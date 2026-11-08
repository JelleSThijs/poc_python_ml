from __future__ import annotations

import logging
import random
import threading
from typing import Any, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from torch.nn import functional

from fruitmatch_ml.features import (
    candidate_state,
    fruit_name_features,
    normalized_features,
    preference_state,
)
from fruitmatch_ml.preferences import (
    profile_key,
    read_preferences,
    save_feedback,
    write_preferences,
)
from fruitmatch_ml.training import load_policy, train_profile

logger = logging.getLogger("fruitmatch_ml.api")
app = FastAPI(title="FruitMatch CleanRL recommender", version="1.0.0")
_profile_locks: dict[str, threading.RLock] = {}
_profile_locks_guard = threading.Lock()
NUTRIENT_WEIGHTS = np.ones(6, dtype=np.float32)
REPEAT_RECOMMENDATION_MULTIPLIER = 2.0
REPEAT_ALTERNATIVE_THRESHOLD = 0.5
RECOMMENDATION_TEMPERATURE = 0.18


class FruitFeatures(BaseModel):
    energy: float = Field(ge=0, le=1000)
    protein: float = Field(ge=0, le=100)
    carbohydrates: float = Field(ge=0, le=100)
    fat: float = Field(ge=0, le=100)
    sugars: float = Field(ge=0, le=100)
    salt: float = Field(ge=0, le=10)


class FruitCandidate(BaseModel):
    id: int = Field(gt=0)
    fruit_key: str = Field(pattern=r"^[a-z0-9-]{1,40}$")
    name: str = Field(min_length=1, max_length=100)
    features: FruitFeatures


class RecommendationRequest(BaseModel):
    profile_id: str = Field(min_length=1, max_length=128)
    fruits: list[FruitCandidate] = Field(min_length=1, max_length=64)


class FeedbackRequest(BaseModel):
    profile_id: str = Field(min_length=1, max_length=128)
    feedback_id: str = Field(min_length=1, max_length=64)
    fruit_id: int = Field(gt=0)
    fruit_key: str = Field(pattern=r"^[a-z0-9-]{1,40}$")
    action: Literal["like", "dislike"]
    reward: Literal[-1, 1]
    features: FruitFeatures

    @model_validator(mode="after")
    def reward_matches_action(self) -> FeedbackRequest:
        expected_reward = 1 if self.action == "like" else -1
        if self.reward != expected_reward:
            raise ValueError("reward must match the like/dislike action")

        return self


def profile_lock(profile_id: str) -> threading.RLock:
    key = profile_key(profile_id)
    with _profile_locks_guard:
        if key not in _profile_locks:
            _profile_locks[key] = threading.RLock()

        return _profile_locks[key]


def fruit_similarity(candidate: FruitCandidate, response: dict) -> float:
    candidate_name = fruit_name_features(candidate.fruit_key)
    response_name = fruit_name_features(response["fruit_key"])
    name_denominator = float(
        np.linalg.norm(candidate_name) * np.linalg.norm(response_name)
    )
    name_similarity = (
        float(np.dot(candidate_name, response_name) / name_denominator)
        if name_denominator
        else 0.0
    )

    candidate_nutrients = normalized_features(candidate.features.model_dump())
    response_nutrients = normalized_features(response["features"])
    nutrient_distance = float(
        np.average(
            np.abs(candidate_nutrients - response_nutrients),
            weights=NUTRIENT_WEIGHTS,
        )
    )
    nutrient_similarity = 1.0 - min(1.0, nutrient_distance)

    return 0.4 * name_similarity + 0.6 * nutrient_similarity


def content_preference_score(
    candidate: FruitCandidate,
    feedback: list[dict],
) -> float:
    if not feedback:
        return 0.5

    weighted_similarity = sum(
        response["reward"] * fruit_similarity(candidate, response)
        for response in feedback
    )
    rating_count = len(feedback)

    return min(1.0, max(0.0, 0.5 + 0.5 * weighted_similarity / rating_count))


def direct_preference_score(
    candidate: FruitCandidate,
    feedback: list[dict],
) -> float:
    direct_responses = [
        response
        for response in feedback
        if response["fruit_key"] == candidate.fruit_key
    ]
    if not direct_responses:
        return 0.5

    positive_responses = sum(response["reward"] > 0 for response in direct_responses)

    return positive_responses / len(direct_responses)


def policy_preference_score(
    policy: torch.nn.Module,
    profile: np.ndarray,
    candidate: FruitCandidate,
) -> float:
    observation = torch.as_tensor(
        candidate_state(
            profile,
            candidate.fruit_key,
            candidate.features.model_dump(),
        ),
        dtype=torch.float32,
    )
    with torch.inference_mode():
        logits = policy.actor(observation.unsqueeze(0)).squeeze(0)
        probability_of_like = functional.softmax(logits, dim=0)[1]

    return float(probability_of_like.item())


def personalized_preference_score(
    candidate: FruitCandidate,
    feedback: list[dict],
    policy: torch.nn.Module | None,
    profile: np.ndarray,
) -> float:
    content_score = content_preference_score(candidate, feedback)
    policy_score = (
        policy_preference_score(policy, profile, candidate)
        if policy is not None
        else content_score
    )
    direct_score = direct_preference_score(candidate, feedback)

    return 0.7 * direct_score + 0.2 * content_score + 0.1 * policy_score


def recommend_candidate(
    request: RecommendationRequest,
    feedback: list[dict],
    recent_recommendations: list[str],
) -> tuple[FruitCandidate, float, bool, list[dict[str, int | str | float]]]:
    profile = preference_state(feedback)
    policy = load_policy(request.profile_id)

    scored: list[tuple[float, float, FruitCandidate]] = []
    for candidate in request.fruits:
        preference_score = personalized_preference_score(
            candidate,
            feedback,
            policy,
            profile,
        )
        scored.append((preference_score, preference_score, candidate))

    preference_scores = np.asarray(
        [score for score, _, _ in scored],
        dtype=np.float64,
    )
    weights = np.exp(
        (preference_scores - preference_scores.max()) / RECOMMENDATION_TEMPERATURE
    )
    if recent_recommendations:
        recent_index = next(
            (
                index
                for index, (_, _, candidate) in enumerate(scored)
                if candidate.fruit_key == recent_recommendations[0]
            ),
            None,
        )
        if recent_index is not None and len(scored) > 1:
            best_alternative_score = max(
                score
                for index, score in enumerate(preference_scores)
                if index != recent_index
            )
            if (
                best_alternative_score
                >= preference_scores[recent_index] * REPEAT_ALTERNATIVE_THRESHOLD
            ):
                weights[recent_index] = 0.0
            else:
                weights[recent_index] *= REPEAT_RECOMMENDATION_MULTIPLIER

    probabilities = weights / weights.sum()
    selected_index = random.choices(
        range(len(scored)),
        weights=probabilities.tolist(),
        k=1,
    )[0]
    selected_score, _, selected_fruit = scored[selected_index]
    likelihoods = [
        {
            "fruit_id": candidate.id,
            "fruit_key": candidate.fruit_key,
            "recommendation_probability": float(probability),
            "like_probability": preference_score,
        }
        for (_, preference_score, candidate), probability in zip(
            scored,
            probabilities,
            strict=True,
        )
    ]
    likelihoods.sort(
        key=lambda item: item["recommendation_probability"],
        reverse=True,
    )

    return (
        selected_fruit,
        selected_score,
        policy is not None,
        likelihoods,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "algorithm": "CleanRL PPO",
    }


@app.post("/recommend")
def recommend(request: RecommendationRequest) -> dict[str, Any]:
    with profile_lock(request.profile_id):
        document = read_preferences(request.profile_id)
        fruit, confidence, trained, likelihoods = recommend_candidate(
            request,
            document["feedback"],
            document["recent_recommendations"],
        )
        document["recommendations"][fruit.fruit_key] = (
            document["recommendations"].get(fruit.fruit_key, 0) + 1
        )
        document["recent_recommendations"] = (
            [fruit.fruit_key, *document["recent_recommendations"]][:4]
        )
        write_preferences(request.profile_id, document)

    return {
        "fruit_id": fruit.id,
        "fruit_key": fruit.fruit_key,
        "confidence": confidence,
        "trained": trained,
        "scores": likelihoods,
    }


@app.post("/preferences")
def preferences(request: RecommendationRequest) -> dict[str, Any]:
    with profile_lock(request.profile_id):
        document = read_preferences(request.profile_id)
        feedback_items = document["feedback"]
        profile = preference_state(feedback_items)
        policy = load_policy(request.profile_id)
        scores = []

        for candidate in request.fruits:
            like_probability = personalized_preference_score(
                candidate,
                feedback_items,
                policy,
                profile,
            )
            scores.append(
                {
                    "fruit_id": candidate.id,
                    "fruit_key": candidate.fruit_key,
                    "like_probability": like_probability,
                }
            )

    return {
        "scores": scores,
        "trained": policy is not None,
        "feedback_count": len(feedback_items),
    }


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict[str, int | str | bool]:
    response = {
        "fruit_id": request.fruit_id,
        "fruit_key": request.fruit_key,
        "action": request.action,
        "reward": request.reward,
        "features": request.features.model_dump(),
    }

    with profile_lock(request.profile_id):
        document, created = save_feedback(
            request.profile_id,
            request.feedback_id,
            response,
        )
        if not created:
            return {
                "status": "already_saved",
                "trained": document["training"]["trained_responses"] > 0,
                "feedback_count": len(document["feedback"]),
            }

        try:
            training_result = train_profile(request.profile_id)
        except Exception:
            logger.exception(
                "CleanRL PPO training failed after saving fruit feedback."
            )
            return {
                "status": "saved",
                "trained": False,
                "feedback_count": len(document["feedback"]),
            }

    return {
        "status": "learned",
        "trained": bool(training_result["trained"]),
        "feedback_count": int(training_result["feedback_count"]),
    }


@app.post("/train/{profile_id}")
def train(profile_id: str) -> dict[str, int | str | bool]:
    with profile_lock(profile_id):
        try:
            result = train_profile(profile_id)
        except Exception as error:
            logger.exception("Manual CleanRL PPO training failed.")
            raise HTTPException(status_code=500, detail="Training failed.") from error

    if not result["trained"]:
        raise HTTPException(
            status_code=409,
            detail="At least one saved fruit response is needed to train.",
        )

    return {
        "status": "trained",
        "algorithm": "CleanRL PPO",
        "feedback_count": int(result["feedback_count"]),
    }
