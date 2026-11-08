from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from typing import Any

import torch

from fruitmatch_ml.environment import FruitPreferenceEnv
from fruitmatch_ml.features import (
    OBSERVATION_SIZE,
    training_examples,
)
from fruitmatch_ml.preferences import (
    model_path,
    profile_key,
    read_preferences,
    write_preferences,
)
from fruitmatch_ml.ppo import ActorCritic, train_policy


def train_profile(profile_id: str, total_timesteps: int = 256) -> dict[str, Any]:
    document = read_preferences(profile_id)
    feedback = document["feedback"]
    if not feedback:
        return {
            "trained": False,
            "feedback_count": 0,
        }

    observations, labels = training_examples(feedback)
    environment = FruitPreferenceEnv(observations, labels)
    policy_file = model_path(profile_id)
    initial_weights = None
    if policy_file.exists():
        try:
            initial_weights = torch.load(
                policy_file,
                map_location="cpu",
                weights_only=True,
            )
        except (OSError, RuntimeError, ValueError):
            initial_weights = None

    seed = int(profile_key(profile_id)[:8], 16) + len(feedback)
    agent = train_policy(
        environment,
        seed=seed,
        initial_weights=initial_weights,
        total_timesteps=total_timesteps,
    )

    policy_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=policy_file.parent,
            prefix=f".{policy_file.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name

        torch.save(agent.state_dict(), temporary_path)
        os.replace(temporary_path, policy_file)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)

    document["training"] = {
        "trained_responses": len(feedback),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "CleanRL PPO",
    }
    write_preferences(profile_id, document)

    return {
        "trained": True,
        "feedback_count": len(feedback),
        "model_path": str(policy_file),
    }


def load_policy(profile_id: str) -> ActorCritic | None:
    policy_file = model_path(profile_id)
    if not policy_file.exists():
        return None

    try:
        agent = ActorCritic(
            observation_size=OBSERVATION_SIZE,
            action_count=2,
        )
        agent.load_state_dict(
            torch.load(
                policy_file,
                map_location="cpu",
                weights_only=True,
            )
        )
        agent.eval()

        return agent
    except (OSError, RuntimeError, ValueError):
        return None
