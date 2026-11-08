from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

FRUIT_ACTIONS = (
    "apple",
    "banana",
    "blueberry",
    "mango",
    "orange",
    "pineapple",
    "strawberry",
    "watermelon",
)
ACTION_INDEX = {fruit_key: index for index, fruit_key in enumerate(FRUIT_ACTIONS)}
FRUIT_TRAITS = ("berry", "citrus", "melon", "pome", "stone", "tropical")
FRUIT_NAME_TRAITS = {
    "apple": ("pome",),
    "banana": ("tropical",),
    "blueberry": ("berry",),
    "mango": ("stone", "tropical"),
    "orange": ("citrus",),
    "pineapple": ("tropical",),
    "strawberry": ("berry",),
    "watermelon": ("melon",),
}
TRAIT_KEYWORDS = {
    "berry": ("berry", "berries", "strawberry", "blueberry", "raspberry", "blackberry", "cranberry"),
    "citrus": ("citrus", "orange", "lemon", "lime", "grapefruit", "tangerine", "mandarin"),
    "melon": ("melon", "watermelon", "cantaloupe", "honeydew"),
    "pome": ("apple", "pear", "quince"),
    "stone": ("stone", "peach", "nectarine", "plum", "cherry", "apricot"),
    "tropical": ("tropical", "mango", "pineapple", "banana", "papaya", "guava", "kiwi", "dragonfruit", "passionfruit", "coconut"),
}
GENERIC_NAME_WORDS = {"fruit", "fresh", "raw"}
NUTRIENT_SIZE = 6
NAME_FEATURE_SIZE = len(FRUIT_ACTIONS) + len(FRUIT_TRAITS)
PROFILE_SIZE = 2 * (NUTRIENT_SIZE + NAME_FEATURE_SIZE) + 2
OBSERVATION_SIZE = PROFILE_SIZE + NAME_FEATURE_SIZE + NUTRIENT_SIZE


def normalized_features(features: dict[str, Any]) -> np.ndarray:
    values = np.asarray(
        [
            float(features.get("energy", 0)) / 1000,
            float(features.get("protein", 0)) / 100,
            float(features.get("carbohydrates", 0)) / 100,
            float(features.get("fat", 0)) / 100,
            float(features.get("sugars", 0)) / 100,
            float(features.get("salt", 0)) / 10,
        ],
        dtype=np.float32,
    )

    return np.clip(values, 0, 1)


def fruit_name_features(fruit_key: str) -> np.ndarray:
    features = np.zeros(NAME_FEATURE_SIZE, dtype=np.float32)
    trait_offset = len(FRUIT_ACTIONS)
    if fruit_key in FRUIT_ACTIONS:
        features[FRUIT_ACTIONS.index(fruit_key)] = 1.0
        traits = FRUIT_NAME_TRAITS[fruit_key]
    else:
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", fruit_key.lower())
            if token not in GENERIC_NAME_WORDS
        ]
        for token in tokens or [fruit_key.lower()]:
            token_hash = hashlib.sha256(token.encode("utf-8")).digest()
            token_bucket = int.from_bytes(token_hash[:4], "big") % len(FRUIT_ACTIONS)
            features[token_bucket] = 1.0

        compact_name = fruit_key.lower().replace("-", "")
        traits = tuple(
            trait
            for trait, keywords in TRAIT_KEYWORDS.items()
            if any(keyword in tokens or keyword in compact_name for keyword in keywords)
        )

    for trait in traits:
        features[trait_offset + FRUIT_TRAITS.index(trait)] = 1.0

    return features


def preference_state(feedback: list[dict[str, Any]]) -> np.ndarray:
    likes = [item for item in feedback if item["reward"] > 0]
    dislikes = [item for item in feedback if item["reward"] < 0]

    def average_features(
        responses: list[dict[str, Any]],
        field: str,
    ) -> np.ndarray:
        if not responses:
            return np.zeros(
                NUTRIENT_SIZE if field == "features" else NAME_FEATURE_SIZE,
                dtype=np.float32,
            )

        if field == "features":
            values = [normalized_features(item["features"]) for item in responses]
        else:
            values = [
                fruit_name_features(item["fruit_key"])
                for item in responses
            ]

        return np.mean(values, axis=0).astype(np.float32)

    total = len(feedback)
    like_ratio = len(likes) / total if total else 0.0

    return np.concatenate(
        [
            average_features(likes, "features"),
            average_features(dislikes, "features"),
            average_features(likes, "name"),
            average_features(dislikes, "name"),
            np.asarray(
                [min(total, 20) / 20, like_ratio],
                dtype=np.float32,
            ),
        ]
    )


def candidate_state(
    profile: np.ndarray,
    fruit_key: str,
    features: dict[str, Any],
) -> np.ndarray:
    return np.concatenate(
        [
            profile.astype(np.float32),
            fruit_name_features(fruit_key),
            normalized_features(features),
        ]
    ).astype(np.float32)


def training_examples(
    feedback: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[int]]:
    profile = preference_state(feedback)
    observations = [
        candidate_state(profile, item["fruit_key"], item["features"])
        for item in feedback
    ]
    labels = [1 if item["reward"] > 0 else 0 for item in feedback]
    return observations, labels
