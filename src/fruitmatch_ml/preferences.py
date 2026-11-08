from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = Path(
    os.environ.get("FRUITMATCH_ML_DATA_DIR", PROJECT_ROOT / "data")
)


def profile_key(profile_id: str) -> str:
    return hashlib.sha256(profile_id.encode("utf-8")).hexdigest()


def preferences_path(profile_id: str) -> Path:
    return DATA_DIRECTORY / "user_preferences" / f"{profile_key(profile_id)}.json"


def model_path(profile_id: str) -> Path:
    return DATA_DIRECTORY / "models" / f"{profile_key(profile_id)}.pt"


def read_preferences(profile_id: str) -> dict[str, Any]:
    path = preferences_path(profile_id)
    if not path.exists():
        return {
            "version": 1,
            "profile_key": profile_key(profile_id),
            "feedback": [],
            "recommendations": {},
            "recent_recommendations": [],
            "training": {
                "trained_responses": 0,
                "updated_at": None,
            },
        }

    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("feedback"), list):
        raise ValueError("The saved user preference file has an invalid format.")

    document.setdefault("recommendations", {})
    document.setdefault("recent_recommendations", [])

    return document


def write_preferences(profile_id: str, document: dict[str, Any]) -> None:
    path = preferences_path(profile_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(json.dumps(document, ensure_ascii=False, indent=2))
            temporary_file.write("\n")
            temporary_path = temporary_file.name

        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def save_feedback(
    profile_id: str,
    feedback_id: str,
    response: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    document = read_preferences(profile_id)
    if any(item["feedback_id"] == feedback_id for item in document["feedback"]):
        return document, False

    document["feedback"].append(
        {
            **response,
            "feedback_id": feedback_id,
            "received_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_preferences(profile_id, document)

    return document, True


def rewards_by_fruit(document: dict[str, Any]) -> dict[str, float]:
    rewards: dict[str, list[float]] = {}
    for response in document["feedback"]:
        rewards.setdefault(response["fruit_key"], []).append(
            float(response["reward"])
        )

    return {
        fruit_key: sum(values) / len(values)
        for fruit_key, values in rewards.items()
    }
