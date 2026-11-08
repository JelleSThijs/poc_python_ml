from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fruitmatch_ml.features import OBSERVATION_SIZE


class FruitPreferenceEnv(gym.Env[np.ndarray, int]):
    """One-step PPO episodes that learn to predict like or pass from fruit context."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        observations: list[np.ndarray],
        labels: list[int],
    ) -> None:
        super().__init__()
        if not observations or len(observations) != len(labels):
            raise ValueError("PPO training requires matching feedback examples.")

        self.observations = [
            np.asarray(observation, dtype=np.float32)
            for observation in observations
        ]
        self.labels = labels
        self.current_index = 0
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_index = int(self.np_random.integers(len(self.observations)))

        return self.observations[self.current_index].copy(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        expected_action = self.labels[self.current_index]
        reward = 1.0 if int(action) == expected_action else -1.0

        return (
            self.observations[self.current_index].copy(),
            reward,
            True,
            False,
            {"correct_prediction": reward > 0},
        )
