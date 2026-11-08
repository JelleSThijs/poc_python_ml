"""PPO training adapted for fruit feedback from CleanRL's single-file PPO.

Algorithm reference: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

from fruitmatch_ml.environment import FruitPreferenceEnv

LEARNING_RATE = 2.5e-4
ROLLOUT_STEPS = 64
UPDATE_EPOCHS = 4
MINIBATCHES = 4
CLIP_COEFFICIENT = 0.2
ENTROPY_COEFFICIENT = 0.01
VALUE_COEFFICIENT = 0.5
MAX_GRADIENT_NORM = 0.5
GAMMA = 0.99
GAE_LAMBDA = 0.95


def layer_init(
    layer: nn.Linear,
    standard_deviation: float = math.sqrt(2),
    bias: float = 0.0,
) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, standard_deviation)
    torch.nn.init.constant_(layer.bias, bias)

    return layer


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_count: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_count), standard_deviation=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), standard_deviation=1.0),
        )

    def get_action_and_value(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = Categorical(logits=self.actor(observation))
        if action is None:
            action = distribution.sample()

        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            self.critic(observation).squeeze(-1),
        )

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation).squeeze(-1)


def train_policy(
    environment: FruitPreferenceEnv,
    *,
    seed: int,
    initial_weights: dict[str, torch.Tensor] | None = None,
    total_timesteps: int = 256,
) -> ActorCritic:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = ActorCritic(
        environment.observation_space.shape[0],
        environment.action_space.n,
    )
    if initial_weights is not None:
        agent.load_state_dict(initial_weights)

    optimizer = torch.optim.Adam(
        agent.parameters(),
        lr=LEARNING_RATE,
        eps=1e-5,
    )
    num_iterations = max(1, math.ceil(total_timesteps / ROLLOUT_STEPS))
    batch_size = ROLLOUT_STEPS
    minibatch_size = batch_size // MINIBATCHES
    observation, _ = environment.reset(seed=seed)

    for _ in range(num_iterations):
        observations = torch.zeros((ROLLOUT_STEPS, *environment.observation_space.shape))
        actions = torch.zeros(ROLLOUT_STEPS, dtype=torch.long)
        log_probabilities = torch.zeros(ROLLOUT_STEPS)
        rewards = torch.zeros(ROLLOUT_STEPS)
        dones = torch.zeros(ROLLOUT_STEPS)
        values = torch.zeros(ROLLOUT_STEPS)

        for step in range(ROLLOUT_STEPS):
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
            observations[step] = observation_tensor

            with torch.no_grad():
                action, log_probability, _, value = agent.get_action_and_value(
                    observation_tensor.unsqueeze(0)
                )

            actions[step] = action.item()
            log_probabilities[step] = log_probability.item()
            values[step] = value.item()

            observation, reward, terminated, truncated, _ = environment.step(action.item())
            done = terminated or truncated
            rewards[step] = reward
            dones[step] = float(done)
            if done:
                observation, _ = environment.reset()

        with torch.no_grad():
            next_value = agent.get_value(
                torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            ).reshape(())

        advantages = torch.zeros_like(rewards)
        last_advantage = 0.0
        for step in reversed(range(ROLLOUT_STEPS)):
            next_nonterminal = 1.0 - dones[step]
            following_value = next_value if step == ROLLOUT_STEPS - 1 else values[step + 1]
            delta = (
                rewards[step]
                + GAMMA * following_value * next_nonterminal
                - values[step]
            )
            last_advantage = (
                delta
                + GAMMA * GAE_LAMBDA * next_nonterminal * last_advantage
            )
            advantages[step] = last_advantage

        returns = advantages + values
        indices = np.arange(batch_size)

        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                minibatch = torch.as_tensor(
                    indices[start : start + minibatch_size],
                    dtype=torch.long,
                )
                _, new_log_probability, entropy, new_value = agent.get_action_and_value(
                    observations[minibatch],
                    actions[minibatch],
                )
                log_ratio = new_log_probability - log_probabilities[minibatch]
                ratio = log_ratio.exp()
                minibatch_advantages = advantages[minibatch]
                minibatch_advantages = (
                    (minibatch_advantages - minibatch_advantages.mean())
                    / (minibatch_advantages.std(unbiased=False) + 1e-8)
                )

                policy_loss_unclipped = -minibatch_advantages * ratio
                policy_loss_clipped = -minibatch_advantages * torch.clamp(
                    ratio,
                    1 - CLIP_COEFFICIENT,
                    1 + CLIP_COEFFICIENT,
                )
                policy_loss = torch.maximum(
                    policy_loss_unclipped,
                    policy_loss_clipped,
                ).mean()
                value_loss = 0.5 * (new_value - returns[minibatch]).pow(2).mean()
                entropy_loss = entropy.mean()
                loss = (
                    policy_loss
                    + VALUE_COEFFICIENT * value_loss
                    - ENTROPY_COEFFICIENT * entropy_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRADIENT_NORM)
                optimizer.step()

    return agent