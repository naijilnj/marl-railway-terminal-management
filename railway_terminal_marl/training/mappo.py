from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from env import RailwayTerminusEnv


class MAPPOPolicy(nn.Module):
    """Parameter-shared local actor and centralized global-state critic."""

    def __init__(self, local_observation_dim: int, global_observation_dim: int, action_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(local_observation_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(global_observation_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def act(self, local_observations: np.ndarray, deterministic: bool = True) -> Dict[str, int]:
        with torch.no_grad():
            device = next(self.parameters()).device
            local = torch.as_tensor(local_observations, dtype=torch.float32, device=device)
            distribution = Categorical(logits=self.actor(local))
            if deterministic:
                actions = distribution.logits.argmax(dim=-1)
            else:
                actions = distribution.sample()
        return {f"platform_{i}": int(value) for i, value in enumerate(actions.tolist())}


def _joint_observation(observations, agents):
    return np.concatenate([observations[agent] for agent in agents]).astype(np.float32)


def train_mappo(
    scenario_factory,
    platform_ids: List[int],
    seed: int,
    total_timesteps: int,
    output_path: str,
    device: str = "cpu",
    rollout_steps: int = 256,
    update_epochs: int = 4,
    minibatch_size: int = 64,
):
    if total_timesteps <= 0 or rollout_steps <= 0:
        raise ValueError("timesteps and rollout_steps must be positive")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_obj = torch.device(device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")

    scenario = scenario_factory(seed)
    env = RailwayTerminusEnv(scenario=scenario, seed=seed)
    observations, _ = env.reset(seed=seed)
    agents = [f"platform_{platform_id}" for platform_id in platform_ids]
    local_dim = env.observation_space(agents[0]).shape[0]
    global_dim = local_dim * len(agents)
    action_dim = env.action_space(agents[0]).n
    policy = MAPPOPolicy(local_dim, global_dim, action_dim).to(device_obj)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    step_count = 0
    episode_count = 0
    while step_count < total_timesteps:
        local_rollout = []
        global_rollout = []
        action_rollout = []
        log_prob_rollout = []
        reward_rollout = []
        value_rollout = []
        done_rollout = []
        next_value_rollout = []

        for _ in range(min(rollout_steps, total_timesteps - step_count)):
            local = np.stack([observations[agent] for agent in agents]).astype(np.float32)
            global_state = _joint_observation(observations, agents)
            local_tensor = torch.as_tensor(local, dtype=torch.float32, device=device_obj)
            global_tensor = torch.as_tensor(global_state, dtype=torch.float32, device=device_obj)
            with torch.no_grad():
                distribution = Categorical(logits=policy.actor(local_tensor))
                sampled_actions = distribution.sample()
                old_log_probs = distribution.log_prob(sampled_actions)
                value = policy.critic(global_tensor).squeeze(-1)
            action_dict = {
                agent: int(action)
                for agent, action in zip(agents, sampled_actions.cpu().tolist())
            }
            next_observations, rewards, terms, truncs, _ = env.step(action_dict)
            done = not env.agents
            team_reward = float(sum(rewards.values()) / max(1, len(rewards)))
            if done:
                next_value = 0.0
            else:
                next_global = _joint_observation(next_observations, agents)
                with torch.no_grad():
                    next_value = float(policy.critic(
                        torch.as_tensor(next_global, dtype=torch.float32, device=device_obj)
                    ).item())

            local_rollout.append(local)
            global_rollout.append(global_state)
            action_rollout.append(sampled_actions.cpu().numpy())
            log_prob_rollout.append(old_log_probs.cpu().numpy())
            reward_rollout.append(team_reward)
            value_rollout.append(float(value.item()))
            done_rollout.append(float(done))
            next_value_rollout.append(next_value)
            step_count += 1

            if done:
                episode_count += 1
                reset_seed = seed + episode_count
                env = RailwayTerminusEnv(
                    scenario=scenario_factory(reset_seed),
                    seed=reset_seed,
                )
                observations, _ = env.reset(seed=reset_seed)
            else:
                observations = next_observations

        rewards_array = np.asarray(reward_rollout, dtype=np.float32)
        values_array = np.asarray(value_rollout, dtype=np.float32)
        dones_array = np.asarray(done_rollout, dtype=np.float32)
        next_values_array = np.asarray(next_value_rollout, dtype=np.float32)
        advantages = np.zeros_like(rewards_array)
        gae = 0.0
        for index in reversed(range(len(rewards_array))):
            delta = (
                rewards_array[index]
                + 0.99 * next_values_array[index] * (1.0 - dones_array[index])
                - values_array[index]
            )
            gae = delta + 0.95 * 0.99 * (1.0 - dones_array[index]) * gae
            advantages[index] = gae
        returns = advantages + values_array
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        local_tensor = torch.as_tensor(
            np.asarray(local_rollout), dtype=torch.float32, device=device_obj
        )
        global_tensor = torch.as_tensor(
            np.asarray(global_rollout), dtype=torch.float32, device=device_obj
        )
        action_tensor = torch.as_tensor(
            np.asarray(action_rollout), dtype=torch.long, device=device_obj
        )
        old_log_prob_tensor = torch.as_tensor(
            np.asarray(log_prob_rollout), dtype=torch.float32, device=device_obj
        )
        advantage_tensor = torch.as_tensor(
            advantages, dtype=torch.float32, device=device_obj
        ).unsqueeze(1).expand(-1, len(agents))
        return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device_obj)

        sample_count = len(rewards_array)
        for _ in range(update_epochs):
            permutation = torch.randperm(sample_count, device=device_obj)
            for start in range(0, sample_count, minibatch_size):
                batch = permutation[start:start + minibatch_size]
                distribution = Categorical(logits=policy.actor(local_tensor[batch]))
                new_log_probs = distribution.log_prob(action_tensor[batch])
                ratio = torch.exp(new_log_probs - old_log_prob_tensor[batch])
                batch_advantages = advantage_tensor[batch]
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 0.8, 1.2) * batch_advantages
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                predicted_values = policy.critic(global_tensor[batch]).squeeze(-1)
                critic_loss = torch.nn.functional.mse_loss(
                    predicted_values, return_tensor[batch]
                )
                entropy = distribution.entropy().mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({
        "state_dict": policy.state_dict(),
        "local_observation_dim": local_dim,
        "global_observation_dim": global_dim,
        "action_dim": action_dim,
        "platform_ids": platform_ids,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "episodes": episode_count,
        "architecture": "shared decentralized actor; centralized training critic",
    }, output_path)
    env.close()
    return policy


def load_mappo(path: str, device: str = "cpu") -> MAPPOPolicy:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    policy = MAPPOPolicy(
        checkpoint["local_observation_dim"],
        checkpoint["global_observation_dim"],
        checkpoint["action_dim"],
    ).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy


def mappo_actions(policy: MAPPOPolicy, env: RailwayTerminusEnv) -> Dict[str, int]:
    local = np.stack([env._observe(agent) for agent in env.agents]).astype(np.float32)
    with torch.no_grad():
        device = next(policy.parameters()).device
        logits = policy.actor(torch.as_tensor(local, dtype=torch.float32, device=device))
        chosen = logits.argmax(dim=-1).tolist()
    return {agent: int(action) for agent, action in zip(env.agents, chosen)}
