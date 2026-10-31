from __future__ import annotations

import os
from typing import Dict

import numpy as np
from gymnasium import Env
from gymnasium import spaces
from stable_baselines3 import PPO

from env import RailwayTerminusEnv


class SinglePlatformGymEnv(Env):
    """Train one platform actor on its local observation; other actors wait."""

    def __init__(self, scenario_factory, platform_id: int, seed: int = 0):
        self.scenario_factory = scenario_factory
        self.platform_id = platform_id
        self.agent = f"platform_{platform_id}"
        self.seed = seed
        self.episode = 0
        self.env = RailwayTerminusEnv(scenario=scenario_factory(seed), seed=seed)
        self.observation_space = self.env.observation_space(self.agent)
        self.action_space = self.env.action_space(self.agent)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = self.seed + self.episode if seed is None else seed
        self.episode += 1
        self.env = RailwayTerminusEnv(
            scenario=self.scenario_factory(episode_seed),
            seed=episode_seed,
        )
        observations, _ = self.env.reset(seed=episode_seed, options=options)
        return observations[self.agent], {"seed": episode_seed}

    def step(self, action):
        actions = {agent: self.env.max_queue for agent in self.env.agents}
        actions[self.agent] = int(np.asarray(action).item())
        observations, rewards, terms, truncs, infos = self.env.step(actions)
        terminated = bool(terms.get(self.agent, False))
        truncated = bool(truncs.get(self.agent, False))
        observation = observations.get(
            self.agent,
            np.zeros(self.observation_space.shape, dtype=np.float32),
        )
        info = {
            "episode_metrics": dict(self.env.episode_metrics),
            "avg_wait": infos.get(self.agent, {}).get("avg_wait"),
            "conflicts": infos.get(self.agent, {}).get("conflicts"),
        }
        return observation, float(rewards.get(self.agent, 0.0)), terminated, truncated, info

    def close(self):
        self.env.close()


def train_independent_ppo(
    scenario_factory,
    platform_ids,
    timesteps_per_platform: int,
    seed: int,
    output_dir: str,
    device: str = "auto",
) -> Dict[str, PPO]:
    if timesteps_per_platform <= 0:
        raise ValueError("timesteps_per_platform must be positive")
    models = {}
    os.makedirs(output_dir, exist_ok=True)
    for platform_id in platform_ids:
        agent_seed = seed + platform_id
        env = SinglePlatformGymEnv(scenario_factory, platform_id, seed=agent_seed)
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=256,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.98,
            ent_coef=0.01,
            seed=agent_seed,
            device=device,
            verbose=0,
        )
        model.learn(total_timesteps=timesteps_per_platform)
        model.save(os.path.join(output_dir, f"platform_{platform_id}"))
        models[env.agent] = model
        env.close()
    return models


def independent_actions(models: Dict[str, PPO], env: RailwayTerminusEnv) -> Dict[str, int]:
    actions = {}
    for agent in env.agents:
        model = models.get(agent)
        if model is None:
            raise ValueError(f"Missing independent policy for {agent}")
        action, _ = model.predict(env._observe(agent), deterministic=True)
        actions[agent] = int(np.asarray(action).item())
    return actions
