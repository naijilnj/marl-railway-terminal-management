from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from gymnasium import Env, spaces

from .terminal_env import RailwayTerminusEnv


def flatten_observations(observations: Dict[str, np.ndarray], agents: List[str]) -> np.ndarray:
    return np.concatenate([observations[agent] for agent in agents]).astype(np.float32)


def split_actions(actions: np.ndarray, agents: List[str]) -> Dict[str, int]:
    flat_actions = np.asarray(actions).reshape(-1)
    return {agent: int(flat_actions[index]) for index, agent in enumerate(agents)}


class TerminusMultiAgentGymEnv(Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, env: RailwayTerminusEnv):
        super().__init__()
        self.env = env
        self.agents = list(self.env.possible_agents)
        self._obs_dict = None
        sample_obs = self.env.observation_space(self.agents[0])
        sample_action = self.env.action_space(self.agents[0])
        self.observation_space = spaces.Box(
            low=np.float32(sample_obs.low.min()),
            high=np.float32(sample_obs.high.max()),
            shape=(sample_obs.shape[0] * len(self.agents),),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([sample_action.n] * len(self.agents))

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        self._obs_dict = obs
        flat_obs = flatten_observations(obs, self.agents)
        info = {"agents": self.agents, "episode_metrics": self.env.episode_metrics}
        return flat_obs, info

    def step(self, action):
        action_dict = split_actions(action, self.agents)
        obs, rewards, terms, truncs, infos = self.env.step(action_dict)
        self.agents = list(self.env.agents)
        self._obs_dict = obs

        flat_obs = flatten_observations(obs, list(obs.keys())) if obs else np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = float(sum(rewards.values()))
        terminated = bool(terms and all(terms.values()))
        truncated = bool(truncs and any(truncs.values()))
        info = {
            "agents": list(obs.keys()),
            "per_agent_rewards": rewards,
            "episode_metrics": self.env.episode_metrics,
            "avg_wait": infos[next(iter(infos))]["avg_wait"] if infos else None,
            "trains_served": infos[next(iter(infos))]["trains_served"] if infos else None,
            "n_waiting": infos[next(iter(infos))]["n_waiting"] if infos else None,
            "invalid_actions": infos[next(iter(infos))]["invalid_actions"] if infos else None,
            "conflicts": infos[next(iter(infos))]["conflicts"] if infos else None,
        }
        return flat_obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()