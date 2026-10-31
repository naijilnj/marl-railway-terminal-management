"""
MetricsLoggingCallback: after each underlying-env episode finishes inside
the SuperSuit-vectorized training loop, pull `episode_metrics` off the
raw PettingZoo env and append a row to a JSONL file the dashboard reads.
"""
import json
import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MetricsLoggingCallback(BaseCallback):
    def __init__(self, log_path: str, seed: int, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.seed = seed
        self._episode_rewards = None
        self._episode_number = 0
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        open(self.log_path, "w").close()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", None)
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float).reshape(-1)
        if dones is None:
            return True
        if self._episode_rewards is None or len(self._episode_rewards) != len(dones):
            self._episode_rewards = np.zeros(len(dones), dtype=float)
        self._episode_rewards += rewards
        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            metrics = info.get("episode_metrics") or {}
            row = dict(metrics)
            row.update({
                "episode": self._episode_number,
                "global_step": int(self.num_timesteps),
                "wall_time": time.time(),
                "seed": self.seed,
                "episode_seed": info.get("episode_seed"),
                "episode_reward": float(self._episode_rewards[i]),
                "avg_wait": metrics.get("avg_wait", info.get("avg_wait")),
                "trains_served": metrics.get("trains_served", info.get("trains_served")),
                "n_waiting": info.get("n_waiting"),
                "invalid_actions": metrics.get("invalid_actions", info.get("invalid_actions")),
                "conflicts": metrics.get("conflicts", info.get("conflicts")),
            })
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            self._episode_number += 1
            self._episode_rewards[i] = 0.0
        return True
