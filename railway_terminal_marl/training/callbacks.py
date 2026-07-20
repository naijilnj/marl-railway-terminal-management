"""
MetricsLoggingCallback: after each underlying-env episode finishes inside
the SuperSuit-vectorized training loop, pull `episode_metrics` off the
raw PettingZoo env and append a row to a JSONL file the dashboard reads.
"""
import json
import os
import time

from stable_baselines3.common.callbacks import BaseCallback


class MetricsLoggingCallback(BaseCallback):
    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self._seen_episode_ends = 0
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # truncate at start of a fresh run
        open(self.log_path, "w").close()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", None)
        if dones is None:
            return True
        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            # SB3 auto-reset puts the pre-reset info under "terminal_observation"'s
            # sibling key "episode" only for Monitor-wrapped envs; here we rely on
            # the custom fields the env attaches directly to info.
            row = {
                "global_step": int(self.num_timesteps),
                "wall_time": time.time(),
                "avg_wait": info.get("avg_wait"),
                "trains_served": info.get("trains_served"),
                "n_waiting": info.get("n_waiting"),
                "invalid_actions": info.get("invalid_actions"),
                "conflicts": info.get("conflicts"),
            }
            if row["avg_wait"] is not None:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                self._seen_episode_ends += 1
        return True
