"""
LiveRunner: runs the actual trained PPO policy against the actual
RailwayTerminusEnv, forever, in a background thread. This is the "real
program" the UI connects to — not a canned replay. Episodes auto-reset
with a fresh random schedule (new seed) when they end, so the control
room keeps running indefinitely instead of stopping after one fixed run.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import PPO

from env import RailwayTerminusEnv, mgr_chennai_central_scenario
from env.multiagent_adapter import flatten_observations, split_actions


class LiveRunner:
    def __init__(self, model_path: str, steps_per_sec: float = 3.0, event_log_size: int = 40):
        self.model = PPO.load(model_path)
        self.steps_per_sec = steps_per_sec
        self._lock = threading.Lock()
        self._events = deque(maxlen=event_log_size)
        self._seed = 0
        self._current_seed = 0
        self._episodes_completed = 0
        self._cumulative_served = 0
        self._cumulative_wait = 0.0
        self._cumulative_conflicts = 0
        self._cumulative_invalid = 0
        self._tick = 0
        self._running = False
        self._env = None
        self._obs = None
        self._episode_wait_total = 0.0
        self._episode_served = 0
        self._episode_conflicts = 0
        self._episode_invalid = 0
        self._episode_index = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._reset_episode()
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def _reset_episode(self):
        self._current_seed = self._seed
        scenario = mgr_chennai_central_scenario(seed=self._current_seed)
        self._seed += 1
        self._episode_index = self._episodes_completed + 1
        self._env = RailwayTerminusEnv(scenario=scenario)
        self._obs, _ = self._env.reset()
        self._episode_wait_total = 0.0
        self._episode_served = 0
        self._episode_conflicts = 0
        self._episode_invalid = 0

    def _loop(self):
        while self._running:
            start = time.time()
            self._step()
            elapsed = time.time() - start
            time.sleep(max(0.0, (1.0 / self.steps_per_sec) - elapsed))

    def _step(self):
        env = self._env
        flat_obs = flatten_observations(self._obs, env.agents)
        action_vec, _ = self.model.predict(flat_obs, deterministic=True)
        actions = split_actions(action_vec, env.agents)
        prev_events_len = len(env.events)
        obs, rewards, terms, truncs, infos = env.step(actions)
        self._obs = obs
        self._tick += 1

        with self._lock:
            for ev in env.events[prev_events_len:]:
                ev = dict(ev)
                ev["tick"] = self._tick
                self._events.append(ev)
                if ev["type"] == "allocate":
                    self._episode_wait_total += ev["wait"]
                    self._episode_served += 1
                elif ev["type"] == "conflict":
                    self._episode_conflicts += 1

        done = not env.agents
        if done:
            with self._lock:
                self._episodes_completed += 1
                self._cumulative_served += self._episode_served
                self._cumulative_wait += self._episode_wait_total
                self._cumulative_conflicts += self._episode_conflicts
                self._cumulative_invalid += env.invalid_actions
            self._reset_episode()

    def snapshot(self) -> dict:
        with self._lock:
            env = self._env
            if env is None:
                return {
                    "tick": 0,
                    "episode": 0,
                    "t_in_episode": 0,
                    "horizon": 0,
                    "platforms": [],
                    "queue": [],
                    "events": [],
                    "recent_allocations": [],
                    "recent_conflicts": [],
                    "next_arrivals": [],
                    "kpis": {
                        "episode_avg_wait": 0.0,
                        "all_time_avg_wait": 0.0,
                        "episode_served": 0,
                        "all_time_served": 0,
                        "episode_conflicts": 0,
                        "all_time_conflicts": 0,
                        "episode_invalid": 0,
                        "all_time_invalid": 0,
                        "episodes_completed": 0,
                        "waiting_now": 0,
                    },
                    "utilization": {"overall": 0.0, "main": 0.0, "suburban": 0.0},
                    "system": {
                        "running": False,
                        "steps_per_sec": self.steps_per_sec,
                        "seed": self._current_seed,
                        "episodes_completed": 0,
                        "episode_index": 0,
                    },
                }

            latest_allocations = {}
            for ev in env.events:
                if ev["type"] == "allocate":
                    latest_allocations[ev["platform"]] = ev

            platforms = []
            occ_by_platform = {}
            for p in env.scenario.platforms:
                ev = latest_allocations.get(p.id)
                if ev is not None:
                    end_t = ev["t"] + ev["dwell"]
                    if env.t < end_t:
                        occ_by_platform[p.id] = {**ev, "end_t": end_t}

            for p in env.scenario.platforms:
                occ = occ_by_platform.get(p.id)
                platforms.append({
                    "id": p.id, "label": p.label, "section": p.section,
                    "length_class": p.length_class, "electrified": p.electrified,
                    "ready_at": self._env.platform_ready_at.get(f"platform_{p.id}", 0),
                    "occupied": bool(occ),
                    "occupant": None if not occ else {
                        "train_label": occ["train_label"], "category": occ["category"],
                        "remaining": max(0, occ["end_t"] - env.t), "dwell": occ["dwell"],
                        "wait": occ["wait"],
                        "priority": occ["priority"],
                    },
                })

            queue = [{
                "train_id": row["spec"].id,
                "train_label": row["spec"].label,
                "category": row["spec"].category,
                "wait": row["wait"],
                "length_class": row["spec"].length_class,
                "priority": row["spec"].priority,
                "electrified_required": row["spec"].electrified_required,
            } for row in env.queue]

            next_arrivals = [{
                "train_id": spec.id,
                "train_label": spec.label,
                "category": spec.category,
                "arrival_in": max(0, spec.scheduled_arrival - env.t),
                "scheduled_arrival": spec.scheduled_arrival,
                "length_class": spec.length_class,
                "priority": spec.priority,
            } for spec in env.trains_pending[:6]]

            recent_allocations = [ev for ev in env.events if ev["type"] == "allocate"][-8:]
            recent_conflicts = [ev for ev in env.events if ev["type"] == "conflict"][-8:]

            main_platforms = [p for p in platforms if p["section"] == "main"]
            suburban_platforms = [p for p in platforms if p["section"] == "suburban"]
            main_util = (sum(1 for p in main_platforms if p["occupied"]) / len(main_platforms)) if main_platforms else 0.0
            suburban_util = (sum(1 for p in suburban_platforms if p["occupied"]) / len(suburban_platforms)) if suburban_platforms else 0.0
            overall_util = (sum(1 for p in platforms if p["occupied"]) / len(platforms)) if platforms else 0.0

            ep_avg_wait = (self._episode_wait_total / self._episode_served) if self._episode_served else 0.0
            all_time_served = self._cumulative_served + self._episode_served
            all_time_wait = self._cumulative_wait + self._episode_wait_total
            all_time_avg_wait = (all_time_wait / all_time_served) if all_time_served else 0.0

            return {
                "tick": self._tick,
                "episode": self._episode_index,
                "t_in_episode": env.t,
                "horizon": env.scenario.horizon,
                "platforms": platforms,
                "queue": queue,
                "events": list(self._events)[-20:],
                "recent_allocations": recent_allocations,
                "recent_conflicts": recent_conflicts,
                "next_arrivals": next_arrivals,
                "kpis": {
                    "episode_avg_wait": round(ep_avg_wait, 2),
                    "all_time_avg_wait": round(all_time_avg_wait, 2),
                    "episode_served": self._episode_served,
                    "all_time_served": all_time_served,
                    "episode_conflicts": self._episode_conflicts,
                    "all_time_conflicts": self._cumulative_conflicts + self._episode_conflicts,
                    "episode_invalid": env.invalid_actions,
                    "all_time_invalid": self._cumulative_invalid + env.invalid_actions,
                    "episodes_completed": self._episodes_completed,
                    "waiting_now": len(env.queue),
                },
                "utilization": {
                    "overall": round(overall_util, 3),
                    "main": round(main_util, 3),
                    "suburban": round(suburban_util, 3),
                },
                "system": {
                    "running": self._running,
                    "steps_per_sec": self.steps_per_sec,
                    "seed": self._current_seed,
                    "episodes_completed": self._episodes_completed,
                    "episode_index": self._episode_index,
                },
            }
