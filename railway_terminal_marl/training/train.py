"""Reproducible single-run PPO training and rollout recording."""
import argparse
import json
import os
import platform
import random
import sys
import time
import subprocess
from importlib import metadata

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from env import (
    RailwayTerminusEnv,
    default_scenario,
    mgr_chennai_central_scenario,
    traffic_scenario,
)
from env.multiagent_adapter import TerminusMultiAgentGymEnv, flatten_observations, split_actions
from training.callbacks import MetricsLoggingCallback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(ROOT, "logs")

SCENARIOS = {
    "default": default_scenario,
    "mgr_chennai_central": mgr_chennai_central_scenario,
    "low_traffic": lambda seed: traffic_scenario("low_traffic", seed=seed),
    "moderate_traffic": lambda seed: traffic_scenario("moderate_traffic", seed=seed),
    "high_traffic": lambda seed: traffic_scenario("high_traffic", seed=seed),
    "peak_hour": lambda seed: traffic_scenario("peak_hour", seed=seed),
    "platform_disruption": lambda seed: traffic_scenario("platform_disruption", seed=seed),
    "extreme_stress": lambda seed: traffic_scenario("extreme_stress", seed=seed),
}


def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _software_versions():
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for package in ("numpy", "gymnasium", "pettingzoo", "stable-baselines3", "matplotlib"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _hardware_metadata(device: str):
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device_requested": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _scenario_definition(scenario):
    return {
        "name": scenario.name,
        "horizon": scenario.horizon,
        "arrival_horizon": scenario.arrival_horizon,
        "max_queue": scenario.max_queue,
        "approach_headway": scenario.approach_headway,
        "platforms": [
            {
                "id": platform.id,
                "length_class": platform.length_class,
                "electrified": platform.electrified,
                "turnaround_buffer": platform.turnaround_buffer,
                "label": platform.label,
                "section": platform.section,
                "compatible_categories": platform.compatible_categories,
                "approach_group": platform.approach_group,
                "closure_intervals": platform.closure_intervals,
            }
            for platform in scenario.platforms
        ],
        "trains": [
            {
                "id": train.id,
                "scheduled_arrival": train.scheduled_arrival,
                "dwell_time": train.dwell_time,
                "length_class": train.length_class,
                "electrified_required": train.electrified_required,
                "priority": train.priority,
                "label": train.label,
                "category": train.category,
                "arrival_delay": train.arrival_delay,
                "unexpected": train.unexpected,
                "cancelled": train.cancelled,
            }
            for train in scenario.trains
        ],
    }


def make_training_env(
    scenario_name: str,
    num_vec_envs: int = 1,
    seed: int = 0,
    coordination: str = "lowest_id",
):
    if num_vec_envs != 1:
        raise ValueError("This adapter currently supports one environment at a time")
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    scenario_factory = SCENARIOS[scenario_name]
    raw_env = RailwayTerminusEnv(
        scenario=scenario_factory(seed), seed=seed, coordination=coordination
    )
    return TerminusMultiAgentGymEnv(
        raw_env,
        scenario_factory=scenario_factory,
        seed=seed,
        coordination=coordination,
    )


def train(
    total_timesteps: int,
    scenario_name: str,
    seed: int = 0,
    output_dir: str = LOGS_DIR,
    coordination: str = "lowest_id",
    device: str = "auto",
    deterministic: bool = False,
):
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    seed_everything(seed, deterministic=deterministic)
    os.makedirs(output_dir, exist_ok=True)
    env = make_training_env(
        scenario_name, seed=seed, coordination=coordination
    )
    initial_scenario = SCENARIOS[scenario_name](seed)
    config = {
        "algorithm": "centralized_shared_parameter_ppo",
        "scenario": scenario_name,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "coordination": coordination,
        "n_steps": 256,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "gamma": 0.98,
        "ent_coef": 0.01,
        "deterministic_torch": deterministic,
        "device": device,
        "training_schedule_seed_rule": "base seed + episode index",
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    with open(os.path.join(output_dir, "scenario.json"), "w", encoding="utf-8") as file:
        json.dump(_scenario_definition(initial_scenario), file, indent=2)
    metadata_payload = {
        "seed": seed,
        "started_at": time.time(),
        "software_versions": _software_versions(),
        "hardware": _hardware_metadata(device),
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    metadata_payload["source_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    metadata_payload["source_tree_dirty"] = bool(subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip())
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata_payload, file, indent=2)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.98,
        ent_coef=0.01,
        seed=seed,
        device=device,
        verbose=1,
    )
    callback = MetricsLoggingCallback(
        log_path=os.path.join(output_dir, "training.jsonl"),
        seed=seed,
    )
    started = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    training_seconds = time.perf_counter() - started
    checkpoint_name = (
        "ppo_terminus"
        if os.path.abspath(output_dir) == os.path.abspath(LOGS_DIR)
        else "model"
    )
    model_path = os.path.join(output_dir, checkpoint_name)
    model.save(model_path)
    metadata_payload["training_seconds"] = training_seconds
    metadata_payload["completed_at"] = time.time()
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata_payload, file, indent=2)
    env.close()
    print(f"Saved model to {model_path}.zip")
    return model


def record_rollout(
    model,
    scenario_name: str,
    seed: int = 0,
    output_path: str = os.path.join(LOGS_DIR, "events.json"),
    coordination: str = "lowest_id",
):
    """Run one deterministic episode and persist events and metrics."""
    scenario = SCENARIOS[scenario_name](seed)
    raw_env = RailwayTerminusEnv(
        scenario=scenario, seed=seed, coordination=coordination
    )
    obs, _ = raw_env.reset(seed=seed)
    while raw_env.agents:
        flat_obs = flatten_observations(obs, raw_env.agents)
        action_vec, _ = model.predict(flat_obs, deterministic=True)
        actions = split_actions(action_vec, raw_env.agents)
        obs, _, _, _, _ = raw_env.step(actions)

    payload = {
        "scenario": {
            "name": scenario.name,
            "seed": seed,
            "arrival_horizon": scenario.arrival_horizon,
            "approach_headway": scenario.approach_headway,
            "horizon": scenario.horizon,
            "platforms": [
                {
                    "id": item.id,
                    "label": item.label,
                    "section": item.section,
                    "length_class": item.length_class,
                    "electrified": item.electrified,
                    "compatible_categories": item.compatible_categories,
                    "approach_group": item.approach_group,
                    "closure_intervals": item.closure_intervals,
                }
                for item in scenario.platforms
            ],
            "total_trains": len(scenario.trains),
        },
        "episode_metrics": raw_env.episode_metrics,
        "train_states": raw_env.train_states,
        "events": raw_env.events,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"Saved rollout events to {output_path}")
    print("Episode metrics:", raw_env.episode_metrics)
    return raw_env.episode_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="default")
    parser.add_argument("--coordination", choices=[
        "lowest_id", "random", "coordinated", "matching",
    ], default="lowest_id")
    parser.add_argument("--output-dir", default=LOGS_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    trained_model = train(
        args.timesteps,
        args.scenario,
        seed=args.seed,
        output_dir=args.output_dir,
        coordination=args.coordination,
        device=args.device,
        deterministic=args.deterministic,
    )
    record_rollout(
        trained_model,
        args.scenario,
        seed=args.seed,
        output_path=os.path.join(args.output_dir, "events.json"),
        coordination=args.coordination,
    )
