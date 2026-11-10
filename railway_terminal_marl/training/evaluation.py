from __future__ import annotations

import json
import os
import time
from typing import Dict, Iterable, Optional

import numpy as np

from env import RailwayTerminusEnv, ScenarioConfig
from env.multiagent_adapter import flatten_observations, split_actions
from training.independent import independent_actions
from training.mappo import mappo_actions
from training.policies import HEURISTIC_ALGORITHMS, heuristic_actions


ALGORITHMS = (
    "fcfs",
    "priority_first",
    "best_fit",
    "centralized_ppo",
    "independent_ppo",
    "mappo",
    "matching",
)


def _actions(
    algorithm: str,
    env: RailwayTerminusEnv,
    model=None,
    independent_models=None,
):
    if algorithm in HEURISTIC_ALGORITHMS:
        return heuristic_actions(env, algorithm)
    if algorithm == "matching":
        return {agent: env.max_queue for agent in env.agents}
    if algorithm == "independent_ppo":
        if independent_models is None:
            raise ValueError("independent_models are required for independent_ppo")
        return independent_actions(independent_models, env)
    if algorithm == "mappo":
        if model is None:
            raise ValueError("a MAPPO policy is required for mappo")
        return mappo_actions(model, env)
    if algorithm == "centralized_ppo":
        if model is None:
            raise ValueError("a centralized PPO model is required")
        action_vector, _ = model.predict(
            flatten_observations(
                {agent: env._observe(agent) for agent in env.agents},
                env.agents,
            ),
            deterministic=True,
        )
        return split_actions(action_vector, env.agents)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def evaluate_episode(
    scenario: ScenarioConfig,
    algorithm: str,
    model=None,
    independent_models=None,
    coordination: str = "lowest_id",
    evaluation_seed: Optional[int] = None,
) -> dict:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if algorithm == "matching":
        coordination = "matching"
    env = RailwayTerminusEnv(
        scenario=scenario,
        seed=evaluation_seed or 0,
        coordination=coordination,
    )
    env.reset(seed=evaluation_seed)
    started = time.perf_counter()
    while env.agents:
        action_dict = _actions(
            algorithm,
            env,
            model=model,
            independent_models=independent_models,
        )
        env.step(action_dict)
    elapsed = time.perf_counter() - started
    result = dict(env.episode_metrics)
    result.update({
        "algorithm": algorithm,
        "coordination": coordination,
        "evaluation_seed": evaluation_seed,
        "computation_seconds": elapsed,
    })
    return result


def evaluate_scenarios(
    scenario_factory,
    scenario_seeds: Iterable[int],
    algorithm: str,
    model=None,
    independent_models=None,
    coordination: str = "lowest_id",
) -> list:
    results = []
    for seed in scenario_seeds:
        scenario = scenario_factory(seed)
        results.append(evaluate_episode(
            scenario=scenario,
            algorithm=algorithm,
            model=model,
            independent_models=independent_models,
            coordination=coordination,
            evaluation_seed=seed,
        ))
    return results


def write_jsonl(path: str, rows: Iterable[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
