from __future__ import annotations

from typing import Dict

from env.terminal_env import RailwayTerminusEnv
from env.config import LENGTH_CLASSES


HEURISTIC_ALGORITHMS = ("fcfs", "priority_first", "best_fit")


def heuristic_actions(env: RailwayTerminusEnv, algorithm: str) -> Dict[str, int]:
    """Create simultaneous platform claims for a documented global heuristic."""
    if algorithm not in HEURISTIC_ALGORITHMS:
        raise ValueError(f"Unknown heuristic: {algorithm}")
    actions = {agent: env.max_queue for agent in env.agents}
    free_agents = [agent for agent in env.agents if env._is_free(agent)]
    free_agents = [
        agent for agent in free_agents
        if env.approach_busy_until.get(
            env._platform_by_agent[agent].approach_group, 0
        ) <= env.t
    ]
    queued = list(range(len(env.queue)))

    def feasible_platforms(q_idx):
        train = env.queue[q_idx]["spec"]
        return [
            agent for agent in free_agents
            if env._is_compatible(env._platform_by_agent[agent], train)
        ]

    if algorithm == "fcfs":
        queued.sort(key=lambda i: (
            env.queue[i]["spec"].scheduled_arrival,
            env.queue[i]["spec"].id,
        ))
    elif algorithm == "priority_first":
        queued.sort(key=lambda i: (
            env.queue[i]["spec"].priority,
            -env.queue[i]["wait"],
            env.queue[i]["spec"].scheduled_arrival,
        ))
    else:
        queued.sort(key=lambda i: (
            len(feasible_platforms(i)),
            -LENGTH_CLASSES.index(env.queue[i]["spec"].length_class),
            env.queue[i]["spec"].priority,
            -env.queue[i]["wait"],
            env.queue[i]["spec"].id,
        ))

    assigned_groups = set()
    assigned_trains = set()
    for q_idx in queued:
        if q_idx in assigned_trains:
            continue
        feasible = [
            agent for agent in free_agents
            if agent in feasible_platforms(q_idx)
            and env._platform_by_agent[agent].approach_group not in assigned_groups
        ]
        if not feasible:
            continue
        train = env.queue[q_idx]["spec"]
        chosen = min(
            feasible,
            key=lambda agent: (
                LENGTH_CLASSES.index(env._platform_by_agent[agent].length_class)
                - LENGTH_CLASSES.index(train.length_class),
                env._platform_by_agent[agent].id,
            ),
        )
        actions[chosen] = q_idx
        assigned_groups.add(env._platform_by_agent[chosen].approach_group)
        assigned_trains.add(q_idx)
        free_agents.remove(chosen)
    return actions
