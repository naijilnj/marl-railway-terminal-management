"""
Train a shared-policy PPO agent across all platform agents using a
custom Gymnasium wrapper that flattens the fixed PettingZoo ParallelEnv
into a single-policy training problem, then:
  1. save the model
  2. write logs/metrics.jsonl  (per-episode metrics -> dashboard)
  3. write logs/events.json    (one clean rollout -> simulator)

Run: python3 -m training.train --timesteps 40000
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from env import RailwayTerminusEnv, default_scenario, mgr_chennai_central_scenario
from env.multiagent_adapter import TerminusMultiAgentGymEnv, flatten_observations, split_actions
from training.callbacks import MetricsLoggingCallback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(ROOT, "logs")
MODEL_PATH = os.path.join(LOGS_DIR, "ppo_terminus.zip")
METRICS_PATH = os.path.join(LOGS_DIR, "metrics.jsonl")
EVENTS_PATH = os.path.join(LOGS_DIR, "events.json")

SCENARIOS = {
    "default": default_scenario,
    "mgr_chennai_central": mgr_chennai_central_scenario,
}


def make_training_env(scenario_name: str, num_vec_envs: int = 4, seed: int = 0):
    raw_env = RailwayTerminusEnv(scenario=SCENARIOS[scenario_name](seed=seed))
    return TerminusMultiAgentGymEnv(raw_env)


def train(total_timesteps: int, scenario_name: str, seed: int = 0):
    os.makedirs(LOGS_DIR, exist_ok=True)
    env = make_training_env(scenario_name, seed=seed)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.98,
        ent_coef=0.01,
        verbose=1,
    )
    callback = MetricsLoggingCallback(log_path=METRICS_PATH)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return model


def record_rollout(model, scenario_name: str, seed: int = 0):
    """Run one deterministic episode with the trained (shared) policy and
    dump the allocation events + summary metrics for the simulator viz."""
    raw_env = RailwayTerminusEnv(scenario=SCENARIOS[scenario_name](seed=seed))
    obs, infos = raw_env.reset()
    while raw_env.agents:
        flat_obs = flatten_observations(obs, raw_env.agents)
        action_vec, _ = model.predict(flat_obs, deterministic=True)
        actions = split_actions(action_vec, raw_env.agents)
        obs, rewards, terms, truncs, infos = raw_env.step(actions)

    payload = {
        "scenario": {
            "horizon": raw_env.scenario.horizon,
            "platforms": [
                {"id": p.id, "label": p.label, "section": p.section,
                 "length_class": p.length_class, "electrified": p.electrified}
                for p in raw_env.scenario.platforms
            ],
            "total_trains": len(raw_env.scenario.trains),
        },
        "episode_metrics": raw_env.episode_metrics,
        "events": raw_env.events,
    }
    with open(EVENTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved rollout events to {EVENTS_PATH}")
    print("Episode metrics:", raw_env.episode_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="default")
    args = parser.parse_args()

    model = train(args.timesteps, args.scenario, seed=args.seed)
    record_rollout(model, args.scenario, seed=args.seed)
