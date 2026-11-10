"""Five-seed PPO experiments with shared unseen test schedules."""
import argparse
import csv
import json
import math
import os
import statistics
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env import TRAFFIC_PROFILES, traffic_scenario
from training.evaluation import evaluate_episode, write_jsonl
from training.independent import train_independent_ppo
from training.mappo import train_mappo
from training.train import SCENARIOS, train

DEFAULT_SEEDS = (11, 22, 33, 44, 55)
CLAIM_RESOLVERS = ("lowest_id", "random", "coordinated")
METRICS = (
    "avg_wait",
    "std_wait",
    "median_wait",
    "max_wait",
    "trains_served",
    "service_completion_percentage",
    "invalid_actions",
    "conflicts",
    "repeated_claims",
    "approach_conflicts",
    "platform_utilization",
    "priority_compliance",
    "computation_seconds",
)


def _mean_sd(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _t_critical(sample_count):
    table = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
    }
    return table.get(sample_count, 1.96)


def _summary(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, group_rows in sorted(grouped.items()):
        record = dict(zip(keys, group_key))
        for metric in METRICS:
            mean, sd = _mean_sd([row.get(metric) for row in group_rows])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
            count = sum(row.get(metric) is not None for row in group_rows)
            half_width = (
                _t_critical(count) * sd / math.sqrt(count)
                if count > 1 and sd is not None else 0.0
            )
            record[f"{metric}_ci95_low"] = mean - half_width if mean is not None else None
            record[f"{metric}_ci95_high"] = mean + half_width if mean is not None else None
            record[f"{metric}_n"] = count
        output.append(record)
    return output


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _plot_training_curves(seed_dirs, output_dir):
    metrics_by_episode = defaultdict(lambda: defaultdict(list))
    reward_by_episode = defaultdict(lambda: defaultdict(list))
    for seed_dir in seed_dirs:
        for row in _read_jsonl(os.path.join(seed_dir, "training.jsonl")):
            episode = row["episode"]
            for metric in ("avg_wait", "invalid_actions", "conflicts"):
                if row.get(metric) is not None:
                    metrics_by_episode[metric][episode].append(float(row[metric]))
            if row.get("episode_reward") is not None:
                reward_by_episode["episode_reward"][episode].append(
                    float(row["episode_reward"])
                )

    plots = [
        ("avg_wait", "Average waiting time", "training_wait.png"),
        ("invalid_actions", "Invalid actions per episode", "training_invalid_actions.png"),
        ("conflicts", "Allocation conflicts per episode", "training_conflicts.png"),
    ]
    for metric, title, filename in plots:
        episodes = sorted(metrics_by_episode[metric])
        if not episodes:
            continue
        means, deviations = [], []
        for episode in episodes:
            mean, sd = _mean_sd(metrics_by_episode[metric][episode])
            means.append(mean)
            deviations.append(sd)
        plt.figure(figsize=(8, 4.5))
        plt.plot(episodes, means, label="Mean across seeds")
        plt.fill_between(
            episodes,
            np.asarray(means) - np.asarray(deviations),
            np.asarray(means) + np.asarray(deviations),
            alpha=0.2,
            label="±1 SD across seeds",
        )
        plt.xlabel("Completed training episode")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=160)
        plt.close()

    episodes = sorted(reward_by_episode["episode_reward"])
    if episodes:
        means = [
            _mean_sd(reward_by_episode["episode_reward"][episode])[0]
            for episode in episodes
        ]
        deviations = [
            _mean_sd(reward_by_episode["episode_reward"][episode])[1]
            for episode in episodes
        ]
        plt.figure(figsize=(8, 4.5))
        plt.plot(episodes, means, label="Mean across seeds")
        plt.fill_between(
            episodes,
            np.asarray(means) - np.asarray(deviations),
            np.asarray(means) + np.asarray(deviations),
            alpha=0.2,
            label="±1 SD across seeds",
        )
        plt.xlabel("Completed training episode")
        plt.ylabel("Episode reward")
        plt.title("PPO episode reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_reward.png"), dpi=160)
        plt.close()


def _plot_test_summary(within_seed_summary, output_dir):
    selected = [
        row for row in within_seed_summary
        if row["algorithm"] == "centralized_ppo"
        and row["coordination"] == "lowest_id"
    ]
    available_profiles = {row["test_profile"] for row in selected}
    profiles = [profile for profile in TRAFFIC_PROFILES if profile in available_profiles]
    values, deviations = [], []
    for profile in profiles:
        seed_means = [
            row["avg_wait_mean"] for row in selected
            if row["test_profile"] == profile
            and row["training_seed"] is not None
        ]
        mean, sd = _mean_sd(seed_means)
        values.append(mean or 0.0)
        deviations.append(sd or 0.0)
    if profiles:
        plt.figure(figsize=(9, 4.5))
        plt.bar(profiles, values, yerr=deviations, capsize=4)
        plt.ylabel("Mean waiting time")
        plt.title("Unseen-scenario performance (mean ± SD across PPO seeds)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "test_performance_by_scenario.png"), dpi=160)
        plt.close()


def _plot_algorithm_comparison(method_summary, output_dir):
    rows = [
        row for row in method_summary
        if row["test_profile"] in {"moderate_traffic", "high_traffic", "peak_hour"}
    ]
    if not rows:
        return
    labels = [
        f"{row['algorithm']} / {row['coordination']}\n{row['test_profile']}"
        for row in rows
    ]
    for metric, title, filename in (
        ("avg_wait", "Mean waiting time", "algorithm_comparison.png"),
        ("computation_seconds", "Computation time per episode (seconds)", "computation_time.png"),
        ("service_completion_percentage", "Service completion (%)", "robustness_comparison.png"),
    ):
        means = [row[f"{metric}_mean"] for row in rows]
        deviations = [row[f"{metric}_sd"] for row in rows]
        plt.figure(figsize=(max(10, len(rows) * 0.8), 5))
        plt.bar(range(len(rows)), means, yerr=deviations, capsize=3)
        plt.ylabel(title)
        plt.title(f"{title} (mean ± SD across run/test rows)")
        plt.xticks(range(len(rows)), labels, rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=160)
        plt.close()

    learning_algorithms = ("fcfs", "priority_first", "best_fit", "centralized_ppo", "matching")
    available_profiles = {row["test_profile"] for row in rows}
    profiles = [profile for profile in TRAFFIC_PROFILES if profile in available_profiles]
    plt.figure(figsize=(8, 4.5))
    for algorithm in learning_algorithms:
        selected = [
            row for row in rows
            if row["algorithm"] == algorithm
            and row["coordination"] in {"lowest_id", "matching"}
        ]
        if selected:
            profile_means = []
            for profile in profiles:
                record = next(
                    (row for row in selected if row["test_profile"] == profile), None
                )
                profile_means.append(
                    record["avg_wait_mean"] if record is not None else float("nan")
                )
            plt.plot(profiles, profile_means, marker="o", label=algorithm)
    plt.xlabel("Traffic profile")
    plt.ylabel("Mean waiting time")
    plt.title("Congestion sensitivity")
    plt.xticks(rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "congestion_sensitivity.png"), dpi=160)
    plt.close()


def _plot_training_vs_test(seed_dirs, raw_results, output_dir):
    training_by_seed = {}
    for seed_dir in seed_dirs:
        metadata_path = os.path.join(seed_dir, "metadata.json")
        with open(metadata_path, encoding="utf-8") as file:
            seed = json.load(file)["seed"]
        rows = _read_jsonl(os.path.join(seed_dir, "training.jsonl"))
        values = [row["avg_wait"] for row in rows if row.get("avg_wait") is not None]
        if values:
            training_by_seed[seed] = statistics.mean(values)
    test_by_seed = defaultdict(list)
    for row in raw_results:
        if (
            row["algorithm"] == "centralized_ppo"
            and row["coordination"] == "lowest_id"
            and row["training_seed"] is not None
            and row.get("avg_wait") is not None
        ):
            test_by_seed[row["training_seed"]].append(row["avg_wait"])
    seeds = sorted(set(training_by_seed) & set(test_by_seed))
    if not seeds:
        return
    training_means = [training_by_seed[seed] for seed in seeds]
    test_means = [statistics.mean(test_by_seed[seed]) for seed in seeds]
    means = [statistics.mean(training_means), statistics.mean(test_means)]
    deviations = [
        statistics.stdev(training_means) if len(seeds) > 1 else 0.0,
        statistics.stdev(test_means) if len(seeds) > 1 else 0.0,
    ]
    plt.figure(figsize=(6, 4.5))
    plt.bar(["Training episodes", "Unseen test scenarios"], means, yerr=deviations, capsize=5)
    plt.ylabel("Mean waiting time")
    plt.title("Training versus held-out performance (mean ± SD across seeds)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_vs_unseen_test.png"), dpi=160)
    plt.close()


def run_multiseed_experiment(
    scenario_name="moderate_traffic",
    seeds=DEFAULT_SEEDS,
    test_episodes=5,
    total_timesteps=300000,
    output_dir=None,
    profiles=None,
    coordination="lowest_id",
    device="auto",
    include_other_architectures=False,
    independent_timesteps=None,
):
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least five distinct training seeds")
    if test_episodes < 2:
        raise ValueError("At least two held-out test episodes are required")
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown training scenario: {scenario_name}")
    profiles = tuple(profiles or TRAFFIC_PROFILES)
    unknown_profiles = set(profiles) - set(TRAFFIC_PROFILES)
    if unknown_profiles:
        raise ValueError(f"Unknown test profiles: {sorted(unknown_profiles)}")
    output_dir = output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "experiments",
        "ppo_multiseed",
    )
    os.makedirs(output_dir, exist_ok=True)
    test_seeds = tuple(1_000_001 + i for i in range(test_episodes))
    run_config = {
        "algorithm": "centralized_shared_parameter_ppo",
        "training_scenario": scenario_name,
        "training_seeds": seeds,
        "test_profiles": profiles,
        "test_scenario_seeds": test_seeds,
        "total_timesteps_per_seed": total_timesteps,
        "coordination": coordination,
        "evaluation_coordination_methods": CLAIM_RESOLVERS,
        "matching_controller_note": (
            "Maximum-weight matching allocates from the full feasible graph and "
            "does not use policy actions; it is evaluated as a separate baseline."
        ),
        "training_test_seed_overlap": False,
        "hardware_runtime_recorded_per_seed": True,
        "additional_learning_architectures": (
            ["independent_ppo", "mappo"] if include_other_architectures else []
        ),
        "independent_training_note": (
            "Each local policy is trained with other platforms held idle."
            if include_other_architectures else None
        ),
    }
    with open(os.path.join(output_dir, "experiment_config.json"), "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)

    raw_results = []
    seed_dirs = []
    started = time.perf_counter()
    for profile in profiles:
        for test_seed in test_seeds:
            scenario = traffic_scenario(profile, seed=test_seed)
            for algorithm in ("fcfs", "priority_first", "best_fit", "matching"):
                row = evaluate_episode(
                    scenario=scenario,
                    algorithm=algorithm,
                    evaluation_seed=test_seed,
                )
                row["training_seed"] = None
                row["test_profile"] = profile
                raw_results.append(row)

    for training_seed in seeds:
        seed_dir = os.path.join(output_dir, f"seed_{training_seed}")
        seed_dirs.append(seed_dir)
        model = train(
            total_timesteps=total_timesteps,
            scenario_name=scenario_name,
            seed=training_seed,
            output_dir=seed_dir,
            coordination=coordination,
            device=device,
        )
        independent_models = None
        mappo_model = None
        if include_other_architectures:
            training_factory = SCENARIOS[scenario_name]
            training_scenario = training_factory(training_seed)
            platform_ids = [platform.id for platform in training_scenario.platforms]
            per_platform_timesteps = independent_timesteps or total_timesteps
            independent_dir = os.path.join(seed_dir, "independent")
            independent_models = train_independent_ppo(
                training_factory,
                platform_ids,
                per_platform_timesteps,
                training_seed,
                independent_dir,
                device=device,
            )
            mappo_path = os.path.join(seed_dir, "mappo.pt")
            mappo_model = train_mappo(
                training_factory,
                platform_ids,
                seed=training_seed,
                total_timesteps=total_timesteps,
                output_path=mappo_path,
                device=device,
            )
            with open(os.path.join(seed_dir, "architecture_config.json"), "w", encoding="utf-8") as file:
                json.dump({
                    "independent_ppo": {
                        "actors": "separate local observation policies",
                        "other_platforms_during_training": "held idle",
                        "timesteps_per_platform": per_platform_timesteps,
                    },
                    "mappo": {
                        "actors": "parameter-shared decentralized local actors",
                        "critic": "centralized joint-observation critic",
                        "total_timesteps": total_timesteps,
                    },
                }, file, indent=2)
        seed_results = []
        for profile in profiles:
            scenario_factory = lambda seed, name=profile: traffic_scenario(name, seed=seed)
            for test_seed in test_seeds:
                for resolver in CLAIM_RESOLVERS:
                    row = evaluate_episode(
                        scenario=scenario_factory(test_seed),
                        algorithm="centralized_ppo",
                        model=model,
                        coordination=resolver,
                        evaluation_seed=test_seed,
                    )
                    row["training_seed"] = training_seed
                    row["test_profile"] = profile
                    raw_results.append(row)
                    seed_results.append(row)
                if include_other_architectures:
                    row = evaluate_episode(
                        scenario=scenario_factory(test_seed),
                        algorithm="independent_ppo",
                        independent_models=independent_models,
                        coordination="lowest_id",
                        evaluation_seed=test_seed,
                    )
                    row["training_seed"] = training_seed
                    row["test_profile"] = profile
                    raw_results.append(row)
                    seed_results.append(row)
                    row = evaluate_episode(
                        scenario=scenario_factory(test_seed),
                        algorithm="mappo",
                        model=mappo_model,
                        coordination="lowest_id",
                        evaluation_seed=test_seed,
                    )
                    row["training_seed"] = training_seed
                    row["test_profile"] = profile
                    raw_results.append(row)
                    seed_results.append(row)
        write_jsonl(os.path.join(seed_dir, "evaluation.jsonl"), seed_results)
        with open(os.path.join(seed_dir, "evaluation.json"), "w", encoding="utf-8") as file:
            json.dump(seed_results, file, indent=2)
        del model

    total_seconds = time.perf_counter() - started
    within_seed_summary = _summary(
        raw_results, ("algorithm", "coordination", "training_seed", "test_profile")
    )
    scenario_summary = _summary(
        raw_results,
        ("algorithm", "coordination", "test_profile", "evaluation_seed"),
    )
    seed_performance = _summary(
        raw_results, ("algorithm", "coordination", "training_seed")
    )
    method_summary = _summary(
        raw_results, ("algorithm", "coordination", "test_profile")
    )
    for profile in profiles:
        ranked = sorted(
            (row for row in method_summary if row["test_profile"] == profile),
            key=lambda row: row["avg_wait_mean"],
        )
        for rank, row in enumerate(ranked, start=1):
            row["waiting_time_rank"] = rank
    _write_csv(os.path.join(output_dir, "results.csv"), raw_results)
    _write_csv(os.path.join(output_dir, "within_seed_summary.csv"), within_seed_summary)
    _write_csv(os.path.join(output_dir, "scenario_summary.csv"), scenario_summary)
    _write_csv(os.path.join(output_dir, "training_seed_summary.csv"), seed_performance)
    _write_csv(os.path.join(output_dir, "algorithm_summary.csv"), method_summary)

    payload = {
        "experiment": run_config,
        "total_execution_seconds": total_seconds,
        "uncertainty_definitions": {
            "within_seed_summary": "variation over held-out scenario seeds for one trained model",
            "scenario_summary": "variation across independently trained PPO seeds on the same held-out scenario seed",
            "training_seed_summary": "variation across held-out scenarios summarized separately for each training seed",
            "confidence_interval": "95% Student-t interval where sample count permits",
        },
        "episode_results": raw_results,
        "within_seed_summary": within_seed_summary,
        "scenario_summary": scenario_summary,
        "training_seed_summary": seed_performance,
        "algorithm_summary": method_summary,
    }
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    _plot_training_curves(seed_dirs, output_dir)
    _plot_test_summary(within_seed_summary, output_dir)
    _plot_algorithm_comparison(method_summary, output_dir)
    _plot_training_vs_test(seed_dirs, raw_results, output_dir)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="moderate_traffic")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--test-episodes", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--profiles", nargs="+", choices=list(TRAFFIC_PROFILES), default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--coordination", choices=[
        "lowest_id", "random", "coordinated", "matching",
    ], default="lowest_id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-other-architectures", action="store_true")
    parser.add_argument("--independent-timesteps", type=int, default=None)
    args = parser.parse_args()
    run_multiseed_experiment(
        scenario_name=args.scenario,
        seeds=args.seeds,
        test_episodes=args.test_episodes,
        total_timesteps=args.timesteps,
        output_dir=args.output_dir,
        profiles=args.profiles,
        coordination=args.coordination,
        device=args.device,
        include_other_architectures=args.include_other_architectures,
        independent_timesteps=args.independent_timesteps,
    )
