# Railway Terminus Platform Allocation Research

This repository contains a synthetic railway-terminal allocation environment,
learning controllers, operational heuristics, optimization-based matching,
and live/static visualizations. It is a research simulator, not an operational
Indian Railways decision-support system.

## Architecture

- `env/config.py` defines platforms, trains, compatibility, station
  configuration, and seeded scenarios.
- `env/terminal_env.py` implements the PettingZoo parallel environment,
  observations, train lifecycle, platform turnaround, closures, approach-group
  capacity, rewards, and metrics.
- `env/multiagent_adapter.py` flattens the platform observations for the
  original Stable-Baselines3 controller.
- `training/train.py` trains the centralized reference PPO policy.
- `training/independent.py` trains separate local-observation PPO policies.
- `training/mappo.py` implements decentralized shared actors with a
  centralized training critic.
- `training/policies.py` implements FCFS, priority-first, and best-fit
  operational heuristics.
- `env/allocation.py` implements maximum-weight one-to-one assignment.
- `training/evaluation.py` evaluates controllers on a common environment and
  produces episode metrics.
- `training/experiments.py` runs five-seed PPO experiments on common held-out
  scenarios and writes CSV, JSON, and plots.
- `server/` and `viz/` serve live snapshots and static replays. The signal
  graphics are illustrative; there is no block-signaling/interlocking model.

## Controller definitions

The original/reference PPO policy is **centralized joint-observation PPO**:
one actor receives the concatenated observations of every platform and emits
the whole joint `MultiDiscrete` action. It is neither decentralized
independent PPO nor MAPPO.

The independent PPO implementation trains one separate actor per platform,
using only that platform's local observation and no communication. MAPPO uses
decentralized local-observation actors and a centralized critic during
training. FCFS, priority-first, and best-fit are deterministic operational
heuristics. Maximum-weight matching is an optimization controller, not MARL.

The environment's allocation resolver can be configured as:

- `lowest_id`: historical platform-ID tie-break reference.
- `random`: seeded randomized tie-break.
- `coordinated`: resolve only submitted claims with a coordinated matching.
- `matching`: allocate from the full feasible platform/train graph.

An approach group permits at most one assignment per headway interval.
Groups, headways, and route constraints are simplified synthetic assumptions,
not a validated signaling model.

## Scenarios and data provenance

`default_scenario()` provides a small iteration environment.
`mgr_chennai_central_scenario()` is a synthetic 17-platform layout and
illustrative 62-train schedule. `traffic_scenario(profile, seed)` supports:
`low_traffic`, `moderate_traffic`, `high_traffic`, `peak_hour`,
`platform_disruption`, and `extreme_stress`. Profiles vary train arrival
patterns, dwell times, categories, cancellation/delay rates, closures, and
approach headway.

The included Excel timetable/delay workbooks are currently not ingested by
the environment. Train schedules, dwell times, compatibility, closures, and
approach constraints should be described as synthetic/modelled assumptions
unless independently validated against authoritative operational sources.

## Training and experiments

Install the declared dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Train a centralized PPO model:

```powershell
Push-Location railway_terminal_marl
python main.py --timesteps 300000 --scenario moderate_traffic
Pop-Location
```

Run five independently seeded trainings and evaluate each checkpoint against
the same unseen schedule seeds across all traffic profiles:

```powershell
Push-Location railway_terminal_marl
python -m training.experiments --scenario moderate_traffic --timesteps 300000
Pop-Location
```

Use `--seeds`, `--test-episodes`, `--profiles`, `--coordination`,
`--device`, and `--output-dir` to configure the experiment.
`--include-other-architectures` additionally trains the independent PPO and
MAPPO implementations. Independent actors are trained with other platforms
held idle; this training-distribution difference is recorded and should be
considered when comparing them. Each
`seed_<n>/` directory contains the configuration, software/hardware metadata,
per-episode reward and training metrics, checkpoint, and held-out evaluation
results. The experiment root contains raw episode results, per-training-seed
and per-scenario summaries, confidence intervals, and plots. Training seeds
and test scenario seeds are distinct. The summaries separately label
within-model test-scenario variation and between-training-seed variation.
PPO is compared under the lowest-ID, seeded-random, and coordinated
claim-resolvers. Maximum-weight matching ignores submitted policy actions,
so it is evaluated as a separate optimization baseline.

The checked-in five-seed pilot used only 512 PPO timesteps per seed and two
held-out scenarios per profile. It validates the pipeline, not convergence,
algorithm rankings, or performance claims.

The saved historical checkpoint/logs predate the revised environment and
multi-seed harness. They are not held-out evidence. The historical MGR log
contains 819 training episodes with mean wait about 0.565, mean 62 trains
served, mean 4,125 invalid actions, and mean 43.8 conflicts per episode. Its
saved rollout reports 3,174 invalid actions and 61 conflicts; older claims
that invalid actions reached zero are not supported by these artifacts.

## Live views

From `railway_terminal_marl/`, `python main.py --serve` starts the FastAPI
server. `/station`, `/signals`, `/dashboard`, and `/api/state` show live
inference state. The visuals communicate simulator state only; they do not
reproduce real signal aspects, route locking, block occupancy, or interlocking.
