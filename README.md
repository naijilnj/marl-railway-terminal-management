# Railway Terminal MARL - System Architecture Reference

## 1. Overview

A synthetic railway-terminus platform-allocation research simulator with a
PettingZoo environment, several learning/heuristic/optimization controllers,
multi-seed experiments, live inference, and static dashboards. It is not an
operational Indian Railways decision-support system.

Platforms are PettingZoo agents and trains wait in a shared queue. The
reference SB3 PPO policy receives all platform observations concatenated and
selects one joint action vector. This is centralized joint-observation PPO,
not independent PPO or MAPPO.

| Concern | File |
|---|---|
| Environment rules | `terminal_env.py` |
| Scenario / data generation | `config.py` |
| Training loop | `train.py` |
| Multi-agent flattening adapter | `multiagent_adapter.py` |
| Live runner (background inference loop) | `live_runner.py` |
| Live HTTP/WebSocket app | `app.py` |
| Static dashboard build | `build.py` |

---

## 2. Core Environment Logic

### 2.1 Observation space

Each platform's observation vector is built from three groups of features:

- **Platform features** — length-class one-hot, encoded platform category,
  busy/closure fraction
- **Queue features** (per visible waiting train) — wait time, train length
  class, electrification requirement, encoded train category/priority
- **Global features** — normalized time, normalized queue size

For a single platform, the observation length is:

$$d = 5 + 6Q + 2$$

where $Q$ is `max_queue`.

In the training wrapper, per-platform observations are concatenated, so
the model sees:

$$D = N \cdot d$$

where $N$ is the number of platforms.

### 2.2 Action space

Actions are discrete claims on queue slots. For each free platform:

$$a \in \{0, 1, \dots, Q\}$$

where the last action ($a = Q$) means **wait / no-op**.

### 2.3 Per-tick environment logic

Each tick, `terminal_env.py`:

1. Collects claims from all free platforms.
2. Uses the configured resolver: lowest ID, seeded random, coordinated
   matching over claims, or maximum-weight matching.
3. Enforces train/platform category, length, electrification, closure,
   turnaround-buffer, and approach-group constraints.
4. Allocates valid trains and records lifecycle events.
5. Advances waiting time and admits, diverts, or cancels trains explicitly.
6. Applies the shared queue penalty.

### 2.4 Reward function

$$
r =
\begin{cases}
-0.5 & \text{empty-slot claim} \\[4pt]
-0.3 & \text{conflict loser} \\[4pt]
-1.0 & \text{incompatible allocation} \\[4pt]
b \cdot m - 0.05\,w - 0.02\,n & \text{valid allocation + shared term}
\end{cases}
$$

where:

- $b = 1.5$ for priority-0 (express/premium) trains, else $b = 1.0$
- $m = 1 - 0.2(\ell_p - \ell_t)$ — the length-match factor, where $\ell_p$
  and $\ell_t$ are the platform's and train's length-class ranks
- $w$ — the train's waiting time
- $n$ — number of trains still waiting (shared across all platforms)

Platform length ordering is **short < medium < long**, defined in `config.py`.
The shared queue term is added to every agent reward and then summed by the
Gym adapter, so its scalar contribution scales with the number of agents.

---

## 3. Training Setup

Training is off-the-shelf **Stable-Baselines3 PPO**, configured in `train.py`.

> **Adapter change:** SuperSuit was removed and replaced with a custom
> flattening adapter, `multiagent_adapter.py`, so the project trains on
> Windows without SuperSuit's C++ build dependency.

The adapter exposes the full concatenated observation and a joint action
vector. It does not train a separate decentralized actor for every platform.

| Hyperparameter | Value |
|---|---|
| Algorithm | PPO |
| Policy | `MlpPolicy` |
| `n_steps` | 256 |
| `batch_size` | 256 |
| `learning_rate` | 3e-4 |
| `gamma` | 0.98 |
| `ent_coef` | 0.01 |

The reference controller is centralized PPO over flattened global
observations and a joint `MultiDiscrete` action. Separate implementations of
independent local PPO and MAPPO are also available. FCFS, priority-first, and
best-fit are operational heuristics; maximum-weight matching is an
optimization controller.

**Training flow:**

1. Build seeded scenario instances from `config.py`.
2. Wrap the multi-agent environment with `multiagent_adapter.py`.
3. Seed Python, NumPy, PyTorch, PPO, and environment randomization.
4. Train PPO and log episode-level metrics.
5. Save model/configuration/software/hardware metadata.
6. Evaluate against shared held-out scenario seeds in
   `experiments/ppo_multiseed/`.

**PPO objective** (standard SB3 clipped surrogate — no custom loss added):

$$
L^{clip}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\hat{A}_t\right)\right]
$$

---

## 4. Scenario / Data Generation

`config.py` defines toy and synthetic MGR-style generators plus six
configurable traffic profiles:

- **`default_scenario()`** — 6 platforms, 24 trains, random-but-seeded timing.
- **`mgr_chennai_central_scenario()`** — 17 platforms, 62 trains,
  approximating MGR Chennai Central's real structure: 12 main-terminal
  platforms (including bay platform **2A**) plus 5 suburban / Moore
  Market Complex platforms.
- **`traffic_scenario(profile, seed)`** — low, moderate, high, peak-hour,
  platform-disruption, and extreme-stress scenarios.

Schedules, dwell times, labels, categories, delays, closures, and approach
headways are synthetic/modelled assumptions. The checked-in Excel timetable
and delay workbooks are not ingested by the current environment.

> The MGR-named layout is an approximation. Do not describe its timetable,
> platform eligibility, or approach constraints as verified operating data.

---

## 5. Live Inference

`live_runner.py`:

1. Loads the saved PPO model (`logs/ppo_terminus.zip`).
2. Starts the MGR Chennai Central scenario.
3. Continuously steps the environment in the background.
4. On episode end, resets with a new seed — the control room runs
   indefinitely rather than stopping after one fixed episode.

`app.py` exposes a WebSocket endpoint that streams snapshots to the
browser. Each snapshot includes:

- Platform occupancy
- Queue contents
- Upcoming arrivals
- Recent allocation and conflict events
- Per-episode and all-time KPIs
- Utilization by section (main vs. suburban)
- Train lifecycle states, platform compatibility/closures, approach-group
  status, and time-average utilization

---

## 6. UI Surfaces

Five distinct UI surfaces exist in the project:

| File | Type | Description |
|---|---|---|
| `live_station.html` | Live, WebSocket | Station board — platforms, occupancy, current trains, dwell bars, queue, allocation ticker |
| `live_signals.html` | Live, WebSocket | Signal/interlocking-style schematic — platform branches, signal state, contention alerts |
| `live_dashboard.html` | Live, WebSocket | Operations dashboard — episode progress, utilization, queue depth, arrivals, recent allocations, alerts |
| `dashboard.html` | Static | Training telemetry, generated from `logs/metrics.jsonl` by `build.py` |
| `mgr_station_simulator.html` | Static | Replay simulator, generated from `logs/events.json` by `build.py` |

The three `live_*` UIs connect to the real running model through
`live_runner.py` and `app.py` — they are not toy playback pages.

---

## 7. Logs

| File | Contents |
|---|---|
| `logs/metrics.jsonl` | One row per completed training episode |
| `logs/events.json` | One deterministic rollout, used to build the static replay UI |

The historical `logs/metrics.jsonl`, `logs/events.json`, and
`logs/ppo_terminus.zip` predate the revised environment and are not held-out
experimental evidence. `experiments/ppo_multiseed/` stores separate seeded
runs, shared held-out evaluation results, summary CSV/JSON, and plots.

Run the five-seed experiment from the nested project directory:

```powershell
Push-Location railway_terminal_marl
python -m training.experiments --scenario moderate_traffic --timesteps 300000
Pop-Location
```

`--include-other-architectures` additionally trains independent PPO and MAPPO;
the independent actors currently train with other platforms held idle, so
that training-distribution difference is recorded in experiment metadata.
Centralized PPO is evaluated with the lowest-ID, seeded-random, and
coordinated claim-resolvers. Maximum-weight matching ignores submitted policy
actions, so it is reported as a separate optimization baseline.
The checked-in five-seed pilot used only 512 PPO timesteps per seed and two
held-out scenarios per profile; it is pipeline validation, not evidence of
convergence or algorithm superiority.
The plots and aggregate CSV/JSON are generated from the same held-out
scenarios used by the baselines.

---

## 8. Current Status

Working now:

- Synthetic configurable traffic profiles and reproducible train lifecycle
- Turnaround, compatibility, closures, and simplified approach-group limits
- Four allocation resolvers and three operational heuristics
- Centralized PPO, independent PPO, and MAPPO implementations
- Five-seed experiment pipeline and held-out evaluation artifacts
- Live WebSocket-fed UI
- Model training, metrics, and replay logs

The approach-group model is an intentionally simplified resource constraint,
not a validated signaling/interlocking simulation. No manuscript is present
in the repository for a code-to-paper comparison. See
`PROJECT_ARCHITECTURE_AUDIT.md` for the verified baseline and implementation
limitations.

---

## 9. File Map (quick reference)

```
Environment rules ................. terminal_env.py
Scenario / data generation ........ config.py
Training .......................... train.py
Multi-agent flattening adapter .... multiagent_adapter.py
Live runner ........................ live_runner.py
Live HTTP/WebSocket app ............ app.py
Static dashboard build ............. build.py
Research audit ..................... PROJECT_ARCHITECTURE_AUDIT.md
Experiment runner .................. training/experiments.py
```

## Railway deployment

This repo is now Railway-ready from the workspace root:

- `Procfile` starts the live backend with `cd railway_terminal_marl && python main.py --serve`
- root `requirements.txt` forwards to `railway_terminal_marl/requirements.txt`
- `railway_terminal_marl/main.py` now binds to Railway's `PORT` environment variable when present

Recommended Railway setup:

1. Point Railway at the repo root.
2. Let Railway detect `requirements.txt` and `Procfile`.
3. Deploy the web service.
4. Open `/station`, `/signals`, and `/dashboard` on the Railway URL.

If you want a static-only deployment, host `railway_terminal_marl/viz` separately and keep Railway for the live backend.
