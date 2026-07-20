# Railway Terminus MARL — Platform Allocation

A working scaffold for allocating trains to platforms at a rail **terminus**
(a terminal/end station with several platforms, e.g. Chennai Central,
Howrah) using multi-agent reinforcement learning.

## How the problem is modeled

- **Agents = platforms** (fixed set, e.g. 6). This is the key design choice:
  the number of platforms at a terminus doesn't change mid-episode, so it
  drops cleanly into PettingZoo's `ParallelEnv` + SuperSuit + Stable-
  Baselines3 with **parameter sharing** (one policy, reused by every
  platform).
- **Trains** arrive on a schedule into a shared waiting queue (visible to
  every platform agent, padded to `max_queue`).
- Each timestep, every **free** platform picks: "accept queue slot *i*" or
  "wait". The environment resolves conflicts (two platforms picking the
  same train — lowest platform id wins), checks compatibility (platform
  length class ≥ train length class, electrification if required), and
  computes reward from match quality, wait-time penalty, and a small
  shared team term for overall queue length.
- This mirrors the real allocation problem: express vs local priority,
  platform length/electrification constraints, turnaround buffers between
  trains, and the goal of minimizing average dwell/wait while keeping
  every platform utilized.

## Scenarios

- `default_scenario()` — a small 6-platform toy scenario, useful for fast
  iteration.
- `mgr_chennai_central_scenario()` — approximates **MGR Chennai Central
  (MAS)**'s real platform structure: 17 platforms total — 12 in the main
  building (including the short bay platform **"2A"**, used in real life
  for premium short-rake trains like the Rajdhani/Shatabdi services) plus
  5 at the **Moore Market Complex** for suburban EMU services, all
  electrified. Traffic mix: ~16 long-distance express/premium trains and
  ~46 suburban EMU services over a ~366-step horizon. Train names/numbers
  and exact schedules are illustrative flavor, not a live timetable — the
  *platform structure* (12 main + 5 suburban + the 2A bay) is the real,
  sourced part.

## Project layout

```
railway_terminal_marl/
  env/
    config.py          # Platform + train scenario definitions (default + MGR Chennai Central)
    terminal_env.py     # PettingZoo ParallelEnv: RailwayTerminusEnv
  training/
    train.py            # SuperSuit + SB3 PPO training, parameter-shared, --scenario flag
    callbacks.py         # Logs per-episode metrics to logs/metrics.jsonl
  viz/
    live_station.html                      # Live station bay view (fetches from the server)
    live_signals.html                       # Live signaling-schematic board (fetches from the server)
    live_dashboard.html                     # Live operations dashboard (fetches from the server)
    mgr_station_simulator_template.html/.html  # Static replay (superseded by the live views, kept for reference)
    dashboard_template.html / dashboard.html   # Training curves — still static, still separate
    build.py                                # Regenerates dashboard.html from logs/metrics.jsonl
  server/
    live_runner.py       # Steps the real env with the trained model forever, in a thread
    app.py                # FastAPI: WebSocket + REST + serves the two live HTML pages
  dashboard.py           # Separate static dashboard server for training telemetry
  logs/
    ppo_terminus.zip     # Trained model (MGR Chennai Central scenario)
    metrics.jsonl        # Per-episode training metrics
    events.json          # One clean rollout: allocation + conflict events
```

## Running it

```bash
pip install -r requirements.txt

# train on the MGR Chennai Central scenario (also runs one deterministic
# rollout at the end and writes events.json for the static dashboard)
python main.py --timesteps 300000 --scenario mgr_chennai_central

# start the live server — runs the trained policy against the real env
# forever in a background thread, looping to a fresh episode whenever one ends
python main.py --serve
```

Then open **http://localhost:8000/** — it links to:

- **`/station`** — live station bay view (Main Terminal + Suburban Terminal /
  Moore Market Complex), streamed over a WebSocket from the actual running
  policy. No play/scrub controls — it's always showing the current live
  state, and episodes loop forever so it never runs out of runway.
- **`/signals`** — a signaling-schematic board (approach lines fanning into
  buffer stops per platform, signal heads, a contention/alert panel) in the
  style of an interlocking control board, also live.
- **`/dashboard`** — a live operations dashboard for queue depth, platform
  utilization, upcoming arrivals, and recent allocations.
- **`/api/state`** — raw JSON snapshot, if you want to build another view.

`viz/dashboard.html` (training metrics) is still a separate static file.
Regenerate it from a training run with `python dashboard.py` or directly
with `python viz/build.py`.

## Who's deciding what, and what model is running

- **Agents = platforms.** Each of the 17 platforms (`platform_0`...`platform_16`)
  is its own PettingZoo agent. Every timestep, every *free* platform looks at
  the shared waiting queue and picks "accept queue slot *i*" or "wait." That's
  the entire decision surface — there's no separate dispatcher, no train-side
  agent, and (important caveat) **no signaling/track agent**: the env models
  platform allocation only, not interlocking or approach-track conflicts. The
  `/signals` view's "signal heads" reflect platform occupancy, not real block
  signalling.
- **Model = a single shared PPO policy** (Stable-Baselines3 `MlpPolicy`),
  trained once and reused by all 17 platforms via SuperSuit's parameter-sharing
  vectorization. It's independent PPO with tied weights, not a centralized-critic
  method like MAPPO/QMIX — which is the likely next upgrade if you want to
  reduce platform-vs-platform contention (see below).

## What actually trained

**MGR Chennai Central scenario** (17 platforms, 62 trains, horizon 366),
300k timesteps of PPO:

- Average wait per train: **~1.0 → ~0.4** timesteps, all 62 trains served
- Invalid (incompatible-platform) actions: **→ 0** in the final rollout
- Platform-claim conflicts stayed nontrivial (~560 in the final rollout) —
  since platforms act independently with no direct communication, several
  often "race" for the same desirable train. This doesn't hurt throughput
  here (queue is large enough to absorb it) but is the main thing worth
  improving next — see below.

**Small toy scenario** (6 platforms, 24 trains, horizon 220), 200k
timesteps: average wait **~4.5 → ~0.2**, invalid actions **~940 → 0**.

## Extending this

- **Reduce platform-claim conflicts**: platforms currently act independently
  (parameter sharing, no communication). A centralized critic (MAPPO) or an
  explicit "claim broadcast" in the observation (what did other platforms
  just bid on) would likely cut conflicts a lot on the 17-platform scenario.
- **Bigger/networked scenarios**: swap in a generator with more platforms,
  delayed/disrupted arrivals, or maintenance windows. The env doesn't
  hardcode platform count.
- **Live simulation/dashboard**: the live station, signal board, and live
  dashboard all read the same websocket stream from `server/app.py`. The
  static `viz/dashboard.html` still exists for offline training telemetry.
- **Realistic constraints**: add track conflicts (shared approach lines
  before the platforms), crew/rolling-stock turnaround rules, or real MAS
  timetable data in place of the illustrative train schedule.
