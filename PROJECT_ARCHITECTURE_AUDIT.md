# Project Architecture and Implementation Audit

> The inspection and findings in the first part of this document are the
> verified **pre-implementation baseline**. The post-implementation status
> below preserves that historical record and distinguishes it from the
> current code and pilot evidence.

## Scope and evidence

This audit describes the checked-in Python source, documentation, data files,
and saved outputs in the repository as inspected on 2026-10-31. It does not
treat README statements as implementation evidence. No LaTeX manuscript,
bibliography, notebook, automated test suite, or evaluation script is present
in the inspected project tree. Consequently, a paper-to-code comparison cannot
be completed until a manuscript is supplied.

The root `rule_engine.py` is a separate, unreferenced collection of rule
predicates and reward values. It is not imported by the railway environment,
trainer, or live server, and its relation to an authoritative railway rulebook
is not established by this repository.

## Repository structure

```text
.
|-- .env                              # empty
|-- .gitignore
|-- .python-version                  # project declares Python >=3.13 elsewhere
|-- .vscode/
|   `-- settings.json
|-- README.md                         # deployment and high-level project notes
|-- Procfile                          # starts nested railway_terminal_marl app
|-- main.py                           # unrelated "Hello" placeholder
|-- pyproject.toml                    # root placeholder; no dependencies
|-- requirements.txt                  # forwards to nested requirements
|-- uv.lock                           # root lock
|-- trains_list.csv                   # not loaded by the current environment
|-- rule_engine.py                    # standalone, unused rule engine
|-- files.zip                         # bundled copy of generated/static files
|-- railway_terminal_marl.zip         # archived project snapshot
|-- data/
|   |-- delays_daterange.xlsx
|   |-- mgr_train_timetable.xlsx
|   `-- train_delays.xlsx
|-- files/
|   |-- dashboard.html
|   |-- mgr_station_simulator.html
|   `-- railway_terminal_marl.zip
`-- railway_terminal_marl/
    |-- .gitignore
    |-- .python-version
    |-- README.md
    |-- main.py
    |-- dashboard.py
    |-- pyproject.toml
    |-- requirements.txt
    |-- uv.lock
    |-- env/
    |   |-- __init__.py
    |   |-- config.py
    |   |-- multiagent_adapter.py
    |   `-- terminal_env.py
    |-- training/
    |   |-- __init__.py
    |   |-- callbacks.py
    |   `-- train.py
    |-- server/
    |   |-- __init__.py
    |   |-- app.py
    |   `-- live_runner.py
    |-- logs/
    |   |-- events.json
    |   |-- metrics.jsonl
    |   `-- ppo_terminus.zip
    `-- viz/
        |-- build.py
        |-- dashboard.html
        |-- dashboard_template.html
        |-- live_dashboard.html
        |-- live_signals.html
        |-- live_station.html
        |-- mgr_station_simulator.html
        `-- mgr_station_simulator_template.html
```

The `.zip` archives are snapshots, not active modules; their contents do not
fully match the current nested project. `files/` and nested `viz/` include
generated/static HTML copies.

## Module inventory and responsibilities

| Module/resource | Observed responsibility |
|---|---|
| `railway_terminal_marl/env/config.py` | `Platform`, `TrainSpec`, `ScenarioConfig`, length compatibility, and two seeded synthetic schedule generators. |
| `railway_terminal_marl/env/terminal_env.py` | PettingZoo `ParallelEnv`: state, observations, discrete claims, compatibility checks, conflicts, reward, episode progression, metrics, and event history. |
| `railway_terminal_marl/env/multiagent_adapter.py` | Flattens all platform observations into one Gymnasium observation and maps one `MultiDiscrete` action vector back to the platform agents. |
| `railway_terminal_marl/training/train.py` | Creates the single environment, configures SB3 PPO, trains, saves one model, and runs one deterministic rollout. |
| `railway_terminal_marl/training/callbacks.py` | Appends a small set of episode fields to JSONL when an episode ends. |
| `railway_terminal_marl/server/live_runner.py` | Loads the saved model, runs deterministic inference in a thread, and builds live snapshots. |
| `railway_terminal_marl/server/app.py` | FastAPI routes and WebSocket streaming for the live views. |
| `railway_terminal_marl/viz/build.py` | Inlines saved events/metrics into static HTML templates. |
| `railway_terminal_marl/dashboard.py` | Separate static dashboard server. |
| Root `rule_engine.py` | Standalone rule-checking methods; no current call sites in the active simulator. |
| `data/*.xlsx`, root `trains_list.csv` | Timetable/delay files exist but are not consumed by the active scenario/training code. |
| `railway_terminal_marl/logs/*` | One saved PPO checkpoint, training JSONL, and one rollout event JSON. |

## Execution flow reconstructed from source

1. **Train generation and timetable ingestion.** `default_scenario(seed)` and
   `mgr_chennai_central_scenario(seed)` use Python's `random.Random(seed)` to
   synthesize train IDs, arrival times, dwell times, categories, and
   compatibility properties. The MGR-named scenario creates a 17-platform
   synthetic layout (12 main plus 5 suburban) and 62 synthetic trains.
   Neither generator ingests the Excel files or CSV. Train labels are
   illustrative; the documentation itself says schedules are not live data.
2. **Arrival and queue management.** `RailwayTerminusEnv.reset()` copies the
   sorted train schedule into `trains_pending`, initializes the queue, and
   calls `_admit_arrivals()`. Each step admits scheduled trains after actions.
   Queue capacity is `ScenarioConfig.max_queue`.
3. **Platform-agent creation.** The environment creates one PettingZoo agent
   named `platform_<id>` per configured platform. A platform can submit a
   claim only when `_is_free()` says it is available.
4. **Observation construction.** `_observe(agent)` provides that platform's
   length-class one-hot, electrification, busy fraction, each visible queue
   entry's wait/length/electrification/priority, normalized time, and
   normalized queue length. Every platform sees the same queue features; the
   platform features differ. There is no explicit action/claim broadcast or
   feasible-action mask.
5. **Action space.** Each platform gets `Discrete(max_queue + 1)`: an index
   into the current queue, or the final index for wait/no-op. The wrapper
   combines these into `MultiDiscrete([max_queue + 1] * num_platforms)`.
6. **Policy architecture and action decoding.** SB3 `PPO("MlpPolicy", ...)`
   receives the concatenation of all platforms' observations and emits the
   whole vector of platform actions. Thus, despite the agents and observations
   being named per-platform, the trained policy is a **centralized,
   joint-observation/joint-action PPO controller with a factorized
   MultiDiscrete action distribution**. It is not decentralized independent
   PPO and is not MAPPO (there is no separate decentralized actor or
   centralized training critic). `split_actions()` maps each vector
   coordinate to an agent.
7. **Conflict resolution and compatibility.** `step()` collects requests,
   groups them by queue index, sorts claimants by platform ID, and awards the
   train to the lowest ID. It then rejects an incompatible winning assignment
   (platform too short or not electrified). The train remains queued but the
   same-step losing claimants are not reconsidered.
8. **Assignment and occupancy.** A valid winning claim removes the train from
   the queue, records an allocation event, sets `platform_busy_until` to
   `t + dwell_time`, and computes `platform_ready_at` by adding the configured
   turnaround buffer. The actual free-platform test does not consult
   `platform_ready_at`.
9. **Reward.** An empty-slot claim incurs -0.5; a losing conflict incurs -0.3;
   an incompatible winner incurs -1.0; a valid allocation receives a
   priority/match reward minus a waiting-time term. Each platform additionally
   receives a shared `-0.02 * queue_length` term.
10. **State transition.** Allocated queue entries are removed; all remaining
    queued trains age one step; time advances; scheduled arrivals are admitted.
    No approach-track, route, or headway state is modeled.
11. **Termination.** At or beyond the scenario horizon, the environment ends.
    It reports normal termination only if queue and pending trains are empty;
    otherwise it reports truncation. No explicit disposition is recorded for
    trains still waiting/pending.
12. **Training/checkpointing.** `train()` constructs one raw environment and
    one Gym wrapper, calls SB3 PPO `.learn(total_timesteps=...)`, and saves to
    the shared `logs/ppo_terminus.zip`. Although `make_training_env()` accepts
    `num_vec_envs`, that argument is unused. It does not pass the seed to PPO
    or explicitly seed Python/NumPy/PyTorch. The scenario seed fixes the
    generated schedule; environment resets do not regenerate it.
13. **Evaluation/output.** `record_rollout()` regenerates the same named
    scenario with the supplied seed, runs one deterministic episode, and
    writes `events.json`. There is no separate validation/test schedule or
    baseline harness. The callback writes `global_step`, wall time, average
    wait, served count, remaining queue count, invalid actions, and conflicts.
14. **Live inference and dashboards.** The FastAPI startup loads the one
    checkpoint. `LiveRunner` iterates the MGR-named synthetic scenario and
    resets with increasing schedule seeds. `snapshot()` computes live
    occupancy and queue state; the HTML clients render it. The static builder
    reads the saved JSON/JSONL and generates HTML.

## Mathematical formulation actually implemented

Let there be \(N\) platforms, queue visibility/capacity \(Q\), time \(t\),
and scheduled trains \(j\). A queued item stores its train specification and
integer waiting counter \(w_j\). The pending list stores not-yet-admitted
trains. Platform \(i\) stores a busy-until time \(b_i\) and a nominal ready
time \(r_i\).

### Observation

The per-platform vector is

\[
o_i(t) =
[\operatorname{onehot}(\ell_i), e_i, u_i(t),
  (w_j/20,\operatorname{onehot}(\ell_j),e_j,p_j)_{j=1}^{Q},
  t/H, |\mathcal{Q}_t|/Q],
\]

where absent queue slots are zero-filled, wait is capped at 1 after
normalization, \(e\) is electrification, \(p_j\) is the stored priority value,
and \(u_i(t)=0\) when the platform is free, otherwise
\(\min(1,(b_i-t)/15)\). Its dimension is \(5+6Q+2=7+6Q\).
The adapter observation is \([o_1,\ldots,o_N]\), of dimension \(N(7+6Q)\).

### Action

Each component \(a_i \in \{0,\ldots,Q\}\). Values below \(Q\) claim queue
indices; \(Q\) means no-op. The policy samples/predicts an \(N\)-component
action from the joint observation.

### Feasibility and allocation

Platform \(i\) is considered free iff \(b_i < t\). For each queue index with
claims, the smallest platform ID wins. A winner is feasible iff
\(\operatorname{rank}(\ell_i)\geq\operatorname{rank}(\ell_j)\), and a train
requiring electrification is assigned only to an electrified platform.
There is no category/section compatibility matrix. For a feasible assignment
at time \(t\), the model sets \(b_i=t+d_j\) and \(r_i=t+d_j+\beta_i\), where
\(d_j\) is dwell and \(\beta_i\) the platform's configured buffer. The
implemented availability check uses \(b_i\), not \(r_i\).

### Reward

For a platform with an empty-slot claim, \(r_i=-0.5\); for a conflict loser,
\(r_i=-0.3\); for an incompatible winning claim, \(r_i=-1.0\). For a feasible
allocation:

\[
r_i =
\left(1.5\;\text{if }p_j=0\text{ else }1.0\right)
\left(1-0.2(\operatorname{rank}(\ell_i)-\operatorname{rank}(\ell_j))\right)
-0.05w_j
-0.02|\mathcal{Q}_{t+1}|.
\]

Every agent receives the shared queue term each step, including agents without
a claim. The wrapper sums all per-agent rewards into one scalar, so this
shared term is multiplied by \(N\) in the scalar training reward. Unassigned
waiting trains have no individual penalty other than the shared queue term.

### Transition and PPO

After allocation, allocated queue indices are removed, remaining waiting
counters increment by one, time increments, and arrivals with scheduled time
at or before the new \(t\) are admitted. PPO uses SB3's standard clipped
surrogate; configured parameters are `n_steps=256`, `batch_size=256`,
`learning_rate=3e-4`, `gamma=0.98`, and `ent_coef=0.01`. Other PPO values use
the installed SB3 defaults; there is no project-defined network, custom loss,
centralized critic, or seed setting in `train.py`.

## Existing experimental methodology and saved output

- The only trainer uses one fixed generated schedule per invocation. Resets
  replay that same schedule; `num_vec_envs=4` is an unused argument.
- The only recorded evaluation is a deterministic rollout on the same
  scenario seed used by training.
- There are no independent-seed experiment folders, held-out scenarios,
  validation selection, repeated test episodes, confidence intervals, or
  baseline comparisons in source.
- The callback emits only six episode-level operational fields; it does not
  record episode reward history, max/median/SD waiting, service percentage,
  utilization over time, priority compliance, execution time, or arrival
  dispositions.
- `logs/metrics.jsonl` contains 819 records (about 300k transitions at the
  366-step MGR scenario horizon). Across those logged training episodes:
  mean `avg_wait` is about 0.565; all records report 62 served trains; mean
  `invalid_actions` is about 4,125 and mean `conflicts` about 43.8. The saved
  final rollout in `events.json` reports 62/62 served, average wait about
  0.323, **3,174 invalid actions**, and 61 conflicts. These saved outputs do
  not substantiate the nested README claim that invalid actions reached zero.
- The log episodes reuse one schedule and are during-training observations;
  they are not independent test data. They cannot establish generalization or
  between-seed variation.

## Data, models, documents, and computational requirements

- **Model:** `railway_terminal_marl/logs/ppo_terminus.zip` (one SB3 checkpoint,
  approximately 3.1 MB).
- **Logs:** `metrics.jsonl` (819 rows) and `events.json` (one 62-train
  synthetic rollout).
- **Spreadsheet inventory:** `mgr_train_timetable.xlsx` has one sheet with
  171 rows/13 columns; delay workbooks contain 30 and 51 rows respectively.
  Their presence does not establish provenance or operational validity, and
  none is loaded by the simulator. The available timetable includes fields
  for train number/name, stations, arrival/departure, and days, but no code
  maps it into `TrainSpec`.
- **Manuscript/supplement:** not present. No code-vs-paper claim can be
  verified.
- **Dependencies:** nested `requirements.txt` lists PettingZoo, SB3,
  Gymnasium, FastAPI, and Uvicorn. The nested `pyproject.toml`/`uv.lock`
  include only Uvicorn despite the project requiring the other packages.
  Root `pyproject.toml` is a placeholder and has no dependencies.
- **Runtime inspected:** Python 3.10.7; NumPy, PyTorch, FastAPI, pandas,
  openpyxl, matplotlib, and SciPy are installed. PettingZoo and
  Stable-Baselines3 are not installed in the inspected interpreter, so the
  active environment/training cannot be executed there without provisioning
  those declared dependencies. The checked-in project metadata says Python
  `>=3.13`, which conflicts with the inspected interpreter and needs
  resolution before claiming a reproducible supported runtime.
- **Hardware/determinism:** no training hardware metadata is recorded.
  Reproducible GPU/deterministic training is not implemented or evidenced.

## Baseline limitations and potential bugs (pre-implementation)

1. **Turnaround is not enforced.** `platform_ready_at` is recorded but never
   used by `_is_free()`. There is also no distinct departure, physical
   clearance, or preparation event.
2. **Segregation is display-only.** `section` groups platforms in the UI;
   train eligibility checks only length and electrification. The synthetic
   MGR generator does not prevent suburban trains from being considered for
   mainline platforms or vice versa.
3. **One train can receive multiple claims.** Conflict handling records each
   losing claim, awards only to the lowest platform ID, and does not retry the
   other agents on their next-best train.
4. **Systematic platform-ID bias.** Deterministic lowest-ID tie-breaking
   favors lower IDs, while conflict claimants receive -0.3. No randomized or
   optimization mechanism is available.
5. **No explicit claim information/mask.** Platform observations do not expose
   competitors' current actions; all queue entries are visible but action
   feasibility is not masked. The central flattened actor does receive all
   platform observations, but the action distribution still chooses
   components without explicit reservation resolution.
6. **Queue overflow silently loses trains.** `_admit_arrivals()` removes an
   arriving train from `trains_pending` before checking capacity. If full, the
   train is discarded without a record or disposition.
7. **Unserved trains have no lifecycle/disposition.** Truncation ends the
   episode even when trains remain queued or pending. They are not represented
   as diverted, cancelled, rejected, or otherwise resolved.
8. **Waiting-time measure is incomplete for operational elapsed time.** A train
   is admitted after the decision at its scheduled time and can only be
   claimed at the next decision; that initial interval is not included in its
   `wait` counter. Only served-train wait contributes to `total_wait`; there
   is no max/median/variance or unserved-train wait.
9. **Utilization is not an episode utilization metric.** The live API reports
   current snapshot occupancy fraction, not time-integrated platform use.
   The environment's episode metrics do not calculate utilization.
10. **Invalid action accounting conflates cases.** Empty-slot claims and
    incompatible claims both increment `invalid_actions`; the UI/logs cannot
    separate them. Direct negative actions are not rejected before Python
    list indexing, though the declared Gym space does not generate them.
11. **Reward shaping is weakly aligned/aggregated.** There is no direct cost
    per unserved train, priority compliance metric, or repeated-claim signal.
    The per-agent shared queue penalty is summed N times by the wrapper.
12. **Insufficient evidence of learning/generalization.** Existing logs are
    repeated training episodes, and the final rollout reuses the same seed.
    Deterministic policies can exploit the fixed schedule; no independent
    held-out evaluation exists.
13. **Documented architecture mismatch.** The nested README refers to
    SuperSuit vectorization/independent per-platform observations, while the
    current source uses a custom wrapper that flattens all platform
    observations and controls a joint action vector. The root architecture
    README is closer on this point but does not describe the verified nonzero
    invalid-action values in the saved output.
14. **No terminal, timetable, or interlocking validation.** The MGR layout is
    an approximation; synthetic labels and schedules are not a validated
    operating plan. Approach tracks, routes, signals, headways, and physical
    movements are absent. The separate `rule_engine.py` is not wired into the
    environment and does not establish a validated signaling model.

## Baseline requirement mapping and implementation plan (pre-change)

| Research requirement | Existing implementation | Missing component | Files to modify/create |
|---|---|---|---|
| Complete architecture audit | Partial README and source; no audit | Consolidated verified baseline and code/paper comparison (paper unavailable) | `PROJECT_ARCHITECTURE_AUDIT.md`; manuscript path if supplied |
| Five-seed PPO and reproducibility | One run path; scenario seed only | End-to-end seed setup, per-seed outputs/config/version metadata/checkpoints | `training/train.py`, `training/callbacks.py`, new experiment runner/utilities |
| Train/validation/test schedules | Same generated schedule for training and rollout | Split scenario seeds and shared held-out test sets | `env/config.py`, new evaluation module |
| Statistical metrics/plots | Small JSONL and one event replay | Episode/scenario/seed separation, confidence intervals, plots, CSV/JSON | New evaluation/statistics/plotting modules |
| Challenging configurable scenarios | Two synthetic generators | Six load/disruption profiles, arrival/delay/cancellation/closure parameters, reproducible seeds | `env/config.py`, new tests |
| Coordination alternatives | Lowest-ID tie-break only | Random tie-break, explicit coordinated assignment, max-weight matching, conflict diagnostics | `env/terminal_env.py`, new allocation module/config |
| Fair algorithm benchmarks | No operational benchmarks | FCFS, priority-first, best-fit, independent/centralized PPO, true MAPPO, matching | New policies/benchmark runner; PPO training code |
| Turnaround/segregation realism | Buffer value stored; section label stored | Availability enforcement, station compatibility matrix | `env/config.py`, `env/terminal_env.py` |
| Approach-track occupancy | None | Documented simplified route/headway constraints | `env/config.py`, `env/terminal_env.py` |
| Admission/lifecycle correctness | Queue overflow drops trains; no dispositions | Explicit train states and overflow/diversion/cancellation accounting | `env/config.py`, `env/terminal_env.py`, evaluation metrics |
| Data validation and provenance | Synthetic scenarios plus unused spreadsheets | Provenance, timetable adapter/verification, explicit synthetic assumptions | `env/config.py`, documentation, optional data ingestion module |

## Original implementation order (historical; superseded by status below)

1. Add focused environment tests for current behavior and expose correctness
   defects (buffer, overflow, action bounds, termination/lifecycle, metrics).
2. Add explicit train lifecycle, operational configuration, safe queue
   admission, station compatibility, turnaround, and simplified approach
   occupancy. Preserve a documented legacy mode if backward comparison is
   required.
3. Implement a common allocation interface with the original, randomized,
   coordinated, and maximum-weight matching resolvers; test uniqueness and
   feasibility invariants.
4. Correct the training wrapper/seed plumbing and record complete run
   metadata. Keep centralized PPO, independent PPO, and MAPPO names distinct;
   do not infer MAPPO from the current joint-action policy.
5. Build scenario generation and common schedule evaluation for baselines and
   trained policies. Evaluate all algorithms on the same held-out episodes.
6. Produce uncertainty-aware CSV/JSON and plots, then update README and the
   audit with measured results. Record hardware/runtime and label the station
   assumptions as synthetic unless independently validated.

No experiment should be reported as completed until its inputs, seed, model,
and outputs are saved and its evaluation schedule is demonstrably held out.

## Post-implementation status

The following changes were made after the baseline audit. They address much
of the proposed work, but do not validate the simulator against a real
terminal, signaling plan, or authoritative timetable.

### Implemented

- Added six deterministic synthetic traffic profiles, explicit train
  lifecycle/disposition records, platform/category eligibility, closures,
  cancellations and delays, turnaround enforcement, and simplified approach
  groups/headways.
- Added seeded random, coordinated-claim, and maximum-weight matching
  allocation resolvers while preserving the lowest-ID reference mode.
- Added FCFS, priority-first, and best-fit baselines, independent local PPO,
  and MAPPO with decentralized actors and a centralized critic. The
  independent actors are trained with other platforms held idle.
- Added per-episode operational metrics and common held-out evaluation,
  multi-seed PPO artifacts, statistical summaries, CSV/JSON outputs, and
  plots. Training seeds and test scenario seeds are kept separate.
- Updated the root and nested READMEs and dependency metadata. Added nine
  focused environment regression tests.

### Verification and pilot evidence

The nine environment tests pass. PettingZoo's Parallel API test and
Stable-Baselines3's Gymnasium `check_env` pass, and centralized PPO,
independent PPO, and MAPPO each completed smoke training.

The saved pilot used five centralized PPO training seeds (`11, 22, 33, 44,
55`), **512 timesteps per seed**, and two held-out scenario seeds
(`1000001, 1000002`) for each of the six profiles. Optional independent PPO
used 256 timesteps per platform; the MAPPO checkpoint was also produced.
The pilot contains 348 episode-result rows. Each of the five seed folders
contains the expected centralized checkpoint, metadata, training/evaluation
logs, MAPPO checkpoint, and eight independent actor checkpoints.

These are **pipeline-validation artifacts, not research-grade performance
evidence**. The PPO budgets are far below the configured 300,000-step
experiment example, and two test schedules per profile are insufficient for
strong statistical claims. Do not rank algorithms, claim convergence, or
claim superiority from this pilot. A full-budget run with a suitably larger
held-out set remains necessary.

### Remaining limitations

- All scenarios and operating rules are synthetic. Excel timetables are
  still not ingested, and the standalone root `rule_engine.py` is still not
  integrated or validated.
- Approach groups/headways are simplified capacity constraints, not a
  signaling, interlocking, block-occupancy, or route-locking simulation.
- Independent PPO's idle-platform training distribution differs from
  concurrent multi-platform execution and must be disclosed in comparisons.
- The `.python-version` files request Python 3.13 while package metadata
  permits Python 3.10 and the validation runtime used Python 3.10.7. The
  supported package floor and preferred development interpreter are distinct;
  compatibility at the declared floor does not certify every dependency on
  every platform.
- Historical checkpoints/logs predate the revised observation semantics.
  Their shapes may remain loadable, but they are not evidence of performance
  in the revised environment; retraining is recommended.
