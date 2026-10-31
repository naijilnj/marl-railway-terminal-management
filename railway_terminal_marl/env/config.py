"""
Scenario configuration for the railway terminus MARL environment.

A "terminal" here is a single terminus station with several platforms.
Trains arrive according to a schedule and must be allocated to a
compatible platform. Agents = platforms (fixed set). Trains form a
shared waiting queue that every platform agent observes.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

LENGTH_CLASSES = ["short", "medium", "long"]  # ordinal: short < medium < long


@dataclass
class Platform:
    id: int
    length_class: str      # max train length this platform can accept
    electrified: bool      # whether platform has electrified line / OHE
    turnaround_buffer: int = 2  # min timesteps needed between departure and next arrival
    label: str = ""         # display label, e.g. "2A" — defaults to str(id) if unset
    section: str = "main"    # display grouping, e.g. "main" vs "suburban"
    compatible_categories: List[str] = field(default_factory=list)
    approach_group: str = "default"
    closure_intervals: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        if not self.label:
            self.label = str(self.id)


@dataclass
class TrainSpec:
    id: int
    scheduled_arrival: int   # timestep the train enters the waiting queue
    dwell_time: int          # timesteps the train occupies the platform once allocated
    length_class: str        # required platform length class (platform must be >= this)
    electrified_required: bool
    priority: int             # 0 = express (higher priority), 1 = local
    label: str = ""           # display name, e.g. "12658 Chennai Mail" — defaults to "Train {id}"
    category: str = ""        # display category, e.g. "Express", "Suburban EMU", "Premium"
    arrival_delay: int = 0
    unexpected: bool = False
    cancelled: bool = False

    def __post_init__(self):
        if not self.label:
            self.label = f"Train {self.id}"
        if not self.category:
            self.category = "Express" if self.priority == 0 else "Local"


def length_ok(platform_class: str, train_class: str) -> bool:
    order = {c: i for i, c in enumerate(LENGTH_CLASSES)}
    return order[platform_class] >= order[train_class]


@dataclass
class ScenarioConfig:
    horizon: int = 200
    max_queue: int = 10          # max waiting trains visible in observation (padded)
    platforms: List[Platform] = field(default_factory=list)
    trains: List[TrainSpec] = field(default_factory=list)
    name: str = "custom"
    approach_headway: int = 1
    arrival_horizon: Optional[int] = None


def default_scenario(seed: int = 0) -> ScenarioConfig:
    """A modest, reproducible scenario: 6 platforms, ~24 trains over 200 steps."""
    import random
    rng = random.Random(seed)

    platforms = [
        Platform(id=0, length_class="long", electrified=True, turnaround_buffer=3),
        Platform(id=1, length_class="long", electrified=True, turnaround_buffer=3),
        Platform(id=2, length_class="medium", electrified=True, turnaround_buffer=2),
        Platform(id=3, length_class="medium", electrified=False, turnaround_buffer=2),
        Platform(id=4, length_class="short", electrified=False, turnaround_buffer=1),
        Platform(id=5, length_class="short", electrified=False, turnaround_buffer=1),
    ]

    trains = []
    t = 0
    for i in range(24):
        t += rng.randint(4, 10)
        length_class = rng.choices(LENGTH_CLASSES, weights=[0.3, 0.4, 0.3])[0]
        electrified_required = rng.random() < 0.4 and length_class != "short"
        priority = 0 if rng.random() < 0.25 else 1
        dwell = rng.randint(6, 14) if priority == 1 else rng.randint(4, 8)
        trains.append(TrainSpec(
            id=i,
            scheduled_arrival=t,
            dwell_time=dwell,
            length_class=length_class,
            electrified_required=electrified_required,
            priority=priority,
        ))

    return ScenarioConfig(
        horizon=220, max_queue=10, platforms=platforms, trains=trains, name="default"
    )


# Illustrative long-distance train names, flavored on real MAS-originating services.
# (Names/numbers are for display flavor in the simulator, not sourced from a live timetable.)
_EXPRESS_NAMES = [
    "12163 Chennai Mail", "12621 Tamil Nadu Express", "12433 Chennai Rajdhani",
    "12695 Trivandrum Mail", "12291 Chennai Duronto", "12007 Chennai Shatabdi",
    "12839 Howrah Mail", "12027 Chennai Shatabdi", "12605 Chennai Express",
    "22625 Chennai SF Express", "12665 Ernakulam Exp", "12610 Chennai Express",
    "22638 Nagercoil Exp", "12603 Chennai Karnataka Exp", "12677 Chennai Ernakulam",
]
_SUBURBAN_NAMES = [
    "Chennai Beach EMU", "Tambaram Local", "Gummidipoondi EMU", "Avadi Local",
    "Tiruvallur EMU", "Arakkonam Local", "Chengalpattu EMU", "Velachery Local",
]


def mgr_chennai_central_scenario(seed: int = 0) -> ScenarioConfig:
    """
    Approximate model of MGR Chennai Central (MAS): a terminus with 17
    platforms — 12 in the main building (including the short bay platform
    "2A" used for premium short-rake trains like Rajdhani/Shatabdi
    services) plus 5 at the Moore Market Complex handling suburban EMU
    services. Electrified throughout (since 1931).

    This reflects the station's real platform *structure* (12 main + 5
    suburban, with a 2A bay), not an exact live timetable — arrivals,
    dwell times, and train-to-platform assignments below are illustrative.
    """
    import random
    rng = random.Random(seed)

    platforms: List[Platform] = []
    main_labels = ["1", "2A", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    for i, lbl in enumerate(main_labels):
        if lbl == "2A":
            # short bay platform for premium short-rake trains
            platforms.append(Platform(id=i, length_class="short", electrified=True,
                                       turnaround_buffer=2, label=lbl, section="main",
                                       compatible_categories=["Express", "Premium"],
                                       approach_group="main_throat"))
        else:
            platforms.append(Platform(id=i, length_class="long", electrified=True,
                                       turnaround_buffer=5, label=lbl, section="main",
                                       compatible_categories=["Express", "Premium"],
                                       approach_group="main_throat"))

    suburban_labels = ["13", "14", "15", "16", "17"]
    offset = len(main_labels)
    for j, lbl in enumerate(suburban_labels):
        platforms.append(Platform(id=offset + j, length_class="medium", electrified=True,
                                   turnaround_buffer=2, label=lbl, section="suburban",
                                   compatible_categories=["Suburban EMU"],
                                   approach_group="suburban_throat"))

    trains: List[TrainSpec] = []
    train_id = 0

    # Long-distance / express / premium trains -> main platforms
    t = rng.randint(5, 15)
    for i in range(16):
        name = rng.choice(_EXPRESS_NAMES)
        is_premium = "Rajdhani" in name or "Shatabdi" in name
        length_class = "short" if is_premium else rng.choices(["medium", "long"], weights=[0.3, 0.7])[0]
        dwell = rng.randint(8, 14) if is_premium else rng.randint(14, 26)
        trains.append(TrainSpec(
            id=train_id, scheduled_arrival=t, dwell_time=dwell,
            length_class=length_class, electrified_required=True,
            priority=0, label=f"{name} #{train_id}",
            category="Premium" if is_premium else "Express",
        ))
        train_id += 1
        t += rng.randint(14, 26)

    # Suburban EMU services -> suburban platforms, frequent, short dwell
    t = rng.randint(2, 6)
    for i in range(46):
        name = rng.choice(_SUBURBAN_NAMES)
        dwell = rng.randint(3, 6)
        trains.append(TrainSpec(
            id=train_id, scheduled_arrival=t, dwell_time=dwell,
            length_class="medium", electrified_required=(rng.random() < 0.9),
            priority=1, label=f"{name} #{train_id}", category="Suburban EMU",
        ))
        train_id += 1
        t += rng.randint(5, 10)

    horizon = max(tr.scheduled_arrival + tr.dwell_time for tr in trains) + 20
    return ScenarioConfig(
        horizon=horizon, max_queue=16, platforms=platforms, trains=trains,
        name="mgr_chennai_central",
    )


TRAFFIC_PROFILES = {
    "low_traffic": {
        "train_count": 20, "horizon": 300, "arrival_rate": 0.08,
        "arrival_distribution": "uniform", "dwell_min": 5, "dwell_max": 10,
        "cancellation_rate": 0.0, "unexpected_rate": 0.0, "closure_count": 0,
    },
    "moderate_traffic": {
        "train_count": 60, "horizon": 300, "arrival_rate": 0.22,
        "arrival_distribution": "poisson", "dwell_min": 8, "dwell_max": 18,
        "cancellation_rate": 0.01, "unexpected_rate": 0.03, "closure_count": 0,
    },
    "high_traffic": {
        "train_count": 110, "horizon": 240, "arrival_rate": 0.5,
        "arrival_distribution": "bursty", "dwell_min": 12, "dwell_max": 25,
        "cancellation_rate": 0.02, "unexpected_rate": 0.05, "closure_count": 0,
    },
    "peak_hour": {
        "train_count": 150, "horizon": 240, "arrival_rate": 0.7,
        "arrival_distribution": "peak", "dwell_min": 10, "dwell_max": 24,
        "cancellation_rate": 0.02, "unexpected_rate": 0.08, "closure_count": 0,
    },
    "platform_disruption": {
        "train_count": 100, "horizon": 240, "arrival_rate": 0.5,
        "arrival_distribution": "bursty", "dwell_min": 12, "dwell_max": 25,
        "cancellation_rate": 0.03, "unexpected_rate": 0.08, "closure_count": 2,
    },
    "extreme_stress": {
        "train_count": 200, "horizon": 220, "arrival_rate": 1.0,
        "arrival_distribution": "peak", "dwell_min": 16, "dwell_max": 32,
        "cancellation_rate": 0.04, "unexpected_rate": 0.12, "closure_count": 3,
    },
}


def traffic_scenario(
    profile: str = "moderate_traffic",
    seed: int = 0,
    overrides: Optional[Dict[str, object]] = None,
    platform_count: int = 8,
) -> ScenarioConfig:
    """Create one reproducible synthetic load/disruption scenario."""
    import math
    import random

    if profile not in TRAFFIC_PROFILES:
        raise ValueError(f"Unknown traffic profile: {profile}")
    if platform_count < 2:
        raise ValueError("platform_count must be at least 2")
    settings = dict(TRAFFIC_PROFILES[profile])
    if overrides:
        unknown = set(overrides) - set(settings)
        if unknown:
            raise ValueError(f"Unsupported scenario overrides: {sorted(unknown)}")
        settings.update(overrides)
    if settings["arrival_distribution"] not in {"uniform", "poisson", "peak", "bursty"}:
        raise ValueError(f"Unsupported arrival distribution: {settings['arrival_distribution']}")
    if not 0 <= float(settings["cancellation_rate"]) <= 1:
        raise ValueError("cancellation_rate must be between 0 and 1")
    if not 0 <= float(settings["unexpected_rate"]) <= 1:
        raise ValueError("unexpected_rate must be between 0 and 1")
    if int(settings["dwell_min"]) < 1 or int(settings["dwell_max"]) < int(settings["dwell_min"]):
        raise ValueError("dwell range must be positive and ordered")

    rng = random.Random(seed)
    main_count = max(1, math.ceil(platform_count * 0.625))
    platforms = []
    for platform_id in range(platform_count):
        mainline = platform_id < main_count
        platforms.append(Platform(
            id=platform_id,
            length_class=("long" if platform_id % 3 else "medium") if mainline else "medium",
            electrified=True,
            turnaround_buffer=4 if mainline else 2,
            label=str(platform_id + 1),
            section="main" if mainline else "suburban",
            compatible_categories=["Express", "Premium", "Passenger"] if mainline else ["Suburban EMU"],
            approach_group="main_throat" if mainline else "suburban_throat",
        ))

    count = int(settings["train_count"])
    horizon = int(settings["horizon"])
    rate = float(settings["arrival_rate"])
    if count < 0 or horizon < 1 or rate <= 0:
        raise ValueError("train_count must be nonnegative and horizon/arrival_rate positive")
    distribution = str(settings["arrival_distribution"])
    arrival_samples = []
    current_time = 0.0
    for index in range(count):
        if distribution == "uniform":
            current_time = index / rate
        elif distribution == "poisson":
            current_time += rng.expovariate(rate)
        elif distribution == "peak":
            progress = (current_time % horizon) / horizon
            current_time += rng.expovariate(rate * (1.7 if 0.35 <= progress <= 0.7 else 0.45))
        else:
            current_time += rng.expovariate(rate * (2.0 if index % 8 < 3 else 0.7))
        planned_arrival = max(0, int(round(current_time)))
        is_unexpected = rng.random() < float(settings["unexpected_rate"])
        delay = rng.randint(1, max(2, horizon // 40)) if is_unexpected else 0
        arrival = planned_arrival + delay
        arrival_samples.append((arrival, planned_arrival, is_unexpected))
    latest_arrival = max((arrival for arrival, _, _ in arrival_samples), default=0)
    if latest_arrival >= horizon:
        scale = (horizon - 1) / latest_arrival
        arrival_samples = [
            (int(round(arrival * scale)), int(round(planned * scale)), unexpected)
            for arrival, planned, unexpected in arrival_samples
        ]
    arrival_samples.sort(key=lambda item: item[0])

    trains = []
    for train_id, (arrival, planned_arrival, is_unexpected) in enumerate(arrival_samples):
        category = rng.choices(
            ["Express", "Premium", "Passenger", "Suburban EMU"],
            weights=[0.25, 0.08, 0.17, 0.50],
        )[0]
        if category == "Suburban EMU":
            length_class = "medium"
            priority = 1
        else:
            length_class = rng.choices(LENGTH_CLASSES, weights=[0.2, 0.35, 0.45])[0]
            priority = 0 if category in {"Express", "Premium"} else 1
        trains.append(TrainSpec(
            id=train_id,
            scheduled_arrival=arrival,
            dwell_time=rng.randint(int(settings["dwell_min"]), int(settings["dwell_max"])),
            length_class=length_class,
            electrified_required=True,
            priority=priority,
            label=f"Synthetic {category} #{train_id}",
            category=category,
            arrival_delay=arrival - planned_arrival,
            unexpected=is_unexpected,
            cancelled=rng.random() < float(settings["cancellation_rate"]),
        ))

    closure_count = int(settings["closure_count"])
    if closure_count < 0:
        raise ValueError("closure_count must be nonnegative")
    candidates = list(range(platform_count))
    rng.shuffle(candidates)
    for platform_id in candidates[:min(closure_count, platform_count)]:
        start = rng.randint(max(1, horizon // 4), max(1, horizon // 2))
        duration = rng.randint(max(2, horizon // 20), max(3, horizon // 8))
        platforms[platform_id].closure_intervals = [(start, min(horizon, start + duration))]

    arrival_horizon = horizon
    latest_arrival = max((train.scheduled_arrival for train in trains), default=0)
    drain_time = max((train.dwell_time for train in trains), default=0)
    turnaround = max((platform.turnaround_buffer for platform in platforms), default=0)
    approach_headway = 2 if profile in {
        "high_traffic", "peak_hour", "platform_disruption", "extreme_stress",
    } else 1
    simulation_horizon = max(
        horizon,
        latest_arrival + drain_time + turnaround + approach_headway + 1,
    )
    return ScenarioConfig(
        horizon=simulation_horizon,
        max_queue=max(16, platform_count * 2),
        platforms=platforms,
        trains=trains,
        name=profile,
        approach_headway=approach_headway,
        arrival_horizon=arrival_horizon,
    )
