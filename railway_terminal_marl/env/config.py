"""
Scenario configuration for the railway terminus MARL environment.

A "terminal" here is a single terminus station with several platforms.
Trains arrive according to a schedule and must be allocated to a
compatible platform. Agents = platforms (fixed set). Trains form a
shared waiting queue that every platform agent observes.
"""
from dataclasses import dataclass, field
from typing import List

LENGTH_CLASSES = ["short", "medium", "long"]  # ordinal: short < medium < long


@dataclass
class Platform:
    id: int
    length_class: str      # max train length this platform can accept
    electrified: bool      # whether platform has electrified line / OHE
    turnaround_buffer: int = 2  # min timesteps needed between departure and next arrival
    label: str = ""         # display label, e.g. "2A" — defaults to str(id) if unset
    section: str = "main"    # display grouping, e.g. "main" vs "suburban"

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

    return ScenarioConfig(horizon=220, max_queue=10, platforms=platforms, trains=trains)


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
                                       turnaround_buffer=2, label=lbl, section="main"))
        else:
            platforms.append(Platform(id=i, length_class="long", electrified=True,
                                       turnaround_buffer=5, label=lbl, section="main"))

    suburban_labels = ["13", "14", "15", "16", "17"]
    offset = len(main_labels)
    for j, lbl in enumerate(suburban_labels):
        platforms.append(Platform(id=offset + j, length_class="medium", electrified=True,
                                   turnaround_buffer=2, label=lbl, section="suburban"))

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
    return ScenarioConfig(horizon=horizon, max_queue=16, platforms=platforms, trains=trains)
