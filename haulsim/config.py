"""Configuration objects, defaults and input validation.

Units
-----
User-facing inputs use the units a mine planner would naturally quote
(activity durations in minutes, failure intervals and shift intervals in
hours). Everything is converted to **minutes** before it reaches the
simulation engine, which never sees any other time unit.

All default values are SYNTHETIC demonstration values. They are plausible
in magnitude but are not measurements from any real mine.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

MINUTES_PER_HOUR = 60.0
MINUTES_PER_DAY = 1440.0


@dataclass(frozen=True)
class Triangular:
    """Triangular distribution described by (minimum, most likely, maximum).

    * ``low``  - the shortest duration considered possible,
    * ``mode`` - the most likely (most frequent) duration,
    * ``high`` - the longest duration considered possible.

    Triangular distributions are common in early-stage studies because
    they are bounded and can be elicited from people who know the
    operation ("usually 4 minutes, never less than 3 or more than 5")
    without a large data set. The mean is (low + mode + high) / 3.
    """

    low: float
    mode: float
    high: float

    @property
    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3.0

    def validate(self, name: str) -> None:
        values = (self.low, self.mode, self.high)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError(f"{name}: all triangular parameters must be finite numbers")
        if self.low < 0:
            raise ValueError(f"{name}: minimum must be >= 0 (got {self.low})")
        if not self.low <= self.mode <= self.high:
            raise ValueError(
                f"{name}: require minimum <= most likely <= maximum "
                f"(got {self.low}, {self.mode}, {self.high})"
            )
        if self.high <= 0:
            raise ValueError(f"{name}: maximum must be > 0")

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.low, self.mode, self.high)


def _tri(value: Any) -> Triangular:
    if isinstance(value, Triangular):
        return value
    if isinstance(value, dict):
        return Triangular(value["low"], value["mode"], value["high"])
    low, mode, high = value
    return Triangular(float(low), float(mode), float(high))


@dataclass(frozen=True)
class MineConfig:
    """Physical and operational description of the haulage system."""

    n_trucks: int = 8
    payload_t: float = 100.0
    n_shovels: int = 1
    n_crushers: int = 1  # crusher receiving stations (dump + crush combined)

    load_min: Triangular = field(default_factory=lambda: Triangular(3, 4, 5))
    haul_min: Triangular = field(default_factory=lambda: Triangular(10, 12, 14))
    dump_min: Triangular = field(default_factory=lambda: Triangular(5, 6, 7))
    return_min: Triangular = field(default_factory=lambda: Triangular(8, 10, 12))

    mtbf_operating_h: float = 40.0  # mean OPERATING hours between truck failures
    repair_min: Triangular = field(default_factory=lambda: Triangular(30, 60, 120))

    shift_interval_h: float = 12.0
    shift_stop_min: float = 20.0

    breakdowns_enabled: bool = True
    shifts_enabled: bool = True
    randomness_enabled: bool = True

    def __post_init__(self) -> None:
        # Accept tuples / dicts for convenience (e.g. when loaded from JSON).
        for name in ("load_min", "haul_min", "dump_min", "return_min", "repair_min"):
            object.__setattr__(self, name, _tri(getattr(self, name)))
        self.validate()

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        if not isinstance(self.n_trucks, int) or self.n_trucks < 1:
            raise ValueError(f"n_trucks must be an integer >= 1 (got {self.n_trucks!r})")
        if not isinstance(self.n_shovels, int) or self.n_shovels < 1:
            raise ValueError(f"n_shovels must be an integer >= 1 (got {self.n_shovels!r})")
        if not isinstance(self.n_crushers, int) or self.n_crushers < 1:
            raise ValueError(f"n_crushers must be an integer >= 1 (got {self.n_crushers!r})")
        if not (self.payload_t > 0 and math.isfinite(self.payload_t)):
            raise ValueError(f"payload_t must be > 0 (got {self.payload_t})")
        for name in ("load_min", "haul_min", "dump_min", "return_min", "repair_min"):
            getattr(self, name).validate(name)
        if self.load_min.mean <= 0 or self.dump_min.mean <= 0:
            raise ValueError("loading and dumping must take a positive mean time")
        if not (self.mtbf_operating_h > 0 and math.isfinite(self.mtbf_operating_h)):
            raise ValueError(f"mtbf_operating_h must be > 0 (got {self.mtbf_operating_h})")
        if not self.shift_interval_h > 0:
            raise ValueError("shift_interval_h must be > 0")
        if self.shift_stop_min < 0:
            raise ValueError("shift_stop_min must be >= 0")
        if self.shifts_enabled and self.shift_stop_min >= self.shift_interval_h * MINUTES_PER_HOUR:
            raise ValueError("shift_stop_min must be shorter than the shift interval")

    # ---- derived quantities in internal units (minutes) --------------- #
    @property
    def mtbf_operating_min(self) -> float:
        return self.mtbf_operating_h * MINUTES_PER_HOUR

    @property
    def shift_interval_min(self) -> float:
        return self.shift_interval_h * MINUTES_PER_HOUR

    @property
    def mean_cycle_work_min(self) -> float:
        """Mean truck cycle time with no queuing, failures or stoppages."""
        return self.load_min.mean + self.haul_min.mean + self.dump_min.mean + self.return_min.mean

    def with_trucks(self, n: int) -> "MineConfig":
        return replace(self, n_trucks=int(n))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for name in ("load_min", "haul_min", "dump_min", "return_min", "repair_min"):
            d[name] = list(getattr(self, name).as_tuple())
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MineConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown MineConfig keys: {sorted(unknown)}")
        return cls(**d)


@dataclass(frozen=True)
class ExperimentConfig:
    """How the model is exercised: run length, warm-up, replications, seeds."""

    fleet_sizes: tuple[int, ...] = tuple(range(1, 21))
    replications: int = 30
    warmup_h: float = 24.0
    measure_days: int = 14
    base_seed: int = 20261017
    workers: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "fleet_sizes", tuple(int(n) for n in self.fleet_sizes))
        self.validate()

    def validate(self) -> None:
        if not self.fleet_sizes:
            raise ValueError("fleet_sizes must not be empty")
        if any(n < 1 for n in self.fleet_sizes):
            raise ValueError("every fleet size must be >= 1")
        if len(set(self.fleet_sizes)) != len(self.fleet_sizes):
            raise ValueError("fleet_sizes must not contain duplicates")
        if self.replications < 2:
            raise ValueError("at least 2 replications are needed for a confidence interval")
        if self.warmup_h < 0:
            raise ValueError("warmup_h must be >= 0")
        if not isinstance(self.measure_days, int) or self.measure_days < 1:
            # Whole days keep the number of shift changes per measurement
            # window identical in every run, whatever the warm-up length.
            raise ValueError("measure_days must be a whole number of days >= 1")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")

    @property
    def warmup_min(self) -> float:
        return self.warmup_h * MINUTES_PER_HOUR

    @property
    def measure_min(self) -> float:
        return self.measure_days * MINUTES_PER_DAY

    @property
    def horizon_min(self) -> float:
        return self.warmup_min + self.measure_min

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fleet_sizes"] = list(self.fleet_sizes)
        return d


@dataclass(frozen=True)
class AnalysisConfig:
    """Decision rules and (illustrative) economics used to interpret results."""

    plateau_fraction: float = 0.95  # tipping point = smallest fleet >= 95 % of plateau
    plateau_tail: int = 3  # plateau estimated from the largest N fleet sizes tested
    plateau_tolerance: float = 0.02  # max relative rise across the tail to call it flat
    target_tph: float = 800.0  # production target, tonnes per scheduled hour

    # ILLUSTRATIVE costs per scheduled hour (not real prices).
    truck_cost_per_h: float = 300.0  # ownership + operating + operator, per truck
    fixed_cost_per_h: float = 2500.0  # shovel, crusher, overheads (independent of fleet)
    value_per_t: float | None = None  # optional margin per tonne for a profit view

    def __post_init__(self) -> None:
        if not 0 < self.plateau_fraction <= 1:
            raise ValueError("plateau_fraction must be in (0, 1]")
        if self.plateau_tail < 2:
            raise ValueError("plateau_tail must be >= 2")
        if self.plateau_tolerance < 0:
            raise ValueError("plateau_tolerance must be >= 0")
        if self.target_tph < 0 or self.truck_cost_per_h < 0 or self.fixed_cost_per_h < 0:
            raise ValueError("target and costs must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
