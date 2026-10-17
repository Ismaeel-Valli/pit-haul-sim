"""SimPy discrete-event model of a shovel -> crusher truck haulage loop.

Event flow for one truck (repeats forever)::

    join shovel queue -> load -> haul loaded -> join crusher queue
        -> dump (tonnes counted here) -> return empty -> (next cycle)

Interruptions
-------------
*Breakdowns* use an **operating-time failure clock**. Each truck carries
``ops_to_failure``: the operating minutes left until its next failure,
sampled from an exponential distribution with mean ``mtbf_operating_h``.
The clock advances only while the truck is *performing an activity*
(loading, hauling, dumping, returning). It does not advance while the
truck is queuing, under repair, or stopped for a shift change. Failures
therefore only happen part-way through an activity, which is resumed
with its *remaining* duration once the repair is finished.

A truck that fails while it occupies the shovel or the crusher keeps
that station blocked until it is repaired (baseline assumption: no tow
or bypass). A truck that fails on the road is repaired where it stands
and does not block other trucks (roads are uncongested in this model).

*Shift changes* are a global stoppage: at every shift change all
productive activities pause and keep their remaining work. Queued trucks
stay in their queues, loads stay on the trucks, and repairs carry on.
A truck whose repair finishes during a stoppage waits for the stoppage
to end before resuming. Because the failure clock does not run during a
stoppage, a breakdown can never *start* inside a stoppage, but a repair
can overlap one; in the accounting, overlapping time is attributed to
repair only (never to both), see ``metrics.py``.

Logging
-------
The model records raw event logs (``SimLog``) rather than computing
statistics on the fly. Every metric, every verification test and the
dashboard replay are derived from the same logs, which keeps them
consistent with each other.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import simpy

from .config import MineConfig

EPS = 1e-9

# Activity codes used in the logs.
LOAD, HAUL, DUMP, RETURN = "load", "haul", "dump", "return"
ACTIVITIES = (LOAD, HAUL, DUMP, RETURN)
SHOVEL, CRUSHER = "shovel", "crusher"

# Independent random streams, one per source of randomness. Changing,
# say, the repair distribution does not disturb the sequence of loading
# times, which makes scenario comparisons less noisy.
STREAMS = ("load", "haul", "dump", "return", "failure", "repair")


class Sampler:
    """Draws activity durations from separate, reproducible random streams.

    With ``randomness_enabled=False`` every draw returns the distribution
    mean, so the model becomes deterministic (useful for hand-checked
    verification). Means are used rather than modes so a deterministic
    run represents the same average workload as the stochastic one.
    """

    def __init__(self, cfg: MineConfig, seed: int):
        self.cfg = cfg
        self.seed = int(seed)
        children = np.random.SeedSequence(self.seed).spawn(len(STREAMS))
        self.rng = {name: np.random.default_rng(child) for name, child in zip(STREAMS, children)}
        self._dist = {
            LOAD: cfg.load_min,
            HAUL: cfg.haul_min,
            DUMP: cfg.dump_min,
            RETURN: cfg.return_min,
        }

    def _tri(self, stream: str, dist) -> float:
        if not self.cfg.randomness_enabled or dist.low == dist.high:
            return dist.mean
        return float(self.rng[stream].triangular(dist.low, dist.mode, dist.high))

    def activity(self, kind: str) -> float:
        return self._tri(kind, self._dist[kind])

    def repair(self) -> float:
        return self._tri("repair", self.cfg.repair_min)

    def operating_time_to_failure(self) -> float:
        if not self.cfg.breakdowns_enabled:
            return float("inf")
        mean = self.cfg.mtbf_operating_min
        if not self.cfg.randomness_enabled:
            return mean
        return float(self.rng["failure"].exponential(mean))


@dataclass
class SimLog:
    """Raw event logs of one simulation run. Times are in minutes."""

    config: MineConfig
    seed: int
    horizon: float
    # (truck, activity, trip_id, seg_start, seg_end, progress_before, total_duration)
    work: list[tuple] = field(default_factory=list)
    # (truck, start, end, holding_station_or_empty, activity_interrupted)
    repairs: list[tuple] = field(default_factory=list)
    # (truck, station, t_join, t_grant)
    waits: list[tuple] = field(default_factory=list)
    # (truck, station, t_grant, t_release)
    holds: list[tuple] = field(default_factory=list)
    # (truck, trip_id, t, tonnes)
    loads: list[tuple] = field(default_factory=list)
    deliveries: list[tuple] = field(default_factory=list)
    # (truck, t_start, t_end): from joining the shovel queue to joining it again
    cycles: list[tuple] = field(default_factory=list)
    # station -> list of (t, number of trucks waiting, excluding those in service)
    queue_len: dict[str, list[tuple]] = field(default_factory=lambda: {SHOVEL: [(0.0, 0)], CRUSHER: [(0.0, 0)]})
    shift_windows: list[tuple] = field(default_factory=list)
    # Truck tonnes still on board at the horizon (for conservation checks)
    on_board_at_end: dict[int, float] = field(default_factory=dict)
    open_cycles: int = 0


class Mine:
    """Owns the SimPy environment, shared resources and the shift controller."""

    def __init__(self, cfg: MineConfig, seed: int, horizon: float):
        self.cfg = cfg
        self.env = simpy.Environment()
        self.sampler = Sampler(cfg, seed)
        self.log = SimLog(config=cfg, seed=int(seed), horizon=float(horizon))
        self.horizon = float(horizon)

        # Finite-capacity resources; SimPy serves requests first-come-first-served.
        self.stations = {
            SHOVEL: simpy.Resource(self.env, capacity=cfg.n_shovels),
            CRUSHER: simpy.Resource(self.env, capacity=cfg.n_crushers),
        }
        self.waiting = {SHOVEL: 0, CRUSHER: 0}

        # Shift-stoppage state shared with every truck.
        self.shift_active = False
        self.shift_start_ev = self.env.event()  # fires when a stoppage begins
        self.shift_end_ev = self.env.event()  # fires when the current stoppage ends

        self._trip_ids = itertools.count()
        self.trucks = [Truck(i, self) for i in range(cfg.n_trucks)]
        if cfg.shifts_enabled and cfg.shift_stop_min > 0:
            self.env.process(self._shift_controller())

    # ------------------------------------------------------------------ #
    def _shift_controller(self):
        """Stop production for ``shift_stop_min`` at every shift change.

        Simulation time 0 is the start of a shift, so stoppages begin at
        interval, 2*interval, ... and the first shift is a full one.
        """
        interval = self.cfg.shift_interval_min
        stop = self.cfg.shift_stop_min
        k = 1
        while True:
            yield self.env.timeout(k * interval - self.env.now)
            self.shift_end_ev = self.env.event()
            self.shift_active = True
            started, self.shift_start_ev = self.shift_start_ev, self.env.event()
            started.succeed()
            self.log.shift_windows.append((self.env.now, self.env.now + stop))
            yield self.env.timeout(stop)
            self.shift_active = False
            self.shift_end_ev.succeed()
            k += 1

    def queue_change(self, station: str, delta: int) -> None:
        self.waiting[station] += delta
        self.log.queue_len[station].append((self.env.now, self.waiting[station]))

    def next_trip_id(self) -> int:
        return next(self._trip_ids)

    # ------------------------------------------------------------------ #
    def run(self) -> SimLog:
        self.env.run(until=self.horizon)
        self._close_open_records()
        return self.log

    def _close_open_records(self) -> None:
        """Close intervals still in progress at the horizon so that time
        accounting covers the whole run. Nothing is *counted* here: a dump
        in progress at the horizon is not a delivery."""
        t_end = self.horizon
        for truck in self.trucks:
            if truck.open_work is not None:
                kind, trip, start, before, total = truck.open_work
                if t_end > start:
                    self.log.work.append((truck.id, kind, trip, start, t_end, before, total))
            if truck.open_repair is not None:
                start, holding, kind = truck.open_repair
                self.log.repairs.append((truck.id, start, t_end, holding, kind))
            if truck.open_wait is not None:
                station, start = truck.open_wait
                # Still waiting at the horizon: grant time unknown (censored).
                self.log.waits.append((truck.id, station, start, float("nan")))
            if truck.open_hold is not None:
                station, start = truck.open_hold
                self.log.holds.append((truck.id, station, start, t_end))
            self.log.on_board_at_end[truck.id] = truck.on_board
            if truck.cycle_start is not None:
                self.log.open_cycles += 1


class Truck:
    """One haul truck. Its SimPy process runs the cycle indefinitely."""

    def __init__(self, truck_id: int, mine: Mine):
        self.id = truck_id
        self.mine = mine
        self.env = mine.env
        self.on_board = 0.0
        self.trip_id: int | None = None
        self.holding: str = ""
        self.ops_to_failure = mine.sampler.operating_time_to_failure()
        # Records that are open (in progress) - closed at the horizon if needed.
        self.open_work = None
        self.open_repair = None
        self.open_wait = None
        self.open_hold = None
        self.cycle_start: float | None = None
        self.env.process(self.run())

    # ------------------------------------------------------------------ #
    def run(self):
        mine, log = self.mine, self.mine.log
        while True:
            now = self.env.now
            if self.cycle_start is not None:
                log.cycles.append((self.id, self.cycle_start, now))
            self.cycle_start = now

            # 1-2. Queue for and load at the shovel.
            yield from self._visit_station(SHOVEL, LOAD)
            # 3. Travel loaded to the crusher.
            yield from self._work(HAUL, mine.sampler.activity(HAUL))
            # 4-5. Queue for and dump at the crusher receiving station.
            yield from self._visit_station(CRUSHER, DUMP)
            # 6. Return empty to the shovel.
            yield from self._work(RETURN, mine.sampler.activity(RETURN))

    def _visit_station(self, station: str, kind: str):
        mine, log = self.mine, self.mine.log
        resource = mine.stations[station]
        t_join = self.env.now
        self.open_wait = (station, t_join)
        mine.queue_change(station, +1)
        with resource.request() as req:
            yield req
            t_grant = self.env.now
            mine.queue_change(station, -1)
            log.waits.append((self.id, station, t_join, t_grant))
            self.open_wait = None
            self.holding = station
            self.open_hold = (station, t_grant)

            if kind == LOAD:
                self.trip_id = mine.next_trip_id()
            yield from self._work(kind, mine.sampler.activity(kind))

            # Service complete. Production is counted ONLY at the end of
            # dumping; finishing loading merely puts tonnes on the truck.
            t_done = self.env.now
            if kind == LOAD:
                self.on_board = mine.cfg.payload_t
                log.loads.append((self.id, self.trip_id, t_done, self.on_board))
            else:
                log.deliveries.append((self.id, self.trip_id, t_done, self.on_board))
                self.on_board = 0.0
                self.trip_id = None
            log.holds.append((self.id, station, t_grant, t_done))
            self.open_hold = None
            self.holding = ""
        # Leaving the ``with`` block releases the station at t_done.

    def _work(self, kind: str, duration: float):
        """Perform an activity of ``duration`` operating minutes.

        The activity advances in chunks. A chunk ends when (a) the activity
        is finished, (b) the truck's failure clock reaches zero, or (c) a
        shift stoppage begins. Elapsed time is always subtracted from the
        remaining duration, so interrupted work is resumed, never restarted.
        """
        mine, log = self.mine, self.mine.log
        trip = self.trip_id
        remaining = float(duration)
        done = 0.0
        while remaining > EPS:
            if mine.shift_active:
                yield mine.shift_end_ev
                continue
            chunk = min(remaining, self.ops_to_failure)
            t0 = self.env.now
            self.open_work = (kind, trip, t0, done, duration)
            yield self.env.timeout(chunk) | mine.shift_start_ev
            elapsed = self.env.now - t0
            self.open_work = None
            if elapsed > 0:
                log.work.append((self.id, kind, trip, t0, self.env.now, done, duration))
            remaining -= elapsed
            done += elapsed
            self.ops_to_failure -= elapsed
            if self.ops_to_failure <= EPS:
                yield from self._breakdown(kind)

    def _breakdown(self, kind: str):
        """Repair in place. Repairs continue through shift stoppages; if a
        stoppage is still active afterwards, ``_work`` waits for it to end."""
        mine, log = self.mine, self.mine.log
        t0 = self.env.now
        self.open_repair = (t0, self.holding, kind)
        yield self.env.timeout(mine.sampler.repair())
        self.open_repair = None
        log.repairs.append((self.id, t0, self.env.now, self.holding, kind))
        self.ops_to_failure = mine.sampler.operating_time_to_failure()


def simulate(cfg: MineConfig, horizon_min: float, seed: int = 0) -> SimLog:
    """Run one replication of ``cfg`` for ``horizon_min`` minutes."""
    if horizon_min <= 0:
        raise ValueError("horizon_min must be > 0")
    return Mine(cfg, seed, horizon_min).run()
