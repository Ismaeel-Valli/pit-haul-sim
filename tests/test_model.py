"""Verification of the simulation engine against hand calculations and invariants.

These tests check that the model does what its documentation says
(verification). They do not show that it represents any real mine
(validation), which would need site data.
"""

import numpy as np
import pytest

from haulsim import MineConfig, Triangular, simulate
from haulsim.metrics import station_minutes, summarise, truck_state_minutes
from haulsim.model import CRUSHER, SHOVEL


def fixed(x):
    return Triangular(x, x, x)


def deterministic(**kw):
    base = dict(
        n_trucks=1,
        randomness_enabled=False,
        breakdowns_enabled=False,
        shifts_enabled=False,
    )
    base.update(kw)
    return MineConfig(**base)


def delivery_times(log):
    return [round(d[2], 9) for d in log.deliveries]


# ---------------------------------------------------------------------- #
# Reproducibility
# ---------------------------------------------------------------------- #
def test_same_seed_reproduces_identical_logs():
    cfg = MineConfig(n_trucks=7)
    a, b = simulate(cfg, 3 * 1440, seed=42), simulate(cfg, 3 * 1440, seed=42)
    assert a.deliveries == b.deliveries
    assert a.work == b.work
    assert a.repairs == b.repairs
    assert a.queue_len == b.queue_len


def test_different_seeds_differ():
    cfg = MineConfig(n_trucks=7)
    a, b = simulate(cfg, 3 * 1440, seed=1), simulate(cfg, 3 * 1440, seed=2)
    assert a.deliveries != b.deliveries


def test_randomness_disabled_ignores_seed():
    cfg = MineConfig(n_trucks=4, randomness_enabled=False)
    assert simulate(cfg, 2 * 1440, seed=1).work == simulate(cfg, 2 * 1440, seed=99).work


# ---------------------------------------------------------------------- #
# Hand-calculated deterministic cycles
# ---------------------------------------------------------------------- #
def test_one_truck_deterministic_cycle_matches_hand_calculation():
    # Cycle = load 4 + haul 12 + dump 6 + return 10 = 32 min.
    # First delivery when the first dump ends at 4 + 12 + 6 = 22 min,
    # then every 32 min: 22 + 32k <= 1440  ->  k = 0..44  ->  45 deliveries.
    log = simulate(deterministic(), 1440)
    expected = [22 + 32 * k for k in range(45)]
    assert delivery_times(log) == pytest.approx(expected)
    assert summarise(log, 0, 1440)["delivered_t"] == 45 * 100
    cycle_lengths = [end - start for _, start, end in log.cycles]
    assert cycle_lengths == pytest.approx([32.0] * len(cycle_lengths))


def test_one_truck_shift_stoppage_shifts_later_deliveries_by_20_min():
    # Stoppage 720-740. Deliveries at 22 + 32k up to 694 (22 of them).
    # The cycle in progress at 720 resumes at 740, so every later delivery
    # is 20 min late: 746 + 32j <= 1440 -> j = 0..21 (22 more). Total 44.
    log = simulate(deterministic(shifts_enabled=True), 1440)
    expected = [22 + 32 * k for k in range(22)] + [746 + 32 * j for j in range(22)]
    assert delivery_times(log) == pytest.approx(expected)
    assert log.shift_windows[0] == pytest.approx((720, 740))


def test_breakdown_resumes_remaining_duration_without_restart():
    # MTBF 30 operating minutes, repair exactly 10 min, no shifts.
    # Op-clock hits 30 at t=30, 8 min into the 10-min return (22-32).
    # Repair 30-40, return finishes 40-42 (2 min remaining, not 10).
    # Next cycle: load 42-46, haul 46-58, dump 58-64 -> delivery at 64.
    # Op-clock (reset at 40) reaches 30 at t=70 during the return 64-74.
    cfg = deterministic(breakdowns_enabled=True, mtbf_operating_h=0.5, repair_min=fixed(10))
    log = simulate(cfg, 85)
    assert delivery_times(log) == pytest.approx([22, 64])
    assert [(r[1], r[2]) for r in log.repairs] == pytest.approx([(30, 40), (70, 80)])
    first_return = [w for w in log.work if w[1] == "return" and w[3] < 45]
    assert [(w[3], w[4]) for w in first_return] == pytest.approx([(22, 30), (40, 42)])
    assert sum(w[4] - w[3] for w in first_return) == pytest.approx(10.0)


def test_failure_at_crusher_blocks_station_and_delivers_once():
    # Both trucks fail after 19 operating minutes, i.e. 3 min into dumping.
    # A: dump 16-19, repair 19-29 (crusher blocked), dump 29-32 -> delivery 32.
    # B: arrives at crusher at 20, waits until 32, dump 32-35, repair 35-45,
    #    dump 45-48 -> delivery 48.
    cfg = deterministic(n_trucks=2, breakdowns_enabled=True, mtbf_operating_h=19 / 60, repair_min=fixed(10))
    log = simulate(cfg, 50)
    assert delivery_times(log) == pytest.approx([32, 48])
    assert [r[3] for r in log.repairs][:2] == [CRUSHER, CRUSHER]
    crusher_waits = [(w[2], w[3]) for w in log.waits if w[1] == CRUSHER]
    assert crusher_waits[1] == pytest.approx((20, 32))
    m = station_minutes(log, CRUSHER, 0, 50)
    assert m["blocked"] == pytest.approx(20)
    assert m["productive"] == pytest.approx(12)
    assert m["idle"] == pytest.approx(18)
    assert len({d[1] for d in log.deliveries}) == len(log.deliveries)


def test_overlapping_repair_and_shift_stoppage():
    # Stoppages every 35 min for 10 min: 35-45, 70-80, 105-115.
    # Failure at t=30 (8 min into the return); repair 30-40 runs THROUGH the
    # 35-45 stoppage; the truck must not restart until 45, then finishes the
    # last 2 min of its return 45-47. Next cycle: load 47-51, haul 51-63,
    # dump 63-69 (delivery 69), return 69-70 | stoppage 70-80 | 80-85, when
    # the op-clock (30 min since 45) fails it again. Repair 85-95.
    cfg = deterministic(
        breakdowns_enabled=True,
        shifts_enabled=True,
        mtbf_operating_h=0.5,
        repair_min=fixed(10),
        shift_interval_h=35 / 60,
        shift_stop_min=10,
    )
    log = simulate(cfg, 100)
    assert delivery_times(log) == pytest.approx([22, 69])
    assert [(r[1], r[2]) for r in log.repairs] == pytest.approx([(30, 40), (85, 95)])
    after_first_repair = [w for w in log.work if w[3] >= 30]
    assert after_first_repair[0][3] == pytest.approx(45)  # not 40: stoppage still active

    states = truck_state_minutes(log, 0, 100).iloc[0]
    assert states["repair"] == pytest.approx(20)
    # 35-45 overlaps repair 35-40 -> only 5 min counted as shift stop; plus 70-80.
    assert states["shift_stop"] == pytest.approx(15)
    assert states[["load", "haul", "dump", "return"]].sum() == pytest.approx(65)
    assert states.sum() == pytest.approx(100)


# ---------------------------------------------------------------------- #
# Invariants on stochastic runs
# ---------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def busy_log():
    cfg = MineConfig(n_trucks=12, n_shovels=2, mtbf_operating_h=6.0)  # frequent failures
    return simulate(cfg, 6 * 1440, seed=7)


def _max_concurrency(intervals):
    events = sorted([(s, 1) for s, e in intervals] + [(e, -1) for s, e in intervals], key=lambda x: (x[0], x[1]))
    level = peak = 0
    for _, d in events:
        level += d
        peak = max(peak, level)
    return peak


def test_resource_capacity_never_exceeded(busy_log):
    for station, cap in ((SHOVEL, 2), (CRUSHER, 1)):
        holds = [(h[2], h[3]) for h in busy_log.holds if h[1] == station]
        assert _max_concurrency(holds) <= cap
    # With 12 trucks both shovels are used at the same time at some point.
    assert _max_concurrency([(h[2], h[3]) for h in busy_log.holds if h[1] == SHOVEL]) == 2


def test_fifo_queue_discipline(busy_log):
    crusher = sorted([w for w in busy_log.waits if w[1] == CRUSHER and not np.isnan(w[3])], key=lambda w: w[2])
    grants = [w[3] for w in crusher]
    assert grants == sorted(grants)  # first to join is first to be served


def test_payload_conservation_and_no_duplicate_deliveries(busy_log):
    loaded = sum(x[3] for x in busy_log.loads)
    delivered = sum(x[3] for x in busy_log.deliveries)
    on_board = sum(busy_log.on_board_at_end.values())
    assert loaded == pytest.approx(delivered + on_board)
    trips = [d[1] for d in busy_log.deliveries]
    assert len(trips) == len(set(trips))
    assert set(trips) <= {x[1] for x in busy_log.loads}
    assert all(d[3] == 100 for d in busy_log.deliveries)


def test_no_productive_work_during_shift_stoppages(busy_log):
    windows = np.array(busy_log.shift_windows)
    assert len(windows) == 11  # every 12 h; the change at the 6-day horizon is not run
    for w in busy_log.work:
        overlap = np.clip(np.minimum(w[4], windows[:, 1]) - np.maximum(w[3], windows[:, 0]), 0, None)
        assert overlap.max() < 1e-9


def test_failures_only_during_activities(busy_log):
    assert len(busy_log.repairs) > 20
    assert {r[4] for r in busy_log.repairs} <= {"load", "haul", "dump", "return"}
    # A repair is a breakdown in place: the truck has no work segment during it.
    for truck, start, end, *_ in busy_log.repairs:
        for w in busy_log.work:
            if w[0] == truck:
                assert min(w[4], end) - max(w[3], start) < 1e-9


def test_state_time_accounting_closes(busy_log):
    a, b = 1440.0, busy_log.horizon
    states = truck_state_minutes(busy_log, a, b)
    assert (states.to_numpy() >= -1e-6).all()
    assert states.sum(axis=1).to_numpy() == pytest.approx(np.full(12, b - a))
    for station in (SHOVEL, CRUSHER):
        m = station_minutes(busy_log, station, a, b)
        parts = [m["productive"], m["blocked"], m["shift"], m["idle"]]
        assert min(parts) >= -1e-6
        assert sum(parts) == pytest.approx(m["scheduled"])


def test_time_weighted_queue_matches_waiting_intervals(busy_log):
    # Integral of queue length over the window must equal the total waiting
    # time inside the window: two independent logs, one quantity.
    a, b = 1440.0, busy_log.horizon
    s = summarise(busy_log, a, b)
    for station in (SHOVEL, CRUSHER):
        waits = [(w[2], busy_log.horizon if np.isnan(w[3]) else w[3]) for w in busy_log.waits if w[1] == station]
        inside = sum(max(0.0, min(e, b) - max(st, a)) for st, e in waits)
        assert s[f"queue_{station}_mean"] * (b - a) == pytest.approx(inside)


def test_deliveries_counted_at_completion_inside_window():
    log = simulate(MineConfig(n_trucks=5), 2 * 1440, seed=3)
    a, b = 600.0, 1700.0
    expected = sum(d[3] for d in log.deliveries if a < d[2] <= b)
    assert summarise(log, a, b)["delivered_t"] == expected


def test_toggles_disable_events():
    log = simulate(MineConfig(n_trucks=5, breakdowns_enabled=False, shifts_enabled=False), 3 * 1440, seed=5)
    assert log.repairs == [] and log.shift_windows == []
    log = simulate(MineConfig(n_trucks=5), 3 * 1440, seed=5)
    assert log.repairs and len(log.shift_windows) == 5  # 12, 24, ..., 60 h
