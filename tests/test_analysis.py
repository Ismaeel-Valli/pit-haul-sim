"""Checks of capacity bounds, decision rules and input validation."""

import numpy as np
import pandas as pd
import pytest

from haulsim import AnalysisConfig, ExperimentConfig, MineConfig, Triangular, simulate
from haulsim.analysis import (
    capacity_bounds,
    economics,
    find_tipping_point,
    fleet_for_target,
    limiting_resource,
    mean_ci,
    summarise_fleet,
)
from haulsim.experiment import replication_seed, run_fleet_study
from haulsim.metrics import summarise


# ---------------------------------------------------------------------- #
# Capacity bounds
# ---------------------------------------------------------------------- #
def test_idealised_bounds_hand_calculation():
    b = capacity_bounds(MineConfig(), 4)
    assert b["fleet_tph"] == pytest.approx(4 * 100 * 60 / 32)  # 750
    assert b["shovel_tph"] == pytest.approx(1500)
    assert b["crusher_tph"] == pytest.approx(1000)
    assert b["ideal_bound_tph"] == pytest.approx(750)
    assert capacity_bounds(MineConfig(), 10)["limiting_resource"] == "crusher"


@pytest.mark.parametrize("n", [1, 3, 5, 7, 10, 14])
def test_throughput_within_availability_adjusted_bound(n):
    cfg = MineConfig(n_trucks=n)
    tph = [summarise(simulate(cfg, 11 * 1440, seed=s), 1440, 11 * 1440)["tph"] for s in range(4)]
    bound = capacity_bounds(cfg)["adj_bound_tph"]
    # The bound is an expected-value bound, so allow a little sampling noise.
    assert np.mean(tph) <= bound * 1.02


def test_large_baseline_fleet_approaches_crusher_capacity():
    cfg = MineConfig(n_trucks=15, breakdowns_enabled=False, shifts_enabled=False)
    s = summarise(simulate(cfg, 8 * 1440, seed=11), 1440, 8 * 1440)
    # 1000 t/h uses the MEAN dump time (6 min), so a run whose sampled dumps
    # are slightly quicker than average may exceed it by a little.
    assert 0.97 * 1000 <= s["tph"] <= 1000 * 1.02
    # Hard sample-path limit: no dump is shorter than 5 min, so at most
    # window/5 dumps finish in the window, plus one already under way.
    assert s["deliveries"] <= 7 * 1440 / 5 + 1
    assert s["crusher_util_available"] > 0.97


def test_single_deterministic_truck_hits_fleet_bound():
    cfg = MineConfig(n_trucks=1, randomness_enabled=False, breakdowns_enabled=False, shifts_enabled=False)
    days = 20
    s = summarise(simulate(cfg, days * 1440), 0, days * 1440)
    # 900 complete cycles of 32 min in 20 days, plus boundary effects of one cycle.
    assert s["tph"] == pytest.approx(capacity_bounds(cfg)["fleet_tph"], rel=0.005)


# ---------------------------------------------------------------------- #
# The bottleneck is an outcome, not an assumption
# ---------------------------------------------------------------------- #
def _limit(cfg, n, days=5):
    s = summarise(simulate(cfg.with_trucks(n), (days + 1) * 1440, seed=2), 1440, (days + 1) * 1440)
    row = pd.Series({f"{k}_mean": v for k, v in s.items()})
    return limiting_resource(row)


def test_crusher_limits_default_large_fleet():
    assert _limit(MineConfig(), 14) == "crusher"


def test_slow_shovel_becomes_the_bottleneck():
    slow_shovel = MineConfig(load_min=Triangular(8, 9, 10))
    assert capacity_bounds(slow_shovel, 14)["limiting_resource"] == "shovel"
    assert _limit(slow_shovel, 14) == "shovel"


def test_unreliable_trucks_make_the_fleet_limiting():
    unreliable = MineConfig(mtbf_operating_h=2.0, repair_min=Triangular(90, 120, 180))
    assert _limit(unreliable, 6) == "truck fleet"


# ---------------------------------------------------------------------- #
# Statistics and decision rules
# ---------------------------------------------------------------------- #
def test_mean_ci_matches_textbook_value():
    m, lo, hi = mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    # t(0.975, 4) = 2.776, s = 1.5811 -> half width 1.963
    assert m == 3.0
    assert hi - m == pytest.approx(1.963, abs=1e-3)
    assert m - lo == pytest.approx(hi - m)


def _synthetic(curve, reps=30, noise=5.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = [
        {"n_trucks": n, "replication": r, "tph": curve(n) + rng.normal(0, noise)}
        for n in range(1, 16)
        for r in range(reps)
    ]
    return pd.DataFrame(rows)


def test_tipping_point_on_known_curve():
    df = _synthetic(lambda n: min(150.0 * n, 900.0))  # saturates exactly at n = 6
    tp = find_tipping_point(df, AnalysisConfig())
    assert tp.status == "ok"
    assert tp.plateau_tph == pytest.approx(900, rel=0.01)
    assert tp.fleet == 6  # 150*6 = 900 >= 855; 150*5 = 750 < 855


def test_rising_curve_reports_plateau_not_reached():
    df = _synthetic(lambda n: 100.0 * n)
    tp = find_tipping_point(df, AnalysisConfig())
    assert tp.status == "plateau_not_reached"
    assert tp.fleet is None
    assert "Extend the fleet range" in tp.messages[0]


def test_noisy_results_are_flagged_uncertain():
    df = _synthetic(lambda n: min(150.0 * n, 900.0), reps=3, noise=150.0)
    assert find_tipping_point(df, AnalysisConfig()).status in {"uncertain", "plateau_not_reached"}


def test_target_and_economics():
    df = _synthetic(lambda n: min(150.0 * n, 900.0))
    summary = summarise_fleet(df, metrics=("tph",))
    t = fleet_for_target(summary, 700)
    assert t["fleet_mean"] == 5 and t["fleet_confident"] == 5
    assert fleet_for_target(summary, 2000)["fleet_mean"] is None
    acfg = AnalysisConfig(target_tph=700, truck_cost_per_h=300, fixed_cost_per_h=2500)
    eco, choice = economics(summary, acfg)
    # Cost/h = 300 N + 2500 rises with N; t/h stops rising at 6, so beyond 6
    # cost per tonne must increase.
    assert choice["least_cost_meeting_target"] == 6
    assert eco.loc[6, "cost_per_t"] == pytest.approx((6 * 300 + 2500) / summary.loc[6, "tph_mean"])
    assert (np.diff(eco.loc[6:, "cost_per_t"].to_numpy()) > 0).all()


# ---------------------------------------------------------------------- #
# Experiment plumbing and validation
# ---------------------------------------------------------------------- #
def test_replication_seeds_are_stable_and_distinct():
    seeds = [replication_seed(123, r) for r in range(50)]
    assert len(set(seeds)) == 50
    assert seeds == [replication_seed(123, r) for r in range(50)]


def test_fleet_study_rows_and_seeds():
    exp = ExperimentConfig(fleet_sizes=(2, 4), replications=3, warmup_h=6, measure_days=1)
    df = run_fleet_study(MineConfig(), exp)
    assert len(df) == 6
    # Common random numbers: same seed for a replication across fleet sizes.
    assert df.groupby("replication")["seed"].nunique().eq(1).all()
    row = df.iloc[0]
    log = simulate(MineConfig(n_trucks=int(row.n_trucks)), exp.horizon_min, int(row.seed))
    assert summarise(log, exp.warmup_min, exp.horizon_min)["tph"] == row.tph


@pytest.mark.parametrize(
    "kwargs",
    [
        {"load_min": (5, 4, 6)},
        {"n_trucks": 0},
        {"payload_t": -1},
        {"mtbf_operating_h": 0},
        {"shift_stop_min": 800},
        {"repair_min": (-1, 2, 3)},
    ],
)
def test_invalid_mine_config_rejected(kwargs):
    with pytest.raises(ValueError):
        MineConfig(**kwargs)


def test_measurement_period_must_be_whole_days():
    with pytest.raises(ValueError):
        ExperimentConfig(measure_days=2.5)
    with pytest.raises(ValueError):
        ExperimentConfig(replications=1)


def test_config_round_trip():
    cfg = MineConfig(n_trucks=9, load_min=Triangular(2, 3, 4))
    assert MineConfig.from_dict(cfg.to_dict()) == cfg
