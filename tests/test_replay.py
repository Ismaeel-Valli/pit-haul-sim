"""The replay must agree with the event logs and with the reported metrics."""

import numpy as np
import pytest

from haulsim import MineConfig, simulate
from haulsim.metrics import summarise, time_weighted_mean
from haulsim.model import CRUSHER, SHOVEL
from haulsim.replay import build_frames, delivered_between


@pytest.fixture(scope="module")
def replay():
    log = simulate(MineConfig(n_trucks=10, mtbf_operating_h=8.0), 2 * 1440, seed=4)
    frames = build_frames(log, 1440.0, 1440.0 + 16 * 60, 1.0)
    return log, frames


def test_every_truck_appears_once_per_frame(replay):
    log, frames = replay
    counts = frames.groupby("t")["truck"].nunique()
    assert (counts == log.config.n_trucks).all()


def test_queued_trucks_match_logged_queue_lengths(replay):
    log, frames = replay
    queued = frames[frames.status == "Queued"]
    for t, g in queued.groupby("t"):
        expected = sum(
            int(time_weighted_mean(log.queue_len[s], t, t + 1e-6) + 0.5) for s in (SHOVEL, CRUSHER)
        )
        assert len(g) == expected


def test_repair_status_matches_repair_log(replay):
    log, frames = replay
    for t, g in frames.groupby("t"):
        under_repair = {r[0] for r in log.repairs if r[1] <= t < r[2]}
        assert set(g.loc[g.status == "Under repair", "truck"]) == under_repair


def test_replay_delivery_counter_matches_summary(replay):
    log, _ = replay
    a, b = 1440.0, 1440.0 + 16 * 60
    counter = delivered_between(log, a, np.arange(a, b + 1e-9, 1.0))
    assert counter[-1] == summarise(log, a, b)["delivered_t"]
    assert (np.diff(counter) >= 0).all()


def test_loaded_trucks_haul_and_empty_trucks_return(replay):
    _, frames = replay
    assert (frames.loc[frames.status == "Hauling (loaded)", "on_board_t"] == 100).all()
    assert (frames.loc[frames.status == "Returning (empty)", "on_board_t"] == 0).all()
