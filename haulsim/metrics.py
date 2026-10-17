"""Turn raw event logs into performance measures for a measurement window.

Window and boundary conventions
-------------------------------
The measurement window is ``(a, b]`` (minutes). Warm-up observations are
discarded by *ignoring* what happened before ``a``; the physical state of
the mine (truck positions, queues, loads, repairs) is never reset.

* **Event counts** (deliveries, completed cycles, completed waits) are
  attributed to the instant the event *completes*. A dump that starts in
  the warm-up and ends inside the window counts as one delivery in the
  window; a dump still in progress at ``b`` does not count.
* **Time-based measures** (state times, utilisation, queue lengths) clip
  every interval to the window, so an activity that crosses a boundary
  contributes exactly the part that lies inside the window.
* **Cycle times** are measured from joining the shovel queue to joining it
  again and are reported for cycles that *complete* inside the window.
  Cycles still open at ``b`` are censored. Cycles interrupted by a long
  repair are the most likely to be censored, so upper percentiles are
  slightly under-estimated; with multi-day windows and ~40-minute cycles
  the effect is small (one open cycle per truck per run).

Denominators
------------
* Truck state fractions: ``n_trucks * (b - a)`` (scheduled truck-time).
* Station fractions: ``capacity * (b - a)`` (scheduled server-time).
  The station time is split into *productive* (a truck is actually being
  loaded/dumped), *blocked* (occupied by a broken truck under repair),
  *shift* (shift stoppage, excluding blocked time) and *idle*.
* ``*_util_available``: productive time divided by station time that is
  neither shift stoppage nor blocked, i.e. how busy the station is when
  it could actually be working. This is the best bottleneck indicator.
* Truck *availability*: 1 - (repair time / scheduled truck-time).

When a repair overlaps a shift stoppage, the overlap is counted as repair
only, so no minute is ever counted twice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import ACTIVITIES, CRUSHER, DUMP, LOAD, SHOVEL, SimLog

TRUCK_STATES = (
    "load",
    "haul",
    "dump",
    "return",
    "queue_shovel",
    "queue_crusher",
    "repair",
    "shift_stop",
)
WORK_STATES = ("load", "haul", "dump", "return")
STATION_STATES = ("productive", "blocked", "shift", "idle")


# ---------------------------------------------------------------------- #
# Interval helpers
# ---------------------------------------------------------------------- #
def clip_len(starts, ends, a: float, b: float) -> np.ndarray:
    """Length of each interval [s, e] that lies inside [a, b]."""
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    return np.clip(np.minimum(ends, b) - np.maximum(starts, a), 0.0, None)


def overlap_with_windows(starts, ends, windows, a: float, b: float) -> np.ndarray:
    """Per-interval overlap with a set of disjoint windows, within [a, b]."""
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    if len(windows) == 0 or len(starts) == 0:
        return np.zeros(len(starts))
    w = np.asarray(windows, dtype=float)
    lo = np.maximum(np.maximum(starts[:, None], w[None, :, 0]), a)
    hi = np.minimum(np.minimum(ends[:, None], w[None, :, 1]), b)
    return np.clip(hi - lo, 0.0, None).sum(axis=1)


def time_weighted_mean(points: list[tuple], a: float, b: float) -> float:
    """Time average over [a, b] of a step function given as (t, value) changes."""
    if b <= a:
        return float("nan")
    t = np.array([p[0] for p in points], dtype=float)
    v = np.array([p[1] for p in points], dtype=float)
    # Value holding over [t_i, t_{i+1}); the last value holds until b.
    t_next = np.append(t[1:], max(b, t[-1]))
    dur = clip_len(t, t_next, a, b)
    return float((dur * v).sum() / (b - a))


def _frame(rows: list[tuple], cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=cols)


def logs_as_frames(log: SimLog) -> dict[str, pd.DataFrame]:
    return {
        "work": _frame(log.work, ["truck", "activity", "trip", "start", "end", "progress_before", "duration"]),
        "repairs": _frame(log.repairs, ["truck", "start", "end", "holding", "activity"]),
        "waits": _frame(log.waits, ["truck", "station", "join", "grant"]),
        "holds": _frame(log.holds, ["truck", "station", "grant", "release"]),
        "loads": _frame(log.loads, ["truck", "trip", "t", "tonnes"]),
        "deliveries": _frame(log.deliveries, ["truck", "trip", "t", "tonnes"]),
        "cycles": _frame(log.cycles, ["truck", "start", "end"]),
    }


# ---------------------------------------------------------------------- #
# Time accounting
# ---------------------------------------------------------------------- #
def shift_minutes(log: SimLog, a: float, b: float) -> float:
    if not log.shift_windows:
        return 0.0
    w = np.asarray(log.shift_windows)
    return float(clip_len(w[:, 0], w[:, 1], a, b).sum())


def truck_state_minutes(log: SimLog, a: float, b: float) -> pd.DataFrame:
    """Minutes each truck spends in each state inside [a, b].

    Rows: trucks. Columns: TRUCK_STATES. Each row sums to (b - a).
    """
    n = log.config.n_trucks
    out = pd.DataFrame(0.0, index=pd.RangeIndex(n, name="truck"), columns=list(TRUCK_STATES))
    windows = log.shift_windows
    shift_total = shift_minutes(log, a, b)

    if log.work:
        w = np.array([(r[0], r[3], r[4]) for r in log.work], dtype=float)
        kinds = np.array([r[1] for r in log.work])
        dur = clip_len(w[:, 1], w[:, 2], a, b)
        for kind in ACTIVITIES:
            m = kinds == kind
            out[kind] = np.bincount(w[m, 0].astype(int), weights=dur[m], minlength=n)

    repair_in_shift = np.zeros(n)
    if log.repairs:
        r = np.array([(x[0], x[1], x[2]) for x in log.repairs], dtype=float)
        trucks = r[:, 0].astype(int)
        out["repair"] = np.bincount(trucks, weights=clip_len(r[:, 1], r[:, 2], a, b), minlength=n)
        repair_in_shift = np.bincount(
            trucks, weights=overlap_with_windows(r[:, 1], r[:, 2], windows, a, b), minlength=n
        )

    if log.waits:
        q = np.array([(x[0], x[2], x[3]) for x in log.waits], dtype=float)
        stations = np.array([x[1] for x in log.waits])
        q[:, 2] = np.where(np.isnan(q[:, 2]), log.horizon, q[:, 2])
        in_window = clip_len(q[:, 1], q[:, 2], a, b)
        in_shift = overlap_with_windows(q[:, 1], q[:, 2], windows, a, b)
        for station in (SHOVEL, CRUSHER):
            m = stations == station
            out[f"queue_{station}"] = np.bincount(
                q[m, 0].astype(int), weights=in_window[m] - in_shift[m], minlength=n
            )

    # During a stoppage every truck that is not under repair is stopped:
    # queued trucks, trucks part-way through an activity and trucks that
    # finished a repair and are waiting to restart.
    out["shift_stop"] = shift_total - repair_in_shift
    return out


def station_minutes(log: SimLog, station: str, a: float, b: float) -> dict[str, float]:
    """Productive / blocked / shift / idle server-minutes for a station in [a, b]."""
    cfg = log.config
    capacity = cfg.n_shovels if station == SHOVEL else cfg.n_crushers
    service = LOAD if station == SHOVEL else DUMP
    scheduled = capacity * (b - a)

    productive = 0.0
    if log.work:
        w = [(r[3], r[4]) for r in log.work if r[1] == service]
        if w:
            arr = np.array(w)
            productive = float(clip_len(arr[:, 0], arr[:, 1], a, b).sum())

    blocked = blocked_in_shift = 0.0
    rep = [(x[1], x[2]) for x in log.repairs if x[3] == station]
    if rep:
        arr = np.array(rep)
        blocked = float(clip_len(arr[:, 0], arr[:, 1], a, b).sum())
        blocked_in_shift = float(overlap_with_windows(arr[:, 0], arr[:, 1], log.shift_windows, a, b).sum())

    shift = capacity * shift_minutes(log, a, b) - blocked_in_shift
    idle = scheduled - productive - blocked - shift
    return {"scheduled": scheduled, "productive": productive, "blocked": blocked, "shift": shift, "idle": idle}


# ---------------------------------------------------------------------- #
# Replication summary
# ---------------------------------------------------------------------- #
def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else float("nan")


def summarise(log: SimLog, a: float, b: float) -> dict[str, float]:
    """All headline measures for one replication over the window (a, b]."""
    cfg = log.config
    hours = (b - a) / 60.0
    n = cfg.n_trucks
    s: dict[str, float] = {"n_trucks": n, "window_h": hours}

    # Production: counted at dump completion only.
    d = np.array([(x[2], x[3]) for x in log.deliveries], dtype=float).reshape(-1, 2)
    in_w = (d[:, 0] > a) & (d[:, 0] <= b)
    s["delivered_t"] = float(d[in_w, 1].sum())
    s["deliveries"] = int(in_w.sum())
    s["tph"] = s["delivered_t"] / hours

    # Cycle times (completed cycles only).
    c = np.array([(x[1], x[2]) for x in log.cycles], dtype=float).reshape(-1, 2)
    ct = c[(c[:, 1] > a) & (c[:, 1] <= b)]
    ct = ct[:, 1] - ct[:, 0]
    s["cycles_completed"] = int(len(ct))
    s["cycle_mean_min"] = float(ct.mean()) if len(ct) else float("nan")
    s["cycle_p50_min"] = _pct(ct, 50)
    s["cycle_p90_min"] = _pct(ct, 90)
    s["cycle_p95_min"] = _pct(ct, 95)

    # Waiting time per visit (visits whose service started inside the window).
    for station in (SHOVEL, CRUSHER):
        wt = np.array([x[3] - x[2] for x in log.waits if x[1] == station and a < x[3] <= b])
        s[f"wait_{station}_mean_min"] = float(wt.mean()) if len(wt) else float("nan")
        s[f"wait_{station}_p90_min"] = _pct(wt, 90)
        s[f"queue_{station}_mean"] = time_weighted_mean(log.queue_len[station], a, b)

    # Station time split.
    for station in (SHOVEL, CRUSHER):
        m = station_minutes(log, station, a, b)
        for k in STATION_STATES:
            s[f"{station}_{k}_frac"] = m[k] / m["scheduled"]
        available = m["scheduled"] - m["shift"] - m["blocked"]
        s[f"{station}_util_available"] = m["productive"] / available if available > 0 else float("nan")

    # Truck time split (fleet totals / scheduled truck-time).
    states = truck_state_minutes(log, a, b)
    total = n * (b - a)
    for k in TRUCK_STATES:
        s[f"truck_{k}_frac"] = float(states[k].sum() / total)
    s["truck_working_frac"] = sum(s[f"truck_{k}_frac"] for k in WORK_STATES)
    s["truck_queue_frac"] = s["truck_queue_shovel_frac"] + s["truck_queue_crusher_frac"]
    s["truck_availability"] = 1.0 - s["truck_repair_frac"]
    s["failures"] = int(sum(1 for x in log.repairs if a < x[1] <= b))
    return s


# ---------------------------------------------------------------------- #
# Series used for warm-up analysis and plotting
# ---------------------------------------------------------------------- #
def hourly_tonnes(log: SimLog) -> np.ndarray:
    """Tonnes delivered in each whole hour of the run (warm-up included)."""
    n_hours = int(np.floor(log.horizon / 60.0))
    t = np.array([x[2] for x in log.deliveries], dtype=float)
    w = np.array([x[3] for x in log.deliveries], dtype=float)
    # Delivery at exactly t = 60k belongs to hour k-1, matching the (a, b] convention.
    idx = np.ceil(t / 60.0).astype(int) - 1
    keep = (idx >= 0) & (idx < n_hours)
    return np.bincount(idx[keep], weights=w[keep], minlength=n_hours)


def queue_timeline(log: SimLog, station: str) -> pd.DataFrame:
    pts = pd.DataFrame(log.queue_len[station], columns=["t_min", "queue"])
    # Keep the last value at each instant so the step plot is clean.
    return pts.groupby("t_min", as_index=False).last()
