"""Truck positions for a 2D replay, reconstructed from simulation event logs.

The replay never re-simulates or invents motion: every frame is a lookup
into the same logs that the metrics use. Simulation time (which instants
are shown) and playback speed (how fast frames are displayed) are
independent; this module only deals with simulation time.

Layout (arbitrary units): shovel at (0, 0), crusher at (10, 0). Loaded
trucks travel along the upper arc, empty trucks return along the lower
arc. Position along a road is proportional to the fraction of that
travel activity completed, so a truck paused by a stoppage or breakdown
stays where it stopped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import CRUSHER, SHOVEL, SimLog

SHOVEL_XY = (0.0, 0.0)
CRUSHER_XY = (10.0, 0.0)
ROAD_HEIGHT = 2.2

STATUS_ORDER = (
    "Hauling (loaded)",
    "Returning (empty)",
    "Loading / dumping",
    "Queued",
    "Under repair",
    "Shift stoppage",
)


def road_xy(activity: str, frac: float) -> tuple[float, float]:
    f = min(max(frac, 0.0), 1.0)
    if activity == "haul":
        return 10.0 * f, ROAD_HEIGHT * np.sin(np.pi * f)
    if activity == "return":
        return 10.0 * (1 - f), -ROAD_HEIGHT * np.sin(np.pi * f)
    return SHOVEL_XY if activity == "load" else CRUSHER_XY


def queue_xy(station: str, slot: int) -> tuple[float, float]:
    col, row = slot % 6, slot // 6
    if station == SHOVEL:
        return -1.0 - 0.6 * col, -0.6 * row
    return 11.0 + 0.6 * col, -0.6 * row


def road_path(activity: str, n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    pts = [road_xy(activity, f) for f in np.linspace(0, 1, n)]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


class _Intervals:
    """Sorted, non-overlapping intervals of one truck with payload rows."""

    def __init__(self, rows: list[tuple], start_idx: int, end_idx: int):
        rows = sorted(rows, key=lambda r: r[start_idx])
        self.rows = rows
        self.starts = np.array([r[start_idx] for r in rows], dtype=float)
        ends = np.array([r[end_idx] for r in rows], dtype=float)
        self.ends = np.where(np.isnan(ends), np.inf, ends)

    def at(self, t: float):
        i = int(np.searchsorted(self.starts, t, side="right")) - 1
        if i >= 0 and self.ends[i] > t:
            return self.rows[i]
        return None

    def last_before(self, t: float):
        """Last interval that started at or before t."""
        i = int(np.searchsorted(self.starts, t, side="right")) - 1
        return self.rows[i] if i >= 0 else None


def build_frames(log: SimLog, t_start: float, t_end: float, step: float) -> pd.DataFrame:
    """One row per (frame time, truck): status, x, y and tonnes on board."""
    n = log.config.n_trucks
    by_truck = lambda rows: [[r for r in rows if r[0] == i] for i in range(n)]  # noqa: E731
    work = [_Intervals(rows, 3, 4) for rows in by_truck(log.work)]
    repairs = [_Intervals(rows, 1, 2) for rows in by_truck(log.repairs)]
    waits = [_Intervals(rows, 2, 3) for rows in by_truck(log.waits)]
    holds = [_Intervals(rows, 2, 3) for rows in by_truck(log.holds)]
    loads = [_Intervals([(r[0], r[2]) for r in log.loads if r[0] == i], 1, 1) for i in range(n)]
    dels = [_Intervals([(r[0], r[2]) for r in log.deliveries if r[0] == i], 1, 1) for i in range(n)]
    shifts = np.array(log.shift_windows, dtype=float).reshape(-1, 2)

    def in_shift(t: float) -> bool:
        return bool(((shifts[:, 0] <= t) & (t < shifts[:, 1])).any())

    def work_xy(seg, t: float):
        _, kind, _, s, e, before, total = seg
        done = before + (min(t, e) - s)
        return road_xy(kind, done / total if total > 0 else 1.0)

    def on_board(i: int, t: float) -> float:
        last_load, last_del = loads[i].last_before(t), dels[i].last_before(t)
        if last_load is None:
            return 0.0
        if last_del is None or last_load[1] > last_del[1]:
            return log.config.payload_t
        return 0.0

    rows = []
    for t in np.arange(t_start, t_end + 1e-9, step):
        waiting = {
            st: sorted((w[2], i) for i in range(n) if (w := waits[i].at(t)) is not None and w[1] == st)
            for st in (SHOVEL, CRUSHER)
        }
        slot = {i: (st, k) for st in waiting for k, (_, i) in enumerate(waiting[st])}
        for i in range(n):
            if (rep := repairs[i].at(t)) is not None:
                seg = work[i].last_before(rep[1])
                xy = work_xy(seg, rep[1]) if seg else SHOVEL_XY
                status = "Under repair"
            elif i in slot:
                xy, status = queue_xy(*slot[i]), "Queued"
            elif (seg := work[i].at(t)) is not None:
                xy = work_xy(seg, t)
                status = {"haul": "Hauling (loaded)", "return": "Returning (empty)"}.get(seg[1], "Loading / dumping")
            elif (h := holds[i].at(t)) is not None:
                xy = SHOVEL_XY if h[1] == SHOVEL else CRUSHER_XY
                status = "Shift stoppage" if in_shift(t) else "Loading / dumping"
            else:
                seg = work[i].last_before(t)
                xy = work_xy(seg, t) if seg else SHOVEL_XY
                if in_shift(t):
                    status = "Shift stoppage"
                elif seg is None:
                    status = "Queued"
                else:
                    status = {"haul": "Hauling (loaded)", "return": "Returning (empty)"}.get(seg[1], "Loading / dumping")
            rows.append((float(t), i, status, float(xy[0]), float(xy[1]), on_board(i, t)))
    return pd.DataFrame(rows, columns=["t", "truck", "status", "x", "y", "on_board_t"])


def delivered_between(log: SimLog, a: float, times) -> np.ndarray:
    """Cumulative tonnes delivered in (a, t] for each t - same rule as metrics."""
    d = np.array([(x[2], x[3]) for x in log.deliveries], dtype=float).reshape(-1, 2)
    return np.array([d[(d[:, 0] > a) & (d[:, 0] <= t), 1].sum() for t in times])
