"""Replicated experiments: fleet-size sweeps, scenarios and run-length checks.

Seeds
-----
Replication ``r`` uses a seed derived from ``(base_seed, r)``. The *same*
replication seed is used for every fleet size and scenario (common random
numbers). Replications remain mutually independent, so per-fleet
confidence intervals are valid, and paired differences between fleet
sizes (marginal gains) have lower variance than unpaired ones.

Every replication row records its seed, so any single run can be
re-created exactly with ``simulate(cfg, horizon, seed)``.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .config import ExperimentConfig, MineConfig
from .metrics import hourly_tonnes, summarise
from .model import simulate

SCENARIOS: dict[str, dict] = {
    "baseline": {"breakdowns_enabled": False, "shifts_enabled": False},
    "breakdowns only": {"breakdowns_enabled": True, "shifts_enabled": False},
    "shifts only": {"breakdowns_enabled": False, "shifts_enabled": True},
    "combined": {"breakdowns_enabled": True, "shifts_enabled": True},
}


def replication_seed(base_seed: int, replication: int) -> int:
    """Deterministic, well-mixed 32-bit seed for one replication."""
    return int(np.random.SeedSequence([int(base_seed), int(replication)]).generate_state(1)[0])


def _run_one(task: tuple) -> dict:
    cfg_dict, n, rep, seed, warmup_min, horizon_min, tags = task
    cfg = MineConfig.from_dict(cfg_dict).with_trucks(n)
    log = simulate(cfg, horizon_min, seed)
    row = {"replication": rep, "seed": seed, **tags}
    row.update(summarise(log, warmup_min, horizon_min))
    return row


def _execute(tasks: list[tuple], workers: int, progress: Callable[[int, int], None] | None) -> pd.DataFrame:
    rows: list[dict] = []
    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            rows.append(_run_one(task))
            if progress:
                progress(i, len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, row in enumerate(pool.map(_run_one, tasks, chunksize=4), 1):
                rows.append(row)
                if progress:
                    progress(i, len(tasks))
    return pd.DataFrame(rows)


def run_fleet_study(
    mine: MineConfig,
    exp: ExperimentConfig,
    progress: Callable[[int, int], None] | None = None,
    scenario: str | None = None,
) -> pd.DataFrame:
    """One row per (fleet size, replication) with every summary measure."""
    tags = {"scenario": scenario} if scenario else {}
    cfg_dict = mine.to_dict()
    tasks = [
        (cfg_dict, n, r, replication_seed(exp.base_seed, r), exp.warmup_min, exp.horizon_min, tags)
        for n in exp.fleet_sizes
        for r in range(exp.replications)
    ]
    return _execute(tasks, exp.workers, progress)


def run_scenarios(
    mine: MineConfig,
    exp: ExperimentConfig,
    scenarios: Iterable[str] = tuple(SCENARIOS),
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Fleet sweep repeated for baseline / breakdown-only / shift-only / combined."""
    tasks = []
    for name in scenarios:
        cfg_dict = replace(mine, **SCENARIOS[name]).to_dict()
        for n in exp.fleet_sizes:
            for r in range(exp.replications):
                seed = replication_seed(exp.base_seed, r)
                tasks.append((cfg_dict, n, r, seed, exp.warmup_min, exp.horizon_min, {"scenario": name}))
    return _execute(tasks, exp.workers, progress)


def warmup_profile(mine: MineConfig, n_trucks: int, hours: int, replications: int, base_seed: int) -> pd.DataFrame:
    """Hourly delivered tonnes from t = 0, averaged over replications.

    Trucks start bunched at the shovel, so the first hours show a start-up
    transient. A warm-up is adequate once the averaged curve has settled
    into its long-run pattern.
    """
    cfg = mine.with_trucks(n_trucks)
    series = np.array(
        [hourly_tonnes(simulate(cfg, hours * 60.0, replication_seed(base_seed, r))) for r in range(replications)]
    )
    return pd.DataFrame(
        {
            "hour": np.arange(1, hours + 1),
            "mean_t": series.mean(axis=0),
            "std_t": series.std(axis=0, ddof=1),
        }
    )


def run_length_sensitivity(
    mine: MineConfig,
    n_trucks: int,
    warmups_h: Iterable[float],
    measure_days: Iterable[int],
    replications: int,
    base_seed: int,
    workers: int = 1,
) -> pd.DataFrame:
    """Throughput estimate for each (warm-up, measurement-length) pair."""
    tasks = []
    cfg_dict = mine.to_dict()
    for w in warmups_h:
        for d in measure_days:
            exp = ExperimentConfig(fleet_sizes=(n_trucks,), replications=replications, warmup_h=w, measure_days=int(d))
            for r in range(replications):
                tags = {"warmup_h": w, "measure_days": int(d)}
                tasks.append(
                    (cfg_dict, n_trucks, r, replication_seed(base_seed, r), exp.warmup_min, exp.horizon_min, tags)
                )
    return _execute(tasks, workers, None)


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def save_run_record(
    out_dir: Path,
    mine: MineConfig,
    exp: ExperimentConfig,
    extra: dict | None = None,
) -> None:
    """Write the parameters and replication seeds needed to reproduce a study."""
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "note": "Synthetic demonstration model. Results are model estimates, not mine measurements.",
        "time_unit_internal": "minutes",
        "mine": mine.to_dict(),
        "experiment": exp.to_dict(),
        "replication_seeds": {r: replication_seed(exp.base_seed, r) for r in range(exp.replications)},
        **(extra or {}),
    }
    (out_dir / "parameters.json").write_text(json.dumps(record, indent=2, default=str))
