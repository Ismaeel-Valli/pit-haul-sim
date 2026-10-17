"""Command-line entry point: ``python -m haulsim <command>``.

Commands
--------
demo         quick look: few replications, short runs (about 10 seconds)
study        full fleet study: >= 30 replications per fleet size, saved to disk
scenarios    baseline / breakdowns-only / shifts-only / combined comparison
sensitivity  warm-up profile and warm-up x run-length check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from .analysis import analyse_study, capacity_bounds, mean_ci
from .config import AnalysisConfig, ExperimentConfig, MineConfig
from .experiment import (
    default_workers,
    run_fleet_study,
    run_length_sensitivity,
    run_scenarios,
    save_run_record,
    warmup_profile,
)

LABEL = "SYNTHETIC MODEL ESTIMATES - not measurements from a real mine."


def parse_fleet(text: str) -> tuple[int, ...]:
    """'1-20' or '2,4,6,8' or '1-10,12,15'."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return tuple(sorted(set(out)))


def load_mine(path: str | None) -> MineConfig:
    if not path:
        return MineConfig()
    return MineConfig.from_dict(json.loads(Path(path).read_text()))


def _progress(prefix: str):
    start = time.perf_counter()

    def report(i: int, n: int) -> None:
        if i == n or i % max(1, n // 20) == 0:
            sys.stderr.write(f"\r{prefix}: {i}/{n} runs ({time.perf_counter() - start:.0f}s)")
            if i == n:
                sys.stderr.write("\n")
            sys.stderr.flush()

    return report


def _print_study(res, exp: ExperimentConfig) -> None:
    s = res.summary
    cols = {
        "tph_mean": "t/h",
        "tph_lo": "CI lo",
        "tph_hi": "CI hi",
        "gain_mean": "gain/truck",
        "cycle_mean_min_mean": "cycle min",
        "wait_crusher_mean_min_mean": "wait crush",
        "wait_shovel_mean_min_mean": "wait shovel",
        "queue_crusher_mean_mean": "Lq crush",
        "crusher_util_available_mean": "crush util",
        "shovel_util_available_mean": "shovel util",
        "truck_queue_frac_mean": "truck queue",
    }
    table = s[list(cols)].rename(columns=cols)
    table["limit"] = pd.Series(res.limiting)
    print(f"\n{LABEL}")
    print(
        f"{exp.replications} replications x {len(exp.fleet_sizes)} fleet sizes; warm-up {exp.warmup_h:g} h; "
        f"measurement {exp.measure_days} day(s). 95% CIs from replication means.\n"
    )
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 2):
        print(table.round(2).to_string())
    tp = res.tipping
    print(f"\nTipping-point rule: {tp.rule}")
    print(f"  status: {tp.status}; plateau ~{tp.plateau_tph:.0f} t/h (95% CI {tp.plateau_lo:.0f}-{tp.plateau_hi:.0f})")
    if tp.fleet is not None:
        print(f"  operational tipping point: {tp.fleet} trucks (plausible range {tp.fleet_range[0]}-{tp.fleet_range[1]})")
    for m in tp.messages:
        print(f"  note: {m}")
    t = res.target
    print(
        f"Production target {t['target_tph']:.0f} t/h: met on average with {t['fleet_mean']} trucks, "
        f"with 95% confidence with {t['fleet_confident']} trucks. {'' if t['message'] == 'ok' else t['message']}"
    )
    c = res.economic_choice
    print(
        f"Illustrative economics: least cost/t overall at {c['least_cost_overall']} trucks; "
        f"least cost/t meeting target at {c['least_cost_meeting_target']} trucks."
    )
    print(f"\n{res.text}\n")


def write_study(out: Path, df: pd.DataFrame, res, mine: MineConfig, exp: ExperimentConfig, acfg: AnalysisConfig) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save_run_record(out, mine, exp, {"analysis": acfg.to_dict()})
    df.to_csv(out / "replications.csv", index=False)
    res.summary.to_csv(out / "summary.csv")
    res.bounds.to_csv(out / "bounds.csv")
    res.economics.to_csv(out / "economics.csv")
    (out / "recommendation.json").write_text(json.dumps(res.recommendation(), indent=2, default=str))
    print(f"Saved results to {out.resolve()}")


# ---------------------------------------------------------------------- #
def cmd_demo(args) -> None:
    mine = load_mine(args.config)
    exp = ExperimentConfig(
        fleet_sizes=parse_fleet(args.fleet), replications=args.reps, warmup_h=args.warmup_h,
        measure_days=args.days, base_seed=args.seed, workers=args.workers,
    )
    df = run_fleet_study(mine, exp, _progress("demo"))
    res = analyse_study(df, mine, AnalysisConfig(target_tph=args.target))
    _print_study(res, exp)
    if args.out:
        write_study(Path(args.out), df, res, mine, exp, AnalysisConfig(target_tph=args.target))


def cmd_study(args) -> None:
    mine = load_mine(args.config)
    exp = ExperimentConfig(
        fleet_sizes=parse_fleet(args.fleet), replications=args.reps, warmup_h=args.warmup_h,
        measure_days=args.days, base_seed=args.seed, workers=args.workers,
    )
    acfg = AnalysisConfig(target_tph=args.target, truck_cost_per_h=args.truck_cost, fixed_cost_per_h=args.fixed_cost)
    if args.reps < 30:
        print("warning: study mode is intended for >= 30 replications per fleet size", file=sys.stderr)
    df = run_fleet_study(mine, exp, _progress("study"))
    res = analyse_study(df, mine, acfg)
    _print_study(res, exp)
    write_study(Path(args.out), df, res, mine, exp, acfg)


def cmd_scenarios(args) -> None:
    mine = load_mine(args.config)
    exp = ExperimentConfig(
        fleet_sizes=parse_fleet(args.fleet), replications=args.reps, warmup_h=args.warmup_h,
        measure_days=args.days, base_seed=args.seed, workers=args.workers,
    )
    df = run_scenarios(mine, exp, progress=_progress("scenarios"))
    rows = []
    for (scen, n), g in df.groupby(["scenario", "n_trucks"]):
        m, lo, hi = mean_ci(g["tph"])
        rows.append({"scenario": scen, "n_trucks": n, "tph_mean": m, "tph_lo": lo, "tph_hi": hi})
    table = pd.DataFrame(rows).pivot(index="n_trucks", columns="scenario", values="tph_mean")
    print(f"\n{LABEL}\nMean throughput (t/h) by scenario:\n")
    print(table.round(0).to_string())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_run_record(out, mine, exp, {"scenarios": "baseline, breakdowns only, shifts only, combined"})
    df.to_csv(out / "scenario_replications.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "scenario_summary.csv", index=False)
    print(f"Saved results to {out.resolve()}")


def cmd_sensitivity(args) -> None:
    mine = load_mine(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "note": "Synthetic demonstration model. Results are model estimates, not mine measurements.",
        "mine": mine.to_dict(),
        "fleet_size": args.trucks,
        "replications": args.reps,
        "base_seed": args.seed,
        "warmups_h": [0.0, 6.0, 24.0, 48.0],
        "measure_days": [3, 7, 14, 28],
        "profile_hours": args.profile_hours,
    }
    (out / "parameters.json").write_text(json.dumps(record, indent=2))
    prof = warmup_profile(mine, args.trucks, args.profile_hours, args.reps, args.seed)
    prof.to_csv(out / "warmup_profile.csv", index=False)
    steady = prof["mean_t"].iloc[48:].mean() if len(prof) > 48 else prof["mean_t"].mean()
    print(f"\n{LABEL}\nWarm-up profile, {args.trucks} trucks, mean of {args.reps} replications (t per hour):")
    print("  hour: " + " ".join(f"{h:>4d}" for h in prof["hour"].iloc[:12]))
    print("  t/h : " + " ".join(f"{v:4.0f}" for v in prof["mean_t"].iloc[:12]))
    print(f"  average from hour 49 onwards: {steady:.0f} t/h")

    grid = run_length_sensitivity(
        mine, args.trucks, [0.0, 6.0, 24.0, 48.0], [3, 7, 14, 28], args.reps, args.seed, args.workers
    )
    rows = []
    for (w, d), g in grid.groupby(["warmup_h", "measure_days"]):
        m, lo, hi = mean_ci(g["tph"])
        rows.append({"warmup_h": w, "measure_days": d, "tph_mean": m, "ci_half_width": (hi - lo) / 2})
    table = pd.DataFrame(rows)
    table.to_csv(out / "run_length_sensitivity.csv", index=False)
    print("\nThroughput estimate (t/h) +/- 95% CI half-width by warm-up (rows) and measurement days (columns):")
    piv = table.assign(cell=table.apply(lambda r: f"{r.tph_mean:.0f} +/- {r.ci_half_width:.1f}", axis=1))
    print(piv.pivot(index="warmup_h", columns="measure_days", values="cell").to_string())
    print(f"Saved results to {out.resolve()}")


def cmd_bounds(args) -> None:
    mine = load_mine(args.config)
    for n in parse_fleet(args.fleet):
        b = capacity_bounds(mine, n)
        print(
            f"{n:>3d} trucks: fleet {b['fleet_tph']:7.1f} | shovel {b['shovel_tph']:7.1f} | "
            f"crusher {b['crusher_tph']:7.1f} | ideal bound {b['ideal_bound_tph']:7.1f} | "
            f"adjusted bound {b['adj_bound_tph']:7.1f} ({b['limiting_resource']})"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="haulsim", description="Open-pit truck haulage simulation (synthetic model)")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, fleet, reps, days, warmup, out):
        sp.add_argument("--config", help="JSON file with MineConfig overrides")
        sp.add_argument("--fleet", default=fleet, help="fleet sizes, e.g. 1-20 or 2,4,6")
        sp.add_argument("--reps", type=int, default=reps, help="replications per fleet size")
        sp.add_argument("--days", type=int, default=days, help="measurement period in whole days")
        sp.add_argument("--warmup-h", type=float, default=warmup, help="warm-up period in hours")
        sp.add_argument("--seed", type=int, default=20261017, help="base random seed")
        sp.add_argument("--workers", type=int, default=default_workers(), help="parallel processes")
        sp.add_argument("--out", default=out, help="output directory")
        sp.add_argument("--target", type=float, default=800.0, help="production target, t per scheduled hour")

    d = sub.add_parser("demo", help="quick demonstration run")
    common(d, "1-12", 5, 3, 12.0, None)
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("study", help="full replicated fleet study")
    common(s, "1-20", 30, 14, 24.0, "results/study")
    s.add_argument("--truck-cost", type=float, default=300.0, help="illustrative cost per truck per hour")
    s.add_argument("--fixed-cost", type=float, default=2500.0, help="illustrative fixed cost per hour")
    s.set_defaults(func=cmd_study)

    sc = sub.add_parser("scenarios", help="baseline / breakdowns / shifts / combined")
    common(sc, "1-14", 30, 14, 24.0, "results/scenarios")
    sc.set_defaults(func=cmd_scenarios)

    se = sub.add_parser("sensitivity", help="warm-up and run-length check")
    common(se, "8", 30, 14, 24.0, "results/sensitivity")
    se.add_argument("--trucks", type=int, default=7, help="fleet size to examine")
    se.add_argument("--profile-hours", type=int, default=96, help="hours shown in warm-up profile")
    se.set_defaults(func=cmd_sensitivity)

    b = sub.add_parser("bounds", help="print idealised capacity bounds")
    b.add_argument("--config")
    b.add_argument("--fleet", default="1-12")
    b.set_defaults(func=cmd_bounds)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
