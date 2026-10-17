"""Statistical summaries and decision rules built on replication results.

Three different questions are kept separate on purpose:

1. **Operational tipping point** - the smallest fleet that reaches a stated
   fraction (default 95 %) of the estimated production plateau. Beyond it,
   extra trucks mostly add queuing.
2. **Fleet for a production target** - the smallest fleet whose throughput
   meets a target (both "on average" and "with 95 % confidence").
3. **Economically preferred fleet** - the least-cost-per-tonne fleet among
   those meeting the target, under explicit, user-supplied costs.

Maximum throughput alone never decides how many trucks to buy.

Confidence intervals are computed from *replication-level* summaries
(one number per independent replication), never from individual trips,
which are autocorrelated within a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .config import AnalysisConfig, MineConfig

KEY_METRICS = (
    "tph",
    "delivered_t",
    "cycle_mean_min",
    "cycle_p90_min",
    "wait_shovel_mean_min",
    "wait_crusher_mean_min",
    "queue_shovel_mean",
    "queue_crusher_mean",
    "shovel_util_available",
    "crusher_util_available",
    "shovel_productive_frac",
    "crusher_productive_frac",
    "shovel_blocked_frac",
    "crusher_blocked_frac",
    "shovel_shift_frac",
    "crusher_shift_frac",
    "shovel_idle_frac",
    "crusher_idle_frac",
    "truck_working_frac",
    "truck_queue_frac",
    "truck_queue_shovel_frac",
    "truck_queue_crusher_frac",
    "truck_repair_frac",
    "truck_shift_stop_frac",
    "truck_availability",
    "failures",
)


# ---------------------------------------------------------------------- #
# Confidence intervals
# ---------------------------------------------------------------------- #
def mean_ci(x, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and two-sided t-based confidence interval of independent values."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return (float("nan"),) * 3
    m = float(x.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    half = float(stats.t.ppf(0.5 + confidence / 2, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return m, m - half, m + half


def summarise_fleet(df: pd.DataFrame, metrics=KEY_METRICS, by: str = "n_trucks") -> pd.DataFrame:
    """Per-fleet mean and 95 % CI (columns ``<metric>_mean/_lo/_hi``)."""
    rows = []
    for key, g in df.groupby(by, sort=True):
        row = {by: key, "replications": len(g)}
        for m in metrics:
            if m in g:
                row[f"{m}_mean"], row[f"{m}_lo"], row[f"{m}_hi"] = mean_ci(g[m])
        rows.append(row)
    return pd.DataFrame(rows).set_index(by)


def marginal_gains(df: pd.DataFrame, metric: str = "tph") -> pd.DataFrame:
    """Throughput gained per added truck between successive tested fleet sizes.

    Uses paired differences by replication (same seed, common random
    numbers), so the CI reflects the difference itself rather than the
    noise in each level.
    """
    fleets = sorted(df["n_trucks"].unique())
    piv = df.pivot_table(index="replication", columns="n_trucks", values=metric)
    rows = []
    for lo, hi in zip(fleets[:-1], fleets[1:]):
        diff = (piv[hi] - piv[lo]) / (hi - lo)
        m, l, h = mean_ci(diff)
        rows.append({"n_trucks": hi, "from_trucks": lo, "gain_mean": m, "gain_lo": l, "gain_hi": h})
    return pd.DataFrame(rows).set_index("n_trucks")


# ---------------------------------------------------------------------- #
# Idealised capacity bounds
# ---------------------------------------------------------------------- #
def capacity_bounds(cfg: MineConfig, n_trucks: int | None = None) -> dict[str, float]:
    """Simple throughput bounds in tonnes per scheduled hour.

    Idealised (no variability, failures or stoppages):

    * fleet  = N * payload * 60 / E[cycle without queuing]
    * shovel = shovels * payload * 60 / E[load time]
    * crusher = stations * payload * 60 / E[dump time]

    The system cannot deliver more than the smallest of the three.

    Availability-adjusted (expected-value approximation): stations lose the
    shift-stoppage fraction ``s``. A truck also needs ``rho = E[repair] /
    MTBF`` minutes of repair per operating minute; repairs continue through
    stoppages, giving a working fraction of ``(1 - s) / (1 + rho (1 - s))``.
    Station blocking by broken trucks is ignored, so the adjusted bound is
    still optimistic.

    Realised throughput is lower than these bounds because random service
    times make trucks bunch (a station can idle while trucks are elsewhere,
    then face a queue), because failures block stations, and because
    stoppages interrupt cycles.
    """
    n = cfg.n_trucks if n_trucks is None else n_trucks
    p = cfg.payload_t
    s = cfg.shift_stop_min / cfg.shift_interval_min if cfg.shifts_enabled else 0.0
    rho = cfg.repair_min.mean / cfg.mtbf_operating_min if cfg.breakdowns_enabled else 0.0
    truck_frac = (1 - s) / (1 + rho * (1 - s))
    ideal = {
        "fleet_tph": n * p * 60.0 / cfg.mean_cycle_work_min,
        "shovel_tph": cfg.n_shovels * p * 60.0 / cfg.load_min.mean,
        "crusher_tph": cfg.n_crushers * p * 60.0 / cfg.dump_min.mean,
    }
    out = dict(ideal)
    out["ideal_bound_tph"] = min(ideal.values())
    out["fleet_adj_tph"] = ideal["fleet_tph"] * truck_frac
    out["shovel_adj_tph"] = ideal["shovel_tph"] * (1 - s)
    out["crusher_adj_tph"] = ideal["crusher_tph"] * (1 - s)
    out["adj_bound_tph"] = min(out["fleet_adj_tph"], out["shovel_adj_tph"], out["crusher_adj_tph"])
    out["truck_working_frac_max"] = truck_frac
    out["shift_frac"] = s
    limits = {"truck fleet": out["fleet_adj_tph"], "shovel": out["shovel_adj_tph"], "crusher": out["crusher_adj_tph"]}
    out["limiting_resource"] = min(limits, key=limits.get)
    return out


def bounds_table(cfg: MineConfig, fleet_sizes) -> pd.DataFrame:
    rows = [{"n_trucks": n, **capacity_bounds(cfg, n)} for n in fleet_sizes]
    return pd.DataFrame(rows).set_index("n_trucks")


def station_capacity_fleet(cfg: MineConfig) -> float:
    """Fleet size at which idealised truck capacity equals station capacity."""
    b = capacity_bounds(cfg, 1)
    return min(b["shovel_adj_tph"], b["crusher_adj_tph"]) / b["fleet_adj_tph"]


# ---------------------------------------------------------------------- #
# Decision rules
# ---------------------------------------------------------------------- #
@dataclass
class TippingPoint:
    status: str  # "ok" | "uncertain" | "plateau_not_reached" | "insufficient_range"
    rule: str
    fleet: int | None = None
    fleet_range: tuple[int | None, int | None] = (None, None)
    plateau_tph: float = float("nan")
    plateau_lo: float = float("nan")
    plateau_hi: float = float("nan")
    threshold_tph: float = float("nan")
    tail_fleets: tuple[int, ...] = ()
    tail_rise_frac: float = float("nan")
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def find_tipping_point(df: pd.DataFrame, acfg: AnalysisConfig, metric: str = "tph") -> TippingPoint:
    """Smallest fleet reaching ``plateau_fraction`` of the estimated plateau.

    Plateau estimate: for each replication, average throughput over the
    ``plateau_tail`` largest fleet sizes tested; the plateau is the mean of
    those replication values (with a 95 % CI). Using several fleets rather
    than the single maximum avoids picking the luckiest noisy estimate.

    Flatness check: the paired rise in throughput from the first to the last
    fleet of that tail is compared with ``plateau_tolerance`` (as a fraction
    of the plateau). If even the lower 95 % CI of the rise exceeds the
    tolerance, the range has not reached a plateau and no tipping point is
    reported. If only the upper CI exceeds it, the result is "uncertain".

    Uncertainty: the threshold is compared with each fleet's CI. The
    earliest fleet whose upper CI reaches the threshold and the earliest
    whose lower CI reaches it bound the plausible tipping point. If they
    are more than one tested size apart, the answer is flagged uncertain.
    """
    k = acfg.plateau_tail
    pct = round(acfg.plateau_fraction * 100)
    rule = f"Smallest fleet whose mean throughput >= {pct}% of the plateau (mean of the {k} largest fleets tested)"
    fleets = sorted(int(n) for n in df["n_trucks"].unique())
    if len(fleets) < k + 1:
        return TippingPoint("insufficient_range", rule, messages=[f"Need at least {k + 1} fleet sizes to estimate a plateau."])

    tail = fleets[-k:]
    piv = df.pivot_table(index="replication", columns="n_trucks", values=metric)
    plateau, p_lo, p_hi = mean_ci(piv[tail].mean(axis=1))
    rise, rise_lo, rise_hi = (v / plateau for v in mean_ci(piv[tail[-1]] - piv[tail[0]]))
    tp = TippingPoint(
        "ok", rule, plateau_tph=plateau, plateau_lo=p_lo, plateau_hi=p_hi,
        threshold_tph=acfg.plateau_fraction * plateau, tail_fleets=tuple(tail),
        tail_rise_frac=rise,
    )

    tol = acfg.plateau_tolerance
    span = f"between {tail[0]} and {tail[-1]} trucks"
    if rise_lo > tol:
        tp.status = "plateau_not_reached"
        tp.messages.append(
            f"Throughput still rises {rise:.1%} {span} (95% CI {rise_lo:.1%} to {rise_hi:.1%}, "
            f"tolerance {tol:.0%}). Extend the fleet range before choosing a tipping point."
        )
        return tp
    plateau_uncertain = rise_hi > tol
    if plateau_uncertain:
        tp.messages.append(
            f"The plateau is not confirmed: the rise {span} is {rise:.1%} (95% CI {rise_lo:.1%} to {rise_hi:.1%}) "
            f"against a {tol:.0%} tolerance. Add replications or extend the fleet range."
        )

    summ = summarise_fleet(df, metrics=(metric,))
    mean, lo, hi = summ[f"{metric}_mean"], summ[f"{metric}_lo"], summ[f"{metric}_hi"]
    thr = tp.threshold_tph

    def first(series):
        hit = series[series >= thr]
        return int(hit.index[0]) if len(hit) else None

    tp.fleet, early, late = first(mean), first(hi), first(lo)
    tp.fleet_range = (early, late)
    if tp.fleet is None or late is None:
        tp.status = "uncertain"
        tp.messages.append("No tested fleet is confidently above the threshold; add replications or fleet sizes.")
    elif fleets.index(late) - fleets.index(early) > 1:
        tp.status = "uncertain"
        tp.messages.append(
            f"Confidence intervals place the tipping point anywhere from {early} to {late} trucks; "
            "add replications to narrow it."
        )
    if plateau_uncertain:
        tp.status = "uncertain"
    if fleets != list(range(fleets[0], fleets[-1] + 1)):
        tp.messages.append("Fleet sizes are not consecutive; the tipping point is only resolved to the tested sizes.")
    return tp


def fleet_for_target(summary: pd.DataFrame, target_tph: float, metric: str = "tph") -> dict:
    """Smallest fleet meeting a target on average and with 95 % confidence."""
    mean, lo = summary[f"{metric}_mean"], summary[f"{metric}_lo"]
    on_avg = mean[mean >= target_tph]
    conf = lo[lo >= target_tph]
    out = {
        "target_tph": target_tph,
        "fleet_mean": int(on_avg.index[0]) if len(on_avg) else None,
        "fleet_confident": int(conf.index[0]) if len(conf) else None,
    }
    if out["fleet_mean"] is None:
        out["message"] = (
            f"No tested fleet reaches {target_tph:.0f} t/h (best {mean.max():.0f} t/h). "
            "The target may exceed station capacity; adding trucks will not fix that."
        )
    elif out["fleet_confident"] is None:
        out["message"] = "The target is met on average but not with 95 % confidence in the tested range."
    else:
        out["message"] = "ok"
    return out


def economics(summary: pd.DataFrame, acfg: AnalysisConfig, metric: str = "tph") -> tuple[pd.DataFrame, dict]:
    """Cost per tonne by fleet size using ILLUSTRATIVE hourly costs.

    All costs are per scheduled hour, the same period as throughput
    (tonnes per scheduled hour), so cost per tonne = cost/h / (t/h).
    """
    n = summary.index.to_numpy()
    tph, lo, hi = (summary[f"{metric}_{s}"].to_numpy() for s in ("mean", "lo", "hi"))
    cost_h = n * acfg.truck_cost_per_h + acfg.fixed_cost_per_h
    eco = pd.DataFrame(
        {
            "cost_per_h": cost_h,
            "tph_mean": tph,
            "cost_per_t": cost_h / tph,
            "cost_per_t_lo": cost_h / hi,
            "cost_per_t_hi": cost_h / lo,
            "meets_target_confident": lo >= acfg.target_tph,
        },
        index=summary.index,
    )
    if acfg.value_per_t is not None:
        eco["profit_per_h"] = acfg.value_per_t * tph - cost_h
    choice: dict = {"target_tph": acfg.target_tph, "least_cost_overall": int(eco["cost_per_t"].idxmin())}
    feasible = eco[eco["meets_target_confident"]]
    choice["least_cost_meeting_target"] = int(feasible["cost_per_t"].idxmin()) if len(feasible) else None
    if acfg.value_per_t is not None:
        choice["max_profit"] = int(eco["profit_per_h"].idxmax())
    return eco, choice


def limiting_resource(row: pd.Series, threshold: float = 0.90) -> str:
    """Which resource is binding at one fleet size, judged from the results.

    A station whose utilisation of *available* time is >= ``threshold``
    is saturated. If no station is saturated, the truck fleet (its size,
    cycle time and availability) is what limits production.
    """
    utils = {
        "shovel": row.get("shovel_util_available_mean", np.nan),
        "crusher": row.get("crusher_util_available_mean", np.nan),
    }
    station = max(utils, key=lambda k: -1 if np.isnan(utils[k]) else utils[k])
    return station if utils[station] >= threshold else "truck fleet"


def interpretation(summary: pd.DataFrame, gains: pd.DataFrame, tp: TippingPoint, cfg: MineConfig) -> str:
    """Plain-language explanation of why (or whether) production plateaus."""
    fleets = list(summary.index)
    small, large = fleets[0], fleets[-1]
    r_small, r_large = summary.loc[small], summary.loc[large]
    lim_small, lim_large = limiting_resource(r_small), limiting_resource(r_large)
    b = capacity_bounds(cfg, large)
    lines = [
        f"With {small} truck(s) the {lim_small} limits output: {r_small['tph_mean']:.0f} t/h, "
        f"trucks queue {r_small['truck_queue_frac_mean']:.0%} of the time.",
        f"With {large} trucks the {lim_large} limits output: {r_large['tph_mean']:.0f} t/h "
        f"(availability-adjusted bound {b['adj_bound_tph']:.0f} t/h, set by the {b['limiting_resource']}). "
        f"The crusher is productively busy {r_large['crusher_util_available_mean']:.0%} and the shovel "
        f"{r_large['shovel_util_available_mean']:.0%} of their available time; trucks spend "
        f"{r_large['truck_queue_frac_mean']:.0%} of scheduled time queuing.",
    ]
    if tp.status in ("ok", "uncertain") and tp.fleet is not None:
        g = gains.loc[gains.index > tp.fleet, "gain_mean"]
        tail_gain = f" Each truck beyond that adds on average {g.mean():.0f} t/h." if len(g) else ""
        lines.append(
            f"Throughput reaches {tp.threshold_tph:.0f} t/h ({tp.threshold_tph / tp.plateau_tph:.0%} of the "
            f"~{tp.plateau_tph:.0f} t/h plateau) at {tp.fleet} trucks.{tail_gain} Once the bottleneck station is "
            "saturated, extra trucks only lengthen its queue: cycle times grow while deliveries do not."
        )
    if tp.status != "ok":
        lines.extend(tp.messages)
    return " ".join(lines)


@dataclass
class StudyAnalysis:
    summary: pd.DataFrame
    gains: pd.DataFrame
    bounds: pd.DataFrame
    tipping: TippingPoint
    target: dict
    economics: pd.DataFrame
    economic_choice: dict
    limiting: dict
    text: str

    def recommendation(self) -> dict:
        return {
            "label": "Estimates from a synthetic model - not measurements from a real mine.",
            "tipping_point": self.tipping.to_dict(),
            "production_target": self.target,
            "economics_illustrative": self.economic_choice,
            "limiting_resource_by_fleet": self.limiting,
            "interpretation": self.text,
        }


def analyse_study(df: pd.DataFrame, mine: MineConfig, acfg: AnalysisConfig) -> StudyAnalysis:
    """Run every analysis step on a fleet-study replication table."""
    summary = summarise_fleet(df)
    gains = marginal_gains(df)
    summary = summary.join(gains[["gain_mean", "gain_lo", "gain_hi"]])
    tp = find_tipping_point(df, acfg)
    target = fleet_for_target(summary, acfg.target_tph)
    eco, choice = economics(summary, acfg)
    limiting = {int(n): limiting_resource(row) for n, row in summary.iterrows()}
    return StudyAnalysis(
        summary=summary,
        gains=gains,
        bounds=bounds_table(mine, summary.index),
        tipping=tp,
        target=target,
        economics=eco,
        economic_choice=choice,
        limiting=limiting,
        text=interpretation(summary, gains, tp, mine),
    )
