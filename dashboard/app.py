"""Streamlit dashboard for the open-pit truck haulage simulation.

Run from the project root:  streamlit run dashboard/app.py

The dashboard only calls the ``haulsim`` package; every number shown is
computed by the same engine and analysis code used by the command line.
"""

from __future__ import annotations

import io
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard import charts  # noqa: E402
from haulsim import AnalysisConfig, ExperimentConfig, MineConfig, Triangular, simulate  # noqa: E402
from haulsim.analysis import analyse_study, capacity_bounds, mean_ci  # noqa: E402
from haulsim.experiment import (  # noqa: E402
    default_workers,
    replication_seed,
    run_fleet_study,
    run_scenarios,
    warmup_profile,
)
from haulsim.metrics import summarise, truck_state_minutes  # noqa: E402
from haulsim.replay import build_frames  # noqa: E402

SAVED_STUDY = ROOT / "results" / "study"
SAVED_SCENARIOS = ROOT / "results" / "scenarios"
LABEL = "Synthetic demonstration model: all values are model estimates, not measurements from a real mine."

CSS = """
    <style>
      .block-container {padding-top: 2rem; max-width: 1400px;}
      h1, h2, h3 {letter-spacing: -0.01em; color: #1F2A33;}
      .cards {display: grid; gap: 12px; margin-bottom: 8px;}
      .cards.kpis {grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));}
      .cards.answers {grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));}
      .kpi {background: #FFFFFF; border: 1px solid #E4E0DA; border-left: 4px solid #A8572A;
            border-radius: 6px; padding: 14px 16px; height: 100%;}
      .kpi .label {font-size: 0.74rem; text-transform: uppercase; letter-spacing: .06em; color: #5B6670;}
      .kpi .value {font-size: 1.6rem; white-space: nowrap; font-weight: 650; color: #1F2A33; line-height: 1.25; margin-top: 2px;}
      .kpi .sub {font-size: 0.82rem; color: #5B6670; margin-top: 2px;}
      .banner {background: #EFECE7; border-radius: 6px; padding: 8px 14px; color: #5B6670; font-size: .85rem;}
      .answer {background: #FFFFFF; border: 1px solid #E4E0DA; border-radius: 6px; padding: 14px 16px; height: 100%;}
      .answer h4 {margin: 0 0 4px 0; font-size: .95rem; color: #A8572A;}
      .answer p {margin: 0; font-size: .9rem; color: #1F2A33;}
    </style>
"""


class Cards:
    """Collects card HTML and renders it as one responsive CSS grid."""

    def __init__(self, kind: str):
        self.kind, self.items = kind, []

    def render(self) -> None:
        st.markdown(f'<div class="cards {self.kind}">{"".join(self.items)}</div>', unsafe_allow_html=True)


def kpi(cards: Cards, label: str, value: str, sub: str = "") -> None:
    cards.items.append(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                       f'<div class="sub">{sub}</div></div>')


def answer(cards: Cards, title: str, body: str) -> None:
    cards.items.append(f'<div class="answer"><h4>{title}</h4><p>{body}</p></div>')


# ---------------------------------------------------------------------- #
# Cached computations (keyed by plain dicts so the cache is reliable)
# ---------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=8)
def cached_study(mine_d: dict, exp_d: dict) -> pd.DataFrame:
    exp = ExperimentConfig(**exp_d)
    try:
        return run_fleet_study(MineConfig.from_dict(mine_d), exp)
    except Exception:  # e.g. process pool unavailable: fall back to one process
        return run_fleet_study(MineConfig.from_dict(mine_d), replace(exp, workers=1))


@st.cache_data(show_spinner=False, max_entries=4)
def cached_scenarios(mine_d: dict, exp_d: dict) -> pd.DataFrame:
    exp = ExperimentConfig(**exp_d)
    try:
        return run_scenarios(MineConfig.from_dict(mine_d), exp)
    except Exception:
        return run_scenarios(MineConfig.from_dict(mine_d), replace(exp, workers=1))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_run(mine_d: dict, horizon: float, seed: int):
    return simulate(MineConfig.from_dict(mine_d), horizon, seed)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_frames(mine_d: dict, horizon: float, seed: int, t0: float, t1: float, step: float) -> pd.DataFrame:
    return build_frames(cached_run(mine_d, horizon, seed), t0, t1, step)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_warmup(mine_d: dict, n: int, reps: int, seed: int) -> pd.DataFrame:
    return warmup_profile(MineConfig.from_dict(mine_d), n, 72, reps, seed)


def load_saved_study():
    params = json.loads((SAVED_STUDY / "parameters.json").read_text())
    df = pd.read_csv(SAVED_STUDY / "replications.csv")
    return MineConfig.from_dict(params["mine"]), ExperimentConfig(**params["experiment"]), df


# ---------------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------------- #
def tri_input(container, label: str, default: Triangular, key: str, step: float = 0.5) -> Triangular:
    container.caption(f"{label} - minimum / most likely / maximum (min)")
    c1, c2, c3 = container.columns(3)
    lo = c1.number_input("min", value=float(default.low), step=step, key=f"{key}_lo", label_visibility="collapsed")
    mo = c2.number_input("mode", value=float(default.mode), step=step, key=f"{key}_mo", label_visibility="collapsed")
    hi = c3.number_input("max", value=float(default.high), step=step, key=f"{key}_hi", label_visibility="collapsed")
    return Triangular(lo, mo, hi)



def main() -> None:
    st.set_page_config(page_title="Pit Haulage Simulator", page_icon="⛏️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    d = MineConfig()
    sb = st.sidebar
    sb.markdown("## ⛏️ Haulage study")
    source = sb.radio(
        "Results source",
        ["Run simulation here", "Load saved CLI study"],
        disabled=not (SAVED_STUDY / "replications.csv").exists(),
        help="Saved study = output of `python -m haulsim study` in results/study.",
    )

    with sb.expander("Experiment", expanded=True):
        mode = st.segmented_control("Mode", ["Quick", "Study"], default="Quick",
                                    help="Quick: 5 replications x 3 days. Study: 30 replications x 14 days.")
        fleet_lo, fleet_hi = st.slider("Fleet sizes to compare", 1, 25, (1, 15 if mode == "Quick" else 20))
        reps = st.number_input("Replications per fleet size", 2, 200, 5 if mode == "Quick" else 30)
        measure_days = st.number_input("Measurement period (whole days)", 1, 60, 3 if mode == "Quick" else 14)
        warmup_h = st.number_input("Warm-up (hours)", 0.0, 240.0, 12.0 if mode == "Quick" else 24.0, step=6.0)
        seed = st.number_input("Base seed", 0, 2**31 - 1, 20261017)
        workers = st.number_input("Parallel processes", 1, 32, default_workers())

    with sb.expander("Fleet & equipment"):
        payload = st.number_input("Truck payload (t)", 1.0, 500.0, d.payload_t, step=10.0)
        n_shovels = st.number_input("Shovels", 1, 5, d.n_shovels)
        n_crushers = st.number_input("Crusher receiving stations", 1, 5, d.n_crushers)
        randomness = st.toggle("Random activity times", True, help="Off: every duration equals its mean (deterministic).")

    with sb.expander("Activity times"):
        load_t = tri_input(st, "Loading", d.load_min, "load")
        haul_t = tri_input(st, "Loaded travel", d.haul_min, "haul")
        dump_t = tri_input(st, "Crusher service", d.dump_min, "dump")
        ret_t = tri_input(st, "Empty return", d.return_min, "ret")

    with sb.expander("Breakdowns"):
        breakdowns = st.toggle("Truck breakdowns", True)
        mtbf = st.number_input("Mean operating hours between failures", 0.5, 1000.0, d.mtbf_operating_h, step=5.0)
        repair_t = tri_input(st, "Repair duration", d.repair_min, "rep", step=5.0)

    with sb.expander("Shift changes"):
        shifts = st.toggle("Shift-change stoppages", True)
        shift_h = st.number_input("Hours between shift changes", 1.0, 48.0, d.shift_interval_h)
        shift_stop = st.number_input("Stoppage length (min)", 0.0, 240.0, d.shift_stop_min, step=5.0)

    with sb.expander("Decision rules & economics"):
        plateau_frac = st.slider("Tipping point: % of plateau", 80, 100, 95) / 100
        plateau_tail = st.number_input("Plateau estimated from largest N fleets", 2, 6, 3)
        plateau_tol = st.slider("Max rise across those fleets to call it flat (%)", 0.5, 10.0, 2.0, step=0.5) / 100
        target = st.number_input("Production target (t per scheduled hour)", 0.0, 10000.0, 800.0, step=50.0)
        st.caption("Illustrative costs per scheduled hour - replace with your own figures.")
        truck_cost = st.number_input("Cost per truck per hour", 0.0, 10000.0, 300.0, step=25.0)
        fixed_cost = st.number_input("Fixed cost per hour (shovel, crusher, overheads)", 0.0, 100000.0, 2500.0, step=100.0)
        use_value = st.toggle("Include value per tonne (profit view)", False)
        value_t = st.number_input("Value per tonne", 0.0, 1000.0, 10.0) if use_value else None

    try:
        mine_ui = MineConfig(
            payload_t=payload, n_shovels=int(n_shovels), n_crushers=int(n_crushers),
            load_min=load_t, haul_min=haul_t, dump_min=dump_t, return_min=ret_t,
            mtbf_operating_h=mtbf, repair_min=repair_t, shift_interval_h=shift_h, shift_stop_min=shift_stop,
            breakdowns_enabled=breakdowns, shifts_enabled=shifts, randomness_enabled=randomness,
        )
        exp_ui = ExperimentConfig(
            fleet_sizes=tuple(range(fleet_lo, fleet_hi + 1)), replications=int(reps), warmup_h=float(warmup_h),
            measure_days=int(measure_days), base_seed=int(seed), workers=int(workers),
        )
        acfg = AnalysisConfig(
            plateau_fraction=plateau_frac, plateau_tail=int(plateau_tail), plateau_tolerance=plateau_tol,
            target_tph=target, truck_cost_per_h=truck_cost, fixed_cost_per_h=fixed_cost, value_per_t=value_t,
        )
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        st.stop()

    run_clicked = sb.button("Run fleet study", type="primary", width="stretch", disabled=source != "Run simulation here")

    # ---------------------------------------------------------------------- #
    # Obtain results
    # ---------------------------------------------------------------------- #
    if source == "Load saved CLI study":
        mine, exp, df = load_saved_study()
        sb.info("Showing the saved CLI study. Model inputs above are ignored; decision rules and costs still apply.")
    else:
        key = (json.dumps(mine_ui.to_dict()), json.dumps(exp_ui.to_dict()))
        if run_clicked or "study_key" not in st.session_state:
            n_runs = len(exp_ui.fleet_sizes) * exp_ui.replications
            with st.spinner(f"Simulating {n_runs} runs ({exp_ui.measure_days} d + {exp_ui.warmup_h:g} h warm-up each)..."):
                st.session_state.study_df = cached_study(mine_ui.to_dict(), exp_ui.to_dict())
            st.session_state.study_key = key
            st.session_state.study_cfg = (mine_ui, exp_ui)
        mine, exp = st.session_state.study_cfg
        df = st.session_state.study_df
        if st.session_state.study_key != key:
            sb.warning("Inputs changed since the last run. Press **Run fleet study** to update the results.")

    res = analyse_study(df, mine, acfg)
    summary, tp = res.summary, res.tipping

    # ---------------------------------------------------------------------- #
    # Header and KPIs
    # ---------------------------------------------------------------------- #
    st.title("Open-pit truck haulage: how many trucks?")
    st.markdown(
        f'<div class="banner">{LABEL} {exp.replications} independent replications per fleet size, '
        f"{exp.measure_days}-day measurement after a {exp.warmup_h:g} h warm-up; intervals are 95% CIs across replications."
        "</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    ref_n = tp.fleet if tp.fleet is not None else int(summary.index.max())
    ref = summary.loc[ref_n]
    k = Cards("kpis")
    kpi(k, "Production plateau", f"{tp.plateau_tph:,.0f} t/h",
        f"95% CI {tp.plateau_lo:,.0f}-{tp.plateau_hi:,.0f}" if tp.status != "insufficient_range" else "not estimated")
    status_txt = {"ok": "confirmed", "uncertain": "uncertain - see notes", "plateau_not_reached": "plateau not reached",
                  "insufficient_range": "fleet range too small"}[tp.status]
    kpi(k, "Operational tipping point", f"{tp.fleet} trucks" if tp.fleet else "n/a", status_txt)
    kpi(k, f"Throughput at {ref_n} trucks", f"{ref.tph_mean:,.0f} t/h",
        f"{ref.tph_mean / tp.plateau_tph:.0%} of plateau" if tp.plateau_tph == tp.plateau_tph else "")
    kpi(k, f"Crusher wait at {ref_n} trucks", f"{ref.wait_crusher_mean_min_mean:.1f} min",
        f"shovel wait {ref.wait_shovel_mean_min_mean:.1f} min per visit")
    kpi(k, f"Utilisation at {ref_n} trucks", f"{ref.crusher_util_available_mean:.0%} crusher",
        f"shovel {ref.shovel_util_available_mean:.0%} (of available time)")
    k.render()
    st.write("")

    tabs = st.tabs(["Recommendation", "Fleet analysis", "Single run & replay", "Scenarios", "Economics",
                    "Verification", "Data & exports", "Model notes"])

    # ---------------------------------------------------------------------- #
    with tabs[0]:
        a = Cards("answers")
        if tp.fleet is not None:
            rng = f" (plausible range {tp.fleet_range[0]}-{tp.fleet_range[1]})" if tp.fleet_range[0] != tp.fleet_range[1] else ""
            body = (f"<b>{tp.fleet} trucks</b>{rng}. Smallest fleet reaching {acfg.plateau_fraction:.0%} of the "
                    f"~{tp.plateau_tph:,.0f} t/h plateau. Status: {status_txt}.")
        else:
            body = "<b>No recommendation.</b> " + " ".join(tp.messages)
        answer(a, "1 - Operational tipping point", body)
        t = res.target
        if t["fleet_mean"] is None:
            body = t["message"]
        else:
            body = (f"<b>{t['fleet_confident'] or '-'} trucks</b> meet {t['target_tph']:,.0f} t/h with 95% confidence "
                    f"({t['fleet_mean']} trucks on average).")
            if t["message"] != "ok":
                body += " " + t["message"]
        answer(a, "2 - Fleet for the production target", body)
        c = res.economic_choice
        body = (f"<b>{c['least_cost_meeting_target']} trucks</b> give the lowest cost per tonne among fleets that meet the "
                f"target (lowest overall: {c['least_cost_overall']}). Depends entirely on the illustrative costs."
                if c["least_cost_meeting_target"] is not None else "No tested fleet meets the target with confidence.")
        if "max_profit" in c:
            body += f" Profit-maximising fleet: {c['max_profit']}."
        answer(a, "3 - Economically preferred fleet", body)
        a.render()
        st.write("")
        st.plotly_chart(charts.throughput(summary, res.bounds, tp, acfg.target_tph), width="stretch")
        st.markdown("#### Why production plateaus")
        st.write(res.text)
        for m in tp.messages:
            st.warning(m)
        st.caption(f"Rule: {tp.rule}. Plateau flatness tolerance {acfg.plateau_tolerance:.0%} across fleets "
                   f"{', '.join(map(str, tp.tail_fleets))}. Grey dotted line: availability-adjusted capacity bound "
                   "min(fleet, shovel, crusher) - realised output is lower because of variability, blocking and stoppages.")

    # ---------------------------------------------------------------------- #
    with tabs[1]:
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.marginal(summary), width="stretch")
        c2.plotly_chart(charts.ci_lines(summary, [("wait_shovel_mean_min", "Shovel", charts.S2),
                                                  ("wait_crusher_mean_min", "Crusher", charts.S3)],
                                        "Mean wait per visit (min)", "Minutes"), width="stretch")
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.ci_lines(summary, [("queue_shovel_mean", "Shovel", charts.S2),
                                                  ("queue_crusher_mean", "Crusher", charts.S3)],
                                        "Time-weighted average queue length", "Trucks waiting", ".2f"), width="stretch")
        c2.plotly_chart(charts.ci_lines(summary, [("shovel_util_available", "Shovel", charts.S2),
                                                  ("crusher_util_available", "Crusher", charts.S3)],
                                        "Productive utilisation of available time", "Fraction", ".3f"), width="stretch")
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.stacked_states(summary), width="stretch")
        c2.plotly_chart(charts.ci_lines(summary, [("cycle_mean_min", "Mean", charts.S1),
                                                  ("cycle_p90_min", "90th percentile", charts.S2)],
                                        "Truck cycle time", "Minutes"), width="stretch")
        st.plotly_chart(charts.station_split(summary), width="stretch")
        st.caption(
            "Denominators: truck fractions use scheduled truck-time (trucks x window). Station fractions use scheduled "
            "server-time. 'Available time' excludes shift stoppages and time blocked by a broken truck. Blocked = station "
            "occupied by a truck under repair; productive = a truck actually being loaded or dumped."
        )

    # ---------------------------------------------------------------------- #
    with tabs[2]:
        st.markdown("One replication from the study, re-run from its recorded seed with full event logs.")
        c = st.columns(4)
        fleets = list(summary.index)
        n_run = c[0].selectbox("Fleet size", fleets, index=fleets.index(ref_n))
        rep = c[1].number_input("Replication", 0, exp.replications - 1, 0)
        run_seed = replication_seed(exp.base_seed, int(rep))
        cfg_run = mine.with_trucks(int(n_run))
        log = cached_run(cfg_run.to_dict(), exp.horizon_min, run_seed)
        s = summarise(log, exp.warmup_min, exp.horizon_min)
        row = df[(df.n_trucks == n_run) & (df.replication == rep)]
        match = "matches" if len(row) and abs(row.tph.iloc[0] - s["tph"]) < 1e-9 else "does NOT match"
        c[2].metric("Throughput (this run)", f"{s['tph']:,.0f} t/h")
        c[3].metric("Deliveries", f"{s['deliveries']:,}")
        st.caption(f"Seed {run_seed}. This run's throughput {match} the study table row for {n_run} trucks, "
                   f"replication {rep}.")

        st.plotly_chart(charts.queue_timelines(log, exp.warmup_min, exp.horizon_min), width="stretch")
        st.caption("Pink bands: shift stoppages. Yellow bands: a station blocked by a truck under repair.")
        st.plotly_chart(charts.per_truck_states(truck_state_minutes(log, exp.warmup_min, exp.horizon_min),
                                                exp.measure_min), width="stretch")

        st.markdown("#### 2D replay (driven by the event log)")
        c = st.columns(4)
        start_h = c[0].number_input("Start (hours into measurement)", 0.0, exp.measure_days * 24.0 - 1, 11.0, step=1.0)
        dur_h = c[1].number_input("Duration (simulated hours)", 0.5, 24.0, 4.0, step=0.5)
        step_min = c[2].number_input("Simulated minutes per frame", 0.25, 10.0, 1.0, step=0.25)
        frame_ms = c[3].number_input("Playback: ms per frame", 10, 1000, 80, step=10)
        t0 = exp.warmup_min + start_h * 60
        t1 = min(exp.horizon_min, t0 + dur_h * 60)
        if (t1 - t0) / step_min > 1500:
            st.warning("Too many frames; increase minutes per frame or shorten the duration.")
        else:
            frames = cached_frames(cfg_run.to_dict(), exp.horizon_min, run_seed, t0, t1, float(step_min))
            st.plotly_chart(charts.replay(frames, log, t0, int(frame_ms)), width="stretch")
            delivered = summarise(log, t0, t1)["delivered_t"]
            st.caption(
                f"Simulation time and playback speed are independent: each frame is {step_min:g} simulated minutes shown "
                f"for {frame_ms} ms. Tonnes delivered in this replay window from the metrics module: {delivered:,.0f} t "
                "(the on-chart counter reads the same delivery log)."
            )

    # ---------------------------------------------------------------------- #
    with tabs[3]:
        st.markdown("Same fleet sizes, replications and seeds with breakdowns and shift stoppages switched on and off.")
        c1, c2 = st.columns(2)
        run_sc = c1.button("Run scenario comparison", disabled=source != "Run simulation here", width="stretch")
        load_sc = c2.button("Load saved CLI scenarios", width="stretch",
                            disabled=not (SAVED_SCENARIOS / "scenario_replications.csv").exists())
        if run_sc:
            with st.spinner("Running four scenarios..."):
                st.session_state.scen = cached_scenarios(mine.to_dict(), exp.to_dict())
        if load_sc:
            st.session_state.scen = pd.read_csv(SAVED_SCENARIOS / "scenario_replications.csv")
        if "scen" in st.session_state:
            sc = st.session_state.scen
            rows = []
            for (name, n), g in sc.groupby(["scenario", "n_trucks"]):
                m, lo, hi = mean_ci(g.tph)
                rows.append({"scenario": name, "n_trucks": n, "tph_mean": m, "tph_lo": lo, "tph_hi": hi})
            table = pd.DataFrame(rows)
            st.plotly_chart(charts.scenarios(table), width="stretch")
            wide = table.pivot(index="n_trucks", columns="scenario", values="tph_mean").round(0)
            st.dataframe(wide, width="stretch")
            st.caption("Breakdowns lower the plateau through lost truck time and crusher blocking; shift stoppages remove "
                       f"{mine.shift_stop_min:g} min of every {mine.shift_interval_h:g} h from every resource at once.")
        else:
            st.info("Run the comparison (or load the saved CLI results) to populate this tab.")

    # ---------------------------------------------------------------------- #
    with tabs[4]:
        st.markdown(f"**Illustrative economics.** Cost per hour = trucks x {acfg.truck_cost_per_h:,.0f} + "
                    f"{acfg.fixed_cost_per_h:,.0f} (fixed). Throughput and cost both use scheduled hours, so cost per tonne "
                    "= cost per hour / tonnes per hour. These are placeholders, not real prices.")
        st.plotly_chart(charts.cost_per_tonne(res.economics, res.economic_choice), width="stretch")
        st.dataframe(res.economics.round(3), width="stretch")

    # ---------------------------------------------------------------------- #
    with tabs[5]:
        st.markdown("#### Idealised capacity bounds")
        b = capacity_bounds(mine, int(summary.index.max()))
        st.markdown(
            f"- Fleet: N x {mine.payload_t:g} t x 60 / {mine.mean_cycle_work_min:.1f} min = "
            f"**{capacity_bounds(mine, 1)['fleet_tph']:.1f} t/h per truck**\n"
            f"- Shovel: {mine.n_shovels} x {mine.payload_t:g} x 60 / {mine.load_min.mean:.2f} = **{b['shovel_tph']:.0f} t/h**\n"
            f"- Crusher: {mine.n_crushers} x {mine.payload_t:g} x 60 / {mine.dump_min.mean:.2f} = **{b['crusher_tph']:.0f} t/h**\n"
            f"- Shift stoppages remove {b['shift_frac']:.1%} of scheduled time; trucks can work at most "
            f"{b['truck_working_frac_max']:.1%} of the time once repairs are included."
        )
        comp = pd.DataFrame({
            "simulated_tph": summary.tph_mean, "ci_hi": summary.tph_hi,
            "ideal_bound": res.bounds.ideal_bound_tph, "adjusted_bound": res.bounds.adj_bound_tph,
            "limiting (bounds)": res.bounds.limiting_resource, "limiting (simulated)": pd.Series(res.limiting),
        })
        comp["within_bound"] = comp.simulated_tph <= comp.adjusted_bound * 1.005
        comp.index.name = "trucks"
        st.dataframe(comp.round(1), width="stretch")
        st.markdown("#### Warm-up check")
        n_w = st.selectbox("Fleet size for warm-up profile", fleets, index=fleets.index(ref_n))
        st.plotly_chart(charts.warmup(cached_warmup(mine.to_dict(), int(n_w), min(exp.replications, 30), exp.base_seed)),
                        width="stretch")
        st.caption("Trucks start bunched at the shovel at t = 0, so the first hour delivers less. Dips every 12 h are "
                   "shift stoppages. The measurement window starts after the warm-up without resetting the mine.")
        st.markdown("Automated checks live in `tests/` (run `python -m pytest`). They verify the model against hand "
                    "calculations and invariants; they do not validate it against a real mine.")

    # ---------------------------------------------------------------------- #
    with tabs[6]:
        params = {"label": LABEL, "mine": mine.to_dict(), "experiment": exp.to_dict(), "analysis": acfg.to_dict(),
                  "replication_seeds": {r: replication_seed(exp.base_seed, r) for r in range(exp.replications)}}
        flat = pd.json_normalize({"mine": mine.to_dict(), "experiment": exp.to_dict(), "analysis": acfg.to_dict()})
        flat = flat.T.reset_index().rename(columns={"index": "parameter", 0: "value"})
        flat["value"] = flat["value"].astype(str)
        c = st.columns(5)

        def csv_bytes(frame: pd.DataFrame, index: bool = True) -> bytes:
            buf = io.StringIO()
            frame.to_csv(buf, index=index)
            return buf.getvalue().encode()

        c[0].download_button("Parameters (CSV)", csv_bytes(flat, False), "parameters.csv", "text/csv")
        c[1].download_button("Parameters + seeds (JSON)", json.dumps(params, indent=2, default=str), "parameters.json")
        c[2].download_button("Replications (CSV)", csv_bytes(df, False), "replications.csv", "text/csv")
        c[3].download_button("Fleet summary (CSV)", csv_bytes(summary), "summary.csv", "text/csv")
        c[4].download_button("Recommendation (JSON)", json.dumps(res.recommendation(), indent=2, default=str),
                             "recommendation.json")
        st.markdown("#### Fleet summary")
        st.dataframe(summary.round(3), width="stretch")
        st.markdown("#### Replication results")
        st.dataframe(df, width="stretch", height=320)
        st.markdown("#### Parameters")
        st.dataframe(flat, width="stretch", hide_index=True)

    # ---------------------------------------------------------------------- #
    with tabs[7]:
        st.markdown((ROOT / "docs" / "MODEL.md").read_text(encoding="utf-8") if (ROOT / "docs" / "MODEL.md").exists()
                    else "See README.md.")


# Streamlit runs this file as __main__. Worker processes started by the
# process pool on Windows import it as __mp_main__ and must not draw the UI.
if __name__ == "__main__":
    main()
