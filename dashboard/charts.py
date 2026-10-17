"""Plotly figures for the dashboard. Pure functions: data in, figure out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from haulsim.metrics import queue_timeline
from haulsim.model import CRUSHER, SHOVEL
from haulsim.replay import CRUSHER_XY, SHOVEL_XY, STATUS_ORDER, delivered_between, road_path

# Validated categorical order (reference palette, light mode). Hues are
# assigned to entities in this fixed order and never cycled.
S1, S2, S3, S4, S5, S8 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#e34948"
INK, MUTED, GRID, NEUTRAL, SURFACE = "#1F2A33", "#5B6670", "#E4E0DA", "#A3A9AE", "#FFFFFF"
FONT = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def fleet_axis(fig: go.Figure, n_values: int) -> go.Figure:
    return fig.update_xaxes(dtick=1 if n_values <= 12 else 2)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def style(fig: go.Figure, title: str = "", x: str = "", y: str = "", height: int = 380, hover="x unified") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left"),
        font=dict(family=FONT, size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE,
        height=height,
        margin=dict(l=10, r=10, t=48 if title else 16, b=96),
        hovermode=hover,
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
        legend=dict(orientation="h", yanchor="top", y=-0.24, xanchor="left", x=0, font=dict(color=MUTED)),
    )
    axis = dict(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_xaxes(title=x, **axis)
    fig.update_yaxes(title=y, **axis)
    return fig


def _band(fig, xs, lo, hi, color, name):
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself",
            fillcolor=_rgba(color, 0.16),
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
            name=f"{name} 95% CI",
        )
    )


def ci_lines(summary: pd.DataFrame, series: list[tuple[str, str, str]], title: str, y: str, fmt: str = ".1f") -> go.Figure:
    """Mean with 95 % CI band for each (metric, label, colour)."""
    fig = go.Figure()
    xs = summary.index.to_numpy()
    for metric, label, color in series:
        lo, hi = summary[f"{metric}_lo"].to_numpy(), summary[f"{metric}_hi"].to_numpy()
        _band(fig, xs, lo, hi, color, label)
        fig.add_trace(
            go.Scatter(
                x=xs, y=summary[f"{metric}_mean"], name=label, mode="lines+markers",
                line=dict(color=color, width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                customdata=np.stack([lo, hi], axis=1),
                hovertemplate=f"%{{y:{fmt}}} (95% CI %{{customdata[0]:{fmt}}}-%{{customdata[1]:{fmt}}})<extra>{label}</extra>",
            )
        )
    style(fig, title, "Trucks in fleet", y)
    fleet_axis(fig, len(xs))
    if len(series) == 1:
        fig.update_layout(showlegend=False)
    return fig


def throughput(summary: pd.DataFrame, bounds: pd.DataFrame, tp, target: float) -> go.Figure:
    fig = ci_lines(summary, [("tph", "Simulated throughput", S1)], "", "Tonnes per scheduled hour", ".0f")
    fig.add_trace(
        go.Scatter(
            x=bounds.index, y=bounds["adj_bound_tph"], name="Capacity bound (availability-adjusted)",
            mode="lines", line=dict(color=NEUTRAL, width=2, dash="dot"),
            hovertemplate="%{y:.0f}<extra>Capacity bound</extra>",
        )
    )
    fig.add_hline(y=target, line=dict(color=S2, width=1.5, dash="dash"))
    fig.add_annotation(x=summary.index.max(), y=target, text=f"Target {target:.0f} t/h", showarrow=False,
                       xanchor="right", yanchor="top", font=dict(color=MUTED, size=12))
    if tp.fleet is not None:
        fig.add_vline(x=tp.fleet, line=dict(color=INK, width=1))
        fig.add_annotation(x=tp.fleet, y=tp.threshold_tph, text=f"Tipping point: {tp.fleet} trucks", showarrow=True,
                           arrowhead=0, ax=60, ay=60, font=dict(color=INK, size=12), bgcolor="white")
    fig.update_layout(showlegend=True)
    style(fig, "Throughput vs fleet size (mean and 95% CI)", "Trucks in fleet", "Tonnes per scheduled hour", 460)
    return fleet_axis(fig, len(summary))


def marginal(summary: pd.DataFrame) -> go.Figure:
    s = summary.dropna(subset=["gain_mean"])
    fig = go.Figure(
        go.Bar(
            x=s.index, y=s["gain_mean"], marker=dict(color=S1, line=dict(color=SURFACE, width=2)),
            error_y=dict(type="data", symmetric=False, array=s["gain_hi"] - s["gain_mean"],
                         arrayminus=s["gain_mean"] - s["gain_lo"], color=MUTED, thickness=1),
            hovertemplate="+%{y:.0f} t/h for truck %{x}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    style(fig, "Extra t/h from each added truck (95% CI)", "Trucks in fleet", "Extra t/h per truck", hover="closest")
    return fleet_axis(fig, len(s))


def stacked_states(summary: pd.DataFrame) -> go.Figure:
    parts = [
        ("truck_working_frac", "Working (load, haul, dump, return)", S1),
        ("truck_queue_shovel_frac", "Queuing at shovel", S2),
        ("truck_queue_crusher_frac", "Queuing at crusher", S3),
        ("truck_repair_frac", "Under repair", S4),
        ("truck_shift_stop_frac", "Shift stoppage", S5),
    ]
    fig = go.Figure()
    for col, label, color in parts:
        fig.add_trace(go.Bar(x=summary.index, y=summary[f"{col}_mean"] * 100, name=label,
                             marker=dict(color=color, line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{y:.1f}%<extra>" + label + "</extra>"))
    fig.update_layout(barmode="stack", bargap=0.25)
    style(fig, "Where truck time goes", "Trucks in fleet", "% of scheduled truck-time", 440)
    return fleet_axis(fig, len(summary))


def station_split(summary: pd.DataFrame) -> go.Figure:
    parts = [("productive", "Productive", S1), ("blocked", "Blocked by broken truck", S2),
             ("shift", "Shift stoppage", S3), ("idle", "Idle (starved)", NEUTRAL)]
    fig = go.Figure()
    for station, label_x in ((SHOVEL, "Shovel"), (CRUSHER, "Crusher")):
        for key, label, color in parts:
            fig.add_trace(go.Bar(
                x=summary.index,
                y=summary[f"{station}_{key}_frac_mean"] * 100, name=label, legendgroup=key,
                showlegend=station == SHOVEL, marker=dict(color=color, line=dict(color=SURFACE, width=2)),
                offsetgroup=station, xaxis="x" if station == SHOVEL else "x2", yaxis="y" if station == SHOVEL else "y2",
                hovertemplate=f"{label_x}: %{{y:.1f}}%<extra>{label}</extra>",
            ))
    fig.update_layout(
        barmode="stack",
        xaxis=dict(domain=[0, 0.48]), xaxis2=dict(domain=[0.52, 1], anchor="y2"),
        yaxis2=dict(anchor="x2", matches="y", showticklabels=False),
    )
    style(fig, "Station time split: shovel (left) and crusher (right)", "", "% of scheduled time", 420, hover="closest")
    fleet_axis(fig, len(summary))
    fig.update_layout(xaxis_title="Shovel - trucks in fleet", xaxis2_title="Crusher - trucks in fleet", yaxis2_title="")
    return fig


def queue_timelines(log, a: float, b: float) -> go.Figure:
    fig = go.Figure()
    for station, label, color in ((SHOVEL, "Shovel queue", S2), (CRUSHER, "Crusher queue", S3)):
        q = queue_timeline(log, station)
        q = pd.concat([q[q.t_min <= a].tail(1).assign(t_min=a), q[(q.t_min > a) & (q.t_min <= b)]])
        fig.add_trace(go.Scatter(x=(q.t_min - a) / 60, y=q.queue, name=label, mode="lines",
                                 line=dict(shape="hv", color=color, width=2),
                                 hovertemplate="%{y} waiting<extra>" + label + "</extra>"))
    for s, e in log.shift_windows:
        if e > a and s < b:
            fig.add_vrect(x0=(max(s, a) - a) / 60, x1=(min(e, b) - a) / 60, fillcolor=_rgba(S5, 0.25), line_width=0)
    for r in log.repairs:
        if r[3] and r[2] > a and r[1] < b:
            fig.add_vrect(x0=(max(r[1], a) - a) / 60, x1=(min(r[2], b) - a) / 60,
                          fillcolor=_rgba(S4, 0.22), line_width=0)
    style(fig, "Trucks waiting (excluding the one in service) - first 48 h shown, drag to pan",
          "Hours into measurement window", "Trucks waiting", 380)
    return fig.update_xaxes(range=[0, min(48.0, (b - a) / 60)])


def per_truck_states(states: pd.DataFrame, window_min: float) -> go.Figure:
    groups = [
        (["load", "haul", "dump", "return"], "Working", S1),
        (["queue_shovel"], "Queuing at shovel", S2),
        (["queue_crusher"], "Queuing at crusher", S3),
        (["repair"], "Under repair", S4),
        (["shift_stop"], "Shift stoppage", S5),
    ]
    fig = go.Figure()
    labels = [f"T{i + 1}" for i in states.index]
    for cols, label, color in groups:
        fig.add_trace(go.Bar(y=labels, x=states[cols].sum(axis=1) / window_min * 100, name=label, orientation="h",
                             marker=dict(color=color, line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{x:.1f}%<extra>" + label + "</extra>"))
    fig.update_layout(barmode="stack")
    style(fig, "Time split per truck (this run)", "% of measurement window", "", max(300, 150 + 30 * len(labels)),
          hover="closest")
    fig.update_layout(legend=dict(y=1.0, yanchor="bottom"), margin=dict(t=90, b=10))
    return fig.update_yaxes(autorange="reversed", dtick=1)


def scenarios(table: pd.DataFrame) -> go.Figure:
    colors = {"baseline": S1, "breakdowns only": S2, "shifts only": S3, "combined": S4}
    fig = go.Figure()
    for name, color in colors.items():
        g = table[table.scenario == name].sort_values("n_trucks")
        if g.empty:
            continue
        _band(fig, g.n_trucks.to_numpy(), g.tph_lo.to_numpy(), g.tph_hi.to_numpy(), color, name)
        fig.add_trace(go.Scatter(x=g.n_trucks, y=g.tph_mean, name=name.capitalize(), mode="lines+markers",
                                 line=dict(color=color, width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                                 hovertemplate="%{y:.0f} t/h<extra>" + name + "</extra>"))
    style(fig, "Throughput by scenario (mean and 95% CI)", "Trucks in fleet", "Tonnes per scheduled hour", 440)
    return fleet_axis(fig, table.n_trucks.nunique())


def cost_per_tonne(eco: pd.DataFrame, choice: dict) -> go.Figure:
    xs = eco.index.to_numpy()
    fig = go.Figure()
    _band(fig, xs, eco.cost_per_t_lo.to_numpy(), eco.cost_per_t_hi.to_numpy(), S1, "cost")
    meets = eco.meets_target_confident
    fig.add_trace(go.Scatter(x=xs, y=eco.cost_per_t, mode="lines", line=dict(color=S1, width=2), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs[meets], y=eco.cost_per_t[meets], mode="markers", name="Meets target (95% conf.)",
                             marker=dict(size=9, color=S1, line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{y:.2f} per t<extra>meets target</extra>"))
    fig.add_trace(go.Scatter(x=xs[~meets], y=eco.cost_per_t[~meets], mode="markers", name="Misses target",
                             marker=dict(size=9, color=SURFACE, symbol="circle-open", line=dict(color=S1, width=2)),
                             hovertemplate="%{y:.2f} per t<extra>misses target</extra>"))
    n = choice.get("least_cost_meeting_target")
    if n is not None:
        fig.add_annotation(x=n, y=eco.loc[n, "cost_per_t"], text=f"Least cost meeting target: {n} trucks",
                           showarrow=True, arrowhead=0, ax=40, ay=-50, bgcolor="white", font=dict(color=INK, size=12))
    style(fig, "Illustrative cost per tonne (95% CI)", "Trucks in fleet", "Cost per tonne (illustrative units)", 400,
          hover="closest")
    return fleet_axis(fig, len(xs))


def warmup(profile: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=profile.hour, y=profile.mean_t, mode="lines", line=dict(color=S1, width=2),
                               hovertemplate="hour %{x}: %{y:.0f} t<extra></extra>"))
    style(fig, "Warm-up check: hourly tonnes from a cold start (mean of replications)", "Hour since start",
          "Tonnes delivered in hour", 340)
    return fig.update_layout(showlegend=False)


# ---------------------------------------------------------------------- #
# Replay
# ---------------------------------------------------------------------- #
STATUS_STYLE = {
    "Hauling (loaded)": (S1, "triangle-right"),
    "Returning (empty)": (S3, "triangle-left"),
    "Loading / dumping": (S2, "square"),
    "Queued": (S4, "circle"),
    "Under repair": (S8, "x"),
    "Shift stoppage": (NEUTRAL, "diamond"),
}


OFFSCREEN = -1000.0


def replay(frames: pd.DataFrame, log, a: float, frame_ms: int) -> go.Figure:
    """Animated plan view. Each frame is one simulation instant from the logs."""
    times = np.sort(frames.t.unique())
    cum = delivered_between(log, a, times)
    hx, hy = road_path("haul")
    rx, ry = road_path("return")

    static = [
        go.Scatter(x=hx, y=hy, mode="lines", line=dict(color=GRID, width=10), hoverinfo="skip", showlegend=False),
        go.Scatter(x=rx, y=ry, mode="lines", line=dict(color=GRID, width=10), hoverinfo="skip", showlegend=False),
        go.Scatter(x=[SHOVEL_XY[0], CRUSHER_XY[0]], y=[SHOVEL_XY[1], CRUSHER_XY[1]], mode="markers+text",
                   marker=dict(size=34, color=INK, symbol="square"), text=["SHOVEL", "CRUSHER"],
                   textposition="bottom center", textfont=dict(color=INK, size=12), hoverinfo="skip", showlegend=False),
    ]

    def status_traces(t):
        f = frames[frames.t == t]
        out = []
        for status in STATUS_ORDER:
            g = f[f.status == status]
            color, symbol = STATUS_STYLE[status]
            # An off-screen anchor point keeps every status in the legend even
            # when no truck is in that state in the current frame.
            xs, ys = np.append(OFFSCREEN, g.x), np.append(OFFSCREEN, g.y)
            cd = np.vstack([[0, 0], np.stack([g.truck + 1, g.on_board_t], axis=1)]) if len(g) else [[0, 0]]
            out.append(go.Scatter(
                x=xs, y=ys, mode="markers+text", name=status, text=[""] + [f"{i + 1}" for i in g.truck],
                textposition="top center", textfont=dict(size=10, color=MUTED),
                marker=dict(size=15, color=color, symbol=symbol, line=dict(color=SURFACE, width=2)),
                customdata=cd,
                hovertemplate="Truck %{customdata[0]}<br>" + status + "<br>%{customdata[1]:.0f} t on board<extra></extra>",
            ))
        return out

    def caption(i, t):
        return f"t = {(t - a) / 60:5.2f} h into replay   |   delivered so far: {cum[i]:,.0f} t"

    xmax = max(15.0, frames.x.max() + 0.6)
    xmin = min(-5.0, frames.x.min() - 0.6)  # frames never contain the off-screen anchor

    def counter(i, t):
        # Clock and delivery counter are a text trace so they animate with the data.
        return go.Scatter(x=[xmin + 0.2], y=[3.4], mode="text", text=[caption(i, t)], textposition="middle right",
                          textfont=dict(family="Consolas, monospace", size=13, color=INK), hoverinfo="skip",
                          showlegend=False)

    n_static = len(static)
    label_every = max(1, len(times) // 8)
    animated = list(range(n_static, n_static + len(STATUS_ORDER) + 1))
    fig = go.Figure(data=static + status_traces(times[0]) + [counter(0, times[0])])
    fig.frames = [
        go.Frame(data=status_traces(t) + [counter(i, t)], name=f"{t:.2f}", traces=animated)
        for i, t in enumerate(times)
    ]
    play = dict(frame=dict(duration=frame_ms, redraw=False), transition=dict(duration=0), fromcurrent=True, mode="immediate")
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", x=0, y=0, xanchor="left", yanchor="top", pad=dict(t=8, r=10),
                          buttons=[dict(label="Play", method="animate", args=[None, play]),
                                   dict(label="Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.2, len=0.8, y=0, yanchor="top", pad=dict(t=8), currentvalue=dict(visible=False),
                      ticklen=0, font=dict(color=MUTED, size=11),
                      steps=[dict(method="animate", label=f"{(t - a) / 60:.1f}h" if i % label_every == 0 else "",
                                  args=[[f"{t:.2f}"], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                             for i, t in enumerate(times)])],
    )
    fig.update_xaxes(range=[xmin, xmax], visible=False)
    fig.update_yaxes(range=[-3.0, 3.8], visible=False, scaleanchor="x", scaleratio=1)
    style(fig, "", "", "", 520, hover="closest")
    fig.update_xaxes(visible=False, showgrid=False)
    fig.update_yaxes(visible=False, showgrid=False)
    fig.update_layout(margin=dict(t=10, b=120),
                      legend=dict(orientation="h", y=-0.2, x=0, xanchor="left", yanchor="top"))
    return fig
