"""
eval/run_report/plot.py
──────────────────────────────────────────────────────────────────────────────
Drawing for the run report: the template look, the declarative `Panel`, and
the renderers for the scenario figures, the training grid and the cover page.

The look is lifted from `plot_metric_evolution` in the former
eval/generate_training_visualizations.py: seaborn-v0_8-whitegrid, figsize
(10, 5), dpi 300, the raw series faint behind a centred rolling mean, bands at
alpha 0.15, "n_samples = N" in grey in the lower right.

What is plotted lives in specs.py; this module only knows how.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .loader import RunData, Segment, update_grid


STYLE        = "seaborn-v0_8-whitegrid"
FIGSIZE      = (10, 5)
FIGSIZE_TALL = (10, 6.5)          # two stacked panels keep the main one template-sized
GRID_FIGSIZE = (20, 15)
COVER_SIZE   = (11.69, 8.27)      # A4 landscape
DPI          = 300
WINDOW       = 10

RAW_ALPHA, RAW_LW = 0.4, 1.0
ROLL_LW           = 2.2
BAND_ALPHA        = 0.15
BAR_ALPHA         = 0.5

# Fixed order, never cycled; the template's blue first. Stacked bars use an
# order whose neighbours stay colour-blind-distinguishable at alpha 0.5, and a
# lone bar series stays neutral so it never competes with the lines.
LINE_COLORS = ("#4878CF", "#eb6834", "#1baf7a")
BAR_COLORS  = ("#eb6834", "#4a3aa7", "#1baf7a")
BAR_SINGLE  = "#b9b8b3"
MUTED       = "#898781"
RATE_YLIM   = (-0.02, 1.02)       # a flat-zero rate stays visible above the spine

_CONFIG_NAMES = {
    "envs": "envs", "steps": "steps", "batch": "batch", "minibatch": "mb",
    "epochs": "epochs", "lr": "lr", "beta": "beta", "ent_coef": "ent", "refresh": "refresh",
}


# ══════════════════════════════════════════════════════════════════════════════
# Declarative panels
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Panel:
    """One axes worth of plot, described by the columns it shows.

    lines   column -> legend label. An empty label on a single line gives the
            template legend ("Raw" / "Rolling mean (w=10)").
    std     ±1 std band around the first line (template style).
    minmax  (min, max) band around the first line.
    bars    stacked bars in the background with their own count axis on the
            right; bars_cap scales it: a column name means len(bars) × that
            column's max (n_samples -> 3 × 20 = 60), a number is used as is,
            None means the data maximum.
    weights pooled rolling mean Σ(x·w)/Σw, for per-game ratios.
    """
    lines:          dict[str, str] = field(default_factory=dict)
    title:          str = ""
    ylabel:         str = ""
    ylim:           tuple[float, float] | None = None
    std:            str | None = None
    minmax:         tuple[str, str] | None = None
    bars:           dict[str, str] = field(default_factory=dict)
    bars_cap:       str | float | None = None
    bars_label:     str = ""
    weights:        str | None = None
    scale:          float = 1.0
    logy:           bool = False
    hlines:         tuple[tuple[float, str], ...] = ()
    hband:          tuple[float, float, str] | None = None
    sparse_markers: bool = False
    refresh_ticks:  bool = False
    ratio:          float = 1.0
    custom:         Callable | None = None

    def columns(self) -> list[str]:
        cols = [*self.lines, *self.bars]
        if self.std:
            cols.append(self.std)
        if self.minmax:
            cols.extend(self.minmax)
        if isinstance(self.bars_cap, str):
            cols.append(self.bars_cap)
        if self.weights:
            cols.append(self.weights)
        return cols


@dataclass(frozen=True)
class FigureSpec:
    title:  str
    panels: tuple[Panel, ...]


@dataclass
class Ctx:
    run:      RunData
    window:   int = WINDOW
    regimes:  list[int] | None = None      # updates that start a new rolling window
    template: bool = False                 # template legend for single-line panels
    small:    bool = False                 # grid fonts instead of figure fonts
    data:     pd.DataFrame | None = None
    flags:    list[str] = field(default_factory=list)

    @property
    def fs(self) -> dict[str, float]:
        if self.small:
            return {"title": 12, "label": 10, "legend": 8, "tick": 9, "note": 8}
        return {"title": 14, "label": 12, "legend": 10, "tick": 10, "note": 9}


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def on_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Index per-update rows by update on a regular grid; missing updates
    become NaN rows so lines break instead of bridging gaps."""
    if df.empty:
        return df.set_index("update") if "update" in df.columns else df
    d = df.drop_duplicates("update", keep="last").set_index("update").sort_index()
    return d.reindex(update_grid(d.index))


def rolling(y: pd.Series, window: int, weights: pd.Series | None = None,
            regimes: list[int] | None = None) -> pd.Series:
    """Centred rolling mean on the update grid that never fills a gap.

    `weights` gives the pooled mean Σ(y·w)/Σw, the right average for per-game
    ratios when the number of games per update swings from 2 to 104.
    `regimes` are updates at which the window restarts, so a config change
    (e.g. an 8x larger batch) stays a step rather than becoming a ramp.
    """
    if regimes:
        group = np.searchsorted(np.asarray(sorted(regimes)), y.index.to_numpy(), side="right")
        parts = [rolling(y[group == g], window,
                         None if weights is None else weights[group == g])
                 for g in np.unique(group)]
        return pd.concat(parts).reindex(y.index)
    if weights is None:
        r = y.rolling(window, center=True, min_periods=1).mean()
    else:
        w = weights.where(y.notna())
        r = ((y * w).rolling(window, center=True, min_periods=1).sum()
             / w.rolling(window, center=True, min_periods=1).sum())
    return r.where(y.notna())


def _edges(x: np.ndarray) -> np.ndarray:
    step = float(x[1] - x[0]) if len(x) > 1 else 1.0
    return np.append(x - step / 2, x[-1] + step / 2)


def constant_value(values: pd.Series, min_points: int = 5) -> float | None:
    """The value of a series that never changes (e.g. a metric stuck at 0)."""
    v = values.dropna()
    if len(v) >= min_points and v.nunique() == 1:
        return float(v.iloc[0])
    return None


def config_text(seg: Segment, diff: dict[str, tuple[str, str]] | None = None) -> str:
    """Full config of a segment, or the diff to the previous one."""
    if diff is None:
        return ", ".join(f"{_CONFIG_NAMES[k]} {seg.config[k]}"
                          for k in _CONFIG_NAMES if k in seg.config) or "config not logged"
    return ", ".join(f"{_CONFIG_NAMES[k]} {a}->{b}"
                      for k, (a, b) in diff.items()) or "no config change"


# ══════════════════════════════════════════════════════════════════════════════
# Panel drawing
# ══════════════════════════════════════════════════════════════════════════════

def draw_panel(ax, p: Panel, data: pd.DataFrame, ctx: Ctx) -> None:
    fs = ctx.fs
    if p.custom is not None:
        p.custom(ax, ctx)
        return
    if p.title:
        ax.set_title(p.title, fontsize=fs["title"], fontweight="bold")
    present = [c for c in p.lines if c in data.columns and data[c].notna().any()]
    if not present:
        ax.text(0.5, 0.5, "not logged in this run", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=fs["label"])
        ax.set_yticks([])
        return

    x = data.index.to_numpy()
    weights = data[p.weights] if p.weights and p.weights in data.columns else None
    regimes = ctx.regimes
    if p.bars:
        _draw_bars(ax, p, data, x, fs)

    if p.hband:
        lo, hi, label = p.hband
        ax.axhspan(lo, hi, color=MUTED, alpha=0.15, lw=0, label=label)

    template = ctx.template and len(p.lines) == 1 and not next(iter(p.lines.values()))
    for i, (col, label) in enumerate(p.lines.items()):
        if col not in present:
            continue
        color = LINE_COLORS[i]
        y     = data[col] * p.scale
        const = constant_value(y)
        suffix = "" if const is None else f" (constant {const:g})"
        if p.sparse_markers and y.isna().mean() > 0.2:
            ax.plot(x, y, ".", color=color, alpha=0.6, ms=3)
        else:
            ax.plot(x, y, color=color, alpha=RAW_ALPHA, lw=RAW_LW,
                    label=f"Raw{suffix}" if template else None)
        ax.plot(x, rolling(y, ctx.window, weights, regimes), color=color, lw=ROLL_LW,
                label=f"Rolling mean (w={ctx.window})" if template else f"{label}{suffix}")

    # Bands after the lines, as in the template, so the legend reads Raw,
    # Rolling mean, band; their lower default zorder keeps them behind.
    first_color = LINE_COLORS[list(p.lines).index(present[0])]
    if p.std and p.std in data.columns:
        base, spread = data[present[0]] * p.scale, data[p.std] * p.scale
        ax.fill_between(x, base - spread, base + spread, color=first_color,
                        alpha=BAND_ALPHA, lw=0, label="±1 std")
    if p.minmax and all(c in data.columns for c in p.minmax):
        # min and max over the samples of one update are integer counts, so a
        # stepped band is honest where linear interpolation would draw spikes.
        lo, hi = (data[c] * p.scale for c in p.minmax)
        ax.fill_between(x, lo, hi, step="mid", color=first_color, alpha=BAND_ALPHA, lw=0,
                        label="min–max")

    for yv, label in p.hlines:
        ax.axhline(yv, color=MUTED, ls="--", lw=1.0, label=label or None)
    for b in ctx.run.boundaries:
        ax.axvline(b.update - 0.5, color=MUTED, ls=(0, (4, 3)), lw=1.0, zorder=1.5)
    if p.refresh_ticks and ctx.run.refreshes:
        ax.vlines([u - 0.5 for u in ctx.run.refreshes], 0, 0.04, colors=MUTED, lw=0.8,
                  transform=ax.get_xaxis_transform(), label="opponent refresh")

    if p.logy:
        ax.set_yscale("log")
    if p.ylim:
        ax.set_ylim(*p.ylim)
        _off_scale_note(ax, p, data, present, fs)
    if p.ylabel:
        ax.set_ylabel(p.ylabel, fontsize=fs["label"])
    ax.tick_params(labelsize=fs["tick"])
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2 or (handles and not ctx.small):
        ax.legend(fontsize=fs["legend"], loc="best", framealpha=0.85,
                  ncols=2 if ctx.small and len(handles) >= 4 else 1)


def _draw_bars(ax, p: Panel, data: pd.DataFrame, x: np.ndarray, fs: dict) -> None:
    """Stacked bars behind the lines, drawn in main-axis units (count / cap)
    with a secondary axis for the count scale: correct z-order, one grid and
    one legend, unlike a twinx axis."""
    cols = [c for c in p.bars if c in data.columns and data[c].notna().any()]
    if not cols:
        return
    cap    = bars_cap(p, data)
    edges  = _edges(x)
    bottom = np.zeros(len(x))
    colors = BAR_COLORS if len(p.bars) > 1 else (BAR_SINGLE,)
    for i, (col, label) in enumerate(p.bars.items()):
        if col not in cols:
            continue
        v   = data[col].to_numpy(dtype=float, copy=True) / cap
        top = bottom + v                           # NaN where the update is missing
        const = constant_value(data[col])
        suffix = "" if const is None else f" (constant {const:g})"
        ax.stairs(top, edges, baseline=bottom, fill=True, color=colors[i % len(colors)],
                  alpha=BAR_ALPHA, lw=0, label=f"{label}{suffix}", zorder=1)
        bottom = np.where(np.isnan(v), bottom, top)

    sec = ax.secondary_yaxis("right", functions=(lambda y: y * cap, lambda c: c / cap))
    sec.set_ylabel(p.bars_label, fontsize=fs["label"] - 1, color=MUTED)
    sec.tick_params(labelsize=fs["tick"], colors=MUTED)


def bars_cap(p: Panel, data: pd.DataFrame) -> float:
    """Top of the bar axis: len(bars) × the cap column's max (3 × n_samples =
    60 for three stacked counts), a fixed number, or the tallest stack."""
    if isinstance(p.bars_cap, (int, float)):
        return float(p.bars_cap)
    if isinstance(p.bars_cap, str) and p.bars_cap in data.columns and data[p.bars_cap].notna().any():
        return float(data[p.bars_cap].max()) * len(p.bars)
    cols  = [c for c in p.bars if c in data.columns]
    total = data[cols].sum(axis=1, min_count=1).max() if cols else np.nan
    return float(total) * 1.1 if pd.notna(total) and total > 0 else 1.0


def _off_scale_note(ax, p: Panel, data: pd.DataFrame, cols: list[str], fs: dict) -> None:
    lo, hi = p.ylim
    n = sum(int(((data[c] * p.scale < lo) | (data[c] * p.scale > hi)).sum()) for c in cols)
    if n:
        ax.text(0.01, 0.02, f"{n} point{'s' if n > 1 else ''} off-scale", transform=ax.transAxes,
                fontsize=fs["note"], color=MUTED, ha="left", va="bottom")


def _boundary_labels(ax, run: RunData, fs: dict) -> None:
    """Label resume boundaries once, as ticks of a top axis ("S2", "S3", …)."""
    if not run.boundaries:
        return
    top = ax.secondary_xaxis("top")
    top.set_xticks([b.update - 0.5 for b in run.boundaries],
                   labels=[b.label for b in run.boundaries])
    top.tick_params(length=0, labelsize=fs["note"], colors=MUTED)


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def render_scenario(name: str, spec: FigureSpec, run: RunData, window: int = WINDOW):
    rows = run.scenarios[run.scenarios["scenario"] == name]
    data = on_grid(rows)
    n = len(spec.panels)
    fig, axes = plt.subplots(
        n, 1, sharex=True, squeeze=False, layout="constrained",
        figsize=FIGSIZE if n == 1 else FIGSIZE_TALL,
        height_ratios=[p.ratio for p in spec.panels],
    )
    axes = axes[:, 0]
    for i, (ax, p) in enumerate(zip(axes, spec.panels)):
        draw_panel(ax, p, data, Ctx(run, window, template=(i == 0)))
    fs = Ctx(run).fs
    axes[-1].set_xlabel("Training Update", fontsize=fs["label"])
    if len(data):
        axes[0].set_xlim(data.index.min() - 1, data.index.max() + 1)
    _boundary_labels(axes[0], run, fs)
    if "n_samples" in data.columns and data["n_samples"].notna().any():
        axes[0].annotate(f"n_samples = {int(data['n_samples'].max())}", xy=(0.98, 0.04),
                         xycoords="axes fraction", ha="right", fontsize=fs["note"], color="gray",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    fig.suptitle(spec.title, fontsize=fs["title"], fontweight="bold")
    return fig


def render_training(run: RunData, metrics: pd.DataFrame, rows, window: int = WINDOW,
                    flags: list[str] | None = None):
    data = on_grid(metrics)
    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, squeeze=False,
                             figsize=GRID_FIGSIZE, layout="constrained")
    ctx = Ctx(run, window, regimes=[b.update for b in run.boundaries if b.diff],
              small=True, data=metrics, flags=flags or [])
    for r, row in enumerate(rows):
        for c in range(n_cols):
            if c < len(row):
                draw_panel(axes[r, c], row[c], data, ctx)
            else:
                axes[r, c].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Training Update", fontsize=ctx.fs["label"])
    if len(data):
        axes[0, 0].set_xlim(data.index.min() - 1, data.index.max() + 1)
    for ax in axes[0]:
        _boundary_labels(ax, run, ctx.fs)
    fig.suptitle(f"{run.name} — training metrics", fontsize=16, fontweight="bold")
    return fig


def run_info(ax, ctx: Ctx) -> None:
    """Text panel of the training grid: segments, their config changes, totals."""
    run, metrics = ctx.run, ctx.data if ctx.data is not None else ctx.run.metrics
    ax.axis("off")
    ax.set_title("Run info", fontsize=ctx.fs["title"], fontweight="bold")
    diffs = {b.tag: b.diff for b in run.boundaries}
    lines = []
    for k, seg in enumerate(run.segments):
        span = run.spans.get(seg.tag)
        kept = f"{span[0]}-{span[1]}" if span else "-"
        text = config_text(seg) if k == 0 else config_text(seg, diffs.get(seg.tag, {}))
        lines += textwrap.wrap(f"S{k + 1}  {kept}: {text}", width=54, subsequent_indent="      ")
    lines.append("")
    totals = [f"{len(metrics)} updates"]
    if "wall_time_s" in metrics.columns:
        totals.append(f"{metrics['wall_time_s'].sum() / 3600:.1f} h wall-clock")
    if "n_games" in metrics.columns:
        totals.append(f"{metrics['n_games'].sum():,.0f} games")
    lines.append(", ".join(totals))
    if ctx.flags:
        lines.append(f"{len(ctx.flags)} sanity flags: see the handbook cover")
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=ctx.fs["legend"])


def cover_lines(run: RunData, flags: list[str], notes: tuple[str, ...] = ()) -> list[str]:
    """Plain-text cover: segment table, issues, sanity flags, notes. Written to
    segments.txt, printed, and rendered as the first handbook page."""
    diffs = {b.tag: b.diff for b in run.boundaries}
    m = run.metrics
    lines = [
        f"Run report: {run.name}",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} from {len(run.segments)} segment(s)"
        + (f", updates {m["update"].min()}-{m['update'].max()}" if len(m) else ""),
        "",
        "Segments (lineage order; a later segment wins on overlapping updates)",
        f"  {'':3} {'tag':<20} {'kept updates':<13} {'resumed from':<13} config",
    ]
    for k, seg in enumerate(run.segments):
        span = run.spans.get(seg.tag)
        kept = f"{span[0]}-{span[1]}" if span else "-"
        if seg.resumed_from is not None:
            resumed = f"ckpt {seg.resumed_from}"
        else:
            resumed = "scratch" if seg.from_scratch else "unknown"
        text = config_text(seg) if k == 0 else config_text(seg, diffs.get(seg.tag, {}))
        lines.append(f"  S{k + 1:<2} {seg.tag:<20} {kept:<13} {resumed:<13} {text}")
    lines += ["", "Issues"] + ([f"  {i}" for i in run.issues] or ["  none"])
    lines += ["", "Sanity flags"] + ([f"  {f}" for f in flags] or ["  none"])
    if notes:
        lines += ["", "Notes"] + [f"  {n}" for n in notes]
    return lines


def render_cover(lines: list[str], max_lines: int = 64):
    fig = plt.figure(figsize=COVER_SIZE)
    fig.text(0.04, 0.95, lines[0], fontsize=16, fontweight="bold", va="top")
    body = lines[1:]
    if len(body) > max_lines:
        body = body[: max_lines - 1] + [f"  ... {len(body) - max_lines + 1} more lines in segments.txt"]
    fig.text(0.04, 0.90, "\n".join(body), family="monospace", fontsize=7, va="top")
    return fig
