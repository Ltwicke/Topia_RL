"""
eval/run_report/specs.py
──────────────────────────────────────────────────────────────────────────────
Everything the run report knows about metrics, as data: which columns each
scenario figure and training panel shows, how raw CSV values are cleaned, and
which sanity flags go on the cover page.

Adding a scenario figure or reacting to a renamed column is an edit here only.
Column names must match what scenarios/configs/<name>.py and RL/train.py
write; tests/test_run_report.py checks that contract.
"""

from __future__ import annotations

import math

import pandas as pd

from .loader import RunData
from .plot import RATE_YLIM, FigureSpec, Panel, constant_value, run_info


# ══════════════════════════════════════════════════════════════════════════════
# Critic constants
# ══════════════════════════════════════════════════════════════════════════════

# Mirror TrainConfig (value_n_bins, value_sigma_ratio, support [0, conquest_reward]);
# the .log does not record them.
N_BINS, V_MIN, V_MAX, SIGMA_RATIO = 51, 0.0, 2.0, 0.75
MIN_TERMINAL_EVENTS = 5          # mirrors RL/ppo/batch_processing.py


def hl_gauss_entropy(y: float) -> float:
    """Entropy of the HL-Gauss target for return y: the lowest cross-entropy
    the V_TERM head can reach on that target."""
    width = (V_MAX - V_MIN) / N_BINS
    sigma = SIGMA_RATIO * width
    cdf = [0.5 * (1 + math.erf((V_MIN + i * width - y) / (sigma * math.sqrt(2))))
           for i in range(N_BINS + 1)]
    p = [b - a for a, b in zip(cdf, cdf[1:])]
    total = sum(p)
    return -sum(q / total * math.log(q / total) for q in p if q > 0)


# Targets at the edge of the support (0 or 2) are half-truncated and have the
# lowest entropy; interior targets the highest.
CE_FLOOR   = (hl_gauss_entropy(V_MIN), hl_gauss_entropy((V_MIN + V_MAX) / 2))   # ≈ (0.51, 1.20)
CE_UNIFORM = math.log(N_BINS)                                                    # ≈ 3.93


# ══════════════════════════════════════════════════════════════════════════════
# Scenario figures
# ══════════════════════════════════════════════════════════════════════════════

_UNSAFE_YLIM = (-0.1, 3.1)       # three riders / three key tiles

SCENARIOS: dict[str, FigureSpec] = {
    "Get_defender_and_wall": FigureSpec(
        "Get Defender and Wall — wall upgrade and defender",
        (Panel(lines={"success_both": "success rate (wall and defender)"},
               ylim=RATE_YLIM, ylabel="Success rate",
               bars={"n_any_unit_created": "any unit created",
                     "n_defenders_created": "defender created",
                     "n_wall_chosen": "wall chosen"},
               bars_cap="n_samples",
               bars_label="Count (stacked; defenders are among units)"),)),
    "Simple_dash_dancing2": FigureSpec(
        "Simple Dash Dancing — Tiles Uncovered per Rollout over Training",
        (Panel(lines={"uncovered_delta_mean": ""}, std="uncovered_delta_std",
               ylabel="Mean Tiles Uncovered (delta)"),)),
    "Knight_choice_no_village": FigureSpec(
        "Knight Choice (no village) — direction chosen",
        (Panel(lines={"choice_left_rate": "left", "choice_right_rate": "right"},
               ylim=RATE_YLIM, ylabel="Share of samples"),)),
    "Rider_leapfrogging": FigureSpec(
        "Rider Leapfrogging — Tiles Uncovered per Rollout over Training",
        (Panel(lines={"uncovered_delta_mean": ""}, std="uncovered_delta_std",
               ylabel="Mean Tiles Uncovered (delta)"),)),
    "road_for_kill": FigureSpec(
        "Road for Kill — kill rate and decisions taken",
        (Panel(lines={"success_rate": "success rate"}, ylim=RATE_YLIM, ylabel="Success rate",
               bars={"avg_decisions_taken": "avg decisions taken"},
               bars_cap="n_decisions_max",
               bars_label="Decisions per rollout (axis = cap)"),)),
    "Escaping_riders2": FigureSpec(
        "Escaping Riders — riders left on unsafe tiles",
        (Panel(lines={"unsafe_rider_count_mean": "mean unsafe riders"},
               minmax=("unsafe_rider_count_min", "unsafe_rider_count_max"),
               ylim=_UNSAFE_YLIM, ylabel="Riders on unsafe tiles", ratio=3),
         Panel(lines={"all_safe_rate": "all riders safe"}, ylim=RATE_YLIM,
               ylabel="All safe", ratio=1))),
    "Upgrade_city_order2": FigureSpec(
        "Upgrade City Order — both cities upgraded",
        (Panel(lines={"both_upgraded_rate": "both cities upgraded"}, ylim=RATE_YLIM,
               ylabel="Rate", ratio=2),
         Panel(lines={"avg_decisions_taken": "avg decisions taken",
                      "avg_cities_upgraded": "avg cities upgraded"},
               ylabel="Per rollout", ratio=1))),
    "Giant_Houdini": FigureSpec(
        "Giant Houdini — all three goals",
        (Panel(lines={"all_three_rate": "all three goals"}, ylim=RATE_YLIM, ylabel="Rate",
               bars={"n_rider_on_118": "rider on tile 118",
                     "n_city_upgraded": "city upgraded",
                     "n_superunit_chosen": "superunit chosen"},
               bars_cap="n_samples",
               bars_label="Count (stacked; superunits are among upgrades)"),)),
    "Defender_ZoC": FigureSpec(
        "Defender ZoC — key tiles occupied",
        (Panel(lines={"occupied_count_mean": "mean occupied"},
               minmax=("occupied_count_min", "occupied_count_max"),
               ylim=_UNSAFE_YLIM, ylabel="Key tiles occupied (of 3)", ratio=3),
         Panel(lines={"two_of_three_rate": "at least 2 of 3"}, ylim=RATE_YLIM,
               ylabel="Rate", ratio=1))),
}

# Scenarios whose signal is the rendered board, covered by GIFs only.
GIF_ONLY = ("Dont_attack", "Rider_hit_and_run", "Estimate_Drylands_endgame", "estimate_lakes11")


# ══════════════════════════════════════════════════════════════════════════════
# Training grid (rows follow the story's groups)
# ══════════════════════════════════════════════════════════════════════════════

TRAINING: tuple[tuple[Panel, ...], ...] = (
    (   # cost & outcomes: story groups 1 and 2
        Panel(title="Time per update",
              lines={"wall_time_s": "wall", "t_collect_s": "collect", "t_ppo_s": "PPO"},
              scale=1 / 60, logy=True, ylabel="minutes"),
        Panel(title="Games & outcomes",
              lines={"active_win_rate": "active win rate", "conquest_rate": "conquest rate"},
              weights="n_games", ylim=RATE_YLIM, ylabel="rate (pooled over games)",
              bars={"n_games": "games finished"}, bars_label="games finished",
              refresh_ticks=True),
        Panel(title="Reward per finished game",
              lines={"avg_active_reward": "term + dense, unweighted"},
              weights="n_games", ylabel="reward"),
        Panel(title="Estimator loss", lines={"est_loss": "hidden-tile estimator"},
              ylabel="nats per hidden tile"),
    ),
    (   # behaviour: story group 3
        Panel(title="Decisions per turn", lines={"decisions_per_turn": "decisions per turn"},
              hlines=((1.0, "1.0 = racing the turn limit"),),
              ylabel="active decisions per turn"),
        Panel(title="Episode length",
              lines={"avg_ep_len": "active decisions per finished game"},
              weights="n_games", ylabel="decisions"),
        Panel(title="Policy entropy", lines={"entropy": "entropy"}, ylabel="nats"),
        Panel(title="Voluntary EndTurn share",
              lines={"endturn_vol_share": "voluntary EndTurns"}, scale=100,
              ylabel="per 100 active decisions"),
    ),
    (   # optimisation & critic: open questions A and B
        Panel(title="Policy surrogate gain (p_loss)",
              lines={"p_loss": "L_clip, logged without the minus sign"},
              hlines=((0.0, ""),), ylabel="surrogate objective"),
        Panel(title="Value loss", lines={"v_loss": "CE_term + MSE_dense"},
              hband=(*CE_FLOOR, "CE_term floor"),
              hlines=((CE_UNIFORM, "ln 51: uniform critic"),), ylabel="mixed units"),
        Panel(title="Critic fit (explained variance)",
              lines={"ev_term": "V_TERM", "ev_dense": "V_DENSE"},
              hlines=((0.0, "0 = predicting the mean"),), ylim=(-0.5, 1.02),
              sparse_markers=True, ylabel="explained variance"),
        Panel(title="Run info", custom=run_info),
    ),
)

# Columns derived by clean_metrics(), not logged by train.py.
DERIVED = ("conquest_rate", "endturn_vol_share")

NOTES = (
    "Metrics row N comes from rollouts of theta(N-1) plus the losses of update N;",
    "scenario row N is evaluated with theta(N), after the PPO step of update N.",
    "Simple_dash_dancing2, Knight_choice_no_village, road_for_kill, Giant_Houdini, Dont_attack and",
    "estimate_lakes11 are played from P1's seat, while PPO trains only on P0's transitions.",
    "Stacked bars in Get_defender_and_wall and Giant_Houdini are nested counts: the stack height is not a total.",
)


# ══════════════════════════════════════════════════════════════════════════════
# Cleaning and sanity flags
# ══════════════════════════════════════════════════════════════════════════════

# Per-game ratios train.py divides by max(n_games, 1): meaningless at 0 games.
PER_GAME = ("active_win_rate", "conquest_rate", "avg_ep_len", "avg_active_reward")


def clean_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Mask values that cannot be read as logged and add derived columns.
    Returns the cleaned frame and one note per change, for the cover page."""
    df, notes = df.copy(), []
    cols = set(df.columns)
    if "conquest_rate" not in cols and {"n_conquest", "n_games"} <= cols:
        df["conquest_rate"] = df["n_conquest"] / df["n_games"].where(df["n_games"] > 0)
        notes.append("conquest_rate derived as n_conquest / n_games (not logged by this run)")
    if "n_games" in cols:
        zero = df["n_games"] == 0
        if zero.any():
            for col in PER_GAME:
                if col in df.columns:
                    df[col] = df[col].mask(zero)
            notes.append(f"{int(zero.sum())} updates finished no game: per-game metrics masked there")
    if {"ev_term", "n_terminal_events"} <= cols:
        low = (df["n_terminal_events"] < MIN_TERMINAL_EVENTS) & df["ev_term"].notna()
        if low.any():
            df["ev_term"] = df["ev_term"].mask(low)
            notes.append(f"ev_term masked at {int(low.sum())} updates with fewer than "
                         f"{MIN_TERMINAL_EVENTS} terminal events (the training-side guard "
                         f"never fires, sc-97)")
    if {"endturn_voluntary", "avg_ep_len", "n_games"} <= cols:
        decisions = df["avg_ep_len"] * df["n_games"]
        df["endturn_vol_share"] = df["endturn_voluntary"] / decisions.where(decisions > 0)
    return df, notes


def scenario_figures(run: RunData) -> list[tuple[str, FigureSpec]]:
    """Registered scenarios present in the run, in spec order, then a generic
    figure for each scenario that has no spec (so none vanishes silently)."""
    present = set(run.scenarios["scenario"].unique()) if len(run.scenarios) else set()
    figures = [(name, spec) for name, spec in SCENARIOS.items() if name in present]
    for name in sorted(present - set(SCENARIOS) - set(GIF_ONLY)):
        rows = run.scenarios[run.scenarios["scenario"] == name]
        rates = [c for c in rows.columns if c.endswith("_rate") and rows[c].notna().any()][:3]
        panel = (Panel(lines={c: c for c in rates}, ylim=RATE_YLIM, ylabel="Rate") if rates
                 else Panel(lines={"v_term": "critic V_TERM at the start state"}, ylabel="V_TERM"))
        figures.append((name, FigureSpec(f"{name} — no spec yet, generic view", (panel,))))
    return figures


def sanity_flags(run: RunData, metrics: pd.DataFrame) -> list[str]:
    """Cheap checks that catch broken metrics before anyone reads the plots."""
    flags = []
    present = set(run.scenarios["scenario"].unique()) if len(run.scenarios) else set()
    for name, spec in SCENARIOS.items():
        if name not in present:
            flags.append(f"{name}: not evaluated in this run, figure skipped")
            continue
        rows = run.scenarios[run.scenarios["scenario"] == name]
        for col in dict.fromkeys(c for p in spec.panels for c in (*p.lines, *p.bars)):
            if col in rows.columns:
                value = constant_value(rows[col])
                if value is not None:
                    flags.append(f"{name}.{col} is constant {value:g} over "
                                 f"{int(rows[col].notna().sum())} updates")
    for row in TRAINING:
        for panel in row:
            for col in panel.lines:
                if col in metrics.columns:
                    value = constant_value(metrics[col])
                    if value is not None:
                        flags.append(f"{col} is constant {value:g} over "
                                     f"{int(metrics[col].notna().sum())} updates")
    for label, df in (("metrics", metrics), ("scenarios", run.scenarios)):
        for col in (c for c in df.columns if c.endswith("_rate")):
            outside = (df[col] < 0) | (df[col] > 1)
            if outside.any():
                flags.append(f"{label}.{col} leaves [0, 1] at {int(outside.sum())} updates")
    if "frac_ret_out_of_support" in metrics.columns:
        oos = metrics["frac_ret_out_of_support"] > 0
        if oos.any():
            flags.append(f"frac_ret_out_of_support > 0 at {int(oos.sum())} updates "
                         f"(max {metrics['frac_ret_out_of_support'].max():.3f})")
    for name in sorted(present - set(SCENARIOS) - set(GIF_ONLY)):
        flags.append(f"{name}: no spec in specs.py, plotted with a generic figure")
    return flags
