"""
smoke_critic_init.py - the sc-48 baseline / acceptance table.

Runs the offline probes in RL/models/diagnostics.py against a FRESHLY
INITIALISED policy and prints a table with the pass/fail threshold beside every
number. No training, no checkpoints.

Against the current (pre-redesign) architecture this script is EXPECTED TO FAIL.
That failure is the point: it is the measured baseline the redesign is judged
against, so run it and keep the output before changing anything.

Run from project root:
  python eval/smoke_critic_init.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RL.models.diagnostics import (
    branch_balance,
    collect_states,
    gradient_flow,
    perturbation_probe,
    readout_dispersion,
    saturation_probe,
    value_stats,
)
from RL.models.policy import PolicyNetwork
from RL.ppo.game_manager import TrainConfig

N_STATES = 192
SEED     = 0


# -- Thresholds (sc-48 plan §7) ------------------------------------------------
# (label, key, comparison, threshold, why)
CHECKS = [
    ("P1  unique value fraction",  "unique_frac",      ">", 0.95,
     "values must be continuous, not a few levels"),
    ("P1  gap ratio (max/median)", "gap_ratio",        "<", 20.0,
     "big gaps in sorted values = discrete plateaus"),
    ("P1  std across states",      "std",              ">", 1e-4,
     "non-degenerate"),
    ("P2  graph dead fraction",    "graph_dead_frac",  "<", 0.02,
     "a board change must move the value at all"),
    ("P2  graph jump ratio",       "graph_jump_ratio", "<", 8.0,
     "no single tile should dominate"),
    ("P2  graph L_rel",            "graph_L_rel",      "<", 0.25,
     "one tile << spread across unrelated boards"),
    # The board must not merely be SMOOTH, it must be INFLUENTIAL. Without this
    # check a critic that ignores the graph entirely scores perfectly on
    # graph_L_rel and graph_jump_ratio, which is exactly what the pre-redesign
    # network does (graph dV is ~1.5% of scalar dV).
    ("P2b graph/scalar sensitivity", "graph_vs_scalar", ">", 0.25,
     "the board must move the value comparably to the scalars"),
    ("P3  max saturated fraction", "max_sat_frac",     "<", 0.01,
     "no layer may sit in a saturating tail"),
    ("P3b scalar/board norm ratio", "ratio",           "<", 5.0,
     "the board must be visible next to the scalars"),
    ("P5  readout dispersion",     "dispersion",       ">", 0.5,
     "pooled embedding must vary with the board"),
    # NOTE: this is a connectivity check, not a saturation detector. Saturation
    # attenuates the encoder and head gradients by similar factors, so the
    # RATIO stays healthy while both shrink. P3/P3b are what catch saturation.
    ("P8  encoder/head grad ratio", "ratio",           ">", 1e-3,
     "value loss must reach the encoder at all"),
]


def _cmp(value: float, op: str, thr: float) -> bool:
    return value > thr if op == ">" else value < thr


def main() -> int:
    torch.manual_seed(SEED)

    cfg = TrainConfig()
    # Small boards and short games: the probes need many states, not long ones.
    cfg.board_size_range    = (11, 11)
    cfg.max_turns_per_game  = 6

    policy = PolicyNetwork(cfg)
    policy.eval()

    print("=" * 78)
    print("sc-48 critic init probes -", N_STATES, "states, fresh network, no training")
    print("=" * 78)

    print("\ncollecting states ...", flush=True)
    states = collect_states(cfg, policy, n_states=N_STATES, seed=SEED)
    print(f"  {len(states)} states, board {states[0].Nx}x{states[0].Ny}")

    print("\nrunning probes ...", flush=True)
    vs   = value_stats(policy, states)
    pert = perturbation_probe(policy, states, seed=SEED)
    sat  = saturation_probe(policy, states)
    bb   = branch_balance(policy, states)
    rd   = readout_dispersion(policy, states)
    gf   = gradient_flow(policy, states)

    # -- Raw numbers -----------------------------------------------------------
    print("\n-- P1  value distribution (V_TERM) " + "-" * 42)
    print(f"  mean {vs['mean']:+.4f}   std {vs['std']:.4f}   "
          f"range [{vs['min']:+.4f}, {vs['max']:+.4f}]   ptp {vs['ptp']:.4f}")
    print(f"  unique values {int(vs['n_unique'])}/{int(vs['n_states'])}"
          f"  ({vs['unique_frac']:.3f})     gap ratio {vs['gap_ratio']:.1f}")

    print("\n-- P2  minimal perturbations " + "-" * 48)
    print(f"  graph-only : dead {pert['graph_dead_frac']:.3f}   "
          f"mean dV {pert['graph_mean_dV']:.4f}   max dV {pert['graph_max_dV']:.4f}   "
          f"jump {pert['graph_jump_ratio']:.1f}")
    print(f"  scalar-only: mean dV {pert['scalar_mean_dV']:.4f}   "
          f"max dV {pert['scalar_max_dV']:.4f}")
    print(f"  reference std across unrelated states: {pert['ref_std']:.4f}")

    print("\n-- P3  activation saturation (|input to a saturating act| > 4) " + "-" * 14)
    for name, s in sat.items():
        info = name.startswith("[scale]")
        flag = "" if info else ("  <-- SATURATED" if s["sat_frac"] > 0.01 else "")
        print(f"  {name:<40} absmean {s['absmean']:9.3f}   absmax {s['absmax']:10.3f}"
              f"   sat {s['sat_frac']:.3f}{flag}")

    print("\n-- P3b branch balance at the fusion point " + "-" * 35)
    print(f"  board {bb['board_norm']:.3f}   scalar {bb['scalar_norm']:.3f}"
          f"   ratio {bb['ratio']:.1f}x   board share {bb['board_share']:.4f}")

    print("\n-- P5  readout dispersion " + "-" * 51)
    print(f"  dispersion {rd['dispersion']:.4f}   mean ch std {rd['mean_ch_std']:.4f}"
          f"   mean |ch| {rd['mean_abs_ch']:.4f}   dead ch {int(rd['dead_channels'])}")

    print("\n-- P8  gradient flow " + "-" * 56)
    print(f"  encoder {gf['encoder_grad']:.3e}   critic {gf['critic_grad']:.3e}"
          f"   ratio {gf['ratio']:.3e}   params w/o grad {int(gf['n_no_grad'])}")

    # -- Verdict ---------------------------------------------------------------
    # Derived checks that need more than one probe.
    pert["graph_vs_scalar"] = (
        pert["graph_mean_dV"] / pert["scalar_mean_dV"]
        if pert["scalar_mean_dV"] > 0 else float("inf")
    )
    # [scale] rows are Linear outputs, reported for information only - a Linear
    # followed by a LayerNorm cannot saturate whatever its magnitude. Only the
    # real saturating activations count toward the check.
    sat_summary = {"max_sat_frac": max(
        (s["sat_frac"] for k, s in sat.items() if not k.startswith("[scale]")),
        default=0.0,
    )}

    sources = {
        "unique_frac": vs, "gap_ratio": vs, "std": vs,
        "graph_dead_frac": pert, "graph_jump_ratio": pert, "graph_L_rel": pert,
        "graph_vs_scalar": pert,
        "max_sat_frac": sat_summary,
        "dispersion": rd,
    }
    print("\n" + "=" * 78)
    print(f"{'check':<32}{'value':>14}{'':>4}{'threshold':>14}{'':>4}  result")
    print("-" * 78)
    n_fail = 0
    for label, key, op, thr, _why in CHECKS:
        src = bb if label.startswith("P3b") else (gf if label.startswith("P8") else sources[key])
        val = src[key]
        ok  = _cmp(val, op, thr)
        n_fail += (not ok)
        print(f"{label:<32}{val:>14.4f}{'':>4}{op + ' ' + format(thr, '.4f'):>14}{'':>4}  "
              f"{'PASS' if ok else 'FAIL'}")
    print("-" * 78)
    print(f"{len(CHECKS) - n_fail}/{len(CHECKS)} passed")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
