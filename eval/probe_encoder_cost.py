"""
probe_encoder_cost.py - the P10 gate for the sc-48 encoder rewrite.

The plan replaces the neighbourhood-masked TransformerConv encoder with true
global self-attention (every tile attends to every tile) plus 2-D RoPE and a
scalar global node. That fixes the 2-tile receptive field in one layer instead
of one tile per layer - but it moves the encoder from O(|E|.D) with |E| = 8N to
O(N^2.D), and memory from O(|E|) to an N^2 attention matrix per head.

This measures both BEFORE any of it is wired in, because "30x on a small base"
is an argument, not a number. If the numbers are bad, the documented fallback is
a hybrid: keep local TransformerConv layers and add the global node as the only
global-communication path, which keeps O(|E|) and still fixes the receptive
field.

Pure forward-pass benchmarking - no training, consistent with the offline-only
verification constraint.

Run from project root:
  python eval/probe_encoder_cost.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RL.models.main_modules import GraphTransformerEncoder
from RL.models.utility_modules import _build_grid_edge_index
from game.enums import NODE_FEAT_DIM

D, HEADS, DEPTH = 48, 4, 2
BOARDS  = [11, 14, 16]          # 121, 196, 256 tiles
BATCHES = [1, 32, 128]
N_WARMUP, N_ITER = 3, 10


class _GlobalAttnPrototype(nn.Module):
    """Minimal stand-in for the proposed encoder: pre-norm global attention.

    Deliberately NOT the real thing - no RoPE, no global node - because this
    measures the cost floor of dense N^2 attention, which is the part that
    decides the gate. RoPE is a cheap elementwise rotation on Q/K and the global
    node adds one token; neither changes the asymptotics.
    """

    def __init__(self, d: int = D, heads: int = HEADS, depth: int = DEPTH):
        super().__init__()
        self.input_proj = nn.Linear(NODE_FEAT_DIM, d)
        self.attn  = nn.ModuleList(
            [nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(depth)]
        )
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.n2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.ff = nn.ModuleList([
            nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
            for _ in range(depth)
        ])

    def forward(self, x):                      # (B, N, F) -> (B, N, D)
        x = self.input_proj(x)
        for a, n1, n2, ff in zip(self.attn, self.n1, self.n2, self.ff):
            h = n1(x)
            x = x + a(h, h, h, need_weights=False)[0]
            x = x + ff(n2(x))
        return x


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, dev) -> float:
    for _ in range(N_WARMUP):
        fn()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        fn()
    _sync(dev)
    return (time.perf_counter() - t0) / N_ITER * 1e3      # ms


def _peak_mb(fn, dev) -> float:
    if dev.type != "cuda":
        return float("nan")
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print("=" * 78)
    print(f"P10 encoder cost - device={dev}, D={D}, heads={HEADS}, depth={DEPTH}")
    print(f"    {N_ITER} iters after {N_WARMUP} warmup, forward only, no_grad")
    print("=" * 78)

    local  = GraphTransformerEncoder(hidden_dim=D, n_heads=HEADS, depth=DEPTH).to(dev).eval()
    glob   = _GlobalAttnPrototype().to(dev).eval()

    print(f"\n{'N':>5} {'B':>5} | {'local ms':>10} {'global ms':>10} {'x':>7}"
          f" | {'local MB':>9} {'global MB':>10} {'x':>7}")
    print("-" * 78)

    rows = []
    with torch.no_grad():
        for side in BOARDS:
            N  = side * side
            ei = _build_grid_edge_index(side, side).to(dev)
            for B in BATCHES:
                x_dense = torch.randn(B, N, NODE_FEAT_DIM, device=dev)
                x_flat  = torch.randn(B * N, NODE_FEAT_DIM, device=dev)
                # Batched local path: one big disconnected graph, as PyG's
                # Batch.from_data_list produces.
                offs = torch.arange(B, device=dev).repeat_interleave(ei.shape[1]) * N
                ei_b = ei.repeat(1, B) + offs

                def run_local():
                    h = local.input_proj(x_flat)
                    return local._run_layers(h, ei_b)

                def run_global():
                    return glob(x_dense)

                try:
                    t_l = _time(run_local, dev)
                    m_l = _peak_mb(run_local, dev)
                    t_g = _time(run_global, dev)
                    m_g = _peak_mb(run_global, dev)
                except torch.cuda.OutOfMemoryError:
                    print(f"{N:>5} {B:>5} | {'OOM':>10}")
                    torch.cuda.empty_cache()
                    continue

                rows.append((N, B, t_l, t_g, m_l, m_g))
                print(f"{N:>5} {B:>5} | {t_l:>10.3f} {t_g:>10.3f} {t_g/t_l:>7.1f}"
                      f" | {m_l:>9.1f} {m_g:>10.1f} {m_g/max(m_l,1e-9):>7.1f}")
                del x_dense, x_flat, ei_b
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

    print("=" * 78)
    if rows:
        roll = [r for r in rows if r[1] == 1]        # rollout path: B = 1
        ppo  = [r for r in rows if r[1] == 128]      # PPO path
        if roll:
            w = max(r[3] / r[2] for r in roll)
            print(f"  rollout path (B=1)   : worst slowdown {w:.1f}x, "
                  f"absolute {max(r[3] for r in roll):.2f} ms")
        if ppo:
            w = max(r[3] / r[2] for r in ppo)
            print(f"  PPO path (B=128)     : worst slowdown {w:.1f}x, "
                  f"peak {max(r[5] for r in ppo):.0f} MB")
        print("\n  Gate: the rollout path is the one that matters - it runs once per\n"
              "  decision across all workers, while the PPO path runs a few times per\n"
              "  update. Absolute rollout cost well under the simulator's per-decision\n"
              "  cost means the slowdown factor is not the deciding number.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
