"""
diagnostics.py - offline probes for the critic's value path.

Everything here runs on a FRESHLY INITIALISED network with a handful of forward
passes. No training, no checkpoints, no rollout collection beyond a few short
random games. The point is to make the sc-48 failure measurable rather than
anecdotal: "the critic predicts discrete constants and tiny board changes move
it wildly" becomes a small table of numbers that either passes or fails.

Probe index (see the sc-48 plan for the rationale behind each threshold):

  P1  value spread + discreteness      value_stats()
  P2  local smoothness, by source      perturbation_probe()
  P3  activation saturation            saturation_probe()
  P3b scalar-vs-board branch balance   branch_balance()
  P5  readout discriminability         readout_dispersion()
  P8  gradient flow to the encoder     gradient_flow()

P4 (attention entropy), P4b (RoPE sanity) and P10 (encoder cost) arrive with the
encoder rewrite; they have nothing to measure against the current local encoder.

Used by `eval/smoke_critic_init.py` (human-readable table, both configs) and by
`tests/test_critic_init.py` (hard asserts).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from game.enums import (
    CITY_SLICE,
    OPP_TYPE_SLICE,
    OWN_TYPE_SLICE,
    PLAYER_CTRL_SLICE,
    ROAD_SLICE,
)
from RL.models.main_modules import V_DENSE, V_TERM


# ══════════════════════════════════════════════════════════════════════════════
# State collection
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProbeState:
    """One board observation, kept with the dimensions the encoder needs."""
    obs: dict
    Nx:  int
    Ny:  int

    def copy_with_graph(self, graph: np.ndarray) -> "ProbeState":
        obs = dict(self.obs)
        obs["partial_graph"] = graph
        return ProbeState(obs, self.Nx, self.Ny)

    def copy_with_scalar(self, scalar: np.ndarray) -> "ProbeState":
        obs = dict(self.obs)
        obs["scalar_state"] = scalar
        return ProbeState(obs, self.Nx, self.Ny)


def collect_states(
    cfg,
    policy,
    n_states: int = 256,
    seed:     int = 0,
    max_steps_per_game: int = 40,
) -> List[ProbeState]:
    """Short random self-play rollouts, harvested for board states.

    The policy is only used to produce *legal* actions - at init it is
    effectively uniform, and the same cached states are reused across configs so
    that any difference in the probe output comes from the model, not the data.
    """
    from RL.ppo.game_manager import _make_env

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    states: List[ProbeState] = []
    with torch.no_grad():
        while len(states) < n_states:
            env = _make_env(cfg)
            obs = env.reset()
            for _ in range(max_steps_per_game):
                if len(states) >= n_states:
                    break
                states.append(ProbeState(obs, env.Nx, env.Ny))
                mask = env.get_action_mask()
                action, *_ = policy(obs, mask)
                obs, _, done, _ = env.step(
                    action, n_valid_action_types=int(mask[0].sum())
                )
                if done:
                    break
    return states[:n_states]


# ══════════════════════════════════════════════════════════════════════════════
# Forward helpers
# ══════════════════════════════════════════════════════════════════════════════

def encode_state(policy, st: ProbeState):
    """-> (node_emb (N,D), global_emb (1,D), value (n_streams,))"""
    node_emb, global_emb = policy.encoder.encode(
        st.obs["partial_graph"], st.Nx, st.Ny, st.obs["scalar_state"]
    )
    # `value_from_emb` post-sc-48; plain call on the pre-sc-48 scalar head.
    critic = policy.critic
    if hasattr(critic, "value_from_emb"):
        return node_emb, global_emb, critic.value_from_emb(global_emb)[0]
    return node_emb, global_emb, critic(global_emb)


def values_over(policy, states: Sequence[ProbeState], stream: int = V_TERM) -> np.ndarray:
    out = np.empty(len(states), dtype=np.float64)
    with torch.no_grad():
        for i, st in enumerate(states):
            v = encode_state(policy, st)[2]
            out[i] = float(v.reshape(-1)[stream])
    return out


def embeddings_over(policy, states: Sequence[ProbeState]) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for st in states:
            rows.append(encode_state(policy, st)[1].reshape(-1).cpu().numpy())
    return np.stack(rows, axis=0)      # (K, D)


# ══════════════════════════════════════════════════════════════════════════════
# P1 - value spread and discreteness
# ══════════════════════════════════════════════════════════════════════════════

def value_stats(policy, states, stream: int = V_TERM) -> Dict[str, float]:
    """Is the value distribution narrow, non-degenerate, and CONTINUOUS?

    `gap_ratio` is the discreteness detector. Sort the values and look at
    consecutive gaps: a critic that snaps between a few levels produces a
    handful of enormous gaps in an otherwise dense array, so max/median blows
    up. A genuinely continuous critic keeps it near 1.
    """
    v = values_over(policy, states, stream)
    s = np.sort(v)
    gaps = np.diff(s)
    gaps = gaps[gaps > 0]
    med = float(np.median(gaps)) if gaps.size else 0.0

    # Distinctness must be measured RELATIVE to the spread. Rounding to a fixed
    # number of decimals conflates "the critic has a few plateaus" with "the
    # critic's output range is small", and those are opposite verdicts - a
    # narrow spread at init is exactly what we want. Resolve at 1e-4 of the
    # observed range instead.
    ptp = float(np.ptp(v))
    resolution = max(ptp * 1e-4, 1e-12)
    n_unique = float(np.unique(np.round(v / resolution)).size)
    return {
        "mean":      float(v.mean()),
        "std":       float(v.std()),
        "min":       float(v.min()),
        "max":       float(v.max()),
        "ptp":       ptp,
        "n_unique":  n_unique,
        "n_states":  float(v.size),
        "unique_frac": float(n_unique / max(v.size, 1)),
        "gap_ratio": float(gaps.max() / med) if (gaps.size and med > 0) else float("inf"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# P2 - local smoothness, attributed to the graph path vs the scalar path
# ══════════════════════════════════════════════════════════════════════════════

def _graph_perturbations(st: ProbeState, n: int, rng: np.random.Generator) -> List[ProbeState]:
    """Minimal single-tile edits that leave `scalar_state` untouched.

    Isolating the graph path matters: a jump seen here cannot be blamed on the
    unnormalised score scalars, so it attributes the failure to the readout.
    """
    g = np.asarray(st.obs["partial_graph"])
    n_tiles = g.shape[0]
    out: List[ProbeState] = []
    for _ in range(n):
        h = g.copy()
        t = int(rng.integers(n_tiles))
        which = rng.integers(4)
        if which == 0:                                   # toggle road
            h[t, ROAD_SLICE] = 1.0 - h[t, ROAD_SLICE]
        elif which == 1:                                 # flip tile control
            blk = h[t, PLAYER_CTRL_SLICE].copy()
            h[t, PLAYER_CTRL_SLICE] = blk[::-1]
        elif which == 2:                                 # toggle a city bit
            c = h[t, CITY_SLICE]
            j = int(rng.integers(c.shape[0]))
            h[t, CITY_SLICE][j] = 1.0 - c[j]
        else:                                            # add/remove one unit
            sl = OWN_TYPE_SLICE if rng.integers(2) == 0 else OPP_TYPE_SLICE
            blk = h[t, sl]
            j = int(rng.integers(blk.shape[0]))
            blk[j] = 1.0 - blk[j]
        out.append(st.copy_with_graph(h))
    return out


def _scalar_perturbations(st: ProbeState) -> List[ProbeState]:
    """Small, realistic moves in each scalar channel, graph untouched.

    +5 is one uncovered tile, +20 one controlled tile, +100 one city - the
    granularity the score actually moves at.
    """
    s = np.asarray(st.obs["scalar_state"], dtype=np.float32)
    deltas = [
        ("stars +1",       np.array([1, 0, 0, 0, 0], dtype=np.float32)),
        ("own_score +5",   np.array([0, 0, 5, 0, 0], dtype=np.float32)),
        ("own_score +100", np.array([0, 0, 100, 0, 0], dtype=np.float32)),
        ("opp_score +100", np.array([0, 0, 0, 100, 0], dtype=np.float32)),
    ]
    return [st.copy_with_scalar(s + d) for _, d in deltas]


def perturbation_probe(
    policy,
    states:  Sequence[ProbeState],
    n_base:  int = 16,
    n_pert:  int = 24,
    stream:  int = V_TERM,
    seed:    int = 0,
) -> Dict[str, float]:
    """dV under minimal perturbations, reported separately per source."""
    rng = np.random.default_rng(seed)
    ref_std = float(values_over(policy, states, stream).std())

    d_graph: List[float] = []
    d_scalar: List[float] = []
    with torch.no_grad():
        for st in states[:n_base]:
            v0 = float(encode_state(policy, st)[2].reshape(-1)[stream])
            for p in _graph_perturbations(st, n_pert, rng):
                d_graph.append(abs(float(encode_state(policy, p)[2].reshape(-1)[stream]) - v0))
            for p in _scalar_perturbations(st):
                d_scalar.append(abs(float(encode_state(policy, p)[2].reshape(-1)[stream]) - v0))

    dg = np.asarray(d_graph)
    ds = np.asarray(d_scalar)
    nz = dg[dg > 0]
    return {
        "graph_dead_frac": float((dg == 0).mean()),
        "graph_mean_dV":   float(dg.mean()),
        "graph_max_dV":    float(dg.max()),
        "graph_jump_ratio": float(dg.max() / nz.mean()) if nz.size else float("inf"),
        "graph_L_rel":     float(dg.mean() / ref_std) if ref_std > 0 else float("inf"),
        "scalar_mean_dV":  float(ds.mean()),
        "scalar_max_dV":   float(ds.max()),
        "ref_std":         ref_std,
    }


# ══════════════════════════════════════════════════════════════════════════════
# P3 - activation saturation
# ══════════════════════════════════════════════════════════════════════════════

def value_path_linears(policy) -> Dict[str, nn.Module]:
    """The Linear layers whose OUTPUT is the next non-linearity's pre-activation."""
    mods: Dict[str, nn.Module] = {}
    enc = policy.encoder
    for name in ("input_proj", "scalar_proj", "fuse"):      # scalar_proj: pre-sc-48
        m = getattr(enc, name, None)
        if isinstance(m, nn.Linear):
            mods[f"encoder.{name}"] = m
    sc = getattr(enc, "scalar_enc", None)
    if sc is not None and isinstance(getattr(sc, "proj", None), nn.Linear):
        mods["encoder.scalar_enc"] = sc.proj
    head = getattr(policy.critic, "value_mlp", None) or getattr(policy.critic, "trunk", None)
    if head is not None:
        for i, m in enumerate(head):
            if isinstance(m, nn.Linear):
                mods[f"critic[{i}]"] = m
    return mods


_SATURATING = (nn.Tanh, nn.Sigmoid, nn.Softsign)


def saturating_activations(policy) -> Dict[str, nn.Module]:
    """Saturating activation modules in the value path, by qualified name."""
    out: Dict[str, nn.Module] = {}
    for root_name, root in (("encoder", policy.encoder), ("critic", policy.critic)):
        for name, m in root.named_modules():
            if isinstance(m, _SATURATING):
                out[f"{root_name}.{name} ({type(m).__name__})"] = m
    return out


def saturation_probe(policy, states, sat_threshold: float = 4.0) -> Dict[str, Dict[str, float]]:
    """How deep into a saturating non-linearity's tail its INPUT sits.

    |x| > 4 puts tanh at ~0.9993 with gradient ~1.3e-3: the unit is a constant
    sign, and nothing upstream of it learns.

    Hooks the activations themselves rather than every Linear, because a Linear
    followed by a LayerNorm has no saturation problem no matter how large its
    output is - measuring Linear outputs flags those as false positives. Linear
    outputs are still reported separately, marked [scale], for information.
    """
    stats: Dict[str, List[np.ndarray]] = {}
    handles = []

    def mk_pre_hook(name):
        def hook(_m, inp):
            stats.setdefault(name, []).append(
                inp[0].detach().reshape(-1).abs().cpu().numpy()
            )
        return hook

    def mk_post_hook(name):
        def hook(_m, _inp, out):
            stats.setdefault(name, []).append(
                out.detach().reshape(-1).abs().cpu().numpy()
            )
        return hook

    for name, m in saturating_activations(policy).items():
        handles.append(m.register_forward_pre_hook(mk_pre_hook(name)))
    for name, m in value_path_linears(policy).items():
        handles.append(m.register_forward_hook(mk_post_hook(f"[scale] {name}")))
    try:
        with torch.no_grad():
            for st in states:
                encode_state(policy, st)
    finally:
        for h in handles:
            h.remove()

    out: Dict[str, Dict[str, float]] = {}
    for name, chunks in stats.items():
        a = np.concatenate(chunks)
        out[name] = {
            "absmean":  float(a.mean()),
            "absmax":   float(a.max()),
            "sat_frac": float((a > sat_threshold).mean()),
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# P3b - scalar-vs-board branch balance
# ══════════════════════════════════════════════════════════════════════════════

def branch_balance(policy, states) -> Dict[str, float]:
    """||scalar branch|| vs ||board branch|| at the fusion point.

    These two are ADDED in the current encoder. If the ratio is far from 1 the
    smaller branch is simply not part of the critic's input, whatever the
    architecture diagram says.
    """
    enc = policy.encoder
    # Post-sc-48 the branch is `scalar_enc`; `scalar_proj` is the legacy name.
    scalar_branch = getattr(enc, "scalar_enc", None) or enc.scalar_proj
    board_n, scalar_n = [], []
    with torch.no_grad():
        for st in states:
            x = torch.as_tensor(
                np.asarray(st.obs["partial_graph"]), dtype=torch.float32, device=enc.device
            )
            x = enc.input_proj(x)
            x = enc._run_layers(x, enc._get_edge_index(st.Nx, st.Ny).to(enc.device))
            pooled = x.amax(dim=0, keepdim=True)
            s = torch.as_tensor(
                np.asarray(st.obs["scalar_state"]), dtype=torch.float32, device=enc.device
            ).reshape(1, -1)
            proj = scalar_branch(s)
            board_n.append(float(pooled.norm()))
            scalar_n.append(float(proj.norm()))
    b = float(np.mean(board_n))
    sc = float(np.mean(scalar_n))
    return {
        "board_norm":  b,
        "scalar_norm": sc,
        "ratio":       sc / b if b > 0 else float("inf"),
        "board_share": b / (b + sc) if (b + sc) > 0 else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# P5 - readout discriminability
# ══════════════════════════════════════════════════════════════════════════════

def readout_dispersion(policy, states) -> Dict[str, float]:
    """Does the pooled embedding vary with the board, or is it a DC offset?

    dispersion = mean per-channel std across states / mean |per-channel mean|.
    A value well below 1 means the readout is dominated by a board-independent
    constant, and the critic is reading noise on top of an offset.
    """
    E = embeddings_over(policy, states)         # (K, D)
    ch_std  = E.std(axis=0)
    ch_mean = np.abs(E.mean(axis=0))
    return {
        "dispersion":   float(ch_std.mean() / ch_mean.mean()) if ch_mean.mean() > 0 else float("inf"),
        "mean_ch_std":  float(ch_std.mean()),
        "mean_abs_ch":  float(ch_mean.mean()),
        "dead_channels": float((ch_std < 1e-6).sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# P8 - gradient flow
# ══════════════════════════════════════════════════════════════════════════════

def gradient_flow(policy, states, n: int = 8, stream: int = V_TERM) -> Dict[str, float]:
    """One backward pass from the value loss. Does anything reach the encoder?

    A saturated trunk shows up here as an encoder/head gradient ratio near zero
    long before it shows up as a bad training curve.
    """
    policy.zero_grad(set_to_none=True)
    loss = 0.0
    for st in states[:n]:
        _, _, v = encode_state(policy, st)
        target = torch.tensor(1.0, device=v.device)
        loss = loss + (v.reshape(-1)[stream] - target) ** 2
    (loss / max(n, 1)).backward()

    def gnorm(mod) -> float:
        tot = 0.0
        for p in mod.parameters():
            if p.grad is not None:
                tot += float(p.grad.norm()) ** 2
        return tot ** 0.5

    enc_g  = gnorm(policy.encoder)
    head_g = gnorm(policy.critic)
    n_none = sum(
        1 for p in list(policy.encoder.parameters()) + list(policy.critic.parameters())
        if p.grad is None
    )
    policy.zero_grad(set_to_none=True)
    return {
        "encoder_grad":  enc_g,
        "critic_grad":   head_g,
        "ratio":         enc_g / head_g if head_g > 0 else 0.0,
        "n_no_grad":     float(n_none),
    }
