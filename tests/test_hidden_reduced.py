"""Tests for the HiddenTileEstimator's reduced-feature output layout.

Three concerns are pinned here:
    1. The vectorised `_full_to_reduced_target` helper produces the
       expected one-hot vector for hand-built tile inputs.
    2. `HiddenTileEstimator.predict_proba` returns the right shape and
       per-group normalisation (softmax sums to 1, sigmoid bits in [0,1]).
    3. `EnvWrapper._get_obs()` carries both POVs, so a single obs dict
       drives `policy.estimate_hidden_dual()` end-to-end.
"""

import numpy as np
import pytest
import torch

from RL.models.main_modules import HiddenTileEstimator, _full_to_reduced_target
from game.enums import (
    NODE_FEAT_DIM,
    REDUCED_FEAT_DIM,
    REDUCED_TILE_TYPE_SLICE,
    REDUCED_ROAD_SLICE,
    REDUCED_OPP_CTRL_SLICE,
    REDUCED_CITY_SLICE,
    REDUCED_OPP_UNIT_SLICE,
    MAX_CITY_LEVEL_HIDDEN,
    N_CITY_TYPES,
    N_TILE_TYPES,
    N_UNIT_TYPES,
    TILE_TYPE_SLICE,
    ROAD_SLICE,
    _PLAYER_CTRL_START,
    _CITY_START,
    OPP_TYPE_SLICE,
)


# ── Helpers to build hand-crafted full-feature rows ──────────────────────────

def _empty_full_row(tile_type_idx: int = 0) -> np.ndarray:
    row = np.zeros(NODE_FEAT_DIM, dtype=np.float32)
    row[TILE_TYPE_SLICE.start + tile_type_idx] = 1.0
    return row


def _set_opp_ctrl(row: np.ndarray) -> None:
    row[_PLAYER_CTRL_START + 1] = 1.0


def _set_village(row: np.ndarray) -> None:
    row[_CITY_START] = 1.0


def _set_opp_city_level(row: np.ndarray, level_idx: int) -> None:
    """level_idx is the CityType IntEnum value (1..N_CITY_TYPES)."""
    base = _CITY_START + 1 + N_CITY_TYPES + (level_idx - 1)
    row[base] = 1.0


def _set_opp_unit(row: np.ndarray, unit_idx: int) -> None:
    row[OPP_TYPE_SLICE.start + unit_idx] = 1.0


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_reduced_dims_consistent():
    """Slice arithmetic must agree with REDUCED_FEAT_DIM."""
    assert REDUCED_TILE_TYPE_SLICE.start == 0
    assert REDUCED_TILE_TYPE_SLICE.stop  == N_TILE_TYPES
    assert REDUCED_ROAD_SLICE.stop       == REDUCED_OPP_CTRL_SLICE.start
    assert REDUCED_OPP_CTRL_SLICE.stop   == REDUCED_CITY_SLICE.start
    assert REDUCED_CITY_SLICE.stop       == REDUCED_OPP_UNIT_SLICE.start
    assert REDUCED_OPP_UNIT_SLICE.stop   == REDUCED_FEAT_DIM
    # city = None + Village + L1..L_cap; opp_unit = None + N_UNIT_TYPES.
    cap = min(N_CITY_TYPES, MAX_CITY_LEVEL_HIDDEN)
    assert (REDUCED_CITY_SLICE.stop - REDUCED_CITY_SLICE.start) == 2 + cap
    assert (REDUCED_OPP_UNIT_SLICE.stop - REDUCED_OPP_UNIT_SLICE.start) \
        == 1 + N_UNIT_TYPES


def test_transform_empty_tile():
    full = _empty_full_row()
    out  = _full_to_reduced_target(torch.from_numpy(full).unsqueeze(0))[0].numpy()

    # tile_type passes through.
    assert out[REDUCED_TILE_TYPE_SLICE.start + 0] == pytest.approx(1.0)
    # opp_ctrl bit off.
    assert out[REDUCED_OPP_CTRL_SLICE.start] == pytest.approx(0.0)
    # city = None (idx 0).
    assert int(out[REDUCED_CITY_SLICE].argmax()) == 0
    # opp_unit = None (idx 0).
    assert int(out[REDUCED_OPP_UNIT_SLICE].argmax()) == 0


def test_transform_village_only():
    full = _empty_full_row()
    _set_village(full)
    out = _full_to_reduced_target(torch.from_numpy(full).unsqueeze(0))[0].numpy()
    # city class 1 = Village.
    assert int(out[REDUCED_CITY_SLICE].argmax()) == 1
    # opp_unit still None.
    assert int(out[REDUCED_OPP_UNIT_SLICE].argmax()) == 0


def test_transform_opp_city_and_unit():
    full = _empty_full_row(tile_type_idx=0)
    _set_opp_ctrl(full)
    _set_opp_city_level(full, level_idx=3)         # CityType.lvl2_explorer
    _set_opp_unit(full, unit_idx=1)                # UnitType.Rider
    out = _full_to_reduced_target(torch.from_numpy(full).unsqueeze(0))[0].numpy()

    assert out[REDUCED_OPP_CTRL_SLICE.start] == pytest.approx(1.0)
    # City class index = 1 (Village offset) + level (3) → 4.
    assert int(out[REDUCED_CITY_SLICE].argmax()) == 1 + 3
    # Opp unit class = 1 (None offset) + unit_idx (1) → 2.
    assert int(out[REDUCED_OPP_UNIT_SLICE].argmax()) == 1 + 1


def test_transform_opp_city_above_cap():
    full = _empty_full_row()
    _set_opp_ctrl(full)
    # Pick a level strictly above MAX_CITY_LEVEL_HIDDEN if the enum allows it.
    high = min(N_CITY_TYPES, MAX_CITY_LEVEL_HIDDEN + 5)
    if high <= MAX_CITY_LEVEL_HIDDEN:
        pytest.skip("N_CITY_TYPES does not exceed MAX_CITY_LEVEL_HIDDEN.")
    _set_opp_city_level(full, level_idx=high)
    out = _full_to_reduced_target(torch.from_numpy(full).unsqueeze(0))[0].numpy()
    # Capped at L_cap → city idx 1 + MAX_CITY_LEVEL_HIDDEN.
    assert int(out[REDUCED_CITY_SLICE].argmax()) == 1 + MAX_CITY_LEVEL_HIDDEN


def test_transform_batch_mixed():
    """Several different rows in one tensor — the vectorised transform must
    produce identical outputs to a per-row loop."""
    rows = [
        _empty_full_row(),                              # empty
        (lambda: (lambda r: (_set_village(r) or r))(_empty_full_row()))(),
        (lambda: (lambda r: (_set_opp_ctrl(r) or r))(_empty_full_row(2)))(),
    ]
    # Add an opp city + unit row.
    r4 = _empty_full_row(1); _set_opp_ctrl(r4); _set_opp_city_level(r4, 2); _set_opp_unit(r4, 4)
    rows.append(r4)

    full_batch = torch.from_numpy(np.stack(rows, axis=0))
    out_batch  = _full_to_reduced_target(full_batch).numpy()

    for i, r in enumerate(rows):
        single = _full_to_reduced_target(torch.from_numpy(r).unsqueeze(0))[0].numpy()
        np.testing.assert_allclose(out_batch[i], single)


def test_predict_proba_shapes_and_normalisation():
    torch.manual_seed(0)
    est = HiddenTileEstimator(node_dim=32)
    emb = torch.randn(7, 32)

    raw   = est(emb)
    probs = est.predict_proba(emb)
    assert raw.shape   == (7, REDUCED_FEAT_DIM)
    assert probs.shape == (7, REDUCED_FEAT_DIM)

    # Softmax groups sum to 1 along the last axis.
    for sl in (REDUCED_TILE_TYPE_SLICE, REDUCED_CITY_SLICE, REDUCED_OPP_UNIT_SLICE):
        sums = probs[:, sl].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    # Sigmoid bits in [0, 1].
    for sl in (REDUCED_ROAD_SLICE, REDUCED_OPP_CTRL_SLICE):
        block = probs[:, sl]
        assert (block >= 0).all() and (block <= 1).all()


def test_loss_finite_and_grads_flow():
    """The loss should be finite, scalar, and produce non-zero gradient
    on the predictor when at least one tile is hidden."""
    torch.manual_seed(0)
    est = HiddenTileEstimator(node_dim=16)
    emb = torch.randn(5, 16, requires_grad=False)
    pred = est(emb)

    # Build a synthetic full-feature target: one row per tile.
    target_rows = []
    for k in range(5):
        r = _empty_full_row(tile_type_idx=k % N_TILE_TYPES)
        if k % 2 == 0:
            _set_opp_ctrl(r)
            _set_opp_unit(r, unit_idx=k % N_UNIT_TYPES)
        target_rows.append(r)
    target = torch.from_numpy(np.stack(target_rows, axis=0))

    hidden_mask = torch.tensor([True, True, False, True, True])
    loss = est.loss(pred, target, hidden_mask)
    assert torch.isfinite(loss)
    assert loss.dim() == 0

    loss.backward()
    grads = [p.grad for p in est.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any((g.abs().sum() > 0).item() for g in grads)


def test_loss_zero_when_no_hidden():
    est = HiddenTileEstimator(node_dim=8)
    emb = torch.randn(3, 8)
    pred = est(emb)
    target = torch.from_numpy(np.stack([_empty_full_row() for _ in range(3)], axis=0))
    mask = torch.zeros(3, dtype=torch.bool)
    loss = est.loss(pred, target, mask)
    assert loss.item() == 0.0


def test_obs_carries_both_povs(fresh_env):
    obs = fresh_env._get_obs()
    for k in ("opp_partial_graph", "opp_scalar_state", "opp_uncovered_tile_ids"):
        assert k in obs, f"missing key {k}"

    assert obs["opp_partial_graph"].shape == obs["partial_graph"].shape
    assert obs["opp_scalar_state"].shape  == obs["scalar_state"].shape
    # Same dtype / 1-D ID array on the opp side.
    assert obs["opp_uncovered_tile_ids"].dtype == obs["uncovered_tile_ids"].dtype
    assert obs["opp_uncovered_tile_ids"].ndim  == 1


def test_estimate_hidden_dual_end_to_end(fresh_env):
    """Build a random-init policy and run the full encoder + estimator path
    on both POVs in one obs dict."""
    from RL.models.policy import PolicyNetwork

    class _Cfg: pass
    policy = PolicyNetwork(_Cfg())

    obs = fresh_env._get_obs()
    est_a, est_b = policy.estimate_hidden_dual(obs)

    N_tiles = obs["partial_graph"].shape[0]
    assert est_a.shape == (N_tiles, REDUCED_FEAT_DIM)
    assert est_b.shape == (N_tiles, REDUCED_FEAT_DIM)

    # Softmax groups sum to 1 row-wise on both outputs.
    for est in (est_a, est_b):
        for sl in (REDUCED_TILE_TYPE_SLICE, REDUCED_CITY_SLICE, REDUCED_OPP_UNIT_SLICE):
            np.testing.assert_allclose(
                est[:, sl].sum(axis=-1), np.ones(N_tiles), atol=1e-4,
            )
