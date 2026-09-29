"""
tests/test_run_report.py
──────────────────────────────────────────────────────────────────────────────
The sc-80 run report (eval/run_report). Every test builds synthetic segments
in tmp_path in the real file formats: the train.py .log banner, the metrics
CSV, a summary.csv written the way scenarios/eval/bank.py writes it, and the
update_NNNNN/<Scenario>.png folders. Nothing reads RL/logs.
"""

import ast
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from eval.run_report import gifs, plot, specs
from eval.run_report.__main__ import main
from eval.run_report.loader import load_run, ranges, resolve_segments, update_grid

REPO = Path(__file__).resolve().parents[1]

V2_ONLY_MISSING = {"conquest_rate", "decisions_per_turn", "endturn_voluntary",
                   "ev_term", "ev_dense", "n_terminal_events", "frac_ret_out_of_support"}
METRIC_FIELDS = [
    "update", "wall_time_s", "t_collect_s", "t_ppo_s", "n_games", "n_active_wins",
    "active_win_rate", "n_conquest", "conquest_rate", "decisions_per_turn",
    "endturn_voluntary", "ev_term", "ev_dense", "n_terminal_events",
    "frac_ret_out_of_support", "avg_ep_len", "avg_active_reward", "est_loss",
    "p_loss", "v_loss", "entropy",
]


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic segments
# ══════════════════════════════════════════════════════════════════════════════

def _scenario_metrics(name: str, update: int) -> dict:
    """Plausible values for every column the specs use."""
    rng = np.random.default_rng(update)
    base = {"n_samples": 20, "t_sec": 1.0, "v_term": 1.0, "v_dense": 2.0, "v_term_std": 0.1}
    per = {
        "Get_defender_and_wall": {"success_both": 0.0, "n_any_unit_created": 15,
                                  "n_defenders_created": 2, "n_wall_chosen": 0},
        "Simple_dash_dancing2": {"uncovered_delta_mean": 3 + rng.random(), "uncovered_delta_std": 1.0},
        "Rider_leapfrogging": {"uncovered_delta_mean": 6 + rng.random(), "uncovered_delta_std": 2.0},
        "Knight_choice_no_village": {"choice_left_rate": 0.3, "choice_right_rate": 0.5},
        "road_for_kill": {"success_rate": rng.random(), "avg_decisions_taken": 4.5, "n_decisions_max": 8},
        "Escaping_riders2": {"unsafe_rider_count_mean": 2.5, "unsafe_rider_count_min": 2,
                             "unsafe_rider_count_max": 3, "all_safe_rate": 0.0},
        "Upgrade_city_order2": {"both_upgraded_rate": 0.7, "avg_decisions_taken": 1.8,
                                "avg_cities_upgraded": 1.6},
        "Giant_Houdini": {"all_three_rate": 0.0, "n_rider_on_118": 3, "n_city_upgraded": 20,
                          "n_superunit_chosen": 15},
        "Defender_ZoC": {"occupied_count_mean": 0.3, "occupied_count_min": 0,
                         "occupied_count_max": 1, "two_of_three_rate": 0.05},
    }
    return {**base, **per.get(name, {"brand_new_rate": 0.5})}


def _write_summary(path: Path, rows: list[dict]) -> None:
    """Write like ScenarioBank.append_summary_csv: update, scenario, then the
    sorted union of all metric keys, blanks where a scenario has no value."""
    cols = sorted({k for r in rows for k in r} - {"update", "scenario"})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["update", "scenario"] + cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in ["update", "scenario"] + cols})


def write_segment(root: Path, tag: str, updates, *, start=None, ckpt=None, scratch=False,
                  config=None, sentinel=0.0, v2=False, scenarios=("Get_defender_and_wall",),
                  scen_updates=None, pngs=(), png_sizes=None, log=True, metrics=True,
                  summary=True, truncate=False, errors=()) -> Path:
    """Write one segment `tag` into `root`. `sentinel` lands in p_loss and in
    every scenario row's t_sec, so a test can tell which segment a row came from."""
    root.mkdir(parents=True, exist_ok=True)
    updates = list(updates)
    cfg = {"envs": 8, "batch": "4,096", "mb": 256, "epochs": 2, "beta": 1.0, "lr": 0.0008,
           "refresh": 10, **(config or {})}
    if log:
        ts = "2026-09-22 23:15:54"
        lines = [f"{ts}  Run tag : {tag}"]
        if ckpt is not None:
            lines.append(f"{ts}  Checkpoint loaded (policy + 2 optimisers + 2 scalers). "
                         f"update_in_ckpt={ckpt}")
        if scratch:
            lines.append(f"{ts}  No pretrained checkpoint — training from scratch.")
        lines += [
            f"{ts}    Workers × envs    : 4 × 2  =  {cfg['envs']} parallel envs",
            f"{ts}    Full batch        : {cfg['batch']} samples (512 steps × 8 envs)",
            f"{ts}    Train fraction    : 1.00  →  4,096 samples / epoch  (16 minibatches × {cfg['mb']})",
            f"{ts}    PPO epochs        : {cfg['epochs']}  |  Updates: 1000",
            f"{ts}    clip_eps / vf / ent: 0.2 / 0.5 / 0.005",
            f"{ts}    Reward            : dense=True (beta={cfg['beta']}, scale=0.0367)  conquest=2.0",
            f"{ts}    Self-play         : active seat P0 vs frozen (refresh every {cfg['refresh']} updates)",
            f"{ts}    LR                : {cfg['lr']}",
            f"{ts}    Estimator LR      : 0.0003",
            f"{ts}    Start update      : {updates[0] if start is None else start}",
            f"{ts}  ",
            f"[update {updates[0] if updates else 0:04d}] weights dispatched in 1.0s",
        ]
        (root / f"{tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if metrics:
        fields = [f for f in METRIC_FIELDS if not (v2 and f in V2_ONLY_MISSING)]
        with open(root / f"{tag}_metrics.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for u in updates:
                row = {"update": u, "wall_time_s": 200.0, "t_collect_s": 30.0, "t_ppo_s": 150.0,
                       "n_games": 20, "n_active_wins": 12, "active_win_rate": 0.6,
                       "n_conquest": 5, "conquest_rate": 0.25, "decisions_per_turn": 10.0,
                       "endturn_voluntary": 40, "ev_term": 0.5, "ev_dense": 0.9,
                       "n_terminal_events": 12, "frac_ret_out_of_support": 0.0,
                       "avg_ep_len": 300.0, "avg_active_reward": 5.0, "est_loss": 2.0,
                       "p_loss": sentinel, "v_loss": 1.8, "entropy": 2.5}
                writer.writerow({k: row[k] for k in fields})
        if truncate:
            path = root / f"{tag}_metrics.csv"
            path.write_bytes(path.read_bytes() + b"999,1.0,2")          # a row still being written
    sdir = root / f"{tag}_scenarios"
    scen_updates = updates if scen_updates is None else list(scen_updates)
    if summary or pngs:
        sdir.mkdir(exist_ok=True)
    if summary:
        rows = []
        for u in scen_updates:
            for name in scenarios:
                if (u, name) in errors:
                    rows.append({"update": u, "scenario": name, "_error": "ValueError: boom"})
                else:
                    rows.append({"update": u, "scenario": name,
                                 **_scenario_metrics(name, u), "t_sec": sentinel})
        _write_summary(sdir / "summary.csv", rows)
    for u in scen_updates:
        for name in pngs:
            (sdir / f"update_{u:05d}").mkdir(exist_ok=True)
            size = (png_sizes or {}).get(u, (40, 30))
            # A colour per update: Pillow merges identical consecutive GIF frames.
            Image.new("RGB", size, ((37 * u) % 256, 100, 50)).save(
                sdir / f"update_{u:05d}" / f"{name}.png")
    return root


def _load(root: Path, *tags):
    return load_run(resolve_segments(list(tags), root), "test")


def _texts(run) -> str:
    return "\n".join(str(i) for i in run.issues)


# ══════════════════════════════════════════════════════════════════════════════
# Segment resolution and log parsing
# ══════════════════════════════════════════════════════════════════════════════

def test_segments_resolve_from_tags_and_paths_in_timestamp_order(tmp_path):
    loose, archive = tmp_path / "logs", tmp_path / "logs" / "archive"
    write_segment(loose, "run_20260102_000000", range(3, 6), ckpt=2)
    write_segment(archive, "run_20260101_000000", range(0, 3), scratch=True)
    write_segment(archive, "run_20260103_000000", range(6, 8), ckpt=5)
    segs = resolve_segments(
        ["run_20260102_000000",                                    # bare tag, from log_root
         str(archive / "run_20260103_000000_metrics.csv"),          # one of its outputs
         str(archive / "run_20260101_000000")],                     # a path prefix
        loose)
    assert [s.tag for s in segs] == ["run_20260101_000000", "run_20260102_000000",
                                     "run_20260103_000000"]
    assert all(s.log_file and s.metrics_csv and s.scenarios_dir for s in segs)


def test_segment_resolution_rejects_bad_input(tmp_path):
    write_segment(tmp_path, "run_20260101_000000", range(3))
    with pytest.raises(ValueError):
        resolve_segments(["not_a_segment"], tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_segments(["run_20990101_000000"], tmp_path)


def test_log_banner_parsing(tmp_path):
    write_segment(tmp_path, "run_20260101_000000", range(0, 3), scratch=True)
    write_segment(tmp_path, "run_20260102_000000", range(106, 110), ckpt=105,
                  config={"envs": 32, "batch": "32,768", "mb": 512, "beta": 0.5, "lr": 0.0003})
    first, second = resolve_segments(["run_20260101_000000", "run_20260102_000000"], tmp_path)
    assert (first.start_update, first.resumed_from, first.from_scratch) == (0, None, True)
    assert (second.start_update, second.resumed_from, second.from_scratch) == (106, 105, False)
    assert second.config == {"envs": "32", "batch": "32768", "minibatch": "512", "epochs": "2",
                             "ent_coef": "0.005", "beta": "0.5", "refresh": "10", "lr": "0.0003"}
    run = load_run([first, second], "test")
    assert run.boundaries[0].update == 106 and run.boundaries[0].label == "S2"
    assert run.boundaries[0].diff["batch"] == ("4096", "32768")
    assert "lr" in run.boundaries[0].diff and "epochs" not in run.boundaries[0].diff


def test_unknown_checkpoint_marker(tmp_path):
    write_segment(tmp_path, "run_20260101_000000", range(0, 3), scratch=True)
    write_segment(tmp_path, "run_20260102_000000", range(3, 5), ckpt="?")
    seg = resolve_segments(["run_20260102_000000"], tmp_path)[0]
    assert seg.resumed_from is None and seg.start_update == 3
    run = _load(tmp_path, "run_20260101_000000", "run_20260102_000000")
    assert "resume checkpoint is not recorded" in _texts(run)


# ══════════════════════════════════════════════════════════════════════════════
# Lineage
# ══════════════════════════════════════════════════════════════════════════════

def test_overlap_is_resolved_in_favour_of_the_later_segment(tmp_path):
    """The real case: segment 1 logged 0..106, segment 2 resumed from 105."""
    write_segment(tmp_path, "run_20260101_000000", range(0, 107), scratch=True, sentinel=1.0,
                  pngs=("Get_defender_and_wall",))
    write_segment(tmp_path, "run_20260102_000000", range(106, 110), ckpt=105, sentinel=2.0,
                  pngs=("Get_defender_and_wall",))
    run = _load(tmp_path, "run_20260101_000000", "run_20260102_000000")
    m, s = run.metrics, run.scenarios
    assert m["update"].is_unique and m["update"].tolist() == list(range(110))
    assert m.loc[m["update"] == 106, "p_loss"].item() == 2.0
    assert m.loc[m["update"] == 105, "p_loss"].item() == 1.0
    assert s.loc[s["update"] == 106, "t_sec"].tolist() == [2.0]
    assert run.frames["Get_defender_and_wall"][106].parent.parent.name == "run_20260102_000000_scenarios"
    assert run.spans == {"run_20260101_000000": (0, 105), "run_20260102_000000": (106, 109)}
    assert "update 106 superseded by run_20260102_000000" in _texts(run)


def test_orphaned_tail_of_any_length_is_cut(tmp_path):
    write_segment(tmp_path, "run_20260101_000000", range(0, 111), scratch=True,
                  pngs=("Get_defender_and_wall",))
    write_segment(tmp_path, "run_20260102_000000", range(106, 108), ckpt=105,
                  pngs=("Get_defender_and_wall",))
    run = _load(tmp_path, "run_20260101_000000", "run_20260102_000000")
    assert run.metrics["update"].max() == 107
    assert run.scenarios["update"].max() == 107
    assert max(run.frames["Get_defender_and_wall"]) == 107


def test_segment_without_rows_still_cuts(tmp_path):
    """A resume that crashed in its first update wrote only its log."""
    write_segment(tmp_path, "run_20260101_000000", range(0, 11), scratch=True)
    write_segment(tmp_path, "run_20260102_000000", [6], ckpt=5, metrics=False, summary=False)
    run = _load(tmp_path, "run_20260101_000000", "run_20260102_000000")
    assert run.metrics["update"].max() == 5
    assert "no metrics CSV, scenarios folder" in _texts(run)


def test_first_segment_from_a_checkpoint_is_only_info(tmp_path):
    write_segment(tmp_path, "run_20260101_000000", range(0, 11), ckpt=99, start=0)
    write_segment(tmp_path, "run_20260102_000000", range(11, 15), ckpt=10)
    run = _load(tmp_path, "run_20260101_000000", "run_20260102_000000")
    assert [i.level for i in run.issues] == ["info"]
    assert "starts from checkpoint update 99" in _texts(run)


def test_suspicious_lists_warn_but_never_block(tmp_path):
    write_segment(tmp_path, "run_20260101_000000", range(0, 11), scratch=True)
    write_segment(tmp_path, "run_20260102_000000", range(21, 25), ckpt=20)     # 11..20 missing
    write_segment(tmp_path, "run_20260103_000000", range(0, 3), scratch=True)  # a new run
    run = _load(tmp_path, "run_20260101_000000", "run_20260102_000000", "run_20260103_000000")
    text = _texts(run)
    assert "is a segment missing from the list?" in text
    assert "started without a checkpoint" in text
    assert "run_20260102_000000 is completely superseded" in text
    assert run.metrics["update"].tolist() == [0, 1, 2]


def test_gaps_are_reported_and_never_bridged(tmp_path):
    write_segment(tmp_path, "run_20260101_000000", range(0, 6), scratch=True)
    write_segment(tmp_path, "run_20260102_000000", range(9, 12), ckpt=8)
    run = _load(tmp_path, "run_20260101_000000", "run_20260102_000000")
    assert "metrics: no rows for updates 6-8" in _texts(run)
    data = plot.on_grid(run.metrics)
    assert data.index.tolist() == list(range(12))
    assert data.loc[6:8, "p_loss"].isna().all()
    rolled = plot.rolling(data["entropy"], 10)
    assert rolled.loc[6:8].isna().all() and rolled.loc[[5, 9]].notna().all()


# ══════════════════════════════════════════════════════════════════════════════
# Robust reading
# ══════════════════════════════════════════════════════════════════════════════

def test_header_only_csv_does_not_turn_columns_into_object(tmp_path):
    """pandas 3 regression: concat of a header-only frame used to make every
    column object dtype, including `update`."""
    write_segment(tmp_path, "run_20260101_000000", [], scratch=True, start=0, summary=False)
    write_segment(tmp_path, "run_20260102_000000", range(0, 4), ckpt=None, start=0)
    run = _load(tmp_path, "run_20260101_000000", "run_20260102_000000")
    assert str(run.metrics["update"].dtype) == "int64"
    assert str(run.metrics["p_loss"].dtype) == "float64"
    assert str(run.scenarios["n_samples"].dtype) == "float64"


def test_partial_last_line_and_empty_summary(tmp_path):
    write_segment(tmp_path, "run_20260101_000000", range(0, 5), scratch=True, truncate=True)
    (tmp_path / "run_20260101_000000_scenarios" / "summary.csv").write_bytes(b"")
    run = _load(tmp_path, "run_20260101_000000")
    assert run.metrics["update"].tolist() == [0, 1, 2, 3, 4]
    text = _texts(run)
    assert "dropped a partially written last line" in text
    assert "summary.csv: file is empty" in text
    assert run.scenarios.empty


def test_crashed_scenario_rows_are_excluded_and_counted(tmp_path):
    write_segment(tmp_path, "run_20260101_000000", range(0, 5), scratch=True,
                  scenarios=("Get_defender_and_wall", "road_for_kill"),
                  errors={(1, "road_for_kill"), (3, "road_for_kill")})
    run = _load(tmp_path, "run_20260101_000000")
    s = run.scenarios
    assert "_error" not in s.columns
    assert sorted(s.loc[s["scenario"] == "road_for_kill", "update"]) == [0, 2, 4]
    assert str(s["success_rate"].dtype) == "float64"
    assert "scenario road_for_kill crashed in 2 update(s)" in _texts(run)


def test_duplicate_rows_within_a_segment_keep_the_last(tmp_path):
    write_segment(tmp_path, "run_20260101_000000", [0, 1, 1, 2], scratch=True)
    run = _load(tmp_path, "run_20260101_000000")
    assert run.metrics["update"].tolist() == [0, 1, 2]
    assert "duplicate rows, kept the last" in _texts(run)


def test_helpers():
    assert ranges([3, 4, 5, 9, 11, 12]) == "3-5, 9, 11-12"
    assert update_grid([0, 5, 10, 20]).tolist() == [0, 5, 10, 15, 20]
    assert update_grid([]).size == 0


# ══════════════════════════════════════════════════════════════════════════════
# Cleaning, rolling means, bar axis
# ══════════════════════════════════════════════════════════════════════════════

def test_clean_metrics(tmp_path):
    write_segment(tmp_path, "run_20260101_000000", range(0, 6), scratch=True, v2=True)
    run = _load(tmp_path, "run_20260101_000000")
    df, notes = specs.clean_metrics(run.metrics)
    assert df["conquest_rate"].eq(0.25).all()
    assert any("conquest_rate derived" in n for n in notes)

    raw = pd.DataFrame({"update": [0, 1, 2], "n_games": [0, 10, 10], "active_win_rate": [0.0, 0.5, 0.6],
                        "ev_term": [0.9, 0.8, 0.7], "n_terminal_events": [2, 4, 12],
                        "endturn_voluntary": [5, 30, 60], "avg_ep_len": [0.0, 300.0, 300.0],
                        "conquest_rate": [0.0, 0.1, 0.2]})
    df, notes = specs.clean_metrics(raw)
    assert np.isnan(df.loc[0, "active_win_rate"]) and df.loc[1, "active_win_rate"] == 0.5
    assert df["ev_term"].isna().tolist() == [True, True, False]
    assert np.isnan(df.loc[0, "endturn_vol_share"])
    assert df.loc[2, "endturn_vol_share"] == pytest.approx(60 / 3000)
    assert any("ev_term masked at 2 updates" in n for n in notes)


def test_pooled_rolling_mean_and_regimes():
    idx = pd.Index(range(4), name="update")
    rate = pd.Series([1.0, 0.0, 1.0, 0.0], index=idx)
    games = pd.Series([1.0, 9.0, 1.0, 9.0], index=idx)
    pooled = plot.rolling(rate, 3, weights=games)
    assert pooled.loc[1] == pytest.approx((1 * 1 + 0 * 9 + 1 * 1) / 11)       # Σwins / Σgames
    stepped = plot.rolling(pd.Series([0.0, 0.0, 10.0, 10.0], index=idx), 3, regimes=[2])
    assert stepped.tolist() == [0.0, 0.0, 10.0, 10.0]                         # no ramp at the step


def test_bar_axis_cap_follows_the_data():
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "n_samples": [20, 20],
                         "n_decisions_max": [8, 8]})
    assert plot.bars_cap(specs.SCENARIOS["Get_defender_and_wall"].panels[0],
                         data.rename(columns={"a": "n_any_unit_created"})) == 60.0
    assert plot.bars_cap(specs.SCENARIOS["road_for_kill"].panels[0], data) == 8.0
    assert plot.bars_cap(plot.Panel(bars={"a": "", "b": ""}), data) == pytest.approx(6 * 1.1)


def test_critic_constants():
    assert specs.CE_FLOOR[0] == pytest.approx(0.507, abs=1e-3)
    assert specs.CE_FLOOR[1] == pytest.approx(1.200, abs=1e-3)
    assert specs.CE_UNIFORM == pytest.approx(math.log(51))


# ══════════════════════════════════════════════════════════════════════════════
# Rendering
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def full_run(tmp_path):
    names = tuple(specs.SCENARIOS) + ("Brand_new",)
    write_segment(tmp_path, "run_20260101_000000", range(0, 30), scratch=True, scenarios=names)
    write_segment(tmp_path, "run_20260102_000000", range(30, 60), ckpt=29, scenarios=names,
                  config={"batch": "32,768", "lr": 0.0003})
    return _load(tmp_path, "run_20260101_000000", "run_20260102_000000")


@pytest.mark.parametrize("name", list(specs.SCENARIOS))
def test_every_scenario_figure_renders(full_run, name, tmp_path):
    spec = specs.SCENARIOS[name]
    with plt.style.context(plot.STYLE):
        fig = plot.render_scenario(name, spec, full_run)
        assert len(fig.axes) == len(spec.panels)          # secondary axes are children
        fig.savefig(tmp_path / f"{name}.png", dpi=40)
        plt.close(fig)
    assert (tmp_path / f"{name}.png").stat().st_size > 0


@pytest.mark.parametrize("v2", [False, True])
def test_training_grid_renders_for_v3_and_v2(tmp_path, v2):
    write_segment(tmp_path, "run_20260101_000000", range(0, 40), scratch=True, v2=v2)
    run = _load(tmp_path, "run_20260101_000000")
    metrics, notes = specs.clean_metrics(run.metrics)
    with plt.style.context(plot.STYLE):
        fig = plot.render_training(run, metrics, specs.TRAINING, flags=notes)
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        fig.savefig(tmp_path / "grid.png", dpi=40)
        plt.close(fig)
    assert ("not logged in this run" in texts) is v2          # ev_*, decisions_per_turn in V2


def test_unregistered_scenarios_get_a_generic_figure_and_a_flag(full_run):
    names = [name for name, _ in specs.scenario_figures(full_run)]
    assert names[: len(specs.SCENARIOS)] == list(specs.SCENARIOS)
    assert names[-1] == "Brand_new"
    metrics, _ = specs.clean_metrics(full_run.metrics)
    flags = specs.sanity_flags(full_run, metrics)
    assert "Brand_new: no spec in specs.py, plotted with a generic figure" in flags
    assert "Get_defender_and_wall.success_both is constant 0 over 60 updates" in flags


def test_gif_pads_uneven_frames(tmp_path):
    write_segment(tmp_path, "run_20260101_000000", range(0, 5), scratch=True, summary=False,
                  pngs=("Dont_attack",), png_sizes={2: (44, 31)})
    run = _load(tmp_path, "run_20260101_000000")
    out = tmp_path / "Dont_attack.gif"
    assert gifs.make_gif(run.frames["Dont_attack"], out) == 5
    assert gifs.make_gif(run.frames["Dont_attack"], tmp_path / "thin.gif", every=2) == 3
    with Image.open(out) as gif:
        assert gif.n_frames == 5 and gif.size == (44, 31)
    assert gifs.make_gif(run.frames["Dont_attack"], tmp_path / "half.gif", scale=0.5) == 5
    with Image.open(tmp_path / "half.gif") as gif:
        assert gif.size == (22, 16)


def test_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, "DPI", 40)
    logs = tmp_path / "logs"
    names = ("Get_defender_and_wall", "Escaping_riders2", "Dont_attack")
    write_segment(logs, "run_20260101_000000", range(0, 12), scratch=True, scenarios=names,
                  pngs=("Dont_attack",))
    write_segment(logs, "run_20260102_000000", range(11, 20), ckpt=10, scenarios=names,
                  pngs=("Dont_attack",))
    assert main(["--name", "demo", "--log-root", str(logs),
                 "run_20260102_000000", "run_20260101_000000"]) == 0
    out = logs / "reports" / "demo"
    for rel in ("training_summary.png", "scenarios/Get_defender_and_wall.png",
                "scenarios/Escaping_riders2.png", "handbook.pdf", "segments.txt"):
        assert (out / rel).stat().st_size > 0, rel
    assert not (out / "scenarios" / "Dont_attack.png").exists()        # GIF-only scenario
    with Image.open(out / "gifs" / "Dont_attack.gif") as gif:
        assert gif.n_frames == 20
    text = (out / "segments.txt").read_text(encoding="utf-8")
    assert "S2  run_20260102_000000  11-19" in text
    assert "update 11 superseded by run_20260102_000000" in text


# ══════════════════════════════════════════════════════════════════════════════
# Contracts with the code that writes the CSVs
# ══════════════════════════════════════════════════════════════════════════════

BANK_KEYS = {"v_term", "v_dense", "v_term_std", "t_sec"}       # added by scenarios/eval/bank.py


@pytest.mark.parametrize("name", list(specs.SCENARIOS))
def test_scenario_spec_columns_are_written_by_the_config(name):
    source = (REPO / "scenarios" / "configs" / f"{name}.py").read_text(encoding="utf-8")
    written = set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', source)) | BANK_KEYS
    used = {c for p in specs.SCENARIOS[name].panels for c in p.columns()}
    assert used <= written, f"{name}: {sorted(used - written)} not written by the config"


def test_training_spec_columns_are_written_by_train_py():
    tree = ast.parse((REPO / "RL" / "train.py").read_text(encoding="utf-8"))
    fields = next(ast.literal_eval(node.value) for node in ast.walk(tree)
                  if isinstance(node, ast.Assign)
                  and any(getattr(t, "id", None) == "_CSV_FIELDS" for t in node.targets))
    used = {c for row in specs.TRAINING for p in row for c in p.columns()}
    assert used <= set(fields) | set(specs.DERIVED), sorted(used - set(fields) - set(specs.DERIVED))
