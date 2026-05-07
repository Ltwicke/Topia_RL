"""
generate_training_visualizations.py

Generate GIF animations and scalar evolution plots from Version 2.0 training logs.
Run from the project root: python eval/generate_training_visualizations.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_ROOT = Path("RL/logs")
RUN_DIRS = [
    "run_20260506_015619_scenarios",
    "run_20260506_034854_scenarios",
    "run_20260506_101428_scenarios",
    "run_20260506_140856_scenarios",
]
OUT_DIR = Path("eval/training_progress_v2")
GIF_FPS = 10

# Update indices at which a new run starts (used for boundary markers in plots)
RUN_BOUNDARIES = [5, 19, 32]
RUN_LABELS = ["Run 1", "Run 2", "Run 3", "Run 4"]

ROLLING_WINDOW = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_summary_csvs(run_dirs: list[str]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        csv_path = LOG_ROOT / run_dir / "summary.csv"
        if not csv_path.exists():
            print(f"  [WARN] CSV not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["scenario", "update"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def collect_pngs(run_dirs: list[str], scenario_name: str) -> list[Path]:
    """Return sorted list of PNG paths for a given scenario across all run dirs."""
    results: list[tuple[int, Path]] = []
    for run_dir in run_dirs:
        base = LOG_ROOT / run_dir
        for update_dir in base.glob("update_*"):
            png = update_dir / f"{scenario_name}.png"
            if png.exists():
                m = re.search(r"update_(\d+)", update_dir.name)
                if m:
                    results.append((int(m.group(1)), png))
    results.sort(key=lambda x: x[0])
    return [p for _, p in results]


def _get_font(size: int = 18):
    """Return a PIL font, falling back gracefully."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def make_gif(png_paths: list[Path], out_path: Path, fps: int) -> None:
    if not png_paths:
        print(f"  [WARN] No frames found for {out_path.name}, skipping.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    font = _get_font(20)
    frames: list[Image.Image] = []

    for png in png_paths:
        m = re.search(r"update_(\d+)", str(png))
        update_num = int(m.group(1)) if m else -1

        img = Image.open(png).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        label = f"Update: {update_num:03d}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad = 6
        x0, y0 = 10, 10
        draw.rectangle(
            [x0 - pad, y0 - pad, x0 + text_w + pad, y0 + text_h + pad],
            fill=(0, 0, 0, 160),
        )
        draw.text((x0, y0), label, font=font, fill=(255, 255, 255, 255))

        composite = Image.alpha_composite(img, overlay).convert("RGB")
        frames.append(composite)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"  Saved GIF ({len(frames)} frames) -> {out_path}")


def plot_metric_evolution(
    df: pd.DataFrame,
    scenario: str,
    metric_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    std_col: str | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = df[df["scenario"] == scenario].copy()
    if sub.empty:
        print(f"  [WARN] No data for scenario '{scenario}', skipping plot.")
        return
    sub = sub.sort_values("update")

    updates = sub["update"].values
    values = sub[metric_col].values

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    # Raw values
    ax.plot(updates, values, color="#4878CF", alpha=0.4, linewidth=1.0, label="Raw")

    # Rolling mean
    if len(values) >= ROLLING_WINDOW:
        roll = pd.Series(values).rolling(ROLLING_WINDOW, center=True, min_periods=1).mean().values
        ax.plot(updates, roll, color="#4878CF", linewidth=2.2,
                label=f"Rolling mean (w={ROLLING_WINDOW})")

    # Optional std shading
    if std_col and std_col in sub.columns:
        stds = sub[std_col].values
        ax.fill_between(updates, values - stds, values + stds,
                        alpha=0.15, color="#4878CF", label=r"±1 std")

    # Run boundary markers
    colors_boundary = ["#E84855", "#F4A261", "#2EC4B6"]
    for i, (boundary, color) in enumerate(zip(RUN_BOUNDARIES, colors_boundary)):
        ax.axvline(boundary, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(boundary + 0.4, ax.get_ylim()[1], RUN_LABELS[i + 1],
                color=color, fontsize=8, va="top", alpha=0.9)

    # n_samples annotation (if available)
    if "n_samples" in sub.columns:
        n = sub["n_samples"].iloc[0]
        ax.annotate(f"n_samples = {int(n)}", xy=(0.98, 0.04),
                    xycoords="axes fraction", ha="right", fontsize=9,
                    color="gray")

    ax.set_xlabel("Training Update", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(updates.min() - 1, updates.max() + 1)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved plot -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lakes_pngs = collect_pngs(RUN_DIRS, "estimate_lakes11")
    drylands_pngs = collect_pngs(RUN_DIRS, "Estimate_Drylands_endgame")

    # --- GIFs at 10 fps ---
    print("Generating GIF (10 fps): estimate_lakes11 ...")
    make_gif(lakes_pngs, OUT_DIR / "estimate_lakes11_training_10fps.gif", 10)

    print("Generating GIF (10 fps): Estimate_Drylands_endgame ...")
    make_gif(drylands_pngs, OUT_DIR / "Estimate_Drylands_endgame_training_10fps.gif", 10)

    # --- GIFs at 20 fps ---
    print("Generating GIF (20 fps): estimate_lakes11 ...")
    make_gif(lakes_pngs, OUT_DIR / "estimate_lakes11_training_20fps.gif", 20)

    print("Generating GIF (20 fps): Estimate_Drylands_endgame ...")
    make_gif(drylands_pngs, OUT_DIR / "Estimate_Drylands_endgame_training_20fps.gif", 20)

    # --- Scalar evolution plots ---
    print("Loading summary CSVs ...")
    df = load_summary_csvs(RUN_DIRS)
    print(f"  Loaded {len(df)} rows across {df['scenario'].nunique()} scenarios.")

    print("Plotting Rider_leapfrogging ...")
    plot_metric_evolution(
        df,
        scenario="Rider_leapfrogging",
        metric_col="uncovered_delta_mean",
        ylabel="Mean Tiles Uncovered (delta)",
        title="Rider Leapfrogging — Tiles Uncovered per Rollout over Training",
        out_path=OUT_DIR / "rider_leapfrogging_uncovered_delta.png",
        std_col="uncovered_delta_std",
    )

    print("Plotting Simple_dash_dancing2 ...")
    plot_metric_evolution(
        df,
        scenario="Simple_dash_dancing2",
        metric_col="uncovered_delta_mean",
        ylabel="Mean Tiles Uncovered (delta)",
        title="Simple Dash Dancing — Tiles Uncovered per Rollout over Training",
        out_path=OUT_DIR / "simple_dash_dancing2_uncovered_delta.png",
        std_col="uncovered_delta_std",
    )

    print("Done. Outputs written to:", str(OUT_DIR.resolve()))


if __name__ == "__main__":
    main()
