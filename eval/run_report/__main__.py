"""
python -m eval.run_report — build the plotting handbook of one training run.

List every segment that belongs to the run, as bare tags (resolved against
--log-root) or as path prefixes, in any order:

    python -m eval.run_report --name V3_sc74 run_20260922_231554 \\
        run_20260923_103914 run_20260924_102222 run_20260926_152118

    python -m eval.run_report --name V2_longest \\
        RL/logs/V2_longest_training/run_20260615_004712 \\
        RL/logs/V2_longest_training/run_20260615_015013 \\
        RL/logs/V2_longest_training/run_20260615_112309

Writes training_summary.png, scenarios/<Scenario>.png, gifs/<Scenario>.gif,
handbook.pdf and segments.txt to --out (default <log-root>/reports/<name>/).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # files only; must precede the pyplot import below

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import gifs, plot, specs
from .loader import load_run, resolve_segments

DEFAULT_LOG_ROOT = Path(__file__).resolve().parents[2] / "RL" / "logs"


def _parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m eval.run_report",
        description="Plot scenarios, training metrics and losses of one run made of any "
                    "number of segments.",
    )
    ap.add_argument("segments", nargs="+",
                    help="segment tags (run_YYYYMMDD_HHMMSS) or path prefixes, any order")
    ap.add_argument("--name", required=True, help="name of the run (title and output folder)")
    ap.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT,
                    help=f"folder that bare tags resolve against (default {DEFAULT_LOG_ROOT})")
    ap.add_argument("--out", type=Path, help="output folder (default <log-root>/reports/<name>)")
    ap.add_argument("--window", type=int, default=plot.WINDOW, help="rolling-mean window in updates")
    ap.add_argument("--no-gifs", action="store_true", help="skip the GIF flip-books")
    ap.add_argument("--gif-every", type=int, default=1, help="use every Nth frame in the GIFs")
    ap.add_argument("--gif-scale", type=float, default=1.0, help="resize GIF frames by this factor")
    return ap.parse_args(argv)


def _save(fig, png: Path | None, pdf: PdfPages) -> None:
    if png is not None:
        fig.savefig(png, dpi=plot.DPI)
    pdf.savefig(fig)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")    # a cp1252 pipe must not crash the report
    run  = load_run(resolve_segments(args.segments, args.log_root), args.name)
    metrics, notes = specs.clean_metrics(run.metrics)
    flags = specs.sanity_flags(run, metrics) + notes
    out = args.out or args.log_root / "reports" / args.name
    (out / "scenarios").mkdir(parents=True, exist_ok=True)

    cover = plot.cover_lines(run, flags, specs.NOTES)
    (out / "segments.txt").write_text("\n".join(cover) + "\n", encoding="utf-8")
    print("\n".join(cover), flush=True)

    with plt.style.context(plot.STYLE), PdfPages(out / "handbook.pdf") as pdf:
        _save(plot.render_cover(cover), None, pdf)
        _save(plot.render_training(run, metrics, specs.TRAINING, args.window, flags),
              out / "training_summary.png", pdf)
        print(f"\n  training_summary.png", flush=True)
        for name, spec in specs.scenario_figures(run):
            _save(plot.render_scenario(name, spec, run, args.window),
                  out / "scenarios" / f"{name}.png", pdf)
            print(f"  scenarios/{name}.png", flush=True)

    if not args.no_gifs:
        for name, frames in run.frames.items():
            n = gifs.make_gif(frames, out / "gifs" / f"{name}.gif",
                              every=args.gif_every, scale=args.gif_scale)
            print(f"  gifs/{name}.gif ({n} frames)", flush=True)

    print(f"\nDone. Handbook: {out / 'handbook.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
