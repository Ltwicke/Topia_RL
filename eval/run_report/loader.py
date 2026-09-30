"""
eval/run_report/loader.py
──────────────────────────────────────────────────────────────────────────────
Load the segments of one complete training run and stitch them together along
the checkpoint lineage.

A segment is one invocation of RL/train.py. Each of its outputs is optional:

    run_<ts>.log            banner: resume point + training config
    run_<ts>_metrics.csv    one row per update
    run_<ts>_scenarios/     summary.csv (one row per update x scenario)
                            + update_NNNNN/<Scenario>.png renders

The user lists the segments that form one run; nothing is auto-discovered.

Lineage
───────
A resumed segment starts at its `Start update` (its cut). Everything an
earlier segment logged at an update >= that cut belongs to a branch that was
abandoned when training restarted from an older checkpoint, so it is dropped
before the segment is appended. One rule covers both overlaps
(run_20260922_231554 logged update 106, then run_20260923_103914 resumed from
checkpoint 105 and re-ran 106) and orphaned tails of any length.

This module knows the file layout and the log banner but no metric names
beyond `update`, `scenario` and `_error`.
"""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


TAG_RE          = re.compile(r"run_\d{8}_\d{6}")
_UPDATE_DIR_RE  = re.compile(r"update_(\d+)")
_SUFFIXES       = ("_metrics.csv", "_scenarios", ".log")
TEXT_COLUMNS    = ("scenario", "_error", "segment")

# The banner is the first few dozen lines of the .log; 300 is a generous cap
# that avoids reading multi-hundred-KB logs to the end.
LOG_HEAD_LINES  = 300
_BANNER_RE      = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+(\S.*?)\s*:\s(.*)$")
_CKPT_RE        = re.compile(r"update_in_ckpt=(\d+|\?)")

# Banner key -> (config key, pattern extracting the value). Only settings that
# change what the plotted metrics mean; the diff between consecutive segments
# labels each resume boundary.
_CONFIG_FIELDS = {
    "Workers × envs":      ("envs",      r"(\d+)\s+parallel envs"),
    "Rollout steps":       ("steps",     r"(\d+)"),
    "Full batch":          ("batch",     r"([\d,]+)\s+samples"),
    "Train fraction":      ("minibatch", r"×\s*(\d+)\)"),
    "PPO epochs":          ("epochs",    r"(\d+)"),
    "LR":                  ("lr",        r"(\S+)"),
    "Reward":              ("beta",      r"beta=([^,)\s]+)"),
    "clip_eps / vf / ent": ("ent_coef",  r"/\s*(\S+)\s*$"),
    "Self-play":           ("refresh",   r"refresh every (\d+)"),
}


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Issue:
    level: str          # "info" or "warning"
    text:  str

    def __str__(self) -> str:
        return f"[{self.level}] {self.text}"


@dataclass
class Segment:
    tag:           str
    log_file:      Path | None
    metrics_csv:   Path | None
    scenarios_dir: Path | None
    start_update:  int | None = None    # "Start update : N"
    resumed_from:  int | None = None    # "update_in_ckpt=M"
    from_scratch:  bool = False         # "No pretrained checkpoint"
    config:        dict[str, str] = field(default_factory=dict)


@dataclass
class Boundary:
    update: int                              # first update of the resumed segment
    label:  str                              # "S2", "S3", ... (position in the list)
    tag:    str
    diff:   dict[str, tuple[str, str]]       # config key -> (before, after)


@dataclass
class RunData:
    name:       str
    segments:   list[Segment]
    metrics:    pd.DataFrame                 # one row per update, sorted
    scenarios:  pd.DataFrame                 # one row per (update, scenario)
    frames:     dict[str, dict[int, Path]]   # scenario -> {update: png}
    boundaries: list[Boundary]
    refreshes:  list[int]                    # updates where the frozen opponent changed
    spans:      dict[str, tuple[int, int]]   # tag -> (first, last) update kept
    issues:     list[Issue]


# ══════════════════════════════════════════════════════════════════════════════
# Segment resolution
# ══════════════════════════════════════════════════════════════════════════════

def resolve_segments(args: list[str], log_root: Path) -> list[Segment]:
    """Turn user arguments into segments, ordered by the timestamp in the tag.

    An argument is a bare tag (`run_20260922_231554`, resolved against
    `log_root`) or a path prefix (`RL/logs/V2_longest_training/run_...`).
    A path to one of the segment's own outputs works too.
    """
    prefixes: dict[str, Path] = {}
    for arg in args:
        name = Path(arg).name
        for suffix in _SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        if not TAG_RE.fullmatch(name):
            raise ValueError(f"not a segment (expected run_YYYYMMDD_HHMMSS): {arg}")
        is_path = "/" in arg or "\\" in arg
        folder  = Path(arg).parent if is_path else Path(log_root)
        if name in prefixes and prefixes[name].resolve() != folder.resolve():
            raise ValueError(f"segment {name} is listed twice, in different folders")
        prefixes[name] = folder

    segments = []
    for tag in sorted(prefixes):
        folder = prefixes[tag]
        log     = folder / f"{tag}.log"
        metrics = folder / f"{tag}_metrics.csv"
        sdir    = folder / f"{tag}_scenarios"
        seg = Segment(
            tag           = tag,
            log_file      = log if log.is_file() else None,
            metrics_csv   = metrics if metrics.is_file() else None,
            scenarios_dir = sdir if sdir.is_dir() else None,
        )
        if not (seg.log_file or seg.metrics_csv or seg.scenarios_dir):
            raise FileNotFoundError(f"no outputs of segment {tag} in {folder}")
        if seg.log_file is not None:
            _parse_log_head(seg)
        segments.append(seg)
    return segments


def _parse_log_head(seg: Segment) -> None:
    """Fill the resume point and config of `seg` from its .log banner."""
    with open(seg.log_file, encoding="utf-8", errors="replace") as fh:
        head = [next(fh, "") for _ in range(LOG_HEAD_LINES)]
    for line in head:
        if "[update " in line:          # the banner is over
            break
        m = _CKPT_RE.search(line)
        if m:
            seg.resumed_from = None if m.group(1) == "?" else int(m.group(1))
            continue
        if "No pretrained checkpoint" in line or "training from scratch" in line:
            seg.from_scratch = True
            continue
        m = _BANNER_RE.match(line.rstrip())
        if m is None:
            continue
        key, value = m.group(1).strip(), m.group(2).strip()
        if key == "Start update":
            seg.start_update = int(value) if value.isdigit() else None
        elif key in _CONFIG_FIELDS:
            name, pattern = _CONFIG_FIELDS[key]
            found = re.search(pattern, value)
            if found:
                seg.config[name] = found.group(1).replace(",", "")


# ══════════════════════════════════════════════════════════════════════════════
# Robust reading
# ══════════════════════════════════════════════════════════════════════════════

def _read_bytes_stable(path: Path, retries: int = 3) -> bytes:
    """Read a file a live run may be rewriting. bank.py truncates and rewrites
    summary.csv on every update, so retry until the read saw a stable file."""
    raw = b""
    for attempt in range(retries):
        before = path.stat()
        raw    = path.read_bytes()
        after  = path.stat()
        if before.st_mtime_ns == after.st_mtime_ns and len(raw) == after.st_size:
            return raw
        time.sleep(0.25 * (attempt + 1))
    return raw


def _read_csv(path: Path | None, what: str, issues: list[Issue]) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame()
    raw = _read_bytes_stable(path)
    if not raw.strip():
        issues.append(Issue("warning", f"{what}: file is empty"))
        return pd.DataFrame()
    if not raw.endswith(b"\n"):
        # A row still being written: the csv module ends every complete row
        # with "\r\n", so a finished file always ends in "\n".
        raw = raw[: raw.rfind(b"\n") + 1]
        issues.append(Issue("info", f"{what}: dropped a partially written last line"))
    try:
        return pd.read_csv(io.BytesIO(raw))
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        issues.append(Issue("warning", f"{what}: unreadable ({exc})"))
        return pd.DataFrame()


def _normalise(df: pd.DataFrame, what: str, issues: list[Issue]) -> pd.DataFrame:
    """`update` -> int64, every metric column -> float64.

    A header-only CSV reads as all-object columns, and concatenating it with
    real data would silently turn every column into object; normalising each
    frame before the concat keeps the dtypes honest.
    """
    if df.empty:
        return pd.DataFrame()
    if "update" not in df.columns:
        issues.append(Issue("warning", f"{what}: no 'update' column, ignored"))
        return pd.DataFrame()
    update = pd.to_numeric(df["update"], errors="coerce")
    if update.isna().any():
        issues.append(Issue(
            "warning", f"{what}: dropped {int(update.isna().sum())} rows without a valid update"))
    df = df.loc[update.notna()].copy()
    df["update"] = update.loc[update.notna()].astype("int64")
    for col in df.columns:
        if col == "update" or col in TEXT_COLUMNS:
            continue
        values = df[col]
        if not pd.api.types.is_numeric_dtype(values):
            numeric = pd.to_numeric(values, errors="coerce")
            lost = int((numeric.isna() & values.notna()).sum())
            if lost:
                issues.append(Issue("warning", f"{what}: ignored {lost} non-numeric values in {col!r}"))
            values = numeric
        df[col] = values.astype("float64")
    return df


def _drop_errors(df: pd.DataFrame, tag: str, issues: list[Issue]) -> pd.DataFrame:
    """Remove rows of scenarios that crashed (bank.py writes `_error` for them)."""
    if "_error" not in df.columns:
        return df
    err = df["_error"].notna() & (df["_error"].astype(str).str.strip() != "")
    for name, n in df.loc[err, "scenario"].value_counts().items():
        issues.append(Issue("warning", f"{tag}: scenario {name} crashed in {n} update(s)"))
    return df.loc[~err].drop(columns="_error")


def _dedupe(df: pd.DataFrame, keys: list[str], what: str, issues: list[Issue]) -> pd.DataFrame:
    if df.empty:
        return df
    dup = df.duplicated(keys, keep="last")
    if dup.any():
        issues.append(Issue("warning", f"{what}: {int(dup.sum())} duplicate rows, kept the last"))
    return df.loc[~dup]


def _collect_frames(sdir: Path | None) -> dict[str, dict[int, Path]]:
    frames: dict[str, dict[int, Path]] = {}
    if sdir is None:
        return frames
    for d in sdir.iterdir():
        m = _UPDATE_DIR_RE.fullmatch(d.name)
        if m is None or not d.is_dir():
            continue
        update = int(m.group(1))
        for png in d.glob("*.png"):
            frames.setdefault(png.stem, {})[update] = png
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Helpers shared with the plotting side
# ══════════════════════════════════════════════════════════════════════════════

def update_grid(updates) -> np.ndarray:
    """Regular grid over `updates`, stepping by their smallest positive gap
    (1 when scenarios are evaluated every update)."""
    u = np.unique(np.asarray(updates, dtype=np.int64))
    if u.size == 0:
        return u
    diffs = np.diff(u)
    step  = int(diffs[diffs > 0].min()) if diffs.size else 1
    return np.arange(u[0], u[-1] + 1, step, dtype=np.int64)


def ranges(values) -> str:
    """[3, 4, 5, 9] -> '3-5, 9'."""
    values = sorted(int(v) for v in values)
    if not values:
        return ""
    out, start, prev = [], values[0], values[0]
    for v in values[1:] + [None]:
        if v is not None and v == prev + 1:
            prev = v
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        if v is not None:
            start = prev = v
    return ", ".join(out)


# ══════════════════════════════════════════════════════════════════════════════
# Stitching
# ══════════════════════════════════════════════════════════════════════════════

def _cut(part: pd.DataFrame, cut: int, by: str, what: str, issues: list[Issue]) -> pd.DataFrame:
    drop = part["update"] >= cut
    if drop.any():
        dropped = part.loc[drop, "update"].unique()
        noun = "update" if len(dropped) == 1 else "updates"
        issues.append(Issue(
            "info",
            f"{part['segment'].iat[0]}: {what} for {noun} {ranges(dropped)} "
            f"superseded by {by} (lineage cut at {cut})",
        ))
    return part.loc[~drop]


def load_run(segments: list[Segment], name: str) -> RunData:
    """Read every segment and stitch them along the checkpoint lineage."""
    issues: list[Issue] = []
    loaded = []
    for seg in segments:
        absent = [label for label, p in (("log", seg.log_file),
                                         ("metrics CSV", seg.metrics_csv),
                                         ("scenarios folder", seg.scenarios_dir)) if p is None]
        if absent:
            issues.append(Issue("info", f"{seg.tag}: no {', '.join(absent)}"))

        metrics = _normalise(
            _read_csv(seg.metrics_csv, f"{seg.tag} metrics", issues), f"{seg.tag} metrics", issues)
        summary = seg.scenarios_dir / "summary.csv" if seg.scenarios_dir else None
        scen = _normalise(
            _read_csv(summary, f"{seg.tag} summary.csv", issues), f"{seg.tag} summary.csv", issues)
        if not scen.empty and "scenario" not in scen.columns:
            issues.append(Issue("warning", f"{seg.tag} summary.csv: no 'scenario' column, ignored"))
            scen = pd.DataFrame()
        scen    = _drop_errors(scen, seg.tag, issues)
        metrics = _dedupe(metrics, ["update"], f"{seg.tag} metrics", issues)
        scen    = _dedupe(scen, ["update", "scenario"], f"{seg.tag} summary.csv", issues)
        if not metrics.empty:
            metrics = metrics.assign(segment=seg.tag)
        if not scen.empty:
            scen = scen.assign(segment=seg.tag)
        frames = _collect_frames(seg.scenarios_dir)

        own = set()
        for part in (metrics, scen):
            if not part.empty:
                own.update(part["update"].tolist())
        for per_update in frames.values():
            own.update(per_update)
        span = (min(own), max(own)) if own else None
        loaded.append((seg, metrics, scen, frames, span))

    # ── Cut and append, oldest segment first ──────────────────────────────────
    cuts: list[int | None] = []
    metric_parts: list[pd.DataFrame] = []
    scen_parts:   list[pd.DataFrame] = []
    frames_all:   dict[str, dict[int, Path]] = {}
    for k, (seg, metrics, scen, frames, span) in enumerate(loaded):
        candidates = [x for x in (seg.start_update, span[0] if span else None) if x is not None]
        cut = min(candidates) if candidates else None
        cuts.append(cut)
        if k > 0 and cut is not None:
            metric_parts = [_cut(p, cut, seg.tag, "metrics", issues) for p in metric_parts]
            scen_parts   = [_cut(p, cut, seg.tag, "scenario rows", issues) for p in scen_parts]
            for per_update in frames_all.values():
                for update in [u for u in per_update if u >= cut]:
                    del per_update[update]
        if not metrics.empty:
            metric_parts.append(metrics)
        if not scen.empty:
            scen_parts.append(scen)
        for scenario, per_update in frames.items():
            frames_all.setdefault(scenario, {}).update(per_update)

    metric_parts = [p for p in metric_parts if not p.empty]
    scen_parts   = [p for p in scen_parts if not p.empty]
    metrics_df = (pd.concat(metric_parts, ignore_index=True)
                  .sort_values("update", kind="stable").reset_index(drop=True)
                  if metric_parts else pd.DataFrame({"update": pd.Series(dtype="int64")}))
    scen_df = (pd.concat(scen_parts, ignore_index=True)
               .sort_values(["scenario", "update"], kind="stable").reset_index(drop=True)
               if scen_parts else pd.DataFrame({"update": pd.Series(dtype="int64"),
                                                "scenario": pd.Series(dtype="str")}))
    frames_all = {s: dict(sorted(f.items())) for s, f in sorted(frames_all.items()) if f}

    _validate(loaded, cuts, issues)
    _report_gaps(metrics_df, scen_df, issues)

    # ── Boundaries, kept spans and opponent refreshes ─────────────────────────
    boundaries = []
    for k in range(1, len(loaded)):
        if cuts[k] is None:
            continue
        before, after = loaded[k - 1][0].config, loaded[k][0].config
        diff = {key: (before[key], after[key])
                for key in after if key in before and before[key] != after[key]}
        boundaries.append(Boundary(cuts[k], f"S{k + 1}", loaded[k][0].tag, diff))

    spans: dict[str, tuple[int, int]] = {}
    for part in (metrics_df, scen_df):
        if part.empty:
            continue
        for tag, updates in part.groupby("segment")["update"]:
            lo, hi = int(updates.min()), int(updates.max())
            if tag in spans:
                lo, hi = min(lo, spans[tag][0]), max(hi, spans[tag][1])
            spans[tag] = (lo, hi)

    refreshes: list[int] = []
    for seg, *_ in loaded:
        if seg.tag not in spans or "refresh" not in seg.config:
            continue
        first, last = spans[seg.tag]
        start = seg.start_update if seg.start_update is not None else first
        every = int(seg.config["refresh"])
        # train.py re-seeds the frozen opponent from the loaded weights at the
        # start of every segment, then every `every` updates after that.
        refreshes.append(start)
        refreshes.extend(u for u in range(start + 1, last + 1) if every > 0 and u % every == 0)

    return RunData(
        name       = name,
        segments   = [seg for seg, *_ in loaded],
        metrics    = metrics_df,
        scenarios  = scen_df,
        frames     = frames_all,
        boundaries = boundaries,
        refreshes  = sorted(set(refreshes)),
        spans      = spans,
        issues     = issues,
    )


def _validate(loaded, cuts, issues: list[Issue]) -> None:
    """Sanity-check the user's segment list. Warnings only, never fatal."""
    for k, (seg, _m, _s, _f, span) in enumerate(loaded):
        if span and seg.start_update is not None and span[0] < seg.start_update:
            issues.append(Issue(
                "warning", f"{seg.tag}: has rows from update {span[0]}, "
                           f"before its start update {seg.start_update}"))
        if k == 0:
            if seg.resumed_from is not None:
                issues.append(Issue(
                    "info", f"{seg.tag}: the first segment starts from checkpoint "
                            f"update {seg.resumed_from}"))
            continue
        prev, prev_span = loaded[k - 1][0], loaded[k - 1][4]
        if seg.from_scratch:
            issues.append(Issue(
                "warning", f"{seg.tag}: started without a checkpoint; is it part of this run?"))
        elif seg.log_file is not None and seg.resumed_from is None:
            issues.append(Issue("info", f"{seg.tag}: the resume checkpoint is not recorded"))
        if (seg.start_update is not None and seg.resumed_from is not None
                and seg.start_update != seg.resumed_from + 1):
            issues.append(Issue(
                "warning", f"{seg.tag}: start update {seg.start_update} does not follow "
                           f"its checkpoint update {seg.resumed_from}"))
        if seg.resumed_from is not None and prev_span and seg.resumed_from > prev_span[1]:
            issues.append(Issue(
                "warning", f"{seg.tag}: resumes from checkpoint {seg.resumed_from}, but "
                           f"{prev.tag} ends at update {prev_span[1]}; is a segment "
                           f"missing from the list?"))
        if cuts[k] is not None and prev_span and cuts[k] <= prev_span[0]:
            issues.append(Issue(
                "warning", f"{prev.tag} is completely superseded by {seg.tag}"))


def _report_gaps(metrics: pd.DataFrame, scenarios: pd.DataFrame, issues: list[Issue]) -> None:
    if not metrics.empty:
        have = set(metrics["update"].tolist())
        missing = [u for u in range(min(have), max(have) + 1) if u not in have]
        if missing:
            issues.append(Issue("warning", f"metrics: no rows for updates {ranges(missing)}"))
    if not scenarios.empty:
        have = set(scenarios["update"].tolist())
        missing = [int(u) for u in update_grid(list(have)) if int(u) not in have]
        if missing:
            issues.append(Issue(
                "warning", f"scenario eval: no rows for updates {ranges(missing)}"))
