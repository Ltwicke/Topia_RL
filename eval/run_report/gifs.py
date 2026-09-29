"""
eval/run_report/gifs.py
──────────────────────────────────────────────────────────────────────────────
Flip-books of the per-update scenario renders, ported from make_gif in the
former eval/generate_training_visualizations.py: every frame gets an
"Update: NNN" badge in the top-left corner.

Render sizes jitter by a few pixels between updates, so frames are padded onto
a white canvas of the largest size. Frames are produced lazily; Pillow still
keeps the palette-mode frames it has written, which is ~1 byte per pixel.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

GIF_FPS = 30


def _font(size: int = 20):
    """Return a PIL font, falling back gracefully."""
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _frame(update: int, path: Path, size: tuple[int, int], out_size: tuple[int, int], font):
    canvas = Image.new("RGBA", size, (255, 255, 255, 255))
    with Image.open(path) as img:
        canvas.alpha_composite(img.convert("RGBA"))

    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    label   = f"Update: {update:03d}"
    bbox    = draw.textbbox((0, 0), label, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad, x0, y0 = 6, 10, 10
    draw.rectangle([x0 - pad, y0 - pad, x0 + text_w + pad, y0 + text_h + pad], fill=(0, 0, 0, 160))
    draw.text((x0, y0), label, font=font, fill=(255, 255, 255, 255))

    frame = Image.alpha_composite(canvas, overlay).convert("RGB")
    return frame if out_size == size else frame.resize(out_size, Image.Resampling.LANCZOS)


def make_gif(frames: dict[int, Path], out_path: Path, fps: int = GIF_FPS,
             every: int = 1, scale: float = 1.0) -> int:
    """Write one GIF from {update: png}; returns the number of frames written."""
    items = sorted(frames.items())[:: max(every, 1)]
    if not items:
        return 0
    sizes = []
    for _, path in items:
        with Image.open(path) as img:
            sizes.append(img.size)
    size     = (max(w for w, _ in sizes), max(h for _, h in sizes))
    out_size = (max(1, round(size[0] * scale)), max(1, round(size[1] * scale)))
    font     = _font(20)

    rendered = (_frame(update, path, size, out_size, font) for update, path in items)
    first    = next(rendered)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first.save(out_path, save_all=True, append_images=rendered,
               duration=int(1000 / fps), loop=0, optimize=False)
    return len(items)
