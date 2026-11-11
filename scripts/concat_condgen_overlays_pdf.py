#!/usr/bin/env python3
"""Assemble CondGen mask overlays into a labelled PDF contact sheet.

The script is intended for qualitative appendix figures.  It reads one image
asset from each sample directory, keeps every image's aspect ratio, and gives
all rows the same image height.  The default asset is the display-only
``mask_overlay_bilinear_alpha0.45.png`` produced by ``render_condgen_1024.py``.

The output is a single high-resolution raster page embedded in a PDF.  This is
deliberate: the source overlays are raster images, and the ``--dpi`` argument
controls both their output sampling and the physical page resolution.

Examples
--------
Render the complete 50-sample 1024 collection:

    python concat_condgen_overlays_pdf.py \
        --source-dir ../object_discovery_compare/VOC07_condgen_best_1024_all50 \
        --output ../object_discovery_compare/VOC07_condgen_best_1024_all50/condgen_overlay_concat.pdf \
        --strict-count

Require a complete 10 x 5 collection once all 50 samples have been rendered:

    python concat_condgen_overlays_pdf.py --strict-count

``--page-height-in`` can be supplied when fitting the result to a particular
LaTeX page.  If omitted, the height is derived from ``--row-height-in`` and
the requested number of rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE = PROJECT_DIR / "object_discovery_compare" / "VOC07_condgen_best_1024_all50"
DEFAULT_ASSET = "mask_overlay_bilinear_alpha0.45.png"
DEFAULT_OUTPUT = DEFAULT_SOURCE / "condgen_overlay_bilinear_alpha0.45_concat.pdf"


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_rank(path: Path) -> int | None:
    match = re.match(r"^(\d+)(?:_|$)", path.name)
    return int(match.group(1)) if match else None


def natural_sample_key(path: Path) -> tuple[int, str]:
    """Sort ``005_005625`` and ``005_iou_...`` by their rank prefix."""

    rank = sample_rank(path)
    return (rank if rank is not None else 10**9, path.name)


def label_for_sample(directory: Path, label_template: str = "{image_stem}") -> str:
    """Return only the image id, e.g. ``005625``.

    ``render_condgen_1024.py`` stores the original file name in
    ``rendering.json``.  We prefer that source, then fall back to the final
    six-digit component of the directory name.
    """

    metadata = directory / "rendering.json"
    image_stem = None
    image_name = None
    if metadata.is_file():
        try:
            data = json.loads(metadata.read_text())
            image_name = data.get("image")
            if image_name:
                image_stem = Path(str(image_name)).stem
        except (OSError, ValueError, TypeError):
            pass

    if image_stem is None:
        match = re.search(r"(?:^|_)(\d{6})(?:$|\.)", directory.name)
        if match:
            image_stem = match.group(1)
        else:
            match = re.search(r"(\d+)$", directory.name)
            image_stem = match.group(1) if match else directory.name
    rank = sample_rank(directory)
    values = {"image_stem": image_stem, "image": image_name or f"{image_stem}.jpg",
              "rank": f"{rank:03d}" if rank is not None else "", "directory": directory.name}
    try:
        return label_template.format(**values)
    except KeyError as error:
        raise ValueError(f"Unknown field in --label-template: {error.args[0]}") from error


def find_samples(source_dir: Path, asset: str, ranks: list[int] | None = None,
                 label_template: str = "{image_stem}") -> tuple[list[dict[str, Any]], list[str]]:
    """Find direct sample directories and report directories lacking ``asset``."""

    samples: list[dict[str, Any]] = []
    missing: list[str] = []
    directories = [p for p in source_dir.iterdir() if p.is_dir()]
    if ranks is None:
        directories.sort(key=natural_sample_key)
    else:
        requested = {rank: position for position, rank in enumerate(ranks)}
        directories = [p for p in directories if sample_rank(p) in requested]
        directories.sort(key=lambda p: requested[sample_rank(p)])
        present = {sample_rank(p) for p in directories}
        missing.extend(f"rank_{rank:03d}" for rank in ranks if rank not in present)
    for directory in directories:
        image_path = directory / asset
        if not image_path.is_file():
            missing.append(directory.name)
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {image_path}")
        samples.append({
            "directory": directory.name,
            "path": image_path,
            "label": label_for_sample(directory, label_template),
            "width": width,
            "height": height,
            "aspect": width / height,
            "sha256": sha256(image_path),
        })
    return samples, missing


def default_font_path() -> Path | None:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    return next((path for path in candidates if path.is_file()), None)


def load_font(font_path: Path | None, size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path is not None:
        return ImageFont.truetype(str(font_path), size_px)
    fallback = default_font_path()
    if fallback is not None:
        return ImageFont.truetype(str(fallback), size_px)
    return ImageFont.load_default()


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    target_width = max(1, round(image.width * target_height / image.height))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def build_pdf(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = args.source_dir.expanduser().resolve(strict=True)
    if not source_dir.is_dir():
        raise ValueError(f"Source is not a directory: {source_dir}")
    samples, missing = find_samples(source_dir, args.asset, args.ranks, args.label_template)
    capacity = args.rows * args.cols
    if not samples:
        raise ValueError(f"No sample directories contain {args.asset!r}: {source_dir}")
    if len(samples) > capacity:
        raise ValueError(
            f"Found {len(samples)} samples but the layout has capacity {capacity}; "
            "increase --rows/--cols or select a smaller source directory"
        )
    if args.strict_count and len(samples) != capacity:
        raise ValueError(
            f"--strict-count requires exactly {capacity} samples, found {len(samples)} "
            f"({len(missing)} directories lack {args.asset})"
        )
    if missing and not args.allow_fewer:
        raise ValueError(
            f"{len(missing)} sample directories lack {args.asset}; use --allow-fewer "
            "for a partial sheet or fix the source collection first."
        )

    # Page width and horizontal geometry are specified in physical units.
    dpi = args.dpi
    margin_left = round(args.margin_left_in * dpi)
    margin_right = round(args.margin_right_in * dpi)
    margin_top = round(args.margin_top_in * dpi)
    margin_bottom = round(args.margin_bottom_in * dpi)
    hspace = round(args.hspace_in * dpi)
    vspace = round(args.vspace_in * dpi)
    label_gap = round(args.label_gap_in * dpi)
    page_width = round(args.page_width_in * dpi)
    cell_width = (page_width - margin_left - margin_right - (args.cols - 1) * hspace) / args.cols
    if cell_width <= 0:
        raise ValueError("Margins and horizontal spacing leave no room for cells")

    font_size_px = max(1, round(args.font_size_pt / 72 * dpi))
    font = load_font(args.font_path, font_size_px)
    # The exact label height varies slightly by font.  Measure representative
    # text so it remains correct when the user changes font or size.
    scratch = Image.new("RGB", (8, 8), "white")
    draw = ImageDraw.Draw(scratch)
    label_height = max(1, text_bbox(draw, "000000", font)[1])

    max_aspect = max(sample["aspect"] for sample in samples)
    image_height = round(args.row_height_in * dpi)
    max_height_for_cell = math.floor(cell_width / max_aspect)
    if image_height > max_height_for_cell:
        image_height = max_height_for_cell
        print(
            f"Warning: row height reduced to {image_height / dpi:.3f} in so the widest "
            f"image ({max_aspect:.3f}:1) fits each cell.",
            flush=True,
        )
    if image_height <= 0:
        raise ValueError("Image height became non-positive; reduce margins or increase page width")

    row_block_height = label_height + label_gap + image_height
    auto_page_height = (
        margin_top + margin_bottom + args.rows * row_block_height + (args.rows - 1) * vspace
    )
    if args.page_height_in is None:
        page_height = auto_page_height
    else:
        page_height = round(args.page_height_in * dpi)
        if page_height < auto_page_height:
            raise ValueError(
                f"Page height {args.page_height_in:.3f} in is too short; need at least "
                f"{auto_page_height / dpi:.3f} in for the requested row height"
            )

    background = tuple(args.background)
    canvas = Image.new("RGB", (page_width, page_height), background)
    draw = ImageDraw.Draw(canvas)
    entries: list[dict[str, Any]] = []
    for index in range(capacity):
        row, col = divmod(index, args.cols)
        cell_x = margin_left + col * (cell_width + hspace)
        label_y = margin_top + row * (row_block_height + vspace)
        image_y = label_y + label_height + label_gap
        if index >= len(samples):
            entries.append({"row": row, "column": col, "empty": True})
            continue
        sample = samples[index]
        with Image.open(sample["path"]) as opened:
            image = opened.convert("RGB")
        image = resize_to_height(image, image_height)
        image_x = round(cell_x + (cell_width - image.width) / 2)
        draw.text(
            (round(cell_x + cell_width / 2), label_y),
            sample["label"],
            fill=args.text_color,
            font=font,
            anchor="ma",
        )
        canvas.paste(image, (image_x, image_y))
        entries.append({
            "row": row,
            "column": col,
            "empty": False,
            "label": sample["label"],
            "directory": sample["directory"],
            "asset": str(sample["path"]),
            "asset_sha256": sample["sha256"],
            "original_size": [sample["width"], sample["height"]],
            "rendered_size": [image.width, image.height],
        })

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Pillow embeds the RGB raster as a single PDF page and records the chosen
    # physical resolution.  A PNG sidecar is useful for inspecting layout
    # before placing the PDF into LaTeX, but is optional.
    canvas.save(output, "PDF", resolution=dpi)
    if args.preview_png is not None:
        preview = args.preview_png.expanduser().resolve()
        preview.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(preview, "PNG", optimize=True)

    manifest = {
        "source_dir": str(source_dir),
        "asset": args.asset,
        "output_pdf": str(output),
        "rows": args.rows,
        "columns": args.cols,
        "sample_count": len(samples),
        "empty_cells": capacity - len(samples),
        "missing_directories": missing,
        "dpi": dpi,
        "page_size_px": [page_width, page_height],
        "page_size_in": [page_width / dpi, page_height / dpi],
        "font_size_pt": args.font_size_pt,
        "font": str(args.font_path) if args.font_path else str(default_font_path() or "PIL default"),
        "image_height_px": image_height,
        "image_height_in": image_height / dpi,
        "cell_width_px": cell_width,
        "margins_in": [args.margin_left_in, args.margin_right_in,
                       args.margin_top_in, args.margin_bottom_in],
        "hspace_in": args.hspace_in,
        "vspace_in": args.vspace_in,
        "label_gap_in": args.label_gap_in,
        "background": list(background),
        "entries": entries,
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE,
                        help="Directory containing one subdirectory per sample")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output PDF path")
    parser.add_argument("--asset", default=DEFAULT_ASSET,
                        help="Image file inside each sample directory")
    parser.add_argument("--ranks", type=_positive_int, nargs="+", default=None,
                        help="Optional sample rank prefixes, in the desired output order")
    parser.add_argument("--label-template", default="{image_stem}",
                        help="Label fields: {image_stem}, {image}, {rank}, {directory}")
    parser.add_argument("--rows", type=_positive_int, default=10)
    parser.add_argument("--cols", type=_positive_int, default=5)
    parser.add_argument("--strict-count", action="store_true",
                        help="Require rows*cols available assets")
    parser.add_argument("--allow-fewer", action="store_true",
                        help="Allow missing assets/directories and leave blank cells")
    parser.add_argument("--page-width-in", type=_positive_float, default=15.0,
                        help="PDF page width in inches")
    parser.add_argument("--page-height-in", type=_positive_float, default=None,
                        help="PDF page height; by default it is derived from rows")
    parser.add_argument("--row-height-in", type=_positive_float, default=1.75,
                        help="Target image height for every row")
    parser.add_argument("--font-size-pt", type=_positive_float, default=10.0,
                        help="Label font size in points")
    parser.add_argument("--font-path", type=Path, default=None,
                        help="Optional TrueType/OpenType font path")
    parser.add_argument("--margin-left-in", type=_nonnegative_float, default=0.25)
    parser.add_argument("--margin-right-in", type=_nonnegative_float, default=0.25)
    parser.add_argument("--margin-top-in", type=_nonnegative_float, default=0.20)
    parser.add_argument("--margin-bottom-in", type=_nonnegative_float, default=0.20)
    parser.add_argument("--hspace-in", type=_nonnegative_float, default=0.08,
                        help="Horizontal gap between cells")
    parser.add_argument("--vspace-in", type=_nonnegative_float, default=0.08,
                        help="Vertical gap between rows")
    parser.add_argument("--label-gap-in", type=_nonnegative_float, default=0.03,
                        help="Gap between label and image")
    parser.add_argument("--dpi", type=_positive_int, default=300,
                        help="Rasterisation and PDF resolution")
    parser.add_argument("--background", type=int, nargs=3, default=(255, 255, 255),
                        metavar=("R", "G", "B"), help="Canvas RGB background")
    parser.add_argument("--text-color", default="black", help="Pillow text colour")
    parser.add_argument("--preview-png", type=Path, default=None,
                        help="Optional PNG preview written alongside the PDF")
    args = parser.parse_args()
    if args.ranks is not None and len(args.ranks) != len(set(args.ranks)):
        parser.error("--ranks must not contain duplicates")
    if any(channel < 0 or channel > 255 for channel in args.background):
        parser.error("--background channels must be in [0, 255]")
    if args.font_path is not None and not args.font_path.is_file():
        parser.error(f"Font does not exist: {args.font_path}")
    return args


def main() -> None:
    args = parse_args()
    manifest = build_pdf(args)
    print(
        f"Wrote {manifest['sample_count']} samples ({manifest['empty_cells']} empty cells) to "
        f"{manifest['output_pdf']} ({manifest['page_size_in'][0]:.2f} x "
        f"{manifest['page_size_in'][1]:.2f} in at {manifest['dpi']} dpi)",
        flush=True,
    )
    if manifest["missing_directories"]:
        print("Directories without the requested asset:", ", ".join(manifest["missing_directories"]), flush=True)


if __name__ == "__main__":
    main()
