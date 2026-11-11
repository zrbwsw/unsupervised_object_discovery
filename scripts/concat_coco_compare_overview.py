#!/usr/bin/env python3
"""Make the two-row COCO object-discovery comparison panel.

Each sample directory produced by ``prepare_object_discovery_compare.py`` has
one ground-truth rendering and four method renderings.  This script arranges
those nine images in the appendix layout:

                 ground truth | CondGen | MAE | iBOT | DINO
                                  bbox       bbox   bbox   bbox
                                  mask       mask   mask   mask

The ground-truth image is vertically centred over the two method rows. Before
layout, every rendering is resized to a common 640x427 reference resolution
by default, so all cells use one aspect ratio; source files are not modified.
The resized image is then fitted to the cell with ``ImageOps.contain``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


METHODS = ("condgen", "mae", "ibot", "dino")
METHOD_LABELS = {
    "condgen": "CondGen(Ours)",
    "mae": "MAE",
    "ibot": "iBOT",
    "dino": "DINO",
}
IREPA_MS_FONT_DIR = Path(sys.prefix) / "share" / "fonts" / "truetype" / "msttcorefonts"
FONT_REGULAR_CANDIDATES = (
    str(IREPA_MS_FONT_DIR / "Times_New_Roman.ttf"),
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
FONT_BOLD_CANDIDATES = (
    str(IREPA_MS_FONT_DIR / "Times_New_Roman_Bold.ttf"),
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


def selected_font_path(bold: bool = False) -> str:
    candidates = FONT_BOLD_CANDIDATES if bold else FONT_REGULAR_CANDIDATES
    return next((path for path in candidates if Path(path).is_file()), candidates[-1])


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a deterministic system font, with Pillow's fallback as backup."""
    candidate = selected_font_path(bold)
    try:
        return ImageFont.truetype(candidate, size)
    except OSError:
        return ImageFont.truetype(selected_font_path(False), size)


def point_pixels(points: float, dpi: int) -> int:
    return max(1, round(float(points) * float(dpi) / 72.0))


def parse_image_label(sample: Path, dataset_name: str | None) -> str:
    """Return a short label such as ``COCO 501038``.

    The COCO compact sample names end in ``COCO_train2014_000000501038``.
    For another dataset, the final numeric component is retained and the
    first component of the name is used as the dataset label.  ``--image-label``
    can always override this automatic value.
    """
    stem = sample.name
    index_match = re.search(r"(?P<index>\d{4,})$", stem)
    if index_match:
        # Prefer an explicit dataset token anywhere in the compact directory
        # name.  This avoids interpreting ``gap`` or ``iou`` as the dataset
        # when names start with ``001_gap_...`` or ``001_iou_...``.
        token_match = re.search(r"(?:^|_)(COCO|VOC(?:07|12)?)(?:_|$)", stem, re.IGNORECASE)
        inferred = token_match.group(1) if token_match else None
        name = dataset_name or inferred or "Dataset"
        index = index_match.group("index").lstrip("0") or "0"
        return f"{name} {index}"
    return dataset_name or stem


def read_iou(sample: Path, method: str) -> float:
    path = sample / method / "prediction.json"
    payload = json.loads(path.read_text())
    # The compact compare tree stores max_iou at the top level.  Supporting an
    # explicit ``metrics`` object makes this useful with older exported trees.
    value = payload.get("max_iou")
    if value is None:
        value = payload.get("metrics", {}).get("max_iou")
    if value is None:
        raise ValueError(f"No max_iou in {path}")
    return float(value)


def center_text(
    canvas: Image.Image,
    text: str,
    x: int,
    y: int,
    width: int,
    height: int,
    text_font: ImageFont.FreeTypeFont,
    *,
    fill: tuple[int, int, int] = (25, 30, 36),
) -> None:
    draw = ImageDraw.Draw(canvas)
    # Shrink long labels to remain inside the cell while preserving the
    # requested nominal font size for normal labels.
    current = text_font
    nominal = getattr(text_font, "size", 1)
    while nominal > 8 and draw.textlength(text, font=current) > max(4, width - 8):
        nominal -= 1
        current = load_font(
            nominal,
            bold=(getattr(text_font, "path", None) == selected_font_path(True)),
        )
    bounds = draw.textbbox((0, 0), text, font=current)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    draw.text(
        (x + (width - text_width) / 2 - bounds[0],
         y + (height - text_height) / 2 - bounds[1]),
        text,
        font=current,
        fill=fill,
    )


def paste_contained(
    canvas: Image.Image,
    path: Path,
    box: tuple[int, int, int, int],
    input_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Resize a rendering to the common input size, then fit it in its cell."""
    x, y, width, height = box
    with Image.open(path) as source:
        # Direct resize is intentional: all visualizations are compared at the
        # same 640x427 aspect ratio before they enter the appendix panel.
        image = source.convert("RGB").resize(input_size, Image.Resampling.LANCZOS)
        image = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
    image_x = x + (width - image.width) // 2
    image_y = y + (height - image.height) // 2
    canvas.paste(image, (image_x, image_y))
    return image_x, image_y, image.width, image.height


def save_image(canvas: Image.Image, output: Path, dpi: int, quality: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    suffix = output.suffix.lower()
    if suffix == ".pdf":
        canvas.save(temporary, format="PDF", resolution=dpi)
    elif suffix == ".png":
        canvas.save(temporary, format="PNG", dpi=(dpi, dpi))
    else:
        canvas.save(temporary, format="JPEG", quality=quality, subsampling=0, dpi=(dpi, dpi))
    temporary.replace(output)


def make_overview(args: argparse.Namespace) -> dict:
    sample = args.sample.expanduser().resolve()
    if not sample.is_dir():
        raise FileNotFoundError(f"Sample directory does not exist: {sample}")

    required = [sample / "ground_truth_boxes.png"]
    for method in METHODS:
        required.extend((sample / method / "prediction_bbox.png",
                         sample / method / args.mask_file))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required rendering(s):\n" + "\n".join(missing))

    ious = {method: read_iou(sample, method) for method in METHODS}
    image_label = args.image_label or parse_image_label(sample, args.dataset_name)

    # Dimensions are pixels so the script is deterministic.  ``--dpi`` controls
    # the metadata when saving PNG/JPEG/PDF and the conversion from point size.
    cell_width = int(args.cell_width if args.cell_width is not None else
                     round(args.cell_width_cm / 2.54 * args.dpi))
    input_width = int(args.input_width)
    input_height = int(args.input_height)
    if input_width <= 0 or input_height <= 0:
        raise ValueError("input width and height must be positive")
    image_height = int(args.image_height if args.image_height is not None else
                       round(cell_width * input_height / input_width))
    label_height = int(args.label_height if args.label_height is not None else
                       round(point_pixels(args.font_size_pt, args.dpi) * 1.25))
    row_gap = int(args.row_gap)
    col_gap = int(args.col_gap)
    label_gap = int(args.label_gap if args.label_gap is not None else
                    round(args.label_gap_cm / 2.54 * args.dpi))
    margin_left = int(args.margin_left if args.margin_left is not None else
                      round(args.margin_cm / 2.54 * args.dpi))
    margin_right = int(args.margin_right if args.margin_right is not None else
                       round(args.margin_cm / 2.54 * args.dpi))
    margin_top = int(args.margin_top if args.margin_top is not None else
                     round(args.margin_cm / 2.54 * args.dpi))
    margin_bottom = int(args.margin_bottom if args.margin_bottom is not None else
                        round(args.margin_cm / 2.54 * args.dpi))
    if min(cell_width, image_height, label_height) <= 0:
        raise ValueError("cell and label dimensions must be positive")

    row_block = image_height + label_gap + label_height
    method_area_height = row_block * 2 + row_gap
    width = margin_left + 5 * cell_width + 4 * col_gap + margin_right
    height = margin_top + method_area_height + margin_bottom
    canvas = Image.new("RGB", (width, height), "white")

    label_px = point_pixels(args.font_size_pt, args.dpi)
    bold_px = point_pixels(args.bold_font_size_pt or args.font_size_pt, args.dpi)
    regular_font = load_font(label_px)
    bold_font = load_font(bold_px, bold=True)
    text_color = tuple(args.text_color)

    # The left cell is one image slot spanning both rows.  It is centred in the
    # combined image area and gets one dataset-index label beneath it.
    x0 = margin_left
    gt_box = paste_contained(canvas, sample / "ground_truth_boxes.png",
                             (x0, margin_top, cell_width, image_height * 2 + row_gap),
                             (input_width, input_height))
    center_text(canvas, image_label, x0, gt_box[1] + gt_box[3] + label_gap,
                cell_width, label_height, regular_font, fill=text_color)

    # Four method columns: bbox on row 1, mask overlay on row 2.
    for index, method in enumerate(METHODS, start=1):
        x = margin_left + index * (cell_width + col_gap)
        bbox_box = paste_contained(canvas, sample / method / "prediction_bbox.png",
                                   (x, margin_top, cell_width, image_height),
                                   (input_width, input_height))
        center_text(canvas, f"IoU {ious[method]:.2f}", x,
                    bbox_box[1] + bbox_box[3] + label_gap, cell_width, label_height,
                    regular_font, fill=text_color)

        y = margin_top + row_block + row_gap
        mask_box = paste_contained(canvas, sample / method / args.mask_file,
                                   (x, y, cell_width, image_height),
                                   (input_width, input_height))
        method_font = bold_font if method == "condgen" else regular_font
        center_text(canvas, METHOD_LABELS[method], x, mask_box[1] + mask_box[3] + label_gap,
                    cell_width, label_height, method_font, fill=text_color)

    output = args.output.expanduser().resolve()
    save_image(canvas, output, args.dpi, args.quality)
    return {
        "sample": str(sample),
        "output": str(output),
        "image_label": image_label,
        "methods": {method: {"iou": ious[method],
                              "bbox": str(sample / method / "prediction_bbox.png"),
                              "mask": str(sample / method / args.mask_file)}
                     for method in METHODS},
        "layout": {"columns": 5, "rows": 2, "cell_width": cell_width,
                   "image_height": image_height, "font_size_pt": args.font_size_pt,
                   "cell_width_cm": cell_width * 2.54 / args.dpi,
                   "input_width": input_width, "input_height": input_height,
                   "input_resize": "direct_resize_before_layout",
                   "label_gap_px": label_gap, "label_gap_cm": label_gap * 2.54 / args.dpi,
                   "mask_file": args.mask_file, "dpi": args.dpi,
                   "font_regular": selected_font_path(False),
                   "font_bold": selected_font_path(True),
                   "font_note": "Microsoft Times New Roman in the active environment when available; otherwise Nimbus Roman fallback"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample", type=Path,
                        help="one VOC/COCO comparison sample directory")
    source.add_argument("--root", type=Path,
                        help="process every immediate sample directory below this root")
    parser.add_argument("--output", type=Path, default=None,
                        help="output .jpg/.png/.pdf path (required with --sample)")
    parser.add_argument("--output-name", default="overview_appendix_layout.jpg",
                        help="filename used for each sample with --root")
    parser.add_argument("--dataset-name", default=None,
                        help="dataset label used in the left caption (default: inferred)")
    parser.add_argument("--image-label", default=None,
                        help="override the left caption, e.g. 'COCO 501038'")
    parser.add_argument("--mask-file", default="mask_overlay.png",
                        help="mask rendering inside each method directory")
    parser.add_argument("--cell-width", type=int, default=None,
                        help="width of every column in pixels; overrides --cell-width-cm")
    parser.add_argument("--cell-width-cm", type=float, default=5.0,
                        help="width of every subfigure in centimetres")
    parser.add_argument("--image-height", type=int, default=None,
                        help="height of each method image slot in pixels; default follows common input ratio")
    parser.add_argument("--input-width", type=int, default=640,
                        help="common width applied to every source rendering before layout")
    parser.add_argument("--input-height", type=int, default=427,
                        help="common height applied to every source rendering before layout")
    parser.add_argument("--label-height", type=int, default=None,
                        help="height reserved for each centred caption; default follows font size")
    parser.add_argument("--row-gap", type=int, default=16,
                        help="vertical gap between the two method rows")
    parser.add_argument("--col-gap", type=int, default=16,
                        help="horizontal gap between columns")
    parser.add_argument("--label-gap", type=int, default=None,
                        help="caption-to-image gap in pixels; overrides --label-gap-cm")
    parser.add_argument("--label-gap-cm", type=float, default=0.10,
                        help="same caption-to-image gap for every cell, in centimetres")
    parser.add_argument("--margin-left", type=int, default=None)
    parser.add_argument("--margin-right", type=int, default=None)
    parser.add_argument("--margin-top", type=int, default=None)
    parser.add_argument("--margin-bottom", type=int, default=None)
    parser.add_argument("--margin-cm", type=float, default=0.20,
                        help="default margin on each side, in centimetres")
    parser.add_argument("--font-size-pt", type=float, default=18,
                        help="caption font size in points")
    parser.add_argument("--bold-font-size-pt", type=float, default=None,
                        help="bold caption size in points (default: --font-size-pt)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality, ignored for PNG/PDF")
    parser.add_argument("--write-metadata", action="store_true",
                        help="write a sibling JSON description of the generated panel")
    args = parser.parse_args()
    args.text_color = (25, 30, 36)
    if args.sample is not None:
        if args.output is None:
            parser.error("--output is required with --sample")
        metadata = make_overview(args)
        if args.write_metadata:
            output = args.output.expanduser().resolve()
            meta = output.with_suffix(".json")
            meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return 0

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"--root does not exist or is not a directory: {root}")
    sample_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not sample_dirs:
        parser.error(f"No sample directories found below --root: {root}")
    outputs = []
    for sample in sample_dirs:
        args.sample = sample
        args.output = sample / args.output_name
        metadata = make_overview(args)
        if args.write_metadata:
            args.output.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        outputs.append(metadata)
    print(json.dumps({"root": str(root), "samples": outputs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
