#!/usr/bin/env python3
"""Validate selected TokenCut materials and build compact visual contact sheets.

This script never runs inference and never modifies the original experiments.
It consumes ``selection.json`` and the rendered sample files beneath ``--root``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps


METHODS = ("dino", "ibot", "mae", "condgen")
LABELS = {"dino": "DINO", "ibot": "iBOT", "mae": "MAE", "condgen": "CondGen"}
TAUS = {"dino": 0.30, "ibot": 0.15, "mae": 0.35, "condgen": 0.45}
METHOD_FILES = (
    "prediction_bbox.png",
    "mask_overlay.png",
    "mask.png",
    "foreground_patch.npy",
    "eigenvector.npy",
    "rendering.json",
    "prediction.json",
    "source_run_config.json",
)
IMAGE_FILES = ("prediction_bbox.png", "mask_overlay.png", "mask.png")
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def font(size: int) -> ImageFont.FreeTypeFont:
    """Use a readable installed font, with Pillow's bundled fallback if needed."""
    try:
        return ImageFont.truetype(str(FONT_PATH), size)
    except OSError:
        return ImageFont.truetype("DejaVuSans.ttf", size)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def is_compare(collection: dict) -> bool:
    kind = str(collection["kind"]).lower()
    if kind in ("compare", "comparison"):
        return True
    if kind in ("condgen_best", "best"):
        return False
    raise ValueError(f"Unknown collection kind: {collection['kind']!r}")


def sample_dir(root: Path, sample: dict) -> Path:
    directory = Path(sample["directory"])
    if not directory.is_absolute():
        directory = root / directory
    directory = directory.resolve()
    if not directory.is_relative_to(root):
        raise ValueError(f"Sample directory lies outside --root: {directory}")
    return directory


def metric(sample: dict, method: str) -> float:
    value = sample["metrics"][method]
    # The canonical selection stores floats; accept an explicit metric object too.
    if isinstance(value, dict):
        value = value["max_iou"]
    return float(value)


def inspect_materials(root: Path, collections: list[dict]) -> tuple[list[dict], list[dict]]:
    """Validate every required file before publishing any successful summary."""
    errors = []
    summaries = []
    for collection in collections:
        name = collection["name"]
        methods = METHODS if is_compare(collection) else ("condgen",)
        samples = collection["samples"]
        summary = {"name": name, "dataset": collection["dataset"],
                   "kind": collection["kind"], "samples": len(samples),
                   "method_renderings": 0, "bbox_matches_saved": 0,
                   "required_files": 0, "present_files": 0, "verified_images": 0}
        if len(samples) != 50:
            errors.append({"collection": name, "error": "Expected 50 selected samples",
                           "found": len(samples)})
        image_names = [sample["image"] for sample in samples]
        if len(image_names) != len(set(image_names)):
            errors.append({"collection": name, "error": "Duplicate selected image names"})
        for sample in samples:
            directory = sample_dir(root, sample)
            paths = [directory / "original.jpg"]
            images = [directory / "original.jpg"]
            for method in methods:
                # Fail explicitly if the selection and rendering use different methods.
                metric(sample, method)
                method_dir = directory / method
                paths.extend(method_dir / filename for filename in METHOD_FILES)
                images.extend(method_dir / filename for filename in IMAGE_FILES)
                rendering_file = method_dir / "rendering.json"
                summary["method_renderings"] += 1
                if rendering_file.is_file():
                    try:
                        rendering = json.loads(rendering_file.read_text())
                        if rendering.get("bbox_matches_saved") is True:
                            summary["bbox_matches_saved"] += 1
                        else:
                            errors.append({"file": str(rendering_file),
                                           "error": "bbox_matches_saved must be true"})
                    except (OSError, ValueError) as exc:
                        errors.append({"file": str(rendering_file), "error": str(exc)})
                for json_file in (method_dir / "prediction.json", method_dir / "source_run_config.json"):
                    if json_file.is_file():
                        try:
                            json.loads(json_file.read_text())
                        except (OSError, ValueError) as exc:
                            errors.append({"file": str(json_file), "error": str(exc)})
            summary["required_files"] += len(paths)
            for path in paths:
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append({"file": str(path), "error": "Missing or empty required file"})
                else:
                    summary["present_files"] += 1
            for path in images:
                if path.is_file():
                    try:
                        with Image.open(path) as image:
                            image.verify()
                        summary["verified_images"] += 1
                    except (OSError, ValueError) as exc:
                        errors.append({"file": str(path), "error": f"Invalid image: {exc}"})
        summaries.append(summary)
    return summaries, errors


def put_image(canvas: Image.Image, path: Path, box: tuple[int, int, int, int]) -> None:
    """Fit and center an image without cropping or changing its aspect ratio."""
    x, y, width, height = box
    with Image.open(path) as source:
        image = ImageOps.contain(source.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
    canvas.paste(image, (x + (width - image.width) // 2, y + (height - image.height) // 2))


def draw_text(canvas: Image.Image, text: str, box: tuple[int, int, int, int], size: int = 22) -> None:
    """Center a single label and reduce the font size if the label is too wide."""
    x, y, width, height = box
    draw = ImageDraw.Draw(canvas)
    text_font = font(size)
    while size > 11 and draw.textlength(text, font=text_font) > width - 12:
        size -= 1
        text_font = font(size)
    bounds = draw.textbbox((0, 0), text, font=text_font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    draw.text((x + (width - text_width) / 2, y + (height - text_height) / 2 - bounds[1]),
              text, fill=(25, 30, 36), font=text_font)


def save_jpeg(canvas: Image.Image, path: Path) -> None:
    if max(canvas.size) > 2048:
        canvas = ImageOps.contain(canvas, (2048, 2048), Image.Resampling.LANCZOS)
    temporary = path.with_name(path.name + ".tmp")
    canvas.save(temporary, format="JPEG", quality=94, subsampling=0)
    temporary.replace(path)


def make_overview(root: Path, collection: dict, sample: dict, rank: int) -> None:
    directory = sample_dir(root, sample)
    compare = is_compare(collection)
    title = f"{collection['dataset']} | #{rank:03d} | {sample['image']}"
    if compare:
        width, margin, gap, cell_width = 2048, 24, 16, 488
        image_height, label_height = 292, 38
        canvas = Image.new("RGB", (width, 1112), "white")
        draw_text(canvas, title, (0, 4, width, 42), 26)
        draw_text(canvas, f"CondGen total IoU advantage: {float(sample['gap_sum']):+.6f}",
                  (0, 46, width, 36), 23)
        put_image(canvas, directory / "original.jpg", (margin, 88, width - 2 * margin, 270))
        draw_text(canvas, "Original image", (0, 360, width, label_height), 22)
        for index, method in enumerate(METHODS):
            x = margin + index * (cell_width + gap)
            label = f"{LABELS[method]} | tau {TAUS[method]:.2f} | IoU {metric(sample, method):.4f}"
            draw_text(canvas, label, (x, 402, cell_width, label_height), 22)
            put_image(canvas, directory / method / "prediction_bbox.png",
                      (x, 444, cell_width, image_height))
            draw_text(canvas, f"{LABELS[method]} | predicted box", (x, 738, cell_width, 30), 20)
            put_image(canvas, directory / method / "mask_overlay.png",
                      (x, 774, cell_width, image_height))
            draw_text(canvas, f"{LABELS[method]} | red mask overlay", (x, 1070, cell_width, 30), 20)
    else:
        width, margin, gap, cell_width = 2048, 24, 16, 656
        canvas = Image.new("RGB", (width, 694), "white")
        draw_text(canvas, title, (0, 4, width, 42), 26)
        draw_text(canvas, f"CondGen | tau {TAUS['condgen']:.2f} | max IoU {metric(sample, 'condgen'):.6f}",
                  (0, 46, width, 36), 23)
        tiles = ((directory / "original.jpg", "Original image"),
                 (directory / "condgen" / "prediction_bbox.png", "CondGen predicted box"),
                 (directory / "condgen" / "mask_overlay.png", "CondGen red mask overlay"))
        for index, (path, label) in enumerate(tiles):
            x = margin + index * (cell_width + gap)
            put_image(canvas, path, (x, 96, cell_width, 542))
            draw_text(canvas, label, (x, 644, cell_width, 40), 24)
    save_jpeg(canvas, directory / "overview.jpg")


def make_contact_sheet(root: Path, collection: dict) -> Path:
    """Use CondGen overlays so the 50-image contact sheet remains readable."""
    samples = collection["samples"]
    columns, cell_width, cell_height = 5, 384, 180
    margin, gap, top = 24, 12, 100
    rows = (len(samples) + columns - 1) // columns
    width = 2 * margin + columns * cell_width + (columns - 1) * gap
    height = top + rows * cell_height + (rows - 1) * gap + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw_text(canvas, collection["name"], (0, 8, width, 40), 29)
    compare = is_compare(collection)
    subtitle = ("CondGen masks, ranked by summed IoU advantage; open overview.jpg for all methods"
                if compare else "CondGen masks, ranked by max IoU; tau = 0.45")
    draw_text(canvas, subtitle, (0, 52, width, 32), 23)
    for index, sample in enumerate(samples):
        directory = sample_dir(root, sample)
        row, column = divmod(index, columns)
        x = margin + column * (cell_width + gap)
        y = top + row * (cell_height + gap)
        put_image(canvas, directory / "condgen" / "mask_overlay.png", (x, y, cell_width, 132))
        score = (f"sum gap {float(sample['gap_sum']):+.4f}" if compare
                 else f"IoU {metric(sample, 'condgen'):.4f}")
        draw_text(canvas, f"#{index + 1:03d} | {score}", (x, y + 132, cell_width, 25), 21)
        image_id = Path(sample["image"]).stem.removeprefix("COCO_train2014_")
        draw_text(canvas, image_id, (x, y + 158, cell_width, 21), 18)
    collection_dir = root / collection["name"]
    collection_dir.mkdir(parents=True, exist_ok=True)
    output = collection_dir / "contact_sheet.jpg"
    save_jpeg(canvas, output)
    return output


def finalize(root: Path) -> int:
    selection_file = root / "selection.json"
    selection = json.loads(selection_file.read_text())
    collections = selection["collections"]
    if not collections:
        raise ValueError("selection.json contains no collections")
    names = [collection["name"] for collection in collections]
    if len(names) != len(set(names)):
        raise ValueError("selection.json contains duplicate collection names")
    summaries, errors = inspect_materials(root, collections)
    totals = Counter()
    for summary in summaries:
        for key in ("samples", "method_renderings", "bbox_matches_saved", "required_files",
                    "present_files", "verified_images"):
            totals[key] += summary[key]
    validation = {"status": "incomplete", "generated_at": datetime.now(timezone.utc).isoformat(),
                  "selection_file": str(selection_file), "collections": summaries,
                  "totals": dict(totals), "errors": errors}
    if errors:
        write_json(root / "validation.json", validation)
        print(f"Validation failed: {len(errors)} problem(s); see {root / 'validation.json'}", file=sys.stderr)
        return 1
    try:
        for collection, summary in zip(collections, summaries):
            for rank, sample in enumerate(collection["samples"], start=1):
                make_overview(root, collection, sample, rank)
            contact_sheet = make_contact_sheet(root, collection)
            summary["overviews"] = len(collection["samples"])
            summary["contact_sheet"] = str(contact_sheet)
            print(f"Finalized {collection['name']}: {len(collection['samples'])} samples", flush=True)
    except Exception as exc:
        validation["errors"].append({"error": f"Contact sheet generation failed: {exc}"})
        write_json(root / "validation.json", validation)
        raise
    validation["status"] = "complete"
    validation["totals"]["overviews"] = sum(item["overviews"] for item in summaries)
    validation["totals"]["contact_sheets"] = len(collections)
    write_json(root / "validation.json", validation)
    print(f"Validation complete: {totals['samples']} samples, {totals['method_renderings']} method renderings")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="object_discovery_compare directory")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    try:
        return finalize(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Could not finalize materials: {exc}", file=sys.stderr)
        # Avoid retaining a stale successful validation if the selection is malformed.
        if root.is_dir():
            write_json(root / "validation.json", {
                "status": "incomplete", "generated_at": datetime.now(timezone.utc).isoformat(),
                "errors": [{"error": str(exc)}],
            })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
