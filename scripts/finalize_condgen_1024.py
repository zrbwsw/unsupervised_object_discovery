#!/usr/bin/env python3
"""Validate and assemble the selected CondGen VOC07 1024 qualitative results."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_RANKS = (5, 11, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26,
                  27, 28, 29, 31, 32, 36, 43, 44, 47, 48, 49, 50)
REQUIRED_FILES = (
    "original.jpg", "input_1024.png", "prediction_bbox.png", "mask.png",
    "mask_overlay_nearest.png", "mask_overlay_bilinear.png",
    "mask_overlay_bilinear_alpha0.45.png", "foreground_patch.npy",
    "eigenvector.npy", "prediction.json", "rendering.json",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def opened_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def panel(canvas, path, xy, size, title, text_size=18):
    """Use one layout rule for all panels; native-resolution PNGs remain separate."""
    x, y = xy
    width, height = size
    draw = ImageDraw.Draw(canvas)
    draw.text((x + 10, y + 7), title, font=font(text_size), fill="#242424")
    image = ImageOps.contain(opened_rgb(path), (width - 20, height - 46),
                            Image.Resampling.LANCZOS)
    canvas.paste(image, (x + (width - image.width) // 2,
                         y + 36 + (height - 46 - image.height) // 2))


def validate_sample(root, sample):
    directory = root / sample["directory"]
    if directory.parent.resolve() != root.resolve():
        raise ValueError("Sample directory must be directly inside the collection")
    for filename in REQUIRED_FILES:
        if not (directory / filename).is_file():
            raise ValueError(f"Missing artifact: {directory / filename}")

    prediction = read_json(directory / "prediction.json")
    rendering = read_json(directory / "rendering.json")
    foreground = np.load(directory / "foreground_patch.npy", allow_pickle=False)
    eigenvector = np.load(directory / "eigenvector.npy", allow_pickle=False)
    image = np.asarray(opened_rgb(directory / "input_1024.png"))
    height, width = image.shape[:2]
    grid = ((height + 15) // 16, (width + 15) // 16)
    if min(width, height) != 1024:
        raise ValueError(f"Short side is not 1024: {(width, height)}")
    if foreground.shape != grid or eigenvector.shape != grid:
        raise ValueError(f"Patch grid mismatch: expected {grid}")
    if not np.isin(foreground, (0, 1)).all() or not foreground.any():
        raise ValueError("Patch mask must be nonempty and binary")
    if not np.isfinite(eigenvector).all():
        raise ValueError("Eigenvector contains nonfinite values")
    if tuple(rendering["image_size_wh"]) != (width, height):
        raise ValueError("Image dimensions differ from rendering metadata")
    if tuple(rendering["patch_grid_hw"]) != grid:
        raise ValueError("Grid dimensions differ from rendering metadata")
    if int(rendering["rank"]) != int(sample["rank"]):
        raise ValueError("Source rank differs from rendering metadata")
    if not np.isclose(float(rendering["tau"]), 0.45):
        raise ValueError("Expected CondGen tau=0.45")
    protocol = rendering.get("protocol", {})
    if isinstance(protocol, dict) and protocol.get("patch_size", 16) != 16:
        raise ValueError("Expected patch_size=16")

    full_mask = np.repeat(np.repeat(foreground.astype(bool), 16, axis=0), 16, axis=1)
    full_mask = full_mask[:height, :width]
    with Image.open(directory / "mask.png") as raw_mask:
        saved_mask = np.asarray(raw_mask.convert("L"))
    if saved_mask.shape != (height, width) or not np.isin(saved_mask, (0, 1, 255)).all():
        raise ValueError("Saved mask must have input dimensions and binary values")
    if not np.array_equal(saved_mask > 0, full_mask):
        raise ValueError("Saved mask differs from the raw patch mask expanded by 16")

    rows, columns = np.nonzero(foreground)
    bbox = [int(columns.min()) * 16, int(rows.min()) * 16,
            min((int(columns.max()) + 1) * 16, width),
            min((int(rows.max()) + 1) * 16, height)]
    if not np.allclose(prediction["bbox_xyxy_input"], bbox, rtol=0, atol=1e-5):
        raise ValueError(f"Predicted box differs from patch-mask bounds: {bbox}")

    source = Path(sample["source_directory"])
    if sha256(directory / "original.jpg") != sha256(source / "original.jpg"):
        raise ValueError("Original image copy differs from the selected source image")
    original_width, original_height = opened_rgb(directory / "original.jpg").size
    expected_long_side = 1024 * max(original_width, original_height) / min(original_width, original_height)
    if abs(max(width, height) - expected_long_side) > 1.01:
        raise ValueError("1024 input does not preserve original aspect ratio")
    original_bbox = np.asarray(bbox) / np.asarray(
        [width / original_width, height / original_height] * 2)
    if not np.allclose(prediction["bbox_xyxy_original"], original_bbox, rtol=0, atol=1e-4):
        raise ValueError("Original-coordinate box has incorrect resize scaling")

    nearest_expected = image.astype(np.float32)
    nearest_expected[full_mask] = (0.3 * nearest_expected[full_mask]
                                  + 0.7 * np.array([255.0, 0.0, 0.0], dtype=np.float32))
    nearest_expected = np.clip(nearest_expected, 0, 255).astype(np.uint8)
    nearest_actual = np.asarray(opened_rgb(directory / "mask_overlay_nearest.png"))
    if not np.array_equal(nearest_expected, nearest_actual):
        raise ValueError("Nearest overlay does not reproduce the raw mask with alpha=0.7")
    for filename in ("prediction_bbox.png", "mask_overlay_nearest.png",
                     "mask_overlay_bilinear.png", "mask_overlay_bilinear_alpha0.45.png"):
        if opened_rgb(directory / filename).size != (width, height):
            raise ValueError(f"Wrong image dimensions: {filename}")

    source_metrics = read_json(source / "metrics.json")
    return {
        "rank": int(sample["rank"]), "image": sample["image"],
        "directory": sample["directory"], "passed": True,
        "original_size_wh": [original_width, original_height],
        "image_size_wh": [width, height], "patch_grid_hw": list(grid),
        "bbox_xyxy_input": bbox, "bbox_xyxy_original": original_bbox.tolist(),
        "source_full_resolution_iou": source_metrics["metrics"]["condgen"],
        "runtime_seconds": rendering.get("runtime_seconds"),
        "original_copy_sha256": sha256(directory / "original.jpg"),
        "patch_mask_sha256": sha256(directory / "foreground_patch.npy"),
        "checks": ["original_copy", "short_side_1024", "aspect_ratio", "patch16_grid",
                   "binary_mask", "mask_matches_patch_grid", "finite_eigenvector",
                   "bbox_matches_mask", "bbox_coordinate_scaling", "raw_nearest_pixels",
                   "overlay_dimensions", "rank_and_tau_metadata"],
    }


def create_overview(root, record):
    directory = root / record["directory"]
    canvas = Image.new("RGB", (1280, 1060), "#f7f7f7")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), f"CondGen | VOC07 source rank {record['rank']:03d} | {record['image']}"
              " | tau=0.45 | patch=16", font=font(22), fill="black")
    views = (("input_1024.png", "Input: short side 1024"),
             ("prediction_bbox.png", "Predicted bounding box"),
             ("mask_overlay_nearest.png", "Raw mask: nearest, alpha=0.70"),
             ("mask_overlay_bilinear.png", "Display only: bilinear, alpha=0.70"))
    for i, (filename, title) in enumerate(views):
        panel(canvas, directory / filename, ((i % 2) * 640, 48 + (i // 2) * 490),
              (640, 490), title, text_size=20)
    draw.text((14, 1032), "Overview thumbnails are resized for layout; inspect native PNGs for mask boundaries.",
              font=font(17), fill="#555555")
    canvas.save(directory / "overview.jpg", quality=95, subsampling=0)


def create_contact_sheets(root, records):
    variants = (("prediction_bbox.png", "contact_sheet_bbox.jpg", "Bounding boxes"),
                ("mask_overlay_nearest.png", "contact_sheet_nearest.jpg", "Nearest mask overlays | alpha=0.70"),
                ("mask_overlay_bilinear.png", "contact_sheet_bilinear.jpg", "Bilinear display overlays | alpha=0.70"))
    files = []
    columns, cell_width, cell_height = 4, 360, 306
    row_count = (len(records) + columns - 1) // columns
    for source_file, output_file, title in variants:
        canvas = Image.new("RGB", (columns * cell_width, 66 + row_count * cell_height), "#f7f7f7")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 9), "CondGen / VOC07 / 1024 / patch16 / tau=0.45", font=font(23), fill="black")
        draw.text((12, 38), title + " (rank labels refer to the original full-resolution collection)",
                  font=font(17), fill="#444444")
        for i, record in enumerate(records):
            panel(canvas, root / record["directory"] / source_file,
                  ((i % columns) * cell_width, 66 + (i // columns) * cell_height),
                  (cell_width, cell_height), f"Rank {record['rank']:03d} | {record['image']}")
        canvas.save(root / output_file, quality=95, subsampling=0)
        files.append(output_file)
    return files


def create_index(root, records):
    with (root / "index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("rank", "image", "directory",
            "source_full_resolution_iou", "input_width", "input_height", "patch_rows",
            "patch_columns", "runtime_seconds"))
        writer.writeheader()
        for record in records:
            writer.writerow({
                "rank": record["rank"], "image": record["image"], "directory": record["directory"],
                "source_full_resolution_iou": record["source_full_resolution_iou"],
                "input_width": record["image_size_wh"][0], "input_height": record["image_size_wh"][1],
                "patch_rows": record["patch_grid_hw"][0], "patch_columns": record["patch_grid_hw"][1],
                "runtime_seconds": record["runtime_seconds"],
            })


def create_readme(root, records):
    text = """# VOC07 CondGen 1024 定性展示

本目录对用户选出的 24 张 VOC07 图像重新运行 CondGen ViT-B/16 + TokenCut，固定 `tau=0.45`。
输入图像按原纵横比缩放，使**短边为 1024**；不是强制拉伸为 1024×1024，也不是仅将原有 mask 放大。
编码器使用 final-layer output patch features，patch size 为 16。归一化之后在右侧和下侧补零到 16 的倍数。
TokenCut 保持二值图、弱边权重 `eps=1e-5`、种子所在连通区域，不使用标注生成预测，不添加 CRF 或形态学后处理。

这是独立的 qualitative 展示：使用选出的 24 张图像与新的 1024 输入推理，**不能替代论文表格中的原图分辨率 CorLoc**。
目录名和图中 rank 沿用 `VOC07_condgen_best` 的原有排名；`index.csv` 中的
`source_full_resolution_iou` 仅记录原始评测的 IoU，不是本次 1024 推理的新指标，也没有据本次结果重新排序。
后续与 DINO、iBOT、MAE 的主要对比图使用 bounding box；本目录只展示 CondGen。

## 快速浏览

- [框图总览](contact_sheet_bbox.jpg)
- [原始 nearest 红色 mask 总览](contact_sheet_nearest.jpg)
- [bilinear 红色 mask 总览](contact_sheet_bilinear.jpg)
- [完整索引](index.csv) · [选图记录](selection.json) · [验证结果](validation.json)

每个样本的 `overview.jpg` 左上为 1024 输入，右上为预测框，左下为原始 nearest mask，右下为 bilinear mask。
总览和 overview 为排版缩小的 JPEG，缩略图缩放本身会柔化边缘；判断真实边界请打开各样本的完整分辨率 PNG。

## 每个样本的文件

| 文件 | 含义 |
| --- | --- |
| `original.jpg` | 原始图像副本，已与源文件校验 SHA-256 |
| `input_1024.png` | 真正用于本次推理的缩放后 RGB 图像，归一化和补边前 |
| `prediction_bbox.png` | 在 1024 输入上画出的预测框 |
| `mask.png` | patch 级预测最近邻展开的二值 mask，裁去补边 |
| `foreground_patch.npy` | 原始 patch 网格二值连通区域，无显示插值 |
| `eigenvector.npy` | TokenCut 使用的特征向量网格 |
| `mask_overlay_nearest.png` | **无平滑显示插值**：每个 patch 展开为 16×16 像素，红色 alpha=0.70 |
| `mask_overlay_bilinear.png` | 同一 patch mask 用 bilinear 生成连续 alpha，红色 alpha=0.70 |
| `mask_overlay_bilinear_alpha0.45.png` | 同一 bilinear 显示方案，附加较淡版本 alpha=0.45 |
| `prediction.json` | 1024 输入坐标和映射回原图坐标的预测框 |
| `rendering.json` | 输入尺寸、网格尺寸、推理协议及运行记录 |
| `overview.jpg` | 输入、框图、nearest 与 bilinear 四联图 |

“无插值”指 **mask rendering 不做平滑插值**；输入图像缩放到短边 1024 本身仍需图像重采样。
两个主要 overlay 固定使用相同 `alpha=0.70`，因此差别只来自 mask 显示插值。
bilinear 先将 patch mask 映射到补边后的像素尺寸，再裁去补边；连续值仅控制红色透明度，
不阈值化，不改变原始 mask、预测框或任何评价指标。较淡的 alpha=0.45 版本仅供选图展示。

本次短边 1024 的约定和显示插值已明确记录；DINOv3 论文提到 res. 1024，
但没有公开 Figure 14 的具体缩放边长约定与 mask 显示插值实现，因此不宣称这里逐项复现其渲染代码。

## 样本索引

| 原排名 | 图像 | 1024 输入 W×H | patch 网格 H×W | 原始分辨率 IoU（来源） | 样本 |
| ---: | --- | --- | --- | ---: | --- |
"""
    for record in records:
        width, height = record["image_size_wh"]
        rows, columns = record["patch_grid_hw"]
        text += (f"| {record['rank']} | {record['image']} | {width}×{height} | {rows}×{columns} | "
                 f"{record['source_full_resolution_iou']:.6f} | "
                 f"[四联图]({record['directory']}/overview.jpg) |\n")
    text += """
## 再次生成汇总

推理完成后可在项目根目录运行：

```bash
python scripts/finalize_condgen_1024.py
```

该脚本只读取已保存的推理结果，验证 24 个指定样本、输入尺寸、patch 网格、二值 mask、
mask 与框的一致性、坐标映射、nearest overlay 像素及原图副本，重新生成本目录中的汇总文件。
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=PROJECT / "object_discovery_compare" / "VOC07_condgen_best_1024")
    args = parser.parse_args()
    root = args.root.resolve()
    selection = read_json(root / "selection.json")
    samples = selection["samples"]
    ranks = [int(sample["rank"]) for sample in samples]
    if sorted(ranks) != list(EXPECTED_RANKS):
        raise SystemExit(f"Expected exactly ranks {EXPECTED_RANKS}, found {ranks}")
    if len({sample["directory"] for sample in samples}) != len(samples):
        raise SystemExit("Duplicate sample directory in selection.json")
    records, errors = [], []
    for sample in sorted(samples, key=lambda item: int(item["rank"])):
        try:
            records.append(validate_sample(root, sample))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({"rank": sample["rank"], "directory": sample["directory"], "error": str(error)})
    validation = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "passed": not errors, "expected_sample_count": 24,
                  "validated_sample_count": len(records), "errors": errors,
                  "samples": records}
    write_json(root / "validation.json", validation)
    if errors:
        raise SystemExit(f"Validation failed for {len(errors)} samples; see {root / 'validation.json'}")
    for record in records:
        create_overview(root, record)
    validation["contact_sheets"] = create_contact_sheets(root, records)
    validation["overview_count"] = len(records)
    create_index(root, records)
    create_readme(root, records)
    write_json(root / "validation.json", validation)
    print(f"Validated {len(records)} / 24 samples; wrote overviews, contact sheets, index and README to {root}")


if __name__ == "__main__":
    main()
