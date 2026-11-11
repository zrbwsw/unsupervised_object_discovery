#!/usr/bin/env python3
"""Compare display interpolation using the same saved masks for all methods."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "TokenCut"))
from visualizations import visualize_mask_overlay

METHODS = ("dino", "ibot", "mae", "condgen")
VARIANTS = (("nearest", 0.7), ("bilinear", 0.7), ("bilinear", 0.45))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sample = args.sample.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    methods = ("condgen",) if sample.parent.name.endswith("_condgen_best") else METHODS
    protected = [sample / "original.jpg", sample / "metrics.json"]
    for method in methods:
        protected.extend(sample / method / filename for filename in
                         ("foreground_patch.npy", "mask.png", "mask_overlay.png", "prediction.json", "rendering.json"))
    before = {str(path): digest(path) for path in protected}
    with Image.open(sample / "original.jpg") as source:
        rgb = np.asarray(source.convert("RGB"))
    height, width = rgb.shape[:2]
    font = ImageFont.truetype("DejaVuSans.ttf", 19)
    canvas = Image.new("RGB", (3 * 516, 48 + len(methods) * 383), "white")
    draw = ImageDraw.Draw(canvas)
    titles = ("Original: nearest, alpha=0.70", "Display: bilinear, alpha=0.70", "Display: bilinear, alpha=0.45")
    for column, title in enumerate(titles):
        draw.text((column * 516 + 12, 14), title, font=font, fill="black")
    records = []
    for row, method in enumerate(methods):
        mask_path = sample / method / "foreground_patch.npy"
        foreground = np.load(mask_path)
        for column, (interpolation, alpha) in enumerate(VARIANTS):
            rendered = visualize_mask_overlay(rgb, foreground, "", "", patch_size=16,
                                              alpha=alpha, save=False, interpolation=interpolation)
            if interpolation == "nearest":
                with Image.open(sample / method / "mask_overlay.png") as original:
                    assert np.array_equal(rendered, np.asarray(original.convert("RGB"))), method
            filename = f"{method}_{interpolation}_alpha{alpha:.2f}.png"
            Image.fromarray(rendered).save(output / filename)
            thumb = ImageOps.contain(Image.fromarray(rendered), (500, 333), Image.Resampling.LANCZOS)
            x = column * 516 + 8 + (500 - thumb.width) // 2
            y = 48 + row * 383 + (333 - thumb.height) // 2
            canvas.paste(thumb, (x, y))
            draw.text((column * 516 + 12, 48 + row * 383 + 342), method, font=font, fill="black")
            records.append({"method": method, "file": filename, "interpolation": interpolation,
                            "alpha": alpha, "source_mask_sha256": before[str(mask_path)]})
    assert before == {str(path): digest(path) for path in protected}, "Source artifacts changed"
    canvas.save(output / "comparison.jpg", quality=95, subsampling=0)
    (output / "rendering.json").write_text(json.dumps({
        "sample": str(sample), "image_size_wh": [width, height], "patch_size": 16,
        "display_only": True, "inference_rerun": False, "sources_unchanged": True,
        "raw_overlay_pixel_match": True, "variants": records,
        "bilinear_alignment": "resize patch grid to padded dimensions, then crop to image dimensions",
    }, indent=2) + "\n")
    (output / "README.md").write_text("""# Mask overlay display comparison

同一张原图和同一批保存的 patch masks，四个方法统一渲染三列：

1. 原始 nearest、alpha=0.70（已逐像素核对与已有 overlay 一致）。
2. bilinear soft alpha、alpha=0.70（只改变显示插值）。
3. bilinear soft alpha、alpha=0.45（降低红色强度）。

打开 comparison.jpg 查看对照。各 PNG 保持原图分辨率。
bilinear 先映射到补边尺寸再裁剪，连续值仅作为透明度，不阈值化或重算框。
这是显示插值示例；没有增加 mask 的真实空间信息，不是 DINOv3 1024 输入推理复现。
源 patch mask、二值 mask、预测框、指标和既有 overlay 均经哈希校验未改动。
若论文采用该显示方式，应注明所有方法统一使用 bilinear alpha interpolation for visualization only。
""")
    print(f"Saved display comparison: {output / 'comparison.jpg'}")
    print("Original overlays match pixel-for-pixel; all source hashes unchanged.")


if __name__ == "__main__":
    main()
