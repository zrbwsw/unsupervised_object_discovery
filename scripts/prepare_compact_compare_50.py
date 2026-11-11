#!/usr/bin/env python3
"""Export compact nine-image comparison folders for three datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


PROJECT = Path(__file__).resolve().parents[1]
COMPARE = PROJECT / "object_discovery_compare"
DATASETS = {
    "COCO20k": "COCO20k_compare",
    "VOC07": "VOC07_compare",
    "VOC12": "VOC12_compare",
}
METHODS = ("condgen", "mae", "ibot", "dino")
SOURCE_IMAGES = {
    "ground_truth_boxes.png": "ground_truth_boxes.png",
    "condgen_prediction_bbox.png": "condgen/prediction_bbox.png",
    "condgen_mask_overlay.png": "condgen/mask_overlay.png",
    "mae_prediction_bbox.png": "mae/prediction_bbox.png",
    "mae_mask_overlay.png": "mae/mask_overlay.png",
    "ibot_prediction_bbox.png": "ibot/prediction_bbox.png",
    "ibot_mask_overlay.png": "ibot/mask_overlay.png",
    "dino_prediction_bbox.png": "dino/prediction_bbox.png",
    "dino_mask_overlay.png": "dino/mask_overlay.png",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_samples(source: Path):
    samples = []
    for directory in source.iterdir():
        if not directory.is_dir() or not (directory / "metrics.json").is_file():
            continue
        metrics = json.loads((directory / "metrics.json").read_text())
        required = {"rank", "image", "metrics", "dataset"}
        if not required <= metrics.keys():
            raise ValueError(f"Missing fields in {directory / 'metrics.json'}")
        values = metrics["metrics"]
        if any(method not in values for method in METHODS):
            raise ValueError(f"Missing method IoU in {directory / 'metrics.json'}")
        for output_name, relative in SOURCE_IMAGES.items():
            path = directory / relative
            if not path.is_file():
                raise FileNotFoundError(path)
        samples.append((int(metrics["rank"]), directory, metrics))
    samples.sort(key=lambda item: item[0])
    if len(samples) != 50:
        raise ValueError(f"Expected 50 samples in {source}, found {len(samples)}")
    ranks = [rank for rank, _, _ in samples]
    if ranks != list(range(1, 51)):
        raise ValueError(f"Expected ranks 1..50 in {source}, found {ranks}")
    return samples


def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def export_dataset(dataset: str, source_name: str, destination_name: str):
    source = COMPARE / source_name
    destination = COMPARE / destination_name
    samples = read_samples(source)
    destination.mkdir(parents=True, exist_ok=True)
    if any(path.is_dir() for path in destination.iterdir()):
        existing = {path.name for path in destination.iterdir() if path.is_dir()}
        expected = {f"{rank:03d}_" for rank, _, _ in samples}
        if not all(any(name.startswith(prefix) for name in existing) for prefix in expected):
            raise RuntimeError(f"Refusing to mix an unrelated existing destination: {destination}")

    rows = []
    for rank, source_dir, metrics in samples:
        scores = metrics["metrics"]
        image_stem = Path(metrics["image"]).stem
        score_text = [f"{float(scores[method]):.6f}" for method in METHODS]
        folder_name = f"{rank:03d}_{image_stem}_{'_'.join(score_text)}"
        target = destination / folder_name
        target.mkdir(parents=True, exist_ok=True)
        copied = {}
        for output_name, relative in SOURCE_IMAGES.items():
            source_file = source_dir / relative
            target_file = target / output_name
            if target_file.exists() and digest(target_file) != digest(source_file):
                raise RuntimeError(f"Existing file differs from source: {target_file}")
            if not target_file.exists():
                shutil.copy2(source_file, target_file)
            copied[output_name] = digest(target_file)
        rows.append({
            "rank": rank, "dataset": dataset, "image": metrics["image"],
            "directory": folder_name, "source_directory": str(source_dir),
            "condgen_iou": float(scores["condgen"]), "mae_iou": float(scores["mae"]),
            "ibot_iou": float(scores["ibot"]), "dino_iou": float(scores["dino"]),
            "image_count": len(SOURCE_IMAGES), "sha256": json.dumps(copied, sort_keys=True),
        })

    with (destination / "index.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("rank", "dataset", "image", "directory", "source_directory",
                  "condgen_iou", "mae_iou", "ibot_iou", "dino_iou", "image_count", "sha256")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_text(destination / "README.md", f"""# {destination_name}

这是 `{dataset}` 的 50 个 compare 样本精简版，按照原 `object_discovery_compare/{source_name}` 的 rank 排序。
每个子目录命名为：

`顺序索引_数据集索引_CondGen IoU_MAE IoU_iBOT IoU_DINO`

IoU 固定保留 6 位小数，目录名中的方法顺序固定为 CondGen、MAE、iBOT、DINO。
每个子目录只保留 9 张图片：GT 框图，以及四个方法各自的 prediction bbox 和 TokenCut mask overlay。
源目录和全部原始数值结果保持不变；详细映射和源路径见 `index.csv`。

文件名：`ground_truth_boxes.png`、`<method>_prediction_bbox.png`、`<method>_mask_overlay.png`。
""")
    return rows


def main():
    global COMPARE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=COMPARE)
    args = parser.parse_args()
    COMPARE = args.root.resolve()
    all_rows = {}
    for dataset, source_name in DATASETS.items():
        destination_name = {
            "COCO20k": "coco20k_compare_50",
            "VOC07": "voc07_compare_50",
            "VOC12": "voc12_compare_50",
        }[dataset]
        all_rows[dataset] = export_dataset(dataset, source_name, destination_name)
        print(f"{dataset}: exported {len(all_rows[dataset])} samples to {destination_name}")
    write_text(COMPARE / "compact_compare_50_manifest.json",
               json.dumps({"datasets": {k: [{"rank": r["rank"], "directory": r["directory"]}
                                               for r in rows] for k, rows in all_rows.items()}},
                          ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
