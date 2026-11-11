#!/usr/bin/env python3
"""Select examples from saved full evaluations and prepare visualization jobs."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
TAUS = {"dino": "0.30", "ibot": "0.15", "mae": "0.35", "condgen": "0.45"}
DATASETS = {
    "VOC07": (5011, Path("/data0/datasets/dino_downstream/voc2007/VOCdevkit/VOC2007/JPEGImages")),
    "VOC12": (11540, Path("/data0/datasets/dino_object_discovery/voc2012/VOCdevkit/VOC2012/JPEGImages")),
    "COCO20k": (19817, Path("/data0/datasets/dino_object_discovery/coco2014/images/train2014")),
}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def box_image(image, boxes, color):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for box in boxes:
        x0, y0, x1, y1 = box
        # The stored xyxy maximum is exclusive; keep the outer edge visible.
        draw.rectangle((x0, y0, min(x1, image.width - 1), min(y1, image.height - 1)),
                       outline=color, width=3)
    return result


def load_sources(dataset, expected):
    records, sources = {}, {}
    for method, tau in TAUS.items():
        candidates = list((PROJECT / "outputs" / dataset / method / f"tau_{tau}").glob("*/TokenCut-*/predictions.jsonl"))
        if len(candidates) != 1:
            raise ValueError((dataset, method, candidates))
        prediction_path = candidates[0]
        result = json.loads((prediction_path.parent / "results.json").read_text())
        config = json.loads((prediction_path.parent / "run_config.json").read_text())
        assert result["encoder"] == method and result["tau"] == float(tau)
        assert result["evaluated_images"] == expected and not result["subset_evaluation"]
        assert result["which_features"] == "output" and result["eps"] == 1e-5
        assert not result["remove_hards"] and config["args"]["resize"] is None
        assert not config["args"]["no_binary_graph"]
        rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
        keyed = {row["image"]: row for row in rows}
        assert len(keyed) == len(rows) == expected
        assert all(0 <= row["max_iou"] <= 1 and row["correct"] == (row["max_iou"] >= 0.5) for row in rows)
        records[method] = keyed
        sources[method] = {
            "tau": float(tau), "predictions": str(prediction_path),
            "predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            "run_config": str(prediction_path.parent / "run_config.json"),
            "results": str(prediction_path.parent / "results.json"),
            "checkpoint": config["checkpoint"]["checkpoint_path"],
            "corloc_percent": result["corloc_percent"],
        }
    assert all(set(rows) == set(records["condgen"]) for rows in records.values())
    return records, sources


def load_coco_annotations():
    data = json.loads((PROJECT / "TokenCut/datasets/instances_train2014_sel20k.json").read_text())
    images = {row["file_name"]: row for row in data["images"]}
    annotations = defaultdict(list)
    for row in data["annotations"]:
        annotations[row["image_id"]].append(row)
    return images, annotations, data["categories"]


def save_gt(dataset, image_path, sample_dir, image, coco):
    if dataset.startswith("VOC"):
        xml_path = image_path.parent.parent / "Annotations" / (image_path.stem + ".xml")
        shutil.copy2(xml_path, sample_dir / "annotation.xml")
        objects = []
        for obj in ET.parse(xml_path).getroot().findall("object"):
            box = obj.find("bndbox")
            coords = [int(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
            coords[0] -= 1
            coords[1] -= 1
            objects.append({"category": obj.findtext("name"), "bbox_xyxy": coords,
                            "difficult": obj.findtext("difficult"), "truncated": obj.findtext("truncated")})
    else:
        images, annotations, categories = coco
        info = images[image_path.name]
        anns = annotations[info["id"]]
        write_json(sample_dir / "annotation.json", {"image": info, "annotations": anns, "categories": categories})
        names = {cat["id"]: cat["name"] for cat in categories}
        objects = []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            objects.append({"category": names[ann["category_id"]],
                            "bbox_xyxy": [int(round(v)) for v in (x, y, x+w, y+h)]})
    write_json(sample_dir / "ground_truth.json", {"objects": objects, "used_for_inference": False})
    box_image(image, [obj["bbox_xyxy"] for obj in objects], "blue").save(sample_dir / "ground_truth_boxes.png")


def collection_readme(collection, sources, total, whole_image_count):
    kind = collection["kind"]
    count = len(collection["samples"])
    ranking = ("按 CondGen 单图 max IoU 降序；同分按原图文件名升序。" if kind == "condgen_best" else
               "按 gap_sum = (CondGen IoU − DINO IoU) + (CondGen IoU − iBOT IoU) + (CondGen IoU − MAE IoU) 降序；同分按 CondGen IoU 降序、原图文件名升序。")
    text = f"""# {collection['name']}

从 {collection['dataset']} 的 {total} 张完整评测图片中选择前 {count} 张。{ranking}
指标均取自已有 predictions.jsonl 的 max_iou（范围 0–1）；它是预测框与全部有效 GT 框的最大 IoU。单图 CorLoc 成功标记为 max IoU >= 0.5。

固定使用用户指定阈值：DINO=0.30，iBOT=0.15，MAE=0.35，CondGen=0.45；没有对每张图重新选择 tau。
compare 的 gap_sum 范围为 −3 到 3，gap_mean = gap_sum/3，另记录相对最强基线的 gap_min。
这是按 GT 指标筛选的定性示例，不是随机样本或全数据集性能估计；框 IoU 不等同于分割质量。本目录中有 {whole_image_count}/{count} 张 CondGen 预测框覆盖整图，严格保留指标排名而不作外观过滤。

## 图像与协议

原图直接复制；prediction_bbox.png 绘制已有评测框（红色）。mask 由相同 checkpoint 在**原始分辨率**重新运行 TokenCut 得到，要求重算 bbox 与已保存 bbox 完全一致。
使用冻结 ViT-B/16 的最后一层 LayerNorm 后 output patch features、tau 如上、eps=1e-5、二值图，补齐右/下边到 16 的倍数，不使用 CRF、bilateral solver、形态学等后处理。
红色 mask 使用 seed 所在连通分量，按 patch_size 最近邻展开再裁剪；前景颜色为 0.3×原像素+0.7×红色。
本批次没有使用 1024 输入重推理，避免改变排名对应的检测结果；overview/contact_sheet 的缩放仅用于浏览。
GT 只用于排序、核对与单独蓝框展示，不输入模型或 mask 推理。

## 每个样本的材料

- original.jpg：原始 JPEG 文件。
- metrics.json：所有四方法 IoU、成功标记、差值、框、原始结果文件位置。
- ground_truth.json、annotation.xml/annotation.json、ground_truth_boxes.png：评分所用标注和蓝色 GT 框图。
- condgen/ 或四方法子目录：prediction_bbox.png、mask_overlay.png、mask.png（二值前景）、foreground_patch.npy、eigenvector.npy、rendering.json（重推理核对记录）、prediction.json、source_run_config.json。
- overview.jpg：原图和预测框/mask 拼图。

index.csv 保存本目录排名，all_scores.csv 保存全数据集排序，contact_sheet.jpg 便于浏览所选 50 张。来源详情见 sources.json。

| 方法 | tau | 全量 CorLoc (%) |
| --- | ---: | ---: |
"""
    for method, source in sources.items():
        text += f"| {method} | {source['tau']:.2f} | {source['corloc_percent']:.4f} |\n"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "object_discovery_compare")
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    jobs, collections = {}, []
    coco = load_coco_annotations()
    for dataset, (expected, image_root) in DATASETS.items():
        records, sources = load_sources(dataset, expected)
        assert args.top_k <= expected
        scores = []
        for image_name, condgen in records["condgen"].items():
            metrics = {m: r[image_name]["max_iou"] for m, r in records.items()}
            gaps = {m: metrics["condgen"] - metrics[m] for m in ("dino", "ibot", "mae")}
            scores.append({"image": image_name, **metrics, "gap_sum": sum(gaps.values()),
                           "gap_mean": sum(gaps.values()) / 3, "gap_min": min(gaps.values())})
        for kind in ("condgen_best", "compare"):
            name = f"{dataset}_{kind}"
            directory = root / name
            directory.mkdir()
            ranked = sorted(scores, key=(lambda s: (-s["condgen"], s["image"])) if kind == "condgen_best"
                            else (lambda s: (-s["gap_sum"], -s["condgen"], s["image"])))
            with (directory / "all_scores.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["rank", *ranked[0]])
                writer.writeheader()
                writer.writerows({"rank": i, **score} for i, score in enumerate(ranked, 1))
            collection = {"name": name, "dataset": dataset, "kind": kind, "samples": []}
            index = []
            whole_image_count = 0
            for rank, score in enumerate(ranked[:args.top_k], 1):
                filename = score["image"]
                metric_tag = f"iou_{score['condgen']:.6f}" if kind == "condgen_best" else f"gap_{score['gap_sum']:.6f}"
                sample_dir = directory / f"{rank:03d}_{metric_tag}_{Path(filename).stem}"
                sample_dir.mkdir()
                image_path = image_root / filename
                shutil.copy2(image_path, sample_dir / "original.jpg")
                with Image.open(image_path) as original:
                    image = original.convert("RGB")
                assert [3, image.height, image.width] == records["condgen"][filename]["image_shape_chw"]
                whole_image_count += records["condgen"][filename]["bbox_xyxy"] == [0, 0, image.width, image.height]
                save_gt(dataset, image_path, sample_dir, image, coco)
                sample = {"rank": rank, "directory": str(sample_dir), "image": filename,
                          "metrics": {m: score[m] for m in TAUS}, "gap_sum": score["gap_sum"],
                          "gap_mean": score["gap_mean"], "gap_min": score["gap_min"],
                          "taus": {m: float(t) for m, t in TAUS.items()}}
                write_json(sample_dir / "metrics.json", {
                    **sample, "dataset": dataset, "image_path": str(image_path),
                    "gap_by_method": {m: score["condgen"] - score[m] for m in ("dino", "ibot", "mae")},
                    "predictions": {m: records[m][filename] for m in TAUS}, "sources": sources,
                })
                for method in (("condgen",) if kind == "condgen_best" else tuple(TAUS)):
                    method_dir = sample_dir / method
                    method_dir.mkdir()
                    prediction = records[method][filename]
                    box_image(image, [prediction["bbox_xyxy"]], "red").save(method_dir / "prediction_bbox.png")
                    write_json(method_dir / "prediction.json", prediction)
                    shutil.copy2(sources[method]["run_config"], method_dir / "source_run_config.json")
                    key = (dataset, method, filename)
                    if key not in jobs:
                        jobs[key] = {"dataset": dataset, "image": filename, "image_path": str(image_path),
                                     "method": method, "tau": float(TAUS[method]),
                                     "checkpoint": sources[method]["checkpoint"],
                                     "bbox_xyxy": prediction["bbox_xyxy"],
                                     "image_shape_chw": prediction["image_shape_chw"], "output_dirs": []}
                    jobs[key]["output_dirs"].append(str(method_dir))
                collection["samples"].append(sample)
                index.append({"rank": rank, "directory": sample_dir.name, **score})
            with (directory / "index.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(index[0]))
                writer.writeheader()
                writer.writerows(index)
            write_json(directory / "sources.json", sources)
            (directory / "README.md").write_text(collection_readme(collection, sources, expected, whole_image_count))
            collections.append(collection)
            print(f"Prepared {name}: {args.top_k} examples", flush=True)
    write_json(root / "selection.json", {"top_k": args.top_k, "taus": TAUS, "collections": collections})
    write_json(root / "render_manifest.json", {"tasks": list(jobs.values())})
    (root / "README.md").write_text("""# Object discovery comparison

三个数据集各有 condgen_best、compare 两组，每组 50 张，共 300 个样本目录。
固定 tau：DINO=0.30、iBOT=0.15、MAE=0.35、CondGen=0.45。

condgen_best 按 CondGen 的 max IoU 降序；compare 按三项 (CondGen IoU − 基线 IoU) 之和降序。
这是使用标注指标挑选的定性示例；IoU 是框指标，不是 mask 指标，整图框也可能排在前列。
详见各目录 README.md、index.csv、all_scores.csv。原图、红框、红色 mask 及原始数值材料均保存在样本目录中。
mask 使用与全量评测相同的原分辨率推理，重算框必须匹配既有框，未额外缩放到 1024 或进行分割后处理。

## 查看

打开各目录 contact_sheet.jpg 快速浏览；进入样本目录查看 overview.jpg。
compare 样本的四方法子目录各含独立预测框、红色 mask、二值 mask、patch mask、eigenvector、源评测配置与重推理记录。
validation.json 记录导出完成状态和核验结果。selection.json 记录筛选，render_manifest.json 记录需要重新推理的去重任务。

## 复现

使用项目 scripts/prepare_object_discovery_compare.py --root NEW_DIRECTORY 生成索引、原图和预测框（拒绝覆盖已有目录）。
然后对四个 method 各运行 scripts/render_object_discovery_compare.py --manifest NEW_DIRECTORY/render_manifest.json --method METHOD --device cuda。
最后运行 scripts/finalize_object_discovery_compare.py --root NEW_DIRECTORY 生成拼图并验证。使用现有 irepa Python 环境，设置 OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS=1。
""")
    print(f"Prepared {len(collections)} collections, {sum(len(c['samples']) for c in collections)} sample directories, {len(jobs)} unique inference tasks")


if __name__ == "__main__":
    main()
