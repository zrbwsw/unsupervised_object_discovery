#!/usr/bin/env python3
"""Check the local VOC07/TokenCut setup on CPU without running an evaluation."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT / "TokenCut"


@contextmanager
def repository_context():
    previous_directory = Path.cwd()
    previous_path = sys.path[:]
    try:
        os.chdir(REPO_ROOT)
        sys.path.insert(0, str(REPO_ROOT))
        yield
    finally:
        os.chdir(previous_directory)
        sys.path[:] = previous_path


def package_versions():
    versions = {}
    for name in (
        "torch", "torchvision", "numpy", "scipy", "Pillow", "tqdm",
        "scikit-image", "matplotlib", "opencv-python", "timm", "pycocotools",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=TASK_ROOT / "reports" / "setup_check.json",
        help="JSON report path (default: task-root/reports/setup_check.json)",
    )
    args = parser.parse_args()
    report_path = args.report.resolve()

    # These settings affect this short-lived check only, not the user's shell.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    sys.dont_write_bytecode = True
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "repository": str(REPO_ROOT),
        "scope": "CPU setup checks only; no checkpoint loading or benchmark evaluation",
        "packages": package_versions(),
        "checks": {},
    }
    report["optional_dependencies"] = {
        "pycocotools": {
            "installed": report["packages"]["pycocotools"] is not None,
            "needed_for": "COCO loading, not VOC07",
        },
    }

    def check(name, operation):
        try:
            details = operation()
            report["checks"][name] = {"status": "passed", **details}
        except Exception as error:
            report["checks"][name] = {
                "status": "failed", "error": f"{type(error).__name__}: {error}",
            }

    def check_imports():
        loaded = {}
        for name in ("networks", "object_discovery", "datasets", "visualizations"):
            module = importlib.import_module(name)
            loaded[name] = str(Path(module.__file__).resolve())
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        return {"modules": loaded, "torch_runtime_version": torch.__version__}

    def check_voc():
        import torch
        from datasets import Dataset

        dataset = Dataset("VOC07", "trainval", False)
        count = len(dataset.dataloader)
        if count != 5011:
            raise ValueError(f"Expected 5011 VOC07 trainval images, found {count}")
        samples = []
        for index in (0, count - 1):
            tensor, annotation = dataset.dataloader[index]
            name = dataset.get_image_name(annotation)
            boxes, classes = dataset.extract_gt(annotation, name)
            image = dataset.load_image(name)
            if tensor.ndim != 3 or tensor.shape[0] != 3:
                raise ValueError(f"Unexpected image tensor shape for {name}: {tensor.shape}")
            if tuple(image.shape[:2]) != tuple(tensor.shape[1:]):
                raise ValueError(f"Image loader paths disagree for {name}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite image values for {name}")
            if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) == 0:
                raise ValueError(f"Invalid ground-truth boxes for {name}")
            if not (boxes[:, 2:] > boxes[:, :2]).all():
                raise ValueError(f"Non-positive ground-truth box area for {name}")
            samples.append({
                "index": index, "filename": name, "image_shape_chw": list(tensor.shape),
                "gt_count": len(boxes), "gt_boxes_xyxy": boxes.tolist(), "classes": classes,
            })
        return {
            "dataset": dataset.name,
            "resolved_voc_directory": str((REPO_ROOT / dataset.root_path / "VOCdevkit" / "VOC2007").resolve()),
            "image_count": count, "remove_hards": False, "samples": samples,
        }

    def check_box_convention():
        import torch
        from datasets import bbox_iou, extract_gt_VOC

        annotation = {"annotation": {"object": [{
            "name": "synthetic", "truncated": "0", "difficult": "0",
            "bndbox": {"xmin": "1", "ymin": "1", "xmax": "10", "ymax": "20"},
        }]}}
        boxes, _ = extract_gt_VOC(annotation)
        if boxes.tolist() != [[0, 0, 10, 20]]:
            raise ValueError(f"Unexpected VOC coordinate conversion: {boxes.tolist()}")
        overlap = float(bbox_iou(torch.tensor([0, 0, 10, 20]), torch.from_numpy(boxes))[0])
        if abs(overlap - 1.0) > 1e-6:
            raise ValueError(f"Unexpected identical-box IoU: {overlap}")
        return {"converted_xyxy": boxes.tolist()[0], "identical_box_iou": overlap}

    def check_toy_graph():
        import numpy as np
        import torch
        from object_discovery import ncut

        # Upstream ncut currently requires a leading CLS token; its zero value is ignored.
        features = torch.zeros(1, 7, 4)
        features[0, 1:4, 0] = 1
        features[0, 4:, 1] = 1
        box, _, mask, seed, _, eigenvector = ncut(
            features, [2, 3], [16, 16], torch.Size([3, 31, 47]), tau=0.2,
        )
        if not np.isfinite(box).all() or not np.isfinite(eigenvector).all():
            raise ValueError("Non-finite toy graph output")
        if not (0 <= box[0] < box[2] <= 47 and 0 <= box[1] < box[3] <= 31):
            raise ValueError(f"Toy bounding box lies outside original image: {box.tolist()}")
        if mask.shape != (2, 3) or eigenvector.shape != (2, 3):
            raise ValueError("Toy mask or eigenvector shape differs from patch grid")
        return {"patch_grid_hw": [2, 3], "bbox_xyxy": box.tolist(), "seed": int(seed)}

    def check_rectangular_tokens():
        import torch
        from dino.vision_transformer import VisionTransformer

        # This tiny random model checks the existing API, not any of the four encoders.
        model = VisionTransformer(
            img_size=[32], patch_size=16, embed_dim=24, depth=2, num_heads=3,
        ).eval()
        with torch.inference_mode():
            tokens = model.get_intermediate_layers(torch.zeros(1, 3, 32, 48), n=1)[0]
        if tuple(tokens.shape) != (1, 7, 24) or not torch.isfinite(tokens).all():
            raise ValueError(f"Unexpected rectangular output tokens: {tuple(tokens.shape)}")
        return {"input_shape": [1, 3, 32, 48], "tokens_with_cls_shape": list(tokens.shape)}

    def check_cli_help():
        completed = subprocess.run(
            [sys.executable, "-B", str(REPO_ROOT / "main_tokencut.py"), "--help"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=45,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"Exit code {completed.returncode}")
        if "--which_features" not in completed.stdout:
            raise ValueError("Expected TokenCut CLI options missing")
        return {"exit_code": completed.returncode}

    try:
        with repository_context():
            check("imports", check_imports)
            if report["checks"]["imports"]["status"] == "passed":
                check("voc07_trainval", check_voc)
                check("voc_box_convention", check_box_convention)
                check("tokencut_toy_graph", check_toy_graph)
                check("rectangular_output_tokens", check_rectangular_tokens)
                check("upstream_cli_help", check_cli_help)
    except Exception as error:
        report["checks"]["repository_access"] = {
            "status": "failed", "error": f"{type(error).__name__}: {error}",
        }

    report["success"] = all(item["status"] == "passed" for item in report["checks"].values())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, result in report["checks"].items():
        print(f"{name}: {result['status']}" + (f" ({result['error']})" if "error" in result else ""))
    print(f"Report: {report_path}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
