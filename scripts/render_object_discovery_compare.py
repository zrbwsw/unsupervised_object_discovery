#!/usr/bin/env python3
"""Recompute selected TokenCut masks at the original evaluation resolution.

The manifest contains a ``tasks`` list.  Each selected task supplies an image,
checkpoint, archived prediction and one or more output directories.  Mask
generation must reproduce the archived bounding box before any result is saved.
"""

import os

# Set these before importing NumPy/SciPy/PyTorch.  Every renderer is one worker.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT / "TokenCut"
ASSET_NAMES = (
    "mask_overlay.png", "mask.png", "foreground_patch.npy", "eigenvector.npy",
)
PATCH_SIZE = 16


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_task(task):
    required = ("dataset", "image", "image_path", "method", "tau", "checkpoint",
                "bbox_xyxy", "image_shape_chw", "output_dirs")
    missing = [name for name in required if name not in task]
    if missing:
        raise ValueError(f"Manifest task lacks fields {missing}: {task}")
    normalized = dict(task)
    for name in ("image_path", "checkpoint"):
        path = Path(task[name]).expanduser()
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"{name} must be an existing absolute file path: {path}")
        normalized[name] = str(path.resolve())
    if len(task["bbox_xyxy"]) != 4 or len(task["image_shape_chw"]) != 3:
        raise ValueError(f"Invalid archived bbox or image shape: {task['image']}")
    if task["image_shape_chw"][0] != 3:
        raise ValueError(f"Expected an RGB input: {task['image']}")
    normalized["tau"] = float(task["tau"])
    directories = []
    for value in task["output_dirs"]:
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(f"output_dirs must contain absolute paths: {path}")
        path = path.resolve()
        if path not in directories:
            directories.append(path)
    if not directories:
        raise ValueError(f"No output directories for {task['image']}")
    normalized["output_dirs"] = directories
    return normalized


def task_identity(task):
    return {name: task[name] for name in (
        "dataset", "image", "image_path", "method", "tau", "checkpoint",
        "bbox_xyxy", "image_shape_chw",
    )}


def cached_rendering(directory, task, checkpoint_sha256, renderer_sha256):
    """Return a complete compatible cache, rejecting a mismatched saved result."""
    metadata_path = directory / "rendering.json"
    if not metadata_path.is_file():
        return None
    with metadata_path.open() as stream:
        metadata = json.load(stream)
    if metadata.get("task") != task_identity(task):
        raise ValueError(f"Cached rendering belongs to a different task: {directory}")
    expected = task["bbox_xyxy"]
    if (metadata.get("bbox_match") is not True
            or metadata.get("expected_bbox_xyxy") != expected
            or metadata.get("recomputed_bbox_xyxy") != expected):
        raise ValueError(f"Cached rendering does not reproduce the archived bbox: {directory}")
    if metadata.get("checkpoint", {}).get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"Checkpoint changed since rendering: {directory}")
    if metadata.get("source", {}).get("renderer_sha256") != renderer_sha256:
        # Recompute when this renderer changes, instead of silently reusing old assets.
        return None
    if not all((directory / name).is_file() and (directory / name).stat().st_size
               for name in ASSET_NAMES):
        return None
    return metadata


def copy_rendering(source, destination):
    """Make independent copies and publish the completion metadata last."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in ASSET_NAMES:
        shutil.copy2(source / name, destination / name)
    temporary = destination / f".rendering.{os.getpid()}.json"
    shutil.copy2(source / "rendering.json", temporary)
    temporary.replace(destination / "rendering.json")


def write_rendering(directory, overlay, full_mask, foreground, eigenvector, metadata,
                    np, Image):
    directory.mkdir(parents=True, exist_ok=True)
    # A stale completion marker must never describe a partially rewritten result.
    (directory / "rendering.json").unlink(missing_ok=True)
    Image.fromarray(overlay).save(directory / "mask_overlay.png")
    Image.fromarray(full_mask).save(directory / "mask.png")
    np.save(directory / "foreground_patch.npy", foreground)
    np.save(directory / "eigenvector.npy", eigenvector)
    temporary = directory / f".rendering.{os.getpid()}.json"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(directory / "rendering.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", choices=("dino", "ibot", "mae", "condgen"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    tasks = [normalize_task(task) for task in manifest["tasks"] if task["method"] == args.method]
    if not tasks:
        parser.error(f"Manifest contains no tasks for {args.method}")

    # Ensure that accidental duplicate destinations cannot overwrite another image.
    destinations = {}
    for task in tasks:
        for directory in task["output_dirs"]:
            identity = task_identity(task)
            if directory in destinations and destinations[directory] != identity:
                raise ValueError(f"Output directory assigned to different tasks: {directory}")
            destinations[directory] = identity

    sys.path.insert(0, str(REPO_ROOT))
    import numpy as np
    from PIL import Image
    import torch
    from datasets import transform
    from main_tokencut import extract_features, source_record
    from networks import get_model
    from object_discovery import ncut
    from visualizations import visualize_mask_overlay

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    renderer_sha256 = file_sha256(Path(__file__))
    source = {**source_record(), "renderer": str(Path(__file__).resolve()),
              "renderer_sha256": renderer_sha256}
    checkpoint_digests = {}
    groups = {}
    for task in tasks:
        checkpoint = task["checkpoint"]
        if checkpoint not in checkpoint_digests:
            checkpoint_digests[checkpoint] = file_sha256(checkpoint)
        groups.setdefault(checkpoint, []).append(task)

    completed = 0
    computed = 0
    started = time.perf_counter()
    print(f"Rendering {args.method}: {len(tasks)} selected images, device={device}", flush=True)
    for checkpoint, checkpoint_tasks in groups.items():
        model = None
        for task in checkpoint_tasks:
            cached = []
            missing = []
            for directory in task["output_dirs"]:
                result = cached_rendering(directory, task, checkpoint_digests[checkpoint],
                                          renderer_sha256)
                (cached if result is not None else missing).append(directory)
            label = f"{task['dataset']}/{task['image']} tau={task['tau']:.2f}"
            if cached:
                for directory in missing:
                    copy_rendering(cached[0], directory)
                completed += 1
                print(f"[{completed}/{len(tasks)}] cached {label}; copied={len(missing)}", flush=True)
                continue

            if model is None:
                model = get_model("vit_base", PATCH_SIZE, device, checkpoint, "model")
                if model.checkpoint_report["checkpoint_sha256"] != checkpoint_digests[checkpoint]:
                    raise ValueError(f"Checkpoint changed while loading: {checkpoint}")
            task_start = time.perf_counter()
            with Image.open(task["image_path"]) as opened:
                image = opened.convert("RGB")
            image_array = np.asarray(image)
            tensor = transform(image)
            if list(tensor.shape) != task["image_shape_chw"]:
                raise ValueError(f"Image dimensions differ from archived prediction for {label}: "
                                 f"{list(tensor.shape)} != {task['image_shape_chw']}")
            if not np.isfinite(task["tau"]) or not np.isfinite(task["bbox_xyxy"]).all():
                raise ValueError(f"Nonfinite tau or archived bbox for {label}")
            height, width = tensor.shape[-2:]
            padded_height = ((height + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
            padded_width = ((width + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
            padded = torch.zeros((3, padded_height, padded_width), dtype=tensor.dtype)
            padded[:, :height, :width] = tensor
            grid = [padded_height // PATCH_SIZE, padded_width // PATCH_SIZE]
            with torch.no_grad():
                features = extract_features(model, padded.to(device)[None], "output")
                expected_shape = (1, grid[0] * grid[1] + 1, model.embed_dim)
                if tuple(features.shape) != expected_shape:
                    raise ValueError(f"Feature/grid mismatch for {label}: {features.shape}")
                predicted, _, foreground, seed, _, eigenvector = ncut(
                    features, grid, [PATCH_SIZE, PATCH_SIZE], tensor.shape,
                    tau=task["tau"], eps=1e-5, im_name=task["image"], no_binary_graph=False,
                )
            bbox_match = bool(np.array_equal(predicted, np.asarray(task["bbox_xyxy"])))
            if not bbox_match:
                raise RuntimeError(f"Recomputed bbox differs for {label}: "
                                   f"expected={task['bbox_xyxy']}, recomputed={predicted.tolist()}")
            overlay = visualize_mask_overlay(image_array, foreground, "", task["image"],
                                             patch_size=PATCH_SIZE, alpha=0.7, save=False)
            full_mask = np.repeat(np.repeat(foreground > 0.5, PATCH_SIZE, axis=0),
                                  PATCH_SIZE, axis=1)[:height, :width].astype(np.uint8) * 255
            runtime = time.perf_counter() - task_start
            metadata = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "manifest": str(manifest_path), "task": task_identity(task),
                "expected_bbox_xyxy": task["bbox_xyxy"],
                "recomputed_bbox_xyxy": predicted.tolist(), "bbox_match": bbox_match,
                "bbox_matches_saved": bbox_match,
                "runtime_seconds": runtime, "device": str(device),
                "patch_grid_hw": grid, "seed_patch": int(seed),
                "checkpoint": model.checkpoint_report, "source": source,
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "protocol": {
                    "arch": "vit_base", "patch_size": PATCH_SIZE, "features": "output",
                    "final_layernorm": True, "cls_in_graph": False, "encoder_frozen": True,
                    "full_resolution": True, "resize": None,
                    "image_mode": "RGB", "tau": task["tau"], "eps": 1e-5,
                    "binary_graph": True,
                    "input_normalization": {"mean": [0.485, 0.456, 0.406],
                                            "std": [0.229, 0.224, 0.225]},
                    "padding": "right/bottom zeros after normalization to patch_size multiple",
                    "mask": "TokenCut connected component containing the seed patch",
                    "mask_upsampling": "nearest repeat by patch_size and crop to original dimensions",
                    "mask_overlay_rgb": [255, 0, 0], "mask_overlay_alpha": 0.7,
                    "annotation_used_for_inference": False, "postprocessing": None,
                },
                "files": list(ASSET_NAMES),
            }
            first = task["output_dirs"][0]
            write_rendering(first, overlay, full_mask, foreground, eigenvector, metadata, np, Image)
            for directory in task["output_dirs"][1:]:
                copy_rendering(first, directory)
            completed += 1
            computed += 1
            print(f"[{completed}/{len(tasks)}] rendered {label}; bbox_match=true "
                  f"seconds={runtime:.3f}; copies={len(task['output_dirs'])}", flush=True)
        if model is not None:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"Completed {args.method}: {completed}/{len(tasks)} images, "
          f"recomputed={computed}, elapsed_seconds={time.perf_counter() - started:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
