#!/usr/bin/env python3
"""Rerun selected CondGen examples at short-side 1024 and export both overlays."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
COLLECTIONS = PROJECT / "object_discovery_compare"
# Default qualitative selection: ranks in VOC07_condgen_best (not image IDs).
# The command-line --ranks option overrides this list.
RANKS = (5, 11, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 31,
         32, 36, 43, 44, 47, 48, 49, 50)
ASSETS = ("original.jpg", "input_1024.png", "prediction_bbox.png",
          "mask_overlay_nearest.png", "mask_overlay_bilinear.png",
          "mask_overlay_bilinear_alpha0.45.png", "mask.png",
          "foreground_patch.npy", "eigenvector.npy", "prediction.json",
          "source_prediction.json", "source_run_config.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, content):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=COLLECTIONS / "VOC07_condgen_best")
    parser.add_argument("--root", type=Path, default=COLLECTIONS / "VOC07_condgen_best_1024")
    parser.add_argument("--ranks", type=int, nargs="+", default=list(RANKS))
    parser.add_argument("--only-ranks", type=int, nargs="+", help="Subset of selection to process")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or len(args.ranks) != len(set(args.ranks)):
        parser.error("Positive threads and unique ranks are required")
    if args.only_ranks and not set(args.only_ranks) <= set(args.ranks):
        parser.error("--only-ranks must be a subset of --ranks")

    source = args.source.resolve(strict=True)
    root = args.root.resolve()
    if root == source or source in root.parents:
        parser.error("Output must be separate from the original collection")
    samples = []
    checkpoints = set()
    protected = {}
    for rank in args.ranks:
        matches = [p for p in source.glob(f"{rank:03d}_*") if p.is_dir()]
        if len(matches) != 1:
            raise ValueError(f"Rank {rank} has {len(matches)} matches")
        directory = matches[0]
        metadata = json.loads((directory / "condgen/rendering.json").read_text())
        task = metadata["task"]
        if task["method"] != "condgen" or task["tau"] != 0.45:
            raise ValueError(f"Unexpected method or tau at rank {rank}")
        checkpoints.add(str(Path(task["checkpoint"]).resolve(strict=True)))
        for filename in ("original.jpg", "metrics.json", "condgen/prediction.json",
                         "condgen/rendering.json", "condgen/source_run_config.json",
                         "condgen/foreground_patch.npy", "condgen/mask.png",
                         "condgen/mask_overlay.png"):
            p = directory / filename
            protected[str(p)] = sha256(p)
        samples.append({"rank": rank, "image": task["image"],
                        "directory": f"{rank:03d}_{Path(task['image']).stem}",
                        "source_directory": str(directory),
                        "source_max_iou": json.loads((directory / "metrics.json").read_text())["metrics"]["condgen"],
                        "original_sha256": protected[str(directory / "original.jpg")]})
    if len(checkpoints) != 1:
        raise ValueError("All selected samples must use the same checkpoint")
    checkpoint = checkpoints.pop()
    protocol = {
        "method": "condgen", "dataset": "VOC07", "arch": "vit_base", "patch_size": 16,
        "tau": 0.45, "eps": 1e-5, "binary_graph": True, "features": "output",
        "final_layernorm": True, "encoder_frozen": True, "cls_in_graph": False,
        "resize": 1024, "resize_convention": "short side, preserve aspect ratio, no crop",
        "image_resize": "torchvision tensor bilinear with antialias=True (before normalization)",
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "padding": "right/bottom zeros after normalization to patch_size multiple",
        "full_resolution": False, "qualitative_only": True,
        "annotation_used_for_inference": False, "mask_postprocessing": None,
        "mask": "TokenCut connected component containing the seed patch",
        "rendering": {"rgb": [255, 0, 0], "nearest_alpha": 0.7,
                      "bilinear_alpha": 0.7, "additional_bilinear_alpha": 0.45,
                      "alignment": "upsample to padded dimensions then crop to input dimensions",
                      "bilinear_display_only": True, "rethreshold_after_interpolation": False},
    }
    selection = {"source": str(source), "samples": samples, "protocol": protocol,
                 "checkpoint": checkpoint, "checkpoint_sha256": sha256(checkpoint),
                 "source_hashes": protected}
    if (root / "selection.json").is_file():
        if json.loads((root / "selection.json").read_text()) != selection:
            raise ValueError("Existing collection has different inputs or protocol")
    else:
        root.mkdir(parents=True, exist_ok=True)
        if any(p.name != "inference.log" for p in root.iterdir()):
            raise ValueError(f"Refusing to reuse an unrelated output directory: {root}")
        write_json(root / "selection.json", selection)
    if args.prepare_only:
        print(f"Prepared {len(samples)} selected samples: {root}", flush=True)
        return

    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(PROJECT / "TokenCut"))
    import numpy as np
    from PIL import Image
    import torch
    import torchvision
    from torchvision.transforms import functional as TF, InterpolationMode
    from main_tokencut import extract_features, source_record
    from networks import get_model
    from object_discovery import ncut
    from visualizations import visualize_mask_overlay, visualize_predictions

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    sources = {**source_record(), "renderer": str(Path(__file__).resolve()),
               "renderer_sha256": sha256(__file__),
               "visualizations_sha256": sha256(PROJECT / "TokenCut/visualizations.py")}
    model = None
    selected = [s for s in samples if not args.only_ranks or s["rank"] in args.only_ranks]
    started = time.perf_counter()
    for number, sample in enumerate(selected, 1):
        output = root / sample["directory"]
        marker = output / "rendering.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            if (saved["source"] == sources and saved["protocol"] == protocol
                    and saved["checkpoint"]["checkpoint_sha256"] == selection["checkpoint_sha256"]
                    and all((output / f).is_file() and sha256(output / f) == saved["asset_sha256"][f]
                            for f in ASSETS)):
                print(f"[{number}/{len(selected)}] cached rank={sample['rank']}", flush=True)
                continue
            raise ValueError(f"Incompatible or damaged existing rendering: {output}")
        if model is None:
            model = get_model("vit_base", 16, device, checkpoint, "model")
            if model.checkpoint_report["checkpoint_sha256"] != selection["checkpoint_sha256"]:
                raise ValueError("Loaded checkpoint changed")
        output.mkdir(parents=True, exist_ok=True)
        sample_start = time.perf_counter()
        original_path = Path(sample["source_directory"]) / "original.jpg"
        with Image.open(original_path) as original:
            original = original.convert("RGB")
            original_size = original.size
            rgb_tensor = TF.resize(TF.to_tensor(original), 1024,
                                   interpolation=InterpolationMode.BILINEAR, antialias=True)
        height, width = rgb_tensor.shape[-2:]
        image = (rgb_tensor.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        tensor = TF.normalize(rgb_tensor, protocol["normalization"]["mean"],
                              protocol["normalization"]["std"])
        padded_height, padded_width = ((height + 15) // 16) * 16, ((width + 15) // 16) * 16
        padded = torch.zeros((3, padded_height, padded_width), dtype=tensor.dtype)
        padded[:, :height, :width] = tensor
        grid = [padded_height // 16, padded_width // 16]
        print(f"[{number}/{len(selected)}] start rank={sample['rank']} {sample['image']} "
              f"input={width}x{height} grid={grid} device={device}", flush=True)
        feature_start = time.perf_counter()
        with torch.inference_mode():
            features = extract_features(model, padded.to(device)[None], "output")
            if tuple(features.shape) != (1, grid[0] * grid[1] + 1, model.embed_dim):
                raise ValueError("Feature/grid mismatch")
            if device.type == "cuda":
                torch.cuda.synchronize()
            feature_seconds = time.perf_counter() - feature_start
            ncut_start = time.perf_counter()
            predicted, _, foreground, seed, _, eigenvector = ncut(
                features, grid, [16, 16], tensor.shape, tau=0.45, eps=1e-5,
                im_name=sample["image"], no_binary_graph=False)
        ncut_seconds = time.perf_counter() - ncut_start
        del features, padded, tensor, rgb_tensor
        mask = np.repeat(np.repeat(foreground > 0.5, 16, axis=0), 16, axis=1)[:height, :width]
        ys, xs = np.where(mask)
        if not np.array_equal(predicted, [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]):
            raise ValueError("Bounding box does not enclose the saved raw mask exactly")
        shutil.copy2(original_path, output / "original.jpg")
        for src, dst in (("prediction.json", "source_prediction.json"),
                         ("source_run_config.json", "source_run_config.json")):
            shutil.copy2(Path(sample["source_directory"]) / "condgen" / src, output / dst)
        Image.fromarray(image).save(output / "input_1024.png")
        # ``image`` is produced by a tensor transpose and may be non-contiguous;
        # OpenCV's rectangle helper requires a contiguous RGB buffer.
        bbox_image = visualize_predictions(np.ascontiguousarray(image), predicted, "", "", save=False)
        Image.fromarray(np.ascontiguousarray(bbox_image)).save(output / "prediction_bbox.png")
        for mode, alpha, filename in (("nearest", 0.7, "mask_overlay_nearest.png"),
                                      ("bilinear", 0.7, "mask_overlay_bilinear.png"),
                                      ("bilinear", 0.45, "mask_overlay_bilinear_alpha0.45.png")):
            overlay = visualize_mask_overlay(image, foreground, "", "", patch_size=16,
                                              alpha=alpha, save=False, interpolation=mode)
            Image.fromarray(overlay).save(output / filename)
        Image.fromarray(mask.astype(np.uint8) * 255).save(output / "mask.png")
        np.save(output / "foreground_patch.npy", foreground)
        np.save(output / "eigenvector.npy", eigenvector)
        scale_back = np.array([original_size[0] / width, original_size[1] / height] * 2)
        write_json(output / "prediction.json", {
            "rank": sample["rank"], "image": sample["image"], "tau": 0.45,
            "bbox_xyxy_input": predicted.tolist(),
            "bbox_xyxy_original": (predicted * scale_back).tolist(),
            "image_size_wh": [width, height], "original_size_wh": list(original_size),
            "qualitative_only": True, "max_iou": None,
            "note": "New 1024 prediction; source_prediction.json is the original full-resolution evaluation.",
        })
        runtime = time.perf_counter() - sample_start
        write_json(marker, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(), **sample,
            "image_size_wh": [width, height], "original_size_wh": list(original_size),
            "padded_size_wh": [padded_width, padded_height], "patch_grid_hw": grid,
            "rank": sample["rank"], "tau": 0.45, "seed_patch": int(seed),
            "protocol": protocol, "checkpoint": model.checkpoint_report,
            "device": str(device), "threads": args.threads, "runtime_seconds": runtime,
            "feature_seconds": feature_seconds, "ncut_seconds": ncut_seconds,
            "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
            "numpy_version": np.__version__, "source": sources,
            "asset_sha256": {filename: sha256(output / filename) for filename in ASSETS},
        })
        print(f"[{number}/{len(selected)}] done rank={sample['rank']} "
              f"seconds={runtime:.2f} features={feature_seconds:.2f} ncut={ncut_seconds:.2f}", flush=True)
    if any(sha256(path) != value for path, value in protected.items()):
        raise RuntimeError("A source artifact changed during this run")
    print(f"Completed {len(selected)} samples; source hashes unchanged; "
          f"elapsed={time.perf_counter() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
