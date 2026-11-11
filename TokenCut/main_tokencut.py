"""
Main experiment file. Code adapted from LOST: https://github.com/valeoai/LOST
"""
import os
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import pickle

import torch
import datetime
import numpy as np

from tqdm import tqdm
from dino.utils import bool_flag

from networks import get_model
from datasets import ImageDataset, Dataset, bbox_iou
from visualizations import visualize_eigvec, visualize_mask_overlay, visualize_predictions
from object_discovery import ncut 
import time


@torch.no_grad()
def extract_features(model, image, which_features):
    """Return [CLS + patches]; ncut removes CLS exactly once."""
    if which_features == "output":
        return model.get_intermediate_layers(image, n=1)[0]
    if which_features not in ("q", "k", "v"):
        raise ValueError(f"Unknown feature type: {which_features}")
    captured = {}

    def capture_qkv(module, inputs, output):
        captured["qkv"] = output

    handle = model.blocks[-1].attn.qkv.register_forward_hook(capture_qkv)
    try:
        model.get_last_selfattention(image)
    finally:
        handle.remove()
    qkv = captured["qkv"]
    batch, count, _ = qkv.shape
    return qkv.reshape(batch, count, 3, -1)[:, :, {"q": 0, "k": 1, "v": 2}[which_features], :]


def source_record():
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    names = ("main_tokencut.py", "networks.py", "datasets.py", "object_discovery.py", "dino/vision_transformer.py", "dinov3_adapter.py")
    return {"upstream_commit": commit, "sha256": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
    }}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TokenCut unsupervised object discovery")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "vit_7b",
            "moco_vit_small",
            "moco_vit_base",
            "mae_vit_base",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )

    parser.add_argument("--encoder", choices=("dino", "ibot", "mae", "condgen", "dinov3"), default=None,
                        help="Local encoder label; requires --pretrained_weights.")
    parser.add_argument("--pretrained_weights", default=None,
                        help="Local checkpoint file, or DINOv3 Hugging Face directory; no download fallback.")
    parser.add_argument("--checkpoint_key", default="model", help="State dict container in the local checkpoint.")
    parser.add_argument("--checkpoint_revision", default=None,
                        help="Source revision recorded for a DINOv3 Hugging Face checkpoint.")
    parser.add_argument("--data_path", default=None, help="VOC root containing VOCdevkit, e.g. .../voc2007.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Use only the first N dataset images for a smoke check (default: all).")

    # Use a dataset
    parser.add_argument(
        "--dataset",
        default="VOC07",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k"],
        help="Dataset name.",
    )
    
    parser.add_argument(
        "--save-feat-dir",
        type=str,
        default=None,
        help="if save-feat-dir is not None, only computing features and save it into save-feat-dir",
    )
    
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="If want to apply only on one image, give file path.",
    )

    # Folder used to output visualizations and 
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory to store predictions and visualizations."
    )

    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool_flag, help="Save predicted bounding boxes (true/false).")

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["attn", "pred", "mask", "all", None],
        default=None,
        help="Select visualizations: pred=bbox, mask=red TokenCut mask overlay, attn=eigenvector.",
    )

    # TokenCut parameters
    parser.add_argument(
        "--which_features",
        type=str,
        default="k",
        choices=["k", "q", "v", "output"],
        help="k/q/v from last attention, or output after last block and final LayerNorm",
    )
    parser.add_argument(
        "--k_patches",
        type=int,
        default=100,
        help="Number of patches with the lowest degree considered."
    )
    parser.add_argument("--resize", type=int, default=None, help="Resize input image to fix size")
    parser.add_argument("--tau", type=float, default=0.2, help="Tau for seperating the Graph.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Eps for defining the Graph.")
    parser.add_argument("--no-binary-graph", action="store_true", default=False, help="Generate a binary graph where edge of the Graph will binary. Or using similarity score as edge weight.")

    # Use dino-seg proposed method
    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)

    args = parser.parse_args()
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max_images must be positive")
    if args.encoder and not args.pretrained_weights:
        parser.error("--encoder requires --pretrained_weights; local comparisons never download weights")
    if args.encoder == "dinov3":
        if args.arch != "vit_7b" or args.patch_size != 16:
            parser.error("DINOv3 requires --arch=vit_7b --patch_size=16")
        if args.which_features != "output":
            parser.error("DINOv3 TokenCut evaluation supports only --which_features=output")
    elif args.encoder and (args.arch != "vit_base" or args.patch_size != 16):
        parser.error("The four original local encoders require --arch=vit_base --patch_size=16")
    if args.dinoseg:
        parser.error("This checkout does not include dino_seg(); use the TokenCut path (omit --dinoseg)")
    if args.resize is not None and args.image_path is None:
        parser.error("--resize is only supported for a single --image_path; datasets use full resolution")
    if not np.isfinite(args.tau) or not np.isfinite(args.eps) or args.eps <= 0:
        parser.error("--tau must be finite and --eps must be positive and finite")

    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        dataset = ImageDataset(args.image_path, args.resize)
    else:
        dataset = Dataset(args.dataset, args.set, args.no_hard, data_path=args.data_path)

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    model = get_model(args.arch, args.patch_size, device, args.pretrained_weights, args.checkpoint_key,
                      checkpoint_revision=args.checkpoint_revision)

    # -------------------------------------------------------------------------------------------------------
    # Directories
    if args.image_path is None:
        args.output_dir = os.path.join(args.output_dir, dataset.name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Record the encoder, feature type and graph threshold separately in every run.
    encoder_name = args.encoder or (Path(args.pretrained_weights).stem if args.pretrained_weights else "reference")
    exp_name = f"TokenCut-{encoder_name}-{args.arch}{args.patch_size}_{args.which_features}_tau{args.tau}"
    folder = Path(args.output_dir) / exp_name
    folder.mkdir(parents=True, exist_ok=True)
    dataset_images = len(dataset.dataloader)
    selected_images = min(args.max_images, dataset_images) if args.max_images is not None else dataset_images
    config = {
        "args": vars(args), "command": [sys.executable, *sys.argv],
        "checkpoint": getattr(model, "checkpoint_report", None), "source": source_record(),
        "device": str(device), "torch_version": torch.__version__, "numpy_version": np.__version__,
        "dataset_images": dataset_images, "selected_images": selected_images,
        "data_root": str(Path(dataset.root_path).resolve()) if hasattr(dataset, "root_path") else None,
        "protocol": {
            "features": args.which_features,
            "final_layernorm": args.which_features == "output",
            "cls_in_graph": False, "encoder_frozen": True,
            "register_tokens_in_graph": False,
            "full_resolution": args.resize is None,
            "input_normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "padding": "right/bottom zeros after normalization to patch_size multiple",
            "position_interpolation": ("DINOv3 RoPE in eval mode without random shift, jitter, or rescale"
                                       if args.encoder == "dinov3" else "DINO bicubic +0.1 scale offset"),
            "tau": args.tau, "eps": args.eps, "binary_graph": not args.no_binary_graph,
            "remove_hards": args.no_hard, "iou_threshold": 0.5, "iou_comparison": ">=",
        },
    }
    # A fresh timestamp in the shell template avoids overwriting an earlier run.
    with (folder / "run_config.json").open("x") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    print(f"Running TokenCut on {dataset.name} (exp: {exp_name}, images: {selected_images}/{dataset_images})")
    print(f"Results directory: {folder}")
    vis_folder = str(folder)
    if args.save_feat_dir is not None:
        Path(args.save_feat_dir).mkdir(parents=True, exist_ok=False)

    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    preds_dict = {}
    cnt = 0
    skipped = 0
    prediction_records = []
    corloc = np.zeros(len(dataset.dataloader))
    
    start_time = time.time() 
    pbar = tqdm(itertools.islice(dataset.dataloader, selected_images), total=selected_images)
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]

        init_image_size = img.shape

        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])
        # Pass in case of no gt boxes in the image
        if im_name is None:
            skipped += 1
            continue

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded

        # # Move to gpu
        if device == torch.device('cuda'):
            img = img.cuda(non_blocking=True)
        # Size for transformers
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        # ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    skipped += 1
                    continue

        # ------------ EXTRACT FEATURES -------------------------------------------
        # Both output and q/k/v include CLS; ncut removes it exactly once.
        feats = extract_features(model, img[None], args.which_features)
        expected_tokens = w_featmap * h_featmap + 1
        if feats.shape != (1, expected_tokens, model.embed_dim):
            raise ValueError(f"Feature/grid mismatch for {im_name}: {feats.shape}")
        scales = [args.patch_size, args.patch_size]
        if args.save_feat_dir is not None:
            np.save(Path(args.save_feat_dir) / (Path(im_name).stem + ".npy"), feats.cpu().numpy())
            continue

        # ------------ Apply TokenCut -------------------------------------------
        pred, objects, foreground, seed, bins, eigenvector = ncut(
            feats, [w_featmap, h_featmap], scales, init_image_size,
            args.tau, args.eps, im_name=im_name, no_binary_graph=args.no_binary_graph,
        )
        if args.visualize in ("pred", "mask", "all"):
            image = dataset.load_image(im_name)
        if args.visualize in ("pred", "all"):
            visualize_predictions(image, pred, vis_folder, im_name)
        if args.visualize in ("mask", "all"):
            visualize_mask_overlay(image, foreground, vis_folder, im_name, patch_size=args.patch_size)
        if args.visualize in ("attn", "all"):
            visualize_eigvec(eigenvector, vis_folder, im_name, [w_featmap, h_featmap], scales)

        # ------------ Visualizations -------------------------------------------
        # Save the prediction
        preds_dict[im_name] = pred
        record = {"image": im_name, "bbox_xyxy": pred.tolist(), "image_shape_chw": list(init_image_size),
                  "patch_grid_hw": [w_featmap, h_featmap], "max_iou": None, "correct": None}
        prediction_records.append(record)

        # Evaluation
        if args.no_evaluation:
            continue

        # Compare prediction to GT boxes
        ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))
        
        if torch.any(ious >= 0.5):
            corloc[im_id] = 1
        record["max_iou"] = float(ious.max()) if ious.numel() else 0.0
        record["correct"] = bool(corloc[im_id])
        record["gt_count"] = int(len(gt_bbxs))

        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")

    end_time = time.time()
    print(f'Time cost: {str(datetime.timedelta(milliseconds=int((end_time - start_time)*1000)))}')
    if args.save_feat_dir is not None:
        print(f"Feature export complete: {args.save_feat_dir}; no CorLoc evaluation performed")
        sys.exit(0)

    # Save original TokenCut predictions plus per-image scoring details.
    if args.save_predictions:
        filename = folder / "preds.pkl"
        with filename.open("wb") as stream:
            pickle.dump(preds_dict, stream)
        with (folder / "predictions.jsonl").open("w") as stream:
            for record in prediction_records:
                stream.write(json.dumps(record) + "\n")
        print(f"Predictions saved at {filename}")

    if not args.no_evaluation:
        if cnt == 0:
            raise RuntimeError("No images were evaluated; CorLoc is undefined")
        successful_images = int(np.sum(corloc))
        corloc_percent = 100 * successful_images / cnt
        results = {
            "corloc_percent": corloc_percent, "successful_images": successful_images,
            "evaluated_images": cnt, "skipped_images": skipped,
            "dataset_images": dataset_images, "selected_images": selected_images,
            "subset_evaluation": selected_images < dataset_images,
            "dataset": dataset.name, "encoder": encoder_name,
            "which_features": args.which_features, "tau": args.tau, "eps": args.eps,
            "remove_hards": args.no_hard, "iou_threshold": 0.5, "iou_comparison": ">=",
            "elapsed_seconds": end_time - start_time,
        }
        with (folder / "results.json").open("w") as stream:
            json.dump(results, stream, indent=2)
            stream.write("\n")
        print(f"corloc: {corloc_percent:.2f} ({successful_images}/{cnt})")
        result_file = folder / "results.txt"
        with result_file.open("w") as stream:
            stream.write('corloc,%.1f,,\n' % corloc_percent)
        print(f"File saved at {result_file}")
