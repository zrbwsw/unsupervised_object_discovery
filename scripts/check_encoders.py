#!/usr/bin/env python3
"""Audit the four local ViT-B/16 checkpoints on CPU; this is not CorLoc evaluation.

Run from any directory. The upstream TokenCut checkout is imported without edits.
Only explicitly identified head.* tensors are excluded; the backbone must load
with strict=True. No checkpoint is rewritten and no GPU is used.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch


TASK_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("dino", "ibot", "mae", "condgen")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_upstream_vit():
    """Resolve upstream's unqualified utils import independently of cwd."""
    source_dir = TASK_ROOT / "TokenCut" / "dino"
    modules = {}
    previous_utils = sys.modules.get("utils")
    try:
        for name, filename in (("utils", "utils.py"), ("tokencut_audit_vit", "vision_transformer.py")):
            spec = importlib.util.spec_from_file_location(name, source_dir / filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[name] = module
    finally:
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils
    return modules["tokencut_audit_vit"]


def audit_one(method, checkpoint, vit):
    checkpoint = checkpoint.absolute()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Expected a model container in {checkpoint}")
    state = payload["model"]
    if not isinstance(state, dict) or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError(f"Expected a tensor state dict in {checkpoint}")
    ignored = {key: list(value.shape) for key, value in state.items() if key.startswith("head.")}
    backbone = {key: value for key, value in state.items() if not key.startswith("head.")}
    model = vit.vit_base(patch_size=16, num_classes=0).cpu().eval()
    model.requires_grad_(False)
    incompatibility = model.load_state_dict(backbone, strict=True)
    norm_eps = sorted({module.eps for module in model.modules() if isinstance(module, torch.nn.LayerNorm)})
    assert norm_eps == [1e-6], norm_eps
    forward_checks = []
    for height, width in ((224, 224), (224, 320), (320, 224)):
        # Same deterministic nonconstant tensor for every encoder.
        generator = torch.Generator(device="cpu").manual_seed(0)
        inputs = torch.randn(1, 3, height, width, generator=generator, device="cpu")
        with torch.inference_mode():
            all_tokens = model.get_intermediate_layers(inputs, n=1)[0]
            patch_tokens = all_tokens[:, 1:, :]
        expected = (1, (height // 16) * (width // 16), 768)
        assert tuple(patch_tokens.shape) == expected, (patch_tokens.shape, expected)
        finite = bool(torch.isfinite(patch_tokens).all().item())
        assert finite, f"Non-finite features: {method}, {height}x{width}"
        forward_checks.append({
            "input_shape": list(inputs.shape),
            "all_token_shape": list(all_tokens.shape),
            "patch_token_shape": list(patch_tokens.shape),
            "patch_grid_hw": [height // 16, width // 16],
            "finite": finite,
            "device": str(patch_tokens.device),
            "dtype": str(patch_tokens.dtype),
        })
    selected_keys = (
        "cls_token", "pos_embed", "patch_embed.proj.weight",
        "blocks.0.attn.qkv.weight", "blocks.0.attn.qkv.bias",
        "blocks.0.mlp.fc1.weight", "blocks.11.norm2.weight", "norm.weight",
    )
    return {
        "method": method,
        "checkpoint_path": str(checkpoint),
        "resolved_checkpoint_path": str(checkpoint.resolve(strict=True)),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
        "top_level_keys": list(payload),
        "state_dict_container": "model",
        "state_dict_tensor_count": len(state),
        "excluded_head_tensors": ignored,
        "backbone_tensor_count": len(backbone),
        "strict_load": True,
        "missing_keys": incompatibility.missing_keys,
        "unexpected_keys": incompatibility.unexpected_keys,
        "architecture": {
            "name": "ViT-B/16",
            "patch_size": model.patch_embed.patch_size,
            "embedding_dim": model.embed_dim,
            "depth": len(model.blocks),
            "attention_heads": model.blocks[0].attn.num_heads,
            "mlp_ratio": model.blocks[0].mlp.fc1.out_features / model.embed_dim,
            "qkv_bias": model.blocks[0].attn.qkv.bias is not None,
            "backbone_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "layernorm_eps": norm_eps,
            "selected_tensor_shapes": {key: list(backbone[key].shape) for key in selected_keys},
        },
        "feature_definition": "last block output after pretrained final LayerNorm, CLS removed",
        "position_interpolation": "TokenCut DINO bicubic, scale_factor=(grid_hw + 0.1)/14; no interpolation for 224x224",
        "frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "cpu_forward_checks": forward_checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-dir", type=Path, default=TASK_ROOT.parent / "encoder")
    parser.add_argument("--output", type=Path, default=TASK_ROOT / "reports" / "encoder_check.json")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    vit = import_upstream_vit()
    repo = TASK_ROOT / "TokenCut"
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "CPU encoder compatibility checks only; no dataset inference or CorLoc scores",
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": importlib.metadata.version("numpy"),
            "pillow_version": importlib.metadata.version("pillow"),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "model_implementation": {
            "path": str(Path(vit.__file__).resolve()),
            "sha256": sha256(Path(vit.__file__)),
            "tokencut_commit": commit,
        },
        "encoders": [],
    }
    for method in METHODS:
        result = audit_one(method, args.encoder_dir / f"{method}_vitb16.pth", vit)
        report["encoders"].append(result)
        print(f"{method}: strict load OK, {result['backbone_tensor_count']} backbone tensors, three CPU forwards OK", flush=True)
    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
