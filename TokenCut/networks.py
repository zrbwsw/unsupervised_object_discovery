"""
Loads model. 
Code adapted from LOST: https://github.com/valeoai/LOST
"""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16

import dino.vision_transformer as vits
#import moco.vits as vits_moco

def _load_local_checkpoint(model, checkpoint_path, checkpoint_key):
    """Load an exported encoder without hiding missing or unexpected parameters."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a checkpoint mapping in {checkpoint_path}")
    if checkpoint_key and checkpoint_key in payload:
        state_dict = payload[checkpoint_key]
        container = checkpoint_key
    elif payload and all(isinstance(key, str) and torch.is_tensor(value)
                         for key, value in payload.items()):
        state_dict = payload
        container = "flat"
    else:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no {checkpoint_key!r} tensor state dict "
            "and is not a flat tensor state dict. Set --checkpoint_key explicitly."
        )
    if not isinstance(state_dict, Mapping) or not state_dict or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in state_dict.items()
    ):
        raise ValueError(f"Expected a nonempty tensor state dict in {checkpoint_path}:{container}")
    excluded = {key: list(value.shape) for key, value in state_dict.items()
                if key.startswith("head.")}
    backbone = {key: value for key, value in state_dict.items()
                if not key.startswith("head.")}
    msg = model.load_state_dict(backbone, strict=True)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    model.checkpoint_report = {
        "source": "local",
        "checkpoint_path": str(checkpoint_path),
        "resolved_checkpoint_path": str(checkpoint_path.resolve(strict=True)),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": digest.hexdigest(),
        "checkpoint_key": checkpoint_key,
        "state_dict_container": container,
        "state_dict_tensor_count": len(state_dict),
        "excluded_head_tensors": excluded,
        "backbone_tensor_count": len(backbone),
        "strict_load": True,
        "missing_keys": msg.missing_keys,
        "unexpected_keys": msg.unexpected_keys,
    }
    print(f"Local weights loaded strictly from {checkpoint_path}: {msg}; "
          f"excluded {len(excluded)} head tensors.")


def get_model(arch, patch_size, device, pretrained_weights=None, checkpoint_key="model", checkpoint_revision=None):
    if arch == "vit_7b":
        if patch_size != 16:
            raise ValueError("DINOv3 ViT-7B requires patch_size=16")
        if pretrained_weights is None:
            raise ValueError("DINOv3 ViT-7B requires a local Hugging Face checkpoint directory")
        from dinov3_adapter import load_dinov3_model
        return load_dinov3_model(pretrained_weights, device, checkpoint_revision)

    checkpoint_path = None
    if pretrained_weights is not None:
        checkpoint_path = Path(pretrained_weights).expanduser().absolute()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Local pretrained checkpoint does not exist: {checkpoint_path}")
    if arch not in vits.__dict__ or not callable(vits.__dict__[arch]):
        raise ValueError(f"Unsupported encoder architecture: {arch}")
    if ("moco" in arch or "mae" in arch) and patch_size != 16:
        raise ValueError(f"Architecture {arch} only supports patch_size=16")

    # Initialize model with pretraining
    url = None
    if "moco" in arch:
        if arch == "moco_vit_small" and patch_size == 16:
            url = "moco-v3/vit-s-300ep/vit-s-300ep.pth.tar"
        elif arch == "moco_vit_base" and patch_size == 16:
            url = "moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
        model = vits.__dict__[arch](num_classes=0)
    elif "mae" in arch:
        if arch == "mae_vit_base" and patch_size == 16:
            url = "mae/visualize/mae_visualize_vit_base.pth"
        model = vits.__dict__[arch](num_classes=0)
    elif "vit" in arch:
        if arch == "vit_small" and patch_size == 16:
            url = "dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth" 
        elif arch == "vit_base" and patch_size == 16:
            url = "dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif arch == "resnet50":
            url = "dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    else:
        raise NotImplementedError 

    for p in model.parameters():
        p.requires_grad = False

    if checkpoint_path is not None:
        _load_local_checkpoint(model, checkpoint_path, checkpoint_key)
    elif url is not None:
        print(
            "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
        )
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/" + url
        )
        if "moco" in arch:
            state_dict = state_dict['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        elif "mae" in arch:
            state_dict = state_dict['model']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('decoder') or k.startswith('mask_token'):
                    # remove prefix
                    #state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=True)
        model.checkpoint_report = {
            "source": "official_url",
            "url": "https://dl.fbaipublicfiles.com/" + url,
            "backbone_tensor_count": len(state_dict),
            "strict_load": True,
            "missing_keys": msg.missing_keys,
            "unexpected_keys": msg.unexpected_keys,
        }
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                url, msg
            )
        )
    else:
        raise ValueError(
            f"No reference pretrained weights for {arch} with patch_size={patch_size}; "
            "provide --pretrained_weights."
        )


    model.eval()
    model.to(device)
    return model
