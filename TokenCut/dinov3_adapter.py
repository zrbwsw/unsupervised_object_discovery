"""DINOv3 Hugging Face checkpoint loading and TokenCut feature adaptation."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors import safe_open
from transformers import DINOv3ViTConfig, DINOv3ViTModel, __version__ as transformers_version


EXPECTED_CONFIG = {
    "model_type": "dinov3_vit",
    "hidden_size": 4096,
    "intermediate_size": 8192,
    "num_hidden_layers": 40,
    "num_attention_heads": 32,
    "patch_size": 16,
    "num_register_tokens": 4,
    "hidden_act": "silu",
    "layer_norm_eps": 1e-5,
    "rope_theta": 100.0,
    "use_gated_mlp": True,
    "query_bias": False,
    "key_bias": False,
    "value_bias": False,
    "proj_bias": True,
    "mlp_bias": True,
}
DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "F32": 4,
    "I64": 8,
    "F64": 8,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_safetensors_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Truncated safetensors header length: {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"Truncated safetensors JSON header: {path}")
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid safetensors JSON header: {path}") from error
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header object: {path}")

    data_start = 8 + header_length
    data_end = 0
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    if not tensors:
        raise ValueError(f"No tensors in safetensors file: {path}")
    for name, info in tensors.items():
        try:
            dtype = info["dtype"]
            shape = info["shape"]
            begin, end = info["data_offsets"]
            item_size = DTYPE_SIZES[dtype]
            expected_bytes = math.prod(shape) * item_size
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid tensor metadata for {name!r} in {path}") from error
        if not isinstance(begin, int) or not isinstance(end, int) or not 0 <= begin <= end:
            raise ValueError(f"Invalid data offsets for {name!r} in {path}: {begin}, {end}")
        if end - begin != expected_bytes:
            raise ValueError(
                f"Tensor byte size mismatch for {name!r} in {path}: "
                f"offsets={end - begin}, shape/dtype={expected_bytes}"
            )
        data_end = max(data_end, end)
    actual_size = path.stat().st_size
    if actual_size != data_start + data_end:
        raise ValueError(
            f"Safetensors file size mismatch: {path}; "
            f"expected {data_start + data_end}, found {actual_size}"
        )
    return data_start, header


def _tensor_counts(header: dict) -> tuple[int, int]:
    parameters = 0
    size_bytes = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        parameters += math.prod(info["shape"])
        begin, end = info["data_offsets"]
        size_bytes += end - begin
    return parameters, size_bytes


def load_dinov3_config(checkpoint_path: str | Path) -> tuple[Path, DINOv3ViTConfig]:
    source = Path(checkpoint_path).expanduser().resolve()
    if source.is_file() and source.suffix == ".safetensors":
        root = source.parent
    elif source.is_dir():
        root = source
    else:
        raise FileNotFoundError(
            f"DINOv3 checkpoint must be a directory or .safetensors file: {source}"
        )
    config = DINOv3ViTConfig.from_pretrained(root, local_files_only=True)
    mismatches = {
        key: {"expected": expected, "actual": getattr(config, key, None)}
        for key, expected in EXPECTED_CONFIG.items()
        if getattr(config, key, None) != expected
    }
    if mismatches:
        raise ValueError(f"Checkpoint is not the expected DINOv3 ViT-7B/16: {mismatches}")
    return root, config


def inspect_dinov3_checkpoint(checkpoint_path: str | Path) -> dict:
    source = Path(checkpoint_path).expanduser().resolve()
    root, config = load_dinov3_config(source)
    model_config = {key: getattr(config, key) for key in EXPECTED_CONFIG}
    if source.is_file():
        _, header = _read_safetensors_header(source)
        parameters, size_bytes = _tensor_counts(header)
        return {
            "source": "local_safetensors_file",
            "checkpoint_path": str(source),
            "resolved_checkpoint_path": str(source),
            "config_directory": str(root),
            "checkpoint_index": None,
            "checkpoint_index_sha256": None,
            "shards": [{
                "name": source.name,
                "size_bytes": source.stat().st_size,
                "tensors": len(header) - ("__metadata__" in header),
            }],
            "total_parameters": parameters,
            "total_size_bytes": size_bytes,
            "model_config": model_config,
        }

    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing sharded checkpoint index: {index_path}")
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid or empty weight_map in {index_path}")
    if any(Path(name).name != name for name in weight_map.values()):
        raise ValueError(f"Shard entries must be plain filenames in {index_path}")

    shard_names = sorted(set(weight_map.values()))
    missing = [name for name in shard_names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "DINOv3 checkpoint is incomplete; wait for the download to finish. "
            f"Missing {len(missing)}/{len(shard_names)} shards: {missing}"
        )
    pending_candidates = sorted((root / ".cache" / "huggingface" / "download").rglob("*.incomplete"))
    pending = [path for path in pending_candidates if path.stat().st_size > 0]
    empty_pending = [path for path in pending_candidates if path.stat().st_size == 0]
    if pending:
        raise RuntimeError(
            "DINOv3 download still has incomplete files: "
            + ", ".join(str(path) for path in pending)
        )

    shards = []
    parameters = 0
    size_bytes = 0
    for name in shard_names:
        path = root / name
        expected_keys = {key for key, shard in weight_map.items() if shard == name}
        _, header = _read_safetensors_header(path)
        actual_keys = set(header) - {"__metadata__"}
        if actual_keys != expected_keys:
            raise ValueError(
                f"Tensor keys in {path} do not match {index_path}: "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}"
            )
        shard_parameters, shard_bytes = _tensor_counts(header)
        parameters += shard_parameters
        size_bytes += shard_bytes
        shards.append({"name": name, "size_bytes": path.stat().st_size, "tensors": len(actual_keys)})

    metadata = index.get("metadata", {})
    for key, actual in (("total_parameters", parameters), ("total_size", size_bytes)):
        expected = metadata.get(key)
        if expected is not None and expected != actual:
            raise ValueError(f"Checkpoint index {key} mismatch: expected {expected}, found {actual}")

    return {
        "source": "local_huggingface_safetensors",
        "checkpoint_path": str(root),
        "resolved_checkpoint_path": str(root),
        "config_directory": str(root),
        "checkpoint_index": str(index_path),
        "checkpoint_index_sha256": _sha256(index_path),
        "ignored_empty_incomplete_files": [str(path) for path in empty_pending],
        "shards": shards,
        "total_parameters": parameters,
        "total_size_bytes": size_bytes,
        "model_config": model_config,
    }


def _hash_output_tensors(path: Path, names: list[str]) -> dict[str, str]:
    data_start, header = _read_safetensors_header(path)
    hashes = {}
    with path.open("rb") as stream:
        for name in names:
            begin, end = header[name]["data_offsets"]
            stream.seek(data_start + begin)
            remaining = end - begin
            digest = hashlib.sha256()
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"Truncated tensor data for {name!r} in {path}")
                digest.update(block)
                remaining -= len(block)
            hashes[name] = digest.hexdigest()
    return hashes


def consolidate_dinov3_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict:
    source = Path(checkpoint_path).expanduser().resolve()
    report = inspect_dinov3_checkpoint(source)
    if source.is_file():
        return {**report, "output_path": str(source), "already_consolidated": True}

    root = source
    output = Path(output_path).expanduser().resolve() if output_path else root / "dinov3_vit7b16.safetensors"
    if output.suffix != ".safetensors":
        raise ValueError(f"Consolidated output must use .safetensors: {output}")
    if output in {root / shard["name"] for shard in report["shards"]}:
        raise ValueError("Consolidated output cannot replace a source shard")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite consolidated checkpoint: {output}")
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary checkpoint: {temporary}")
    output.parent.mkdir(parents=True, exist_ok=True)

    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    source_headers = {
        name: _read_safetensors_header(root / name)
        for name in sorted(set(weight_map.values()))
    }
    metadata = {}
    for _, header in source_headers.values():
        metadata.update(header.get("__metadata__", {}))
    metadata["format"] = "pt"
    metadata["source_index_sha256"] = report["checkpoint_index_sha256"]

    output_header = {}
    sources = {}
    destination_offset = 0
    for name, shard in weight_map.items():
        _, header = source_headers[shard]
        info = header[name]
        begin, end = info["data_offsets"]
        size = end - begin
        output_header[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [destination_offset, destination_offset + size],
        }
        sources[name] = (root / shard, source_headers[shard][0], begin, end)
        destination_offset += size
    output_header["__metadata__"] = metadata
    header_bytes = json.dumps(output_header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    required_bytes = 8 + len(header_bytes) + destination_offset
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient disk space for consolidated checkpoint: need {required_bytes} bytes, "
            f"available {free_bytes} bytes at {output.parent}"
        )

    source_hashes = {}
    try:
        with temporary.open("xb") as destination:
            destination.write(struct.pack("<Q", len(header_bytes)))
            destination.write(header_bytes)
            for name, (path, data_start, begin, end) in sources.items():
                digest = hashlib.sha256()
                with path.open("rb") as source_stream:
                    source_stream.seek(data_start + begin)
                    remaining = end - begin
                    while remaining:
                        block = source_stream.read(min(8 * 1024 * 1024, remaining))
                        if not block:
                            raise ValueError(f"Truncated tensor data for {name!r} in {path}")
                        destination.write(block)
                        digest.update(block)
                        remaining -= len(block)
                source_hashes[name] = digest.hexdigest()
            destination.flush()
        output_hashes = _hash_output_tensors(temporary, list(weight_map))
        if output_hashes != source_hashes:
            mismatches = [name for name in weight_map if output_hashes[name] != source_hashes[name]]
            raise RuntimeError(f"Consolidated tensor hash mismatch: {mismatches}")
        parameters, size_bytes = _tensor_counts(_read_safetensors_header(temporary)[1])
        if (parameters, size_bytes) != (report["total_parameters"], report["total_size_bytes"]):
            raise RuntimeError("Consolidated parameter or byte count does not match the source shards")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "source": "consolidated_safetensors",
        "output_path": str(output),
        "output_size_bytes": output.stat().st_size,
        "source_index_sha256": report["checkpoint_index_sha256"],
        "source_shards": len(report["shards"]),
        "tensors": len(weight_map),
        "total_parameters": report["total_parameters"],
        "total_size_bytes": report["total_size_bytes"],
        "tensor_sha256_verified": len(weight_map),
        "already_consolidated": False,
    }


class DINOv3TokenCutAdapter(torch.nn.Module):
    def __init__(self, model: DINOv3ViTModel, checkpoint_report: dict):
        super().__init__()
        self.model = model
        self.embed_dim = model.config.hidden_size
        self.patch_size = model.config.patch_size
        self.num_register_tokens = model.config.num_register_tokens
        self.checkpoint_report = checkpoint_report
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def get_intermediate_layers(self, image: torch.Tensor, n: int = 1):
        if n != 1:
            raise ValueError("TokenCut DINOv3 adaptation exposes only the final layer (n=1)")
        output = self.model(pixel_values=image, return_dict=True).last_hidden_state
        patches = (image.shape[-2] // self.patch_size) * (image.shape[-1] // self.patch_size)
        expected = 1 + self.num_register_tokens + patches
        if output.shape != (image.shape[0], expected, self.embed_dim):
            raise ValueError(
                f"Unexpected DINOv3 output shape {tuple(output.shape)}; "
                f"expected {(image.shape[0], expected, self.embed_dim)}"
            )
        patch_output = output[:, 1 + self.num_register_tokens:, :]
        return [torch.cat((output[:, :1, :], patch_output), dim=1)]


def load_dinov3_model(
    checkpoint_path: str | Path,
    device: torch.device,
    checkpoint_revision: str | None = None,
) -> DINOv3TokenCutAdapter:
    source = Path(checkpoint_path).expanduser().resolve()
    report = inspect_dinov3_checkpoint(source)
    config_root, config = load_dinov3_config(source)
    source_files = [source] if source.is_file() else [
        config_root / shard["name"] for shard in report["shards"]
    ]
    checkpoint_shapes = {}
    for path in source_files:
        _, header = _read_safetensors_header(path)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in checkpoint_shapes:
                raise RuntimeError(f"Duplicate DINOv3 tensor across checkpoint files: {name}")
            if info["dtype"] != "F32":
                raise ValueError(f"DINOv3 evaluation requires FP32 weights; {name} is {info['dtype']}")
            checkpoint_shapes[name] = tuple(info["shape"])

    config._attn_implementation = "sdpa"
    with init_empty_weights(include_buffers=False):
        model = DINOv3ViTModel(config)
    model_shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    missing = sorted(set(model_shapes) - set(checkpoint_shapes))
    unexpected = sorted(set(checkpoint_shapes) - set(model_shapes))
    if missing or unexpected:
        raise RuntimeError(f"DINOv3 checkpoint keys do not match the model: missing={missing}, unexpected={unexpected}")
    mismatched = {
        name: {"checkpoint": checkpoint_shapes[name], "model": model_shapes[name]}
        for name in set(model_shapes) & set(checkpoint_shapes)
        if checkpoint_shapes[name] != model_shapes[name]
    }
    if mismatched:
        raise RuntimeError(f"DINOv3 checkpoint tensor shapes do not match the model: {mismatched}")

    for path in source_files:
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            names = list(stream.keys())
            print(f"Loading {len(names)} DINOv3 tensors from {path}", flush=True)
            for name in names:
                value = stream.get_tensor(name)
                set_module_tensor_to_device(model, name, "cpu", value=value, dtype=torch.float32)
                del value
    meta_parameters = [name for name, value in model.named_parameters() if value.is_meta]
    if meta_parameters:
        raise RuntimeError(f"DINOv3 parameters remain uninitialized after loading: {meta_parameters}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if report["total_parameters"] != parameter_count:
        raise RuntimeError(
            f"DINOv3 parameter count mismatch: model={parameter_count}, checkpoint={report['total_parameters']}"
        )

    report.update({
        "checkpoint_revision": checkpoint_revision,
        "transformers_version": transformers_version,
        "attention_implementation": "sdpa",
        "loading_backend": "accelerate_meta_initialization_and_safetensors_per_tensor_loading",
        "dtype": "float32",
        "strict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "loaded_parameter_count": parameter_count,
        "verified_tensor_count": len(checkpoint_shapes),
        "register_tokens_removed_before_ncut": model.config.num_register_tokens,
    })
    adapter = DINOv3TokenCutAdapter(model, report)
    adapter.to(device)
    return adapter
