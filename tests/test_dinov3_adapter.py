"""CPU regression checks for the DINOv3 TokenCut adapter."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel


TASK_ROOT = Path(__file__).resolve().parents[1]
TOKENCUT_ROOT = TASK_ROOT / "TokenCut"
sys.path.insert(0, str(TOKENCUT_ROOT))

from dinov3_adapter import (
    EXPECTED_CONFIG,
    DINOv3TokenCutAdapter,
    consolidate_dinov3_checkpoint,
    inspect_dinov3_checkpoint,
    load_dinov3_model,
)
from main_tokencut import extract_features
from networks import get_model
from object_discovery import ncut


class DINOv3AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(321)
        cls.model = DINOv3ViTModel(DINOv3ViTConfig(
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_register_tokens=4,
            patch_size=16,
            use_gated_mlp=True,
            query_bias=False,
            key_bias=False,
            value_bias=False,
            hidden_act="silu",
        )).eval()
        cls.adapter = DINOv3TokenCutAdapter(cls.model, {})

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def write_config(root: Path):
        (root / "config.json").write_text(json.dumps({
            "architectures": ["DINOv3ViTModel"], **EXPECTED_CONFIG,
        }))

    def test_output_matches_final_norm_and_removes_register_tokens(self):
        for height, width in ((32, 48), (48, 32)):
            with self.subTest(height=height, width=width):
                image = torch.randn(1, 3, height, width)
                actual = extract_features(self.adapter, image, "output")
                direct = self.model(pixel_values=image, return_dict=True).last_hidden_state
                self.assertEqual(actual.shape, (1, 1 + (height // 16) * (width // 16), 48))
                torch.testing.assert_close(actual[:, 0], direct[:, 0])
                torch.testing.assert_close(actual[:, 1:], direct[:, 5:])
                self.assertFalse(actual.requires_grad)

    def test_adapted_output_runs_ncut_on_rectangular_grids(self):
        for height, width in ((32, 48), (48, 32)):
            with self.subTest(height=height, width=width):
                image = torch.randn(1, 3, height, width)
                features = extract_features(self.adapter, image, "output")
                pred, _, foreground, _, _, _ = ncut(
                    features, [height // 16, width // 16], [16, 16],
                    [3, height, width], 0.2, 1e-5,
                )
                self.assertTrue(0 <= pred[0] <= pred[2] <= width)
                self.assertTrue(0 <= pred[1] <= pred[3] <= height)
                self.assertEqual(foreground.shape, (height // 16, width // 16))
                self.assertTrue(np.isin(foreground, (0.0, 1.0)).all())

    def test_checkpoint_inspection_reads_sharded_safetensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_config(root)
            shard = "model-00001-of-00001.safetensors"
            save_file({"weight": torch.arange(4, dtype=torch.float32)}, str(root / shard))
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_parameters": 4, "total_size": 16},
                "weight_map": {"weight": shard},
            }))
            report = inspect_dinov3_checkpoint(root)
            self.assertEqual(report["total_parameters"], 4)
            self.assertEqual(report["shards"][0]["name"], shard)
            self.assertEqual(report["shards"][0]["tensors"], 1)
            self.assertEqual(report["model_config"]["hidden_size"], 4096)

    def test_checkpoint_inspection_rejects_missing_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_config(root)
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {}, "weight_map": {"weight": "missing.safetensors"},
            }))
            with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
                inspect_dinov3_checkpoint(root)

    def test_checkpoint_inspection_rejects_active_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_config(root)
            shard = "model-00001-of-00001.safetensors"
            save_file({"weight": torch.arange(4, dtype=torch.float32)}, str(root / shard))
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {}, "weight_map": {"weight": shard},
            }))
            pending = root / ".cache" / "huggingface" / "download" / "weights.incomplete"
            pending.parent.mkdir(parents=True)
            pending.write_bytes(b"partial")
            with self.assertRaisesRegex(RuntimeError, "still has incomplete"):
                inspect_dinov3_checkpoint(root)

    def test_checkpoint_inspection_ignores_empty_stale_download_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_config(root)
            shard = "model-00001-of-00001.safetensors"
            save_file({"weight": torch.arange(4, dtype=torch.float32)}, str(root / shard))
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {}, "weight_map": {"weight": shard},
            }))
            pending = root / ".cache" / "huggingface" / "download" / "weights.incomplete"
            pending.parent.mkdir(parents=True)
            pending.touch()
            report = inspect_dinov3_checkpoint(root)
            self.assertEqual(report["ignored_empty_incomplete_files"], [str(pending)])

    def test_streaming_consolidation_preserves_tensor_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_config(root)
            tensors = {
                "first": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                "second": torch.tensor(7, dtype=torch.float32),
            }
            save_file({"first": tensors["first"]}, str(root / "shard1.safetensors"))
            save_file({"second": tensors["second"]}, str(root / "shard2.safetensors"))
            parameter_count = sum(tensor.numel() for tensor in tensors.values())
            size_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_parameters": parameter_count, "total_size": size_bytes},
                "weight_map": {"first": "shard1.safetensors", "second": "shard2.safetensors"},
            }))
            output = root / "complete.safetensors"
            report = consolidate_dinov3_checkpoint(root, output)
            self.assertEqual(report["tensor_sha256_verified"], 2)
            self.assertEqual(report["total_parameters"], parameter_count)
            self.assertEqual(report["total_size_bytes"], size_bytes)
            inspected = inspect_dinov3_checkpoint(output)
            self.assertEqual(inspected["source"], "local_safetensors_file")
            self.assertEqual(inspected["total_parameters"], parameter_count)
            restored = load_file(output)
            self.assertEqual(set(restored), set(tensors))
            for name, tensor in tensors.items():
                self.assertTrue(torch.equal(restored[name], tensor), name)
            self.assertTrue((root / "shard1.safetensors").is_file())
            self.assertTrue((root / "shard2.safetensors").is_file())
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                consolidate_dinov3_checkpoint(root, output)
            self.assertTrue(consolidate_dinov3_checkpoint(output)["already_consolidated"])

    def test_local_sharded_and_single_file_models_load_through_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.model.config.save_pretrained(root)
            self.model.save_pretrained(root, safe_serialization=True, max_shard_size="10KB")
            single = root / "complete.safetensors"
            save_file({name: value.contiguous() for name, value in self.model.state_dict().items()}, single)
            parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
            index = json.loads((root / "model.safetensors.index.json").read_text())
            shard_names = sorted(set(index["weight_map"].values()))
            for source in (root, single):
                with self.subTest(source=source.name):
                    with mock.patch(
                        "dinov3_adapter.inspect_dinov3_checkpoint",
                        return_value={
                            "total_parameters": parameter_count,
                            "shards": [{"name": name} for name in shard_names],
                        },
                    ), mock.patch(
                        "dinov3_adapter.load_dinov3_config",
                        return_value=(root, self.model.config),
                    ):
                        loaded = load_dinov3_model(source, torch.device("cpu"), "test-revision")
                    self.assertEqual(loaded.checkpoint_report["loaded_parameter_count"], parameter_count)
                    self.assertEqual(loaded.checkpoint_report["checkpoint_revision"], "test-revision")
                    self.assertTrue(loaded.checkpoint_report["strict_load"])
                    for name, value in self.model.state_dict().items():
                        self.assertTrue(torch.equal(loaded.model.state_dict()[name], value), name)
                    image = torch.randn(1, 3, 32, 48)
                    actual = extract_features(loaded, image, "output")
                    expected = extract_features(self.adapter, image, "output")
                    torch.testing.assert_close(actual, expected)

    def test_invalid_local_directory_fails_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "does_not_exist"
            with mock.patch.object(DINOv3ViTModel, "from_pretrained") as load:
                with self.assertRaisesRegex(FileNotFoundError, "does_not_exist"):
                    get_model("vit_7b", 16, torch.device("cpu"), str(missing), "model")
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
