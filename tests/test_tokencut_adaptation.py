"""CPU regression checks for the local TokenCut encoder adaptation.

Run from any directory with the evaluation Python environment. No dataset,
downloaded checkpoint, GPU, or network connection is required. The unchanged
upstream NCut reference is read from the TokenCut checkout's Git HEAD.
"""

from functools import partial
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


TASK_ROOT = Path(__file__).resolve().parents[1]
TOKENCUT_ROOT = TASK_ROOT / "TokenCut"
sys.path.insert(0, str(TOKENCUT_ROOT))

from dino import vision_transformer as vits
from main_tokencut import extract_features
from networks import _load_local_checkpoint, get_model
from object_discovery import ncut


class TokenCutAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(123)
        cls.model = vits.vit_tiny(patch_size=16, num_classes=0).eval()
        cls.model.requires_grad_(False)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def small_checkpoint_model():
        return vits.VisionTransformer(
            img_size=[32], patch_size=16, num_classes=0, embed_dim=24,
            depth=1, num_heads=3, mlp_ratio=2, qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        )

    def test_output_is_complete_last_block_with_final_norm(self):
        for height, width in ((32, 48), (48, 32)):
            with self.subTest(height=height, width=width):
                image = torch.randn(1, 3, height, width)
                actual = extract_features(self.model, image, "output")
                expected = self.model.prepare_tokens(image)
                for block in self.model.blocks:
                    expected = block(expected)
                expected = self.model.norm(expected)
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(actual.shape, (1, 1 + height // 16 * (width // 16), 192))
                self.assertFalse(actual.requires_grad)

    def test_qkv_matches_upstream_and_hooks_do_not_accumulate(self):
        qkv_layer = self.model.blocks[-1].attn.qkv
        original_hooks = len(qkv_layer._forward_hooks)
        for height, width in ((32, 48), (48, 32)):
            image = torch.randn(1, 3, height, width)
            captured = {}
            handle = qkv_layer.register_forward_hook(
                lambda module, inputs, output: captured.update(qkv=output)
            )
            try:
                attention = self.model.get_last_selfattention(image)
            finally:
                handle.remove()
            batch, heads, count, _ = attention.shape
            upstream = captured["qkv"].reshape(batch, count, 3, heads, -1 // heads)
            upstream = upstream.permute(2, 0, 3, 1, 4)
            for index, feature_type in enumerate(("q", "k", "v")):
                with self.subTest(shape=(height, width), feature_type=feature_type):
                    expected = upstream[index].transpose(1, 2).reshape(batch, count, -1)
                    actual = extract_features(self.model, image, feature_type)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(len(qkv_layer._forward_hooks), original_hooks)

    def test_failed_forward_also_removes_qkv_hook(self):
        qkv_layer = self.model.blocks[-1].attn.qkv
        original_hooks = len(qkv_layer._forward_hooks)
        with mock.patch.object(self.model, "get_last_selfattention", side_effect=RuntimeError("forward failed")):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                extract_features(self.model, torch.zeros(1, 3, 32, 48), "k")
        self.assertEqual(len(qkv_layer._forward_hooks), original_hooks)

    def test_ncut_matches_upstream_on_rectangular_grids(self):
        upstream_source = subprocess.check_output(
            ["git", "-C", str(TOKENCUT_ROOT), "show", "HEAD:object_discovery.py"],
            text=True,
        )
        upstream = {}
        exec(compile(upstream_source, "upstream/object_discovery.py", "exec"), upstream)
        generator = torch.Generator().manual_seed(72)
        for dims in ((3, 5), (5, 3)):
            feats = torch.randn(1, 1 + int(np.prod(dims)), 32, generator=generator)
            for no_binary_graph in (False, True):
                with self.subTest(dims=dims, no_binary_graph=no_binary_graph):
                    args = dict(dims=dims, scales=[16, 16],
                                init_image_size=[3, dims[0] * 16 - 5, dims[1] * 16 - 7],
                                tau=0.2, eps=1e-5, no_binary_graph=no_binary_graph)
                    expected = upstream["ncut"](feats.clone(), **args)
                    actual = ncut(feats.clone(), **args)
                    for index in (0, 1, 2, 3, 5):
                        np.testing.assert_array_equal(actual[index], expected[index])
                    self.assertIsNone(actual[4])
                    self.assertLessEqual(actual[0][2], args["init_image_size"][2])
                    self.assertLessEqual(actual[0][3], args["init_image_size"][1])
                    changed_cls = feats.clone()
                    changed_cls[:, 0, :] = 1e6
                    np.testing.assert_array_equal(ncut(changed_cls, **args)[0], actual[0])

    def test_ncut_rejects_missing_cls_and_incorrect_grid(self):
        for shape, dims in (((1, 6, 8), (2, 3)), ((1, 7, 8), (2, 4)), ((2, 7, 8), (2, 3))):
            with self.subTest(shape=shape, dims=dims):
                with self.assertRaisesRegex(ValueError, "expects CLS"):
                    ncut(torch.randn(shape), dims, [16, 16], [3, 32, 48])
        with self.assertRaisesRegex(ValueError, "at least three"):
            ncut(torch.randn(1, 3, 8), [1, 2], [16, 16], [3, 16, 32])

    def test_local_checkpoint_loads_strictly_and_excludes_head(self):
        source = self.small_checkpoint_model()
        target = self.small_checkpoint_model()
        state = dict(source.state_dict())
        state.update({"head.weight": torch.randn(3, 24), "head.bias": torch.randn(3)})
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "encoder.pth"
            torch.save({"model": state}, checkpoint)
            _load_local_checkpoint(target, checkpoint, "model")
        for name, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[name], value), name)
        self.assertTrue(target.checkpoint_report["strict_load"])
        self.assertEqual(target.checkpoint_report["missing_keys"], [])
        self.assertEqual(target.checkpoint_report["unexpected_keys"], [])
        self.assertEqual(set(target.checkpoint_report["excluded_head_tensors"]), {"head.weight", "head.bias"})

    def test_local_checkpoint_rejects_missing_or_extra_backbone_parameters(self):
        model = self.small_checkpoint_model()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid.pth"
            for failure in ("missing", "unexpected"):
                with self.subTest(failure=failure):
                    state = dict(model.state_dict())
                    if failure == "missing":
                        del state["blocks.0.attn.qkv.weight"]
                    else:
                        state["unrecognized_backbone.weight"] = torch.zeros(1)
                    torch.save({"model": state}, checkpoint)
                    with self.assertRaisesRegex(RuntimeError, "Missing key|Unexpected key"):
                        _load_local_checkpoint(model, checkpoint, "model")

    def test_invalid_local_path_fails_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "does_not_exist.pth"
            with mock.patch("torch.hub.load_state_dict_from_url") as download:
                with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                    get_model("vit_base", 16, torch.device("cpu"), str(missing), "model")
                download.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
