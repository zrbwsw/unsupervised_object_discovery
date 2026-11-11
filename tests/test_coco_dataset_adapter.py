"""COCO adapter checks using temporary images and annotations only.

Run with the evaluation Python environment. The torchvision loader is mocked;
the adapter reads actual fixture files without the full dataset or a GPU.
"""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image


TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "TokenCut"))

from datasets import Dataset


class CocoDatasetAdapterTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        previous_cwd = os.getcwd()
        os.chdir(directory.name)
        self.addCleanup(os.chdir, previous_cwd)

        self.image_root = Path("datasets/COCO/images/train2014")
        self.image_root.mkdir(parents=True)
        self.names = {
            image_id: f"COCO_train2014_{image_id:012d}.jpg"
            for image_id in (101, 202)
        }
        for image_id, color in ((101, (240, 20, 30)), (202, (20, 230, 40))):
            Image.new("RGB", (8, 6), color).save(self.image_root / self.names[image_id])

        Path("datasets/coco_20k_filenames.txt").write_text(
            "".join(f"train2014/{self.names[image_id]}\n" for image_id in (101, 202))
        )
        # Deliberately reverse the image-record order relative to the fixed list.
        self.cache = {
            "images": [
                {"id": image_id, "file_name": self.names[image_id], "width": 8, "height": 6}
                for image_id in (202, 101)
            ],
            "annotations": [
                {"id": 1, "image_id": 101, "category_id": 7,
                 "bbox": [1.2, 0.6, 4.1, 3.7], "iscrowd": 0},
                {"id": 2, "image_id": 101, "category_id": 9,
                 "bbox": [0, 0, 8, 6], "iscrowd": 1},
                {"id": 3, "image_id": 101, "category_id": 4,
                 "bbox": [2, 1, 3, 4], "iscrowd": 0},
                {"id": 4, "image_id": 202, "category_id": 4,
                 "bbox": [2, 1, 3, 4], "iscrowd": 0},
            ],
            "categories": [{"id": i, "name": str(i)} for i in (4, 7, 9)],
        }
        self.write_cache(self.cache)
        loader = mock.patch("datasets.torchvision.datasets.CocoDetection", return_value=[])
        self.coco_loader = loader.start()
        self.addCleanup(loader.stop)

    @staticmethod
    def write_cache(cache):
        Path("datasets/instances_train2014_sel20k.json").write_text(json.dumps(cache))

    def test_image_names_and_loaded_images_ignore_json_record_order(self):
        dataset = Dataset("COCO20k", "train", remove_hards=False)
        for image_id in (101, 202):
            with self.subTest(image_id=image_id):
                targets = [a for a in self.cache["annotations"] if a["image_id"] == image_id]
                name = dataset.get_image_name(targets)
                self.assertEqual(name, self.names[image_id])
                with Image.open(self.image_root / self.names[image_id]) as expected:
                    np.testing.assert_array_equal(dataset.load_image(name), np.asarray(expected))

    def test_cache_with_wrong_missing_or_duplicate_image_ids_is_rejected(self):
        for failure in ("wrong_id", "missing_id", "duplicate_id"):
            with self.subTest(failure=failure):
                cache = copy.deepcopy(self.cache)
                if failure == "wrong_id":
                    cache["images"][0]["id"] = 999
                elif failure == "missing_id":
                    cache["images"].pop()
                else:
                    cache["images"].append(copy.deepcopy(cache["images"][0]))
                self.write_cache(cache)
                with self.assertRaisesRegex(ValueError, "does not match the fixed image list"):
                    Dataset("COCO20k", "train", remove_hards=False)
        self.coco_loader.assert_not_called()

    def test_crowd_filter_and_rounded_xywh_conversion_are_preserved(self):
        dataset = Dataset("COCO20k", "train", remove_hards=False)
        targets = [a for a in self.cache["annotations"] if a["image_id"] == 101]
        original_targets = copy.deepcopy(targets)
        boxes, classes = dataset.extract_gt(targets, self.names[101])
        np.testing.assert_array_equal(boxes, [[1, 1, 5, 4], [2, 1, 5, 5]])
        self.assertEqual(classes, [7, 4])
        self.assertEqual(targets, original_targets)

    def test_fixed_coco20k_list_rejects_nontrain_splits(self):
        for split in ("val", "trainval"):
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, "use --set=train"):
                    Dataset("COCO20k", split, remove_hards=False)
        self.coco_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
