"""Verify real VOC12/COCO20K loaders, symlinks, IDs and sample annotations on CPU."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT / 'TokenCut'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=TASK_ROOT / 'reports/voc12_coco_loader_check.json')
    args = parser.parse_args()
    report_path = args.report.resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO_ROOT))
    from datasets import Dataset
    import torch
    torch.set_num_threads(1)
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.executable, 'pycocotools': importlib.metadata.version('pycocotools'),
        'datasets': [],
    }
    previous_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        for name, split, expected in (('VOC12', 'trainval', 11540), ('COCO20k', 'train', 19817)):
            dataset = Dataset(name, split, False)
            if len(dataset.dataloader) != expected:
                raise ValueError(f'{name}: expected {expected}, got {len(dataset.dataloader)}')
            result = {'name': dataset.name, 'image_count': expected, 'remove_hards': False,
                      'resolved_image_root': str(Path(dataset.root_path).resolve()), 'samples': []}
            if name == 'COCO20k':
                selected_ids = {int(value) for value in dataset.sel_20k}
                if set(dataset.dataloader.ids) != selected_ids:
                    raise ValueError('Actual COCO loader IDs differ from the fixed list')
                noncrowd_counts = [sum(ann.get('iscrowd', 0) == 0 for ann in
                                      dataset.dataloader.coco.imgToAnns[image_id])
                                   for image_id in dataset.dataloader.ids]
                if not all(noncrowd_counts):
                    raise ValueError('COCO subset contains an image without non-crowd ground truth')
                result.update(
                    actual_ids_match_fixed_list=True,
                    total_annotations=len(dataset.dataloader.coco.anns),
                    noncrowd_annotations=sum(noncrowd_counts),
                    all_images_have_noncrowd_gt=True,
                    list_sha256=hashlib.sha256(Path(dataset.sel20k).read_bytes()).hexdigest(),
                    annotation_cache=str(Path(dataset.annfile).resolve()),
                )
            for index in (0, expected // 2, expected - 1):
                tensor, targets = dataset.dataloader[index]
                filename = dataset.get_image_name(targets)
                boxes, classes = dataset.extract_gt(targets, filename)
                image = dataset.load_image(filename)
                if tuple(image.shape[:2]) != tuple(tensor.shape[1:]):
                    raise ValueError(f'{filename}: tensor/visualization dimensions disagree')
                if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) == 0:
                    raise ValueError(f'{filename}: invalid ground-truth boxes')
                if not torch.isfinite(tensor).all():
                    raise ValueError(f'{filename}: nonfinite input')
                if name == 'COCO20k':
                    image_id = dataset.dataloader.ids[index]
                    original = dataset.dataloader.coco.imgs[image_id]
                    if original['file_name'] != filename:
                        raise ValueError('Filename and image_id disagree')
                result['samples'].append({'index': index, 'filename': filename,
                    'image_shape_chw': list(tensor.shape), 'gt_count': len(boxes),
                    'boxes_xyxy': boxes.tolist(), 'classes': classes})
            report['datasets'].append(result)
            print(f'{name}: {expected} images; first/middle/last samples and GT verified', flush=True)
    finally:
        os.chdir(previous_cwd)
    report['success'] = True
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Report: {report_path}')


if __name__ == '__main__':
    main()
