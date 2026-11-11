"""Extract local VOC2007 archives and verify the original detection layout."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path('/data0/datasets/dino_downstream/voc2007')
ARCHIVES = ('VOCtrainval_06-Nov-2007.tar', 'VOCtest_06-Nov-2007.tar')
EXPECTED = {'train': 2501, 'val': 2510, 'trainval': 5011, 'test': 4952}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def extract_archive(path, destination):
    """Re-running verifies existing files instead of overwriting them."""
    created = existing = 0
    with tarfile.open(path) as archive:
        for member in archive:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f'Archive member outside destination: {member.name}')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f'Unsupported archive member: {member.name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source:
                if target.exists():
                    with target.open('rb') as current:
                        while True:
                            block = source.read(1024 * 1024)
                            if block != current.read(1024 * 1024):
                                raise ValueError(f'Existing file differs from archive: {target}')
                            if not block:
                                break
                    existing += 1
                else:
                    # A temporary file avoids leaving a partial target if interrupted.
                    temporary = target.with_name(target.name + '.extracting')
                    with temporary.open('wb') as output:
                        shutil.copyfileobj(source, output)
                    temporary.replace(target)
                    created += 1
    return {'created_files': created, 'identical_existing_files': existing}


def verify_dataset(destination):
    voc = destination / 'VOCdevkit/VOC2007'
    splits = {}
    split_report = {}
    for split, count in EXPECTED.items():
        path = voc / 'ImageSets/Main' / f'{split}.txt'
        ids = path.read_text().split()
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError(f'{split}: expected {count} unique IDs, got {len(ids)}')
        splits[split] = set(ids)
        split_report[split] = {'images': count, 'list_sha256': sha256(path)}
    if splits['train'] & splits['val'] or splits['train'] | splits['val'] != splits['trainval']:
        raise ValueError('train/val are not a disjoint partition of trainval')
    if splits['trainval'] & splits['test']:
        raise ValueError('trainval/test overlap')
    all_ids = splits['trainval'] | splits['test']
    for folder, suffix in (('JPEGImages', '.jpg'), ('Annotations', '.xml')):
        file_ids = {p.stem for p in (voc / folder).glob('*' + suffix)}
        if file_ids != all_ids:
            raise ValueError(f'{folder}: missing={len(all_ids-file_ids)}, extra={len(file_ids-all_ids)}')
    object_counts = {'trainval': 0, 'test': 0}
    hard_only_counts = {'trainval': 0, 'test': 0}
    zero_coordinate_boxes = 0
    for index, image_id in enumerate(sorted(all_ids), 1):
        image_path = voc / 'JPEGImages' / f'{image_id}.jpg'
        annotation = ET.parse(voc / 'Annotations' / f'{image_id}.xml').getroot()
        if annotation.findtext('filename') != image_path.name:
            raise ValueError(f'XML/image filename mismatch: {image_id}')
        with Image.open(image_path) as image:
            image.load()  # Decode every image, rather than checking file presence alone.
            width, height = image.size
        xml_size = (int(annotation.findtext('size/width')), int(annotation.findtext('size/height')))
        if (width, height) != xml_size:
            raise ValueError(f'XML/image size mismatch: {image_id}')
        objects = annotation.findall('object')
        if not objects:
            raise ValueError(f'No ground-truth objects: {image_id}')
        split = 'trainval' if image_id in splits['trainval'] else 'test'
        object_counts[split] += len(objects)
        hard_only_counts[split] += all(
            obj.findtext('difficult') == '1' or obj.findtext('truncated') == '1'
            for obj in objects
        )
        for obj in objects:
            box = [int(obj.findtext('bndbox/' + field)) for field in ('xmin', 'ymin', 'xmax', 'ymax')]
            x1, y1, x2, y2 = box
            if not (0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height):
                raise ValueError(f'Invalid bounding box: {image_id}: {box}')
            zero_coordinate_boxes += min(box) == 0
        if index % 1000 == 0:
            print(f'Verified {index}/{len(all_ids)} JPEG/XML pairs', flush=True)
    return {
        'voc_directory': str(voc), 'splits': split_report,
        'jpeg_count': len(all_ids), 'xml_count': len(all_ids),
        'all_images_decoded': True, 'all_xml_parsed': True,
        'xml_image_dimensions_match': True, 'bounding_boxes_in_image_bounds': True,
        'object_counts': object_counts, 'images_with_only_difficult_or_truncated_objects': hard_only_counts,
        'boxes_with_zero_coordinate': zero_coordinate_boxes,
        'evaluation_split': 'trainval', 'evaluation_images': EXPECTED['trainval'],
        'annotations_modified': False,
    }


def link_tokencut(destination):
    # Upstream already has datasets/VOC2007/download_voc07.sh. Link its child.
    link = ROOT / 'TokenCut/datasets/VOC2007/VOCdevkit'
    target = destination / 'VOCdevkit'
    if link.is_symlink() and link.resolve() == target:
        return str(link)
    if link.exists() or link.is_symlink():
        raise FileExistsError(f'Refusing to replace existing data link/directory: {link}')
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    return str(link)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/voc2007_preparation.json')
    args = parser.parse_args()
    destination = args.data_root.expanduser().resolve()
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'data_root': str(destination), 'archives': []}
    for name in ARCHIVES:
        path = destination / 'archives' / name
        entry = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
        if not args.verify_only:
            print(f'Extracting/verifying {name}', flush=True)
            entry.update(extract_archive(path, destination))
        report['archives'].append(entry)
    report.update(verify_dataset(destination))
    report['tokencut_data_link'] = link_tokencut(destination)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
