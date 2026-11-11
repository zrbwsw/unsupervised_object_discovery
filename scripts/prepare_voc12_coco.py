#!/usr/bin/env python3
"""Prepare local VOC2012 and COCO2014 archives for TokenCut evaluation.

Existing extracted files are compared with archive bytes, never overwritten.
The COCO20K cache preserves TokenCut's exact filename and annotation order.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path('/data0/datasets/dino_object_discovery')
VOC_SPLITS = {'train': 5717, 'val': 5823, 'trainval': 11540}
COCO_SELECTED_COUNT = 19817
LOG = logging.getLogger('prepare_voc12_coco')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def member_target(destination, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError(f'Unsafe archive member: {name}')
    candidate = destination / relative
    if candidate.is_symlink():
        raise ValueError(f'Refusing archive file through existing symlink: {candidate}')
    target = candidate.resolve()
    if not target.is_relative_to(destination.resolve()):
        raise ValueError(f'Archive member escapes destination: {name}')
    return target


def install_file(source, target):
    """Compare an existing file or atomically publish a new file without replacement."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file():
            raise ValueError(f'Existing target is not a regular file: {target}')
        with target.open('rb') as existing:
            while True:
                block = source.read(1024 * 1024)
                if block != existing.read(1024 * 1024):
                    raise ValueError(f'Existing file differs from archive: {target}')
                if not block:
                    break
        return False
    descriptor, name = tempfile.mkstemp(prefix='.' + target.name + '.', suffix='.extracting', dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            shutil.copyfileobj(source, output, 1024 * 1024)
        temporary.chmod(0o644)
        # link() fails if another process created the target; it cannot overwrite it.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def extract_archive(path, destination):
    created = existing = directories = 0
    destination.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.zip':
        archive = zipfile.ZipFile(path)
        members = archive.infolist()
    else:
        archive = tarfile.open(path)
        members = archive.getmembers()
    with archive:
        for index, member in enumerate(members, 1):
            is_zip = isinstance(member, zipfile.ZipInfo)
            name = member.filename if is_zip else member.name
            target = member_target(destination, name)
            if is_zip:
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(f'Archive symlink is unsupported: {name}')
                directory = member.is_dir()
                file_type = stat.S_IFMT(mode)
                if not directory and file_type not in (0, stat.S_IFREG):
                    raise ValueError(f'Unsupported ZIP member type: {name}')
            else:
                directory = member.isdir()
                if not directory and not member.isfile():
                    raise ValueError(f'Unsupported TAR member type: {name}')
            if directory:
                target.mkdir(parents=True, exist_ok=True)
                directories += 1
                continue
            with (archive.open(member) if is_zip else archive.extractfile(member)) as source:
                if install_file(source, target):
                    created += 1
                else:
                    existing += 1
            if index % 5000 == 0:
                LOG.info('%s: processed %d/%d members', path.name, index, len(members))
    return {'created_files': created, 'identical_existing_files': existing,
            'directories': directories, 'archive_members': len(members)}


def verify_voc(destination):
    voc = destination / 'VOCdevkit/VOC2012'
    split_sets = {}
    split_report = {}
    for split, expected in VOC_SPLITS.items():
        path = voc / 'ImageSets/Main' / f'{split}.txt'
        image_ids = path.read_text().split()
        if len(image_ids) != expected or len(set(image_ids)) != expected:
            raise ValueError(f'VOC {split}: expected {expected} unique IDs, got {len(image_ids)}')
        split_sets[split] = set(image_ids)
        split_report[split] = {'images': expected, 'list_sha256': sha256(path)}
    if (split_sets['train'] & split_sets['val'] or
            split_sets['train'] | split_sets['val'] != split_sets['trainval']):
        raise ValueError('VOC train and val are not a disjoint partition of trainval')
    jpeg_ids = {path.stem for path in (voc / 'JPEGImages').glob('*.jpg')}
    xml_ids = {path.stem for path in (voc / 'Annotations').glob('*.xml')}
    if len(jpeg_ids) != 17125 or jpeg_ids != xml_ids or not split_sets['trainval'] <= jpeg_ids:
        raise ValueError(f'Unexpected VOC2012 full image/XML sets: {len(jpeg_ids)}, {len(xml_ids)}')
    object_counts = total_objects = zero_coordinate = hard_only = 0
    fractional_boxes = []
    for index, image_id in enumerate(sorted(jpeg_ids), 1):
        image_path = voc / 'JPEGImages' / f'{image_id}.jpg'
        annotation = ET.parse(voc / 'Annotations' / f'{image_id}.xml').getroot()
        if annotation.findtext('filename') != image_path.name:
            raise ValueError(f'VOC XML/image filename mismatch: {image_id}')
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
        xml_size = tuple(int(annotation.findtext(f'size/{field}')) for field in ('width', 'height'))
        if (width, height) != xml_size:
            raise ValueError(f'VOC XML/image dimension mismatch: {image_id}')
        objects = annotation.findall('object')
        if not objects:
            raise ValueError(f'VOC image without ground-truth objects: {image_id}')
        total_objects += len(objects)
        if image_id in split_sets['trainval']:
            object_counts += len(objects)
            hard_only += all(obj.findtext('difficult') == '1' or obj.findtext('truncated') == '1'
                             for obj in objects)
        for obj in objects:
            raw_box = [obj.findtext('bndbox/' + field) for field in ('xmin', 'ymin', 'xmax', 'ymax')]
            box = [float(value) for value in raw_box]
            if any('.' in value for value in raw_box):
                fractional_boxes.append({'image_id': image_id, 'bbox': box,
                                         'in_detection_trainval': image_id in split_sets['trainval']})
            x1, y1, x2, y2 = box
            if not (0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height):
                raise ValueError(f'VOC invalid bounding box: {image_id}: {box}')
            zero_coordinate += min(box) == 0
        if index % 2000 == 0:
            LOG.info('VOC2012: decoded and verified %d/%d JPEG/XML pairs', index, len(jpeg_ids))
    return {'directory': str(voc), 'splits': split_report, 'jpeg_count': len(jpeg_ids),
            'xml_count': len(xml_ids), 'all_images_decoded': True, 'all_xml_parsed': True,
            'xml_image_dimensions_match': True, 'bounding_boxes_in_image_bounds': True,
            'total_objects': total_objects, 'trainval_objects': object_counts,
            'trainval_images_with_only_difficult_or_truncated_objects': hard_only,
            'boxes_with_zero_coordinate': zero_coordinate, 'evaluation_split': 'trainval',
            'boxes_with_decimal_coordinates': fractional_boxes,
            'evaluation_images': VOC_SPLITS['trainval'], 'annotations_modified': False}


def ensure_cache(path, expected, verify_only):
    if path.is_symlink():
        raise ValueError(f'Refusing to use a symlink as the dataset-owned cache: {path}')
    if path.exists():
        with path.open() as source:
            if json.load(source) != expected:
                raise ValueError(f'Existing COCO20K cache differs from source selection: {path}')
        return 'verified_existing'
    if verify_only:
        raise FileNotFoundError(f'COCO20K cache missing in --verify-only mode: {path}')
    descriptor, temporary_name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w') as output:
            json.dump(expected, output)
            output.write('\n')
        temporary.chmod(0o644)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return 'created'


def verify_coco(destination, verify_only):
    local_list = destination / 'splits/coco_20k_filenames.txt'
    upstream_list = ROOT / 'TokenCut/datasets/coco_20k_filenames.txt'
    if local_list.read_bytes() != upstream_list.read_bytes():
        raise ValueError('Local COCO20K filename list differs from the TokenCut checkout')
    filenames = local_list.read_text().splitlines()
    if len(filenames) != COCO_SELECTED_COUNT or len(set(filenames)) != COCO_SELECTED_COUNT:
        raise ValueError(f'Expected {COCO_SELECTED_COUNT} unique COCO20K filenames')
    for filename in filenames:
        relative = PurePosixPath(filename)
        if len(relative.parts) != 2 or relative.parts[0] != 'train2014' or relative.suffix != '.jpg':
            raise ValueError(f'Unexpected COCO20K filename: {filename}')
    selected_ids = [int(Path(name).stem.split('_')[-1]) for name in filenames]
    if len(set(selected_ids)) != COCO_SELECTED_COUNT:
        raise ValueError('Duplicate COCO20K image IDs')
    source_path = destination / 'annotations/instances_train2014.json'
    LOG.info('Loading source COCO2014 instance annotations')
    with source_path.open() as source:
        data = json.load(source)
    all_images = {item['id']: item for item in data['images']}
    if len(all_images) != 82783 or len(all_images) != len(data['images']):
        raise ValueError(f'Unexpected COCO train2014 image count: {len(all_images)}')
    jpeg_names = {path.name for path in (destination / 'images/train2014').glob('*.jpg')}
    if jpeg_names != {item['file_name'] for item in all_images.values()}:
        raise ValueError('Full extracted COCO train2014 JPEG set differs from annotations')
    category_ids = {item['id'] for item in data['categories']}
    if len(category_ids) != 80 or len(category_ids) != len(data['categories']):
        raise ValueError('Unexpected COCO category list')
    selected_set = set(selected_ids)
    annotations_by_image = defaultdict(list)
    all_annotation_ids = set()
    for annotation in data['annotations']:
        if annotation['id'] in all_annotation_ids:
            raise ValueError(f'Duplicate COCO annotation ID: {annotation["id"]}')
        all_annotation_ids.add(annotation['id'])
        if annotation['image_id'] not in all_images or annotation['category_id'] not in category_ids:
            raise ValueError(f'Invalid COCO annotation image/category reference: {annotation["id"]}')
        if annotation['image_id'] in selected_set:
            annotations_by_image[annotation['image_id']].append(annotation)
    # O(images + annotations + selected images), preserving the original nested loop order.
    selected = {'images': [all_images[image_id] for image_id in selected_ids],
                'annotations': [annotation for image_id in selected_ids
                                for annotation in annotations_by_image[image_id]],
                'categories': data['categories']}
    empty_annotations = []
    empty_noncrowd = []
    no_valid_noncrowd = []
    invalid_boxes = []
    outside_bounds = []
    degenerate_after_rounding = []
    crowd_count = noncrowd_count = valid_noncrowd_count = 0
    min_width = min_height = float('inf')
    max_width = max_height = 0
    for index, (filename, image_id) in enumerate(zip(filenames, selected_ids), 1):
        metadata = all_images[image_id]
        if metadata['file_name'] != Path(filename).name:
            raise ValueError(f'COCO filename/image ID mismatch: {filename}')
        with Image.open(destination / 'images' / filename) as image:
            image.load()
            width, height = image.size
        if (width, height) != (metadata['width'], metadata['height']):
            raise ValueError(f'COCO annotation/image dimension mismatch: {filename}')
        min_width, max_width = min(min_width, width), max(max_width, width)
        min_height, max_height = min(min_height, height), max(max_height, height)
        annotations = annotations_by_image[image_id]
        if not annotations:
            empty_annotations.append(image_id)
        noncrowd = valid_noncrowd = 0
        for annotation in annotations:
            iscrowd = annotation['iscrowd'] == 1
            crowd_count += iscrowd
            noncrowd_count += not iscrowd
            noncrowd += not iscrowd
            box = annotation['bbox']
            valid = len(box) == 4 and all(math.isfinite(value) for value in box)
            if valid:
                x, y, box_width, box_height = box
                valid = box_width > 0 and box_height > 0
                if not (-0.01 <= x <= x + box_width <= width + 0.01 and
                        -0.01 <= y <= y + box_height <= height + 0.01):
                    outside_bounds.append(annotation['id'])
                if not iscrowd and (round(x + box_width) <= round(x) or round(y + box_height) <= round(y)):
                    degenerate_after_rounding.append(annotation['id'])
            if not valid:
                invalid_boxes.append(annotation['id'])
            valid_noncrowd += valid and not iscrowd
        valid_noncrowd_count += valid_noncrowd
        if not noncrowd:
            empty_noncrowd.append(image_id)
        if not valid_noncrowd:
            no_valid_noncrowd.append(image_id)
        if index % 2000 == 0:
            LOG.info('COCO20K: decoded and verified %d/%d selected images', index, len(selected_ids))
    cache_path = destination / 'annotations/instances_train2014_sel20k.json'
    cache_status = ensure_cache(cache_path, selected, verify_only)
    return {'directory': str(destination), 'full_train2014_images': len(all_images),
            'full_train2014_annotations': len(data['annotations']), 'categories': len(category_ids),
            'selection_file': str(local_list), 'selection_sha256': sha256(local_list),
            'selection_matches_tokencut_bytes': True, 'selected_images': len(selected_ids),
            'selected_annotations': len(selected['annotations']), 'crowd_annotations': crowd_count,
            'noncrowd_annotations': noncrowd_count, 'valid_noncrowd_annotations': valid_noncrowd_count,
            'images_with_no_annotations': {'count': len(empty_annotations), 'image_ids': empty_annotations},
            'images_with_no_noncrowd_annotations': {'count': len(empty_noncrowd), 'image_ids': empty_noncrowd},
            'images_with_no_valid_noncrowd_boxes': {'count': len(no_valid_noncrowd), 'image_ids': no_valid_noncrowd},
            'invalid_bboxes': {'count': len(invalid_boxes), 'annotation_ids': invalid_boxes},
            'bboxes_outside_image_bounds': {'count': len(outside_bounds), 'annotation_ids': outside_bounds},
            'noncrowd_boxes_degenerate_after_tokencut_rounding': {
                'count': len(degenerate_after_rounding), 'annotation_ids': degenerate_after_rounding},
            'all_selected_images_decoded': True, 'image_annotation_dimensions_match': True,
            'full_train2014_jpeg_filename_set_matches_annotations': True,
            'selected_width_range': [min_width, max_width], 'selected_height_range': [min_height, max_height],
            'cache_path': str(cache_path), 'cache_status': cache_status, 'cache_sha256': sha256(cache_path),
            'cache_image_order_matches_selection': True,
            'cache_annotation_order_matches_upstream_selection': True,
            'source_instances_sha256': sha256(source_path), 'source_annotations_modified': False}


def ensure_link(link, target, verify_only):
    target = target.resolve(strict=True)
    if link.is_symlink() and link.resolve() == target:
        return {'link': str(link), 'target': str(target), 'status': 'verified_existing'}
    if link.exists() or link.is_symlink():
        raise FileExistsError(f'Refusing to replace an existing path: {link}')
    if verify_only:
        raise FileNotFoundError(f'Dataset link missing in --verify-only mode: {link}')
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())
    return {'link': str(link), 'target': str(target), 'status': 'created'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--verify-only', action='store_true', help='Require extracted files, cache and links to already exist.')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Validate existing extraction and create missing cache/links without comparing every archive member.')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/voc12_coco_preparation.json')
    parser.add_argument('--log', type=Path, default=ROOT / 'reports/voc12_coco_preparation.log')
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.log)])
    destination = args.data_root.expanduser().resolve()
    voc = destination / 'voc2012'
    coco = destination / 'coco2014'
    report = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'data_root': str(destination), 'verify_only': args.verify_only, 'archives': []}
    archives = [(voc / 'archives/VOCtrainval_11-May-2012.tar', voc),
                (coco / 'archives/annotations_trainval2014.zip', coco),
                (coco / 'archives/train2014.zip', coco / 'images')]
    for path, output in archives:
        LOG.info('Hashing %s (%d bytes)', path.name, path.stat().st_size)
        entry = {'path': str(path), 'bytes': path.stat().st_size,
                 'sha256': sha256(path), 'destination': str(output)}
        if not args.verify_only and not args.skip_extraction:
            LOG.info('Extracting or comparing %s', path.name)
            entry.update(extract_archive(path, output))
        report['archives'].append(entry)
        LOG.info('Archive ready: %s', path.name)
    report['voc2012'] = verify_voc(voc)
    report['coco2014'] = verify_coco(coco, args.verify_only)
    datasets = ROOT / 'TokenCut/datasets'
    links = [(datasets / 'VOC2012/VOCdevkit', voc / 'VOCdevkit'),
             (datasets / 'COCO/images', coco / 'images'),
             (datasets / 'COCO/annotations', coco / 'annotations'),
             (datasets / 'instances_train2014_sel20k.json', coco / 'annotations/instances_train2014_sel20k.json')]
    report['links'] = [ensure_link(link, target, args.verify_only) for link, target in links]
    report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    LOG.info('Complete: VOC12 trainval=%d, COCO20K=%d; report=%s',
             report['voc2012']['evaluation_images'], report['coco2014']['selected_images'], args.report)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        LOG.exception('Dataset preparation failed')
        raise
