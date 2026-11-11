"""Collect VOC12/COCO20k sweep results without selecting a best tau."""

import argparse
import csv
import json
from pathlib import Path


DATASETS = {'VOC12': ('VOC12_trainval', 11540), 'COCO20k': ('COCO20k_train', 19817)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--gpu', choices=('6', '7'))
    parser.add_argument('--require-complete', action='store_true',
                        help='Return nonzero unless all selected rows have valid full-dataset results.')
    args = parser.parse_args()
    root = args.run_root.resolve()
    dataset_name, expected_images = DATASETS[args.dataset]
    statuses = {}
    for gpu in ('6', '7'):
        status_file = root / f'gpu{gpu}_status.tsv'
        if status_file.is_file():
            with status_file.open() as stream:
                for row in csv.DictReader(stream, delimiter='\t'):
                    statuses[gpu, row['method'], row['tau']] = row['exit_code']

    rows = []
    with (root / 'manifest.tsv').open() as stream:
        for gpu, method, tau, directory in csv.reader(stream, delimiter='\t'):
            if args.gpu and gpu != args.gpu:
                continue
            row = {'dataset': args.dataset, 'gpu': gpu, 'method': method, 'tau': tau,
                   'status': 'pending_or_running', 'corloc_percent': '',
                   'successful_images': '', 'evaluated_images': '', 'output_dir': directory}
            results = list(Path(directory).glob(f'{dataset_name}/TokenCut-*/results.json'))
            if results:
                if len(results) != 1:
                    row['status'] = 'ambiguous_results'
                else:
                    try:
                        result = json.loads(results[0].read_text())
                        valid = (result['encoder'] == method and result['tau'] == float(tau)
                                 and result['dataset'] == dataset_name
                                 and result['evaluated_images'] == expected_images
                                 and result['dataset_images'] == expected_images
                                 and not result['subset_evaluation']
                                 and result['which_features'] == 'output'
                                 and result['eps'] == 1e-5 and not result['remove_hards']
                                 and result['iou_threshold'] == 0.5
                                 and result['iou_comparison'] == '>=')
                        row['status'] = 'completed' if valid else 'invalid_protocol'
                        for key in ('corloc_percent', 'successful_images', 'evaluated_images'):
                            row[key] = result[key]
                    except (OSError, ValueError, KeyError, TypeError):
                        row['status'] = 'invalid_result'
            exit_code = statuses.get((gpu, method, tau))
            if exit_code is not None and exit_code != '0':
                row['status'] = 'failed_exit_' + exit_code
            elif exit_code == '0' and not results:
                row['status'] = 'missing_result'
            rows.append(row)
    if not rows:
        parser.error('No matching rows in manifest.tsv')

    suffix = f'_gpu{args.gpu}' if args.gpu else ''
    csv_path = root / f'summary{suffix}.csv'
    temporary = csv_path.with_suffix('.csv.tmp')
    with temporary.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    lines = [f'Dataset: {dataset_name} ({expected_images} images).', '',
             '| GPU | Method | tau | Status | CorLoc (%) | Correct / evaluated |',
             '| --- | --- | --- | --- | ---: | ---: |']
    for row in rows:
        metric = f"{row['corloc_percent']:.4f}" if row['corloc_percent'] != '' else '--'
        count = f"{row['successful_images']} / {row['evaluated_images']}" if row['evaluated_images'] != '' else '--'
        lines.append(f"| {row['gpu']} | {row['method']} | {row['tau']} | {row['status']} | {metric} | {count} |")
    markdown_path = root / f'summary{suffix}.md'
    temporary_markdown = markdown_path.with_suffix('.md.tmp')
    temporary_markdown.write_text('\n'.join(lines) + '\n')
    temporary_markdown.replace(markdown_path)
    completed = sum(row['status'] == 'completed' for row in rows)
    print(f'Summary: {csv_path}; {completed}/{len(rows)} completed')
    if args.require_complete:
        expected_combinations = {
            (str(6 + index % 2), method, tau)
            for index, (method, tau) in enumerate(
                (method, tau)
                for method in ('dino', 'ibot', 'mae', 'condgen')
                for tau in ('0.10', '0.15', '0.20', '0.25', '0.30')
            )
            if args.gpu is None or str(6 + index % 2) == args.gpu
        }
        combinations = {(row['gpu'], row['method'], row['tau']) for row in rows}
        if (completed != len(rows) or len(rows) != len(expected_combinations)
                or combinations != expected_combinations):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
