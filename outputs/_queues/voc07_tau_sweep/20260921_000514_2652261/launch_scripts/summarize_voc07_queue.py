"""Collect completed VOC07 sweep runs without selecting a best tau."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--gpu', choices=('6', '7'))
    args = parser.parse_args()
    root = args.run_root.resolve()
    rows = []
    with (root / 'manifest.tsv').open() as stream:
        for gpu, method, tau, directory in csv.reader(stream, delimiter='\t'):
            if args.gpu and gpu != args.gpu:
                continue
            row = {'gpu': gpu, 'method': method, 'tau': tau, 'status': 'pending_or_running',
                   'corloc_percent': '', 'successful_images': '', 'evaluated_images': '',
                   'output_dir': directory}
            results = list(Path(directory).glob('VOC07_trainval/TokenCut-*/results.json'))
            if results:
                if len(results) != 1:
                    raise ValueError(f'Multiple results for {method}, tau={tau}')
                result = json.loads(results[0].read_text())
                valid = (result['encoder'] == method and result['tau'] == float(tau)
                         and result['dataset'] == 'VOC07_trainval'
                         and result['evaluated_images'] == 5011 and not result['subset_evaluation']
                         and result['which_features'] == 'output' and not result['remove_hards']
                         and result['iou_comparison'] == '>=')
                row['status'] = 'completed' if valid else 'invalid_protocol'
                for key in ('corloc_percent', 'successful_images', 'evaluated_images'):
                    row[key] = result[key]
            status_file = root / f'gpu{gpu}_status.tsv'
            if status_file.exists():
                with status_file.open() as statuses:
                    for status in csv.DictReader(statuses, delimiter='\t'):
                        if status['method'] == method and status['tau'] == tau and status['exit_code'] != '0':
                            row['status'] = 'failed_exit_' + status['exit_code']
            rows.append(row)
    suffix = f'_gpu{args.gpu}' if args.gpu else ''
    csv_path = root / f'summary{suffix}.csv'
    temporary = csv_path.with_suffix('.csv.tmp')
    with temporary.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    lines = ['| GPU | Method | tau | Status | CorLoc (%) | Correct / evaluated |',
             '| --- | --- | --- | --- | ---: | ---: |']
    for row in rows:
        metric = f"{row['corloc_percent']:.4f}" if row['corloc_percent'] != '' else '--'
        count = f"{row['successful_images']} / {row['evaluated_images']}" if row['evaluated_images'] != '' else '--'
        lines.append(f"| {row['gpu']} | {row['method']} | {row['tau']} | {row['status']} | {metric} | {count} |")
    (root / f'summary{suffix}.md').write_text('\n'.join(lines) + '\n')
    print(f"Summary: {csv_path}; {sum(row['status'] == 'completed' for row in rows)}/{len(rows)} completed")


if __name__ == '__main__':
    main()
