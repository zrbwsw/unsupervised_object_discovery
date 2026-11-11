#!/usr/bin/env python3
"""Summarize the VOC07/VOC12/COCO20K high-tau sweep."""

import argparse
import csv
import json
from pathlib import Path


DATASETS = {
    "VOC07": ("VOC07_trainval", 5011),
    "VOC12": ("VOC12_trainval", 11540),
    "COCO20k": ("COCO20k_train", 19817),
}
METHODS = ("dino", "ibot", "mae", "condgen")
TAUS = ("0.35", "0.40")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--gpu", choices=("6", "7"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()

    statuses = {}
    for gpu in ("6", "7"):
        path = root / f"gpu{gpu}_status.tsv"
        if path.is_file():
            with path.open() as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    statuses[(gpu, row["dataset"], row["method"], row["tau"])] = row["exit_code"]

    rows = []
    with (root / "manifest.tsv").open() as stream:
        for gpu, dataset, method, tau, directory in csv.reader(stream, delimiter="\t"):
            if args.gpu and gpu != args.gpu:
                continue
            result_name, expected = DATASETS[dataset]
            row = {
                "gpu": gpu, "dataset": dataset, "method": method, "tau": tau,
                "status": "pending_or_running", "corloc_percent": "",
                "successful_images": "", "evaluated_images": "", "elapsed_seconds": "",
                "output_dir": directory,
            }
            result_files = list(Path(directory).glob(f"{result_name}/TokenCut-*/results.json"))
            if len(result_files) == 1:
                try:
                    result = json.loads(result_files[0].read_text())
                    valid = (
                        result["encoder"] == method
                        and result["tau"] == float(tau)
                        and result["dataset"] == result_name
                        and result["evaluated_images"] == expected
                        and result["dataset_images"] == expected
                        and result["selected_images"] == expected
                        and result["skipped_images"] == 0
                        and not result["subset_evaluation"]
                        and result["which_features"] == "output"
                        and result["eps"] == 1e-5
                        and not result["remove_hards"]
                        and result["iou_threshold"] == 0.5
                        and result["iou_comparison"] == ">="
                    )
                    row["status"] = "completed" if valid else "invalid_protocol"
                    for key in ("corloc_percent", "successful_images", "evaluated_images", "elapsed_seconds"):
                        row[key] = result.get(key, "")
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    row["status"] = "invalid_result"
            elif len(result_files) > 1:
                row["status"] = "ambiguous_results"
            exit_code = statuses.get((gpu, dataset, method, tau))
            if exit_code is not None and exit_code != "0":
                row["status"] = "failed_exit_" + exit_code
            elif exit_code == "0" and not result_files:
                row["status"] = "missing_result"
            elif exit_code is None and row["status"] == "completed":
                row["status"] = "awaiting_process_exit"
            rows.append(row)

    if not rows:
        parser.error("No matching rows in manifest.tsv")
    fieldnames = list(rows[0])
    csv_path = root / (f"summary_gpu{args.gpu}.csv" if args.gpu else "summary.csv")
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)

    md_path = csv_path.with_suffix(".md")
    lines = ["| GPU | Dataset | Method | tau | Status | CorLoc (%) | Evaluated | Time (s) |", "| --- | --- | --- | --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        corloc = f"{float(row['corloc_percent']):.4f}" if row["corloc_percent"] != "" else "--"
        elapsed = f"{float(row['elapsed_seconds']):.1f}" if row["elapsed_seconds"] != "" else "--"
        evaluated = f"{row['successful_images']}/{row['evaluated_images']}" if row["evaluated_images"] != "" else "--"
        lines.append(f"| {row['gpu']} | {row['dataset']} | {row['method']} | {row['tau']} | {row['status']} | {corloc} | {evaluated} | {elapsed} |")
    md_path.write_text("\n".join(lines) + "\n")
    completed = sum(row["status"] == "completed" for row in rows)
    print(f"Summary: {csv_path}; {completed}/{len(rows)} completed")

    if args.require_complete:
        expected = {
            (str(6 + index % 2), dataset, method, tau)
            for index, (dataset, method, tau) in enumerate(
                (dataset, method, tau)
                for dataset in DATASETS
                for method in METHODS
                for tau in TAUS
            )
            if args.gpu is None or str(6 + index % 2) == args.gpu
        }
        actual = {(row["gpu"], row["dataset"], row["method"], row["tau"]) for row in rows}
        if completed != len(rows) or len(rows) != len(expected) or actual != expected:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
