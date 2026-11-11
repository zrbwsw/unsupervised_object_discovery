#!/usr/bin/env python3
"""Summarize and validate the DINOv3 ViT-7B/16 tau sweep."""

import argparse
import csv
import json
import os
from pathlib import Path


DATASETS = {
    "VOC07": ("VOC07_trainval", 5011),
    "VOC12": ("VOC12_trainval", 11540),
    "COCO20k": ("COCO20k_train", 19817),
}
DEFAULT_TAUS = "0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40"
METHOD = "dinov3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--gpu")
    parser.add_argument("--taus", default=os.environ.get("TAUS", DEFAULT_TAUS))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    taus = tuple(args.taus.split(","))

    statuses = {}
    status_paths = [root / f"gpu{args.gpu}_status.tsv"] if args.gpu else sorted(root.glob("gpu*_status.tsv"))
    for path in status_paths:
        if not path.is_file():
            continue
        with path.open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                statuses[(row["dataset"], row["method"], row["tau"])] = row["exit_code"]

    rows = []
    seen = set()
    with (root / "manifest.tsv").open() as stream:
        for gpu, dataset, method, tau, directory in csv.reader(stream, delimiter="\t"):
            key = (dataset, method, tau)
            if key in seen:
                raise ValueError(f"Duplicate manifest combination: {key}")
            seen.add(key)
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
                    result_file = result_files[0]
                    result = json.loads(result_file.read_text())
                    run_config = json.loads(result_file.with_name("run_config.json").read_text())
                    checkpoint = run_config["checkpoint"]
                    valid = (
                        result["encoder"] == method == METHOD
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
                        and run_config["args"]["arch"] == "vit_7b"
                        and run_config["protocol"]["register_tokens_in_graph"] is False
                        and run_config["protocol"]["position_interpolation"].startswith("DINOv3 RoPE")
                        and checkpoint["strict_load"] is True
                        and checkpoint["dtype"] == "float32"
                        and checkpoint["loaded_parameter_count"] == checkpoint["total_parameters"]
                        and checkpoint["register_tokens_removed_before_ncut"] == 4
                        and checkpoint["model_config"]["hidden_size"] == 4096
                        and checkpoint["model_config"]["num_hidden_layers"] == 40
                    )
                    row["status"] = "completed" if valid else "invalid_protocol"
                    for key in ("corloc_percent", "successful_images", "evaluated_images", "elapsed_seconds"):
                        row[key] = result.get(key, "")
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    row["status"] = "invalid_result"
            elif len(result_files) > 1:
                row["status"] = "ambiguous_results"
            exit_code = statuses.get((dataset, method, tau))
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

    lines = [
        "| GPU | Dataset | Method | tau | Status | CorLoc (%) | Evaluated | Time (s) |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        corloc = f"{float(row['corloc_percent']):.4f}" if row["corloc_percent"] != "" else "--"
        elapsed = f"{float(row['elapsed_seconds']):.1f}" if row["elapsed_seconds"] != "" else "--"
        evaluated = f"{row['successful_images']}/{row['evaluated_images']}" if row["evaluated_images"] != "" else "--"
        lines.append(
            f"| {row['gpu']} | {row['dataset']} | {row['method']} | {row['tau']} | "
            f"{row['status']} | {corloc} | {evaluated} | {elapsed} |"
        )
    csv_path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    completed = sum(row["status"] == "completed" for row in rows)
    print(f"Summary: {csv_path}; {completed}/{len(rows)} completed")

    if args.require_complete:
        expected = {(dataset, METHOD, tau) for dataset in DATASETS for tau in taus}
        actual = {(row["dataset"], row["method"], row["tau"]) for row in rows}
        if completed != len(rows) or len(rows) != len(expected) or actual != expected:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
