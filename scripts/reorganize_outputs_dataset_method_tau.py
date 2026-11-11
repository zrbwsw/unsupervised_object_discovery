#!/usr/bin/env python3
"""Reorganize completed object-discovery outputs by dataset/method/tau.

The completed queues wrote result directories below queue-specific run roots.
This script moves each complete tau directory as a unit, archives queue
metadata separately, and records an old-to-new path mapping.  It deliberately
requires ``--apply`` for mutations and refuses to run while an evaluator is
alive.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT / "outputs"


def result_count(path: Path) -> int:
    return sum(1 for _ in path.rglob("results.json"))


def choose_run(sweep: str, expected: int) -> Path:
    root = OUTPUTS / sweep
    candidates = sorted(
        p for p in root.iterdir()
        if p.is_dir() and result_count(p) == expected
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {expected}-result run under {root}, found {candidates}"
        )
    return candidates[0]


def canonical_records():
    records = []
    queues = []

    base = [
        ("voc07_tau_sweep", "VOC07", 20),
        ("voc12_tau_sweep", "VOC12", 20),
        ("coco20k_tau_sweep", "COCO20k", 20),
    ]
    for sweep, dataset, expected in base:
        run = choose_run(sweep, expected)
        queues.append((sweep, run))
        for method_dir in sorted(p for p in run.iterdir() if p.is_dir() and p.name != "launch_scripts"):
            for tau_dir in sorted(p for p in method_dir.glob("tau_*") if p.is_dir()):
                records.append((dataset, method_dir.name, tau_dir.name, tau_dir, run, sweep))

    for sweep, expected in (("high_tau_sweep", 24), ("condgen_tau045_050_sweep", 6)):
        run = choose_run(sweep, expected)
        queues.append((sweep, run))
        for dataset_dir in sorted(p for p in run.iterdir() if p.is_dir() and p.name != "launch_scripts"):
            for method_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
                for tau_dir in sorted(p for p in method_dir.glob("tau_*") if p.is_dir()):
                    records.append((dataset_dir.name, method_dir.name, tau_dir.name,
                                    tau_dir, run, sweep))
    if len(records) != 90:
        raise RuntimeError(f"Expected 90 complete experiment directories, found {len(records)}")
    return records, queues


def stale_queue_roots(queues):
    """Return completed/abandoned queue roots that contain no canonical result run."""
    known = {run for _, run in queues}
    roots = []
    for sweep in ("high_tau_sweep", "voc12_then_coco_tau_sweep"):
        root = OUTPUTS / sweep
        if root.is_dir():
            for child in sorted(p for p in root.iterdir() if p.is_dir()):
                if child not in known:
                    roots.append((sweep, child))
    return roots


def check_no_evaluator():
    proc = subprocess.run(
        ["bash", "-lc", "pgrep -af 'main_tokencut.py|eval_object_discovery_.*_b16.sh' || true"],
        text=True, capture_output=True, check=True,
    )
    lines = [
        line for line in proc.stdout.splitlines()
        if "reorganize_outputs" not in line and "pgrep -af" not in line
    ]
    if lines:
        raise RuntimeError("An evaluator appears to be running; refusing to move outputs:\n" + "\n".join(lines))


def replace_paths(path: Path, mapping):
    if not path.is_file() or path.is_symlink():
        return
    try:
        text = path.read_text()
    except (UnicodeDecodeError, PermissionError):
        return
    old = text
    for source, target in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(source, target)
    if text != old:
        path.write_text(text)


def add_relocation_metadata(target: Path, source: Path):
    config_files = list(target.rglob("run_config.json"))
    if len(config_files) != 1:
        raise RuntimeError(f"Expected one run_config.json below {target}, found {config_files}")
    config_path = config_files[0]
    payload = json.loads(config_path.read_text())
    payload["relocation"] = {
        "original_output_dir": str(source),
        "relocated_output_dir": str(target),
    }
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="perform the moves")
    args = parser.parse_args()
    records, queues = canonical_records()
    stale = stale_queue_roots(queues)
    smoke = OUTPUTS / "voc12_coco_cpu_smoke_20260921_011820"

    mapping = {}
    planned = []
    for dataset, method, tau, source, run, sweep in records:
        target = OUTPUTS / dataset / method / tau
        mapping[str(source)] = str(target)
        planned.append((dataset, method, tau, source, target, run, sweep))
        if target.exists():
            raise RuntimeError(f"Target already exists; refusing to merge: {target}")
    for sweep, run in queues + stale:
        archive = OUTPUTS / "_queues" / sweep / run.name
        mapping[str(run)] = str(archive)
        if archive.exists():
            raise RuntimeError(f"Archive already exists; refusing to merge: {archive}")
    if smoke.is_dir():
        mapping[str(smoke)] = str(OUTPUTS / "_smoke" / smoke.name)

    print(f"Formal experiments: {len(planned)}")
    print(f"Queue runs to archive: {len(queues) + len(stale)}")
    print(f"Smoke directory: {smoke if smoke.is_dir() else 'none'}")
    if not args.apply:
        for dataset, method, tau, source, target, *_ in planned:
            print(f"{source} -> {target}")
        print("Dry run only. Re-run with --apply to perform the reorganization.")
        return

    check_no_evaluator()
    archive_roots = {}
    for sweep, run in queues + stale:
        archive = OUTPUTS / "_queues" / sweep / run.name
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run), str(archive))
        archive_roots[run] = archive

    # Move each tau directory from its archived queue root into the canonical tree.
    for dataset, method, tau, source, target, run, sweep in planned:
        archived_source = archive_roots[run] / source.relative_to(run)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(archived_source), str(target))
        add_relocation_metadata(target, source)

    if smoke.is_dir():
        smoke_target = OUTPUTS / "_smoke" / smoke.name
        smoke_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(smoke), str(smoke_target))

    # Move queue-level latest_run pointers into their archive roots and remove
    # now-empty legacy sweep directories.
    for sweep, _run in queues + stale:
        legacy_root = OUTPUTS / sweep
        latest = legacy_root / "latest_run.txt"
        archive_parent = OUTPUTS / "_queues" / sweep
        if latest.is_file():
            target_latest = archive_parent / "latest_run.txt"
            target_latest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(latest), str(target_latest))
        if legacy_root.is_dir() and not any(legacy_root.iterdir()):
            legacy_root.rmdir()

    manifest = OUTPUTS / "relocation_manifest.tsv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("kind", "dataset", "method", "tau", "original_path", "current_path"))
        for dataset, method, tau, source, target, *_ in planned:
            writer.writerow(("experiment", dataset, method, tau, source, target))
        for sweep, run in queues + stale:
            writer.writerow(("queue", "", "", "", run, archive_roots[run]))
        if smoke.is_dir():
            writer.writerow(("smoke", "", "", "", smoke, OUTPUTS / "_smoke" / smoke.name))

    # Queue metadata and repository reports keep useful absolute paths. Point
    # them to their new locations while preserving run_config execution args;
    # each relocated run_config also carries its explicit relocation record.
    for path in (OUTPUTS / "_queues").rglob("*"):
        replace_paths(path, mapping)
    for path in (OUTPUTS / "_smoke").rglob("*") if (OUTPUTS / "_smoke").exists() else []:
        replace_paths(path, mapping)
    reports = PROJECT / "reports"
    for path in reports.glob("*"):
        replace_paths(path, mapping)
    for path in (PROJECT / "configs").glob("*.json"):
        replace_paths(path, mapping)
    replace_paths(PROJECT / "README.md", mapping)

    layout = OUTPUTS / "README.md"
    layout.write_text(
        "# Output layout\n\n"
        "Canonical experiment results use `outputs/<dataset>/<method>/tau_<value>/`.\n\n"
        "Queue logs, manifests, copied launch scripts and status files are archived under `_queues/`; "
        "CPU smoke runs are under `_smoke/`; qualitative visualizations remain under `visualizations/`.\n\n"
        "The complete old-to-new mapping is in `relocation_manifest.tsv`. Each relocated run_config.json "
        "also records its original and current output directory.\n"
    )
    print(f"Applied. Mapping written to {manifest}")


if __name__ == "__main__":
    main()
