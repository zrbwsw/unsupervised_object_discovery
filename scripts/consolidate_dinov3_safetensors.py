#!/usr/bin/env python3
"""Consolidate DINOv3 sharded safetensors without materializing all tensors."""

import argparse
import json
from pathlib import Path
import sys


TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "TokenCut"))

from dinov3_adapter import consolidate_dinov3_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=Path("/data1/pretrain_models"),
                        help="sharded checkpoint directory or an existing single .safetensors file")
    parser.add_argument("--output", type=Path,
                        help="output file (default: SOURCE/dinov3_vit7b16.safetensors)")
    parser.add_argument("--overwrite", action="store_true",
                        help="atomically replace an existing consolidated output; source shards are never replaced")
    args = parser.parse_args()
    report = consolidate_dinov3_checkpoint(args.source, args.output, overwrite=args.overwrite)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
