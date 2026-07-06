#!/usr/bin/env python3
"""Load tiles from data/benchmarks/<organ>/ into the demo test cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from pathassist.test_datasets import import_local_folder, local_organ_dir


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("organ", help="organ id, e.g. breast, pulmonary")
  parser.add_argument(
    "--folder",
    type=Path,
    help="override source folder (default: data/benchmarks/<organ>)",
  )
  parser.add_argument("--max-samples", type=int, default=None)
  parser.add_argument("--no-activate", action="store_true")
  args = parser.parse_args()

  folder = args.folder or local_organ_dir(args.organ, ROOT)
  result = import_local_folder(
    args.organ,
    folder=folder,
    max_samples=args.max_samples,
    root=ROOT,
    activate=not args.no_activate,
  )
  print(
    f"Loaded {result['sample_count']} tiles for {result['organ_id']} "
    f"from {result['source_dir']} -> {result['output_dir']}"
  )


if __name__ == "__main__":
  main()
