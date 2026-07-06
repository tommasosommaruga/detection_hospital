#!/usr/bin/env python3
"""Copy a Colab-downloaded ensemble checkpoint into models/<organ>/ensemble.pt."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "checkpoint",
    type=Path,
    help="Downloaded file (e.g. pathassist_gastrointestinal_ensemble.pt)",
  )
  parser.add_argument(
    "--organ",
    help="Organ id override (default: read organ_id from checkpoint)",
  )
  args = parser.parse_args()

  src = args.checkpoint.expanduser().resolve()
  if not src.is_file():
    print(f"Not found: {src}", file=sys.stderr)
    return 1

  payload = torch.load(src, map_location="cpu", weights_only=False)
  organ_id = args.organ or payload.get("organ_id")
  if not organ_id:
    dataset = str(payload.get("dataset", ""))
    if "PatchCamelyon" in dataset or payload.get("tile_size") == 96:
      organ_id = "lymph_node"
    elif "NCT" in dataset or "CRC" in dataset or payload.get("tile_size") == 224:
      organ_id = "gastrointestinal"
    else:
      print(
        "Could not detect organ_id — pass --organ (e.g. gastrointestinal)",
        file=sys.stderr,
      )
      return 1

  dest = ROOT / "models" / organ_id / "ensemble.pt"
  dest.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(src, dest)
  print(f"Installed {organ_id} model → {dest}")
  print(f"  tile_size={payload.get('tile_size')}  members={len(payload.get('members', []))}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
