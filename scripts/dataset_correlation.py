#!/usr/bin/env python3
"""Compare pathology datasets to PatchCamelyon and correlate with NN explainability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))


def _load_ensemble_member(checkpoint: Path, device: str, index: int = 0):
  from pathassist.ensemble import load_ensemble_checkpoint

  ensemble, _meta = load_ensemble_checkpoint(checkpoint, device=device)
  member = ensemble.members[index]
  return member["model"], member["model_config"]


def _collect_pcam_samples(reference_dir: Path, limit: int) -> list[tuple[str, np.ndarray, int | None]]:
  from pathassist.correlation import load_reference_tiles

  tiles = load_reference_tiles(reference_dir, limit=limit)
  out = []
  for i, img in enumerate(tiles):
    label = 1 if "meta" in reference_dir.joinpath(f"../").name else None
    out.append((f"pcam_{i:02d}", img, label))
  # infer label from filename when exported by test_real_pcam
  labeled = []
  for path in sorted(reference_dir.glob("*.png"))[:limit]:
    name = path.stem
    label = 1 if name.startswith("metastasis") else 0 if name.startswith("normal") else None
    from PIL import Image

    img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    labeled.append((path.stem, img, label))
  return labeled if labeled else out


def _try_bach_samples(limit: int = 3) -> list[tuple[str, np.ndarray, int | None]]:
  try:
    from datasets import load_dataset
    from PIL import Image
  except ImportError:
    return []

  try:
    ds = load_dataset("1aurent/BACH", split="train")
  except Exception:
    return []

  out = []
  for i in range(min(limit, len(ds))):
    row = ds[i]
    img = np.array(row["image"].convert("RGB"), dtype=np.uint8)
  # resize to 96 for fair comparison with PCam
    from PIL import Image as PILImage

    small = PILImage.fromarray(img).resize((96, 96), PILImage.Resampling.BILINEAR)
    out.append((f"bach_{i:02d}", np.array(small, dtype=np.uint8), int(row.get("label", -1))))
  return out


def _synthetic_domain_shift(ref: np.ndarray, seed: int) -> np.ndarray:
  """Simulate stain/brightness shift when external data unavailable."""
  rng = np.random.default_rng(seed)
  shifted = ref.astype(np.float32)
  shifted[:, :, 0] *= float(rng.uniform(0.85, 1.15))
  shifted[:, :, 2] *= float(rng.uniform(0.80, 1.20))
  shifted += rng.uniform(-15, 15, size=3)
  return np.clip(shifted, 0, 255).astype(np.uint8)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--reference-dir", type=Path, default=ROOT / "outputs" / "real_pcam")
  from pathassist.organs import default_checkpoint_path

  parser.add_argument("--checkpoint", type=Path, default=default_checkpoint_path(ROOT))
  parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "dataset_correlation.json")
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--member", type=int, default=0, help="Ensemble member index for explainability")
  parser.add_argument("--limit", type=int, default=10)
  args = parser.parse_args()

  from pathassist.correlation import (
    analyze_sample,
    build_correlation_report,
    load_reference_tiles,
    save_report,
  )
  from pathassist.dataset_similarity import summarize_tiles

  ref_tiles = load_reference_tiles(args.reference_dir, limit=args.limit)
  if not ref_tiles:
    print(f"No PNG tiles in {args.reference_dir}; run scripts/test_real_pcam.py first.")
    return 1

  ref_summary = summarize_tiles(ref_tiles)
  model, model_config = _load_ensemble_member(args.checkpoint, args.device, args.member)

  analyses = []
  for name, img, label in _collect_pcam_samples(args.reference_dir, args.limit):
    analyses.append(
      analyze_sample(
        img, "patchcamelyon", name, model, model_config, args.device, ref_summary, label
      )
    )

  for name, img, label in _try_bach_samples(limit=3):
    analyses.append(
      analyze_sample(img, "bach", name, model, model_config, args.device, ref_summary, label)
    )

  # Synthetic proxies for Camelyon+/CRC when HF unavailable
  for dataset_id, seed in [("camelyon_plus", 1), ("nct_crc_he", 2), ("breakhis", 3)]:
    proxy = _synthetic_domain_shift(ref_tiles[seed % len(ref_tiles)], seed)
    analyses.append(
      analyze_sample(
        proxy, dataset_id, f"{dataset_id}_proxy", model, model_config,
        args.device, ref_summary, label=None,
      )
    )

  report = build_correlation_report(analyses)
  save_report(report, args.output)

  print(f"Wrote {args.output}")
  print("\nDataset catalog (heuristic similarity to PCam):")
  for row in report["catalog"][:5]:
    print(f"  {row['composite_similarity']:.2f}  {row['name']}")
  print("\nCorrelations:")
  for key, val in report["correlations"].items():
    print(f"  {key}: {val}")
  print("\nInterpretation:")
  for line in report["interpretation"]:
    print(f"  - {line}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
