#!/usr/bin/env python3
"""Find a recall-first detection threshold on real PatchCamelyon tiles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from pathassist.config import load_config
from pathassist.detection import classify_case, sweep_thresholds
from pathassist.pipeline import analyze_case
from pathassist.scoring import EnsembleScorer

MANIFEST = ROOT / "outputs" / "real_pcam" / "manifest.csv"
from pathassist.organs import default_checkpoint_path

CHECKPOINT = default_checkpoint_path(ROOT)
CONFIG = ROOT / "config" / "pcam_test.yaml"


def score_cases() -> tuple[list[float], list[int]]:
  if not MANIFEST.exists():
    raise SystemExit("Run: python scripts/test_real_pcam.py first")

  config = load_config(CONFIG)
  scorer = EnsembleScorer(str(CHECKPOINT), device_preference="auto")
  scores: list[float] = []
  labels: list[int] = []

  import csv

  with MANIFEST.open(encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
      path = Path(row["image"])
      if not path.is_absolute():
        path = ROOT / path
      image = np.array(Image.open(path).convert("RGB"))
      case, *_ = analyze_case(
        case_id=row["case_id"],
        image=image,
        scorer=scorer,
        config=config,
        output_dir=ROOT / "outputs" / "threshold_sweep" / row["case_id"],
        audit_store=None,
      )
      scores.append(case.case_score)
      labels.append(int(row["label"]))
      print(f"  {row['case_id']}: score={case.case_score:.4f} label={row['label']}")
  return scores, labels


def main() -> None:
  print("Scoring benchmark tiles with recall-first config...")
  scores, labels = score_cases()
  current_thr = float(load_config(CONFIG)["triage"]["detection_threshold"])

  print(f"\nThreshold sweep ({len(scores)} tiles):")
  print(f"{'thr':>5}  {'recall':>7}  {'precision':>9}  {'fn':>3}  {'fp':>3}")
  best_recall_row = None
  for row in sweep_thresholds(scores, labels):
    mark = " <-- current" if abs(row["threshold"] - current_thr) < 0.001 else ""
    if row["fn"] == 0 and (best_recall_row is None or row["fp"] < best_recall_row["fp"]):
      best_recall_row = row
    print(
      f"{row['threshold']:5.2f}  {row['recall']:7.1%}  {row['precision']:9.1%}  "
      f"{int(row['fn']):3d}  {int(row['fp']):3d}{mark}"
    )

  if best_recall_row:
    print(
      f"\nBest zero-miss threshold: {best_recall_row['threshold']:.2f} "
      f"(recall {best_recall_row['recall']:.0%}, "
      f"precision {best_recall_row['precision']:.0%}, "
      f"fp={int(best_recall_row['fp'])})"
    )
    print("Set triage.detection_threshold in config/pcam_test.yaml to this value.")

  out = ROOT / "outputs" / "threshold_sweep.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(
    json.dumps({"scores": scores, "labels": labels, "sweep": sweep_thresholds(scores, labels)}, indent=2),
    encoding="utf-8",
  )
  print(f"Saved: {out}")


if __name__ == "__main__":
  main()
