#!/usr/bin/env python3
"""Export real PatchCamelyon test tiles and run the ensemble pipeline on them."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "real_pcam"
MANIFEST = OUT_DIR / "manifest.csv"
from pathassist.organs import default_checkpoint_path

CHECKPOINT = default_checkpoint_path(ROOT)
CONFIG = ROOT / "config" / "pcam_test.yaml"


def export_samples(n_each: int = 5) -> list[dict]:
  from datasets import load_dataset

  OUT_DIR.mkdir(parents=True, exist_ok=True)
  ds = load_dataset("1aurent/PatchCamelyon", split="test")
  labels = np.array(ds["label"])

  rows = []
  for label, name in [(0, "normal"), (1, "metastasis")]:
    idx = np.where(labels == label)[0][:n_each]
    for i, row_idx in enumerate(idx):
      img = np.array(ds[int(row_idx)]["image"], dtype=np.uint8)
      fname = f"{name}_{i:02d}.png"
      path = OUT_DIR / fname
      Image.fromarray(img).save(path)
      rows.append(
        dict(case_id=f"PCAM-{name[:4].upper()}-{i:02d}", image=path, label=int(label))
      )
      print(f"  saved {fname}  label={label}")

  with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["case_id", "image", "label"])
    writer.writeheader()
    for row in rows:
      writer.writerow(
        dict(case_id=row["case_id"], image=str(row["image"]), label=row["label"])
      )
  return rows


def run_pipeline(rows: list[dict]) -> list[dict]:
  results = []
  for row in rows:
    cmd = [
      sys.executable,
      "-m",
      "pathassist.cli",
      "--config",
      str(CONFIG),
      "run",
      "--case-id",
      row["case_id"],
      "--image",
      str(row["image"]),
      "--scorer",
      "ensemble",
      "--checkpoint",
      str(CHECKPOINT),
      "--output-dir",
      str(ROOT / "outputs"),
      "--device",
      "cpu",
    ]
    print(f"\n>>> {row['case_id']} ({'metastasis' if row['label'] else 'normal'})")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(proc.stdout.strip())
    if proc.returncode != 0:
      print(proc.stderr.strip())
      raise SystemExit(proc.returncode)

    # parse case score from stdout
    score = 0.0
    for line in proc.stdout.splitlines():
      if "score" in line and "priority" in line:
        score = float(line.split("score")[1].strip())
    pred = 1 if score >= 0.5 else 0
    ok = pred == row["label"]
    results.append(dict(**row, score=score, pred=pred, correct=ok))
  return results


def main() -> None:
  print("Exporting real PatchCamelyon test tiles...")
  rows = export_samples(n_each=5)
  print(f"\nRunning ensemble on {len(rows)} real tiles (tile_size=96)...")
  results = run_pipeline(rows)

  correct = sum(r["correct"] for r in results)
  print(f"\n{'='*50}")
  print(f"Results: {correct}/{len(results)} correct tile-level calls")
  for r in results:
    tag = "OK" if r["correct"] else "MISS"
    truth = "metastasis" if r["label"] else "normal"
    pred = "metastasis" if r["pred"] else "normal"
    print(f"  [{tag}] {r['case_id']}: true={truth} pred={pred} score={r['score']:.2f}")

  normals = [r for r in results if r["label"] == 0]
  mets = [r for r in results if r["label"] == 1]
  if mets:
    recall = sum(r["pred"] == 1 for r in mets) / len(mets)
    print(f"  Recall (metastasis): {recall:.0%}")
  if normals:
    spec = sum(r["pred"] == 0 for r in normals) / len(normals)
    print(f"  Specificity (normal): {spec:.0%}")


if __name__ == "__main__":
  main()
