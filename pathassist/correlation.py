"""Correlate dataset visual similarity with model behaviour and layer activations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset_similarity import (
  CATALOG,
  REFERENCE_ID,
  catalog_as_dicts,
  compare_image_fingerprints,
  fingerprint_vector,
  image_fingerprint,
  summarize_tiles,
)


@dataclass
class SampleAnalysis:
  dataset_id: str
  case_id: str
  label: int | None
  stain_similarity: float
  catalog_similarity: float
  model_score: float
  grad_cam_max: float
  grad_cam_entropy: float
  late_layer_activation: float
  score_error: float | None  # |score - label| when label known

  def to_dict(self) -> dict[str, Any]:
    return {
      "dataset_id": self.dataset_id,
      "case_id": self.case_id,
      "label": self.label,
      "stain_similarity": round(self.stain_similarity, 4),
      "catalog_similarity": round(self.catalog_similarity, 4),
      "model_score": round(self.model_score, 4),
      "grad_cam_max": round(self.grad_cam_max, 4),
      "grad_cam_entropy": round(self.grad_cam_entropy, 4),
      "late_layer_activation": round(self.late_layer_activation, 4),
      "score_error": round(self.score_error, 4) if self.score_error is not None else None,
    }


def _entropy_normalized(arr: np.ndarray) -> float:
  flat = arr.astype(np.float64).ravel()
  flat = flat / max(flat.sum(), 1e-8)
  flat = flat[flat > 1e-12]
  if flat.size == 0:
    return 0.0
  ent = -float(np.sum(flat * np.log(flat)))
  max_ent = float(np.log(flat.size))
  return ent / max_ent if max_ent > 0 else 0.0


def _catalog_by_id(dataset_id: str) -> float:
  for entry in CATALOG:
    if entry.id == dataset_id:
      return entry.composite_similarity
  return 0.0


def load_reference_tiles(reference_dir: Path, limit: int = 10) -> list[np.ndarray]:
  paths = sorted(reference_dir.glob("*.png"))[:limit]
  if not paths:
    return []
  from PIL import Image

  return [np.array(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]


def stain_similarity_to_reference(image: np.ndarray, ref_summary: dict[str, float]) -> float:
  ref_vec = np.array(list(ref_summary.values()), dtype=np.float32)
  other_vec = fingerprint_vector(image_fingerprint(image))
  denom = float(np.linalg.norm(ref_vec) * np.linalg.norm(other_vec))
  if denom < 1e-8:
    return 0.0
  return float(np.dot(ref_vec, other_vec) / denom)


def analyze_sample(
  image: np.ndarray,
  dataset_id: str,
  case_id: str,
  model,
  model_config: dict,
  device: str,
  ref_summary: dict[str, float],
  label: int | None = None,
) -> SampleAnalysis:
  from .nn_explain import explain_tile

  from .nn_explain import grad_cam

  stain_sim = stain_similarity_to_reference(image, ref_summary)
  explanation = explain_tile(model, image, model_config, device)
  cam = grad_cam(model, image, model_config, device)
  layers = explanation.get("layers", [])
  late_act = float(layers[-1]["mean_activation"]) if layers else 0.0
  score = float(explanation["score"])
  err = abs(score - float(label)) if label is not None else None

  return SampleAnalysis(
    dataset_id=dataset_id,
    case_id=case_id,
    label=label,
    stain_similarity=stain_sim,
    catalog_similarity=_catalog_by_id(dataset_id),
    model_score=score,
    grad_cam_max=float(cam.max()),
    grad_cam_entropy=_entropy_normalized(cam),
    late_layer_activation=late_act,
    score_error=err,
  )


def pearson(xs: list[float], ys: list[float]) -> float | None:
  if len(xs) < 2 or len(xs) != len(ys):
    return None
  a, b = np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
  if a.std() < 1e-12 or b.std() < 1e-12:
    return None
  return float(np.corrcoef(a, b)[0, 1])


def build_correlation_report(samples: list[SampleAnalysis]) -> dict[str, Any]:
  stain_sims = [s.stain_similarity for s in samples]
  catalog_sims = [s.catalog_similarity for s in samples]
  scores = [s.model_score for s in samples]
  cam_max = [s.grad_cam_max for s in samples]
  cam_ent = [s.grad_cam_entropy for s in samples]
  late_act = [s.late_layer_activation for s in samples]
  errors = [s.score_error for s in samples if s.score_error is not None]

  labeled = [s for s in samples if s.label is not None and s.score_error is not None]
  stain_vs_error = (
    pearson([s.stain_similarity for s in labeled], [s.score_error for s in labeled])
    if labeled
    else None
  )
  catalog_vs_error = (
    pearson([s.catalog_similarity for s in labeled], [s.score_error for s in labeled])
    if labeled
    else None
  )

  return {
    "catalog": catalog_as_dicts(),
    "reference_dataset": REFERENCE_ID,
    "sample_count": len(samples),
    "samples": [s.to_dict() for s in samples],
    "correlations": {
      "stain_similarity_vs_score_error": stain_vs_error,
      "catalog_similarity_vs_score_error": catalog_vs_error,
      "stain_similarity_vs_grad_cam_max": pearson(stain_sims, cam_max),
      "stain_similarity_vs_late_activation": pearson(stain_sims, late_act),
      "catalog_similarity_vs_model_score": pearson(catalog_sims, scores),
      "grad_cam_max_vs_model_score": pearson(cam_max, scores),
    },
    "interpretation": _interpret_correlations(
      stain_vs_error, catalog_vs_error, pearson(stain_sims, cam_max)
    ),
  }


def _interpret_correlations(
  stain_vs_error: float | None,
  catalog_vs_error: float | None,
  stain_vs_cam: float | None,
) -> list[str]:
  notes: list[str] = []
  if stain_vs_error is not None:
    if stain_vs_error < -0.3:
      notes.append(
        "Higher stain similarity to PCam tends to lower prediction error "
        "(model generalises better on visually similar tiles)."
      )
    elif stain_vs_error > 0.3:
      notes.append(
        "Prediction error rises with stain similarity — possible overfitting to PCam colour stats."
      )
  if catalog_vs_error is not None and catalog_vs_error < -0.3:
    notes.append(
      "Datasets closer in task/organ (Camelyon family) show lower score error."
    )
  if stain_vs_cam is not None and abs(stain_vs_cam) > 0.3:
    direction = "more" if stain_vs_cam > 0 else "less"
    notes.append(
      f"Domain-shifted tiles trigger {direction} focused Grad-CAM peaks "
      f"(r={stain_vs_cam:.2f})."
    )
  if not notes:
    notes.append(
      "Insufficient samples or weak correlation — export more PCam tiles and "
      "external datasets for a stable estimate."
    )
  return notes


def save_report(report: dict[str, Any], output_path: Path) -> Path:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
  return output_path
