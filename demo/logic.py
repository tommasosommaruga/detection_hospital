"""Pure helpers for the PathAssist workstation (unit-testable, no FastAPI)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PRIORITY_ORDER = {"URGENT": 0, "REVIEW_QC": 1, "HIGH": 2, "ROUTINE": 3}
ALLOWED_IMAGE_FILES = frozenset({
  "source.png", "heatmap.png", "uncertainty.png",
})


def safe_image_filename(filename: str) -> bool:
  """Allow standard artifacts and NN explainability overlays."""
  if filename in ALLOWED_IMAGE_FILES:
    return True
  if not filename.endswith(".png"):
    return False
  return "_explain_" in filename
VALID_DECISIONS = frozenset({"approve", "modify", "reject"})


def infer_tile_size(height: int, width: int) -> int:
  return 96 if max(height, width) <= 128 else 256


def pick_config_name(tile_size: int) -> str:
  return "pcam_test.yaml" if tile_size <= 128 else "default.yaml"


from pathassist.detection import (
  classify_case,
  detection_threshold,
  metastasis_threshold,
  min_review_score,
  prediction_correct as detection_correct,
)


TRIAGE_OVERRIDE_KEYS = ("detection_threshold", "metastasis_threshold", "min_review_score")


def recall_first_triage() -> dict[str, float]:
  """Production defaults (lymph node / pcam_test.yaml)."""
  return {
    "detection_threshold": 0.25,
    "metastasis_threshold": 0.55,
    "min_review_score": 0.15,
  }


def extract_triage_thresholds(config: dict[str, Any]) -> dict[str, float]:
  """Current triage knobs from a loaded organ config."""
  triage = config.get("triage", recall_first_triage())
  return {
    "detection_threshold": float(detection_threshold(config)),
    "metastasis_threshold": float(metastasis_threshold(config)),
    "min_review_score": float(min_review_score(config)),
  }


def merge_triage_overrides(
  config: dict[str, Any],
  overrides: dict[str, float] | None,
) -> dict[str, Any]:
  """Return a copy of config with optional triage threshold overrides."""
  if not overrides:
    return config
  merged = dict(config)
  triage = dict(merged.get("triage", {}))
  for key in TRIAGE_OVERRIDE_KEYS:
    if key in overrides and overrides[key] is not None:
      triage[key] = float(overrides[key])
  merged["triage"] = triage
  return merged


def default_triage_config() -> dict[str, Any]:
  return {"triage": recall_first_triage()}


def predict_label(case_score: float, config: dict | None = None) -> str:
  cfg = config or default_triage_config()
  return classify_case(case_score, cfg)["predicted"]


def prediction_correct(case_score: float, label: int | None, config: dict | None = None) -> bool | None:
  cfg = config or default_triage_config()
  return detection_correct(case_score, label, cfg)


def uncertainty_available(uncertainty_path: Path | None) -> bool:
  return uncertainty_path is not None and uncertainty_path.exists()


def safe_case_id(case_id: str) -> bool:
  """Reject path traversal in URL segments."""
  return bool(case_id) and case_id not in {".", ".."} and "/" not in case_id and "\\" not in case_id


def worklist_sort_key(case: dict[str, Any]) -> tuple:
  return (
    PRIORITY_ORDER.get(case.get("priority", ""), 9),
    -float(case.get("case_score", 0)),
  )


def confusion_cell(predicted: str, label: int, *, borderline_positive: bool = True) -> str:
  """Map prediction + ground truth to tp | tn | fp | fn."""
  pred_pos = predicted in {"metastasis", "borderline"} if borderline_positive else predicted == "metastasis"
  actual_pos = int(label) == 1
  if pred_pos and actual_pos:
    return "tp"
  if not pred_pos and not actual_pos:
    return "tn"
  if pred_pos and not actual_pos:
    return "fp"
  return "fn"


def production_predicted(case_score: float, config: dict[str, Any]) -> str:
  return classify_case(float(case_score), config)["predicted"]


def research_predicted_0_5(case_score: float) -> str:
  """Notebook-style single threshold — no borderline bucket."""
  return "metastasis" if float(case_score) >= 0.5 else "normal"


def _count_metrics(cells: list[str]) -> dict[str, Any]:
  tp = sum(1 for c in cells if c == "tp")
  tn = sum(1 for c in cells if c == "tn")
  fp = sum(1 for c in cells if c == "fp")
  fn = sum(1 for c in cells if c == "fn")
  total = tp + tn + fp + fn
  acc = (tp + tn) / max(1, total)
  prec = tp / max(1, tp + fp)
  rec = tp / max(1, tp + fn)
  f1 = 2 * prec * rec / max(1, prec + rec)
  return {
    "total": total,
    "tp": tp,
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1": round(f1, 4),
  }


def compute_validation_metrics(
  samples: list[dict[str, Any]],
  config: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Confusion-matrix metrics using production recall-first rules (recomputed from scores)."""
  cfg = config or default_triage_config()
  labeled = [s for s in samples if s.get("ready") and s.get("label") is not None]
  if not labeled:
    return {"ready": False, "message": "Run validation on benchmark tiles first"}

  if not all(s.get("case_score") is not None for s in labeled):
    return {
      "ready": False,
      "message": "Re-run benchmark — stored cases lack scores for production metrics",
    }

  prod_cells: list[str] = []
  research_cells: list[str] = []
  cases: list[dict[str, Any]] = []
  for sample in labeled:
    score = float(sample["case_score"])
    label = int(sample["label"])
    predicted = production_predicted(score, cfg)
    research_pred = research_predicted_0_5(score)
    prod_cell = confusion_cell(predicted, label, borderline_positive=True)
    research_cell = confusion_cell(research_pred, label, borderline_positive=False)
    prod_cells.append(prod_cell)
    research_cells.append(research_cell)
    cases.append({
      **sample,
      "predicted": predicted,
      "research_predicted": research_pred,
      "confusion_cell": prod_cell,
      "research_confusion_cell": research_cell,
      "correct": detection_correct(score, label, cfg),
      "research_correct": int(research_pred == "metastasis") == label,
      "detection_threshold": detection_threshold(cfg),
      "metastasis_threshold": metastasis_threshold(cfg),
      "min_review_score": min_review_score(cfg),
    })

  production = _count_metrics(prod_cells)
  research = _count_metrics(research_cells)

  thresholds = extract_triage_thresholds(cfg)
  return {
    "ready": True,
    "operating_mode": "recall_first",
    "borderline_counts_as_positive": True,
    "thresholds": {
      **thresholds,
      "research_threshold": 0.5,
    },
    "config_source": cfg.get("_validation_config_source"),
    "threshold_source": cfg.get("_threshold_source", "yaml"),
    **production,
    "research_0_5": research,
    "cases": cases,
  }


def build_case_payload(
  case,
  case_id: str,
  label: int | None,
  tile_size: int,
  uncertainty_path: Path | None,
  review: dict[str, Any] | None,
  analyzed_at: str,
  config: dict[str, Any] | None = None,
  organ: dict[str, Any] | None = None,
  wsi_info: Any | None = None,
) -> dict[str, Any]:
  cfg = config or default_triage_config()
  detection = classify_case(case.case_score, cfg)
  predicted = detection["predicted"]
  has_uncertainty = uncertainty_available(uncertainty_path)
  organ_info = organ or {}
  payload = {
    "case_id": case_id,
    "analyzed_at": analyzed_at,
    "tile_size": tile_size,
    "label": label,
    "label_name": ("metastasis" if label == 1 else "normal") if label is not None else None,
    "priority": case.priority,
    "case_score": round(case.case_score, 4),
    "mean_uncertainty": round(case.mean_uncertainty, 4),
    "max_uncertainty": round(case.max_uncertainty, 4),
    "mean_disagreement": round(case.mean_disagreement, 4),
    "suspicious_tiles": case.suspicious_tile_count,
    "tile_count": case.tile_count,
    "grade": case.grade.grade if case.grade else None,
    "grade_confidence": round(case.grade.confidence, 4) if case.grade else None,
    "grade_rationale": case.grade.rationale if case.grade else None,
    "qc": case.qc.to_dict() if case.qc else None,
    "review_flags": case.review_flags,
    "regions": [r.to_dict() for r in case.regions_of_interest[:8]],
    "model_name": case.model_name,
    "model_version": case.model_version,
    "predicted": predicted,
    "operating_mode": "recall_first",
    "detection_threshold": detection["detection_threshold"],
    "metastasis_threshold": detection["metastasis_threshold"],
    "min_review_score": min_review_score(cfg),
    "needs_review": detection["needs_review"],
    "correct": prediction_correct(case.case_score, label, cfg),
    "review": review,
    "paths": {
      "source": f"/api/image/{case_id}/source.png",
      "heatmap": f"/api/image/{case_id}/heatmap.png",
      "uncertainty": f"/api/image/{case_id}/uncertainty.png" if has_uncertainty else None,
      "report": f"/api/report/{case_id}",
    },
  }
  if wsi_info is not None:
    payload["wsi"] = wsi_info.to_dict()
    payload["wsi_mode"] = "full_res"
  if organ_info:
    payload.update({
      "organ_id": organ_info.get("organ_id"),
      "organ_name": organ_info.get("organ_name"),
      "organ_specialty": organ_info.get("organ_specialty"),
      "organ_task": organ_info.get("organ_task"),
      "organ_stain": organ_info.get("organ_stain"),
      "model_checkpoint": organ_info.get("model_checkpoint"),
      "metadata_detected_organ_id": organ_info.get("metadata_detected_organ_id"),
      "metadata_mismatch": organ_info.get("metadata_mismatch", False),
      "metadata_sources": organ_info.get("metadata_sources", []),
      "organ_warnings": organ_info.get("warnings", []),
    })
  return payload
