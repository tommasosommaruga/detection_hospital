"""Case-level malignancy detection thresholds (recall-first tuning).

Hospitals care more about missed metastasis than false alarms on triage.
All thresholds live in config — no magic 0.5 in application code.
"""

from __future__ import annotations

from typing import Any

from .types import TileScore

DEFAULT_DETECTION_THRESHOLD = 0.5
DEFAULT_MIN_REVIEW_SCORE = 0.20


def detection_threshold(config: dict[str, Any]) -> float:
  triage = config.get("triage", {})
  return float(triage.get("detection_threshold", triage.get("suspicious_tile_threshold", DEFAULT_DETECTION_THRESHOLD)))


def min_review_score(config: dict[str, Any]) -> float:
  return float(config.get("triage", {}).get("min_review_score", DEFAULT_MIN_REVIEW_SCORE))


def case_score_mode(config: dict[str, Any]) -> str:
  return str(config.get("triage", {}).get("case_score_mode", "top_k_mean"))


def aggregate_case_score(tile_scores: list[TileScore], config: dict[str, Any]) -> float:
  """Aggregate tile scores into one case-level score for detection."""
  if not tile_scores:
    return 0.0
  scores = sorted((ts.score for ts in tile_scores), reverse=True)
  if case_score_mode(config) == "max":
    return float(scores[0])
  top_k = max(1, min(len(scores), 10))
  return float(sum(scores[:top_k]) / top_k)


def metastasis_threshold(config: dict[str, Any]) -> float:
  triage = config.get("triage", {})
  return float(triage.get("metastasis_threshold", triage.get("suspicious_tile_threshold", 0.55)))


def classify_case(case_score: float, config: dict[str, Any]) -> dict[str, Any]:
  """Return prediction label and whether the case needs pathologist review."""
  review_floor = min_review_score(config)
  strong_thr = metastasis_threshold(config)
  review_thr = detection_threshold(config)

  if case_score >= strong_thr:
    return {
      "predicted": "metastasis",
      "detection_threshold": review_thr,
      "metastasis_threshold": strong_thr,
      "needs_review": True,
      "review_reason": "high_confidence_metastasis",
    }

  if case_score >= review_thr:
    return {
      "predicted": "borderline",
      "detection_threshold": review_thr,
      "metastasis_threshold": strong_thr,
      "needs_review": True,
      "review_reason": "recall_first_review",
    }

  if case_score >= review_floor:
    return {
      "predicted": "borderline",
      "detection_threshold": review_thr,
      "metastasis_threshold": strong_thr,
      "needs_review": True,
      "review_reason": "low_score_possible_metastasis",
    }

  return {
    "predicted": "normal",
    "detection_threshold": review_thr,
    "metastasis_threshold": strong_thr,
    "needs_review": False,
    "review_reason": None,
  }


def prediction_correct(case_score: float, label: int | None, config: dict[str, Any]) -> bool | None:
  if label is None:
    return None
  info = classify_case(case_score, config)
  # Borderline counts as positive for recall measurement — pathologist still reviews.
  pred_pos = info["predicted"] in {"metastasis", "borderline"}
  return int(pred_pos) == int(label == 1)


def sweep_thresholds(
  scores: list[float],
  labels: list[int],
  thresholds: list[float] | None = None,
) -> list[dict[str, float]]:
  """Precision/recall at each threshold — use to pick a recall-first operating point."""
  if thresholds is None:
    thresholds = [round(t, 2) for t in __import__("numpy").linspace(0.05, 0.95, 19)]

  rows = []
  labels_arr = __import__("numpy").asarray(labels, dtype=int)
  scores_arr = __import__("numpy").asarray(scores, dtype=float)

  for thr in thresholds:
    pred = (scores_arr >= thr).astype(int)
    tp = int(((pred == 1) & (labels_arr == 1)).sum())
    tn = int(((pred == 0) & (labels_arr == 0)).sum())
    fp = int(((pred == 1) & (labels_arr == 0)).sum())
    fn = int(((pred == 0) & (labels_arr == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    rows.append(
      {
        "threshold": float(thr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / max(1e-8, prec + rec), 4),
      }
    )
  return rows
