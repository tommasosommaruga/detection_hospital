"""Case triage.

This is the stage that turns per-tile scores into the thing a hospital actually
wants: a priority for the case and a short, ranked list of regions to look at
first. It is pure functions over the scores, with all thresholds coming from
config - so tuning the triage policy never means editing model or IO code.
"""

from __future__ import annotations

from typing import Any

from .detection import aggregate_case_score, classify_case
from .types import CaseResult, GradeEstimate, QCReport, RegionOfInterest, TileScore


def _case_score(tile_scores: list[TileScore], config: dict[str, Any] | None = None) -> float:
  """Aggregate tile scores into a single case-level malignancy score."""
  if config is not None:
    return aggregate_case_score(tile_scores, config)
  if not tile_scores:
    return 0.0
  ordered = sorted((ts.score for ts in tile_scores), reverse=True)
  top_k = max(1, min(len(ordered), 10))
  return sum(ordered[:top_k]) / top_k


def _priority_band(case_score: float, bands: list[dict[str, Any]]) -> str:
  """Map a case score to a named priority band (first matching band wins)."""
  for band in bands:
    if case_score >= float(band["min_score"]):
      return str(band["name"])
  return "ROUTINE"


def _review_reason(tile_score: TileScore, config: dict[str, Any]) -> str:
  triage_cfg = config.get("triage", {})
  reasons: list[str] = []
  if tile_score.score >= float(triage_cfg.get("suspicious_tile_threshold", 0.5)):
    reasons.append("high_score")
  if tile_score.uncertainty >= float(triage_cfg.get("uncertainty_threshold", 0.45)):
    reasons.append("uncertain")
  if tile_score.disagreement >= float(triage_cfg.get("disagreement_threshold", 0.25)):
    reasons.append("ensemble_disagreement")
  return ", ".join(reasons)


def _build_review_flags(
  tile_scores: list[TileScore],
  qc: QCReport | None,
  config: dict[str, Any],
) -> list[str]:
  triage_cfg = config.get("triage", {})
  flags: list[str] = []

  if qc and not qc.passed:
    flags.append("QC_FAILED")

  if not tile_scores:
    flags.append("NO_SCORABLE_TILES")
    return flags

  uncertainties = [ts.uncertainty for ts in tile_scores]
  disagreements = [ts.disagreement for ts in tile_scores]
  if max(uncertainties) >= float(triage_cfg.get("uncertainty_threshold", 0.45)):
    flags.append("HIGH_UNCERTAINTY")
  if max(disagreements) >= float(triage_cfg.get("disagreement_threshold", 0.25)):
    flags.append("ENSEMBLE_DISAGREEMENT")

  borderline = [
    ts for ts in tile_scores
    if abs(ts.score - 0.5) <= float(triage_cfg.get("borderline_margin", 0.08))
  ]
  if borderline:
    flags.append("BORDERLINE_TILES")

  return flags


def triage_case(
  case_id: str,
  tile_scores: list[TileScore],
  model_name: str,
  model_version: str,
  config: dict[str, Any],
  grade: GradeEstimate | None = None,
  qc: QCReport | None = None,
) -> CaseResult:
  """Produce a ranked, prioritised result for one case."""
  triage_cfg = config["triage"]
  threshold = float(triage_cfg["suspicious_tile_threshold"])
  max_regions = int(triage_cfg["max_review_regions"])
  bands = list(triage_cfg["priority_bands"])

  suspicious = [ts for ts in tile_scores if ts.score >= threshold]
  uncertain = [
    ts for ts in tile_scores
    if ts.uncertainty >= float(triage_cfg.get("uncertainty_threshold", 0.45))
  ]
  review_candidates = {id(ts.tile): ts for ts in suspicious + uncertain}
  ranked_tiles = sorted(
    review_candidates.values(),
    key=lambda ts: (ts.score, ts.uncertainty),
    reverse=True,
  )

  regions = [
    RegionOfInterest(
      rank=rank,
      tile=ts.tile,
      score=ts.score,
      uncertainty=ts.uncertainty,
      review_reason=_review_reason(ts, config),
    )
    for rank, ts in enumerate(ranked_tiles[:max_regions], start=1)
  ]

  case_score = _case_score(tile_scores, config)
  review_flags = _build_review_flags(tile_scores, qc, config)
  detection = classify_case(case_score, config)
  if detection["predicted"] == "borderline":
    review_flags = list(review_flags) + ["POSSIBLE_METASTASIS"]
  if detection["needs_review"] and case_score < float(triage_cfg["suspicious_tile_threshold"]):
    review_flags = list(review_flags) + ["RECALL_FIRST_REVIEW"]

  priority = _priority_band(case_score, bands)
  if detection["predicted"] in {"metastasis", "borderline"} and priority == "ROUTINE":
    priority = "HIGH"
  if "RECALL_FIRST_REVIEW" in review_flags and priority == "ROUTINE":
    priority = "HIGH"
  if "QC_FAILED" in review_flags and priority != "URGENT":
    priority = "REVIEW_QC"

  uncertainties = [ts.uncertainty for ts in tile_scores] or [0.0]
  disagreements = [ts.disagreement for ts in tile_scores] or [0.0]

  return CaseResult(
    case_id=case_id,
    model_name=model_name,
    model_version=model_version,
    tile_count=len(tile_scores),
    suspicious_tile_count=len(suspicious),
    case_score=case_score,
    priority=priority,
    regions_of_interest=regions,
    mean_uncertainty=float(sum(uncertainties) / len(uncertainties)),
    max_uncertainty=max(uncertainties),
    mean_disagreement=float(sum(disagreements) / len(disagreements)),
    grade=grade,
    qc=qc,
    review_flags=review_flags,
  )


def rank_worklist(cases: list[CaseResult]) -> list[CaseResult]:
  """Order a batch of cases so the most urgent surface at the top of the queue."""
  priority_rank = {"URGENT": 0, "REVIEW_QC": 1, "HIGH": 2, "ROUTINE": 3}
  return sorted(
    cases,
    key=lambda case: (
      priority_rank.get(case.priority, 99),
      -case.case_score,
      -case.max_uncertainty,
    ),
  )
