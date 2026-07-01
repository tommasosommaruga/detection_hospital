"""Case triage.

This is the stage that turns per-tile scores into the thing a hospital actually
wants: a priority for the case and a short, ranked list of regions to look at
first. It is pure functions over the scores, with all thresholds coming from
config - so tuning the triage policy never means editing model or IO code.
"""

from __future__ import annotations

from typing import Any

from .types import CaseResult, RegionOfInterest, TileScore


def _case_score(tile_scores: list[TileScore]) -> float:
  """Aggregate tile scores into a single case-level malignancy score.

  We use the mean of the top-k tiles rather than the global mean: a malignancy
  is usually focal, so a few strongly positive tiles should dominate a slide full
  of benign tissue. k scales with slide size but is capped for stability.
  """
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


def triage_case(
  case_id: str,
  tile_scores: list[TileScore],
  model_name: str,
  model_version: str,
  config: dict[str, Any],
) -> CaseResult:
  """Produce a ranked, prioritised result for one case."""
  triage_cfg = config["triage"]
  threshold = float(triage_cfg["suspicious_tile_threshold"])
  max_regions = int(triage_cfg["max_review_regions"])
  bands = list(triage_cfg["priority_bands"])

  suspicious = [ts for ts in tile_scores if ts.score >= threshold]
  suspicious.sort(key=lambda ts: ts.score, reverse=True)

  regions = [
    RegionOfInterest(rank=rank, tile=ts.tile, score=ts.score)
    for rank, ts in enumerate(suspicious[:max_regions], start=1)
  ]

  case_score = _case_score(tile_scores)
  return CaseResult(
    case_id=case_id,
    model_name=model_name,
    model_version=model_version,
    tile_count=len(tile_scores),
    suspicious_tile_count=len(suspicious),
    case_score=case_score,
    priority=_priority_band(case_score, bands),
    regions_of_interest=regions,
  )


def rank_worklist(cases: list[CaseResult]) -> list[CaseResult]:
  """Order a batch of cases so the most urgent surface at the top of the queue."""
  return sorted(cases, key=lambda case: case.case_score, reverse=True)
