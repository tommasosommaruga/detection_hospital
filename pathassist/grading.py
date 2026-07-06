"""Cancer severity grading (decision-support estimate).

This is a tile-score-derived proxy, not a validated ISUP/Gleason grade. It gives
the pathologist a structured severity hint to review alongside detection scores.
"""

from __future__ import annotations

from typing import Any

from .types import GradeEstimate, TileScore


def estimate_grade(
  tile_scores: list[TileScore],
  case_score: float,
  config: dict[str, Any],
) -> GradeEstimate:
  """Estimate a coarse malignancy grade from tile-level predictions."""
  grading_cfg = config.get("grading", {})
  threshold = float(config["triage"]["suspicious_tile_threshold"])
  suspicious = [ts for ts in tile_scores if ts.score >= threshold]

  if not suspicious:
    return GradeEstimate(
      grade="BENIGN_LIKELY",
      confidence=min(1.0, 1.0 - case_score),
      rationale="No tiles crossed the suspicious threshold.",
    )

  high_score_tiles = sum(1 for ts in suspicious if ts.score >= 0.75)
  suspicious_fraction = len(suspicious) / max(1, len(tile_scores))

  if case_score >= float(grading_cfg.get("high_case_score", 0.80)):
    grade = "HIGH"
    confidence = min(1.0, case_score)
    rationale = (
      f"Strong focal signal: case score {case_score:.2f}, "
      f"{high_score_tiles} highly suspicious tiles."
    )
  elif case_score >= float(grading_cfg.get("moderate_case_score", 0.55)):
    grade = "MODERATE"
    confidence = min(1.0, 0.5 + suspicious_fraction)
    rationale = (
      f"Moderate malignancy pattern: case score {case_score:.2f}, "
      f"{len(suspicious)} suspicious tiles."
    )
  else:
    grade = "LOW"
    confidence = min(1.0, 0.4 + suspicious_fraction)
    rationale = (
      f"Weak or sparse signal: case score {case_score:.2f}, "
      f"{len(suspicious)} suspicious tiles."
    )

  return GradeEstimate(grade=grade, confidence=confidence, rationale=rationale)
