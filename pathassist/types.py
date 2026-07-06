"""Shared data structures passed between pipeline stages.

These are deliberately plain dataclasses so they serialise cleanly into the audit
log and are easy to inspect in a notebook or a test.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class Tile:
  """A single square region cut out of a slide."""

  index: int
  x: int
  y: int
  width: int
  height: int

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class TileScore:
  """A model's malignancy score for one tile, in the range [0, 1]."""

  tile: Tile
  score: float
  uncertainty: float = 0.0
  disagreement: float = 0.0

  def to_dict(self) -> dict[str, Any]:
    return {
      "tile": self.tile.to_dict(),
      "score": self.score,
      "uncertainty": self.uncertainty,
      "disagreement": self.disagreement,
    }


@dataclass(frozen=True)
class RegionOfInterest:
  """A suspicious region surfaced for pathologist review, ranked by score."""

  rank: int
  tile: Tile
  score: float
  uncertainty: float = 0.0
  review_reason: str = ""

  def to_dict(self) -> dict[str, Any]:
    return {
      "rank": self.rank,
      "tile": self.tile.to_dict(),
      "score": self.score,
      "uncertainty": self.uncertainty,
      "review_reason": self.review_reason,
    }


@dataclass
class QCReport:
  """Quality-control summary for one slide."""

  passed: bool
  flags: list[str] = field(default_factory=list)
  tissue_coverage: float = 0.0
  blur_score: float = 0.0

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass
class GradeEstimate:
  """Coarse malignancy grade estimate — advisory only, not a clinical grade."""

  grade: str
  confidence: float
  rationale: str = ""

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass
class CaseResult:
  """The full machine output for one slide/case, before human review."""

  case_id: str
  model_name: str
  model_version: str
  tile_count: int
  suspicious_tile_count: int
  case_score: float
  priority: str
  regions_of_interest: list[RegionOfInterest] = field(default_factory=list)
  mean_uncertainty: float = 0.0
  max_uncertainty: float = 0.0
  mean_disagreement: float = 0.0
  grade: GradeEstimate | None = None
  qc: QCReport | None = None
  review_flags: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    payload = asdict(self)
    payload["regions_of_interest"] = [roi.to_dict() for roi in self.regions_of_interest]
    payload["grade"] = self.grade.to_dict() if self.grade else None
    payload["qc"] = self.qc.to_dict() if self.qc else None
    return payload
