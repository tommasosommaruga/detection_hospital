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

  def to_dict(self) -> dict[str, Any]:
    return {"tile": self.tile.to_dict(), "score": self.score}


@dataclass(frozen=True)
class RegionOfInterest:
  """A suspicious region surfaced for pathologist review, ranked by score."""

  rank: int
  tile: Tile
  score: float

  def to_dict(self) -> dict[str, Any]:
    return {"rank": self.rank, "tile": self.tile.to_dict(), "score": self.score}


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

  def to_dict(self) -> dict[str, Any]:
    payload = asdict(self)
    payload["regions_of_interest"] = [roi.to_dict() for roi in self.regions_of_interest]
    return payload
