"""Audit trail and human-in-the-loop review.

This is the backbone that makes the system deployable rather than just a demo.
Every automated result is persisted with the model name/version, the config used
and a timestamp; every pathologist decision (approve / modify / reject) is
recorded against that same run. Three payoffs fall out of this for free:

  * Traceability - a regulator or validation study can ask "what did the model
    say, with which version, on this exact case?" and get an answer.
  * Second-reader / QC (idea #3) - disagreements between model and pathologist
    are just records where the decision is 'reject' or 'modify'.
  * Continuous learning (idea #10) - the stored corrections become the next
    training set.

Records are newline-delimited JSON so they are trivial to append to, grep, and
load into a notebook. No patient identifiers are stored here - only a case id
that the calling system controls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import CaseResult


def _utc_now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


@dataclass
class ReviewDecision:
  """A pathologist's action on a machine result."""

  case_id: str
  decision: str  # one of: approve, modify, reject
  reviewer: str
  note: str = ""
  timestamp: str = ""

  VALID_DECISIONS = ("approve", "modify", "reject")

  def __post_init__(self) -> None:
    if self.decision not in self.VALID_DECISIONS:
      raise ValueError(
        f"decision must be one of {self.VALID_DECISIONS}, got {self.decision!r}"
      )
    if not self.timestamp:
      self.timestamp = _utc_now_iso()


class AuditStore:
  """Append-only JSON-lines store for machine results and human decisions."""

  def __init__(self, store_dir: str | Path) -> None:
    self.store_dir = Path(store_dir)
    self.store_dir.mkdir(parents=True, exist_ok=True)
    self.results_path = self.store_dir / "results.jsonl"
    self.decisions_path = self.store_dir / "decisions.jsonl"

  def _append(self, path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps(record, sort_keys=True) + "\n")

  def record_result(self, case: CaseResult, config_used: dict[str, Any]) -> dict[str, Any]:
    """Persist a machine result together with the config that produced it."""
    record = {
      "type": "result",
      "recorded_at": _utc_now_iso(),
      "config_used": config_used,
      "result": case.to_dict(),
    }
    self._append(self.results_path, record)
    return record

  def record_decision(self, decision: ReviewDecision) -> dict[str, Any]:
    """Persist a pathologist's review decision."""
    record = {"type": "decision", **asdict(decision)}
    self._append(self.decisions_path, record)
    return record

  def load_results(self) -> list[dict[str, Any]]:
    return self._load(self.results_path)

  def load_decisions(self) -> list[dict[str, Any]]:
    return self._load(self.decisions_path)

  def _load(self, path: Path) -> list[dict[str, Any]]:
    if not path.exists():
      return []
    with path.open("r", encoding="utf-8") as handle:
      return [json.loads(line) for line in handle if line.strip()]

  def disagreements(self) -> list[dict[str, Any]]:
    """Return decisions where the pathologist did not approve the machine result.

    These are exactly the cases a QC / second-reader dashboard should surface,
    and the seed corpus for retraining.
    """
    return [d for d in self.load_decisions() if d.get("decision") in ("modify", "reject")]
