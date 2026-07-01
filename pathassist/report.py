"""Report drafting.

Turns a CaseResult into a human-readable draft the pathologist edits and signs
off. Every draft is explicitly labelled as decision support, never a diagnosis.
"""

from __future__ import annotations

from typing import Any

from .types import CaseResult


def draft_report(case: CaseResult, config: dict[str, Any]) -> str:
  """Render a plain-text report draft for one case."""
  disclaimer = str(config["report"]["disclaimer"]).strip()

  lines: list[str] = []
  lines.append("=" * 68)
  lines.append(f"CASE {case.case_id}  |  PRIORITY: {case.priority}")
  lines.append("=" * 68)
  lines.append("")
  lines.append(f"Model:                 {case.model_name} v{case.model_version}")
  lines.append(f"Tiles analysed:        {case.tile_count}")
  lines.append(f"Suspicious tiles:      {case.suspicious_tile_count}")
  lines.append(f"Case malignancy score: {case.case_score:.2f}")
  lines.append("")

  if case.regions_of_interest:
    lines.append("Suggested review regions (highest score first):")
    for roi in case.regions_of_interest:
      tile = roi.tile
      lines.append(
        f"  {roi.rank:>2}. score {roi.score:.2f}  "
        f"at x={tile.x}, y={tile.y} ({tile.width}x{tile.height} px)"
      )
  else:
    lines.append("No regions crossed the suspicious threshold.")

  lines.append("")
  lines.append("-" * 68)
  lines.append(disclaimer)
  lines.append("-" * 68)
  lines.append("")
  lines.append("Pathologist decision: [ ] Approve   [ ] Modify   [ ] Reject")
  lines.append("Reviewed by: ______________________   Date: ______________")
  return "\n".join(lines)
