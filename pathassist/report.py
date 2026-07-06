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
  organ_ctx = config.get("organ") or {}
  lines.append("=" * 68)
  lines.append(f"CASE {case.case_id}  |  PRIORITY: {case.priority}")
  lines.append("=" * 68)
  lines.append("")
  if organ_ctx.get("organ_name"):
    lines.append(f"Organ / specialty:     {organ_ctx['organ_name']}")
    if organ_ctx.get("organ_task"):
      lines.append(f"Detection task:        {organ_ctx['organ_task']}")
    if organ_ctx.get("metadata_mismatch"):
      lines.append(
        "WARNING:               Image metadata suggests a different organ — verify selection."
      )
    lines.append("")
  lines.append(f"Model:                 {case.model_name} v{case.model_version}")
  lines.append(f"Tiles analysed:        {case.tile_count}")
  lines.append(f"Suspicious tiles:      {case.suspicious_tile_count}")
  lines.append(f"Case malignancy score: {case.case_score:.2f}")
  lines.append(f"Mean uncertainty:      {case.mean_uncertainty:.2f}")
  lines.append(f"Max uncertainty:       {case.max_uncertainty:.2f}")
  if case.mean_disagreement > 0:
    lines.append(f"Mean ensemble disagreement: {case.mean_disagreement:.2f}")

  if case.qc is not None:
    lines.append("")
    lines.append("Quality control:")
    status = "PASS" if case.qc.passed else "REVIEW REQUIRED"
    lines.append(f"  Status:          {status}")
    lines.append(f"  Tissue coverage: {case.qc.tissue_coverage:.1%}")
    lines.append(f"  Blur score:      {case.qc.blur_score:.1f}")
    if case.qc.flags:
      lines.append(f"  Flags:           {', '.join(case.qc.flags)}")

  if case.grade is not None:
    lines.append("")
    lines.append("Severity estimate (advisory — not a clinical grade):")
    lines.append(
      f"  Grade: {case.grade.grade}  (confidence {case.grade.confidence:.2f})"
    )
    lines.append(f"  Rationale: {case.grade.rationale}")

  if case.review_flags:
    lines.append("")
    lines.append("Review flags:")
    for flag in case.review_flags:
      lines.append(f"  - {flag}")

  lines.append("")
  if case.regions_of_interest:
    lines.append("Suggested review regions (highest priority first):")
    for roi in case.regions_of_interest:
      tile = roi.tile
      reason = f"  [{roi.review_reason}]" if roi.review_reason else ""
      lines.append(
        f"  {roi.rank:>2}. score {roi.score:.2f}  uncertainty {roi.uncertainty:.2f}  "
        f"at x={tile.x}, y={tile.y} ({tile.width}x{tile.height} px){reason}"
      )
  else:
    lines.append("No regions crossed the suspicious threshold.")

  lines.append("")
  lines.append("Recommended pathologist actions:")
  if "QC_FAILED" in case.review_flags:
    lines.append("  1. Re-scan or reject slide quality before trusting AI output.")
  if "HIGH_UNCERTAINTY" in case.review_flags or "ENSEMBLE_DISAGREEMENT" in case.review_flags:
    lines.append("  2. Manually review uncertain / disputed regions first.")
  if case.priority in {"URGENT", "HIGH"}:
    lines.append("  3. Prioritize this case in the worklist.")
  lines.append("  4. Approve, modify, or reject this draft before sign-out.")

  lines.append("")
  lines.append("-" * 68)
  lines.append(disclaimer)
  lines.append("-" * 68)
  lines.append("")
  lines.append("Pathologist decision: [ ] Approve   [ ] Modify   [ ] Reject")
  lines.append("Reviewed by: ______________________   Date: ______________")
  return "\n".join(lines)
