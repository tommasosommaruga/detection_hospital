"""Pipeline orchestration.

Wires the stages together in the order from the project diagram:

    tile -> QC -> score -> triage -> grade -> explain -> report -> record (await approval)

Each stage stays independently testable; this module just connects them and
guarantees that every run leaves an audit trail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .audit import AuditStore
from .explain import render_heatmap, render_uncertainty_map
from .nn_explain import explain_tile
from .grading import estimate_grade
from .qc import assess_slide_quality
from .report import draft_report
from .scoring import Scorer
from .tiling import resolve_case_tiles
from .triage import triage_case, _case_score
from .types import CaseResult
from .wsi import WsiScanInfo


def analyze_case(
  case_id: str,
  image: np.ndarray | None,
  scorer: Scorer,
  config: dict[str, Any],
  output_dir: str | Path,
  audit_store: AuditStore | None = None,
  *,
  image_path: str | Path | None = None,
) -> tuple[CaseResult, Path, Path, Path | None, dict[str, Any] | None, WsiScanInfo | None]:
  """Run the full pipeline for one case.

  Returns the machine result plus paths to the score heatmap, report draft,
  optional uncertainty map, and optional NN layer explanation metadata.
  Nothing here decides anything clinically.
  """
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  display_image, tiles, wsi_info = resolve_case_tiles(image, config, image_path=image_path)

  qc = assess_slide_quality(display_image, tile_count=len(tiles), config=config)
  tile_scores = scorer.score_tiles(tiles)
  case_score = _case_score(tile_scores, config)
  grade = estimate_grade(tile_scores, case_score, config)
  case = triage_case(
    case_id=case_id,
    tile_scores=tile_scores,
    model_name=scorer.name,
    model_version=scorer.version,
    config=config,
    grade=grade,
    qc=qc,
  )

  heatmap_path = render_heatmap(
    display_image, tile_scores, output_dir / f"{case_id}_heatmap.png"
  )
  uncertainty_path = render_uncertainty_map(
    display_image,
    tile_scores,
    output_dir / f"{case_id}_uncertainty.png",
    min_uncertainty=float(config["triage"].get("uncertainty_threshold", 0.45)),
  )

  report_text = draft_report(case, config)
  report_path = output_dir / f"{case_id}_report.txt"
  report_path.write_text(report_text, encoding="utf-8")

  nn_explanation: dict[str, Any] | None = None
  explain_cfg = config.get("explainability", {})
  if explain_cfg.get("nn_layers") and tiles:
    nn_explanation = _run_nn_explanation(
      scorer, tiles[0][1], output_dir, case_id, explain_cfg
    )

  if audit_store is not None:
    audit_store.record_result(case, config_used=config)

  return case, heatmap_path, report_path, uncertainty_path, nn_explanation, wsi_info


def _run_nn_explanation(
  scorer: Scorer,
  pixel_array: np.ndarray,
  output_dir: Path,
  case_id: str,
  explain_cfg: dict[str, Any],
) -> dict[str, Any] | None:
  """Grad-CAM + per-layer activations for the first scorable tile."""
  member_index = int(explain_cfg.get("member_index", 0))
  if not hasattr(scorer, "_ensemble"):
    return None
  members = scorer._ensemble.members  # type: ignore[attr-defined]
  if not members or member_index >= len(members):
    return None
  member = members[member_index]
  device = str(getattr(scorer, "device", "cpu"))
  return explain_tile(
    member["model"],
    pixel_array,
    member["model_config"],
    device,
    output_dir=output_dir,
    prefix=f"{case_id}_explain",
  )