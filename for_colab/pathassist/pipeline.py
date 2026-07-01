"""Pipeline orchestration.

Wires the stages together in the order from the project diagram:

    tile -> score -> triage -> explain -> report -> record (await human approval)

Each stage stays independently testable; this module just connects them and
guarantees that every run leaves an audit trail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .audit import AuditStore
from .explain import render_heatmap
from .report import draft_report
from .scoring import Scorer
from .tiling import tile_image
from .triage import triage_case
from .types import CaseResult


def analyze_case(
  case_id: str,
  image: np.ndarray,
  scorer: Scorer,
  config: dict[str, Any],
  output_dir: str | Path,
  audit_store: AuditStore | None = None,
) -> tuple[CaseResult, Path, Path]:
  """Run the full pipeline for one case.

  Returns the machine result plus the paths to the heatmap and the report draft.
  Nothing here decides anything clinically - it produces artifacts for a human.
  """
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  tiling_cfg = config["tiling"]
  tiles = tile_image(
    image,
    tile_size=int(tiling_cfg["tile_size"]),
    overlap=float(tiling_cfg["overlap"]),
    background_intensity_cutoff=int(tiling_cfg["background_intensity_cutoff"]),
  )

  tile_scores = scorer.score_tiles(tiles)

  case = triage_case(
    case_id=case_id,
    tile_scores=tile_scores,
    model_name=scorer.name,
    model_version=scorer.version,
    config=config,
  )

  heatmap_path = render_heatmap(
    image, tile_scores, output_dir / f"{case_id}_heatmap.png"
  )

  report_text = draft_report(case, config)
  report_path = output_dir / f"{case_id}_report.txt"
  report_path.write_text(report_text, encoding="utf-8")

  if audit_store is not None:
    audit_store.record_result(case, config_used=config)

  return case, heatmap_path, report_path
