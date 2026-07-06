"""Slide-level quality control checks.

Flags slides that are too blurry, mostly background, or have unusual staining so
the pathologist knows when to distrust the model output.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import QCReport


def _laplacian_variance(gray: np.ndarray) -> float:
  """Simple focus metric — lower values usually mean blurrier images."""
  kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
  padded = np.pad(gray, 1, mode="edge")
  lap = (
    kernel[0, 0] * padded[:-2, :-2]
    + kernel[0, 1] * padded[:-2, 1:-1]
    + kernel[0, 2] * padded[:-2, 2:]
    + kernel[1, 0] * padded[1:-1, :-2]
    + kernel[1, 1] * padded[1:-1, 1:-1]
    + kernel[1, 2] * padded[1:-1, 2:]
    + kernel[2, 0] * padded[2:, :-2]
    + kernel[2, 1] * padded[2:, 1:-1]
    + kernel[2, 2] * padded[2:, 2:]
  )
  return float(lap.var())


def assess_slide_quality(
  image: np.ndarray,
  tile_count: int,
  config: dict[str, Any],
) -> QCReport:
  """Run lightweight QC checks on a slide thumbnail or downsampled image."""
  qc_cfg = config.get("qc", {})
  flags: list[str] = []

  gray = image.mean(axis=2).astype(np.float32)
  tissue_mask = gray < float(qc_cfg.get("background_intensity_cutoff", 220))
  tissue_coverage = float(tissue_mask.mean())

  if tissue_coverage < float(qc_cfg.get("min_tissue_coverage", 0.05)):
    flags.append("LOW_TISSUE_COVERAGE")

  # Downsample for speed on large images.
  step = max(1, min(image.shape[0], image.shape[1]) // 256)
  small = gray[::step, ::step]
  blur_score = _laplacian_variance(small)
  if blur_score < float(qc_cfg.get("min_blur_score", 8.0)):
    flags.append("BLURRY_SLIDE")

  mean_rgb = image.reshape(-1, 3).mean(axis=0)
  stain_range = float(mean_rgb.max() - mean_rgb.min())
  if stain_range < float(qc_cfg.get("min_stain_range", 8.0)):
    flags.append("UNUSUAL_STAINING")

  if tile_count == 0:
    flags.append("NO_SCORABLE_TILES")

  return QCReport(
    passed=len(flags) == 0,
    flags=flags,
    tissue_coverage=tissue_coverage,
    blur_score=blur_score,
  )
