"""Tests for dataset–model correlation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pathassist.correlation import (
  SampleAnalysis,
  build_correlation_report,
  pearson,
  stain_similarity_to_reference,
)
from pathassist.dataset_similarity import image_fingerprint, summarize_tiles
from pathassist.synthetic import make_synthetic_slide


def test_pearson_perfect_positive():
  assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


def test_pearson_insufficient_data():
  assert pearson([1.0], [2.0]) is None


def test_build_correlation_report_structure():
  samples = [
    SampleAnalysis(
      dataset_id="patchcamelyon",
      case_id="a",
      label=1,
      stain_similarity=0.99,
      catalog_similarity=1.0,
      model_score=0.8,
      grad_cam_max=0.7,
      grad_cam_entropy=0.5,
      late_layer_activation=0.4,
      score_error=0.2,
    ),
    SampleAnalysis(
      dataset_id="bach",
      case_id="b",
      label=0,
      stain_similarity=0.6,
      catalog_similarity=0.5,
      model_score=0.3,
      grad_cam_max=0.5,
      grad_cam_entropy=0.6,
      late_layer_activation=0.3,
      score_error=0.3,
    ),
  ]
  report = build_correlation_report(samples)
  assert report["sample_count"] == 2
  assert "correlations" in report
  assert len(report["interpretation"]) >= 1


def test_stain_similarity_to_reference():
  tiles = [make_synthetic_slide(height=96, width=96, seed=i) for i in range(2)]
  ref = summarize_tiles(tiles)
  sim = stain_similarity_to_reference(tiles[0], ref)
  assert 0.0 < sim <= 1.0
