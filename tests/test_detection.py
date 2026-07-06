"""Tests for recall-first detection thresholds."""

from __future__ import annotations

from pathassist.detection import (
  aggregate_case_score,
  classify_case,
  detection_threshold,
  prediction_correct,
  sweep_thresholds,
)
from pathassist.types import Tile, TileScore


def _cfg(thr=0.25, review=0.15, mode="max"):
  return {
    "triage": {
      "detection_threshold": thr,
      "min_review_score": review,
      "case_score_mode": mode,
      "suspicious_tile_threshold": 0.5,
    }
  }


def test_classify_metastasis_normal_borderline():
  cfg = _cfg(thr=0.25, review=0.15)
  cfg["triage"]["metastasis_threshold"] = 0.55
  assert classify_case(0.8, cfg)["predicted"] == "metastasis"
  assert classify_case(0.05, cfg)["predicted"] == "normal"
  assert classify_case(0.30, cfg)["predicted"] == "borderline"
  assert classify_case(0.54, cfg)["predicted"] == "borderline"


def test_prediction_correct_counts_borderline_as_positive_for_recall():
  cfg = _cfg(thr=0.25, review=0.15)
  assert prediction_correct(0.20, 1, cfg) is True
  assert prediction_correct(0.20, 0, cfg) is False


def test_aggregate_case_score_max_mode():
  tiles = [
    TileScore(tile=Tile(0, 0, 0, 96, 96), score=0.2),
    TileScore(tile=Tile(1, 96, 0, 96, 96), score=0.9),
  ]
  assert aggregate_case_score(tiles, _cfg(mode="max")) == 0.9
  assert aggregate_case_score(tiles, _cfg(mode="top_k_mean")) < 0.9


def test_sweep_thresholds_finds_zero_fn():
  scores = [0.9, 0.28, 0.1, 0.54]
  labels = [1, 1, 0, 0]
  rows = sweep_thresholds(scores, labels, thresholds=[0.25, 0.30, 0.50])
  at_25 = next(r for r in rows if r["threshold"] == 0.25)
  assert at_25["fn"] == 0
  assert at_25["recall"] == 1.0


def test_detection_threshold_from_config():
  assert detection_threshold(_cfg(thr=0.31)) == 0.31
