"""Unit tests for workstation pure helpers."""

from __future__ import annotations

from pathlib import Path

from demo.logic import (
  ALLOWED_IMAGE_FILES,
  build_case_payload,
  compute_validation_metrics,
  extract_triage_thresholds,
  infer_tile_size,
  merge_triage_overrides,
  pick_config_name,
  predict_label,
  prediction_correct,
  safe_case_id,
  safe_image_filename,
  uncertainty_available,
  worklist_sort_key,
)
from pathassist.config import load_config
from pathassist.grading import estimate_grade
from pathassist.qc import assess_slide_quality
from pathassist.scoring import DummyScorer
from pathassist.synthetic import make_synthetic_slide
from pathassist.tiling import tile_image
from pathassist.triage import triage_case


def test_infer_tile_size():
  assert infer_tile_size(96, 96) == 96
  assert infer_tile_size(128, 96) == 96
  assert infer_tile_size(129, 96) == 256
  assert infer_tile_size(512, 512) == 256


def test_pick_config_name():
  assert pick_config_name(96) == "pcam_test.yaml"
  assert pick_config_name(128) == "pcam_test.yaml"
  assert pick_config_name(256) == "default.yaml"


def test_predict_label_and_correctness():
  cfg = {"triage": {"detection_threshold": 0.25, "metastasis_threshold": 0.55, "min_review_score": 0.15}}
  assert predict_label(0.8, cfg) == "metastasis"
  assert predict_label(0.30, cfg) == "borderline"
  assert predict_label(0.05, cfg) == "normal"
  assert prediction_correct(0.30, 1, cfg) is True
  assert prediction_correct(0.8, 0, cfg) is False


def test_safe_case_id_rejects_traversal():
  assert safe_case_id("PCAM-NORM-00") is True
  assert safe_case_id("../etc") is False
  assert safe_case_id("foo/bar") is False
  assert safe_case_id("") is False


def test_uncertainty_available(tmp_path):
  missing = tmp_path / "missing.png"
  assert uncertainty_available(None) is False
  assert uncertainty_available(missing) is False
  missing.write_bytes(b"x")
  assert uncertainty_available(missing) is True


def test_compute_validation_metrics_empty():
  result = compute_validation_metrics([])
  assert result["ready"] is False


def test_compute_validation_metrics_confusion():
  samples = [
    {"ready": True, "label": 1, "case_score": 0.8},
    {"ready": True, "label": 0, "case_score": 0.1},
    {"ready": True, "label": 0, "case_score": 0.6},
    {"ready": True, "label": 1, "case_score": 0.3},
  ]
  cfg = {
    "triage": {
      "detection_threshold": 0.25,
      "metastasis_threshold": 0.55,
      "min_review_score": 0.15,
    },
  }
  result = compute_validation_metrics(samples, cfg)
  assert result["ready"] is True
  assert result["operating_mode"] == "recall_first"
  assert result["tp"] == 2  # 0.8 metastasis + 0.3 borderline
  assert result["tn"] == 1
  assert result["fp"] == 1
  assert result["fn"] == 0
  assert result["recall"] == 1.0
  assert result["research_0_5"]["fn"] == 1  # 0.3 missed at 0.5
  buckets = {c["confusion_cell"] for c in result["cases"]}
  assert buckets == {"tp", "tn", "fp"}


def test_merge_triage_overrides():
  base = {"triage": {"detection_threshold": 0.25, "metastasis_threshold": 0.55, "min_review_score": 0.15}}
  merged = merge_triage_overrides(base, {"detection_threshold": 0.20})
  assert merged["triage"]["detection_threshold"] == 0.20
  assert merged["triage"]["metastasis_threshold"] == 0.55


def test_compute_validation_metrics_lower_threshold_improves_recall():
  samples = [{"ready": True, "label": 1, "case_score": 0.22}]
  strict = compute_validation_metrics(samples, {
    "triage": {"detection_threshold": 0.25, "metastasis_threshold": 0.55, "min_review_score": 0.23},
  })
  loose = compute_validation_metrics(samples, {
    "triage": {"detection_threshold": 0.25, "metastasis_threshold": 0.55, "min_review_score": 0.15},
  })
  assert strict["fn"] == 1
  assert loose["recall"] == 1.0


def test_worklist_sort_key():
  cases = [
    {"priority": "ROUTINE", "case_score": 0.9},
    {"priority": "URGENT", "case_score": 0.1},
    {"priority": "HIGH", "case_score": 0.5},
  ]
  ranked = sorted(cases, key=worklist_sort_key)
  assert ranked[0]["priority"] == "URGENT"
  assert ranked[-1]["priority"] == "ROUTINE"


def test_build_case_payload(tmp_path):
  config = load_config()
  image = make_synthetic_slide(seed=7)
  tiles = tile_image(image, 128, 0.0, 220)
  scores = DummyScorer(seed=7).score_tiles(tiles)
  qc = assess_slide_quality(image, len(tiles), config)
  case = triage_case("PAYLOAD-1", scores, "dummy", "0", config, qc=qc)
  grade = estimate_grade(scores, case.case_score, config)
  case.grade = grade

  unc = tmp_path / "unc.png"
  unc.write_bytes(b"png")

  payload = build_case_payload(
    case,
    "PAYLOAD-1",
    label=0,
    tile_size=128,
    uncertainty_path=unc,
    review=None,
    analyzed_at="2026-01-01T00:00:00Z",
    config=load_config(Path(__file__).resolve().parents[1] / "config" / "pcam_test.yaml"),
    organ={
      "organ_id": "lymph_node",
      "organ_name": "Lymph Node",
      "organ_specialty": "Hematolymphoid",
      "organ_task": "Metastasis detection",
    },
  )
  assert payload["case_id"] == "PAYLOAD-1"
  assert payload["organ_id"] == "lymph_node"
  assert payload["organ_name"] == "Lymph Node"
  assert payload["predicted"] in {"normal", "metastasis", "borderline"}
  assert payload["paths"]["uncertainty"] is not None
  assert payload["correct"] in {True, False}

  payload_no_unc = build_case_payload(
    case, "PAYLOAD-1", None, 128, None, None, "2026-01-01T00:00:00Z"
  )
  assert payload_no_unc["paths"]["uncertainty"] is None
  assert payload_no_unc["correct"] is None


def test_allowed_image_files():
  assert "source.png" in ALLOWED_IMAGE_FILES
  assert "evil.exe" not in ALLOWED_IMAGE_FILES
  assert safe_image_filename("PCAM-01_explain_gradcam.png")
  assert not safe_image_filename("evil.exe")
  assert not safe_image_filename("../source.png")
