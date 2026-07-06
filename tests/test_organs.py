"""Tests for multi-organ registry and metadata detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathassist.organs import (
  detect_organ_from_filename,
  detect_organ_from_sidecar,
  load_organ_registry,
  match_organ_id,
  normalize_organ_id,
  validate_organ_selection,
)


@pytest.fixture
def registry():
  root = Path(__file__).resolve().parents[1]
  return load_organ_registry(root)


def test_normalize_organ_id():
  assert normalize_organ_id("Head & Neck") == "head_and_neck"
  assert normalize_organ_id("lymph-node") == "lymph_node"


def test_match_organ_aliases(registry):
  assert match_organ_id("breast", registry) == "breast"
  assert match_organ_id("camelyon", registry) == "lymph_node"
  assert match_organ_id("colon", registry) == "gastrointestinal"


def test_detect_organ_from_filename(registry):
  assert detect_organ_from_filename("organ=breast_case01.png", registry) == "breast"
  assert detect_organ_from_filename("breast_tile_001.png", registry) == "breast"
  assert detect_organ_from_filename("random.png", registry) is None


def test_detect_organ_from_sidecar(tmp_path):
  img = tmp_path / "slide.png"
  img.write_bytes(b"png")
  sidecar = tmp_path / "slide.png.json"
  sidecar.write_text(json.dumps({"organ": "pulmonary"}), encoding="utf-8")
  assert detect_organ_from_sidecar(img) == "pulmonary"


def test_validate_organ_selection_mismatch(registry):
  metadata = {"detected_organ_id": "breast", "metadata_sources": ["filename"]}
  result = validate_organ_selection(
    "lymph_node",
    metadata,
    require_ready_model=False,
    registry=registry,
  )
  assert result["metadata_mismatch"] is True
  assert result["ok"] is True
  assert result["warnings"]


def test_validate_organ_missing_checkpoint(registry):
  metadata = {"detected_organ_id": None, "metadata_sources": []}
  result = validate_organ_selection(
    "breast",
    metadata,
    require_ready_model=True,
    registry=registry,
  )
  assert result["ok"] is False
  assert result["errors"]
  assert "breast" in result["errors"][0].lower()


def test_lymph_node_model_ready_when_checkpoint_exists(registry):
  if registry.get("lymph_node").checkpoint.is_file():
    result = validate_organ_selection(
      "lymph_node",
      {"detected_organ_id": None, "metadata_sources": []},
      registry=registry,
    )
    assert result["model_ready"] is True
    assert result["ok"] is True
