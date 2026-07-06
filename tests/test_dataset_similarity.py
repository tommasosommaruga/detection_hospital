"""Tests for dataset similarity fingerprints."""

from __future__ import annotations

import numpy as np

from pathassist.dataset_similarity import (
  CATALOG,
  catalog_as_dicts,
  compare_image_fingerprints,
  image_fingerprint,
  summarize_tiles,
)
from pathassist.synthetic import make_synthetic_slide


def test_catalog_sorted_by_similarity():
  rows = catalog_as_dicts()
  assert rows[0]["id"] == "patchcamelyon"
  scores = [r["composite_similarity"] for r in rows]
  assert scores == sorted(scores, reverse=True)


def test_camelyon_family_more_similar_than_breast():
  by_id = {d.id: d.composite_similarity for d in CATALOG}
  assert by_id["camelyon16"] > by_id["bach"]
  assert by_id["camelyon16"] > by_id["nct_crc_he"]


def test_fingerprint_identical_tiles():
  img = make_synthetic_slide(height=96, width=96, seed=3)
  cmp = compare_image_fingerprints(img, img)
  assert cmp["cosine_similarity"] == 1.0


def test_summarize_tiles():
  tiles = [make_synthetic_slide(height=96, width=96, seed=i) for i in range(3)]
  summary = summarize_tiles(tiles)
  assert "mean_r" in summary
  assert 0 <= summary["tissue_fraction"] <= 1


def test_shifted_tile_lower_similarity():
  base = make_synthetic_slide(height=96, width=96, seed=1)
  shifted = np.zeros_like(base)
  shifted[:, :, 0] = 200
  shifted[:, :, 1] = 80
  shifted[:, :, 2] = 120
  cmp = compare_image_fingerprints(base, shifted)
  assert cmp["cosine_similarity"] < 0.95
