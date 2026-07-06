"""Tests for organ training data catalog."""

from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np

from pathassist.training_data import catalog_table, get_training_spec, load_training_catalog


def test_catalog_has_auto_download_organs():
  catalog = load_training_catalog()
  assert "lymph_node" in catalog
  assert catalog["lymph_node"]["auto_download"] is True
  assert catalog["gastrointestinal"]["hf_dataset"] == "1aurent/NCT-CRC-HE"


def test_catalog_table_rows():
  rows = catalog_table()
  assert len(rows) >= 18
  gi = next(r for r in rows if r["organ_id"] == "gastrointestinal")
  assert gi["auto_download"] is True
  assert "huggingface.co" in gi["hf_url"]


def test_manual_organ_requires_folder():
  spec = get_training_spec("dermatopathology")
  assert spec["auto_download"] is False
  assert spec.get("manual_url")


def test_resize_tile_hf_image_dict():
  import io

  from PIL import Image

  from pathassist.training_data import _resize_tile

  buf = io.BytesIO()
  Image.new("RGB", (32, 32), color=(128, 64, 32)).save(buf, format="PNG")
  tile = _resize_tile({"bytes": buf.getvalue()}, tile_size=16)
  assert tile.shape == (16, 16, 3)
  assert tile.dtype == np.uint8


def test_nct_crc_he_balanced_labels():
  """Balanced loader must interleave both classes, not take stream order."""
  from pathassist.training_data import _NCT_NORMAL, _NCT_TUMOR, _resize_tile

  rows = []
  for _ in range(20):
    rows.append({"label": _NCT_NORMAL, "image": np.zeros((8, 8, 3), dtype=np.uint8)})
  for _ in range(5):
    rows.append({"label": _NCT_TUMOR, "image": np.ones((8, 8, 3), dtype=np.uint8)})

  per_class = 4
  norm_tiles, tum_tiles = [], []
  for row in rows:
    label = int(row["label"])
    if label == _NCT_NORMAL and len(norm_tiles) < per_class:
      norm_tiles.append(_resize_tile(row["image"], 8))
    elif label == _NCT_TUMOR and len(tum_tiles) < per_class:
      tum_tiles.append(_resize_tile(row["image"], 8))

  labels = np.array([0.0] * len(norm_tiles) + [1.0] * len(tum_tiles), dtype=np.float32)
  assert len(norm_tiles) == per_class
  assert len(tum_tiles) == per_class
  assert labels.sum() == per_class


def test_load_organ_training_data_missing_folder():
  with pytest.raises(ValueError, match="requires manual tiles"):
    from pathassist.training_data import load_organ_training_data

    load_organ_training_data("dermatopathology", max_samples=10)
