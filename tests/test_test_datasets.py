"""Tests for holdout test-dataset import and hub catalog."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pathassist.test_datasets as td
from pathassist.test_datasets import (
  hub_catalog,
  import_test_samples,
  label_names_for_organ,
  resolve_samples_dir,
  set_active_dataset,
)
from pathassist.synthetic import make_synthetic_slide


@pytest.fixture
def repo(tmp_path):
  shutil.copytree(
    Path(__file__).resolve().parents[1] / "config",
    tmp_path / "config",
    dirs_exist_ok=True,
  )
  return tmp_path


def _write_fake_import(repo: Path, organ_id: str = "lymph_node") -> Path:
  out = td.test_datasets_root(repo) / organ_id
  images = out / "images"
  images.mkdir(parents=True)
  Image.fromarray(make_synthetic_slide(96, 96, seed=1)).save(images / "a.png")
  Image.fromarray(make_synthetic_slide(96, 96, seed=2)).save(images / "b.png")
  with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
      handle,
      fieldnames=["case_id", "image", "label", "organ", "dataset_id", "split"],
    )
    writer.writeheader()
    writer.writerow({
      "case_id": "PCAM-NOR-00",
      "image": "images/a.png",
      "label": 0,
      "organ": organ_id,
      "dataset_id": organ_id,
      "split": "test",
    })
    writer.writerow({
      "case_id": "PCAM-MET-00",
      "image": "images/b.png",
      "label": 1,
      "organ": organ_id,
      "dataset_id": organ_id,
      "split": "test",
    })
  (out / "import_meta.json").write_text(
    json.dumps({
      "organ_id": organ_id,
      "split": "test",
      "label_names": {"0": "normal", "1": "metastasis"},
      "n_samples": 2,
    }),
    encoding="utf-8",
  )
  return out


def test_hub_catalog_merges_training_and_similarity(repo):
  data = hub_catalog(repo)
  assert data["reference"] == "patchcamelyon"
  ln = next(d for d in data["datasets"] if d["organ_id"] == "lymph_node")
  assert ln["importable"] is True
  assert ln["hf_test_split"] == "test"
  assert ln["composite_similarity"] == 1.0
  gi = next(d for d in data["datasets"] if d["organ_id"] == "gastrointestinal")
  assert gi["hf_test_split"] == "CRC_VAL_HE_7K"


def test_resolve_and_activate(repo):
  legacy = repo / "outputs" / "real_pcam"
  legacy.mkdir(parents=True)
  (legacy / "manifest.csv").write_text("case_id,label,image\n", encoding="utf-8")
  assert resolve_samples_dir(repo) == legacy

  _write_fake_import(repo, "lymph_node")
  set_active_dataset("lymph_node", root=repo)
  assert resolve_samples_dir(repo) == td.test_datasets_root(repo) / "lymph_node"


def test_label_names_from_import_meta(repo):
  _write_fake_import(repo, "lymph_node")
  neg, pos = label_names_for_organ("lymph_node", repo)
  assert neg == "normal"
  assert pos == "metastasis"


def test_import_test_samples_monkeypatched(repo, monkeypatch):
  import pathassist.test_datasets as td

  tiles = [
    make_synthetic_slide(96, 96, seed=3),
    make_synthetic_slide(96, 96, seed=4),
  ]

  def fake_collect(spec, split, per_class, tile_size):
    return tiles, [0, 1]

  monkeypatch.setitem(td._COLLECTORS, "patchcamelyon", fake_collect)
  result = import_test_samples("lymph_node", count=2, root=repo, activate=True)
  assert result["sample_count"] == 2
  assert result["split"] == "test"
  assert result["cached"] is False
  manifest = td.test_datasets_root(repo) / "lymph_node" / "manifest.csv"
  assert manifest.is_file()
  assert resolve_samples_dir(repo).name == "lymph_node"


def test_import_reuses_cached_tiles(repo, monkeypatch):
  import pathassist.test_datasets as td

  calls = {"n": 0}

  def fake_collect(spec, split, per_class, tile_size):
    calls["n"] += 1
    return [make_synthetic_slide(96, 96, seed=5), make_synthetic_slide(96, 96, seed=6)], [0, 1]

  monkeypatch.setitem(td._COLLECTORS, "patchcamelyon", fake_collect)

  first = import_test_samples("lymph_node", count=2, root=repo, activate=False)
  assert first["cached"] is False
  assert calls["n"] == 1

  # Second import with same count reuses disk cache — no dataset API call.
  second = import_test_samples("lymph_node", count=2, root=repo, activate=False)
  assert second["cached"] is True
  assert second["sample_count"] == 2
  assert calls["n"] == 1

  # force=True re-downloads.
  forced = import_test_samples("lymph_node", count=2, root=repo, activate=False, force=True)
  assert forced["cached"] is False
  assert calls["n"] == 2


def test_import_local_folder(repo):
  from PIL import Image

  from pathassist.test_datasets import import_local_folder, local_organ_dir

  src = local_organ_dir("breast", repo)
  (src / "benign").mkdir(parents=True)
  (src / "malignant").mkdir(parents=True)
  Image.new("RGB", (64, 64), (200, 180, 160)).save(src / "benign" / "b0.png")
  Image.new("RGB", (64, 64), (180, 60, 60)).save(src / "malignant" / "m0.png")

  result = import_local_folder("breast", root=repo, activate=True)
  assert result["sample_count"] == 2
  assert result["split"] == "local"
  manifest = repo / "outputs" / "test_datasets" / "breast" / "manifest.csv"
  assert manifest.is_file()
  assert (repo / "outputs" / "test_datasets" / "breast" / "images").is_dir()
