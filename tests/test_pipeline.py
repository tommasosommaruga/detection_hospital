"""Smoke tests for the pipeline stages.

These check that the workflow holds together end to end and that the audit trail
is written - not that any clinical claim is correct (there is no real model yet).
"""

from __future__ import annotations

from pathassist.audit import AuditStore, ReviewDecision
from pathassist.config import load_config
from pathassist.pipeline import analyze_case
from pathassist.scoring import DummyScorer, Scorer
from pathassist.synthetic import make_synthetic_slide
from pathassist.tiling import tile_image
from pathassist.triage import rank_worklist


def test_dummy_scorer_satisfies_protocol():
  assert isinstance(DummyScorer(), Scorer)


def test_tiling_skips_background():
  image = make_synthetic_slide(height=512, width=512, seed=1)
  tiles = tile_image(image, tile_size=128, overlap=0.0, background_intensity_cutoff=220)
  # Some tissue tiles should survive, but not the full grid (background dropped).
  assert 0 < len(tiles) <= 16
  for tile, pixels in tiles:
    assert pixels.shape == (128, 128, 3)
    assert tile.width == 128 and tile.height == 128


def test_end_to_end_produces_artifacts_and_audit(tmp_path):
  config = load_config()
  config["audit"]["store_dir"] = str(tmp_path / "runs")
  image = make_synthetic_slide(seed=2)
  store = AuditStore(config["audit"]["store_dir"])

  case, heatmap_path, report_path = analyze_case(
    case_id="TEST-1",
    image=image,
    scorer=DummyScorer(seed=2),
    config=config,
    output_dir=tmp_path / "outputs",
    audit_store=store,
  )

  assert heatmap_path.exists()
  assert report_path.exists()
  assert "DECISION SUPPORT DRAFT" in report_path.read_text()
  assert 0.0 <= case.case_score <= 1.0
  assert case.priority in {"URGENT", "HIGH", "ROUTINE"}

  results = store.load_results()
  assert len(results) == 1
  assert results[0]["result"]["case_id"] == "TEST-1"
  assert results[0]["result"]["model_name"] == "dummy-heuristic"


def test_review_decision_and_disagreements(tmp_path):
  store = AuditStore(tmp_path / "runs")
  store.record_decision(ReviewDecision("C1", "approve", "Dr. A"))
  store.record_decision(ReviewDecision("C2", "reject", "Dr. B", note="benign"))
  disagreements = store.disagreements()
  assert len(disagreements) == 1
  assert disagreements[0]["case_id"] == "C2"


def test_invalid_decision_rejected():
  try:
    ReviewDecision("C3", "maybe", "Dr. C")
  except ValueError:
    return
  raise AssertionError("expected ValueError for invalid decision")


def _write_tile(path, value):
  from PIL import Image
  import numpy as np

  Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(path)


def test_load_from_image_folder(tmp_path):
  from pathassist.data import load_from_image_folder

  (tmp_path / "benign").mkdir()
  (tmp_path / "malignant").mkdir()
  _write_tile(tmp_path / "benign" / "a.png", 230)
  _write_tile(tmp_path / "benign" / "b.png", 220)
  _write_tile(tmp_path / "malignant" / "c.png", 60)

  tiles, labels = load_from_image_folder(tmp_path, tile_size=16)
  assert tiles.shape == (3, 16, 16, 3)
  assert set(labels.tolist()) == {0.0, 1.0}
  assert labels.sum() == 1.0


def test_load_from_csv_with_named_labels(tmp_path):
  from pathassist.data import load_from_csv

  _write_tile(tmp_path / "t0.png", 200)
  _write_tile(tmp_path / "t1.png", 40)
  csv_path = tmp_path / "labels.csv"
  csv_path.write_text("path,label\nt0.png,benign\nt1.png,malignant\n", encoding="utf-8")

  tiles, labels = load_from_csv(
    csv_path, tile_size=16, positive_labels=["malignant"]
  )
  assert tiles.shape == (2, 16, 16, 3)
  assert labels.tolist() == [0.0, 1.0]


def test_torch_train_save_load_score_roundtrip(tmp_path):
  """A model 'trained' here must save CPU-loadable and score tiles via TorchScorer.

  This mirrors the real workflow (train on GPU, run on CPU): training forces CPU
  here so the test is deterministic and hardware-independent.
  """
  import pytest

  pytest.importorskip("torch")

  from pathassist.scoring import Scorer, TorchScorer
  from pathassist.tiling import tile_image
  from pathassist.train import train

  result = train(
    epochs=1,
    num_samples=60,
    tile_size=64,
    device_preference="cpu",
    out_path=str(tmp_path / "model.pt"),
    seed=3,
  )
  assert result.checkpoint_path.exists()
  assert len(result.history) == 1

  scorer = TorchScorer(str(result.checkpoint_path), device_preference="cpu")
  assert isinstance(scorer, Scorer)

  image = make_synthetic_slide(height=512, width=512, seed=3)
  tiles = tile_image(image, tile_size=128, overlap=0.0, background_intensity_cutoff=220)
  scores = scorer.score_tiles(tiles)
  assert len(scores) == len(tiles)
  for tile_score in scores:
    assert 0.0 <= tile_score.score <= 1.0


def test_worklist_ranking_orders_by_score():
  config = load_config()
  cases = []
  for idx, seed in enumerate([1, 2, 3]):
    image = make_synthetic_slide(height=512, width=512, seed=seed)
    from pathassist.tiling import tile_image as _tile
    from pathassist.triage import triage_case

    tiles = _tile(image, 128, 0.0, 220)
    scores = DummyScorer(seed=seed).score_tiles(tiles)
    cases.append(triage_case(f"C{idx}", scores, "dummy", "0", config))

  ranked = rank_worklist(cases)
  scores = [c.case_score for c in ranked]
  assert scores == sorted(scores, reverse=True)
