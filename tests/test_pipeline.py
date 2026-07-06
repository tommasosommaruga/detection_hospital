"""Smoke tests for the pipeline stages.

These check that the workflow holds together end to end and that the audit trail
is written - not that any clinical claim is correct (there is no real model yet).
"""

from __future__ import annotations

from pathassist.audit import AuditStore, ReviewDecision
from pathassist.config import load_config
from pathassist.grading import estimate_grade
from pathassist.pipeline import analyze_case
from pathassist.qc import assess_slide_quality
from pathassist.scoring import DummyScorer, Scorer
from pathassist.synthetic import make_synthetic_slide
from pathassist.tiling import tile_image
from pathassist.triage import rank_worklist, triage_case


def test_dummy_scorer_satisfies_protocol():
  assert isinstance(DummyScorer(), Scorer)


def test_tiling_skips_background():
  image = make_synthetic_slide(height=512, width=512, seed=1)
  tiles = tile_image(image, tile_size=128, overlap=0.0, background_intensity_cutoff=220)
  assert 0 < len(tiles) <= 16
  for tile, pixels in tiles:
    assert pixels.shape == (128, 128, 3)
    assert tile.width == 128 and tile.height == 128


def test_tiling_force_whole_image_for_pcam_patch():
  image = make_synthetic_slide(height=96, width=96, seed=9)
  without = tile_image(image, tile_size=96, overlap=0.0, background_intensity_cutoff=220)
  with_force = tile_image(
    image, tile_size=96, overlap=0.0, background_intensity_cutoff=220, force_whole_image=True
  )
  assert len(with_force) == 1
  assert with_force[0][1].shape == (96, 96, 3)


def test_end_to_end_produces_artifacts_and_audit(tmp_path):
  config = load_config()
  config["audit"]["store_dir"] = str(tmp_path / "runs")
  image = make_synthetic_slide(seed=2)
  store = AuditStore(config["audit"]["store_dir"])

  case, heatmap_path, report_path, uncertainty_path, _, _ = analyze_case(
    case_id="TEST-1",
    image=image,
    scorer=DummyScorer(seed=2),
    config=config,
    output_dir=tmp_path / "outputs",
    audit_store=store,
  )

  assert heatmap_path.exists()
  assert report_path.exists()
  report_text = report_path.read_text()
  assert "PATHOLOGY ASSISTANT DRAFT" in report_text
  assert "Quality control:" in report_text
  assert "Severity estimate" in report_text
  assert 0.0 <= case.case_score <= 1.0
  assert case.priority in {"URGENT", "HIGH", "ROUTINE", "REVIEW_QC"}
  assert case.grade is not None
  assert case.qc is not None

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


def test_export_corrections(tmp_path):
  config = load_config()
  config["audit"]["store_dir"] = str(tmp_path / "runs")
  store = AuditStore(config["audit"]["store_dir"])
  image = make_synthetic_slide(seed=4)

  analyze_case(
    case_id="C-EXPORT",
    image=image,
    scorer=DummyScorer(seed=4),
    config=config,
    output_dir=tmp_path / "outputs",
    audit_store=store,
  )
  store.record_decision(
    ReviewDecision("C-EXPORT", "modify", "Dr. C", note="false positive region")
  )

  out = store.export_corrections(tmp_path / "corrections.csv")
  text = out.read_text()
  assert "C-EXPORT" in text
  assert "modify" in text
  assert "Dr. C" in text


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
  """A model 'trained' here must save CPU-loadable and score tiles via TorchScorer."""
  import pytest

  pytest.importorskip("torch")

  from pathassist.scoring import Scorer, TorchScorer
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


def test_ensemble_checkpoint_roundtrip(tmp_path):
  import pytest

  torch = pytest.importorskip("torch")
  from pathassist.backbone import build_model
  from pathassist.ensemble import ENSEMBLE_CHECKPOINT_FORMAT, load_ensemble_checkpoint
  from pathassist.scoring import EnsembleScorer, Scorer

  members = []
  for idx, seed in enumerate([11, 12]):
    cfg = dict(
      backbone="custom",
      conv_channels=[8, 16],
      kernel_size=3,
      use_batch_norm=True,
      dropout=0.0,
      head_hidden=8,
      pool="max",
    )
    model = build_model(cfg)
    members.append(
      dict(
        name=f"m{idx}",
        model_config=cfg,
        val_acc=0.5,
        weight=1.0,
        seed=seed,
        state_dict={k: v.cpu() for k, v in model.state_dict().items()},
      )
    )

  ckpt_path = tmp_path / "ensemble.pt"
  torch.save(
    {
      "checkpoint_format": ENSEMBLE_CHECKPOINT_FORMAT,
      "tile_size": 64,
      "vote_mode": "soft",
      "metrics": {"test_acc": 0.91},
      "members": members,
    },
    ckpt_path,
  )

  scorer = EnsembleScorer(str(ckpt_path), device_preference="cpu")
  assert isinstance(scorer, Scorer)
  assert scorer.name == "ensemble-voter"

  image = make_synthetic_slide(height=256, width=256, seed=5)
  tiles = tile_image(image, tile_size=64, overlap=0.0, background_intensity_cutoff=220)
  scores = scorer.score_tiles(tiles)
  assert len(scores) == len(tiles)
  for tile_score in scores:
    assert 0.0 <= tile_score.score <= 1.0
    assert tile_score.disagreement >= 0.0

  ensemble, metadata = load_ensemble_checkpoint(ckpt_path, device="cpu")
  assert ensemble.n_models == 2
  assert metadata["checkpoint_format"] == ENSEMBLE_CHECKPOINT_FORMAT


def test_qc_and_grading_helpers():
  config = load_config()
  image = make_synthetic_slide(seed=6)
  tiles = tile_image(image, 128, 0.0, 220)
  scores = DummyScorer(seed=6).score_tiles(tiles)
  qc = assess_slide_quality(image, tile_count=len(tiles), config=config)
  assert 0.0 <= qc.tissue_coverage <= 1.0

  case = triage_case("QC-1", scores, "dummy", "0", config, qc=qc)
  grade = estimate_grade(scores, case.case_score, config)
  assert grade.grade in {"BENIGN_LIKELY", "LOW", "MODERATE", "HIGH"}


def test_worklist_ranking_orders_by_score():
  config = load_config()
  cases = []
  for idx, seed in enumerate([1, 2, 3]):
    image = make_synthetic_slide(height=512, width=512, seed=seed)
    from pathassist.tiling import tile_image as _tile

    tiles = _tile(image, 128, 0.0, 220)
    scores = DummyScorer(seed=seed).score_tiles(tiles)
    cases.append(triage_case(f"C{idx}", scores, "dummy", "0", config))

  ranked = rank_worklist(cases)
  scores = [c.case_score for c in ranked]
  assert scores == sorted(scores, reverse=True)
