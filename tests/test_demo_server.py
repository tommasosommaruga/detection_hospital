"""API tests for the PathAssist workstation (uses DummyScorer, no checkpoint)."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from demo.server import WorkstationState, create_app
from pathassist.scoring import DummyScorer
from pathassist.synthetic import make_synthetic_slide

ROOT = Path(__file__).resolve().parents[1]


def _png_bytes(array: np.ndarray | None = None) -> bytes:
  if array is None:
    array = make_synthetic_slide(height=96, width=96, seed=9)
  buf = io.BytesIO()
  Image.fromarray(array).save(buf, format="PNG")
  return buf.getvalue()


@pytest.fixture
def workstation(tmp_path):
  shutil.copytree(ROOT / "config", tmp_path / "config", dirs_exist_ok=True)
  state = WorkstationState(
    root=tmp_path,
    static_dir=tmp_path / "static",
    scorer_factory=lambda: DummyScorer(seed=1),
    checkpoint=tmp_path / "missing.pt",
  )
  state.static_dir.mkdir(parents=True)
  (state.static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")
  return state


def _upload(client, organ: str = "lymph_node", confirm: bool = False):
  data = {"organ": organ}
  if confirm:
    data["confirm_mismatch"] = "true"
  return client.post(
    "/api/analyze/upload",
    files={"file": ("tile.png", _png_bytes(), "image/png")},
    data=data,
  )


@pytest.fixture
def client(workstation):
  with TestClient(create_app(workstation)) as test_client:
    yield test_client


def test_health_without_checkpoint(client):
  resp = client.get("/api/health")
  assert resp.status_code == 200
  body = resp.json()
  assert body["checkpoint_exists"] is False
  assert body["status"] == "no_checkpoint"


def test_mission_endpoint(client):
  resp = client.get("/api/mission")
  assert resp.status_code == 200
  assert "title" in resp.json()
  assert "pipeline" in resp.json()


def test_index_serves_html(client):
  resp = client.get("/")
  assert resp.status_code == 200
  assert "ok" in resp.text


def test_upload_and_analyze(client, workstation):
  resp = _upload(client)
  assert resp.status_code == 200
  data = resp.json()
  assert data["case_id"].startswith("UPLOAD-")
  assert data["organ_id"] == "lymph_node"
  assert data["organ_name"] == "Lymph Node"
  assert "case_score" in data
  assert (workstation.results_dir / data["case_id"] / "meta.json").exists()


def test_upload_requires_organ(client):
  resp = client.post(
    "/api/analyze/upload",
    files={"file": ("tile.png", _png_bytes(), "image/png")},
  )
  assert resp.status_code == 422


def test_upload_rejects_untrained_organ_without_dummy_scorer(tmp_path):
  shutil.copytree(ROOT / "config", tmp_path / "config", dirs_exist_ok=True)
  state = WorkstationState(
    root=tmp_path,
    static_dir=tmp_path / "static",
    checkpoint=tmp_path / "missing.pt",
  )
  state.static_dir.mkdir(parents=True)
  (state.static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")
  with TestClient(create_app(state)) as c:
    resp = c.post(
      "/api/analyze/upload",
      files={"file": ("tile.png", _png_bytes(), "image/png")},
      data={"organ": "breast"},
    )
    assert resp.status_code == 400
    assert "No trained model" in resp.text


def test_upload_rejects_non_image(client):
  resp = client.post(
    "/api/analyze/upload",
    files={"file": ("notes.txt", b"hello", "text/plain")},
    data={"organ": "lymph_node"},
  )
  assert resp.status_code == 400


def test_get_result_and_report(client, workstation):
  up = _upload(client).json()
  case_id = up["case_id"]

  result = client.get(f"/api/result/{case_id}")
  assert result.status_code == 200
  assert result.json()["case_id"] == case_id

  report = client.get(f"/api/report/{case_id}")
  assert report.status_code == 200
  assert "PATHOLOGY ASSISTANT DRAFT" in report.text


def test_invalid_case_id_rejected(client):
  case_id = _upload(client).json()["case_id"]
  assert client.get(f"/api/image/{case_id}/evil.exe").status_code == 400


def test_review_flow(client, workstation):
  case_id = _upload(client).json()["case_id"]

  bad = client.post(
    "/api/review",
    json={"case_id": case_id, "decision": "maybe", "reviewer": "Dr. X"},
  )
  assert bad.status_code == 400

  ok = client.post(
    "/api/review",
    json={"case_id": case_id, "decision": "approve", "reviewer": "Dr. X", "note": ""},
  )
  assert ok.status_code == 200
  assert ok.json()["decision"] == "approve"

  refreshed = client.get(f"/api/result/{case_id}").json()
  assert refreshed["review"]["decision"] == "approve"


def test_worklist_and_cases(client):
  _upload(client)
  cases = client.get("/api/cases").json()
  assert len(cases) >= 1
  worklist = client.get("/api/worklist").json()
  assert len(worklist) >= 1
  assert "priority" in worklist[0]


def test_sample_manifest_analyze(client, workstation):
  samples_dir = workstation.root / "outputs" / "real_pcam"
  workstation.samples_dir = samples_dir
  samples_dir.mkdir(parents=True, exist_ok=True)
  img_path = samples_dir / "tile.png"
  Image.fromarray(make_synthetic_slide(96, 96, seed=3)).save(img_path)
  manifest = samples_dir / "manifest.csv"
  manifest.write_text(
    "case_id,label,image\nDEMO-01,0,tile.png\n",
    encoding="utf-8",
  )

  missing = client.post("/api/analyze/sample/UNKNOWN")
  assert missing.status_code == 404

  ok = client.post(
    "/api/analyze/sample/DEMO-01",
    data={"organ": "lymph_node"},
  )
  assert ok.status_code == 200
  assert ok.json()["label"] == 0

  samples = client.get("/api/samples").json()
  assert len(samples) == 1
  assert samples[0]["ready"] is True


def test_validation_not_ready_then_ready(client, workstation):
  empty = client.get("/api/validation").json()
  assert empty["ready"] is False

  samples_dir = workstation.root / "outputs" / "real_pcam"
  workstation.samples_dir = samples_dir
  samples_dir.mkdir(parents=True, exist_ok=True)
  (samples_dir / "manifest.csv").write_text(
    "case_id,label,image\nBENCH-1,0,tile.png\n",
    encoding="utf-8",
  )

  meta_dir = workstation.results_dir / "BENCH-1"
  meta_dir.mkdir(parents=True)
  meta = {
    "case_id": "BENCH-1",
    "ready": True,
    "correct": True,
    "predicted": "normal",
    "label": 0,
    "label_name": "normal",
    "priority": "ROUTINE",
    "case_score": 0.1,
  }
  (meta_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

  ready = client.get("/api/validation").json()
  assert ready["ready"] is True
  assert ready["tn"] == 1
  assert ready["thresholds"]["detection_threshold"] == 0.30


def test_validation_threshold_preview_and_session(client, workstation):
  samples_dir = workstation.root / "outputs" / "real_pcam"
  workstation.samples_dir = samples_dir
  samples_dir.mkdir(parents=True, exist_ok=True)
  (samples_dir / "manifest.csv").write_text(
    "case_id,label,image,organ\nTUNE-1,1,tile.png,lymph_node\n",
    encoding="utf-8",
  )
  meta_dir = workstation.results_dir / "TUNE-1"
  meta_dir.mkdir(parents=True)
  (meta_dir / "meta.json").write_text(
    json.dumps({
      "case_id": "TUNE-1",
      "ready": True,
      "label": 1,
      "case_score": 0.22,
      "organ_id": "lymph_node",
    }),
    encoding="utf-8",
  )

  strict = client.get("/api/validation?min_review_score=0.23").json()
  assert strict["fn"] == 1

  loose = client.get("/api/validation?min_review_score=0.15").json()
  assert loose["recall"] == 1.0

  session = client.post("/api/session/triage", json={"detection_threshold": 0.18}).json()
  assert session["applied_to_analysis"] is True
  assert session["active"]["detection_threshold"] == 0.18

  reset = client.post("/api/session/triage", json={"reset": True}).json()
  assert reset["applied_to_analysis"] is False


def test_organs_endpoint(client):
  resp = client.get("/api/organs")
  assert resp.status_code == 200
  body = resp.json()
  assert body["default_organ"] == "gastrointestinal"
  assert len(body["organs"]) >= 10
  ids = {o["id"] for o in body["organs"]}
  assert "breast" in ids
  assert "gastrointestinal" in ids


def test_model_endpoint_with_scorer_factory(client):
  resp = client.get("/api/model?organ=lymph_node")
  assert resp.status_code == 200
  assert resp.json()["organ_id"] == "lymph_node"


def test_dataset_catalog_endpoint(client):
  resp = client.get("/api/datasets/catalog")
  assert resp.status_code == 200
  body = resp.json()
  assert body["reference"] == "patchcamelyon"
  assert len(body["datasets"]) >= 5
  assert body["datasets"][0]["id"] == "patchcamelyon"


def test_dataset_hub_endpoint(client, workstation):
  resp = client.get("/api/datasets/hub")
  assert resp.status_code == 200
  body = resp.json()
  assert "datasets" in body
  assert any(d["organ_id"] == "lymph_node" for d in body["datasets"])


def test_samples_filter_by_organ(client, workstation):
  root = workstation.root / "outputs" / "test_datasets"
  gi = root / "gastrointestinal"
  gi.mkdir(parents=True, exist_ok=True)
  (gi / "manifest.csv").write_text(
    "case_id,label,image,organ,dataset_id,split\n"
    "GI-TUM-00,1,images/a.png,gastrointestinal,gastrointestinal,test\n",
    encoding="utf-8",
  )
  ln = root / "lymph_node"
  ln.mkdir(parents=True, exist_ok=True)
  (ln / "manifest.csv").write_text(
    "case_id,label,image,organ,dataset_id,split\n"
    "PCAM-NOR-00,0,images/b.png,lymph_node,lymph_node,test\n",
    encoding="utf-8",
  )
  gi_samples = client.get("/api/samples", params={"organ": "gastrointestinal"}).json()
  assert len(gi_samples) == 1
  assert gi_samples[0]["case_id"] == "GI-TUM-00"
  assert client.get("/api/samples", params={"organ": "dermatopathology"}).json() == []


def test_set_active_dataset_endpoint(client, workstation):
  out = workstation.root / "outputs" / "test_datasets" / "lymph_node"
  out.mkdir(parents=True)
  (out / "manifest.csv").write_text(
    "case_id,label,image,organ,dataset_id,split\n"
    "PCAM-NOR-00,0,images/a.png,lymph_node,lymph_node,test\n",
    encoding="utf-8",
  )
  resp = client.post("/api/datasets/set-active", json={"organ_id": "lymph_node"})
  assert resp.status_code == 200
  assert resp.json()["organ_id"] == "lymph_node"
  active = client.get("/api/datasets/active").json()
  assert active["organ_id"] == "lymph_node"


def test_get_image_after_analyze(client):
  case_id = _upload(client).json()["case_id"]
  img = client.get(f"/api/image/{case_id}/source.png")
  assert img.status_code == 200
  assert img.headers["content-type"].startswith("image/")

  bad_file = client.get(f"/api/image/{case_id}/evil.exe")
  assert bad_file.status_code == 400


def test_organ_metadata_mismatch_blocks_upload(client):
  mismatch_bytes = _png_bytes()
  resp = client.post(
    "/api/analyze/upload",
    files={"file": ("organ=breast_tile.png", mismatch_bytes, "image/png")},
    data={"organ": "lymph_node"},
  )
  assert resp.status_code == 409
  detail = resp.json()["detail"]
  assert detail["code"] == "organ_metadata_mismatch"

  ok = client.post(
    "/api/analyze/upload",
    files={"file": ("organ=breast_tile.png", mismatch_bytes, "image/png")},
    data={"organ": "lymph_node", "confirm_mismatch": "true"},
  )
  assert ok.status_code == 200
  assert ok.json()["metadata_mismatch"] is True
