"""Tests for production auth and health endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from demo.server import WorkstationState, create_app
from pathassist.scoring import DummyScorer
from pathassist.settings import Settings


@pytest.fixture
def secured_client(tmp_path):
  settings = Settings(
    env="production",
    host="127.0.0.1",
    port=8765,
    root=tmp_path,
    config_path=tmp_path / "config" / "default.yaml",
    checkpoint=tmp_path / "missing.pt",
    runs_dir=tmp_path / "runs",
    results_dir=tmp_path / "results",
    upload_dir=tmp_path / "uploads",
    log_level="WARNING",
    api_key="test-secret-key",
    cors_origins=["*"],
    max_upload_mb=5,
    require_checkpoint=False,
  )
  state = WorkstationState(
    root=tmp_path,
    static_dir=tmp_path / "static",
    scorer_factory=lambda: DummyScorer(seed=1),
    checkpoint=tmp_path / "missing.pt",
    settings=settings,
  )
  state.static_dir.mkdir(parents=True)
  (state.static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")
  return TestClient(create_app(state, settings)), settings


def test_health_live(secured_client):
  client, _ = secured_client
  assert client.get("/health/live").json()["status"] == "alive"


def test_health_ready(secured_client):
  client, _ = secured_client
  assert client.get("/health/ready").json()["status"] == "ready"


def test_auth_blocks_upload_without_key(secured_client):
  client, _ = secured_client
  resp = client.post(
    "/api/analyze/upload",
    files={"file": ("t.png", b"not-an-image", "image/png")},
  )
  assert resp.status_code == 401


def test_auth_allows_upload_with_key(secured_client):
  client, settings = secured_client
  # invalid image still fails after auth — we only check not 401
  resp = client.post(
    "/api/analyze/upload",
    files={"file": ("t.png", b"x", "image/png")},
    headers={"X-API-Key": settings.api_key},
  )
  assert resp.status_code != 401
