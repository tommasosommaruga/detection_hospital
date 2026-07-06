"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _dev_env(monkeypatch):
  monkeypatch.setenv("PATHASSIST_ENV", "development")
  monkeypatch.delenv("PATHASSIST_API_KEY", raising=False)
