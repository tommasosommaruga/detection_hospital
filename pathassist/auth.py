"""API key authentication for the workstation."""

from __future__ import annotations

import secrets
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from .settings import Settings

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(settings: Settings, api_key: Optional[str]) -> Optional[str]:
  """Return reviewer identity when auth passes; None when auth is disabled."""
  if not settings.auth_enabled:
    return None
  if not api_key or not secrets.compare_digest(api_key, settings.api_key):
    raise HTTPException(status_code=401, detail="Invalid or missing API key")
  return "api-key-user"


def make_auth_dependency(settings: Settings):
  def _dep(api_key: Optional[str] = Security(_API_KEY_HEADER)) -> Optional[str]:
    return verify_api_key(settings, api_key)
  return _dep
