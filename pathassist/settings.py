"""Environment-based deployment settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str = "") -> str:
  return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
  raw = _env(name, str(default).lower())
  return raw.lower() in {"1", "true", "yes", "on"}


def _default_checkpoint(root: Path) -> Path:
  from .organs import default_checkpoint_path

  return default_checkpoint_path(root)


@dataclass(frozen=True)
class Settings:
  env: str
  host: str
  port: int
  root: Path
  config_path: Path
  checkpoint: Path
  runs_dir: Path
  results_dir: Path
  upload_dir: Path
  log_level: str
  api_key: str
  cors_origins: list[str]
  max_upload_mb: int
  require_checkpoint: bool

  @property
  def is_production(self) -> bool:
    return self.env == "production"

  @property
  def auth_enabled(self) -> bool:
    return bool(self.api_key)

  @classmethod
  def from_env(cls, root: Path | None = None) -> Settings:
    root = root or Path(_env("PATHASSIST_ROOT", str(Path(__file__).resolve().parent.parent)))
    env = _env("PATHASSIST_ENV", "development")
    cors = [o.strip() for o in _env("PATHASSIST_CORS_ORIGINS", "*").split(",") if o.strip()]
    config_name = "production.yaml" if env == "production" else "default.yaml"
    return cls(
      env=env,
      host=_env("PATHASSIST_HOST", "0.0.0.0"),
      port=int(_env("PATHASSIST_PORT", "8765")),
      root=root,
      config_path=Path(_env("PATHASSIST_CONFIG", str(root / "config" / config_name))),
      checkpoint=Path(_env("PATHASSIST_CHECKPOINT", str(_default_checkpoint(root)))),
      runs_dir=Path(_env("PATHASSIST_RUNS_DIR", str(root / "runs"))),
      results_dir=Path(_env("PATHASSIST_RESULTS_DIR", str(root / "outputs" / "demo_results"))),
      upload_dir=Path(_env("PATHASSIST_UPLOAD_DIR", str(root / "outputs" / "demo_uploads"))),
      log_level=_env("PATHASSIST_LOG_LEVEL", "INFO").upper(),
      api_key=_env("PATHASSIST_API_KEY", ""),
      cors_origins=cors or ["*"],
      max_upload_mb=int(_env("PATHASSIST_MAX_UPLOAD_MB", "50")),
      require_checkpoint=_env_bool("PATHASSIST_REQUIRE_CHECKPOINT", env == "production"),
    )
