"""Configuration loading.

Keeping thresholds and workflow knobs in YAML (see config/default.yaml) means the
clinical behaviour can be tuned per site without touching code - which is exactly
the kind of change a validation study or a new hospital deployment will require.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
  """Load a YAML config file, falling back to the packaged default."""
  config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
  if not config_path.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")
  with config_path.open("r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle)
  if not isinstance(data, dict):
    raise ValueError(f"Config at {config_path} must be a mapping, got {type(data)!r}")
  return data
