"""Organ registry — maps specialty to specialized model checkpoints.

Each organ has its own ensemble checkpoint under ``models/<organ_id>/``.
The workstation requires an explicit organ selection before analysis so the
correct model is always used and recorded in the audit trail.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ORGANS_CONFIG_NAME = "organs.yaml"


@dataclass(frozen=True)
class OrganSpec:
  id: str
  name: str
  specialty: str
  task: str
  stain: str
  checkpoint: Path
  config: str
  aliases: tuple[str, ...] = ()
  model_ready: bool = False

  def to_dict(self) -> dict[str, Any]:
    return {
      "id": self.id,
      "name": self.name,
      "specialty": self.specialty,
      "task": self.task,
      "stain": self.stain,
      "checkpoint": str(self.checkpoint),
      "config": self.config,
      "model_ready": self.model_ready,
      "aliases": list(self.aliases),
    }


@dataclass
class OrganRegistry:
  root: Path
  default_organ: str
  organs: dict[str, OrganSpec] = field(default_factory=dict)

  def get(self, organ_id: str) -> OrganSpec:
    key = normalize_organ_id(organ_id)
    if key not in self.organs:
      raise KeyError(f"Unknown organ: {organ_id!r}")
    return self.organs[key]

  def resolve(self, organ_id: str | None) -> OrganSpec:
    if organ_id is None or not str(organ_id).strip():
      return self.get(self.default_organ)
    return self.get(organ_id)

  def list_organs(self) -> list[OrganSpec]:
    return sorted(self.organs.values(), key=lambda o: o.name.lower())

  def checkpoint_for(self, organ_id: str) -> Path:
    return self.get(organ_id).checkpoint

  def as_dicts(self) -> list[dict[str, Any]]:
    return [o.to_dict() for o in self.list_organs()]


def normalize_organ_id(raw: str) -> str:
  """Map user input / filename tokens to canonical organ id."""
  value = str(raw).strip().lower()
  value = value.replace("&", "and").replace("-", "_").replace(" ", "_")
  value = re.sub(r"[^a-z0-9_]", "", value)
  return value


def _alias_index(organs_cfg: dict[str, Any]) -> dict[str, str]:
  index: dict[str, str] = {}
  for organ_id, spec in organs_cfg.items():
    index[normalize_organ_id(organ_id)] = organ_id
    for alias in spec.get("aliases", []) or []:
      index[normalize_organ_id(alias)] = organ_id
    index[normalize_organ_id(spec.get("name", organ_id))] = organ_id
    index[normalize_organ_id(spec.get("specialty", organ_id))] = organ_id
  return index


def match_organ_id(raw: str, registry: OrganRegistry) -> str | None:
  """Return canonical organ id if *raw* matches a known organ or alias."""
  key = normalize_organ_id(raw)
  if key in registry.organs:
    return key
  cfg_path = registry.root / "config" / ORGANS_CONFIG_NAME
  if not cfg_path.is_file():
    return None
  data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
  index = _alias_index(data.get("organs", {}))
  return index.get(key)


def default_checkpoint_path(root: str | Path | None = None) -> Path:
  """Return the checkpoint path for the default organ in the registry."""
  registry = load_organ_registry(root)
  return registry.get(registry.default_organ).checkpoint


def organ_checkpoint_path(organ_id: str, root: str | Path | None = None) -> Path:
  """Return ``models/<organ_id>/ensemble.pt`` from the registry."""
  return load_organ_registry(root).get(organ_id).checkpoint


def load_organ_registry(root: str | Path | None = None) -> OrganRegistry:
  root = Path(root or Path(__file__).resolve().parents[1])
  cfg_path = root / "config" / ORGANS_CONFIG_NAME
  if not cfg_path.is_file():
    raise FileNotFoundError(f"Organ registry not found: {cfg_path}")

  data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
  default_organ = str(data.get("default_organ", "lymph_node"))
  organs: dict[str, OrganSpec] = {}

  for organ_id, spec in (data.get("organs") or {}).items():
    ckpt = root / str(spec["checkpoint"])
    organs[organ_id] = OrganSpec(
      id=organ_id,
      name=str(spec["name"]),
      specialty=str(spec.get("specialty", spec["name"])),
      task=str(spec.get("task", "")),
      stain=str(spec.get("stain", "H&E")),
      checkpoint=ckpt,
      config=str(spec.get("config", "default.yaml")),
      aliases=tuple(spec.get("aliases", []) or []),
      model_ready=ckpt.is_file(),
    )

  if default_organ not in organs:
    raise ValueError(f"default_organ {default_organ!r} not in registry")

  return OrganRegistry(root=root, default_organ=default_organ, organs=organs)


_FILENAME_ORGAN_RE = re.compile(
  r"(?:^|[_.\-])(organ|specialty|site)[=:_\-]([a-z0-9_&]+)",
  re.IGNORECASE,
)
_FILENAME_ORGAN_PREFIX_RE = re.compile(
  r"^(organ|specialty|site)[=:_\-]([a-z0-9_&]+)",
  re.IGNORECASE,
)


def _resolve_token_to_organ(token: str, registry: OrganRegistry) -> str | None:
  """Map a filename token to a canonical organ id."""
  direct = match_organ_id(token, registry)
  if direct:
    return direct
  for part in re.split(r"[_\-]+", token):
    if len(part) < 3:
      continue
    found = match_organ_id(part, registry)
    if found:
      return found
  return None


def detect_organ_from_filename(filename: str, registry: OrganRegistry) -> str | None:
  """Infer organ from structured filename tokens, e.g. ``organ=breast_tile.png``."""
  name = Path(filename).stem.lower()
  for pattern in (_FILENAME_ORGAN_PREFIX_RE, _FILENAME_ORGAN_RE):
    for match in pattern.finditer(name):
      found = _resolve_token_to_organ(match.group(2), registry)
      if found:
        return found
  # Leading token: breast_case001.png
  for part in re.split(r"[_.\-]+", name):
    if len(part) < 3:
      continue
    found = match_organ_id(part, registry)
    if found:
      return found
  return None


def detect_organ_from_sidecar(image_path: str | Path) -> str | None:
  """Read ``<image>.json`` sidecar for ``{"organ": "breast"}``."""
  path = Path(image_path)
  for sidecar in (path.with_suffix(path.suffix + ".json"), path.with_suffix(".json")):
    if not sidecar.is_file():
      continue
    try:
      payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
      continue
    organ = payload.get("organ") or payload.get("organ_id") or payload.get("specialty")
    if organ:
      return str(organ)
  return None


def detect_organ_metadata(
  filename: str | None,
  image_path: str | Path | None,
  registry: OrganRegistry,
) -> dict[str, Any]:
  """Collect organ hints from filename and sidecar metadata."""
  from_filename = detect_organ_from_filename(filename, registry) if filename else None
  from_sidecar = None
  if image_path is not None:
    raw = detect_organ_from_sidecar(image_path)
    if raw:
      from_sidecar = match_organ_id(raw, registry) or normalize_organ_id(raw)

  detected = from_sidecar or from_filename
  sources = []
  if from_filename:
    sources.append("filename")
  if from_sidecar:
    sources.append("sidecar")

  return {
    "detected_organ_id": detected,
    "detected_from_filename": from_filename,
    "detected_from_sidecar": from_sidecar,
    "metadata_sources": sources,
  }


def validate_organ_selection(
  selected_organ_id: str,
  metadata: dict[str, Any],
  *,
  require_ready_model: bool = True,
  registry: OrganRegistry | None = None,
) -> dict[str, Any]:
  """Validate organ choice; surface mismatches and missing checkpoints."""
  registry = registry or load_organ_registry()
  organ = registry.resolve(selected_organ_id)
  detected = metadata.get("detected_organ_id")
  mismatch = bool(detected and detected != organ.id)

  warnings: list[str] = []
  errors: list[str] = []

  if mismatch:
    detected_name = registry.get(detected).name
    warnings.append(
      f"Metadata suggests {detected_name} but you selected {organ.name}. "
      "Confirm the correct organ before analysis."
    )

  if require_ready_model and not organ.model_ready:
    errors.append(
      f"No trained model for {organ.name} ({organ.checkpoint}). "
      f"Train with: python -m pathassist.train --organ {organ.id} --source folder --data-dir ..."
    )

  return {
    "organ_id": organ.id,
    "organ_name": organ.name,
    "organ_specialty": organ.specialty,
    "organ_task": organ.task,
    "organ_stain": organ.stain,
    "model_ready": organ.model_ready,
    "metadata_detected_organ_id": detected,
    "metadata_mismatch": mismatch,
    "metadata_sources": metadata.get("metadata_sources", []),
    "warnings": warnings,
    "errors": errors,
    "ok": not errors,
  }
