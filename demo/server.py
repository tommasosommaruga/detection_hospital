"""PathAssist workstation — local API for the decision-support pipeline."""

from __future__ import annotations

import csv
import io
import json
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from contextlib import asynccontextmanager
import logging

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from demo.logic import (
  ALLOWED_IMAGE_FILES,
  VALID_DECISIONS,
  TRIAGE_OVERRIDE_KEYS,
  build_case_payload,
  compute_validation_metrics,
  extract_triage_thresholds,
  infer_tile_size,
  merge_triage_overrides,
  pick_config_name,
  safe_case_id,
  safe_image_filename,
  worklist_sort_key,
)

from pathassist.auth import make_auth_dependency
from pathassist.logging_config import setup_logging
from pathassist.settings import Settings

logger = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"


class TriageSessionBody(BaseModel):
  detection_threshold: Optional[float] = None
  metastasis_threshold: Optional[float] = None
  min_review_score: Optional[float] = None
  reset: bool = False


@dataclass
class WorkstationState:
  """Injectable paths and scorer for tests vs production."""

  root: Path = ROOT
  static_dir: Path = field(default_factory=lambda: STATIC)
  samples_dir: Path | None = None
  upload_dir: Path | None = None
  results_dir: Path | None = None
  checkpoint: Path | None = None
  runs_dir: Path | None = None
  scorer_factory: Callable[[], Any] | None = None
  settings: Settings | None = None
  triage_overrides: dict[str, float] = field(default_factory=dict)

  @classmethod
  def from_settings(cls, settings: Settings) -> WorkstationState:
    return cls(
      root=settings.root,
      checkpoint=settings.checkpoint,
      runs_dir=settings.runs_dir,
      results_dir=settings.results_dir,
      upload_dir=settings.upload_dir,
      settings=settings,
    )

  def __post_init__(self) -> None:
    if self.upload_dir is None:
      self.upload_dir = self.root / "outputs" / "demo_uploads"
    if self.results_dir is None:
      self.results_dir = self.root / "outputs" / "demo_results"
    if self.checkpoint is None:
      from pathassist.organs import default_checkpoint_path

      self.checkpoint = default_checkpoint_path(self.root)
    if self.runs_dir is None:
      self.runs_dir = self.root / "runs"

  @property
  def effective_samples_dir(self) -> Path:
    if self.samples_dir is not None:
      return self.samples_dir
    from pathassist.test_datasets import resolve_samples_dir

    return resolve_samples_dir(self.root)

  @property
  def manifest(self) -> Path:
    return self.effective_samples_dir / "manifest.csv"

  _scorer: Any = field(default=None, init=False, repr=False)
  _ckpt_meta: dict[str, Any] | None = field(default=None, init=False, repr=False)
  _scorers_by_organ: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
  _organ_registry: Any = field(default=None, init=False, repr=False)

  def reset_cache(self) -> None:
    self._scorer = None
    self._ckpt_meta = None
    self._scorers_by_organ = {}

  def get_organ_registry(self):
    if self._organ_registry is None:
      from pathassist.organs import load_organ_registry

      self._organ_registry = load_organ_registry(self.root)
    return self._organ_registry

  def ensure_dirs(self) -> None:
    from pathassist.test_datasets import test_datasets_root

    for path in (
      self.upload_dir,
      self.results_dir,
      self.effective_samples_dir,
      self.runs_dir,
      test_datasets_root(self.root),
    ):
      path.mkdir(parents=True, exist_ok=True)

  def get_checkpoint_meta(self, checkpoint: Path | None = None) -> dict[str, Any]:
    ckpt = checkpoint or self.checkpoint
    if checkpoint is None and self._ckpt_meta is not None:
      return self._ckpt_meta
    if not ckpt.exists():
      return {}
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = {
      "format": payload.get("checkpoint_format"),
      "vote_mode": payload.get("vote_mode"),
      "tile_size": payload.get("tile_size"),
      "dataset": payload.get("dataset"),
      "organ_id": payload.get("organ_id"),
      "metrics": payload.get("metrics"),
      "cv_summary": payload.get("cv_summary"),
      "members": [
        {
          "name": m["name"],
          "backbone": m.get("model_config", {}).get("backbone", "custom"),
          "weight": m.get("weight"),
          "val_acc": m.get("val_acc"),
        }
        for m in payload.get("members", [])
      ],
    }
    if checkpoint is None:
      self._ckpt_meta = meta
    return meta

  def get_scorer_for_organ(self, organ_id: str):
    registry = self.get_organ_registry()
    organ = registry.resolve(organ_id)
    if organ.id in self._scorers_by_organ:
      return self._scorers_by_organ[organ.id], organ

    if self.scorer_factory is not None:
      scorer = self.scorer_factory()
      self._scorers_by_organ[organ.id] = scorer
      return scorer, organ

    from pathassist.scoring import EnsembleScorer

    if not organ.checkpoint.exists():
      raise RuntimeError(f"No checkpoint for {organ.name}: {organ.checkpoint}")
    scorer = EnsembleScorer(str(organ.checkpoint), device_preference="auto")
    self._scorers_by_organ[organ.id] = scorer
    return scorer, organ

  def get_scorer(self):
    if self._scorer is None:
      registry = self.get_organ_registry()
      self._scorer, _ = self.get_scorer_for_organ(registry.default_organ)
    return self._scorer


class ReviewBody(BaseModel):
  case_id: str
  decision: str
  reviewer: str
  note: str = ""


class DatasetImportBody(BaseModel):
  count: int = 20
  activate: bool = True
  force: bool = False


class DatasetImportLocalBody(BaseModel):
  folder: Optional[str] = None
  activate: bool = True
  max_samples: Optional[int] = None


class ActiveDatasetBody(BaseModel):
  organ_id: str


def _utc_now() -> str:
  return datetime.now(timezone.utc).isoformat()


def _resolve_tile_size(
  state: WorkstationState,
  organ_spec,
  height: int,
  width: int,
) -> int:
  """Use the checkpoint's tile size for single-patch uploads (e.g. 224×224 GI tiles)."""
  if organ_spec.checkpoint.exists():
    meta = state.get_checkpoint_meta(organ_spec.checkpoint)
    ckpt_tile = meta.get("tile_size")
    if ckpt_tile:
      ckpt_tile = int(ckpt_tile)
      if height <= ckpt_tile and width <= ckpt_tile:
        return ckpt_tile
  return infer_tile_size(height, width)


def _pick_config(state: WorkstationState, tile_size: int, organ_id: str | None = None) -> dict:
  from pathassist.config import load_config

  registry = state.get_organ_registry()
  organ = registry.resolve(organ_id)
  config = load_config(state.root / "config" / organ.config)
  config["tiling"]["tile_size"] = tile_size
  if organ.id == registry.default_organ and tile_size <= 128:
    pcam = load_config(state.root / "config" / "pcam_test.yaml")
    config["tiling"].update(pcam.get("tiling", {}))
    config["tiling"]["tile_size"] = tile_size
    config["tiling"]["force_whole_image"] = True
  return _apply_effective_triage(config, state)


def _organ_context_from_validation(validation: dict[str, Any], organ) -> dict[str, Any]:
  return {
    "organ_id": organ.id,
    "organ_name": organ.name,
    "organ_specialty": organ.specialty,
    "organ_task": organ.task,
    "organ_stain": organ.stain,
    "model_checkpoint": str(organ.checkpoint),
    "metadata_detected_organ_id": validation.get("metadata_detected_organ_id"),
    "metadata_mismatch": validation.get("metadata_mismatch", False),
    "metadata_sources": validation.get("metadata_sources", []),
    "warnings": validation.get("warnings", []),
  }


def _parse_organ_form(
  state: WorkstationState,
  organ: str | None,
  filename: str | None = None,
  image_path: Path | None = None,
  *,
  allow_mismatch: bool = False,
  require_ready: bool = True,
) -> tuple[dict[str, Any], Any]:
  from pathassist.organs import detect_organ_metadata, validate_organ_selection

  if not organ or not str(organ).strip():
    raise HTTPException(
      400,
      "Organ selection is required. Choose the organ/specialty that matches this image.",
    )

  registry = state.get_organ_registry()
  metadata = detect_organ_metadata(filename, image_path, registry)
  validation = validate_organ_selection(
    organ,
    metadata,
    require_ready_model=require_ready and state.scorer_factory is None,
    registry=registry,
  )

  if validation["errors"]:
    raise HTTPException(400, "; ".join(validation["errors"]))

  if validation["metadata_mismatch"] and not allow_mismatch:
    raise HTTPException(
      409,
      detail={
        "code": "organ_metadata_mismatch",
        "message": validation["warnings"][0] if validation["warnings"] else "Organ mismatch",
        "selected_organ_id": validation["organ_id"],
        "detected_organ_id": validation["metadata_detected_organ_id"],
        "warnings": validation["warnings"],
      },
    )

  organ_spec = registry.get(validation["organ_id"])
  return _organ_context_from_validation(validation, organ_spec), organ_spec


def _review_status(state: WorkstationState, case_id: str) -> dict[str, Any] | None:
  from pathassist.audit import AuditStore

  for row in reversed(AuditStore(state.runs_dir).load_decisions()):
    if row.get("case_id") == case_id:
      return row
  return None


def _analyze_image(
  state: WorkstationState,
  case_id: str,
  image: np.ndarray,
  label: int | None = None,
  organ_id: str | None = None,
  filename: str | None = None,
  image_path: Path | None = None,
  allow_organ_mismatch: bool = False,
) -> dict:
  from pathassist.audit import AuditStore
  from pathassist.pipeline import analyze_case

  organ_ctx, organ_spec = _parse_organ_form(
    state,
    organ_id,
    filename=filename,
    image_path=image_path,
    allow_mismatch=allow_organ_mismatch,
  )

  state.ensure_dirs()
  h, w = image.shape[:2]
  tile_size = _resolve_tile_size(state, organ_spec, h, w)
  config = _pick_config(state, tile_size, organ_spec.id)
  config["organ"] = organ_ctx

  case_dir = state.results_dir / case_id
  case_dir.mkdir(parents=True, exist_ok=True)

  scorer, _ = state.get_scorer_for_organ(organ_spec.id)

  case, heatmap_path, report_path, uncertainty_path, nn_explanation, wsi_info = analyze_case(
    case_id=case_id,
    image=image,
    scorer=scorer,
    config=config,
    output_dir=case_dir,
    audit_store=AuditStore(state.runs_dir),
    image_path=image_path,
  )

  from pathassist.wsi import load_wsi_thumbnail

  source_image = (
    load_wsi_thumbnail(image_path, max_side=int(config["tiling"].get("thumbnail_max_side", 2048)))
    if wsi_info is not None and image_path is not None
    else image
  )
  Image.fromarray(source_image).save(case_dir / "source.png")
  shutil.copy(heatmap_path, case_dir / "heatmap.png")
  if uncertainty_available := (uncertainty_path and uncertainty_path.exists()):
    shutil.copy(uncertainty_path, case_dir / "uncertainty.png")
  shutil.copy(report_path, case_dir / "report.txt")

  if nn_explanation:
    explain_meta = {
      "score": nn_explanation.get("score"),
      "backbone": nn_explanation.get("backbone"),
      "layers": nn_explanation.get("layers", []),
      "grad_cam": nn_explanation.get("grad_cam"),
      "paths": {},
    }
    for key, src in (nn_explanation.get("paths") or {}).items():
      src_path = Path(src).resolve()
      if src_path.exists():
        dest = (case_dir / src_path.name).resolve()
        if src_path != dest:
          shutil.copy(src_path, dest)
        explain_meta["paths"][key] = f"/api/image/{case_id}/{dest.name}"
    result_explain = explain_meta
  else:
    result_explain = None

  result = build_case_payload(
    case,
    case_id,
    label,
    tile_size,
    uncertainty_path if uncertainty_available else None,
    _review_status(state, case_id),
    _utc_now(),
    config,
    organ=organ_ctx,
    wsi_info=wsi_info,
  )
  result["report"] = report_path.read_text(encoding="utf-8")
  if result_explain:
    result["nn_explanation"] = result_explain
  (case_dir / "meta.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
  return result


def _parse_label(raw: Any) -> int | None:
  if raw is None:
    return None
  text = str(raw).strip()
  if not text:
    return None
  return int(text)


def _samples_dir_for(state: WorkstationState, organ_id: str | None = None) -> Path:
  if organ_id:
    from pathassist.test_datasets import dataset_dir

    return dataset_dir(state.root, organ_id)
  return state.effective_samples_dir


def _manifest_path_for(state: WorkstationState, organ_id: str | None = None) -> Path:
  return _samples_dir_for(state, organ_id) / "manifest.csv"


def _load_samples(state: WorkstationState, organ_id: str | None = None) -> list[dict]:
  path = _manifest_path_for(state, organ_id)
  if not path.is_file():
    return []
  with path.open(encoding="utf-8") as handle:
    return list(csv.DictReader(handle))


def _find_sample_row(
  state: WorkstationState,
  case_id: str,
  organ_hint: str | None = None,
) -> tuple[dict, Path] | tuple[None, None]:
  """Locate a manifest row and its on-disk images directory."""
  from pathassist.test_datasets import test_datasets_root

  candidates: list[tuple[str | None, Path]] = []
  if organ_hint:
    candidates.append((organ_hint, _samples_dir_for(state, organ_hint)))
  candidates.append((None, state.effective_samples_dir))
  root = test_datasets_root(state.root)
  if root.is_dir():
    for child in sorted(root.iterdir()):
      if child.is_dir() and (child / "manifest.csv").is_file():
        candidates.append((child.name, child))

  seen: set[Path] = set()
  for organ_id, samples_dir in candidates:
    if samples_dir in seen:
      continue
    seen.add(samples_dir)
    row = next((r for r in _load_samples(state, organ_id) if r["case_id"] == case_id), None)
    if row is not None:
      return row, samples_dir
  return None, None


def _all_case_ids(state: WorkstationState) -> list[str]:
  ids = set()
  if state.results_dir.exists():
    for path in state.results_dir.iterdir():
      if path.is_dir() and (path / "meta.json").exists():
        ids.add(path.name)
  return sorted(ids)


  ids = set()
  if state.results_dir.exists():
    for path in state.results_dir.iterdir():
      if path.is_dir() and (path / "meta.json").exists():
        ids.add(path.name)
  return sorted(ids)


def _validation_base_config(state: WorkstationState, samples: list[dict]) -> dict:
  """Organ YAML triage thresholds without session/preview overrides."""
  from pathassist.config import load_config

  registry = state.get_organ_registry()
  organ_ids = {s.get("organ_id") for s in samples if s.get("organ_id")}
  if "lymph_node" in organ_ids:
    organ_id = "lymph_node"
  elif organ_ids:
    organ_id = sorted(organ_ids)[0]
  else:
    organ_id = registry.default_organ
  organ = registry.get(organ_id)
  config = load_config(state.root / "config" / organ.config)
  config["_validation_config_source"] = f"config/{organ.config} ({organ.name})"
  return config


def _apply_effective_triage(
  config: dict[str, Any],
  state: WorkstationState,
  preview: dict[str, float] | None = None,
) -> dict[str, Any]:
  """Merge session + optional preview overrides onto an organ config."""
  effective: dict[str, float] = dict(state.triage_overrides)
  if preview:
    effective.update(preview)
  source = "preview" if preview else ("session" if state.triage_overrides else "yaml")
  merged = merge_triage_overrides(config, effective if effective else None)
  merged["_threshold_source"] = source
  return merged


def _validation_config(
  state: WorkstationState,
  samples: list[dict],
  preview: dict[str, float] | None = None,
) -> dict:
  return _apply_effective_triage(_validation_base_config(state, samples), state, preview)


def _parse_triage_preview(
  detection_threshold: float | None,
  metastasis_threshold: float | None,
  min_review_score: float | None,
) -> dict[str, float] | None:
  preview = {
    key: value
    for key, value in {
      "detection_threshold": detection_threshold,
      "metastasis_threshold": metastasis_threshold,
      "min_review_score": min_review_score,
    }.items()
    if value is not None
  }
  return preview or None


def _validation_summary_data(
  state: WorkstationState,
  preview: dict[str, float] | None = None,
) -> dict:
  samples = _sample_entries(state)
  config = _validation_config(state, samples, preview=preview)
  result = compute_validation_metrics(samples, config)
  if result.get("ready"):
    base = _validation_base_config(state, samples)
    result["yaml_defaults"] = extract_triage_thresholds(base)
  return result


def _clamp_triage_value(value: float) -> float:
  return float(max(0.01, min(0.99, value)))


def _sample_entries(state: WorkstationState, organ_id: str | None = None) -> list[dict]:
  samples_dir = _samples_dir_for(state, organ_id)
  out = []
  for row in _load_samples(state, organ_id):
    case_id = row["case_id"]
    organ_id = row.get("organ") or state.get_organ_registry().default_organ
    label = _parse_label(row.get("label"))
    if label is None:
      label_name = None
    else:
      from pathassist.test_datasets import label_names_for_organ

      neg, pos = label_names_for_organ(organ_id, state.root)
      label_name = pos if label == 1 else neg
    entry = {
      "case_id": case_id,
      "label": label,
      "label_name": label_name,
      "organ_id": organ_id,
      "dataset_id": row.get("dataset_id"),
      "split": row.get("split"),
      "source_image": f"/api/sample-tile/{case_id}/thumb.png",
      "ready": (state.results_dir / case_id / "meta.json").exists(),
      "has_label": label is not None,
    }
    meta = state.results_dir / case_id / "meta.json"
    if meta.exists():
      entry.update(json.loads(meta.read_text(encoding="utf-8")))
    out.append(entry)
  return out


def _resolve_image_path(
  state: WorkstationState,
  image_field: str,
  samples_dir: Path | None = None,
) -> Path:
  path = Path(image_field)
  if path.is_absolute():
    return path
  base = samples_dir or state.effective_samples_dir
  in_samples = base / path
  if in_samples.exists():
    return in_samples
  legacy = state.root / "outputs" / "real_pcam" / path
  if legacy.exists():
    return legacy
  return state.root / path


def create_app(state: WorkstationState | None = None, settings: Settings | None = None) -> FastAPI:
  settings = settings or Settings.from_env(ROOT)
  setup_logging(settings.log_level)
  ws = state or WorkstationState.from_settings(settings)

  @asynccontextmanager
  async def lifespan(_app: FastAPI):
    ws.ensure_dirs()
    logger.info("PathAssist started env=%s host=%s port=%s", settings.env, settings.host, settings.port)
    yield

  app = FastAPI(title="PathAssist Workstation", version="1.1.0", lifespan=lifespan)

  app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
  )

  require_auth = make_auth_dependency(settings)

  def _max_upload_bytes() -> int:
    return settings.max_upload_mb * 1024 * 1024

  @app.get("/", response_class=HTMLResponse)
  def index() -> HTMLResponse:
    return HTMLResponse((ws.static_dir / "index.html").read_text(encoding="utf-8"))

  @app.get("/api/health")
  def health() -> dict:
    meta = ws.get_checkpoint_meta()
    return {
      "status": "ok" if ws.checkpoint.exists() else "no_checkpoint",
      "env": settings.env,
      "checkpoint": str(ws.checkpoint),
      "checkpoint_exists": ws.checkpoint.exists(),
      "model_loaded": ws._scorer is not None,
      "cases_cached": len(_all_case_ids(ws)),
      "samples_available": ws.manifest.exists(),
      "metrics": meta.get("metrics"),
      "wsi_openslide": __import__("pathassist.wsi", fromlist=["openslide_available"]).openslide_available(),
    }

  @app.get("/health/live")
  def health_live() -> dict:
    return {"status": "alive"}

  @app.get("/health/ready")
  def health_ready() -> dict:
    if settings.require_checkpoint and not ws.checkpoint.exists():
      raise HTTPException(503, "Checkpoint not loaded")
    try:
      ws.runs_dir.mkdir(parents=True, exist_ok=True)
      probe = ws.runs_dir / ".ready"
      probe.write_text("ok", encoding="utf-8")
    except OSError as exc:
      raise HTTPException(503, f"Storage not writable: {exc}") from exc
    return {"status": "ready", "checkpoint": str(ws.checkpoint)}

  @app.get("/api/mission")
  def mission() -> dict:
    return {
      "title": "PathAssist — Digital Pathology Assistant",
      "aim": (
        "Help pathologists work faster and safer. PathAssist scans cases, prioritizes "
        "urgent slides, highlights suspicious regions with explainable evidence, drafts "
        "reports, flags quality issues, and learns from pathologist corrections — "
        "so critical cases are seen first and fewer cancers are missed under workload pressure."
      ),
      "clinical_value": [
        "AI-assisted triage — high-risk cases at the top of the queue",
        "Explainable detection — heatmaps, uncertainty, ranked review regions",
        "QC assistant — blur, tissue, staining checks before sign-off",
        "Second reader — surfaces model vs pathologist disagreement",
        "Report drafting — structured draft for review and edit",
        "Continuous learning — corrections export for retraining",
      ],
      "pipeline": [
        "Whole-slide / patch ingestion",
        "Quality control (blur, tissue coverage, staining)",
        "Suspicious region detection (ensemble scoring)",
        "Case triage & priority worklist",
        "Severity estimate & explainability overlays",
        "Report draft for pathologist review",
        "Human approve / modify / reject",
        "Export corrections for retraining",
      ],
      "not": (
        "Not an autonomous diagnostician. The pathologist reviews every case and "
        "signs off. Regulatory clearance and hospital validation are required before "
        "clinical deployment."
      ),
      "deployment_ready": [
        "Ensemble checkpoint (GPU train, CPU inference)",
        "Audit trail (JSONL)",
        "Configurable thresholds (YAML)",
        "Docker + health probes",
        "API key authentication",
        "WSI thumbnail ingest (OpenSlide optional)",
      ],
      "roadmap": [
      "OpenSlide WSI — scan full slides, show only suspicious regions",
      "Cell-level detection (bounding boxes)",
      "Similar historical cases for rare-pattern alerts",
      "LIS / EHR integration",
      "Clinical ISUP / Gleason grading with labeled data",
      "Multi-modal inputs (labs, history, genomics)",
    ],
    }

  @app.get("/api/organs")
  def list_organs() -> dict:
    registry = ws.get_organ_registry()
    ready = sum(1 for o in registry.list_organs() if o.model_ready)
    return {
      "default_organ": registry.default_organ,
      "ready_count": ready,
      "total": len(registry.organs),
      "organs": registry.as_dicts(),
    }

  @app.get("/api/model")
  def model_info(organ: Optional[str] = None) -> dict:
    registry = ws.get_organ_registry()
    organ_spec = registry.resolve(organ)
    if ws.scorer_factory is None and not organ_spec.checkpoint.exists():
      raise HTTPException(404, f"No model for {organ_spec.name}")
    meta = ws.get_checkpoint_meta(organ_spec.checkpoint)
    if ws.scorer_factory is not None:
      scorer = ws.scorer_factory()
    else:
      scorer, _ = ws.get_scorer_for_organ(organ_spec.id)
    return {
      **meta,
      "organ_id": organ_spec.id,
      "organ_name": organ_spec.name,
      "scorer_name": scorer.name,
      "scorer_version": scorer.version,
      "n_members": len(meta.get("members", [])),
      "checkpoint": str(organ_spec.checkpoint),
      "model_ready": organ_spec.model_ready,
    }

  @app.get("/api/samples")
  def list_samples(organ: Optional[str] = Query(None)) -> list[dict]:
    organ_id = str(organ).strip() if organ else None
    if organ_id:
      ws.get_organ_registry().resolve(organ_id)
    return _sample_entries(ws, organ_id=organ_id or None)

  @app.get("/api/cases")
  def list_cases() -> list[dict]:
    cases = []
    for case_id in _all_case_ids(ws):
      meta = json.loads((ws.results_dir / case_id / "meta.json").read_text())
      meta["review"] = _review_status(ws, case_id)
      cases.append(meta)
    cases.sort(key=worklist_sort_key)
    return cases

  @app.get("/api/worklist")
  def worklist() -> list[dict]:
    from pathassist.audit import AuditStore

    store = AuditStore(ws.runs_dir)
    latest: dict[str, dict] = {}
    for record in store.load_results():
      latest[record["result"]["case_id"]] = record

    ranked_meta = list_cases()
    for item in ranked_meta:
      item["audit_recorded"] = item["case_id"] in latest
    return ranked_meta

  @app.get("/api/validation")
  def validation_summary(
    detection_threshold: Optional[float] = Query(None, ge=0.01, le=0.99),
    metastasis_threshold: Optional[float] = Query(None, ge=0.01, le=0.99),
    min_review_score: Optional[float] = Query(None, ge=0.01, le=0.99),
  ) -> dict:
    samples = _sample_entries(ws)
    preview = _parse_triage_preview(
      detection_threshold, metastasis_threshold, min_review_score,
    )
    return _validation_summary_data(ws, preview=preview)

  @app.get("/api/session/triage")
  def get_triage_session() -> dict:
    samples = _sample_entries(ws)
    base = _validation_base_config(ws, samples)
    defaults = extract_triage_thresholds(base)
    active = extract_triage_thresholds(_validation_config(ws, samples))
    return {
      "defaults": defaults,
      "active": active,
      "session_overrides": dict(ws.triage_overrides),
      "config_source": base.get("_validation_config_source"),
      "applied_to_analysis": bool(ws.triage_overrides),
    }

  @app.post("/api/session/triage")
  def set_triage_session(body: TriageSessionBody) -> dict:
    if body.reset:
      ws.triage_overrides = {}
    else:
      for key in TRIAGE_OVERRIDE_KEYS:
        value = getattr(body, key)
        if value is not None:
          ws.triage_overrides[key] = _clamp_triage_value(value)
    samples = _sample_entries(ws)
    base = _validation_base_config(ws, samples)
    return {
      "defaults": extract_triage_thresholds(base),
      "active": extract_triage_thresholds(_validation_config(ws, samples)),
      "session_overrides": dict(ws.triage_overrides),
      "applied_to_analysis": bool(ws.triage_overrides),
      "config_source": base.get("_validation_config_source"),
    }

  @app.get("/api/datasets/catalog")
  def dataset_catalog() -> dict:
    from pathassist.dataset_similarity import catalog_as_dicts

    return {"reference": "patchcamelyon", "datasets": catalog_as_dicts()}

  @app.get("/api/datasets/hub")
  def dataset_hub() -> dict:
    from pathassist.test_datasets import hub_catalog

    return hub_catalog(ws.root)

  @app.get("/api/datasets/active")
  def dataset_active() -> dict:
    from pathassist.test_datasets import read_active_organ, resolve_samples_dir

    organ_id = read_active_organ(ws.root)
    return {
      "organ_id": organ_id,
      "samples_dir": str(resolve_samples_dir(ws.root)),
      "manifest_exists": ws.manifest.exists(),
    }

  @app.get("/api/explain/{case_id}")
  def get_explanation(case_id: str) -> dict:
    if not safe_case_id(case_id):
      raise HTTPException(status_code=400, detail="Invalid case id")
    meta_path = ws.results_dir / case_id / "meta.json"
    if not meta_path.exists():
      raise HTTPException(status_code=404, detail="Case not analyzed")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "nn_explanation" not in meta:
      raise HTTPException(status_code=404, detail="No NN explanation for this case")
    return meta["nn_explanation"]

  @app.get("/api/correlation")
  def correlation_summary() -> dict:
    report_path = ws.root / "outputs" / "dataset_correlation.json"
    if not report_path.exists():
      return {"ready": False, "message": "Run scripts/dataset_correlation.py first"}
    return {"ready": True, **json.loads(report_path.read_text(encoding="utf-8"))}

  @app.get("/api/result/{case_id}")
  def get_result(case_id: str) -> dict:
    if not safe_case_id(case_id):
      raise HTTPException(400, "Invalid case id")
    meta_path = ws.results_dir / case_id / "meta.json"
    if not meta_path.exists():
      raise HTTPException(404, f"No result for {case_id}")
    result = json.loads(meta_path.read_text(encoding="utf-8"))
    result["review"] = _review_status(ws, case_id)
    return result

  @app.get("/api/report/{case_id}", response_class=PlainTextResponse)
  def get_report(case_id: str) -> PlainTextResponse:
    if not safe_case_id(case_id):
      raise HTTPException(400, "Invalid case id")
    path = ws.results_dir / case_id / "report.txt"
    if not path.exists():
      raise HTTPException(404, "Report not found")
    return PlainTextResponse(path.read_text(encoding="utf-8"))

  @app.post("/api/analyze/sample/{case_id}")
  def analyze_sample(
    case_id: str,
    organ: Optional[str] = Form(None),
    confirm_mismatch: bool = Form(False),
    _auth=Depends(require_auth),
  ) -> dict:
    if not safe_case_id(case_id):
      raise HTTPException(400, "Invalid case id")
    row, samples_dir = _find_sample_row(ws, case_id, organ_hint=organ)
    if row is None or samples_dir is None:
      raise HTTPException(404, "Unknown sample")
    path = _resolve_image_path(ws, row["image"], samples_dir=samples_dir)
    if not path.exists():
      raise HTTPException(404, f"Image not found: {path}")
    selected_organ = organ or row.get("organ") or ws.get_organ_registry().default_organ
    from pathassist.wsi import is_wsi_path, load_wsi_thumbnail

    if is_wsi_path(path):
      img = load_wsi_thumbnail(path)
    else:
      img = np.array(Image.open(path).convert("RGB"))
    return _analyze_image(
      ws,
      case_id,
      img,
      label=_parse_label(row.get("label")),
      organ_id=selected_organ,
      filename=path.name,
      image_path=path,
      allow_organ_mismatch=confirm_mismatch,
    )

  @app.post("/api/analyze/upload")
  async def analyze_upload(
    file: UploadFile = File(...),
    organ: str = Form(...),
    confirm_mismatch: bool = Form(False),
    _auth=Depends(require_auth),
  ) -> dict:
    from pathassist.wsi import WSI_EXTENSIONS, is_wsi_path, load_wsi_thumbnail, openslide_available

    raw = await file.read()
    if len(raw) > _max_upload_bytes():
      raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")

    upload_name = file.filename or "image.png"
    ext = Path(upload_name).suffix.lower()
    is_wsi = ext in WSI_EXTENSIONS or is_wsi_path(upload_name)

    if is_wsi:
      if not openslide_available():
        raise HTTPException(
          400,
          "WSI upload requires OpenSlide on the server (install openslide + openslide-python).",
        )
    elif not file.content_type or not file.content_type.startswith("image/"):
      raise HTTPException(400, "Please upload an image or whole-slide file (.svs, .ndpi, …)")

    case_id = f"UPLOAD-{uuid.uuid4().hex[:8].upper()}"
    ws.ensure_dirs()
    upload_path = ws.upload_dir / f"{case_id}_{upload_name}"
    upload_path.write_bytes(raw)

    try:
      if is_wsi:
        img = load_wsi_thumbnail(upload_path)
      else:
        img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception as exc:
      raise HTTPException(400, f"Invalid image: {exc}") from exc

    return _analyze_image(
      ws,
      case_id,
      img,
      organ_id=organ,
      filename=upload_name,
      image_path=upload_path,
      allow_organ_mismatch=confirm_mismatch,
    )

  @app.post("/api/analyze/batch")
  async def analyze_batch(
    files: list[UploadFile] = File(...),
    organ: str = Form(...),
    confirm_mismatch: bool = Form(False),
    _auth=Depends(require_auth),
  ) -> dict:
    results = []
    from pathassist.wsi import WSI_EXTENSIONS, is_wsi_path, load_wsi_thumbnail, openslide_available

    for file in files:
      raw = await file.read()
      upload_name = file.filename or "image.png"
      ext = Path(upload_name).suffix.lower()
      is_wsi = ext in WSI_EXTENSIONS or is_wsi_path(upload_name)
      if is_wsi and not openslide_available():
        continue
      if not is_wsi and (not file.content_type or not file.content_type.startswith("image/")):
        continue
      try:
        case_id = f"BATCH-{uuid.uuid4().hex[:6].upper()}"
        upload_path = ws.upload_dir / f"{case_id}_{upload_name}"
        upload_path.write_bytes(raw)
        if is_wsi:
          img = load_wsi_thumbnail(upload_path)
        else:
          img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
      except Exception:
        continue
      results.append(
        _analyze_image(
          ws,
          case_id,
          img,
          organ_id=organ,
          filename=upload_name,
          image_path=upload_path,
          allow_organ_mismatch=confirm_mismatch,
        )
      )
    return {"count": len(results), "results": results}

  @app.post("/api/review")
  def submit_review(body: ReviewBody, _auth=Depends(require_auth)) -> dict:
    from dataclasses import asdict

    from pathassist.audit import AuditStore, ReviewDecision

    if body.decision not in VALID_DECISIONS:
      raise HTTPException(400, "Invalid decision")
    store = AuditStore(ws.runs_dir)
    decision = ReviewDecision(
      case_id=body.case_id,
      decision=body.decision,
      reviewer=body.reviewer,
      note=body.note,
    )
    store.record_decision(decision)
    return asdict(decision)

  @app.post("/api/export/corrections")
  def export_corrections(_auth=Depends(require_auth)) -> dict:
    from pathassist.audit import AuditStore

    out = ws.root / "outputs" / "corrections.csv"
    path = AuditStore(ws.runs_dir).export_corrections(out)
    return {"path": str(path), "download": "/api/download/corrections.csv"}

  @app.get("/api/download/corrections.csv")
  def download_corrections() -> FileResponse:
    path = ws.root / "outputs" / "corrections.csv"
    if not path.exists():
      raise HTTPException(404, "No corrections yet")
    return FileResponse(path, filename="corrections.csv")

  if not settings.is_production:

    @app.post("/api/datasets/set-active")
    def set_active_dataset_route(payload: ActiveDatasetBody, _auth=Depends(require_auth)) -> dict:
      from pathassist.test_datasets import set_active_dataset

      try:
        out_dir = set_active_dataset(payload.organ_id, root=ws.root)
      except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
      ws.reset_cache()
      return {"organ_id": payload.organ_id, "samples_dir": str(out_dir)}

    @app.post("/api/datasets/{organ_id}/import")
    def import_dataset(
      organ_id: str,
      payload: DatasetImportBody,
      _auth=Depends(require_auth),
    ) -> dict:
      from pathassist.test_datasets import get_training_spec, start_import_job

      try:
        get_training_spec(organ_id, root=ws.root)
      except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
      job_id = start_import_job(
        organ_id,
        count=payload.count,
        root=ws.root,
        activate=payload.activate,
        force=payload.force,
      )
      return {"job_id": job_id, "organ_id": organ_id, "status": "running"}

    @app.post("/api/datasets/{organ_id}/import-local")
    def import_local_dataset(
      organ_id: str,
      payload: DatasetImportLocalBody,
      _auth=Depends(require_auth),
    ) -> dict:
      from pathassist.test_datasets import get_training_spec, import_local_folder

      try:
        get_training_spec(organ_id, root=ws.root)
      except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
      folder = Path(payload.folder).expanduser() if payload.folder else None
      try:
        result = import_local_folder(
          organ_id,
          folder=folder,
          max_samples=payload.max_samples,
          root=ws.root,
          activate=payload.activate,
        )
      except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
      except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
      ws.reset_cache()
      return result

    @app.get("/api/datasets/import/{job_id}")
    def import_job_status(job_id: str) -> dict:
      from pathassist.test_datasets import get_job

      job = get_job(job_id)
      if job is None:
        raise HTTPException(404, "Unknown import job")
      if job.get("status") == "done":
        ws.reset_cache()
      return job

    @app.post("/api/run-full-test")
    def run_full_test(_auth=Depends(require_auth)) -> dict:
      if not ws.manifest.exists():
        raise HTTPException(400, "Import a holdout test set from Dataset Hub first")
      results = []
      skipped = 0
      for row in _load_samples(ws):
        label = _parse_label(row.get("label"))
        if label is None:
          skipped += 1
          continue
        path = _resolve_image_path(ws, row["image"])
        if not path.exists():
          raise HTTPException(404, f"Missing sample image: {path}")
        img = np.array(Image.open(path).convert("RGB"))
        organ = row.get("organ") or ws.get_organ_registry().default_organ
        results.append(
          _analyze_image(
            ws,
            row["case_id"],
            img,
            label=label,
            organ_id=organ,
            filename=path.name,
            image_path=path,
            allow_organ_mismatch=False,
          )
        )
      if not results:
        raise HTTPException(400, "No labeled samples in active manifest")
      correct = sum(1 for r in results if r.get("correct"))
      return {
        "total": len(results),
        "skipped_unlabeled": skipped,
        "correct": correct,
        "accuracy": round(correct / max(1, len(results)), 4),
        "validation": _validation_summary_data(ws),
        "results": results,
      }

    @app.post("/api/warm-cache")
    def warm_cache(
      organ: Optional[str] = Query(None),
      _auth=Depends(require_auth),
    ) -> dict:
      from pathassist.test_datasets import read_active_organ

      organ_id = str(organ).strip() if organ else read_active_organ(ws.root)
      rows = _load_samples(ws, organ_id)
      if not rows:
        return {"warmed": 0, "total_cached": len(_all_case_ids(ws))}
      samples_dir = _samples_dir_for(ws, organ_id)
      warmed = 0
      for row in rows:
        if not (ws.results_dir / row["case_id"] / "meta.json").exists():
          path = _resolve_image_path(ws, row["image"], samples_dir=samples_dir)
          if path.exists():
            img = np.array(Image.open(path).convert("RGB"))
            sample_organ = row.get("organ") or organ_id or ws.get_organ_registry().default_organ
            _analyze_image(
              ws,
              row["case_id"],
              img,
              label=_parse_label(row.get("label")),
              organ_id=sample_organ,
              filename=path.name,
              image_path=path,
              allow_organ_mismatch=False,
            )
            warmed += 1
      return {"warmed": warmed, "total_cached": len(_all_case_ids(ws))}

  @app.get("/api/sample-tile/{case_id}/{filename}")
  def get_sample_tile(case_id: str, filename: str) -> FileResponse:
    if not safe_case_id(case_id):
      raise HTTPException(400, "Invalid case id")
    if filename not in ("thumb.png", "source.png"):
      raise HTTPException(400, "Invalid file")
    row, samples_dir = _find_sample_row(ws, case_id)
    if row is None or samples_dir is None:
      raise HTTPException(404, "Unknown sample")
    path = _resolve_image_path(ws, row["image"], samples_dir=samples_dir)
    if not path.is_file():
      raise HTTPException(404, "Sample image not found")
    return FileResponse(path)

  @app.get("/api/image/{case_id}/{filename}")
  def get_image(case_id: str, filename: str) -> FileResponse:
    if not safe_case_id(case_id):
      raise HTTPException(400, "Invalid case id")
    allowed = safe_image_filename(filename)
    if not allowed:
      raise HTTPException(400, "Invalid file")
    path = ws.results_dir / case_id / filename
    if not path.exists():
      raise HTTPException(404, "Image not found")
    return FileResponse(path)

  app.mount("/static", StaticFiles(directory=ws.static_dir), name="static")
  return app


app = create_app()


def main() -> None:
  import uvicorn

  settings = Settings.from_env(ROOT)
  print(f"PathAssist Workstation → http://{settings.host}:{settings.port} ({settings.env})")
  uvicorn.run(
    "demo.server:app",
    host=settings.host,
    port=settings.port,
    log_level=settings.log_level.lower(),
  )


if __name__ == "__main__":
  main()
