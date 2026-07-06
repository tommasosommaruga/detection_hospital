"""Holdout test-tile import for workstation benchmarking.

Imports balanced samples from public holdout splits only (never training splits).
"""

from __future__ import annotations

import csv
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from pathassist.dataset_similarity import CATALOG as SIMILARITY_CATALOG
from pathassist.training_data import (
  TRAINING_DATASETS_NAME,
  _BACH_NEGATIVE,
  _BACH_POSITIVE,
  _LC25000_LUNG_BENIGN,
  _LC25000_LUNG_MALIGNANT,
  _NCT_NORMAL,
  _NCT_TUMOR,
  _repo_root,
  _resize_tile,
  catalog_table,
  get_training_spec,
)

MANIFEST_FIELDS = [
  "case_id",
  "image",
  "label",
  "organ",
  "dataset_id",
  "split",
]

ORGAN_SIMILARITY_ID: dict[str, str] = {
  "lymph_node": "patchcamelyon",
  "breast": "bach",
  "gastrointestinal": "nct_crc_he",
}

LABEL_NAMES: dict[str, tuple[str, str]] = {
  "lymph_node": ("normal", "metastasis"),
  "breast": ("benign", "malignant"),
  "gastrointestinal": ("normal mucosa", "tumor"),
  "pulmonary": ("benign", "malignant"),
}

DEFAULT_IMPORT_COUNT = 20
MAX_IMPORT_COUNT = 200

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def test_datasets_root(root: Path | None = None) -> Path:
  return _repo_root(root) / "outputs" / "test_datasets"


def local_benchmarks_root(root: Path | None = None) -> Path:
  """User-owned tiles for the demo — not downloaded from Hugging Face."""
  return _repo_root(root) / "data" / "benchmarks"


def local_organ_dir(organ_id: str, root: Path | None = None) -> Path:
  return local_benchmarks_root(root) / organ_id


def active_state_path(root: Path | None = None) -> Path:
  return test_datasets_root(root) / "active.json"


def dataset_dir(root: Path | None, organ_id: str) -> Path:
  return test_datasets_root(root) / organ_id


def import_meta_path(root: Path | None, organ_id: str) -> Path:
  return dataset_dir(root, organ_id) / "import_meta.json"


def manifest_path(root: Path | None, organ_id: str) -> Path:
  return dataset_dir(root, organ_id) / "manifest.csv"


def read_active_organ(root: Path | None = None) -> str | None:
  path = active_state_path(root)
  if not path.is_file():
    return None
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except json.JSONDecodeError:
    return None
  organ_id = data.get("organ_id")
  return str(organ_id) if organ_id else None


def read_import_meta(root: Path | None, organ_id: str) -> dict[str, Any] | None:
  path = import_meta_path(root, organ_id)
  if not path.is_file():
    return None
  return json.loads(path.read_text(encoding="utf-8"))


def label_names_for_organ(organ_id: str, root: Path | None = None) -> tuple[str, str]:
  meta = read_import_meta(root, organ_id)
  if meta and "label_names" in meta:
    names = meta["label_names"]
    return str(names["0"]), str(names["1"])
  return LABEL_NAMES.get(organ_id, ("negative", "positive"))


def set_active_dataset(organ_id: str, root: Path | None = None) -> Path:
  out_dir = dataset_dir(root, organ_id)
  if not (out_dir / "manifest.csv").is_file():
    raise FileNotFoundError(f"No imported manifest for organ {organ_id!r}")
  if _existing_sample_count(root, organ_id) < 1:
    raise FileNotFoundError(f"No imported samples for organ {organ_id!r}")
  test_datasets_root(root).mkdir(parents=True, exist_ok=True)
  active_state_path(root).write_text(
    json.dumps({"organ_id": organ_id, "updated_at": _utc_now()}, indent=2),
    encoding="utf-8",
  )
  return out_dir


def resolve_samples_dir(root: Path | None = None) -> Path:
  repo = _repo_root(root)
  active = read_active_organ(root)
  if active:
    active_dir = dataset_dir(root, active)
    if (active_dir / "manifest.csv").is_file():
      return active_dir
  legacy = repo / "outputs" / "real_pcam"
  if (legacy / "manifest.csv").is_file():
    return legacy
  if active:
    return dataset_dir(root, active)
  return legacy


def import_status(root: Path | None = None) -> dict[str, dict[str, Any]]:
  status: dict[str, dict[str, Any]] = {}
  root_dir = test_datasets_root(root)
  if not root_dir.is_dir():
    return status
  for path in root_dir.iterdir():
    if not path.is_dir():
      continue
    manifest = path / "manifest.csv"
    if not manifest.is_file():
      continue
    organ_id = path.name
    with manifest.open(encoding="utf-8") as handle:
      rows = list(csv.DictReader(handle))
    meta = read_import_meta(root, organ_id) or {}
    status[organ_id] = {
      "organ_id": organ_id,
      "sample_count": len(rows),
      "labeled_count": sum(1 for r in rows if str(r.get("label", "")).strip() != ""),
      "split": meta.get("split"),
      "imported_at": meta.get("imported_at"),
      "is_active": read_active_organ(root) == organ_id,
    }
  return status


def local_folder_tile_count(organ_id: str, root: Path | None = None) -> int:
  """Count image files under data/benchmarks/<organ>/ class subfolders."""
  from pathassist.data import IMAGE_EXTENSIONS

  base = local_organ_dir(organ_id, root)
  if not base.is_dir():
    return 0
  n = 0
  for sub in base.iterdir():
    if not sub.is_dir():
      continue
    n += sum(1 for p in sub.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
  return n


def hub_catalog(root: Path | None = None) -> dict[str, Any]:
  sim_by_id = {d.id: d for d in SIMILARITY_CATALOG}
  statuses = import_status(root)
  active = read_active_organ(root)
  rows = []
  for spec in catalog_table(root):
    organ_id = spec["organ_id"]
    sim_id = ORGAN_SIMILARITY_ID.get(organ_id)
    sim = sim_by_id.get(sim_id) if sim_id else None
    st = statuses.get(organ_id, {})
    neg, pos = LABEL_NAMES.get(organ_id, ("class 0", "class 1"))
    rows.append({
      **spec,
      "similarity_id": sim_id,
      "composite_similarity": sim.composite_similarity if sim else None,
      "stain_similarity": sim.stain_similarity if sim else None,
      "access_notes": sim.access if sim else None,
      "importable": bool(spec.get("auto_download")),
      "local_dir": str(local_organ_dir(organ_id, root)),
      "local_tile_count": local_folder_tile_count(organ_id, root),
      "label_names": {"0": neg, "1": pos},
      "hf_test_split": spec.get("hf_test_split") or spec.get("hf_split"),
      "import_status": st or None,
      "is_active": active == organ_id,
    })
  rows.sort(
    key=lambda r: (
      0 if r.get("is_active") else 1,
      0 if r.get("import_status") else 1,
      -(r.get("composite_similarity") or 0),
      r["name"],
    ),
  )
  return {
    "reference": "patchcamelyon",
    "active_organ": active,
    "active_samples_dir": str(resolve_samples_dir(root)),
    "local_benchmarks_root": str(local_benchmarks_root(root)),
    "default_import_count": DEFAULT_IMPORT_COUNT,
    "datasets": rows,
  }


def _utc_now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _organ_prefix(organ_id: str) -> str:
  return {
    "lymph_node": "PCAM",
    "breast": "BACH",
    "gastrointestinal": "GI",
    "pulmonary": "LUNG",
  }.get(organ_id, organ_id[:4].upper())


def _balanced_per_class(count: int) -> int:
  return max(1, count // 2)


def _write_manifest(
  out_dir: Path,
  rows: list[dict[str, Any]],
  organ_id: str,
  split: str,
  meta_extra: dict[str, Any] | None = None,
) -> None:
  images_dir = out_dir / "images"
  images_dir.mkdir(parents=True, exist_ok=True)
  neg, pos = LABEL_NAMES.get(organ_id, ("neg", "pos"))
  manifest_rows = []
  for row in rows:
    fname = row["filename"]
    rel = f"images/{fname}"
    manifest_rows.append({
      "case_id": row["case_id"],
      "image": rel,
      "label": row["label"],
      "organ": organ_id,
      "dataset_id": organ_id,
      "split": split,
    })

  with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
    writer.writeheader()
    writer.writerows(manifest_rows)

  meta = {
    "organ_id": organ_id,
    "split": split,
    "label_names": {"0": neg, "1": pos},
    "imported_at": _utc_now(),
    "n_samples": len(manifest_rows),
    **(meta_extra or {}),
  }
  loader = (meta_extra or {}).get("loader")
  if split == "train" and loader == "lc25000_lung":
    meta["holdout_note"] = "LC25000 has no public test split; stratified sample from train stream."
  elif split == "train" and loader == "bach":
    meta["holdout_note"] = "BACH HF test split is unlabeled; stratified sample from train."
  (out_dir / "import_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _save_rows_to_dir(
  out_dir: Path,
  organ_id: str,
  split: str,
  tiles: list[np.ndarray],
  labels: list[int],
  class_tags: tuple[str, str],
) -> list[dict[str, Any]]:
  prefix = _organ_prefix(organ_id)
  images_dir = out_dir / "images"
  images_dir.mkdir(parents=True, exist_ok=True)
  saved: list[dict[str, Any]] = []
  counters = {0: 0, 1: 0}
  for tile, label in zip(tiles, labels):
    idx = counters[label]
    counters[label] += 1
    tag = class_tags[0].upper().replace(" ", "")[:4] if label == 0 else class_tags[1].upper().replace(" ", "")[:4]
    fname = f"{prefix.lower()}_{tag.lower()}_{idx:02d}.png"
    Image.fromarray(tile).save(images_dir / fname)
    saved.append({
      "case_id": f"{prefix}-{tag[:3]}-{idx:02d}",
      "filename": fname,
      "label": label,
    })
  return saved


def _collect_patchcamelyon(spec: dict[str, Any], split: str, per_class: int, tile_size: int) -> tuple[list, list]:
  from datasets import load_dataset

  ds = load_dataset(spec["hf_dataset"], split=split)
  labels = np.array(ds["label"])
  tiles, y = [], []
  rng = np.random.default_rng(42)
  for label in (0, 1):
    idx = np.where(labels == label)[0]
    rng.shuffle(idx)
    for row_idx in idx[:per_class]:
      tiles.append(_resize_tile(ds[int(row_idx)]["image"], tile_size))
      y.append(label)
  return tiles, y


def _collect_bach(spec: dict[str, Any], split: str, per_class: int, tile_size: int) -> tuple[list, list]:
  from datasets import load_dataset

  ds = load_dataset(spec["hf_dataset"], split=split)
  pos_idx, neg_idx = [], []
  for i, row in enumerate(ds):
    label = int(row["label"])
    if label in _BACH_POSITIVE:
      pos_idx.append(i)
    elif label in _BACH_NEGATIVE:
      neg_idx.append(i)
  rng = np.random.default_rng(42)
  rng.shuffle(pos_idx)
  rng.shuffle(neg_idx)
  if len(pos_idx) < per_class or len(neg_idx) < per_class:
    raise ValueError(
      f"BACH split {split!r}: need {per_class} per class, "
      f"got neg={len(neg_idx)} pos={len(pos_idx)}"
    )
  tiles, y = [], []
  for idx in neg_idx[:per_class]:
    tiles.append(_resize_tile(ds[int(idx)]["image"], tile_size))
    y.append(0)
  for idx in pos_idx[:per_class]:
    tiles.append(_resize_tile(ds[int(idx)]["image"], tile_size))
    y.append(1)
  return tiles, y


def _collect_nct_crc(spec: dict[str, Any], split: str, per_class: int, tile_size: int) -> tuple[list, list]:
  from datasets import load_dataset

  ds = load_dataset(spec["hf_dataset"], split=split, streaming=True)
  norm, tum = [], []
  for row in ds:
    label = int(row["label"])
    tile = _resize_tile(row["image"], tile_size)
    if label == _NCT_NORMAL and len(norm) < per_class:
      norm.append((tile, 0))
    elif label == _NCT_TUMOR and len(tum) < per_class:
      tum.append((tile, 1))
    if len(norm) >= per_class and len(tum) >= per_class:
      break
  if len(norm) < per_class or len(tum) < per_class:
    raise ValueError(f"Could not collect balanced NCT tiles from split {split!r}")
  pairs = norm + tum
  return [p[0] for p in pairs], [p[1] for p in pairs]


def _collect_lc25000_lung(spec: dict[str, Any], split: str, per_class: int, tile_size: int) -> tuple[list, list]:
  from datasets import load_dataset

  ds = load_dataset(spec["hf_dataset"], split=split, streaming=True)
  benign, malig = [], []
  for row in ds:
    if int(row["organ"]) != 0:
      continue
    label = int(row["label"])
    tile = _resize_tile(row["image"], tile_size)
    if label in _LC25000_LUNG_BENIGN and len(benign) < per_class:
      benign.append((tile, 0))
    elif label in _LC25000_LUNG_MALIGNANT and len(malig) < per_class:
      malig.append((tile, 1))
    if len(benign) >= per_class and len(malig) >= per_class:
      break
  pairs = benign + malig
  if len(pairs) < per_class * 2:
    raise ValueError(f"Could not collect balanced lung tiles from split {split!r}")
  return [p[0] for p in pairs], [p[1] for p in pairs]


_COLLECTORS: dict[str, Callable[..., tuple[list, list]]] = {
  "patchcamelyon": _collect_patchcamelyon,
  "bach": _collect_bach,
  "nct_crc_he": _collect_nct_crc,
  "lc25000_lung": _collect_lc25000_lung,
}


def _existing_sample_count(root: Path | None, organ_id: str) -> int:
  """Tiles on disk with manifest rows and image files present."""
  manifest = manifest_path(root, organ_id)
  if not manifest.is_file():
    return 0
  out_dir = dataset_dir(root, organ_id)
  count = 0
  with manifest.open(encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
      if (out_dir / row["image"]).is_file():
        count += 1
  return count


def import_test_samples(
  organ_id: str,
  *,
  count: int = DEFAULT_IMPORT_COUNT,
  root: Path | None = None,
  activate: bool = True,
  force: bool = False,
) -> dict[str, Any]:
  """Import balanced holdout tiles for one organ. Returns import summary.

  Cached on disk: if enough tiles were already imported, reuse them instead of
  hitting the dataset API again. Pass ``force=True`` to re-download.
  """
  count = max(2, min(int(count), MAX_IMPORT_COUNT))
  if count % 2:
    count -= 1
  per_class = _balanced_per_class(count)

  spec = get_training_spec(organ_id, root=root)
  if not spec.get("auto_download"):
    raise ValueError(
      f"{spec['name']} requires manual tiles — use the manual URL and folder import (not wired yet)."
    )

  split = str(spec.get("hf_test_split") or spec.get("hf_split") or "test")

  existing = _existing_sample_count(root, organ_id)
  if not force and existing >= count:
    if activate:
      set_active_dataset(organ_id, root=root)
    return {
      "organ_id": organ_id,
      "split": split,
      "sample_count": existing,
      "per_class": per_class,
      "output_dir": str(dataset_dir(root, organ_id)),
      "activated": activate,
      "cached": True,
    }

  loader = str(spec.get("loader", ""))
  collector = _COLLECTORS.get(loader)
  if collector is None:
    raise ValueError(f"No holdout importer for loader {loader!r}")

  tile_size = int(spec.get("tile_size", 256))
  tiles, labels = collector(spec, split, per_class, tile_size)
  if not tiles:
    raise ValueError(
      f"No labeled tiles collected for {spec['name']} (split={split!r}). "
      "Re-download from Dataset Hub or check the dataset catalog."
    )

  out_dir = dataset_dir(root, organ_id)
  if out_dir.exists():
    for child in out_dir.iterdir():
      if child.is_dir():
        import shutil
        shutil.rmtree(child)
      else:
        child.unlink()
  out_dir.mkdir(parents=True, exist_ok=True)

  class_tags = LABEL_NAMES.get(organ_id, ("negative", "positive"))
  saved = _save_rows_to_dir(out_dir, organ_id, split, tiles, labels, class_tags)
  _write_manifest(out_dir, saved, organ_id, split, {"loader": loader, "hf_dataset": spec.get("hf_dataset")})

  if activate:
    set_active_dataset(organ_id, root=root)

  return {
    "organ_id": organ_id,
    "split": split,
    "sample_count": len(saved),
    "per_class": per_class,
    "output_dir": str(out_dir),
    "activated": activate,
    "cached": False,
  }


def import_local_folder(
  organ_id: str,
  *,
  folder: Path | str | None = None,
  max_samples: int | None = None,
  root: Path | None = None,
  activate: bool = True,
) -> dict[str, Any]:
  """Register tiles from ``data/benchmarks/<organ>/benign|malignant/`` for the demo.

  Copies resized tiles into ``outputs/test_datasets/<organ>/`` (the demo cache).
  No Hugging Face or external API calls.
  """
  from pathassist.data import load_from_image_folder

  spec = get_training_spec(organ_id, root=root)
  tile_size = int(spec.get("tile_size", 256))
  source = Path(folder) if folder else local_organ_dir(organ_id, root=root)
  if not source.is_dir():
    raise FileNotFoundError(
      f"Local folder not found: {source}\n"
      f"Create {source}/benign/ and {source}/malignant/ and add tile images."
    )

  tiles, labels = load_from_image_folder(source, tile_size=tile_size)
  labels_list = labels.astype(int).tolist()

  if max_samples and len(labels_list) > max_samples:
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(labels_list))[:max_samples]
    tiles = tiles[idx]
    labels_list = [labels_list[i] for i in idx]

  out_dir = dataset_dir(root, organ_id)
  if out_dir.exists():
    import shutil

    for child in out_dir.iterdir():
      if child.is_dir():
        shutil.rmtree(child)
      else:
        child.unlink()
  out_dir.mkdir(parents=True, exist_ok=True)

  class_tags = LABEL_NAMES.get(organ_id, ("negative", "positive"))
  saved = _save_rows_to_dir(out_dir, organ_id, "local", tiles, labels_list, class_tags)
  _write_manifest(
    out_dir,
    saved,
    organ_id,
    "local",
    {"loader": "local_folder", "source_dir": str(source.resolve())},
  )

  if activate:
    set_active_dataset(organ_id, root=root)

  return {
    "organ_id": organ_id,
    "split": "local",
    "sample_count": len(saved),
    "output_dir": str(out_dir),
    "source_dir": str(source.resolve()),
    "activated": activate,
    "cached": False,
  }


def get_job(job_id: str) -> dict[str, Any] | None:
  with _jobs_lock:
    job = _jobs.get(job_id)
    return dict(job) if job else None


def start_import_job(
  organ_id: str,
  *,
  count: int = DEFAULT_IMPORT_COUNT,
  root: Path | None = None,
  activate: bool = True,
  force: bool = False,
) -> str:
  job_id = uuid.uuid4().hex[:12]
  with _jobs_lock:
    _jobs[job_id] = {
      "job_id": job_id,
      "status": "running",
      "organ_id": organ_id,
      "started_at": _utc_now(),
    }

  def _run() -> None:
    try:
      result = import_test_samples(
        organ_id,
        count=count,
        root=root,
        activate=activate,
        force=force,
      )
      with _jobs_lock:
        _jobs[job_id].update({"status": "done", "finished_at": _utc_now(), "result": result})
    except Exception as exc:
      with _jobs_lock:
        _jobs[job_id].update({
          "status": "error",
          "finished_at": _utc_now(),
          "error": str(exc),
        })

  threading.Thread(target=_run, daemon=True).start()
  return job_id
