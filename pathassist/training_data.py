"""Organ training data catalog and Hugging Face auto-download loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

TRAINING_DATASETS_NAME = "training_datasets.yaml"

_BACH_POSITIVE = {1, 2}  # InSitu, Invasive
_BACH_NEGATIVE = {0, 3}  # Benign, Normal
_NCT_TUMOR = 8
_NCT_NORMAL = 6
_LC25000_LUNG_BENIGN = {0}  # lung_n
_LC25000_LUNG_MALIGNANT = {1, 2, 3, 4}  # adeno, squamous, large-cell, etc.


def _repo_root(root: Path | None = None) -> Path:
  return Path(root or Path(__file__).resolve().parents[1])


def load_training_catalog(root: Path | None = None) -> dict[str, Any]:
  path = _repo_root(root) / "config" / TRAINING_DATASETS_NAME
  if not path.is_file():
    raise FileNotFoundError(f"Training catalog not found: {path}")
  data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
  return data.get("organs", {})


def get_training_spec(organ_id: str, root: Path | None = None) -> dict[str, Any]:
  catalog = load_training_catalog(root)
  if organ_id not in catalog:
    raise KeyError(f"No training dataset configured for organ: {organ_id!r}")
  return catalog[organ_id]


def catalog_table(root: Path | None = None) -> list[dict[str, Any]]:
  rows = []
  for organ_id, spec in sorted(load_training_catalog(root).items(), key=lambda x: x[1]["name"]):
    rows.append({
      "organ_id": organ_id,
      "name": spec["name"],
      "auto_download": bool(spec.get("auto_download")),
      "hf_dataset": spec.get("hf_dataset"),
      "hf_split": spec.get("hf_split"),
      "hf_test_split": spec.get("hf_test_split"),
      "hf_url": spec.get("hf_url"),
      "manual_url": spec.get("manual_url"),
      "tile_size": spec.get("tile_size", 256),
      "approx_tiles": spec.get("approx_tiles"),
      "task": spec.get("task", ""),
      "notes": spec.get("notes", ""),
    })
  return rows


def _as_pil_image(image) -> "Image.Image":
  import io

  from PIL import Image

  if hasattr(image, "convert"):
    return image.convert("RGB")
  if isinstance(image, dict):
    raw = image.get("bytes")
    if raw is not None:
      return Image.open(io.BytesIO(raw)).convert("RGB")
    path = image.get("path")
    if path:
      return Image.open(path).convert("RGB")
  return Image.fromarray(np.asarray(image)).convert("RGB")


def _resize_tile(image, tile_size: int) -> np.ndarray:
  from PIL import Image

  img = _as_pil_image(image)
  if img.size != (tile_size, tile_size):
    img = img.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
  return np.asarray(img, dtype=np.uint8)


def _load_patchcamelyon(
  hf_dataset: str,
  hf_split: str,
  max_samples: int,
  tile_size: int,
) -> tuple[np.ndarray, np.ndarray]:
  from datasets import load_dataset

  limit = f"[:{max_samples}]" if max_samples else ""
  ds = load_dataset(hf_dataset, split=f"{hf_split}{limit}")
  tiles = np.stack([_resize_tile(img, tile_size) for img in ds["image"]])
  labels = np.array(ds["label"], dtype=np.float32)
  return tiles, labels


def _load_bach(
  hf_dataset: str,
  hf_split: str,
  max_samples: int,
  tile_size: int,
) -> tuple[np.ndarray, np.ndarray]:
  from datasets import load_dataset

  ds = load_dataset(hf_dataset, split=hf_split)
  tiles: list[np.ndarray] = []
  labels: list[float] = []
  for row in ds:
    label = int(row["label"])
    if label in _BACH_POSITIVE:
      y = 1.0
    elif label in _BACH_NEGATIVE:
      y = 0.0
    else:
      continue
    tiles.append(_resize_tile(row["image"], tile_size))
    labels.append(y)
    if max_samples and len(tiles) >= max_samples:
      break
  return np.stack(tiles), np.asarray(labels, dtype=np.float32)


def _load_nct_crc_he(
  hf_dataset: str,
  hf_split: str,
  max_samples: int,
  tile_size: int,
) -> tuple[np.ndarray, np.ndarray]:
  from datasets import load_dataset

  ds = load_dataset(hf_dataset, split=hf_split, streaming=True)
  ds = ds.shuffle(seed=0, buffer_size=10_000)

  per_class = max_samples // 2 if max_samples else 50_000
  norm_tiles: list[np.ndarray] = []
  tum_tiles: list[np.ndarray] = []
  scanned = 0
  for row in ds:
    scanned += 1
    label = int(row["label"])
    if label == _NCT_NORMAL and len(norm_tiles) < per_class:
      norm_tiles.append(_resize_tile(row["image"], tile_size))
    elif label == _NCT_TUMOR and len(tum_tiles) < per_class:
      tum_tiles.append(_resize_tile(row["image"], tile_size))
    if len(norm_tiles) >= per_class and len(tum_tiles) >= per_class:
      break

  if not norm_tiles or not tum_tiles:
    raise ValueError(
      "No balanced TUM/NORM tiles from NCT-CRC-HE "
      f"(norm={len(norm_tiles)}, tum={len(tum_tiles)}, scanned={scanned})."
    )

  tiles = np.stack(norm_tiles + tum_tiles)
  labels = np.array([0.0] * len(norm_tiles) + [1.0] * len(tum_tiles), dtype=np.float32)
  perm = np.random.default_rng(0).permutation(len(labels))
  return tiles[perm], labels[perm]


def _load_lc25000_lung(
  hf_dataset: str,
  hf_split: str,
  max_samples: int,
  tile_size: int,
) -> tuple[np.ndarray, np.ndarray]:
  from datasets import load_dataset

  ds = load_dataset(hf_dataset, split=hf_split, streaming=True)
  per_class = max(1, max_samples // 2) if max_samples else 0
  benign_tiles: list[np.ndarray] = []
  malig_tiles: list[np.ndarray] = []
  for row in ds:
    if int(row["organ"]) != 0:
      continue
    label = int(row["label"])
    tile = _resize_tile(row["image"], tile_size)
    if label in _LC25000_LUNG_BENIGN and len(benign_tiles) < per_class:
      benign_tiles.append(tile)
    elif label in _LC25000_LUNG_MALIGNANT and len(malig_tiles) < per_class:
      malig_tiles.append(tile)
    if per_class and len(benign_tiles) >= per_class and len(malig_tiles) >= per_class:
      break
  if not benign_tiles or not malig_tiles:
    raise ValueError(
      "No balanced lung tiles from LC25000 "
      f"(benign={len(benign_tiles)}, malig={len(malig_tiles)})."
    )
  tiles = np.stack(benign_tiles + malig_tiles)
  labels = np.array(
    [0.0] * len(benign_tiles) + [1.0] * len(malig_tiles),
    dtype=np.float32,
  )
  perm = np.random.default_rng(0).permutation(len(labels))
  return tiles[perm], labels[perm]


_LOADERS = {
  "patchcamelyon": _load_patchcamelyon,
  "bach": _load_bach,
  "nct_crc_he": _load_nct_crc_he,
  "lc25000_lung": _load_lc25000_lung,
}


def load_organ_training_data(
  organ_id: str,
  *,
  max_samples: int | None = 32768,
  tile_size: int | None = None,
  data_dir: str | Path | None = None,
  root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
  """Load (tiles, labels) for training one organ model.

  Returns metadata describing which dataset was used.
  """
  spec = get_training_spec(organ_id, root=root)
  tile_size = int(tile_size or spec.get("tile_size", 256))
  meta = {
    "organ_id": organ_id,
    "name": spec["name"],
    "auto_download": bool(spec.get("auto_download")),
    "hf_dataset": spec.get("hf_dataset"),
    "hf_split": spec.get("hf_split"),
    "hf_url": spec.get("hf_url"),
    "manual_url": spec.get("manual_url"),
    "tile_size": tile_size,
    "task": spec.get("task", ""),
  }

  if spec.get("auto_download"):
    loader_name = str(spec.get("loader", ""))
    loader = _LOADERS.get(loader_name)
    if loader is None:
      raise ValueError(f"No auto loader implemented for {organ_id!r} ({loader_name})")
    tiles, labels = loader(
      spec["hf_dataset"],
      spec.get("hf_split", "train"),
      max_samples or 0,
      tile_size,
    )
    meta["source"] = "huggingface"
    meta["loader"] = loader_name
    return tiles, labels, meta

  if not data_dir:
    raise ValueError(
      f"{spec['name']} requires manual tiles. Set DATA_DIR with benign/ and malignant/ "
      f"subfolders. See: {spec.get('manual_url', 'config/training_datasets.yaml')}"
    )
  from .data import load_from_image_folder

  tiles, labels = load_from_image_folder(data_dir, tile_size=tile_size)
  if max_samples and len(tiles) > max_samples:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(tiles), max_samples, replace=False)
    tiles, labels = tiles[idx], labels[idx]
  meta["source"] = "folder"
  meta["data_dir"] = str(data_dir)
  return tiles, labels, meta
