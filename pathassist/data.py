"""Real tile-dataset loaders.

Two ways to bring your own annotated tiles into training, both returning the same
`(tiles, labels)` pair the trainer expects:

  * `load_from_image_folder` - one subfolder per class:
        data/tiles/benign/*.png
        data/tiles/malignant/*.png
  * `load_from_csv` - a CSV listing image paths and labels:
        path,label
        tiles/img_0001.png,malignant
        tiles/img_0002.png,benign

Both resize every tile to a common `tile_size` so batching is uniform, and both
return `(N, tile_size, tile_size, 3)` uint8 tiles plus `(N,)` float32 labels in
{0.0, 1.0}. Swapping synthetic data for real data is then a one-line change in
the trainer or notebook - nothing downstream needs to know.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

# File extensions we treat as loadable tile images.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _load_one_tile(path: str | Path, tile_size: int) -> np.ndarray:
  """Load a single image, convert to RGB and resize to (tile_size, tile_size)."""
  from PIL import Image

  with Image.open(path) as img:
    resized = img.convert("RGB").resize((tile_size, tile_size))
    return np.asarray(resized, dtype=np.uint8)


def _coerce_label(raw: str, positive_labels: set[str] | None) -> float:
  """Turn a raw label cell into 0.0/1.0.

  If `positive_labels` is given, any value in that set is positive (1.0). Else the
  value is parsed as a number and thresholded at >= 0.5 (so 0/1 or probabilities
  both work).
  """
  value = str(raw).strip()
  if positive_labels is not None:
    return 1.0 if value.lower() in {p.lower() for p in positive_labels} else 0.0
  try:
    return 1.0 if float(value) >= 0.5 else 0.0
  except ValueError as exc:
    raise ValueError(
      f"Label {value!r} is not numeric; pass positive_labels to map class names."
    ) from exc


def _stack(tiles: list[np.ndarray], labels: list[float]) -> tuple[np.ndarray, np.ndarray]:
  if not tiles:
    raise ValueError("No tiles were loaded; check the path, extensions and labels.")
  return np.stack(tiles, axis=0), np.asarray(labels, dtype=np.float32)


def load_from_image_folder(
  root: str | Path,
  tile_size: int = 64,
  class_to_label: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Load tiles from a folder with one subdirectory per class.

  By convention 'benign'/'normal'/'negative' -> 0 and 'malignant'/'tumor'/
  'positive' -> 1. Override with `class_to_label` (e.g. {"tumor": 1, "normal": 0})
  to control the mapping or restrict which subfolders are used.
  """
  root = Path(root)
  if not root.is_dir():
    raise NotADirectoryError(f"Not a directory: {root}")

  default_map = {
    "benign": 0.0, "normal": 0.0, "negative": 0.0, "0": 0.0,
    "malignant": 1.0, "tumor": 1.0, "tumour": 1.0, "positive": 1.0, "1": 1.0,
  }
  mapping = class_to_label if class_to_label is not None else default_map

  tiles: list[np.ndarray] = []
  labels: list[float] = []
  for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    key = class_dir.name.lower()
    if key not in mapping:
      continue  # skip subfolders that aren't mapped to a class
    label = float(mapping[key])
    for image_path in sorted(class_dir.iterdir()):
      if image_path.suffix.lower() in IMAGE_EXTENSIONS:
        tiles.append(_load_one_tile(image_path, tile_size))
        labels.append(label)

  return _stack(tiles, labels)


def load_from_csv(
  csv_path: str | Path,
  image_root: str | Path | None = None,
  tile_size: int = 64,
  path_column: str = "path",
  label_column: str = "label",
  positive_labels: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Load tiles listed in a CSV file.

  `image_root` is prepended to relative paths in the CSV (defaults to the CSV's
  own directory). `positive_labels` maps class-name labels to 1.0; omit it if the
  label column is already numeric (0/1 or probabilities).
  """
  csv_path = Path(csv_path)
  if not csv_path.is_file():
    raise FileNotFoundError(f"CSV not found: {csv_path}")
  base = Path(image_root) if image_root is not None else csv_path.parent
  positive_set = set(positive_labels) if positive_labels is not None else None

  tiles: list[np.ndarray] = []
  labels: list[float] = []
  with csv_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None or path_column not in reader.fieldnames:
      raise ValueError(f"CSV must have a '{path_column}' column; got {reader.fieldnames}")
    if label_column not in reader.fieldnames:
      raise ValueError(f"CSV must have a '{label_column}' column; got {reader.fieldnames}")
    for row in reader:
      rel = row[path_column].strip()
      if not rel:
        continue
      image_path = (base / rel) if not Path(rel).is_absolute() else Path(rel)
      tiles.append(_load_one_tile(image_path, tile_size))
      labels.append(_coerce_label(row[label_column], positive_set))

  return _stack(tiles, labels)
