"""Synthetic slide generation for development and tests.

Generates a fake H&E-like slide: a light tissue background with a few darker,
denser "cell clusters". This is purely so the pipeline can be exercised without
any real data or model weights. It is not a substitute for real slides.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_synthetic_slide(
  height: int = 1024,
  width: int = 1024,
  num_clusters: int = 6,
  seed: int = 0,
) -> np.ndarray:
  """Create a synthetic (H, W, 3) uint8 slide with dark clustered regions."""
  rng = np.random.default_rng(seed)

  # Light pinkish tissue background with mild noise.
  background = np.full((height, width, 3), fill_value=235, dtype=np.float32)
  background[..., 1] -= 15  # slightly less green -> pinkish
  background += rng.normal(0.0, 6.0, size=background.shape)

  yy, xx = np.mgrid[0:height, 0:width]
  for _ in range(num_clusters):
    cy = rng.integers(0, height)
    cx = rng.integers(0, width)
    radius = rng.integers(40, 120)
    strength = rng.uniform(60, 130)
    dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
    blob = np.exp(-dist_sq / (2.0 * radius**2)) * strength
    background -= blob[..., None]  # darken (denser nuclei)

  return np.clip(background, 0, 255).astype(np.uint8)


def make_tile_dataset(
  num_samples: int = 400,
  tile_size: int = 64,
  seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
  """Create a labelled synthetic tile dataset for training/testing.

  Returns (tiles, labels) where tiles is (N, tile_size, tile_size, 3) uint8 and
  labels is (N,) float32 in {0.0, 1.0}. Positive ("malignant") tiles are darker
  and denser; negatives are light tissue. This is a learnable toy signal, not a
  clinical one - replace with real annotated tiles when you have them.
  """
  rng = np.random.default_rng(seed)
  tiles = np.empty((num_samples, tile_size, tile_size, 3), dtype=np.uint8)
  labels = np.empty((num_samples,), dtype=np.float32)

  for i in range(num_samples):
    is_positive = bool(rng.integers(0, 2))
    base_level = 150.0 if is_positive else 225.0
    noise_scale = 25.0 if is_positive else 12.0
    tile = np.full((tile_size, tile_size, 3), base_level, dtype=np.float32)
    tile[..., 1] -= 15  # pinkish tint
    tile += rng.normal(0.0, noise_scale, size=tile.shape)
    if is_positive:
      # Add a few dark nuclei blobs.
      yy, xx = np.mgrid[0:tile_size, 0:tile_size]
      for _ in range(rng.integers(3, 7)):
        cy, cx = rng.integers(0, tile_size), rng.integers(0, tile_size)
        radius = rng.integers(4, 10)
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * radius**2)) * 90.0
        tile -= blob[..., None]
    tiles[i] = np.clip(tile, 0, 255).astype(np.uint8)
    labels[i] = 1.0 if is_positive else 0.0

  return tiles, labels


def export_tile_folder(
  root: str | Path,
  num_samples: int = 400,
  tile_size: int = 64,
  seed: int = 0,
) -> Path:
  """Write a synthetic labelled dataset to disk as an image folder.

  Produces `root/benign/*.png` and `root/malignant/*.png` so you can exercise the
  real-data loader (`load_from_image_folder`) end to end before you have real
  slides. Returns the root path.
  """
  from PIL import Image

  root = Path(root)
  benign_dir = root / "benign"
  malignant_dir = root / "malignant"
  benign_dir.mkdir(parents=True, exist_ok=True)
  malignant_dir.mkdir(parents=True, exist_ok=True)

  tiles, labels = make_tile_dataset(num_samples=num_samples, tile_size=tile_size, seed=seed)
  for i, (tile, label) in enumerate(zip(tiles, labels)):
    target_dir = malignant_dir if label >= 0.5 else benign_dir
    Image.fromarray(tile).save(target_dir / f"tile_{i:05d}.png")
  return root
