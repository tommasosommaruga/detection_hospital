"""Whole-slide tiling.

Real whole-slide images (.svs, .ndpi) are gigapixel and must be read region by
region with a library such as OpenSlide. To keep the scaffold runnable anywhere
(including a plain Colab CPU runtime with no OpenSlide), this module operates on
an in-memory image array and exposes a single `tile_image` entry point. Swapping
in an OpenSlide-backed reader later only means feeding tiles from disk into the
same downstream stages - the interface below does not change.
"""

from __future__ import annotations

import numpy as np

from .types import Tile


def load_image(path: str) -> np.ndarray:
  """Load a standard image file into an (H, W, 3) uint8 array.

  For real WSIs replace this with an OpenSlide region reader; every downstream
  stage only sees the tiles, so nothing else needs to change.
  """
  from PIL import Image

  with Image.open(path) as img:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _is_background(tile_pixels: np.ndarray, cutoff: int) -> bool:
  """A tile is treated as background when it is mostly bright/empty glass."""
  return float(tile_pixels.mean()) >= cutoff


def tile_image(
  image: np.ndarray,
  tile_size: int,
  overlap: float = 0.0,
  background_intensity_cutoff: int = 220,
) -> list[tuple[Tile, np.ndarray]]:
  """Cut an image into tiles, dropping near-white background tiles.

  Returns a list of (Tile metadata, tile pixel array) pairs. Coordinates are in
  pixels relative to the top-left of the input image.
  """
  if tile_size <= 0:
    raise ValueError("tile_size must be positive")
  if not 0.0 <= overlap < 1.0:
    raise ValueError("overlap must be in the range [0.0, 1.0)")

  height, width = image.shape[:2]
  stride = max(1, int(round(tile_size * (1.0 - overlap))))

  tiles: list[tuple[Tile, np.ndarray]] = []
  index = 0
  for y in range(0, max(1, height - tile_size + 1), stride):
    for x in range(0, max(1, width - tile_size + 1), stride):
      pixels = image[y : y + tile_size, x : x + tile_size]
      if pixels.shape[0] != tile_size or pixels.shape[1] != tile_size:
        continue
      if _is_background(pixels, background_intensity_cutoff):
        continue
      tile = Tile(index=index, x=x, y=y, width=tile_size, height=tile_size)
      tiles.append((tile, pixels))
      index += 1
  return tiles
