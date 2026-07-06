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
from pathlib import Path

from .types import Tile
from .wsi import WsiScanInfo, is_wsi_path, load_wsi_thumbnail, openslide_available, prepare_wsi_tiles


def load_image(path: str) -> np.ndarray:
  """Load a standard image or WSI thumbnail into an (H, W, 3) uint8 array."""
  if is_wsi_path(path):
    return load_wsi_thumbnail(path)
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
  force_whole_image: bool = False,
) -> list[tuple[Tile, np.ndarray]]:
  """Cut an image into tiles, dropping near-white background tiles.

  Returns a list of (Tile metadata, tile pixel array) pairs. Coordinates are in
  pixels relative to the top-left of the input image.

  When ``force_whole_image`` is true and the image fits in one tile (e.g. 96×96
  PatchCamelyon patches), the entire image is scored as a single tile even if it
  looks background-heavy — the patch itself is the clinical sample.
  """
  if tile_size <= 0:
    raise ValueError("tile_size must be positive")
  if not 0.0 <= overlap < 1.0:
    raise ValueError("overlap must be in the range [0.0, 1.0)")

  height, width = image.shape[:2]
  if force_whole_image and height <= tile_size and width <= tile_size:
    tile = Tile(index=0, x=0, y=0, width=width, height=height)
    return [(tile, image)]

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


def should_use_wsi_full_res(image_path: str | Path | None, tiling_cfg: dict) -> bool:
  """True when we can tile a WSI at model MPP instead of thumbnail-only."""
  if not tiling_cfg.get("wsi_full_res", False):
    return False
  if image_path is None:
    return False
  return is_wsi_path(image_path) and openslide_available()


def resolve_case_tiles(
  image: np.ndarray | None,
  config: dict,
  *,
  image_path: str | Path | None = None,
) -> tuple[np.ndarray, list[tuple[Tile, np.ndarray]], WsiScanInfo | None]:
  """Choose patch tiling or full-res WSI tiling for one case."""
  tiling_cfg = config["tiling"]
  path = Path(image_path) if image_path else None

  if should_use_wsi_full_res(path, tiling_cfg):
    assert path is not None
    thumbnail, tiles, info = prepare_wsi_tiles(path, tiling_cfg)
    return thumbnail, tiles, info

  if image is None:
    if path is None:
      raise ValueError("Either image or image_path is required")
    image = load_image(str(path))

  tiles = tile_image(
    image,
    tile_size=int(tiling_cfg["tile_size"]),
    overlap=float(tiling_cfg.get("overlap", 0.0)),
    background_intensity_cutoff=int(tiling_cfg.get("background_intensity_cutoff", 220)),
    force_whole_image=bool(tiling_cfg.get("force_whole_image", False)),
  )
  return image, tiles, None
