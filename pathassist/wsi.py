"""Whole-slide image loading and full-resolution tiling (OpenSlide when available)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import Tile

WSI_EXTENSIONS = {".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".scn"}

# PatchCamelyon patches ≈ 10× from 40× (0.243 µm/px at 40×).
DEFAULT_LYMPH_TARGET_MPP = 0.972
# NCT-CRC-HE patches are 224 px at 0.5 µm/px.
DEFAULT_GI_TARGET_MPP = 0.5


@dataclass(frozen=True)
class WsiScanInfo:
  """Metadata for a full-resolution WSI scan (serialisable into case meta.json)."""

  path: str
  level: int
  level_mpp: float
  target_mpp: float
  level0_width: int
  level0_height: int
  thumbnail_width: int
  thumbnail_height: int
  tile_size: int
  tiles_scored: int
  tiles_skipped_background: int

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


def is_wsi_path(path: str | Path) -> bool:
  return Path(path).suffix.lower() in WSI_EXTENSIONS


def openslide_available() -> bool:
  try:
    import openslide  # noqa: F401
    return True
  except ImportError:
    return False


def read_slide_mpp(slide: Any) -> float | None:
  """Read microns-per-pixel from OpenSlide properties (level 0)."""
  values: list[float] = []
  for key in ("openslide.mpp-x", "openslide.mpp-y"):
    raw = slide.properties.get(key)
    if not raw:
      continue
    try:
      values.append(float(raw))
    except ValueError:
      continue
  if not values:
    return None
  return sum(values) / len(values)


def select_level_for_mpp(slide: Any, target_mpp: float) -> tuple[int, float]:
  """Pick the pyramid level whose MPP is closest to ``target_mpp``."""
  if target_mpp <= 0:
    raise ValueError("target_mpp must be positive")

  base_mpp = read_slide_mpp(slide)
  if base_mpp is None:
    downsample = max(1.0, target_mpp / 0.25)
    level = slide.get_best_level_for_downsample(downsample)
    ds = float(slide.level_downsamples[level])
    return level, target_mpp

  best_level = 0
  best_error = float("inf")
  for level in range(slide.level_count):
    ds = float(slide.level_downsamples[level])
    level_mpp = base_mpp * ds
    error = abs(math.log(level_mpp) - math.log(target_mpp))
    if error < best_error:
      best_level = level
      best_error = error

  ds = float(slide.level_downsamples[best_level])
  return best_level, base_mpp * ds


def _is_background(tile_pixels: np.ndarray, cutoff: int) -> bool:
  return float(tile_pixels.mean()) >= cutoff


def _level0_to_thumbnail(
  x0: int,
  y0: int,
  tile_w0: int,
  tile_h0: int,
  *,
  level0_width: int,
  level0_height: int,
  thumb_width: int,
  thumb_height: int,
) -> tuple[int, int, int, int]:
  """Map a level-0 region onto thumbnail pixel coordinates for heatmaps."""
  sx = thumb_width / max(1, level0_width)
  sy = thumb_height / max(1, level0_height)
  tx = int(round(x0 * sx))
  ty = int(round(y0 * sy))
  tw = max(1, int(round(tile_w0 * sx)))
  th = max(1, int(round(tile_h0 * sy)))
  return tx, ty, tw, th


def tile_wsi(
  path: str | Path,
  *,
  tile_size: int,
  target_mpp: float,
  overlap: float = 0.0,
  background_intensity_cutoff: int = 220,
  thumbnail_max_side: int = 2048,
  max_tiles: int | None = 10_000,
) -> tuple[np.ndarray, list[tuple[Tile, np.ndarray]], WsiScanInfo]:
  """Tile a whole-slide image at native magnification matching ``target_mpp``.

  Returns a downsampled thumbnail (for display/heatmaps), model-resolution tiles,
  and scan metadata. Tile coordinates are in **thumbnail space** so existing
  heatmap rendering works unchanged.
  """
  if not openslide_available():
    raise RuntimeError("OpenSlide is required for full-resolution WSI tiling")

  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(path)
  if tile_size <= 0:
    raise ValueError("tile_size must be positive")
  if not 0.0 <= overlap < 1.0:
    raise ValueError("overlap must be in [0.0, 1.0)")

  import openslide

  slide = openslide.OpenSlide(str(path))
  try:
    level0_w, level0_h = slide.dimensions
    level, level_mpp = select_level_for_mpp(slide, target_mpp)
    downsample = float(slide.level_downsamples[level])
    level_w, level_h = slide.level_dimensions[level]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))

    thumbnail = load_wsi_thumbnail(path, max_side=thumbnail_max_side)
    thumb_h, thumb_w = thumbnail.shape[:2]

    tiles: list[tuple[Tile, np.ndarray]] = []
    skipped = 0
    index = 0

    for y in range(0, max(1, level_h - tile_size + 1), stride):
      for x in range(0, max(1, level_w - tile_size + 1), stride):
        if max_tiles is not None and len(tiles) >= max_tiles:
          break

        x0 = int(round(x * downsample))
        y0 = int(round(y * downsample))
        w0 = int(round(tile_size * downsample))
        h0 = int(round(tile_size * downsample))

        region = slide.read_region((x0, y0), level, (tile_size, tile_size)).convert("RGB")
        pixels = np.asarray(region, dtype=np.uint8)
        if pixels.shape[0] != tile_size or pixels.shape[1] != tile_size:
          continue
        if _is_background(pixels, background_intensity_cutoff):
          skipped += 1
          continue

        tx, ty, tw, th = _level0_to_thumbnail(
          x0,
          y0,
          w0,
          h0,
          level0_width=level0_w,
          level0_height=level0_h,
          thumb_width=thumb_w,
          thumb_height=thumb_h,
        )
        tile = Tile(index=index, x=tx, y=ty, width=tw, height=th)
        tiles.append((tile, pixels))
        index += 1

      if max_tiles is not None and len(tiles) >= max_tiles:
        break

    info = WsiScanInfo(
      path=str(path),
      level=level,
      level_mpp=float(level_mpp),
      target_mpp=float(target_mpp),
      level0_width=int(level0_w),
      level0_height=int(level0_h),
      thumbnail_width=int(thumb_w),
      thumbnail_height=int(thumb_h),
      tile_size=int(tile_size),
      tiles_scored=len(tiles),
      tiles_skipped_background=int(skipped),
    )
    return thumbnail, tiles, info
  finally:
    slide.close()


def prepare_wsi_tiles(
  path: str | Path,
  tiling_cfg: dict[str, Any],
) -> tuple[np.ndarray, list[tuple[Tile, np.ndarray]], WsiScanInfo]:
  """Run full-res WSI tiling using values from config ``tiling`` section."""
  return tile_wsi(
    path,
    tile_size=int(tiling_cfg["tile_size"]),
    target_mpp=float(tiling_cfg.get("target_mpp", DEFAULT_GI_TARGET_MPP)),
    overlap=float(tiling_cfg.get("overlap", 0.0)),
    background_intensity_cutoff=int(tiling_cfg.get("background_intensity_cutoff", 220)),
    thumbnail_max_side=int(tiling_cfg.get("thumbnail_max_side", 2048)),
    max_tiles=tiling_cfg.get("wsi_max_tiles"),
  )


def load_wsi_thumbnail(path: str | Path, max_side: int = 2048) -> np.ndarray:
  """Load a downsampled RGB view of a whole-slide image for display."""
  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(path)

  if is_wsi_path(path) and openslide_available():
    import openslide

    slide = openslide.OpenSlide(str(path))
    try:
      w, h = slide.dimensions
      scale = max(w, h) / max_side
      level = 0
      if scale > 1:
        level = slide.get_best_level_for_downsample(scale)
      size = slide.level_dimensions[level]
      img = slide.read_region((0, 0), level, size).convert("RGB")
      return np.asarray(img, dtype=np.uint8)
    finally:
      slide.close()

  from PIL import Image

  with Image.open(path) as img:
    rgb = img.convert("RGB")
    w, h = rgb.size
    if max(w, h) > max_side:
      scale = max_side / max(w, h)
      rgb = rgb.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return np.asarray(rgb, dtype=np.uint8)
