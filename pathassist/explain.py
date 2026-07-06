"""Explainability overlays.

Pathologists distrust black boxes, so the pipeline always emits visual artifacts
alongside the numbers: a heatmap of tile scores and, when available, an
uncertainty overlay for regions that need closer review.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import TileScore


def _score_to_rgb(score: float) -> tuple[int, int, int]:
  """Map a score in [0, 1] to a green-to-red heatmap colour."""
  score = min(1.0, max(0.0, score))
  red = int(round(255 * score))
  green = int(round(255 * (1.0 - score)))
  return (red, green, 0)


def _uncertainty_to_rgb(uncertainty: float) -> tuple[int, int, int]:
  """Map uncertainty in [0, 1] to a blue-to-yellow warning colour."""
  uncertainty = min(1.0, max(0.0, uncertainty))
  blue = int(round(255 * (1.0 - uncertainty)))
  yellow = int(round(255 * uncertainty))
  return (yellow, yellow, blue)


def render_heatmap(
  image: np.ndarray,
  tile_scores: list[TileScore],
  output_path: str | Path,
  alpha: float = 0.45,
) -> Path:
  """Blend a per-tile score heatmap over the original image and save it."""
  from PIL import Image

  base = Image.fromarray(image).convert("RGB")
  overlay = Image.new("RGB", base.size, (0, 0, 0))
  overlay_pixels = overlay.load()

  for tile_score in tile_scores:
    tile = tile_score.tile
    colour = _score_to_rgb(tile_score.score)
    for yy in range(tile.y, min(tile.y + tile.height, base.height)):
      for xx in range(tile.x, min(tile.x + tile.width, base.width)):
        overlay_pixels[xx, yy] = colour

  blended = Image.blend(base, overlay, alpha)
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  blended.save(output_path)
  return output_path


def render_uncertainty_map(
  image: np.ndarray,
  tile_scores: list[TileScore],
  output_path: str | Path,
  alpha: float = 0.40,
  min_uncertainty: float = 0.35,
) -> Path | None:
  """Highlight uncertain tiles so the pathologist knows where the model hesitates."""
  from PIL import Image

  uncertain = [ts for ts in tile_scores if ts.uncertainty >= min_uncertainty]
  if not uncertain:
    return None

  base = Image.fromarray(image).convert("RGB")
  overlay = Image.new("RGB", base.size, (0, 0, 0))
  overlay_pixels = overlay.load()

  for tile_score in uncertain:
    tile = tile_score.tile
    colour = _uncertainty_to_rgb(tile_score.uncertainty)
    for yy in range(tile.y, min(tile.y + tile.height, base.height)):
      for xx in range(tile.x, min(tile.x + tile.width, base.width)):
        overlay_pixels[xx, yy] = colour

  blended = Image.blend(base, overlay, alpha)
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  blended.save(output_path)
  return output_path
