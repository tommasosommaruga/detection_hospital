"""WSI helper tests."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from pathassist.tiling import resolve_case_tiles, should_use_wsi_full_res
from pathassist.wsi import (
  WsiScanInfo,
  is_wsi_path,
  load_wsi_thumbnail,
  openslide_available,
  read_slide_mpp,
  select_level_for_mpp,
  tile_wsi,
)


class FakeSlide:
  """Minimal OpenSlide stand-in for unit tests."""

  def __init__(self, width: int = 10_000, height: int = 8000, mpp: float = 0.25) -> None:
    self.dimensions = (width, height)
    self.level_count = 3
    self.level_downsamples = [1.0, 4.0, 16.0]
    self.level_dimensions = [
      (width, height),
      (width // 4, height // 4),
      (width // 16, height // 16),
    ]
    self.properties = {
      "openslide.mpp-x": str(mpp),
      "openslide.mpp-y": str(mpp),
    }

  def get_best_level_for_downsample(self, downsample: float) -> int:
    best = 0
    best_err = float("inf")
    for level, ds in enumerate(self.level_downsamples):
      err = abs(ds - downsample)
      if err < best_err:
        best = level
        best_err = err
    return best

  def read_region(self, location, level, size):
    w, h = size
    rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    rgb[0, 0] = (200, 50, 50)
    return Image.fromarray(rgb, mode="RGB")

  def close(self) -> None:
    return None


def test_is_wsi_path():
  assert is_wsi_path("slide.svs")
  assert is_wsi_path("slide.ndpi")
  assert not is_wsi_path("tile.png")


def test_load_wsi_thumbnail_png_fallback(tmp_path):
  path = tmp_path / "tile.png"
  Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path)
  arr = load_wsi_thumbnail(path, max_side=128)
  assert arr.shape == (64, 64, 3)


def test_openslide_available_is_bool():
  assert isinstance(openslide_available(), bool)


def test_read_slide_mpp():
  slide = FakeSlide(mpp=0.243)
  assert read_slide_mpp(slide) == pytest.approx(0.243)


def test_select_level_for_mpp_picks_closest_level():
  slide = FakeSlide(mpp=0.243)
  level, mpp = select_level_for_mpp(slide, target_mpp=0.972)
  assert level == 1
  assert mpp == pytest.approx(0.243 * 4.0)


def test_select_level_for_mpp_without_metadata():
  slide = FakeSlide(mpp=0.25)
  slide.properties = {}
  level, mpp = select_level_for_mpp(slide, target_mpp=1.0)
  assert 0 <= level < slide.level_count
  assert mpp == 1.0


def test_tile_wsi_returns_thumbnail_coords(tmp_path, monkeypatch):
  monkeypatch.setattr("pathassist.wsi.openslide_available", lambda: True)
  path = tmp_path / "slide.svs"
  path.write_bytes(b"fake")
  mock_openslide = MagicMock()
  mock_openslide.OpenSlide.return_value = FakeSlide()
  monkeypatch.setitem(sys.modules, "openslide", mock_openslide)

  thumbnail, tiles, info = tile_wsi(
    path,
    tile_size=32,
    target_mpp=0.972,
    background_intensity_cutoff=250,
    thumbnail_max_side=512,
    max_tiles=50,
  )

  assert thumbnail.ndim == 3
  assert len(tiles) > 0
  assert isinstance(info, WsiScanInfo)
  tile, pixels = tiles[0]
  assert pixels.shape == (32, 32, 3)
  assert 0 <= tile.x < thumbnail.shape[1]
  assert 0 <= tile.y < thumbnail.shape[0]
  assert info.tiles_scored == len(tiles)


def test_should_use_wsi_full_res_requires_config_and_path():
  cfg = {"wsi_full_res": True}
  assert not should_use_wsi_full_res("tile.png", cfg)
  assert not should_use_wsi_full_res("slide.svs", {"wsi_full_res": False})


def test_resolve_case_tiles_falls_back_without_openslide(monkeypatch):
  monkeypatch.setattr("pathassist.tiling.openslide_available", lambda: False)
  image = np.full((128, 128, 3), 100, dtype=np.uint8)
  config = {
    "tiling": {
      "tile_size": 64,
      "overlap": 0.0,
      "background_intensity_cutoff": 220,
      "wsi_full_res": True,
      "target_mpp": 0.5,
    },
  }
  display, tiles, info = resolve_case_tiles(image, config, image_path="slide.svs")
  assert info is None
  assert len(tiles) >= 1
  assert display.shape == image.shape
