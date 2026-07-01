"""Tile preprocessing shared by training and inference.

Using one transform in both places is what guarantees the model sees identical
inputs whether it is being trained on a Colab GPU or run later on a CPU. If these
diverged, a model could look accurate in training and misbehave at inference.
"""

from __future__ import annotations

import numpy as np

# Simple, dependency-free normalisation. Kept here (not hardcoded in the model)
# so it can be tuned or swapped for dataset-specific stats in one place.
PIXEL_SCALE = 255.0
NORM_MEAN = 0.5
NORM_STD = 0.5


def tile_to_tensor(pixels: np.ndarray):
  """Convert one HWC uint8 tile to a normalised CHW float32 torch tensor."""
  import torch

  array = pixels.astype(np.float32) / PIXEL_SCALE
  array = (array - NORM_MEAN) / NORM_STD
  # HWC -> CHW
  tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
  return tensor


def tiles_to_batch(pixel_arrays: list[np.ndarray]):
  """Stack several HWC uint8 tiles into a single (N, C, H, W) float32 batch."""
  import torch

  tensors = [tile_to_tensor(pixels) for pixels in pixel_arrays]
  return torch.stack(tensors, dim=0)
