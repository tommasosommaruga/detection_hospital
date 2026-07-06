"""Tile preprocessing shared by training and inference.

Using one transform in both places is what guarantees the model sees identical
inputs whether it is being trained on a Colab GPU or run later on a CPU. If these
diverged, a model could look accurate in training and misbehave at inference.
"""

from __future__ import annotations

import numpy as np

PIXEL_SCALE = 255.0
NORM_MEAN = 0.5
NORM_STD = 0.5
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tile_to_tensor(pixels: np.ndarray, normalize: str = "custom"):
  """Convert one HWC uint8 tile to a normalised CHW float32 torch tensor."""
  import torch

  array = pixels.astype(np.float32) / PIXEL_SCALE
  if normalize == "imagenet":
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
  else:
    array = (array - NORM_MEAN) / NORM_STD
  return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tiles_to_batch(pixel_arrays: list[np.ndarray], normalize: str = "custom"):
  """Stack several HWC uint8 tiles into a single (N, C, H, W) float32 batch."""
  import torch

  tensors = [tile_to_tensor(pixels, normalize=normalize) for pixels in pixel_arrays]
  return torch.stack(tensors, dim=0)
