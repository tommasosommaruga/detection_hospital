"""The tile classifier model.

A small, self-contained CNN that outputs a single malignancy probability per
tile. It is intentionally simple so it trains quickly on a Colab GPU and runs
fast on a CPU. The architecture is device-agnostic - the same nn.Module trains
on cuda and does inference on cpu unchanged.

Swap this for a stronger backbone (ResNet, EfficientNet, a pathology foundation
model) later without touching the rest of the pipeline; only this file and the
checkpoint format need to agree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CHECKPOINT_FORMAT = 1


def build_model(tile_size: int = 256):
  """Construct the tile classifier. Import torch lazily so the rest of the
  package (tiling, triage, reporting, audit) works without torch installed."""
  import torch.nn as nn

  class TileClassifier(nn.Module):
    """Compact CNN: 3 conv blocks -> global pooling -> single logit."""

    def __init__(self) -> None:
      super().__init__()
      self.features = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
      )
      self.classifier = nn.Linear(64, 1)

    def forward(self, x):
      feats = self.features(x)
      feats = feats.flatten(1)
      return self.classifier(feats).squeeze(1)  # raw logits, shape (N,)

  return TileClassifier()


def save_checkpoint(model, path: str | Path, tile_size: int, version: str) -> Path:
  """Save weights plus the metadata needed to reload and run on any device.

  We always move weights to CPU before saving so the checkpoint loads on a
  machine with no GPU (the common inference case). Training device is irrelevant
  to the saved file.
  """
  import torch

  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
  payload: dict[str, Any] = {
    "checkpoint_format": CHECKPOINT_FORMAT,
    "tile_size": tile_size,
    "version": version,
    "state_dict": cpu_state,
  }
  torch.save(payload, path)
  return path


def load_checkpoint(path: str | Path, device: str = "cpu"):
  """Load a checkpoint and return (model_on_device, metadata_dict).

  map_location=cpu first guarantees a GPU-trained checkpoint loads even where no
  GPU exists; the model is then moved to the requested device.
  """
  import torch

  payload = torch.load(Path(path), map_location="cpu")
  if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
    raise ValueError(
      f"Unsupported checkpoint format {payload.get('checkpoint_format')!r}; "
      f"expected {CHECKPOINT_FORMAT}."
    )
  model = build_model(tile_size=int(payload["tile_size"]))
  model.load_state_dict(payload["state_dict"])
  model.to(device)
  model.eval()
  metadata = {k: v for k, v in payload.items() if k != "state_dict"}
  return model, metadata
