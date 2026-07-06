"""Region scoring.

The scorer is the one component you will eventually replace with a trained model
(YOLO, a tile classifier, a foundation-model embedding + head, etc.). Everything
else in the pipeline depends only on the small `Scorer` protocol below, so a real
model can be dropped in without changing tiling, triage, reporting or auditing.

`DummyScorer` lets the whole pipeline run today on synthetic data with zero
weights to download - useful for wiring up and testing the workflow first.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import Tile, TileScore


@runtime_checkable
class Scorer(Protocol):
  """Anything that assigns a malignancy score in [0, 1] to a tile."""

  name: str
  version: str

  def score_tiles(self, tiles: list[tuple[Tile, np.ndarray]]) -> list[TileScore]:
    ...


class DummyScorer:
  """A deterministic placeholder scorer for development and tests.

  It uses a simple, explainable heuristic: darker, higher-saturation tiles (a
  rough proxy for dense, hematoxylin-stained nuclei) score higher. This is NOT a
  clinical signal - it exists only so the pipeline produces varied, reproducible
  output before a real model is trained.
  """

  name = "dummy-heuristic"
  version = "0.1.0"

  def __init__(self, seed: int = 0) -> None:
    self._rng = np.random.default_rng(seed)

  def score_tiles(self, tiles: list[tuple[Tile, np.ndarray]]) -> list[TileScore]:
    scores: list[TileScore] = []
    for tile, pixels in tiles:
      channel_means = pixels.reshape(-1, pixels.shape[-1]).mean(axis=0)
      darkness = 1.0 - float(channel_means.mean()) / 255.0
      saturation = float(channel_means.max() - channel_means.min()) / 255.0
      jitter = float(self._rng.uniform(-0.03, 0.03))
      raw = 0.7 * darkness + 0.3 * saturation + jitter
      score = min(1.0, max(0.0, raw))
      scores.append(TileScore(tile=tile, score=score))
    return scores


class TorchScorer:
  """Runs a trained CNN checkpoint over tiles to produce malignancy scores.

  Loading, device placement and preprocessing all defer to shared modules, so a
  GPU-trained checkpoint runs identically (just slower) on CPU. Import torch is
  deferred to construction so the rest of the package works without torch.
  """

  def __init__(
    self,
    checkpoint_path: str,
    device_preference: str = "auto",
    batch_size: int = 32,
  ) -> None:
    from .device import resolve_device
    from .model import load_checkpoint

    self.device = resolve_device(device_preference)
    self._model, metadata = load_checkpoint(checkpoint_path, device=self.device)
    self._batch_size = batch_size
    self.name = "torch-tile-classifier"
    self.version = str(metadata.get("version", "unknown"))

  def score_tiles(self, tiles: list[tuple[Tile, np.ndarray]]) -> list[TileScore]:
    import torch

    from .preprocess import tiles_to_batch

    if not tiles:
      return []

    results: list[TileScore] = []
    with torch.no_grad():
      for start in range(0, len(tiles), self._batch_size):
        chunk = tiles[start : start + self._batch_size]
        batch = tiles_to_batch([pixels for _, pixels in chunk]).to(self.device)
        probs = torch.sigmoid(self._model(batch)).detach().cpu().numpy()
        for (tile, _), prob in zip(chunk, probs):
          score = float(prob)
          uncertainty = 1.0 - abs(score - 0.5) * 2.0
          results.append(
            TileScore(tile=tile, score=score, uncertainty=uncertainty)
          )
    return results


class EnsembleScorer:
  """Runs a weighted ensemble checkpoint with uncertainty and disagreement."""

  def __init__(
    self,
    checkpoint_path: str,
    device_preference: str = "auto",
    batch_size: int = 32,
  ) -> None:
    from .device import resolve_device
    from .ensemble import load_ensemble_checkpoint

    self.device = resolve_device(device_preference)
    self._ensemble, metadata = load_ensemble_checkpoint(
      checkpoint_path, device=self.device
    )
    self._batch_size = batch_size
    self.name = "ensemble-voter"
    metrics = metadata.get("metrics") or {}
    test_acc = metrics.get("test_acc")
    version_bits = [f"{self._ensemble.n_models} models"]
    if test_acc is not None:
      version_bits.append(f"test_acc={test_acc:.3f}")
    self.version = ", ".join(version_bits)

  def score_tiles(self, tiles: list[tuple[Tile, np.ndarray]]) -> list[TileScore]:
    if not tiles:
      return []

    results: list[TileScore] = []
    for start in range(0, len(tiles), self._batch_size):
      chunk = tiles[start : start + self._batch_size]
      pixel_arrays = [pixels for _, pixels in chunk]
      vote = self._ensemble.predict_batch(
        pixel_arrays, self.device, batch_size=self._batch_size
      )
      for (tile, _), prob, disagreement in zip(
        chunk, vote["prob"], vote["disagreement"]
      ):
        score = float(prob)
        borderline = 1.0 - abs(score - 0.5) * 2.0
        uncertainty = min(1.0, float(disagreement) + 0.5 * borderline)
        results.append(
          TileScore(
            tile=tile,
            score=score,
            uncertainty=uncertainty,
            disagreement=float(disagreement),
          )
        )
    return results
