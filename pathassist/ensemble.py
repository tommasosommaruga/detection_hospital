"""Ensemble inference — weighted voting across multiple trained models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .backbone import build_model, get_model_norm
from .preprocess import tiles_to_batch

ENSEMBLE_CHECKPOINT_FORMAT = 3


class VotingEnsemble:
  """Multiple models vote on each tile with optional performance-based weights."""

  def __init__(self, members: list[dict[str, Any]], vote_mode: str = "soft") -> None:
    self.members = members
    self.vote_mode = vote_mode

  @property
  def n_models(self) -> int:
    return len(self.members)

  def predict_batch(
    self,
    pixel_arrays: list[np.ndarray],
    device: str,
    batch_size: int = 32,
  ) -> dict[str, np.ndarray]:
    """Return ensemble probabilities and per-tile disagreement for a tile batch."""
    import torch

    if not pixel_arrays:
      return {
        "prob": np.array([], dtype=np.float32),
        "disagreement": np.array([], dtype=np.float32),
      }

    member_probs = []
    for member in self.members:
      member_probs.append(
        self._member_probs(member, pixel_arrays, device, batch_size)
      )

    stacked = np.stack(member_probs, axis=0)
    weights = np.array([m["weight"] for m in self.members], dtype=np.float32)
    weights /= max(weights.sum(), 1e-8)

    soft_prob = (stacked * weights[:, None]).sum(axis=0)
    hard_votes = (stacked >= 0.5).astype(np.float32)
    hard_prob = (hard_votes.mean(axis=0) >= 0.5).astype(np.float32)
    disagreement = stacked.std(axis=0)

    final_prob = soft_prob if self.vote_mode == "soft" else hard_prob
    return {"prob": final_prob, "disagreement": disagreement, "member_probs": stacked}

  def _member_probs(
    self,
    member: dict[str, Any],
    pixel_arrays: list[np.ndarray],
    device: str,
    batch_size: int,
  ) -> np.ndarray:
    import torch

    model = member["model"]
    norm = get_model_norm(member["model_config"])
    outputs: list[np.ndarray] = []

    with torch.no_grad():
      for start in range(0, len(pixel_arrays), batch_size):
        chunk = pixel_arrays[start : start + batch_size]
        batch = tiles_to_batch(chunk, normalize=norm).to(device)
        probs = torch.sigmoid(model(batch)).detach().cpu().numpy()
        outputs.append(probs)

    return np.concatenate(outputs).astype(np.float32)


def load_ensemble_checkpoint(
  path: str | Path,
  device: str = "cpu",
) -> tuple[VotingEnsemble, dict[str, Any]]:
  """Load an ensemble.pt checkpoint produced by the training notebook."""
  import torch

  payload = torch.load(Path(path), map_location="cpu", weights_only=False)
  fmt = payload.get("checkpoint_format")
  if fmt != ENSEMBLE_CHECKPOINT_FORMAT:
    raise ValueError(
      f"Unsupported checkpoint format {fmt!r}; expected {ENSEMBLE_CHECKPOINT_FORMAT}."
    )

  members = []
  for spec in payload["members"]:
    # Weights are in the checkpoint — do not re-download ImageNet pretrained weights.
    cfg = dict(spec["model_config"])
    cfg["pretrained"] = False
    model = build_model(cfg)
    model.load_state_dict(spec["state_dict"])
    model.to(device)
    model.eval()
    members.append(
      {
        "name": spec["name"],
        "model": model,
        "model_config": spec["model_config"],
        "weight": float(spec.get("weight", 1.0)),
        "val_acc": float(spec.get("val_acc", 0.0)),
      }
    )

  ensemble = VotingEnsemble(members, vote_mode=str(payload.get("vote_mode", "soft")))
  metadata = {k: v for k, v in payload.items() if k != "members"}
  return ensemble, metadata
