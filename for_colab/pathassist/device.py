"""Device selection.

One place decides where tensors go, so the same code path trains on a Colab GPU
and runs on a laptop CPU. "auto" prefers CUDA, then Apple Silicon (MPS), then CPU.
"""

from __future__ import annotations


def resolve_device(preference: str = "auto") -> str:
  """Return a concrete torch device string given a preference.

  preference: "auto" | "cuda" | "mps" | "cpu". "auto" picks the best available.
  """
  import torch

  preference = preference.lower()
  if preference in ("cuda", "mps", "cpu"):
    return preference

  if torch.cuda.is_available():
    return "cuda"
  if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    return "mps"
  return "cpu"
