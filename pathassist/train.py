"""Training entry point.

Designed to run on a Colab GPU but stay fully functional on CPU (just slower).
The device is chosen automatically, weights are always saved CPU-loadable, and
the same preprocessing used at inference is used here - so a model trained on GPU
runs identically on a laptop.

Data sources (see `pathassist/data.py`):
  * synthetic (default) - a learnable toy signal, needs no files.
  * image folder        - one subfolder per class.
  * CSV                 - a list of image paths + labels.

`train(...)` returns a `TrainResult` carrying the checkpoint path, the per-epoch
history and the validation predictions, so a notebook can plot curves, a
confusion matrix and sample predictions without re-running anything.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import load_from_csv, load_from_image_folder
from .synthetic import make_tile_dataset


@dataclass
class TrainResult:
  """Everything needed to save, reload and inspect a training run."""

  checkpoint_path: Path
  tile_size: int
  device: str
  history: list[dict] = field(default_factory=list)
  # Validation set predictions from the final epoch (for confusion matrix, etc.).
  val_true: np.ndarray | None = None
  val_prob: np.ndarray | None = None


def build_dataset(
  source: str = "synthetic",
  data_dir: str | None = None,
  data_csv: str | None = None,
  image_root: str | None = None,
  path_column: str = "path",
  label_column: str = "label",
  positive_labels: list[str] | None = None,
  tile_size: int = 64,
  num_samples: int = 400,
  seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
  """Return `(tiles, labels)` from the requested source.

  This is the single seam between "toy" and "real": point it at a folder or CSV
  and the rest of training is unchanged.
  """
  if source == "folder":
    if not data_dir:
      raise ValueError("source='folder' requires data_dir.")
    return load_from_image_folder(data_dir, tile_size=tile_size)
  if source == "csv":
    if not data_csv:
      raise ValueError("source='csv' requires data_csv.")
    return load_from_csv(
      data_csv,
      image_root=image_root,
      tile_size=tile_size,
      path_column=path_column,
      label_column=label_column,
      positive_labels=positive_labels,
    )
  if source == "synthetic":
    return make_tile_dataset(num_samples=num_samples, tile_size=tile_size, seed=seed)
  raise ValueError(f"Unknown data source: {source!r}")


def _split(tiles: np.ndarray, labels: np.ndarray, val_fraction: float, seed: int):
  rng = np.random.default_rng(seed)
  order = rng.permutation(len(tiles))
  n_val = max(1, int(len(tiles) * val_fraction))
  val_idx, train_idx = order[:n_val], order[n_val:]
  return (tiles[train_idx], labels[train_idx]), (tiles[val_idx], labels[val_idx])


def _iter_batches(tiles: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool):
  indices = np.arange(len(tiles))
  if shuffle:
    np.random.shuffle(indices)
  for start in range(0, len(indices), batch_size):
    chunk = indices[start : start + batch_size]
    yield tiles[chunk], labels[chunk]


def train(
  epochs: int = 5,
  batch_size: int = 32,
  learning_rate: float = 1e-3,
  tile_size: int = 64,
  val_fraction: float = 0.2,
  device_preference: str = "auto",
  version: str = "0.1.0",
  out_path: str = "models/tile_classifier.pt",
  seed: int = 0,
  tiles: np.ndarray | None = None,
  labels: np.ndarray | None = None,
  num_samples: int = 400,
) -> TrainResult:
  """Train the tile classifier and return a TrainResult.

  Pass `tiles`/`labels` (e.g. from `build_dataset`) to train on real data; if
  omitted, a synthetic dataset is generated so this always runs.
  """
  import torch
  import torch.nn as nn

  from .device import resolve_device
  from .model import build_model, save_checkpoint
  from .preprocess import tiles_to_batch

  device = resolve_device(device_preference)
  print(f"Training on device: {device}")

  if tiles is None or labels is None:
    tiles, labels = make_tile_dataset(num_samples=num_samples, tile_size=tile_size, seed=seed)
  else:
    tile_size = int(tiles.shape[1])  # trust the data's actual tile size
  print(f"Dataset: {len(tiles)} tiles, {int(labels.sum())} positive, tile_size={tile_size}")

  (train_x, train_y), (val_x, val_y) = _split(tiles, labels, val_fraction, seed)

  model = build_model(tile_size=tile_size).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  loss_fn = nn.BCEWithLogitsLoss()

  history: list[dict] = []
  val_true = val_prob = None
  for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    start_time = time.time()
    for batch_x, batch_y in _iter_batches(train_x, train_y, batch_size, shuffle=True):
      inputs = tiles_to_batch(list(batch_x)).to(device)
      targets = torch.from_numpy(batch_y).to(device)
      optimizer.zero_grad()
      logits = model(inputs)
      loss = loss_fn(logits, targets)
      loss.backward()
      optimizer.step()
      epoch_loss += float(loss.item())
      n_batches += 1

    val_true, val_prob, val_acc, val_loss = _evaluate(
      model, val_x, val_y, batch_size, device, loss_fn
    )
    train_loss = epoch_loss / max(1, n_batches)
    elapsed = time.time() - start_time
    history.append(
      {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
    )
    print(
      f"epoch {epoch:>2}/{epochs}  train_loss {train_loss:.4f}  "
      f"val_loss {val_loss:.4f}  val_acc {val_acc:.3f}  ({elapsed:.1f}s)"
    )

  saved = save_checkpoint(model, out_path, tile_size=tile_size, version=version)
  print(f"Saved CPU-loadable checkpoint to: {saved}")
  return TrainResult(
    checkpoint_path=saved,
    tile_size=tile_size,
    device=device,
    history=history,
    val_true=val_true,
    val_prob=val_prob,
  )


def _evaluate(model, val_x, val_y, batch_size, device, loss_fn):
  """Return (true_labels, probs, accuracy, mean_loss) over the validation set."""
  import torch

  from .preprocess import tiles_to_batch

  model.eval()
  all_probs: list[np.ndarray] = []
  all_true: list[np.ndarray] = []
  total_loss = 0.0
  n_batches = 0
  with torch.no_grad():
    for batch_x, batch_y in _iter_batches(val_x, val_y, batch_size, shuffle=False):
      inputs = tiles_to_batch(list(batch_x)).to(device)
      targets = torch.from_numpy(batch_y).to(device)
      logits = model(inputs)
      total_loss += float(loss_fn(logits, targets).item())
      n_batches += 1
      all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
      all_true.append(batch_y)

  probs = np.concatenate(all_probs) if all_probs else np.array([])
  true = np.concatenate(all_true) if all_true else np.array([])
  preds = (probs >= 0.5).astype(np.float32)
  accuracy = float((preds == true).mean()) if len(true) else 0.0
  return true, probs, accuracy, total_loss / max(1, n_batches)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="pathassist.train",
    description="Train the tile classifier (GPU on Colab, CPU anywhere).",
  )
  parser.add_argument(
    "--source", default="synthetic", choices=["synthetic", "folder", "csv"],
    help="Where tiles come from.",
  )
  parser.add_argument("--data-dir", default=None, help="Root folder for --source folder.")
  parser.add_argument("--data-csv", default=None, help="CSV file for --source csv.")
  parser.add_argument("--image-root", default=None, help="Base dir for relative CSV paths.")
  parser.add_argument("--path-column", default="path")
  parser.add_argument("--label-column", default="label")
  parser.add_argument(
    "--positive-labels", nargs="*", default=None,
    help="Label values that mean positive/malignant (for non-numeric labels).",
  )
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--tile-size", type=int, default=64)
  parser.add_argument("--num-samples", type=int, default=400, help="Synthetic set size.")
  parser.add_argument("--val-fraction", type=float, default=0.2)
  parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
  parser.add_argument("--version", default="0.1.0")
  parser.add_argument("--out", default="models/tile_classifier.pt")
  parser.add_argument("--seed", type=int, default=0)
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  tiles, labels = build_dataset(
    source=args.source,
    data_dir=args.data_dir,
    data_csv=args.data_csv,
    image_root=args.image_root,
    path_column=args.path_column,
    label_column=args.label_column,
    positive_labels=args.positive_labels,
    tile_size=args.tile_size,
    num_samples=args.num_samples,
    seed=args.seed,
  )
  train(
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    tile_size=args.tile_size,
    val_fraction=args.val_fraction,
    device_preference=args.device,
    version=args.version,
    out_path=args.out,
    seed=args.seed,
    tiles=tiles,
    labels=labels,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
