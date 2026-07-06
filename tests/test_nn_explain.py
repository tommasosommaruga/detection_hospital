"""Tests for NN layer explainability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathassist.backbone import build_model
from pathassist.nn_explain import explain_tile, extract_layer_activations, grad_cam
from pathassist.synthetic import make_synthetic_slide


@pytest.fixture
def custom_model():
  cfg = {
    "backbone": "custom",
    "conv_channels": [8, 16],
    "kernel_size": 3,
    "use_batch_norm": True,
    "pool": "max",
    "dropout": 0.0,
    "head_hidden": 0,
  }
  model = build_model(cfg)
  model.eval()
  return model, cfg


def test_extract_layer_activations(custom_model):
  model, cfg = custom_model
  img = make_synthetic_slide(height=96, width=96, seed=4)
  layers = extract_layer_activations(model, img, cfg, "cpu")
  assert len(layers) >= 2
  assert layers[0].activation_map.shape == (96, 96)
  assert layers[-1].stage == "late"


def test_grad_cam_shape(custom_model):
  model, cfg = custom_model
  img = make_synthetic_slide(height=96, width=96, seed=5)
  cam = grad_cam(model, img, cfg, "cpu")
  assert cam.shape == (96, 96)
  assert cam.max() <= 1.0 + 1e-5


def test_explain_tile_writes_files(custom_model, tmp_path):
  model, cfg = custom_model
  img = make_synthetic_slide(height=96, width=96, seed=6)
  result = explain_tile(model, img, cfg, "cpu", output_dir=tmp_path, prefix="t")
  assert 0.0 <= result["score"] <= 1.0
  assert Path(result["paths"]["gradcam"]).exists()
  assert Path(result["paths"]["heatmap_conv1"]).exists()


def test_layer_maps_differ_across_depth(custom_model):
  """Early and mid layer salience maps must not collapse to the same pattern."""
  model, cfg = custom_model
  img = make_synthetic_slide(height=96, width=96, seed=7)
  layers = extract_layer_activations(model, img, cfg, "cpu")
  assert len(layers) >= 2
  active = [layer for layer in layers if layer.activation_map.std() > 1e-4]
  assert len(active) >= 2, "expected at least two non-flat layer maps"
  a = active[0].activation_map.ravel()
  b = active[-1].activation_map.ravel()
  corr = float(np.corrcoef(a, b)[0, 1])
  mad = float(np.mean(np.abs(a - b)))
  assert corr < 0.98 or mad > 0.05, f"maps too similar (corr={corr:.3f}, mad={mad:.3f})"
