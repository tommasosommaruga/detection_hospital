"""Neural-network explainability: layer activations and Grad-CAM.

Each stage uses **layer-wise Grad-CAM** (gradients × activations for the malignancy
score) so early/mid/late maps show genuinely different, prediction-relevant patterns —
not a channel-average that looks identical at every depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

LAYER_DESCRIPTIONS = {
  "early": "Edges, stain boundaries, tissue vs background",
  "mid": "Textures, cell clusters, stromal patterns",
  "late": "Semantic regions linked to malignancy score",
  "head": "Final malignancy logit (not spatial)",
}

# Approximate turbo colormap LUT (256 × RGB) for readable heatmaps.
_TURBO_LUT = np.array(
  [
    (48, 18, 59), (65, 30, 90), (72, 40, 120), (68, 51, 146), (59, 63, 167),
    (48, 76, 182), (40, 90, 191), (29, 105, 195), (20, 120, 196), (13, 135, 194),
    (13, 150, 191), (15, 165, 186), (26, 179, 178), (42, 192, 168), (63, 204, 155),
    (87, 214, 141), (114, 222, 125), (142, 228, 108), (170, 232, 90), (198, 234, 72),
    (224, 233, 54), (246, 228, 37), (254, 217, 28), (254, 202, 28), (254, 185, 32),
    (252, 166, 40), (248, 146, 49), (242, 125, 58), (234, 104, 67), (224, 83, 76),
    (212, 62, 85), (198, 42, 94), (182, 24, 102), (163, 8, 110), (142, 0, 117),
    (119, 2, 122), (94, 8, 124), (68, 16, 122), (42, 26, 115), (18, 38, 102),
  ],
  dtype=np.float32,
)
# Interpolate to 256 steps
_TURBO = np.zeros((256, 3), dtype=np.float32)
for i in range(256):
  t = i / 255.0 * (len(_TURBO_LUT) - 1)
  lo = int(t)
  hi = min(lo + 1, len(_TURBO_LUT) - 1)
  frac = t - lo
  _TURBO[i] = _TURBO_LUT[lo] * (1 - frac) + _TURBO_LUT[hi] * frac


@dataclass
class LayerExplanation:
  name: str
  stage: str
  description: str
  activation_map: np.ndarray  # H×W float32 in [0,1] — layer Grad-CAM salience
  mean_activation: float
  max_activation: float
  dominant_channel: int | None = None
  raw_spatial_shape: tuple[int, ...] | None = None

  def to_dict(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "stage": self.stage,
      "description": self.description,
      "mean_activation": round(self.mean_activation, 4),
      "max_activation": round(self.max_activation, 4),
      "dominant_channel": self.dominant_channel,
      "raw_spatial_shape": list(self.raw_spatial_shape) if self.raw_spatial_shape else None,
      "shape": list(self.activation_map.shape),
      "method": "layer_grad_cam",
    }


def _percentile_normalize(arr: np.ndarray, lo_pct: float = 5.0, hi_pct: float = 99.0) -> np.ndarray:
  arr = arr.astype(np.float32)
  lo = float(np.percentile(arr, lo_pct))
  hi = float(np.percentile(arr, hi_pct))
  if hi - lo < 1e-8:
    return np.zeros_like(arr)
  out = (arr - lo) / (hi - lo)
  return np.clip(out, 0.0, 1.0)


def _upsample_map(act: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
  from PIL import Image

  if act.ndim == 3:
    raise ValueError("Pass a 2D spatial map to _upsample_map")
  small = Image.fromarray(_percentile_normalize(act))
  return np.asarray(small.resize((out_w, out_h), Image.Resampling.BILINEAR), dtype=np.float32)


def _dominant_filter_map(act: np.ndarray) -> tuple[np.ndarray, int]:
  """Single most active convolutional filter at this depth (C×H×W)."""
  if act.ndim != 3:
    return _percentile_normalize(act), 0
  channel_energy = act.reshape(act.shape[0], -1).mean(axis=1)
  ch = int(np.argmax(channel_energy))
  return _percentile_normalize(act[ch]), ch


def _max_projection(act: np.ndarray) -> np.ndarray:
  if act.ndim == 3:
    return _percentile_normalize(np.max(act, axis=0))
  return _percentile_normalize(act)


def _layer_plan(model, backbone: str) -> list[tuple[str, str, str]]:
  if backbone == "resnet18":
    return [
      ("net.conv1", "conv1", "early"),
      ("net.layer1", "layer1", "early"),
      ("net.layer2", "layer2", "mid"),
      ("net.layer3", "layer3", "mid"),
      ("net.layer4", "layer4", "late"),
    ]
  if backbone == "efficientnet_b0":
    return [
      ("net.features.0", "block0", "early"),
      ("net.features.2", "block2", "mid"),
      ("net.features.4", "block4", "mid"),
      ("net.features.6", "block6", "late"),
    ]
  return [
    ("features.0", "conv1", "early"),
    ("features.4", "conv2", "mid"),
    ("features.8", "conv3", "late"),
  ]


def _resolve_target_layer(model, backbone: str) -> str:
  if backbone == "resnet18":
    return "net.layer4"
  if backbone == "efficientnet_b0":
    return "net.features"
  return "features"


def _compute_layer_grad_cams(
  model,
  pixel_array: np.ndarray,
  model_config: dict,
  device: str,
  plan: list[tuple[str, str, str]],
) -> dict[str, np.ndarray]:
  """One backward pass — Grad-CAM salience map per hooked layer."""
  import torch

  from .preprocess import tile_to_tensor

  backbone = model_config.get("backbone", "custom")
  norm = "imagenet" if backbone != "custom" else "custom"
  tensor = tile_to_tensor(pixel_array, normalize=norm).unsqueeze(0).to(device)
  tensor.requires_grad_(True)

  module_names = [p[0] for p in plan]
  activations: dict[str, torch.Tensor] = {}
  gradients: dict[str, torch.Tensor] = {}
  handles = []
  named = dict(model.named_modules())

  for name in module_names:
    if name not in named:
      continue

    def _fwd(n: str) -> Callable:
      def hook(_m, _i, out) -> None:
        activations[n] = out

      return hook

    def _bwd(n: str) -> Callable:
      def hook(_m, _gi, go) -> None:
        gradients[n] = go[0]

      return hook

    handles.append(named[name].register_forward_hook(_fwd(name)))
    handles.append(named[name].register_full_backward_hook(_bwd(name)))

  model.zero_grad()
  logit = model(tensor)
  score = logit if logit.ndim == 0 else logit.squeeze()
  score.backward()

  h, w = pixel_array.shape[:2]
  cams: dict[str, np.ndarray] = {}
  for name in module_names:
    if name not in activations or name not in gradients:
      continue
    act = activations[name].detach()[0]
    grad = gradients[name].detach()[0]
    if act.ndim != 3 or grad.ndim != 3:
      continue
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = (weights * act).sum(dim=0).relu().cpu().numpy()
    if float(cam.max()) < 1e-7:
      cam = np.maximum((weights * act).sum(dim=0).abs().cpu().numpy(), 0.0)
    cams[name] = _upsample_map(cam, h, w)

  for hnd in handles:
    hnd.remove()

  return cams


def _capture_raw_activations(
  model,
  pixel_array: np.ndarray,
  model_config: dict,
  device: str,
  module_names: list[str],
) -> dict[str, np.ndarray]:
  import torch

  from .preprocess import tile_to_tensor

  backbone = model_config.get("backbone", "custom")
  norm = "imagenet" if backbone != "custom" else "custom"
  tensor = tile_to_tensor(pixel_array, normalize=norm).unsqueeze(0).to(device)

  storage: dict[str, np.ndarray] = {}
  handles = []
  named = dict(model.named_modules())

  for name in module_names:
    if name not in named:
      continue

    def _hook(n: str) -> Callable:
      def hook(_m, _i, out) -> None:
        t = out.detach().cpu()
        storage[n] = t[0].numpy() if t.ndim == 4 else t.numpy()

      return hook

    handles.append(named[name].register_forward_hook(_hook(name)))

  model.eval()
  with torch.no_grad():
    model(tensor)
  for hnd in handles:
    hnd.remove()
  return storage


def grad_cam(
  model,
  pixel_array: np.ndarray,
  model_config: dict,
  device: str,
  target_layer_name: str | None = None,
) -> np.ndarray:
  """Grad-CAM at the deepest layer (standard class-discriminative map)."""
  import torch

  from .preprocess import tile_to_tensor

  backbone = model_config.get("backbone", "custom")
  norm = "imagenet" if backbone != "custom" else "custom"
  tensor = tile_to_tensor(pixel_array, normalize=norm).unsqueeze(0).to(device)
  tensor.requires_grad_(True)

  target = target_layer_name or _resolve_target_layer(model, backbone)
  activations: dict[str, torch.Tensor] = {}
  gradients: dict[str, torch.Tensor] = {}

  def fwd_hook(_m, _i, out):
    activations["target"] = out

  def bwd_hook(_m, _gi, go):
    gradients["target"] = go[0]

  named = dict(model.named_modules())
  if target not in named:
    target = _resolve_target_layer(model, backbone)
  handle_f = named[target].register_forward_hook(fwd_hook)
  handle_b = named[target].register_full_backward_hook(bwd_hook)

  model.zero_grad()
  logit = model(tensor)
  score = logit if logit.ndim == 0 else logit.squeeze()
  score.backward()

  act = activations["target"].detach()[0]
  grad = gradients["target"].detach()[0]
  if act.ndim == 3 and grad.ndim == 3:
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = (weights * act).sum(dim=0).relu().cpu().numpy()
  else:
    cam = act.cpu().numpy() if act.ndim == 2 else act.mean(0).cpu().numpy()

  handle_f.remove()
  handle_b.remove()

  h, w = pixel_array.shape[:2]
  return _upsample_map(cam, h, w)


def extract_layer_activations(
  model,
  pixel_array: np.ndarray,
  model_config: dict,
  device: str,
) -> list[LayerExplanation]:
  """Per-layer Grad-CAM maps — each depth shows different prediction-linked salience."""
  plan = _layer_plan(model, model_config.get("backbone", "custom"))
  module_names = [p[0] for p in plan]

  model.eval()
  layer_cams = _compute_layer_grad_cams(model, pixel_array, model_config, device, plan)
  raw = _capture_raw_activations(model, pixel_array, model_config, device, module_names)

  out: list[LayerExplanation] = []
  h, w = pixel_array.shape[:2]
  for module_name, display_name, stage in plan:
    salience: np.ndarray | None = None
    if module_name in layer_cams and layer_cams[module_name].max() > 1e-5:
      salience = layer_cams[module_name]
    elif module_name in raw:
      act = raw[module_name]
      if stage == "early":
        salience = _upsample_map(_max_projection(act), h, w)
      else:
        dom, _ = _dominant_filter_map(act)
        salience = _upsample_map(dom, h, w)
    if salience is None:
      continue

    dom_ch = None
    raw_shape = None
    if module_name in raw and raw[module_name].ndim == 3:
      _, dom_ch = _dominant_filter_map(raw[module_name])
      raw_shape = tuple(raw[module_name].shape)

    out.append(
      LayerExplanation(
        name=display_name,
        stage=stage,
        description=LAYER_DESCRIPTIONS.get(stage, ""),
        activation_map=salience,
        mean_activation=float(salience.mean()),
        max_activation=float(salience.max()),
        dominant_channel=dom_ch,
        raw_spatial_shape=raw_shape,
      )
    )
  return out


def _colormap_rgb(heat: np.ndarray) -> np.ndarray:
  idx = np.clip((heat * 255).astype(np.int32), 0, 255)
  return _TURBO[idx].astype(np.uint8)


def _blend_heatmap(base: np.ndarray, heat: np.ndarray, alpha: float = 0.52) -> np.ndarray:
  rgb = _colormap_rgb(heat)
  base_f = base.astype(np.float32)
  blend = (1.0 - alpha) * base_f + alpha * rgb
  return np.clip(blend, 0, 255).astype(np.uint8)


def render_explanation_overlays(
  pixel_array: np.ndarray,
  layers: list[LayerExplanation],
  grad_cam_map: np.ndarray,
  output_dir: str | Path,
  prefix: str = "explain",
) -> dict[str, Path]:
  """Save per-layer Grad-CAM overlays and pure heatmap thumbnails."""
  from PIL import Image

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  paths: dict[str, Path] = {}

  for layer in layers:
    heat_path = output_dir / f"{prefix}_{layer.name}_heatmap.png"
    Image.fromarray(_colormap_rgb(layer.activation_map)).save(heat_path)
    paths[f"heatmap_{layer.name}"] = heat_path

    blended = _blend_heatmap(pixel_array, layer.activation_map)
    path = output_dir / f"{prefix}_{layer.name}_activation.png"
    Image.fromarray(blended).save(path)
    paths[f"activation_{layer.name}"] = path

  gc_blend = _blend_heatmap(pixel_array, grad_cam_map, alpha=0.55)
  gc_path = output_dir / f"{prefix}_gradcam.png"
  Image.fromarray(gc_blend).save(gc_path)
  paths["gradcam"] = gc_path

  gc_pure = output_dir / f"{prefix}_gradcam_heat.png"
  Image.fromarray(_colormap_rgb(grad_cam_map)).save(gc_pure)
  paths["gradcam_heat"] = gc_pure
  return paths


def explain_tile(
  model,
  pixel_array: np.ndarray,
  model_config: dict,
  device: str,
  output_dir: str | Path | None = None,
  prefix: str = "explain",
) -> dict[str, Any]:
  """Full explanation package for one tile."""
  import torch

  from .preprocess import tile_to_tensor

  backbone = model_config.get("backbone", "custom")
  norm = "imagenet" if backbone != "custom" else "custom"
  tensor = tile_to_tensor(pixel_array, normalize=norm).unsqueeze(0).to(device)
  model.eval()
  with torch.no_grad():
    logit = model(tensor)
    prob = float(torch.sigmoid(logit).cpu().item())

  layers = extract_layer_activations(model, pixel_array, model_config, device)
  cam = grad_cam(model, pixel_array, model_config, device)

  result: dict[str, Any] = {
    "score": round(prob, 4),
    "backbone": backbone,
    "layers": [layer.to_dict() for layer in layers],
    "grad_cam": {"mean": round(float(cam.mean()), 4), "max": round(float(cam.max()), 4)},
  }

  if output_dir is not None:
    paths = render_explanation_overlays(pixel_array, layers, cam, output_dir, prefix)
    result["paths"] = {k: str(v) for k, v in paths.items()}

  return result
