"""Configurable tile classifiers used by the ensemble checkpoint loader.

Supports the same architectures trained in notebooks/train_and_evaluate.ipynb so
GPU-trained ensemble.pt files load directly for CPU inference.
"""

from __future__ import annotations

import copy

import torch.nn as nn


def merge_config(base: dict, overrides: dict) -> dict:
  out = copy.deepcopy(base)
  out.update(overrides)
  return out


def get_model_norm(cfg: dict) -> str:
  return "imagenet" if cfg.get("backbone", "custom") != "custom" else "custom"


class TileClassifier(nn.Module):
  """Small custom CNN — fast and CPU-friendly."""

  def __init__(self, cfg: dict) -> None:
    super().__init__()
    self.cfg = cfg
    channels = cfg["conv_channels"]
    kernel = cfg["kernel_size"]
    use_bn = cfg.get("use_batch_norm", True)
    pool_type = cfg.get("pool", "max")
    layers, in_ch = [], 3
    for out_ch in channels:
      layers += [nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2)]
      if use_bn:
        layers += [nn.BatchNorm2d(out_ch)]
      layers += [nn.ReLU(inplace=True)]
      layers += [
        nn.MaxPool2d(2) if pool_type == "max" else nn.AvgPool2d(2)
      ]
      in_ch = out_ch
    layers += [nn.AdaptiveAvgPool2d(1)]
    self.features = nn.Sequential(*layers)
    drop = cfg.get("dropout", 0.0)
    hidden = cfg.get("head_hidden", 0)
    if hidden > 0:
      self.head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(channels[-1], hidden),
        nn.ReLU(),
        nn.Dropout(drop),
        nn.Linear(hidden, 1),
      )
    else:
      self.head = nn.Sequential(
        nn.Flatten(), nn.Dropout(drop), nn.Linear(channels[-1], 1)
      )

  def forward(self, x):
    return self.head(self.features(x)).squeeze(-1)


class PretrainedBackbone(nn.Module):
  """Torchvision backbone with a dropout classification head."""

  def __init__(self, net: nn.Module, cfg: dict) -> None:
    super().__init__()
    self.cfg = cfg
    self.net = net

  def forward(self, x):
    return self.net(x).squeeze(-1)


def _build_classifier_head(feat_dim: int, cfg: dict) -> nn.Module:
  drop = cfg.get("dropout", 0.5)
  hidden = cfg.get("head_hidden", 256)
  return nn.Sequential(
    nn.Dropout(drop),
    nn.Linear(feat_dim, hidden),
    nn.ReLU(inplace=True),
    nn.Dropout(drop),
    nn.Linear(hidden, 1),
  )


def _build_resnet18(cfg: dict) -> PretrainedBackbone:
  from torchvision.models import ResNet18_Weights, resnet18

  weights = ResNet18_Weights.IMAGENET1K_V1 if cfg.get("pretrained", True) else None
  net = resnet18(weights=weights)
  feat_dim = net.fc.in_features
  net.fc = _build_classifier_head(feat_dim, cfg)
  if cfg.get("freeze_backbone", False):
    for name, param in net.named_parameters():
      if not name.startswith("fc."):
        param.requires_grad = False
  return PretrainedBackbone(net, cfg)


def _build_efficientnet_b0(cfg: dict) -> PretrainedBackbone:
  from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

  weights = (
    EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg.get("pretrained", True) else None
  )
  net = efficientnet_b0(weights=weights)
  feat_dim = net.classifier[1].in_features
  net.classifier = _build_classifier_head(feat_dim, cfg)
  if cfg.get("freeze_backbone", False):
    for param in net.features.parameters():
      param.requires_grad = False
  return PretrainedBackbone(net, cfg)


def _build_resnet34(cfg: dict) -> PretrainedBackbone:
  from torchvision.models import ResNet34_Weights, resnet34

  weights = ResNet34_Weights.IMAGENET1K_V1 if cfg.get("pretrained", True) else None
  net = resnet34(weights=weights)
  feat_dim = net.fc.in_features
  net.fc = _build_classifier_head(feat_dim, cfg)
  if cfg.get("freeze_backbone", False):
    for name, param in net.named_parameters():
      if not name.startswith("fc."):
        param.requires_grad = False
  return PretrainedBackbone(net, cfg)


def _build_densenet121(cfg: dict) -> PretrainedBackbone:
  from torchvision.models import DenseNet121_Weights, densenet121

  weights = DenseNet121_Weights.IMAGENET1K_V1 if cfg.get("pretrained", True) else None
  net = densenet121(weights=weights)
  feat_dim = net.classifier.in_features
  net.classifier = _build_classifier_head(feat_dim, cfg)
  if cfg.get("freeze_backbone", False):
    for param in net.features.parameters():
      param.requires_grad = False
  return PretrainedBackbone(net, cfg)


def build_model(cfg: dict) -> nn.Module:
  """Factory: custom CNN or pretrained backbone."""
  backbone = cfg.get("backbone", "custom")
  if backbone == "custom":
    return TileClassifier(cfg)
  if backbone == "resnet18":
    return _build_resnet18(cfg)
  if backbone == "resnet34":
    return _build_resnet34(cfg)
  if backbone == "efficientnet_b0":
    return _build_efficientnet_b0(cfg)
  if backbone == "densenet121":
    return _build_densenet121(cfg)
  raise ValueError(f"Unknown backbone: {backbone!r}")
