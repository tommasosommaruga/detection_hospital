"""Public pathology datasets comparable to PatchCamelyon (PCam).

Similarity scores are heuristic composites (task, organ, stain, resolution, label type).
Use `compare_image_fingerprints` for quantitative stain/colour similarity on actual tiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Reference: user's training data (PatchCamelyon 96×96 H&E lymph-node metastasis tiles).
REFERENCE_ID = "patchcamelyon"


@dataclass(frozen=True)
class DatasetProfile:
  id: str
  name: str
  source: str
  task: str
  organ: str
  stain: str
  label_type: str
  typical_tile_px: str
  access: str
  # Heuristic similarity to PatchCamelyon in [0, 1] (higher = closer).
  task_similarity: float
  stain_similarity: float
  resolution_similarity: float
  notes: str

  @property
  def composite_similarity(self) -> float:
    return round(
      0.45 * self.task_similarity
      + 0.30 * self.stain_similarity
      + 0.25 * self.resolution_similarity,
      3,
    )


CATALOG: list[DatasetProfile] = [
  DatasetProfile(
    id="patchcamelyon",
    name="PatchCamelyon (PCam)",
    source="1aurent/PatchCamelyon (Hugging Face)",
    task="Binary metastasis in lymph-node patches",
    organ="Lymph node",
    stain="H&E",
    label_type="Tile-level binary",
    typical_tile_px="96×96",
    access="HF datasets",
    task_similarity=1.0,
    stain_similarity=1.0,
    resolution_similarity=1.0,
    notes="Your training benchmark — 327,680 tiles from CAMELYON16 WSIs.",
  ),
  DatasetProfile(
    id="camelyon16",
    name="CAMELYON16",
    source="camelyon16.grand-challenge.org",
    task="Metastasis detection in lymph-node WSI",
    organ="Lymph node",
    stain="H&E",
    label_type="Slide + pixel annotations",
    typical_tile_px="WSI (multi-GB)",
    access="Registration + GigaDB",
    task_similarity=0.98,
    stain_similarity=0.95,
    resolution_similarity=0.35,
    notes="Parent dataset for PCam; same biology, different format (WSI vs tiles).",
  ),
  DatasetProfile(
    id="camelyon17",
    name="CAMELYON17",
    source="camelyon17.grand-challenge.org",
    task="Lymph-node metastasis + patient-level labels",
    organ="Lymph node",
    stain="H&E",
    label_type="Slide-level (+ hospital one-hot)",
    typical_tile_px="WSI",
    access="Registration",
    task_similarity=0.95,
    stain_similarity=0.95,
    resolution_similarity=0.35,
    notes="Adds domain shift (multiple hospitals). Best for external validation.",
  ),
  DatasetProfile(
    id="camelyon_plus",
    name="Camelyon+ (refined)",
    source="ScienceDB / Nature Scientific Data 2025",
    task="4-class metastasis (neg / micro / macro / ITC)",
    organ="Lymph node",
    stain="H&E",
    label_type="Refined slide + pixel labels",
    typical_tile_px="WSI",
    access="ScienceDB 10.57760/sciencedb.16442",
    task_similarity=0.96,
    stain_similarity=0.96,
    resolution_similarity=0.35,
    notes="Cleaned CAMELYON16/17; better labels, still WSI-level.",
  ),
  DatasetProfile(
    id="bach",
    name="BACH (ICIAR 2018)",
    source="Hugging Face: 1aurent/BACH",
    task="Breast histology: normal / benign / in situ / invasive",
    organ="Breast",
    stain="H&E",
    label_type="Whole-image 4-class",
    typical_tile_px="2048×1536",
    access="HF / ICIAR",
    task_similarity=0.35,
    stain_similarity=0.88,
    resolution_similarity=0.15,
    notes="Same stain, different organ & task; useful stain-transfer fine-tune only.",
  ),
  DatasetProfile(
    id="breakhis",
    name="BreakHis",
    source="Kaggle / UFPR",
    task="Breast tumour benign vs malignant",
    organ="Breast",
    stain="H&E",
    label_type="Patch 4 magnifications",
    typical_tile_px="700×460 – 2048×1536",
    access="Kaggle",
    task_similarity=0.40,
    stain_similarity=0.85,
    resolution_similarity=0.20,
    notes="Magnification varies (40×–400×); domain gap vs 10× lymph tiles.",
  ),
  DatasetProfile(
    id="nct_crc_he",
    name="NCT-CRC-HE-100K",
    source="Zenodo / HF mirrors",
    task="Colon tissue type (9 classes)",
    organ="Colon",
    stain="H&E",
    label_type="Patch 224×224",
    typical_tile_px="224×224",
    access="Zenodo",
    task_similarity=0.20,
    stain_similarity=0.80,
    resolution_similarity=0.55,
    notes="Similar patch workflow; different organ and labels.",
  ),
  DatasetProfile(
    id="panda",
    name="PANDA",
    source="panda.grand-challenge.org",
    task="Prostate ISUP grade",
    organ="Prostate",
    stain="H&E",
    label_type="Slide-level grade",
    typical_tile_px="WSI",
    access="Kaggle / challenge",
    task_similarity=0.15,
    stain_similarity=0.82,
    resolution_similarity=0.30,
    notes="Grading not metastasis detection; future clinical extension.",
  ),
  DatasetProfile(
    id="idc_breast",
    name="IDC Breast Histopathology",
    source="Kaggle / HF splits",
    task="Invasive ductal carcinoma patch detection",
    organ="Breast",
    stain="H&E",
    label_type="50×50 IDC vs background",
    typical_tile_px="198×198",
    access="Kaggle",
    task_similarity=0.45,
    stain_similarity=0.86,
    resolution_similarity=0.70,
    notes="Small patches; closer resolution to PCam than BACH.",
  ),
]


def catalog_as_dicts() -> list[dict[str, Any]]:
  return [
    {
      "id": d.id,
      "name": d.name,
      "source": d.source,
      "task": d.task,
      "organ": d.organ,
      "composite_similarity": d.composite_similarity,
      "task_similarity": d.task_similarity,
      "stain_similarity": d.stain_similarity,
      "resolution_similarity": d.resolution_similarity,
      "access": d.access,
      "notes": d.notes,
    }
    for d in sorted(CATALOG, key=lambda x: -x.composite_similarity)
  ]


def image_fingerprint(image: np.ndarray) -> dict[str, float]:
  """Colour/stain fingerprint comparable across datasets (no ML required)."""
  flat = image.reshape(-1, 3).astype(np.float32)
  mean_rgb = flat.mean(axis=0)
  std_rgb = flat.std(axis=0)
  gray = image.mean(axis=2).astype(np.float32)
  tissue_frac = float((gray < 220).mean())
  stain_range = float(mean_rgb.max() - mean_rgb.min())
  # H&E proxy: eosin (pink) vs hematoxylin (purple-blue) ratio
  r, g, b = mean_rgb
  eosin_proxy = float((r + g) / 2.0 - b)
  return {
    "mean_r": float(r),
    "mean_g": float(g),
    "mean_b": float(b),
    "std_r": float(std_rgb[0]),
    "std_g": float(std_rgb[1]),
    "std_b": float(std_rgb[2]),
    "tissue_fraction": tissue_frac,
    "stain_range": stain_range,
    "eosin_proxy": eosin_proxy,
    "brightness": float(gray.mean()),
  }


def fingerprint_vector(fp: dict[str, float]) -> np.ndarray:
  keys = [
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "tissue_fraction", "stain_range", "eosin_proxy", "brightness",
  ]
  return np.array([fp[k] for k in keys], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
  denom = float(np.linalg.norm(a) * np.linalg.norm(b))
  if denom < 1e-8:
    return 0.0
  return float(np.dot(a, b) / denom)


def compare_image_fingerprints(ref: np.ndarray, other: np.ndarray) -> dict[str, float]:
  """Quantitative stain/colour similarity between two tiles."""
  fp_ref = image_fingerprint(ref)
  fp_other = image_fingerprint(other)
  vec_ref = fingerprint_vector(fp_ref)
  vec_other = fingerprint_vector(fp_other)
  return {
    "cosine_similarity": round(cosine_similarity(vec_ref, vec_other), 4),
    "reference": fp_ref,
    "other": fp_other,
  }


def summarize_tiles(images: list[np.ndarray]) -> dict[str, float]:
  if not images:
    return {}
  fps = [fingerprint_vector(image_fingerprint(img)) for img in images]
  mean_vec = np.mean(np.stack(fps), axis=0)
  return {k: float(v) for k, v in zip(
    ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
     "tissue_fraction", "stain_range", "eosin_proxy", "brightness"],
    mean_vec,
  )}
