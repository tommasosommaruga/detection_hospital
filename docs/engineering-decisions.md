# PathAssist — Engineering decisions & status log

Living record of what we built, what we chose, and why.  
**Last updated:** 2026-07-05

Use this when resuming work, onboarding, or planning Swiss hospital deployment.

---

## 1. Product direction

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary user | Pathologist (decision support) | Not autonomous diagnosis; human sign-off always |
| Core value | **Triage + second reader + explainability** | Hospitals adopt workflow tools, not raw classifiers |
| First clinical target | **Sentinel lymph node metastasis** (`lymph_node`) | Best public-data proxy for real LN staging (CAMELYON / PCam) |
| Second target | Colon CRC screening (`gastrointestinal`) | NCT-CRC-HE; only after lymph demo is solid |
| Out of scope (for now) | Microbiology, cytology (Pap), MRI, full WSI prod | Different modality/product; needs OpenSlide pipeline first |
| Regulatory stance | Research / demo prototype | MDR, GDPR, LIS, prospective validation = hospital-side later |
| Metric priority | **Recall over accuracy** | Missed metastasis >> extra false alarms (pathologist reviews anyway) |

---

## 2. Hospital fit (Swiss / EU digital pathology)

### What fits today

- **H&E patch classifier** for lymph node and colon mucosa
- **Recall-first thresholds** in YAML + UI tuning
- **Workstation demo**: Analyze, Worklist, Validation, audit/corrections export
- **On-prem friendly**: CPU inference, Docker path in repo

### Gaps before real hospital go-live

| Gap | Priority | Notes |
|-----|----------|-------|
| **WSI ingest** (`.svs`, `.ndpi`, Leica/Philips/Hamamatsu) | **Partial** | Full-res tiling via OpenSlide when installed; thumbnail fallback otherwise |
| **Local validation** on hospital scanner + stain | #1 | Public pretrain + threshold retune minimum |
| LIS / EMR integration | #2 | Not started |
| Prospective clinical study | #2 | Required for cleared device |
| MDR / Swiss FADP compliance | Process | Engineering enables; legal/clinical owns |

### Training data vs hospital data

| Organ | Public training set | Correct for task? | Hospital gap |
|-------|---------------------|-------------------|--------------|
| `lymph_node` | PatchCamelyon → CAMELYON16 sentinel LN H&E | **Yes** — canonical benchmark | Scanner, population, WSI tiling |
| `gastrointestinal` | NCT-CRC-HE (German CRC, Macenko-normalized) | **Yes** for tumor vs normal mucosa | Stain shift; not grading/subtyping |
| `breast` | BACH (~400 images) | Demo only | Too small for deploy |
| `pulmonary` | LC25000 lung subset | Moderate | Not priority |

**When Swiss data arrives:** threshold retune first → optional fine-tune on local patches at **same geometry** (96×96 @ ~2.43 µm/px for lymph; 224×224 @ 0.5 MPP for GI). Do not throw away public pretraining.

---

## 3. Training architecture (Colab notebook)

**Single source of truth:** `notebooks/train_and_evaluate.ipynb`  
**Version marker:** `NOTEBOOK_VERSION = '2026-07-05-prod-ensemble'`

### Decisions

| Topic | Decision |
|-------|----------|
| Colab packaging | **Self-contained notebook** — no `import pathassist` required in Colab |
| Default organ for prod training | `ORGAN_ID = 'lymph_node'` |
| Default run mode | `RUN_MODE = 'deploy'` (~1–2 h T4); `full` for CV + deploy (~6–8 h) |
| Colab profile | `MACHINE_PROFILE = 'colab_gpu'` (32k tiles); `colab_gpu_high` for 65k |
| Ensemble | **ResNet18 + ResNet34 + EfficientNet-B0 + DenseNet121** (soft vote + val weights) |
| Optimizer | AdamW, `grad_clip_norm=1.0`, AMP on GPU |
| Loss | BCE — **no recall in loss**; recall via threshold at inference |
| Checkpoint selection | `checkpoint_metric='val_recall_at_threshold'` (not val_loss alone) |
| Lymph checkpoint threshold | `0.25` (matches `config/pcam_test.yaml`) |
| GI checkpoint threshold | `0.30` (matches `config/gi_crc.yaml`) |
| Early stop | On checkpoint metric (recall @ threshold) |
| Class balance (GI) | **Balanced NORM + TUM** per class — stream order is wrong |
| GI Colab cap | 8192 tiles, batch 64 + AMP (RAM/session limits) |
| SMOTE on images | **No** |
| Weighted loss for imbalance | Only if training on natural prevalence — not for balanced GI subsample |
| ViT / foundation models | **Deferred** — Colab limits; 96px patches; current ensemble sufficient on PCam |

### Bugs fixed in notebook (do not revert)

1. **GI HF images** — `row['image']` is dict not PIL → `_as_pil_image()` helper  
2. **GI all-negative labels** — stream had NORM before TUM → balanced per-class loader  
3. **Colab disconnects** — tile caps, AMP, `cudnn.benchmark`, `channels_last`  
4. **Checkpoint save** — duplicate `weight` line syntax error fixed  

### Production code mirrored in `pathassist/`

- `pathassist/backbone.py` — `resnet34`, `densenet121` builders for prod checkpoint load  
- `pathassist/training_data.py` — GI balanced loader for local training path  
- `pathassist/ensemble.py` — `VotingEnsemble` + `load_ensemble_checkpoint()` (format 3)  
- `pathassist/scoring.py` — `EnsembleScorer` wraps ensemble for tile/case inference  

### Ensemble design — one `ensemble.pt`, four voters

We do **not** train one monolithic network. We train **four independent binary classifiers** on the same organ tiles, then save them together in a single PyTorch checkpoint. At inference, every tile is scored by all four; their outputs are combined by weighted voting.

#### The four members (`ENSEMBLE_MEMBERS` in notebook)

| Member name | Backbone | Notes |
|-------------|----------|-------|
| `resnet18` | ResNet-18 | ImageNet pretrained, seed 42 |
| `resnet34` | ResNet-34 | Different depth + light color jitter |
| `efficientnet_b0` | EfficientNet-B0 | Different architecture family |
| `densenet121` | DenseNet-121 | Different connectivity pattern |

Diversity comes from **architecture + seed + augmentation + LR**, not from one shared weights tensor. Each member has its own `state_dict` (full learned weights).

#### Voting at inference (`VOTE_MODE = 'soft'` default)

For each tile, each member outputs metastasis probability `p ∈ [0,1]` (sigmoid of logit).

- **Soft vote:** `final_prob = Σ (p_i × w_i) / Σ w_i`  
- **Hard vote (optional):** each member votes positive if `p_i ≥ 0.5`; majority wins  
- **Disagreement:** std dev across the four `p_i` values → uncertainty signal in UI (`pathassist/scoring.py`)

Vote weights `w_i` are **not equal by default** (`USE_VAL_WEIGHTS = True`). During training, each model’s weight is decayed every epoch by its validation error rate (`vote_weight *= 1 - error_rate`, floor `WEIGHT_FLOOR = 0.05`). Models that make more mistakes get less influence in the final vote. Final `weight` per member is stored in the checkpoint.

Implementation: `pathassist/ensemble.py` → `VotingEnsemble.predict_batch()`.

#### Why one file?

`torch.save()` writes a **Python dict bundle**, not a single `nn.Module`. Checkpoint **format 3** (`ENSEMBLE_CHECKPOINT_FORMAT = 3`):

```python
{
  "checkpoint_format": 3,
  "organ_id": "lymph_node",
  "tile_size": 96,
  "vote_mode": "soft",
  "metrics": { "test_acc": ..., "test_recall": ... },
  "members": [
    {
      "name": "resnet18",
      "model_config": { "backbone": "resnet18", ... },  # how to rebuild arch
      "state_dict": { ... },                             # all weights for this model
      "weight": 0.87,                                    # voting weight
      "val_acc": 0.94,
      "val_recall": ...,
      "seed": 42,
    },
    # ... three more members
  ],
}
```

Saved path: `models/<organ_id>/ensemble.pt` (see `resolve_ensemble_path()` in notebook).  
There are **no separate `.pt` files per member** in production — one file ≈ sum of four models (tens–low hundreds of MB).

#### Load path at runtime

1. `EnsembleScorer` / CLI `--scorer ensemble` → `load_ensemble_checkpoint(path)`  
2. For each `members[]` entry: `build_model(model_config)` → `load_state_dict` → `model.eval()`  
3. Wrap in `VotingEnsemble(members, vote_mode=...)`  
4. Each tile batch: run all four → weighted soft vote → `case_score` + `disagreement`

`pretrained=False` on load — ImageNet weights are already inside each member’s `state_dict`; we do not re-download torchvision weights.

#### Explainability caveat

Grad-CAM / layer heatmaps (`pathassist/explain.py`, `pipeline.py`) run on **one ensemble member at a time** (default `member_index=0`), because activation maps are per-network. The **case decision** still uses all four voters.

#### Training → deploy flow

```
notebook: train resnet18, resnet34, efficientnet_b0, densenet121
    → torch.save(checkpoint) → models/lymph_node/ensemble.pt
demo/CLI: EnsembleScorer(checkpoint)
    → 4 models in memory → weighted vote per tile → triage / validation UI
```

Old single-model checkpoint `ensemble_old_training_04_07.pt` predates this format — **superseded**.

---

## 4. Model metrics (lymph node, July 2026 run)

Official **PatchCamelyon test** split (32,768 tiles), ensemble after `2026-07-05-prod-ensemble`:

| Operating point | Accuracy | Recall | Notes |
|-----------------|----------|--------|-------|
| @ 0.5 (research default) | ~90.0% | ~85.8% | Notebook held-out eval prints this |
| @ **0.25 (production)** | — | **~95.7%** | ~697 FN / ~16k positive tiles; matches deploy threshold |

**Do not compare** deployment holdout confusion matrix (~3k tiles, ~96% recall) to full test — different splits.

Old checkpoint (`ensemble_old_training_04_07.pt`) predates ensemble + recall-checkpoint — **superseded**.

---

## 5. Inference & thresholds

### Config files

| Organ | Config | `detection_threshold` | `metastasis_threshold` | `min_review_score` |
|-------|--------|----------------------|--------------------------|-------------------|
| Lymph | `config/pcam_test.yaml` | **0.25** | 0.55 | 0.15 |
| GI | `config/gi_crc.yaml` | **0.30** | 0.55 (default) | 0.15 |

### Classification rules (`pathassist/detection.py`)

- `score >= metastasis_threshold` → **metastasis** (strong call)  
- `score >= detection_threshold` → **borderline** (review)  
- `score >= min_review_score` → **borderline** (low-score review band)  
- else → **normal**  

**Borderline counts as positive** for recall metrics (pathologist still reviews).

### Workstation default organ

`config/organs.yaml` still has `default_organ: gastrointestinal`.  
**Hospital pitch:** switch to `lymph_node` when demoing LN workflow.

---

## 6. Demo / workstation engineering

### Validation page

- Metrics **recomputed** from stored `case_score` using organ YAML (not stale meta flags)  
- Side-by-side: **production recall-first** vs **research @ 0.5**  
- **Threshold tuning UI** (2026-07-05): sliders for `detection_threshold`, `metastasis_threshold`, `min_review_score`  
  - Live metric update (no re-inference)  
  - **Use for Analyze** → `POST /api/session/triage` applies to session  
  - **Reset to YAML** → clears session overrides  

### Key API routes

| Route | Purpose |
|-------|---------|
| `GET /api/validation?detection_threshold=…` | Benchmark metrics with preview thresholds |
| `GET/POST /api/session/triage` | Read/apply session threshold overrides |
| `POST /api/analyze/upload` | Uses organ config + session overrides via `_pick_config()` |

### Files touched

- `demo/logic.py` — `compute_validation_metrics`, `merge_triage_overrides`  
- `demo/server.py` — validation config, session triage  
- `demo/static/app.js` — validation UI + threshold sliders  
- `demo/static/app.css` — tuner styles  

---

## 7. What to train next (decision)

```
Priority 1: lymph_node  — retrain if checkpoint lacks NOTEBOOK_VERSION 2026-07-05-prod-ensemble
Priority 2: gastrointestinal — only if colon demo needed; must use fixed balanced loader
Skip: breast, pulmonary, other organs until WSI + one organ is hospital-demo ready
```

### Lymph deploy checklist

1. Colab: `ORGAN_ID='lymph_node'`, `RUN_MODE='deploy'`, verify `NOTEBOOK_VERSION` in logs  
2. Save → `models/lymph_node/ensemble.pt`  
3. Run demo Validation → tune thresholds → **Use for Analyze**  
4. Optional: `RUN_MODE='full'` once for CV report  

### GI retrain checklist

1. Only after lymph is deployed  
2. Confirm balanced NORM+TUM in loader logs  
3. Deploy to `models/gastrointestinal/ensemble.pt`  
4. Validate on `CRC_VAL_HE_7K` holdout via benchmark import (notebook skips GI `EVAL_TEST`)  

---

## 8. Explicit non-decisions (avoid rework)

| Rejected / deferred | Why |
|---------------------|-----|
| 100% recall @ 0.5 on full PCam | Unrealistic; use threshold tuning |
| Recall in training loss | Threshold handles prevalence at inference |
| Training all 20 organ slots | No data; dilutes focus |
| Microbiology module | Different product |
| Colab dependency on repo package | User trains from notebook only |
| Browser-only threshold tuning without backend | `classify_case()` is source of truth in Python |

---

## 9. File map (quick reference)

| Path | Role |
|------|------|
| `notebooks/train_and_evaluate.ipynb` | **Train all organs (Colab)** |
| `config/pcam_test.yaml` | Lymph prod thresholds + 96px tiling |
| `config/gi_crc.yaml` | GI prod thresholds + 224px tiling |
| `config/training_datasets.yaml` | Organ → HF dataset catalog |
| `config/organs.yaml` | Organ registry + checkpoint paths |
| `models/lymph_node/ensemble.pt` | Prod lymph checkpoint (target path) |
| `models/lymph_node/ensemble_old_training_04_07.pt` | **Superseded** |
| `pathassist/detection.py` | Threshold logic + `classify_case` |
| `pathassist/backbone.py` | Ensemble member architecture builders |
| `pathassist/ensemble.py` | `VotingEnsemble`, checkpoint format 3 loader |
| `pathassist/scoring.py` | `EnsembleScorer` — tile scoring + disagreement |
| `demo/` | Pathologist workstation |
| `guide.md` | Full runbook + roadmap |
| `scripts/threshold_sweep.py` | Offline threshold analysis |
| `.firecrawl/hospital-fit/` | Research notes on hospital data fit |

---

## 10. Open engineering (not done)

- [x] Full-res WSI tiling (`pathassist/wsi.py` + `resolve_case_tiles`) — needs OpenSlide on server
- [x] Demo/CLI accept `.svs` / `.ndpi` uploads
- [ ] Persist WSI level-0 coords in ROI metadata (for deep-zoom viewer)
- [ ] `default_organ: lymph_node` for hospital demos  
- [ ] Persist session thresholds to YAML (optional; today session-only)  
- [ ] Notebook: print test recall @ 0.25 and @ 0.5 side by side  
- [ ] GI training above 8k tiles (needs more RAM or streaming training)  
- [ ] Local Swiss slide validation protocol (document when data exists)  

---

## 11. One-paragraph summary

PathAssist is a **recall-first H&E patch triage assistant** aimed first at **sentinel lymph node metastasis** (PatchCamelyon/CAMELYON), with colon CRC as a secondary organ. Training is done in a **standalone Colab notebook** using a **four-backbone ensemble** (ResNet-18, ResNet-34, EfficientNet-B0, DenseNet-121) saved as **one `ensemble.pt`** with weighted soft voting at inference; checkpoints are selected by **validation recall at production threshold** (0.25 lymph / 0.30 GI). The demo workstation applies the same rules as YAML config, lets pathologists **tune thresholds on the Validation page**, and exports corrections for future retraining. Swiss hospital production requires **WSI ingest**, **local scanner validation**, and **threshold retune on hospital data** — public pretraining is the correct starting point, not the final deploy artifact.
