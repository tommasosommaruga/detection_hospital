# PathAssist — Engineering Guide

A **digital pathology assistant** built for hospital adoption: it saves pathologist
time, surfaces urgent cases first, reduces missed cancers, and fits into real
clinical workflows — not a black-box that outputs "cancer: yes/no."

The ML model is roughly **20–30% of the work**. The rest is triage, explainability,
QC, reporting, audit, integration, and continuous improvement from pathologist
feedback.

---

## 1. What we are building

Hospitals adopt tools that solve **workflow problems**:

| Clinical need | What PathAssist does |
|---------------|----------------------|
| Urgent cases wait too long | **AI-assisted triage** — scan cases, rank by risk, put suspicious slides at the top of the queue |
| Pathologists don't trust black boxes | **Explainable detection** — heatmaps, uncertainty, ranked regions of interest, confidence scores |
| Fatigue causes errors | **QC assistant** — flags blur, poor tissue, staining issues; surfaces model–pathologist disagreement as second-reader signal |
| Whole slides are overwhelming | **WSI prioritization** (roadmap) — find suspicious regions, show only what needs review |
| Grading takes time | **Grading assistant** — advisory severity estimate with rationale (clinical ISUP/Gleason: future) |
| Reporting is repetitive | **Automated report drafting** — structured draft with counts, scores, and review regions |
| Models go stale | **Continuous learning** — pathologist corrections stored and exportable for retraining |
| Rare patterns get missed | **Similar-case retrieval** (roadmap) — flag patterns resembling rare conditions |

**The pathologist always signs off.** The assistant reduces workload and catches
what humans miss under time pressure — it does not replace clinical judgment.

### Target clinical pipeline

```
Whole-slide / patch ingest
        ↓
Quality control (slide + staining)
        ↓
Suspicious region detection (ensemble scoring)
        ↓
Case triage & priority worklist
        ↓
Severity / grade estimate
        ↓
Explainability (heatmap + uncertainty)
        ↓
Report draft
        ↓
Pathologist review (approve / modify / reject)
        ↓
Audit trail → corrections → retrain
```

This is the architecture hospitals find deployable: **triage + second reader +
explainability**, not full automation.

---

## 2. Ten clinical capabilities — status map

| # | Capability | Status in this repo |
|---|------------|---------------------|
| 1 | **AI-assisted triage** | **Built** — `triage.py`, priority bands (URGENT → ROUTINE), worklist UI + CLI |
| 2 | **Explainable detection** | **Built** — heatmaps, uncertainty maps, ranked ROIs, ensemble disagreement |
| 3 | **QC / second reader** | **Partial** — QC flags built; `AuditStore.disagreements()` flags modify/reject; no live LIS hook yet |
| 4 | **Whole-slide prioritization** | **Planned** — tiling works; OpenSlide WSI reader not wired |
| 5 | **Cancer grading assistant** | **Partial** — advisory grade in `grading.py`; not trained on ISUP/Gleason labels |
| 6 | **Multi-modal diagnostics** | **Planned** — images only today; demographics/labs/genomics not integrated |
| 7 | **Automated report drafting** | **Built** — `report.py` + downloadable draft in workstation |
| 8 | **Treatment response prediction** | **Research** — out of scope for current checkpoint |
| 9 | **Rare disease detection** | **Planned** — needs case bank + similarity retrieval |
| 10 | **Continuous learning** | **Built** — review decisions in JSONL, `export-corrections` for retraining corpus |

---

## 3. Engineering completed

### 3.1 Core pipeline (`pathassist/`)

```
tile → QC → score → triage → grade → explain → report → audit (await review)
```

| Module | Clinical role |
|--------|---------------|
| `tiling.py` | Slide → scorable regions (WSI-ready design) |
| `qc.py` | Catch unusable slides before they waste pathologist time |
| `scoring.py` | Suspicious-region detection (ensemble) |
| `ensemble.py` | Robust scoring via weighted multi-model vote |
| `triage.py` | **Queue prioritization** — who gets seen first |
| `grading.py` | Grade/aggressiveness hint for review |
| `explain.py` | Trust-building visual evidence |
| `report.py` | Time-saving draft for sign-off |
| `audit.py` | Safety, traceability, second-reader analytics |
| `cli.py` | Batch processing, worklist, corrections export |

### 3.2 Trained ensemble (`models/lymph_node/ensemble.pt`)

- PatchCamelyon training via `notebooks/train_and_evaluate.ipynb`
- 4 models: ResNet-18 ×3 + EfficientNet-B0
- Holdout ~96.7% accuracy; external test ~85.8% acc / ~73.8% recall
- **High recall on triage matters more than headline accuracy** for catching metastasis

### 3.3 Pathologist workstation (`demo/`)

| View | Doctor-facing purpose |
|------|----------------------|
| Dashboard | System status, model transparency |
| Analyze | Upload case, see ROIs, heatmaps, QC, edit/sign report |
| Worklist | **Today's queue — urgent cases first** |
| Validation | Benchmark performance on real tiles |
| Vision | Roadmap and hospital integration gaps |

### 3.4 Tests & benchmark

- **41 pytest tests** — pipeline + detection + API (`PYTHONPATH=. pytest tests/ -v`)
- `scripts/test_real_pcam.py` — 10 real PatchCamelyon tiles
- `scripts/threshold_sweep.py` — recall/precision tradeoff analysis

### 3.5 Recall-first tuning (applied)

| Setting | Value | Purpose |
|---------|-------|---------|
| `detection_threshold` | 0.25 | Review any case above this |
| `metastasis_threshold` | 0.55 | Strong metastasis call only |
| `force_whole_image` | true | Fix 96×96 PCam patches skipped as background |

**Results on 10 real PCam tiles after tuning:**

| Metric | Before | After |
|--------|--------|-------|
| Metastasis recall | 80% (4/5) | **100% (5/5)** |
| Accuracy | 80% | **90% (9/10)** |
| Missed cancer | `PCAM-META-02` | **None** |

The one remaining flag (`PCAM-NORM-01`) is **borderline** — sent for review, not called metastasis.

---

## 4. How to run

### Setup

```bash
cd /path/to/detection_hospital
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

### Web workstation (pathologist demo)

```bash
PYTHONPATH=. python demo/server.py
```

Open **http://127.0.0.1:8765** — use **Worklist** for triage, **Analyze** for case review.

### Production deployment (Docker)

```bash
cp .env.example .env          # set PATHASSIST_API_KEY and paths
# place models/lymph_node/ensemble.pt (or per-organ under models/<organ>/)
docker compose up --build -d
curl http://localhost:8765/health/ready
```

| Endpoint | Purpose |
|----------|---------|
| `GET /health/live` | Liveness (process up) |
| `GET /health/ready` | Readiness (checkpoint + disk) |
| `GET /api/health` | Full status + metrics |

Write endpoints require header: `X-API-Key: <your-key>` when `PATHASSIST_API_KEY` is set.

```bash
PATHASSIST_ENV=production PATHASSIST_API_KEY=secret PYTHONPATH=. python demo/server.py
```


```bash
python -m pathassist.cli --config config/pcam_test.yaml run \
  --case-id CASE-001 \
  --image outputs/real_pcam/metastasis_00.png \
  --scorer ensemble \
  --checkpoint models/lymph_node/ensemble.pt
```

> `--config` must come **before** `run`.

### CLI — triage queue & second-reader workflow

```bash
python -m pathassist.cli worklist
python -m pathassist.cli review --case-id CASE-001 --decision modify \
  --reviewer "Dr. Smith" --note "ROI 2 is reactive, not malignant"
python -m pathassist.cli export-corrections --output outputs/corrections.csv
```

### Real-tile benchmark

```bash
pip install datasets huggingface_hub   # first time only
python scripts/test_real_pcam.py
```

### Retrain ensemble (Google Colab GPU)

1. Open `notebooks/train_and_evaluate.ipynb` in [Google Colab](https://colab.research.google.com)
2. **Runtime → Change runtime type → GPU**
3. Section 2: set `ORGAN_ID` (`lymph_node` for PCam, or `breast`, `gastrointestinal`, …)
4. `RUN_MODE = 'deploy'`, `MACHINE_PROFILE = 'colab_gpu'`
5. Run all — checkpoint saves to `models/<ORGAN_ID>/ensemble.pt`
6. Download (section 12) and copy into this repo at the same path

For non–lymph-node organs: `DATA_SOURCE = 'folder'`, mount Drive, set `DATA_DIR` with `benign/` + `malignant/` tiles.

Notebook `notebooks/train_and_evaluate.ipynb` — `RUN_MODE = 'deploy'` for fast
training; `full` for cross-validation.

---

## 5. Repository layout

```
config/                 Triage thresholds, QC cutoffs, report templates
models/lymph_node/ensemble.pt   Default lymph-node ensemble
models/<organ>/ensemble.pt      One checkpoint per organ (see config/organs.yaml)
pathassist/             Clinical pipeline modules
demo/                   Pathologist workstation (FastAPI + UI)
notebooks/              Training & evaluation
scripts/                Real-data benchmarks
outputs/                Heatmaps, reports, demo results
runs/                   Audit log (machine + human decisions)
tests/                  Automated regression tests
```

---

## 6. Limitations (honest)

### Production-ready today

- Docker + docker-compose with health probes
- API key auth on write endpoints
- Environment-based config (`.env`)
- Recall-first triage (100% metastasis recall on 10-tile benchmark)
- WSI thumbnail ingest when OpenSlide is installed
- CI test suite (48 tests)
- Structured logging

### Still required before hospital go-live

- **Prospective validation** on local scanner, stain, and patient population
- **Regulatory path** (CE/FDA) if marketed as a medical device
- **LIS / EHR integration** — case ingest, report export, identity management
- **Privacy & security** — HIPAA/GDPR, encryption, access control
- **Monitoring** — drift detection when scanner or population changes

### Current technical gaps

| Gap | Impact |
|-----|--------|
| No OpenSlide WSI (`.svs`/`.ndpi`) | Cannot scan full gigapixel slides yet — **#1 engineering priority** |
| Tile-level PCam model | Domain shift on hospital data; test recall ~74% |
| No bounding-box cell detector | ROIs are tiles, not individual cells (YOLO-style: future) |
| No similar-case retrieval | Rare-disease assist not available |
| No multi-modal inputs | Labs, history, genomics not used |
| Localhost demo only | Not hardened for hospital IT deployment |
| Advisory grade only | Not ISUP/Gleason-trained |

### What this is not (yet)

- Not a replacement pathologist
- Not cleared for autonomous diagnosis
- Not validated for a specific hospital's intended use

---

## 7. Roadmap — what to build next

Ordered by **clinical value × feasibility**:

### Phase A — Make it useful on real hospital slides (highest impact)

1. **OpenSlide WSI ingestion** — scan whole slide, tile pyramid, feed existing pipeline
2. **Region-first UI** — jump to suspicious coordinates; minimize pan-and-scan time
3. **Recall tuning** — lower false negatives on triage (threshold calibration, stain aug)
4. **Second-reader dashboard** — dedicated view for model vs pathologist disagreement

### Phase B — Deeper clinical assist

5. **Cell-level detection** — YOLO or Faster R-CNN on annotated nuclei (if labels exist)
6. **Clinical grading** — train on PANDA / hospital ISUP labels
7. **Similar historical cases** — embedding index for rare-pattern alerts
8. **LIS adapter** — HL7/FHIR or CSV bridge for real queue ingest

### Phase C — Hospital-grade product

9. **Auth, roles, audit immutability**
10. **Multi-site validation study design + metrics**
11. **Scheduled retrain** from `export-corrections` pipeline
12. **Multi-modal** — structured EHR fields alongside imaging

### Beyond ML (often harder than the model)

- Expert annotation budget and governance
- Hospital IT security review
- Change management — pathologist training and trust building
- Legal/commercial model (SaaS vs on-prem)

---

## 8. Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: pathassist` | Repo root + `PYTHONPATH=.` |
| No checkpoint | Train notebook §12 or place `models/<organ>/ensemble.pt` |
| Port 8765 in use | `lsof -ti:8765 \| xargs kill` |
| CLI config ignored | `--config` before subcommand |
| Poor scores on PCam tiles | Use `config/pcam_test.yaml` (96×96) |

---

## 9. Command cheat sheet

```bash
PYTHONPATH=. pytest tests/ -q
PYTHONPATH=. python demo/server.py
python -m pathassist.cli worklist
python -m pathassist.cli --config config/pcam_test.yaml run \
  --case-id X --image tile.png --scorer ensemble --checkpoint models/ensemble.pt
python scripts/test_real_pcam.py
```

---

*PathAssist: triage faster, explain clearly, draft reports, learn from corrections —
built so pathologists spend time on cases that matter.*
