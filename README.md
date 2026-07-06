# PathAssist — Digital Pathology Assistant

A **hospital-oriented pathology assistant** that helps doctors work faster and
safer — not a toy classifier that prints "cancer: yes/no."

It is built around what hospitals actually need:

- **Triage** — scan cases, prioritize urgent slides, reduce missed critical diagnoses
- **Explainability** — heatmaps, uncertainty, ranked regions so pathologists can verify quickly
- **Second reader** — flag disagreements between AI and pathologist after sign-off
- **QC** — catch blurry, understained, or low-tissue slides before they waste review time
- **Report drafting** — structured draft the pathologist edits and signs
- **Continuous learning** — corrections exported for retraining

The pathologist remains in control. The assistant removes repetitive visual search,
surfaces what matters first, and creates an audit trail for safety and improvement.

## Pipeline

```
whole-slide image
    -> tiling            split slide into tiles, drop background
    -> quality control    blur / tissue coverage / staining checks
    -> region scoring     ensemble or single model scores each tile
    -> case triage        priority band, uncertainty flags, ranked regions
    -> severity grading   advisory grade estimate (not clinical sign-out)
    -> explainability     score heatmap + uncertainty map
    -> report drafting    structured draft with recommended actions
    -> human approval     pathologist decision recorded in audit trail
    -> continuous learning export corrections for retraining
```

Each stage is a separate module and is independently testable. The only piece you
replace to go from toy to real is the **scorer** (`scoring.py`); everything else
stays the same.

## Layout

```
config/default.yaml     all thresholds & workflow knobs (nothing clinical hardcoded)
pathassist/
  tiling.py             cut a slide into tiles (swap in OpenSlide for real WSIs)
  scoring.py            Scorer protocol + Dummy / Torch / EnsembleScorer
  backbone.py           configurable CNN + ResNet/EfficientNet builders
  ensemble.py           weighted ensemble loader (ensemble.pt format 3)
  qc.py                 slide quality checks
  grading.py            advisory severity estimate
  model.py              the CNN + CPU-loadable checkpoint save/load
  preprocess.py         one transform shared by training and inference
  device.py             auto GPU/CPU/MPS selection
  train.py              training entry point (GPU on Colab, CPU anywhere)
  triage.py             case score, priority banding, worklist ranking
  explain.py            heatmap overlay rendering
  report.py             report draft text
  audit.py              append-only audit trail + pathologist decisions
  pipeline.py           wires the stages together
  cli.py                command-line interface
  synthetic.py          fake slides + labelled tiles for dev/tests
notebooks/colab_train.ipynb   train on a Colab GPU, download a CPU checkpoint
tests/                  end-to-end smoke tests
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# End-to-end demo on a synthetic slide, no model needed:
python -m pathassist.cli demo --case-id DEMO-001

# Record a pathologist decision, then view the triage worklist:
python -m pathassist.cli review --case-id DEMO-001 --decision approve --reviewer "Dr. Smith"
python -m pathassist.cli worklist
```

Outputs (heatmap + report draft) land in `outputs/`; the audit trail is in `runs/`.

## Tests

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest tests/ -v
```

48 tests — pipeline, detection, API, auth, WSI helpers. CI runs on push via GitHub Actions.

## Production deployment

```bash
cp .env.example .env
# Edit PATHASSIST_API_KEY; mount models/ (per-organ checkpoints)
docker compose up --build -d
```

See [guide.md](guide.md) for full runbook, recall tuning, and hospital integration roadmap.

## Train on GPU (Colab), run on CPU

The workflow is designed so a model trained on a GPU runs unchanged on a CPU:

- `device.py` selects CUDA/MPS/CPU automatically (`--device auto`).
- Checkpoints are always saved on CPU and loaded with `map_location="cpu"`, so a
  GPU-trained file loads on a machine with no GPU.
- The same `preprocess.py` transform is used in training and inference, so inputs
  never diverge between the two.

Train (locally on CPU, or on a Colab GPU via `notebooks/colab_train.ipynb`):

```bash
python -m pathassist.train --epochs 10 --tile-size 64 --out models/tile_classifier.pt
# device defaults to "auto": GPU if present, else CPU
```

Run inference with the trained **ensemble** (recommended):

```bash
python -m pathassist.cli run \
    --case-id CASE-001 --image path/to/slide.png \
    --scorer ensemble --checkpoint models/lymph_node/ensemble.pt --device cpu
```

Or with the older single-model checkpoint:

```bash
python -m pathassist.cli run \
    --case-id CASE-001 --image path/to/slide.png \
    --scorer torch --checkpoint models/tile_classifier.pt --device cpu
```

The CNN uses adaptive pooling, so it accepts tiles of any size; still, prefer
matching the training `tile-size` to the config `tiling.tile_size` for best results.

## Using real data

The scaffold trains on a synthetic toy signal so it runs out of the box. To use
real data, replace `make_tile_dataset` in `train.py` with a loader that returns
`(tiles, labels)` from annotated slides. Public starting points:

- **CAMELYON16/17** — lymph-node metastasis WSIs (detection).
- **PANDA** — prostate biopsies with ISUP grades (grading).
- **TCGA** — multi-cancer WSIs + molecular/clinical data (multimodal work later).

For gigapixel `.svs`/`.ndpi` files, add `openslide-python` and feed tiles from an
OpenSlide region reader into `tile_image`'s place — downstream stages don't change.

## Why the audit trail matters

Every machine result is stored with its model name/version and the config used;
every pathologist decision (approve/modify/reject) is stored against it. This is
what turns a demo into something deployable:

- **Traceability** for validation studies and regulators.
- **Second-reader / QC**: `AuditStore.disagreements()` surfaces cases where the
  pathologist overrode the model.
- **Continuous learning**: those corrections become the next training set.

## What this is *not* (yet)

Not an autonomous pathologist. Not regulatory-cleared for unsupervised diagnosis.
Requires hospital validation, LIS integration, and governance before clinical
deployment. The current PCam ensemble is a strong engineering baseline — WSI
support and local validation are the next steps toward real adoption.

## Tests

```bash
python -m pytest -q
```
