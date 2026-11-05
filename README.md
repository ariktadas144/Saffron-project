# Saffron Authentication Detector

A research prototype for distinguishing real and adulterated saffron samples
from images. The project explores a ConvNeXt-Tiny image encoder, grid
graph-attention features, hybrid attention fusion (HAFM), and weighted
test-time-view decoding. It also includes colour and capture-metadata
baselines, dataset auditing tools, and a Streamlit research interface.

> **Research use only.** Model predictions are not laboratory authentication,
> food-safety advice, or regulatory certification. Read the
> [model card](docs/MODEL_CARD.md) before interpreting results.

## Contents

- [Project status and evaluation cautions](#project-status-and-evaluation-cautions)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Requirements and setup](#requirements-and-setup)
- [Prepare a dataset](#prepare-a-dataset)
- [Train and evaluate](#train-and-evaluate)
- [Run the research app](#run-the-research-app)
- [Tests](#tests)
- [Documentation](#documentation)
- [Limitations and responsible use](#limitations-and-responsible-use)

## Project status and evaluation cautions

The supplied image split is not a valid independent evaluation: multiple
photos depict the same physical sample, samples overlap the original training
and test folders, and exact duplicates exist under different sample IDs.
`scripts/prepare_dataset.py` rebuilds the split around physical-sample groups
and duplicate-linked images. Do not report metrics from the original split.

Image and capture properties—including camera, flash, ISO, exposure, image
size, colour, and editing—can correlate with the labels. Shortcut baselines
and group-aware evaluation are included to expose these confounds; good scores
alone do not demonstrate that the model learned saffron-specific features.

No deep-model performance claim should be made unless a complete run passes
the provenance checks in `scripts/build_report.py`. Smoke tests, incomplete
folds, and interrupted runs are not evaluation results. See the
[model card](docs/MODEL_CARD.md) for the current evidence status.

## How it works

1. **Prepare and audit data.** Create a manifest with physical-sample group
   IDs, detect duplicates, and examine capture metadata.
2. **Establish baselines.** Compare against colour and capture-feature
   shortcuts before interpreting deep-model results.
3. **Train with grouped cross-validation.** Keep every view of a physical
   sample (and duplicate-linked samples) within one fold.
4. **Tune without test leakage.** Validation data is used for model selection
   and decoder tuning; held-out fold data is reserved for evaluation.
5. **Verify and report.** Preserve run configuration, predictions, weights,
   and provenance, and only report eligible complete runs.

The deep path combines:

- **ConvNeXt-Tiny** for spatial image features.
- **Grid graph attention** for messages between neighbouring feature-map
  patches.
- **HAFM** to fuse the global CNN feature and graph features.
- **Wombat-OA weighted-beam decoding** to tune and combine model heads and
  test-time views using validation data.

The implementation describes the experimental design; it is not evidence
that the model has learned diagnostic saffron cues.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/` | Model, decoder, colour baseline, and inference-safety utilities |
| `scripts/` | Dataset preparation, training, evaluation, auditing, and reporting |
| `app/` | Streamlit research app and inference code |
| `tests/` | Unit and pipeline tests |
| `notebooks/`, `colab/` | Notebook workflows |
| `docs/` | Model card, capture guidance, evaluation walkthrough, and preregistration |
| `data/` | Local dataset inputs and generated manifests (not committed) |
| `runs/` | Local checkpoints and evaluation outputs (not committed) |

## Requirements and setup

The core environment supports Python 3.10–3.12. TensorFlow training is
intended for a supported Linux GPU environment, WSL2, or Google Colab; native
Windows does not provide TensorFlow GPU support, and CPU training can be very
slow.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For Linux or WSL2 with a compatible NVIDIA GPU, install the TensorFlow CUDA
extras instead of the plain TensorFlow requirement:

```bash
python -m pip install "tensorflow[and-cuda]>=2.16"
```

Check the TensorFlow installation and available devices:

```bash
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

## Prepare a dataset

Place the dataset archives in `data/` and prepare a five-fold manifest. Replace
the example archive names with the names of the archives you received:

```powershell
python scripts/prepare_dataset.py --src data\Test_dataset.zip data\train_dataset.zip --folds 5
```

The preparation step creates extracted images, normalized working copies, and
`data/manifest.csv`. Keep the manifest's `group` identifiers stable: all
images of a physical sample must stay together when splitting, aggregating, or
resampling results. Inspect duplicate and metadata audit output before
training.

Dataset archives, extracted images, manifests, and run outputs are excluded
from Git. Obtain and use the dataset according to its own terms.

## Train and evaluate

### Smoke test

Run a short fold to check that data loading and the training pipeline work.
This uses random initialization and tiny settings; its outputs are **not
research results**:

```powershell
python scripts/run_cv.py --manifest data/manifest.csv --out runs/smoke --folds 0 --img-size 128 --epochs-warm 1 --epochs-ft 1 --mc-samples 4 --no-imagenet --pop 8 --iters 4
```

### Cross-validation

Run all five folds with the standard settings:

```powershell
python scripts/run_cv.py --manifest data/manifest.csv --out runs/cv1 --img-size 256
```

Each fold holds out one group fold for testing and the next for validation;
the other folds are used for training. Folds run in separate processes to
help control memory use. A run can be resumed, and summaries can be rebuilt
from completed folds:

```powershell
python scripts/run_cv.py --out runs/cv1 --aggregate-only
```

For a quick code check, use a small image size and fewer epochs. Larger images,
batch sizes, and Monte Carlo samples increase compute and memory requirements.
Consult `python scripts/run_cv.py --help` for available options.

### Additional evaluation workflows

- `scripts/run_shortcut_cv.py` evaluates shortcut baselines.
- `scripts/make_loco_manifests.py` creates leave-one-camera-out manifests.
- `scripts/stress_test.py` evaluates saved models against image perturbations;
  it requires the trained fold weights.
- `scripts/build_report.py` builds reports from complete runs that satisfy
  provenance checks.

For a locked independent evaluation, follow
[`docs/preregistration.md`](docs/preregistration.md). Do not use test-set
results to tune thresholds or decoder parameters.

### Results

Cross-validation outputs are written under the selected run directory. They
include per-fold predictions and ablations, decoder parameters, fold
summaries, and pooled out-of-fold predictions. Treat a result as reportable
only when the run is complete and the provenance-gated report accepts it.

## Run the research app

Install the app dependencies and launch Streamlit from the repository root:

```powershell
python -m pip install -r app/requirements_app.txt
streamlit run app/streamlit_app.py
```

The deep-model path requires a complete, provenance-recorded fold ensemble
and calibration for the exact model artifacts and manifest. Missing or
mismatched provenance, calibration, capture support, image support, or
uncertainty checks cause the app to abstain with **INCONCLUSIVE** rather than
issue an unsupported verdict. The colour-statistics baseline is exploratory
and does not issue a primary real/fake decision.

**P(fake)** is a model score, not a probability of laboratory-confirmed
adulteration. **INCONCLUSIVE** means the available evidence is insufficient
for a primary decision; it does not mean real or fake.

## Tests

Run the fast test suite:

```powershell
pytest -m "not slow" -q
```

The marked slow tests exercise the folder-based training and decoding
pipeline:

```powershell
pytest -m slow -q
```

## Documentation

- [Model card](docs/MODEL_CARD.md): intended use, evidence status, and risks.
- [Project walkthrough](docs/README_PROJECT_EXPLAINED.md): architecture and
  evaluation rationale.
- [Safe evaluation walkthrough](docs/SAFE_EVALUATION_WALKTHROUGH.md):
  end-to-end evaluation guidance.
- [Capture protocol](docs/capture_protocol.md): guidance for collecting
  controlled image data.
- [Preregistration](docs/preregistration.md): hash-locked prospective
  evaluation procedure.
- [Model glossary](docs/model_glossary.md): terminology used in the project.

## Limitations and responsible use

- The dataset has only 59 independent physical-sample groups; model estimates
  and validation-based decoder tuning can be unstable.
- Camera, flash, ISO, exposure, colour, background, and image processing may
  act as shortcuts. Report relevant strata and baseline comparisons.
- The meaning of the dataset's adulterant-type labels should be confirmed
  before interpreting per-type results.
- Generalization to new cameras, sessions, locations, saffron sources, or
  adulterant types has not been established.
- Use chemical analysis or independent authentication for real-world
  decisions; this prototype is not a substitute.
