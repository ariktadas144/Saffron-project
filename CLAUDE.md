# Saffron real-vs-fake authenticity (conference paper project)

## Goal
Classify saffron images as real vs fake. Deliverable: a conference paper with honest, reproducible results, supervised by a college faculty member.

## Data facts
- Filenames: <id>[ <TYPE>]_<shot>.jpg. Several shots = the SAME physical sample.
- About 59 independent samples. Splits MUST be sample/group-level (scripts/prepare_dataset.py builds data/manifest.csv with group and fold).
- Known shortcuts: image size/aspect/file size (AUC 0.79), colour mean/std (AUC 0.94).
- Fake types C/S/M are undocumented. Do not interpret per-type results.
- Two cameras: SM-J600G and SM-G973F (see scripts/make_loco_manifests.py).

## Hard rules
1. NEVER fabricate, round, or hand-edit any metric. Every number must come from a file in runs/ and be traceable to a command and git commit.
2. NEVER split at image level. NEVER tune on test folds.
3. NEVER modify or commit anything in data/ or model weights.
4. Report mean +/- std over folds and seeds. The colour baseline appears in every results table.
5. If a result is too good (AUC near 1.0), assume leakage or a shortcut and audit first.
6. One change per experiment. Log each in docs/experiments.md.
7. Do not Read image files in data/ directly; summarise with scripts.

## Environment
- Windows PowerShell, CPU-only. Activate: .venv\Scripts\Activate.ps1
- Never run full CV locally. Smoke runs only. Full runs go to Colab (notebooks/run_saffron.ipynb).
- Ask before any command expected to take over 10 minutes, or any deletion.

## Commands
- Tests (~2 min, 49 tests): python -m pytest -m "not slow" -q
- Smoke: python scripts/run_cv.py --manifest data/manifest.csv --out runs/smoke --folds 0 --img-size 128 --epochs-warm 1 --epochs-ft 1 --mc-samples 4 --no-imagenet --pop 8 --iters 4
- Full CV (Colab): python scripts/run_cv.py --manifest data/manifest.csv --out runs/<name> --img-size 256
- Aggregate: python scripts/run_cv.py --out runs/<name> --aggregate-only
- Robustness: scripts/make_loco_manifests.py, scripts/stress_test.py

## Working style
- Python with type hints, small functions, one set_seed() helper.
- Outputs only in runs/<experiment-name>/. Paper assets only in paper/figures and paper/tables.
- Commit before each experiment: exp(<name>): <what changed>.
- Superpowers: use planning for code changes. For experiments the test is grouped-CV vs the colour baseline. Keep pytest green after src/ or scripts/ edits.
- Paper text, docs and experiment logs are written in normal prose, not caveman style.
