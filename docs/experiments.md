# Experiment log

## 2026-11-04: frozen_embed_v1, frozen ImageNet ConvNeXt-Tiny with a balanced logistic head, plus capture-shortcut audit

**Status: capture-confounded. Do not cite as an authenticity signal.**

**Git commit:** 323a740 (HEAD when run). The working tree was not clean: `.claude/settings.json` was modified and `data/audit_exif.csv` and `saffron_project.zip` were untracked. No `exp(...)` commit was made before the run, so this entry breaks the commit-before-experiment rule. The audit scripts live in the session scratchpad, not in the repository, so the audit rows below cannot be reproduced from this commit alone.

**Command (main run):**
`python scripts/frozen_embedding_cv.py --out runs/frozen_embed_v1/frozen_embedding_cv.csv --cache runs/frozen_embed_v1/frozen_embeddings.npz`
Defaults used: `data/manifest.csv`, `data/audit_exif.csv`, 256 px, seed 42, shuffle seeds 101, 202, 303. Results are in `runs/frozen_embed_v1/` (`frozen_embedding_cv.csv`, `_camera.csv`, `_shuffle.csv`). The run has no `summary_sample.csv` or `summary_image.csv`, and no colour-baseline row of its own.

**Audit commands:** scratchpad scripts `audit_probe.py` (steps `probes_crop`, `sessions`, `nn`), `crop_embed.py` and `audit_probe2.py`, run with `python -I`. Outputs are in `runs/audit_v1/`. `python scripts/matched_subset_eval.py --run runs/cv_repro --out runs/audit_v1/matched_subset_results.csv` failed with `FileNotFoundError: fold 0 OOF predictions or thresholds are missing`. `scripts/label_shuffle_test.py` was not run because it retrains the deep model.

**Hypothesis:** pretrained ImageNet features carry signal about saffron authenticity beyond the colour shortcut (colour mean/std AUC about 0.94 per CLAUDE.md).

**Setup:** manifest folds, 59 groups (31 real, 28 fake), 311 images. Test = fold k, validation = fold k+1, train = the rest. Backbone frozen, pooled 768-d embedding, `StandardScaler` plus `LogisticRegression(class_weight="balanced")`. Sample-level AUC by logit fusion over a group's shots. One seed (42) for the head.

### Results (sample-level AUC, mean +/- std over 5 folds unless stated)

| Probe | Mean AUC | Std | Per-fold range |
|---|---|---|---|
| Frozen embedding (main run) | 1.000 | 0.000 | 1.00 to 1.00 |
| Frozen embedding, 50% centre crop | 1.000 | 0.000 | 1.00 to 1.00 |
| EXIF full (iso, exposure, hour, flash, focal length, width, height, file size) | 0.980 | 0.045 | 0.90 to 1.00 |
| EXIF without hour | 0.974 | 0.043 | 0.90 to 1.00 |
| EXIF without hour or size dims | 0.911 | 0.035 | 0.89 to 0.97 |
| exposure_time alone | 0.944 | 0.054 | 0.87 to 1.00 |
| iso alone | 0.918 | 0.038 | 0.88 to 0.97 |
| Hour only (one-hot) | 0.880 | 0.067 | 0.77 to 0.94 |
| Colour mean/std only (this harness, logistic head) | 0.862 | 0.093 | 0.77 to 1.00 |
| Size + file size only (this harness, logistic head) | 0.598 | 0.102 | 0.42 to 0.67 |
| Group-label shuffle control (15 fold x seed runs) | 0.466 | 0.214 | not recorded here |

Colour baseline for comparison: 0.862 +/- 0.093 above (logistic head, this harness) and 0.909 +/- 0.063 from `runs/audit/phase2_shortcut_same_folds_summary.csv` (sample level, random forest). The frozen embedding does not have a colour-baseline row in its own output.

**SM-G973F only (25 groups: 17 real, 8 fake; sessions 04-05 h15 and 04-06 h10, both contain both classes):**

| Probe | 5-fold mean AUC | Std | Range | Pooled OOF AUC | Leave-one-session-out pooled AUC |
|---|---|---|---|---|---|
| Embedding | 1.000 | 0.000 | 1.00 to 1.00 | 0.985 | 0.662 |
| EXIF full | 0.920 | 0.179 | 0.60 to 1.00 | 0.897 | 0.669 |
| Colour | 1.000 | 0.000 | 1.00 to 1.00 | 0.993 | 0.985 |

Leave-one-session-out per session, embedding: 0.867 (held out 04-05 h15, trained on 12 real and 5 fake groups) and 0.683 (held out 04-06 h10, trained on 5 real and 3 fake groups). Only two sessions exist, so this is thin evidence either way.

**Other findings:**
- On SM-J600G every real sample was captured at hour 12 and every fake at hour 13 or 14 (2021-04-14), so capture hour alone separates the classes for that camera. Of 7 date+hour sessions, 5 contain a single class.
- Nearest-neighbour check: the nearest train sample shares the label for 57 of 59 test samples (96.6%); the neighbour shares the capture session in 88% of cases.
- Camera is confounded with class at image level (SM-G973F 101 real / 40 fake; SM-J600G 60 real / 100 fake).
- `data/audit_exif.csv` has a `fold` column that disagrees with `data/manifest.csv` for 246 images (46 of 59 groups). All analyses here use the manifest folds. Whether the `runs/audit/phase2_*` results depend on the stale column is unchecked.
- Size probe reconciliation: the earlier 0.598 and the prior script's 0.882 differ in both model and features. On these folds: logistic regression on the prior features 0.764, forest on my features 0.787, forest on the prior features 0.886.

### Verdict vs the colour baseline
The embedding's 1.000 exceeds the colour baseline (0.862 to 0.909), but the result is not valid evidence of authenticity signal. Capture-condition features reach 0.91 to 0.98 on their own, the embedding falls to 0.66 when a whole session is held out on SM-G973F, and only colour holds up in that test. A 1.0 on 59 groups with this many capture shortcuts is treated as a shortcut until shown otherwise. Inconclusive, leaning shortcut.

**Caveats:** one seed for the head; 11 to 13 groups per test fold; 25 groups and 2 sessions for the SM-G973F tests; the main run's own colour row is missing, so the colour comparison uses two different harnesses.

Fold pinning: manifest sha256 347b74549aa5b0663df55a0e5402d4148bf4a654bbad74386f003c9b54e5db4c (311 rows, 59 groups); audit_exif.csv fold column is stale and unused; pinned copy at data/manifest_folds.csv.


## 2026-11-05: cv_imagenet_v1 (5-fold deep run, imagenet=True)

**Commit:** 4eaa086 (branch fix-eval). Folds are the pinned `data/manifest_folds.csv` (311 images, 59 groups); the audit confirmed that every fold's test groups and paths in `runs/cv_imagenet_v1/fold*/preds_test.csv` match `data/manifest.csv`, with no duplicate paths and no group in more than one fold.

**Command (reconstructed from the CLAUDE.md template and `fold0/config.json`; the exact Colab cell was not saved in the repo):** `python scripts/run_cv.py --manifest data/manifest.csv --out runs/cv_imagenet_v1 --img-size 256` with `imagenet: true`, seed 42, 3 warm + 25 fine-tune epochs, batch 16, mc_samples 16. Run on Colab. Library versions as reported by the user, not recorded in `runs/`: TF 2.21.0, Keras 3.13.2, numpy 2.4.6, scikit-learn 1.6.1 (local: Keras 3.15.1, scikit-learn 1.9.1).

**Hypothesis:** the Wombat-OA + weighted beam + log-prob fusion model with ImageNet initialisation beats the colour baseline under group-level folds.

**Results, sample level, mean +/- std over 5 folds (`summary_sample.csv`):**

| Model | AUC | Accuracy |
|---|---|---|
| PROPOSED: Wombat-OA + weighted beam + log-prob fusion | 1.0 +/- 0.0 | 1.0 +/- 0.0 |
| single view, cnn / hafm | 1.0 +/- 0.0 | 1.0 +/- 0.0 |
| single view, gnn | 1.0 +/- 0.0 | 0.985 +/- 0.031 |
| colour-statistics random forest (baseline) | 0.909 +/- 0.056 | 0.831 +/- 0.051 |
| exposure iso flash random forest (shortcut) | 0.945 +/- 0.036 | 0.849 +/- 0.093 |
| size file bpp random forest (shortcut) | 0.882 +/- 0.098 | 0.746 +/- 0.093 |

Image level (`summary_image.csv`): PROPOSED AUC 1.0 +/- 0.0 (acc 0.997 +/- 0.006); colour baseline AUC 0.924 +/- 0.052; EXIF 0.95 +/- 0.032; size 0.842 +/- 0.092.

Frozen-embedding probe in `runs/frozen_embed_v2`: AUC 1.0 in all 5 folds (`frozen_embedding_cv.csv`). Group-label shuffle control (`frozen_embedding_cv_shuffle.csv`): mean AUC 0.466 +/- 0.214 over 15 runs.

**Audit (runs/audit_v1/, rebuilt harness, `probe_table_rebuilt.csv`):** SM-G973F-only pooled OOF AUC over 25 samples (8 fake): proposed 1.000, frozen probe 1.000, colour RF 0.985, EXIF RF 0.710, size RF 0.640. Both SM-G973F sessions (2021-04-05 h15, 2021-04-06 h10) give AUC 1.0 for proposed and hafm (`g973f_session_auc.csv`). The rebuilt RF rows differ from the logged phase 2 values (colour 0.904 vs 0.909, size 0.862 vs 0.879; EXIF 0.939 matches exactly), so rebuilt-harness numbers are not interchangeable with the table above.

Matched G973F subset (ISO 100/125/160, flash off; 8 real/8 fake): proposed AUC 1.0, EXIF-only 0.703. Per-session G973F AUC 1.0 in both sessions. Cross-session transfer untested.

**Discrepancy with the user's notes:** the notes quoted colour 0.904, EXIF 0.941 and size 0.827, which match no file in `runs/cv_imagenet_v1` or `runs/audit_v1`; the values in this entry are copied from the files. The 0.904 matches only the rebuilt colour row. The quoted G973F EXIF 0.74 is close to 0.739 in `matched_subset_results.csv` (EXIF-only baseline, all 25 G973F samples), whereas the 0.710 above comes from the rebuilt harness. Both are kept, and they are different computations.

**Matched-subset script:** `scripts/matched_subset_eval.py --run runs/cv_imagenet_v1 --out runs/audit_v1/matched_subset_results.csv` first failed with `AttributeError: 'DataFrame' object has no attribute 'camera'` because `preds_test.csv` already holds camera columns. `load_oof` was then fixed to drop the duplicate columns before the merge (test in `tests/test_matched_subset_load_oof.py`, not yet committed), and the run wrote `runs/audit_v1/matched_subset_results.csv`.

### Verdict vs the colour baseline
Proposed AUC 1.0 +/- 0.0 exceeds the colour baseline 0.909 +/- 0.056, but under rule 5 this is treated as a shortcut, not authenticity signal. Test folds hold 11 to 13 groups, the G973F sessions each have a single capture batch, and capture features alone reach 0.88 to 0.95. Status: capture-confounded, do not cite as an authenticity signal.

**Caveats:** one seed; no per-seed variance; AUC 1.0 in every fold of the deep model, frozen probe, and both G973F sessions.


## 2026-11-05: session_holdout_v1 (SM-G973F cross-session transfer, Colab)

**Commit:** 2a1a087 (HEAD of fix-eval when this entry was written, the commit that added `scripts/session_holdout_cv.py`). The run directory does not record the commit it ran at.

**Command:** `python scripts/session_holdout_cv.py --out runs/session_holdout_v1 --img-size 256` (settings in `session_holdout_run.json`: seed 42, batch 16, 3 warm + 25 fine-tune epochs, mc_samples 16, pop 20, iters 25, imagenet true, `--date-a 2021:04:05 --date-b 2021:04:06`). Colab versions from `pip_freeze.txt`: tensorflow 2.21.0, keras 3.13.2, numpy 2.4.6, scikit-learn 1.6.1.

**Hypothesis:** if the deep model carries authenticity signal beyond capture conditions, it should still separate real from fake when trained on one SM-G973F capture session and tested on the other, and it should beat the colour baseline there.

**Files:** this run has no `summary_sample.csv` or `summary_image.csv`, because each direction is a single train/val/test split with no folds to average. Values below are from `session_holdout_results.csv` and `<direction>/ablation_sample_level.csv`. There is no std or CI: one run, one seed per direction.

**Splits (`split_counts.csv`; validation = random 20% of train groups, stratified, seed 42):**

| Direction | Train real/fake groups | Val real/fake groups | Test real/fake groups |
|---|---|---|---|
| 04-05 -> 04-06 | 4 / 2 | 1 / 1 | 12 / 5 |
| 04-06 -> 04-05 | 10 / 4 | 2 / 1 | 5 / 3 |

**Results, sample level (AUC; accuracy at the validation-tuned threshold in brackets):**

| Method | 04-05 -> 04-06 | 04-06 -> 04-05 |
|---|---|---|
| PROPOSED: Wombat-OA + weighted beam + log-prob fusion | 0.8 (acc 0.882) | 0.7 (acc 0.375) |
| single view, hafm (no beam) | 0.95 | 1.0 |
| colour-statistics random forest (baseline) | 1.0 (acc 0.294) | 1.0 (acc 1.0) |
| exposure iso flash random forest (shortcut) | 0.5 (acc 0.294) | 0.5 (acc 0.625) |

In the 04-06 -> 04-05 direction the proposed model has specificity 0.0 and recall 1.0, so it labels all 8 test samples fake at the tuned threshold. In the 04-05 -> 04-06 direction it has specificity 1.0 and recall 0.6. The beam random search lowered AUC relative to the earlier ablation rows (0.933 to 0.8, and 1.0 to 0.7), and it was tuned on a validation set of 2 and 3 groups.

### Verdict vs the colour baseline
The proposed model does not beat the colour baseline in either direction (0.8 vs 1.0 and 0.7 vs 1.0). The colour random forest separates the classes perfectly in both directions, so colour statistics transfer across these two sessions. This does not show whether the signal is saffron colour or a capture/batch difference; there are only 8 and 17 groups. The deep model's 1.0 in the within-session folds is therefore not evidence of authenticity signal that transfers across sessions. The exposure/ISO/flash baseline is at chance (0.5), so EXIF does not transfer here. The proposed model's cross-session AUC (0.7 to 0.8) is above chance but is based on 8 and 17 test samples.

**Caveats:** one seed per direction; tiny training sets (6 and 14 groups) and validation sets (2 and 3 groups) make the tuned thresholds and beam parameters unreliable; test AUCs rest on 5 and 3 fake samples, so a single sample changes AUC by 0.07 to 0.2; single-view hafm without the beam/random-search stage scores 0.95 and 1.0 across sessions, so the beam stage, tuned on 2-3 validation groups, is what lowered the proposed model's AUC here. This is indicative only (5 and 3 fake test groups); sessions differ in date and hour as well as in sample composition, and only two exist.


## 2026-11-06: session_holdout_v1 probes (frozen, colour, EXIF, shuffle on the same splits)

**Commit:** f365fdf (HEAD when this entry was written; `scripts/session_holdout_probes.py` is tracked at this commit).

**Command:** `python scripts/session_holdout_probes.py` (defaults: `runs/session_holdout_v1` role manifests, `runs/frozen_embed_v2/frozen_embeddings.npz`, `data/audit_exif.csv`). CPU only, no training. Colour/EXIF seed 42; frozen head StandardScaler + LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42), fit on train only.

**Hypothesis:** a frozen ImageNet embedding probe carries more cross-session signal than the deep model's beam stage, and a group-label shuffle control stays near 0.5.

**Files:** the run has no `summary_sample.csv` or `summary_image.csv` (one split per direction, no folds). Values are from `runs/session_holdout_v1/probes_results.csv`; no std or CI exists (one seed per direction, except the three shuffle seeds). Test groups: 12 real / 5 fake for 04-05 -> 04-06, 5 real / 3 fake for 04-06 -> 04-05.

**Results, sample-level AUC:**

| Method | 04-05 -> 04-06 | 04-06 -> 04-05 |
|---|---|---|
| frozen probe | 0.85 | 0.80 |
| colour-statistics RF (baseline) | 1.0 | 1.0 |
| colour mean only RF (RGB+HSV means) | 0.933333 | 1.0 |
| exposure iso flash RF | 0.5 | 0.5 |
| frozen probe, shuffle seed 101 | 0.616667 | 0.333333 |
| frozen probe, shuffle seed 202 | 0.033333 | 0.0 |
| frozen probe, shuffle seed 303 | 0.8 | 1.0 |

For comparison, from the same run's `session_holdout_results.csv`: proposed model 0.8 and 0.7. The colour and EXIF rows reproduce `session_holdout_results.csv` with a gap of 0.0 in both directions.

### Verdict vs the colour baseline
The frozen probe (0.85, 0.80) is above the proposed model (0.8, 0.7) but below the colour baseline (1.0 in both directions), so neither beats colour. The shuffle control does not stay near 0.5: with 5 and 3 fake test groups it ranges from 0.0 to 1.0, so this control cannot separate signal from noise on these two splits. The colour-mean-only variant also reaches 0.933 and 1.0, so simple colour averages already transfer across these two sessions. Whether that is saffron colour or a capture/batch difference cannot be told from 8 and 17 test groups.

**Caveats:** one seed for the real heads; 6 and 14 training groups; AUC steps of about 0.07 to 0.2 per test group; the shuffle AUCs of 0.0 and 1.0 show how unstable AUC is at this size.
