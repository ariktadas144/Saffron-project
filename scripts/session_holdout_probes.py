"""Cheap probes on the SAME train/val/test groups as runs/session_holdout_v1 (SM-G973F cross-session).

   python scripts/session_holdout_probes.py

Rows: frozen-embedding probe, colour RF, colour-mean-only RF, EXIF RF, and a group-label shuffle control
for the frozen probe. Colour and EXIF rows must reproduce session_holdout_results.csv; gaps are printed.
Sample-level AUC is threshold-free, so the validation-only threshold is not needed here. No training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DIRECTIONS = ("2021-04-05_to_2021-04-06", "2021-04-06_to_2021-04-05")
SHUFFLE_SEEDS = (101, 202, 303)
MEAN_COLUMNS = [0, 1, 2, 6, 7, 8]  # RGB mean + HSV mean in color_baseline.features' 12-column layout
SEED = 42


def match_rows(manifest_paths, cache_paths) -> np.ndarray:
    """Row index into the cache for every manifest path (separator-insensitive); fail loudly otherwise."""
    from manifest_paths import normalize_manifest_path
    cache = [normalize_manifest_path(p) for p in cache_paths]
    if len(set(cache)) != len(cache):
        raise ValueError("embedding cache contains duplicate paths")
    index = {p: i for i, p in enumerate(cache)}
    wanted = [normalize_manifest_path(p) for p in manifest_paths]
    missing = [p for p in wanted if p not in index]
    if missing:
        raise ValueError(f"{len(missing)} manifest path(s) missing from the embedding cache, e.g. {missing[:5]}")
    return np.array([index[p] for p in wanted], dtype=int)


def _sample_auc(labels, probability, groups) -> float:
    import wombat_beam as wb
    from sklearn.metrics import roc_auc_score
    score, y = wb.aggregate_by_group(probability, labels, np.asarray(groups).astype(str))
    return float(roc_auc_score(y, score)) if len(set(y)) == 2 else float("nan")


def _frozen(embeddings, train_idx, test_idx, train_labels, seed) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed))
    model.fit(embeddings[train_idx], train_labels)
    return model.predict_proba(embeddings[test_idx])[:, 1]


def run_direction(direction: str, run_dir: Path, audit: pd.DataFrame, embeddings, cache_paths) -> pd.DataFrame:
    import color_baseline as cb
    from audit_shortcuts import BASELINE_FEATURES, _classifier
    from label_shuffle_test import shuffle_partition_labels
    from manifest_paths import normalize_manifest_paths
    from session_holdout_cv import COLOUR, EXIF

    man = normalize_manifest_paths(pd.read_csv(run_dir / f"manifest_{direction}.csv")).reset_index(drop=True)
    check = man[["path", "group"]].merge(audit[["path", "group"]], on="path", how="left", validate="one_to_one")
    if check.group_y.isna().any() or not check.group_x.astype(str).eq(check.group_y.astype(str)).all():
        raise ValueError(f"{direction}: manifest paths/groups do not match data/audit_exif.csv")
    frame = man[["path"]].merge(audit.drop(columns=["group", "label", "fold"], errors="ignore"),
                                on="path", how="left", validate="one_to_one")
    frame["label"] = man.label.to_numpy()
    tr, te = (np.flatnonzero((man.fold == c).to_numpy()) for c in (2, 0))
    y_tr, y_te, g_te = man.label.to_numpy()[tr], man.label.to_numpy()[te], man.group.to_numpy()[te]
    test_groups = man.iloc[te].drop_duplicates("group")
    counts = {"n_test_real": int((test_groups.label == 0).sum()), "n_test_fake": int((test_groups.label == 1).sum())}
    rows: list[tuple[str, float]] = []

    emb = embeddings[match_rows(man.path, cache_paths)]
    rows.append(("frozen probe (StandardScaler + LR)", _sample_auc(y_te, _frozen(emb, tr, te, y_tr, SEED), g_te)))
    X = cb.features(man.path.tolist(), cache=None)  # cache=None: never overwrite the run's colour_features.npz
    rows.append((COLOUR, _sample_auc(y_te, cb.fit_predict(X[tr], y_tr, X[te], seed=SEED), g_te)))
    rows.append(("colour mean only random forest (RGB+HSV means)",
                 _sample_auc(y_te, cb.fit_predict(X[tr][:, MEAN_COLUMNS], y_tr, X[te][:, MEAN_COLUMNS], seed=SEED), g_te)))
    feats = BASELINE_FEATURES["exposure_iso_flash"]
    model = _classifier(feats, frame)
    model.set_params(randomforestclassifier__random_state=SEED)
    model.fit(frame.iloc[tr][feats], y_tr)
    p_exif = model.predict_proba(frame.iloc[te][feats])[:, list(model.classes_).index(1)]
    rows.append((EXIF, _sample_auc(y_te, p_exif, g_te)))
    for s in SHUFFLE_SEEDS:
        shuffled = shuffle_partition_labels(man, 0, s)
        p = _frozen(emb, tr, te, shuffled.label.to_numpy()[tr], SEED + s)
        rows.append((f"frozen probe, group-label shuffle seed {s}", _sample_auc(y_te, p, g_te)))
    return pd.DataFrame([{"direction": direction, "method": m, "auc_sample": a, **counts} for m, a in rows])


def reproduction_gaps(results: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    from session_holdout_cv import COLOUR, EXIF
    ref = reference.rename(columns={"auc_sample": "auc_reference"})
    merged = results[results.method.isin([COLOUR, EXIF])].merge(ref, on=["direction", "method"], how="left")
    merged["gap"] = merged.auc_sample - merged.auc_reference
    return merged[["direction", "method", "auc_sample", "auc_reference", "gap"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="runs/session_holdout_v1")
    ap.add_argument("--embeddings", default="runs/frozen_embed_v2/frozen_embeddings.npz")
    ap.add_argument("--audit", default="data/audit_exif.csv")
    a = ap.parse_args()
    rel = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p  # noqa: E731
    from manifest_paths import normalize_manifest_paths
    audit = normalize_manifest_paths(pd.read_csv(rel(a.audit)), columns=("path",))
    z = np.load(rel(a.embeddings), allow_pickle=True)
    run_dir = rel(a.run_dir)
    results = pd.concat([run_direction(d, run_dir, audit, z["embeddings"], z["paths"]) for d in DIRECTIONS],
                        ignore_index=True)
    results.to_csv(run_dir / "probes_results.csv", index=False)
    print(results.to_string(index=False))
    gaps = reproduction_gaps(results, pd.read_csv(run_dir / "session_holdout_results.csv"))
    print("\nReproduction check vs session_holdout_results.csv:")
    print(gaps.to_string(index=False))
    if not (gaps.gap.abs() < 1e-9).all():
        print("WARNING: colour/EXIF rows do not reproduce session_holdout_results.csv exactly")
    print(f"\nWrote {run_dir / 'probes_results.csv'}\nCommand: python scripts/session_holdout_probes.py")


if __name__ == "__main__":
    main()
