"""Cross-session transfer on SM-G973F only: train on one capture date, test on the other, then the reverse.

   python scripts/session_holdout_cv.py --out runs/session_holdout_v1 --img-size 256        # Colab, full
   python scripts/session_holdout_cv.py --smoke                                             # local smoke, 1+1 epochs
   python scripts/session_holdout_cv.py --out runs/audit_v1/session_holdout_prepare --prepare-only   # splits only

Each direction gets its own explicit manifest (data/manifest.csv is never edited). Roles use the same
convention as make_loco_manifests.py: fold 0 = TEST, 1 = VALIDATION, 2 = TRAIN, so scripts/train_fold.py
and src/saffron_cnn_gnn.run_fold train with their unchanged settings. Validation = a random 20% of the
training groups, stratified by class, seed 42. The evaluation below mirrors scripts/run_cv.py (same
colour/EXIF baselines and wb.evaluate_all) but aligns data/audit_exif.csv by path, because run_cv.py
requires the audit file to match the manifest row for row and so cannot run on a subset manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CAMERA = "SM-G973F"
PROPOSED = "PROPOSED: Wombat-OA + weighted beam + log-prob fusion"
COLOUR = "colour-statistics random forest (baseline)"
EXIF = "exposure iso flash random forest (shortcut)"
ROLES = {"test": 0, "val": 1, "train": 2}


def _norm_date(value: str) -> str:
    return str(value).replace("-", ":")


def build_split(
    frame: pd.DataFrame, train_date: str, test_date: str,
    seed: int = 42, val_frac: float = 0.2,
) -> pd.DataFrame:
    """Return the SM-G973F rows of `frame` with `fold` = 0 test / 1 val / 2 train (group-level)."""
    needed = {"group", "label", "camera", "capture_date"}
    if missing := needed - set(frame.columns):
        raise ValueError(f"frame is missing columns: {sorted(missing)}")
    train_date, test_date = _norm_date(train_date), _norm_date(test_date)
    if train_date == test_date:
        raise ValueError("train and test sessions must differ")
    data = frame[frame.camera.astype(str).str.contains(CAMERA, na=False)].copy()
    data["capture_date"] = data.capture_date.astype(str).map(_norm_date)
    per_group = data.groupby("group").agg(
        label=("label", "nunique"), date=("capture_date", "nunique"))
    if (per_group != 1).any().any():
        raise ValueError("each group must have exactly one label and one capture date")
    meta = data.groupby("group").agg(label=("label", "first"), date=("capture_date", "first"))
    test_groups = set(meta.index[meta.date == test_date])
    train_pool = meta[meta.date == train_date]
    rng = np.random.default_rng(seed)
    val_groups: set[str] = set()
    for label, part in train_pool.groupby("label"):
        names = sorted(part.index)
        n_val = max(1, int(round(val_frac * len(names))))
        if len(names) - n_val < 1:
            raise ValueError(f"class {label}: only {len(names)} training group(s), cannot hold out validation")
        val_groups |= set(rng.choice(names, size=n_val, replace=False))
    train_groups = set(train_pool.index) - val_groups
    roles = {**{g: ROLES["test"] for g in test_groups},
             **{g: ROLES["val"] for g in val_groups},
             **{g: ROLES["train"] for g in train_groups}}
    out = data[data.group.isin(roles)].copy()
    out["fold"] = out.group.map(roles).astype(int)
    for role, code in ROLES.items():
        if out[out.fold == code].label.nunique() != 2:
            raise ValueError(f"{role} split does not contain both classes")
    if out.groupby("group").fold.nunique().max() != 1:
        raise AssertionError("a group appears in more than one role")
    return out


def split_counts(split: pd.DataFrame, direction: str) -> pd.DataFrame:
    rows = []
    for role, code in ROLES.items():
        part = split[split.fold == code]
        g = part.drop_duplicates("group")
        rows.append({
            "direction": direction, "split": role, "n_images": len(part),
            "n_groups": len(g), "n_real_groups": int((g.label == 0).sum()),
            "n_fake_groups": int((g.label == 1).sum()),
            "n_real_images": int((part.label == 0).sum()),
            "n_fake_images": int((part.label == 1).sum()),
        })
    return pd.DataFrame(rows)


def load_joined(manifest_path: Path, audit_path: Path) -> pd.DataFrame:
    from manifest_paths import normalize_manifest_paths
    man = normalize_manifest_paths(pd.read_csv(manifest_path))
    audit = normalize_manifest_paths(pd.read_csv(audit_path), columns=("path",))
    ident = man[["path", "group"]].merge(
        audit[["path", "group"]], on="path", how="outer", validate="one_to_one", indicator=True)
    if not ident._merge.eq("both").all() or not ident.group_x.astype(str).eq(ident.group_y.astype(str)).all():
        raise ValueError("audit image/group identity does not match the manifest")
    keep = [c for c in audit.columns if c not in man.columns or c == "path"]
    return man.merge(audit[keep], on="path", how="left", validate="one_to_one")


def train_and_evaluate(split: pd.DataFrame, manifest_file: Path, out: Path, a: argparse.Namespace) -> pd.DataFrame:
    """Train fold 0 of the role manifest in a fresh process, then score the proposed model and the baselines."""
    import color_baseline as cb
    import wombat_beam as wb
    from audit_shortcuts import BASELINE_FEATURES, _classifier

    out.mkdir(parents=True, exist_ok=True)
    if a.retrain or not ((out / "cache_val.npz").exists() and (out / "cache_test.npz").exists()):
        cmd = [sys.executable, str(ROOT / "scripts" / "train_fold.py"),
               "--manifest", str(manifest_file), "--out", str(out), "--fold", "0",
               "--img-size", str(a.img_size), "--batch", str(a.batch),
               "--epochs-warm", str(a.epochs_warm), "--epochs-ft", str(a.epochs_ft),
               "--mc-samples", str(a.mc_samples), "--seed", str(a.seed)]
        if a.no_imagenet:
            cmd.append("--no-imagenet")
        subprocess.run(cmd, check=True)
    val, test = wb.Cache.load(str(out / "cache_val.npz")), wb.Cache.load(str(out / "cache_test.npz"))
    split = split.reset_index(drop=True)
    va, te = split[split.fold == 1].reset_index(drop=True), split[split.fold == 0].reset_index(drop=True)
    for name, frame, cache in (("validation", va, val), ("test", te, test)):
        if (cache.paths is None or cache.groups is None
                or not np.array_equal(frame.path.astype(str).to_numpy(), cache.paths.astype(str))
                or not np.array_equal(frame.group.astype(str).to_numpy(), cache.groups.astype(str))
                or not np.array_equal(frame.label.to_numpy(), cache.y)):
            raise ValueError(f"{name} cache does not match the role manifest")
    tr_idx, va_idx, te_idx = (split.index[split.fold == c].to_numpy() for c in (2, 1, 0))
    X = cb.features(split.path.tolist(), cache=str(out / "colour_features.npz"))
    y_tr = split.label.to_numpy()[tr_idx]
    p_all = cb.fit_predict(X[tr_idx], y_tr, X[np.concatenate([va_idx, te_idx])], seed=a.seed)
    p_col_val, p_col = np.split(p_all, [len(va_idx)])
    feats = BASELINE_FEATURES["exposure_iso_flash"]
    model = _classifier(feats, split)
    model.set_params(randomforestclassifier__random_state=a.seed)
    model.fit(split.iloc[tr_idx][feats], y_tr)
    col = list(model.classes_).index(1)
    p_exif_val = model.predict_proba(split.iloc[va_idx][feats])[:, col]
    p_exif = model.predict_proba(split.iloc[te_idx][feats])[:, col]
    _, info = wb.evaluate_all(
        val, test, pop=a.pop, iters=a.iters, seed=0,
        extra_rows={COLOUR: (p_col, 0.5), EXIF: (p_exif, 0.5)},
        extra_val_probs={COLOUR: p_col_val, EXIF: p_exif_val},
    )
    sample = info["sample_table"]
    sample.to_csv(out / "ablation_sample_level.csv")
    return pd.DataFrame({"method": [PROPOSED, COLOUR, EXIF],
                         "auc_sample": [float(sample.loc[m, "auc"]) for m in (PROPOSED, COLOUR, EXIF)]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--audit", default="data/audit_exif.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--date-a", default="2021:04:05")
    ap.add_argument("--date-b", default="2021:04:06")
    ap.add_argument("--smoke", action="store_true", help="128 px, 1+1 epochs, no ImageNet download")
    ap.add_argument("--prepare-only", action="store_true", help="write manifests and counts, do not train")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--img-size", type=int, default=256); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs-warm", type=int, default=3); ap.add_argument("--epochs-ft", type=int, default=25)
    ap.add_argument("--mc-samples", type=int, default=16); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pop", type=int, default=20); ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--no-imagenet", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.img_size, a.epochs_warm, a.epochs_ft, a.mc_samples = 128, 1, 1, 4
        a.pop, a.iters, a.no_imagenet = 8, 4, True
    out = Path(a.out or ("runs/SMOKE_TEST_NOT_A_RESULT_session_holdout" if a.smoke else "runs/session_holdout_v1"))
    out = out if out.is_absolute() else ROOT / out
    root_rel = lambda p: p if Path(p).is_absolute() else ROOT / p  # noqa: E731
    joined = load_joined(root_rel(a.manifest), root_rel(a.audit))
    g973 = joined[joined.camera.astype(str).str.contains(CAMERA, na=False)]
    hours = pd.to_datetime(g973.DateTimeOriginal, format="%Y:%m:%d %H:%M:%S", errors="coerce").dt.hour
    print("SM-G973F sessions (date: hours, groups):",
          {d: (sorted(set(hours[g973.capture_date == d].dropna().astype(int))), g973[g973.capture_date == d].group.nunique())
           for d in sorted(g973.capture_date.unique())})
    out.mkdir(parents=True, exist_ok=True)
    (out / "session_holdout_run.json").write_text(json.dumps(
        {k: v for k, v in vars(a).items()}, indent=2, default=str), encoding="utf-8")
    counts, results = [], []
    for train_date, test_date in ((a.date_a, a.date_b), (a.date_b, a.date_a)):
        direction = f"{train_date.replace(':', '-')}_to_{test_date.replace(':', '-')}"
        split = build_split(joined, train_date, test_date, seed=a.seed)
        manifest_file = out / f"manifest_{direction}.csv"
        split.to_csv(manifest_file, index=False)
        c = split_counts(split, direction)
        counts.append(c)
        print(f"\n=== train {train_date} -> test {test_date} (wrote {manifest_file}) ===")
        print(c.to_string(index=False))
        if not a.prepare_only:
            r = train_and_evaluate(split, manifest_file, out / direction, a)
            results.append(r.assign(direction=direction))
    pd.concat(counts).to_csv(out / "split_counts.csv", index=False)
    if results:
        res = pd.concat(results)[["direction", "method", "auc_sample"]]
        res.to_csv(out / "session_holdout_results.csv", index=False)
        print("\nSample-level AUC per direction (one test session, 8 or 17 groups; no CI):")
        print(res.to_string(index=False))
    print("\nColab command:\n  python scripts/session_holdout_cv.py --out runs/session_holdout_v1 --img-size 256")


if __name__ == "__main__":
    main()
