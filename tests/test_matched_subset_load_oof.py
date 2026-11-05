from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def test_load_oof_accepts_preds_that_already_have_camera_columns(tmp_path):
    from scripts.matched_subset_eval import EXIF, PROPOSED, load_oof, sample_predictions

    paths = ["data/images/real/a_1.jpg", "data/images/fake/b_1.jpg"]
    manifest = pd.DataFrame({
        "path": paths, "label": [0, 1], "type": ["-", "C"],
        "group": ["a", "b"], "fold": [0, 0],
    })
    audit = manifest.assign(
        camera="samsung SM-G973F", flash_status="off", iso=100.0,
        capture_date="2021:04:05",
    )
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    audit.to_csv(tmp_path / "audit.csv", index=False)
    run = tmp_path / "run" / "fold0"
    run.mkdir(parents=True)
    pd.DataFrame({
        PROPOSED: [0.1, 0.9], EXIF: [0.2, 0.8], "path": paths, "group": ["a", "b"],
        "y": [0, 1], "fold": [0, 0], "camera": "stale camera", "flash_status": "on",
        "iso": 999.0, "capture_date": "1999:01:01", "type": ["-", "C"],
    }).to_csv(run / "preds_test.csv", index=False)
    (run / "sample_thresholds.json").write_text(
        json.dumps({PROPOSED: 0.5, EXIF: 0.5}), encoding="utf-8"
    )

    result = load_oof(tmp_path / "run", tmp_path / "manifest.csv", tmp_path / "audit.csv")

    assert not any(c.endswith(("_x", "_y")) for c in result.columns)
    assert set(result.camera) == {"samsung SM-G973F"}  # audit copy wins
    assert len(sample_predictions(result)) == 2
