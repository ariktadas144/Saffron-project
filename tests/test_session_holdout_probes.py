from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def test_match_rows_ignores_separator_style_and_order():
    from scripts.session_holdout_probes import match_rows

    cache = ["data\\images\\real\\a.jpg", "data/images/fake/b.jpg", "data/images/real/c.jpg"]
    manifest = ["data/images/real/c.jpg", "data\\images\\real\\a.jpg", "data/images/fake/b.jpg"]
    assert match_rows(manifest, cache).tolist() == [2, 0, 1]


def test_match_rows_fails_loudly_on_missing_and_duplicate_paths():
    from scripts.session_holdout_probes import match_rows

    with pytest.raises(ValueError, match="missing"):
        match_rows(["data/images/real/zzz.jpg"], ["data/images/real/a.jpg"])
    with pytest.raises(ValueError, match="duplicate"):
        match_rows(["a.jpg"], ["data/x/a.jpg".replace("data/x/", ""), "a.jpg"])


def test_reproduction_gaps_flags_a_difference():
    from scripts.session_holdout_probes import reproduction_gaps
    from session_holdout_cv import COLOUR, EXIF

    res = pd.DataFrame({"direction": ["d", "d"], "method": [COLOUR, EXIF], "auc_sample": [1.0, 0.6]})
    ref = pd.DataFrame({"direction": ["d", "d"], "method": [COLOUR, EXIF], "auc_sample": [1.0, 0.5]})
    gaps = reproduction_gaps(res, ref)
    assert np.allclose(gaps.gap.to_numpy(), [0.0, 0.1])
