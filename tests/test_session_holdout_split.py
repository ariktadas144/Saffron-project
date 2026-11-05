from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _frame(n_real_a=5, n_fake_a=3, n_real_b=12, n_fake_b=5) -> pd.DataFrame:
    rows = []
    def add(prefix, n, label, date, camera):
        for i in range(n):
            for shot in (1, 2):
                rows.append({"path": f"{prefix}{i}_{shot}.jpg", "group": f"{prefix}{i}", "label": label,
                             "camera": camera, "capture_date": date})
    add("ra", n_real_a, 0, "2021:04:05", "samsung SM-G973F")
    add("fa", n_fake_a, 1, "2021:04:05", "samsung SM-G973F")
    add("rb", n_real_b, 0, "2021:04:06", "samsung SM-G973F")
    add("fb", n_fake_b, 1, "2021:04:06", "samsung SM-G973F")
    add("rj", 4, 0, "2021:04:05", "samsung SM-J600G")  # other camera must be dropped
    return pd.DataFrame(rows)


def test_split_is_group_disjoint_cross_session_and_stratified():
    from scripts.session_holdout_cv import build_split

    s = build_split(_frame(), "2021-04-05", "2021-04-06")
    assert not s.group.str.startswith("rj").any()
    assert s.groupby("group").fold.nunique().max() == 1
    test, val, train = (s[s.fold == c] for c in (0, 1, 2))
    assert set(test.capture_date) == {"2021:04:06"} and test.group.nunique() == 17
    assert set(val.capture_date) | set(train.capture_date) == {"2021:04:05"}
    assert val.drop_duplicates("group").label.value_counts().to_dict() == {0: 1, 1: 1}
    assert train.drop_duplicates("group").label.value_counts().to_dict() == {0: 4, 1: 2}
    assert len(s) == 2 * (5 + 3 + 12 + 5)  # all shots of every kept group


def test_split_is_deterministic_and_reverse_direction_swaps_roles():
    from scripts.session_holdout_cv import build_split

    a = build_split(_frame(), "2021:04:05", "2021:04:06", seed=42)
    b = build_split(_frame(), "2021:04:05", "2021:04:06", seed=42)
    pd.testing.assert_frame_equal(a, b)
    rev = build_split(_frame(), "2021:04:06", "2021:04:05")
    assert set(rev[rev.fold == 0].capture_date) == {"2021:04:05"}
    assert rev[rev.fold == 0].group.nunique() == 8
    assert rev[rev.fold == 1].drop_duplicates("group").label.value_counts().to_dict() == {0: 2, 1: 1}


def test_split_rejects_bad_inputs():
    from scripts.session_holdout_cv import build_split

    with pytest.raises(ValueError, match="differ"):
        build_split(_frame(), "2021:04:05", "2021:04:05")
    with pytest.raises(ValueError, match="cannot hold out"):
        build_split(_frame(n_fake_a=1), "2021:04:05", "2021:04:06")
    bad = _frame()
    bad.loc[bad.group == "ra0", "label"] = [0, 1]
    with pytest.raises(ValueError, match="one label"):
        build_split(bad, "2021:04:05", "2021:04:06")
