"""Frozen-embedding cache must save and reload without pickle (no model download, no real images)."""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def module(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "frozen_embedding_cv", os.path.join(ROOT, "scripts", "frozen_embedding_cv.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_load_image", lambda path, size, train: tf.zeros((size, size, 3)))
    return mod


def _fake_model(*args, **kwargs):
    return lambda batch, training=False: tf.reduce_mean(batch, axis=(1, 2)) + 1.0


def _manifest(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.jpg"
        p.write_bytes(b"x" * (i + 1))
        paths.append(str(p))
    return pd.DataFrame({"path": paths})


def test_cache_round_trips_without_pickle(module, monkeypatch, tmp_path):
    manifest, cache = _manifest(tmp_path), tmp_path / "cache.npz"
    monkeypatch.setattr(module.tf.keras.applications, "ConvNeXtTiny", _fake_model)
    first = module.extract_frozen_embeddings(manifest, cache, img_size=8, batch_size=2)

    with np.load(cache, allow_pickle=False) as saved:  # raises if any array is an object array
        assert saved["paths"].dtype.kind == "U"

    def must_not_build(*args, **kwargs):
        raise AssertionError("cache was not reused")

    monkeypatch.setattr(module.tf.keras.applications, "ConvNeXtTiny", must_not_build)
    second = module.extract_frozen_embeddings(manifest, cache, img_size=8, batch_size=2)
    np.testing.assert_array_equal(first, second)


def test_legacy_object_array_cache_is_rebuilt(module, monkeypatch, tmp_path):
    manifest, cache = _manifest(tmp_path), tmp_path / "cache.npz"
    legacy_paths = np.array(manifest.path.tolist(), dtype=object)
    np.savez_compressed(cache, paths=legacy_paths, sizes=np.array([1, 2, 3]),
                        img_size=np.asarray(8), embeddings=np.zeros((3, 3), dtype=np.float32))
    monkeypatch.setattr(module.tf.keras.applications, "ConvNeXtTiny", _fake_model)
    rebuilt = module.extract_frozen_embeddings(manifest, cache, img_size=8, batch_size=2)
    assert rebuilt.shape == (3, 3) and np.all(rebuilt == 1.0)
    with np.load(cache, allow_pickle=False) as saved:
        assert saved["paths"].dtype.kind == "U"
