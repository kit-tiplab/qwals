"""Tests for the disk cache (``_cache.py`` + the ``cache=`` constructor knob)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qwals import QwalsCalculator
from qwals import _cache


def _write_tiny_dataset(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "wals-data.csv"
    order = tmp_path / "WALS_feature_order.csv"
    pd.DataFrame(
        [
            {"Language_ID": "pol", "Language_name": "Polish",  "Parameter_name": "F1", "Value": "Low"},
            {"Language_ID": "eng", "Language_name": "English", "Parameter_name": "F1", "Value": "High"},
            {"Language_ID": "ger", "Language_name": "German",  "Parameter_name": "F1", "Value": "Average"},
        ]
    ).to_csv(data, index=False)
    pd.DataFrame([{"NAME": "F1", "VALUES IN ORDER": "Low,Average,High"}]).to_csv(order, index=False)
    return data, order


def test_cache_round_trip(tmp_path):
    """A second build with cache=True should populate from the .npz, not the CSV."""
    data, order = _write_tiny_dataset(tmp_path)
    cache_dir = Path(os.environ["QWALS_CACHE_DIR"])

    c1 = QwalsCalculator(data, order, cache=True)
    files = list(cache_dir.glob("*.npz"))
    assert len(files) == 1, f"expected exactly one cache file, got: {files}"

    c2 = QwalsCalculator(data, order, cache=True)

    # Every visible attribute should match exactly.
    assert c1.languages == c2.languages
    assert c1.features == c2.features
    assert c1.feature_orders == c2.feature_orders
    assert c1._alias == c2._alias
    assert c1._synth_to_val == c2._synth_to_val
    assert np.array_equal(c1._ord, c2._ord)
    assert np.array_equal(c1._val, c2._val)
    assert np.array_equal(c1._n_ord, c2._n_ord)

    # And distances must match bit-for-bit.
    assert c1.distance("Polish", "English") == c2.distance("Polish", "English")
    assert c1.distance("pl", "en", method="onehot") == c2.distance("pl", "en", method="onehot")


def test_cache_disabled_writes_nothing(tmp_path):
    data, order = _write_tiny_dataset(tmp_path)
    cache_dir = Path(os.environ["QWALS_CACHE_DIR"])

    QwalsCalculator(data, order, cache=False)
    assert not cache_dir.exists() or not list(cache_dir.glob("*.npz"))


def test_cache_custom_path(tmp_path):
    data, order = _write_tiny_dataset(tmp_path)
    custom = tmp_path / "out" / "my_wals.npz"

    c1 = QwalsCalculator(data, order, cache=custom)
    assert custom.exists() and custom.stat().st_size > 0

    # A second build from the same custom path skips CSV parsing.
    c2 = QwalsCalculator(data, order, cache=custom)
    assert c1.languages == c2.languages
    assert np.array_equal(c1._ord, c2._ord)


def test_cache_invalidates_on_option_change(tmp_path):
    """Changing an option that affects the matrices must produce a new cache file."""
    data, order = _write_tiny_dataset(tmp_path)
    cache_dir = Path(os.environ["QWALS_CACHE_DIR"])

    QwalsCalculator(data, order, cache=True, inferred_order_method="appearance")
    QwalsCalculator(data, order, cache=True, inferred_order_method="sorted")

    files = sorted(cache_dir.glob("*.npz"))
    assert len(files) == 2, f"expected two cache files (one per option), got: {files}"


def test_cache_invalidates_on_data_change(tmp_path, monkeypatch):
    """Editing the source CSV must trigger a rebuild rather than serve stale data."""
    data, order = _write_tiny_dataset(tmp_path)

    c1 = QwalsCalculator(data, order, cache=True)
    assert "Polish" in c1.languages

    # Replace Polish with Ukrainian. Bump mtime to ensure the fingerprint differs
    # even on filesystems with low mtime resolution.
    pd.DataFrame(
        [
            {"Language_ID": "ukr", "Language_name": "Ukrainian", "Parameter_name": "F1", "Value": "Low"},
            {"Language_ID": "eng", "Language_name": "English",   "Parameter_name": "F1", "Value": "High"},
            {"Language_ID": "ger", "Language_name": "German",    "Parameter_name": "F1", "Value": "Average"},
        ]
    ).to_csv(data, index=False)
    st = data.stat()
    os.utime(data, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))  # +10 s

    c2 = QwalsCalculator(data, order, cache=True)
    assert "Polish" not in c2.languages
    assert "Ukrainian" in c2.languages


def test_cache_load_returns_none_for_corrupt_file(tmp_path):
    """A junk .npz at the expected path must be treated as a miss, not crash."""
    data, order = _write_tiny_dataset(tmp_path)
    custom = tmp_path / "corrupt.npz"
    custom.write_bytes(b"this is not a real npz file")

    # Should silently fall back to CSV parsing and overwrite the cache.
    c = QwalsCalculator(data, order, cache=custom)
    assert "Polish" in c.languages
    # And now the file should be a valid cache.
    assert _cache.load(custom, key=_cache.cache_key(
        data_path=data, order_path=order,
        infer_missing_orders=True, inferred_order_method="appearance",
        package_version=__import__("qwals").__version__,
    ), package_version=__import__("qwals").__version__) is not None


def test_cache_key_stable_for_same_inputs(tmp_path):
    """The cache key must not depend on irrelevant call site detail."""
    data, order = _write_tiny_dataset(tmp_path)
    k1 = _cache.cache_key(
        data_path=data, order_path=order,
        infer_missing_orders=True, inferred_order_method="appearance",
        package_version="0.4.0",
    )
    k2 = _cache.cache_key(
        data_path=data, order_path=order,
        infer_missing_orders=True, inferred_order_method="appearance",
        package_version="0.4.0",
    )
    assert k1 == k2

    # Different package version → different key (so cache invalidates on upgrades).
    k3 = _cache.cache_key(
        data_path=data, order_path=order,
        infer_missing_orders=True, inferred_order_method="appearance",
        package_version="0.5.0",
    )
    assert k1 != k3
