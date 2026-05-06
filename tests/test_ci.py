"""Tests for v0.8 distance_ci() bootstrap CI + nearest(sort_by='upper_ci')."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qwals import QwalsCalculator


def _calc(tmp_path):
    """20-feature fixture so the bootstrap has something to resample over."""
    rows = []
    feature_orders = []
    n_features = 20
    for fi in range(n_features):
        feat = f"F{fi:02d}"
        feature_orders.append({"NAME": feat, "VALUES IN ORDER": "v0,v1,v2,v3"})
        # Polish vs English: half match (fi even) and half don't (fi odd).
        rows.append({"Language_ID": "pol", "Language_name": "Polish",  "Parameter_name": feat, "Value": "v0"})
        rows.append({"Language_ID": "eng", "Language_name": "English", "Parameter_name": feat,
                     "Value": "v0" if fi % 2 == 0 else "v3"})
        # Sparse Klingon: only first 4 features documented.
        if fi < 4:
            rows.append({"Language_ID": "tlh", "Language_name": "Klingon", "Parameter_name": feat,
                         "Value": "v0" if fi == 0 else "v3"})
    pd.DataFrame(rows).to_csv(tmp_path / "wals-data.csv", index=False)
    pd.DataFrame(feature_orders).to_csv(tmp_path / "WALS_feature_order.csv", index=False)
    return QwalsCalculator(tmp_path / "wals-data.csv", tmp_path / "WALS_feature_order.csv")


# =====================================================================
# distance_ci()
# =====================================================================

def test_ci_brackets_point_estimate(tmp_path):
    c = _calc(tmp_path)
    res = c.distance_ci("Polish", "English", n_bootstrap=2000, rng=42)
    assert res["ci_low"] <= res["distance"] <= res["ci_high"]


def test_ci_reproducible_with_rng_seed(tmp_path):
    c = _calc(tmp_path)
    a = c.distance_ci("Polish", "English", n_bootstrap=500, rng=123)
    b = c.distance_ci("Polish", "English", n_bootstrap=500, rng=123)
    assert a == b


def test_ci_different_seeds_differ(tmp_path):
    c = _calc(tmp_path)
    a = c.distance_ci("Polish", "English", n_bootstrap=500, rng=1)
    b = c.distance_ci("Polish", "English", n_bootstrap=500, rng=2)
    assert (a["ci_low"], a["ci_high"]) != (b["ci_low"], b["ci_high"])


def test_ci_widens_for_sparse_pair(tmp_path):
    """Klingon has 4 shared features with Polish vs English's 20.
    The bootstrap CI on Polish↔Klingon should be visibly wider."""
    c = _calc(tmp_path)
    well = c.distance_ci("Polish", "English", n_bootstrap=2000, rng=42)
    sparse = c.distance_ci("Polish", "Klingon", n_bootstrap=2000, rng=42)
    well_width = well["ci_high"] - well["ci_low"]
    sparse_width = sparse["ci_high"] - sparse["ci_low"]
    assert sparse_width > well_width


def test_ci_invalid_inputs_raise(tmp_path):
    c = _calc(tmp_path)
    with pytest.raises(ValueError, match="ci must be in"):
        c.distance_ci("Polish", "English", ci=1.5)
    with pytest.raises(ValueError, match="n_bootstrap must be"):
        c.distance_ci("Polish", "English", n_bootstrap=0)


def test_ci_metadata_in_result(tmp_path):
    c = _calc(tmp_path)
    res = c.distance_ci("Polish", "English", n_bootstrap=300, ci=0.9, rng=0)
    assert res["n_bootstrap"] == 300
    assert res["ci_level"] == 0.9
    assert res["language_1"] == "Polish"
    assert res["language_2"] == "English"


# =====================================================================
# nearest(sort_by='upper_ci')
# =====================================================================

def test_nearest_with_coverage(tmp_path):
    c = _calc(tmp_path)
    out = c.nearest("Polish", n=2, with_coverage=True, min_shared=0)
    assert all(len(t) == 3 for t in out)
    # The shared count for Polish↔English is 20 (all features).
    by_name = dict((t[0], t) for t in out)
    if "English" in by_name:
        assert by_name["English"][2] == 20


def test_nearest_sort_by_upper_ci_returns_4tuples(tmp_path):
    c = _calc(tmp_path)
    out = c.nearest(
        "Polish", n=3,
        sort_by="upper_ci", n_bootstrap=200,
        min_shared=0, rng=42,
    )
    assert all(len(t) == 4 for t in out)
    # Within the result, ci_high values should be non-decreasing.
    ci_highs = [t[3] for t in out]
    assert ci_highs == sorted(ci_highs)


def test_nearest_upper_ci_requires_bootstrap(tmp_path):
    c = _calc(tmp_path)
    with pytest.raises(ValueError, match="n_bootstrap"):
        c.nearest("Polish", n=3, sort_by="upper_ci", min_shared=0)


def test_nearest_invalid_sort_by_raises(tmp_path):
    c = _calc(tmp_path)
    with pytest.raises(ValueError, match="sort_by"):
        c.nearest("Polish", n=3, sort_by="banana", min_shared=0)
