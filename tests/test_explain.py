"""Tests for v0.8 explain_distance() + per-distance coverage returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qwals import QwalsCalculator


def _calc(tmp_path):
    """4-language × 3-feature fixture; English shares all 3 features with Polish,
    German shares F1+F3, Russian shares only F1."""
    data = tmp_path / "wals-data.csv"
    order = tmp_path / "WALS_feature_order.csv"
    pd.DataFrame(
        [
            {"Language_ID": "pol", "Language_name": "Polish",  "Parameter_name": "F1", "Value": "Low"},
            {"Language_ID": "pol", "Language_name": "Polish",  "Parameter_name": "F2", "Value": "Yes"},
            {"Language_ID": "pol", "Language_name": "Polish",  "Parameter_name": "F3", "Value": "Type A"},
            {"Language_ID": "eng", "Language_name": "English", "Parameter_name": "F1", "Value": "High"},
            {"Language_ID": "eng", "Language_name": "English", "Parameter_name": "F2", "Value": "Yes"},
            {"Language_ID": "eng", "Language_name": "English", "Parameter_name": "F3", "Value": "Type A"},
            {"Language_ID": "ger", "Language_name": "German",  "Parameter_name": "F1", "Value": "Average"},
            {"Language_ID": "ger", "Language_name": "German",  "Parameter_name": "F3", "Value": "Type B"},
            {"Language_ID": "rus", "Language_name": "Russian", "Parameter_name": "F1", "Value": "Low"},
        ]
    ).to_csv(data, index=False)
    pd.DataFrame(
        [
            {"NAME": "F1", "VALUES IN ORDER": "Low,Average,High"},
            {"NAME": "F2", "VALUES IN ORDER": "No,Yes"},
            {"NAME": "F3", "VALUES IN ORDER": "Type A,Type B"},
        ]
    ).to_csv(order, index=False)
    return QwalsCalculator(data, order)


# =====================================================================
# explain_distance()
# =====================================================================

def test_explain_distance_returns_dataframe(tmp_path):
    c = _calc(tmp_path)
    df = c.explain_distance("Polish", "English")
    assert isinstance(df, pd.DataFrame)
    assert {"Feature", "per_feature_distance", "Weight", "contribution"}.issubset(df.columns)
    # 3 shared features → 3 rows.
    assert len(df) == 3


def test_explain_contributions_sum_to_one(tmp_path):
    c = _calc(tmp_path)
    df = c.explain_distance("Polish", "English", top_k=None)
    assert df["contribution"].sum() == pytest.approx(1.0, abs=1e-6)


def test_explain_distance_sorted_descending(tmp_path):
    """The largest-contribution feature comes first."""
    c = _calc(tmp_path)
    df = c.explain_distance("Polish", "English", top_k=None)
    contribs = df["contribution"].tolist()
    assert contribs == sorted(contribs, reverse=True)


def test_explain_distance_top_k_limits_rows(tmp_path):
    c = _calc(tmp_path)
    df = c.explain_distance("Polish", "English", top_k=2)
    assert len(df) == 2


def test_explain_distance_respects_weights(tmp_path):
    c = _calc(tmp_path)
    c.set_weight("F1", 5.0)            # F1 differs (Low vs High) → big contribution
    df = c.explain_distance("Polish", "English", top_k=None)
    f1_row = df[df["Feature"] == "F1"].iloc[0]
    assert f1_row["Weight"] == 5.0
    # F1 should now dominate the contributions.
    assert f1_row["contribution"] > 0.7


def test_explain_distance_respects_active_mask(tmp_path):
    c = _calc(tmp_path)
    c.use_features(["F1", "F3"])       # drop F2
    df = c.explain_distance("Polish", "English", top_k=None)
    assert set(df["Feature"]) == {"F1", "F3"}


def test_explain_distance_self_returns_empty(tmp_path):
    c = _calc(tmp_path)
    df = c.explain_distance("Polish", "Polish")
    assert df.empty


def test_explain_distance_alias_input(tmp_path):
    c = _calc(tmp_path)
    df_a = c.explain_distance("Polish", "English")
    df_b = c.explain_distance("pl", "en")
    pd.testing.assert_frame_equal(df_a, df_b)


# =====================================================================
# distance(return_coverage=True) + coverage_for()
# =====================================================================

def test_distance_return_coverage(tmp_path):
    c = _calc(tmp_path)
    out = c.distance("Polish", "English", return_coverage=True)
    assert out["distance"] >= 0
    assert out["n_shared"] == 3                      # all three features shared
    assert out["n_total_features"] == 3
    assert out["coverage"] == 1.0


def test_distance_return_coverage_sparse_pair(tmp_path):
    c = _calc(tmp_path)
    out = c.distance("Polish", "Russian", return_coverage=True)
    # Russian has only F1.
    assert out["n_shared"] == 1
    assert out["coverage"] == pytest.approx(1 / 3, abs=1e-6)


def test_distance_return_coverage_self_returns_zero(tmp_path):
    c = _calc(tmp_path)
    out = c.distance("Polish", "Polish", return_coverage=True)
    assert out["distance"] == 0.0
    assert out["language_1"] == out["language_2"] == "Polish"


def test_distance_coverage_and_details_compose(tmp_path):
    c = _calc(tmp_path)
    out = c.distance(
        "Polish", "English",
        return_coverage=True, return_details=True,
    )
    assert "details" in out and "coverage" in out
    assert out["features_used"] == 3


def test_coverage_for_helper(tmp_path):
    c = _calc(tmp_path)
    cov = c.coverage_for("Polish", "Russian")
    assert cov["n_shared"] == 1
    assert cov["lang1"] == "Polish"
    assert cov["lang2"] == "Russian"
    assert cov["lang1_total"] == 3      # Polish has all three
    assert cov["lang2_total"] == 1      # Russian has only F1


def test_coverage_for_respects_active_mask(tmp_path):
    c = _calc(tmp_path)
    c.use_features(["F2"])              # F2 isn't in Russian
    cov = c.coverage_for("Polish", "Russian")
    assert cov["n_shared"] == 0
    assert cov["n_active"] == 1


def test_coverage_for_alias_input(tmp_path):
    c = _calc(tmp_path)
    a = c.coverage_for("pl", "en")
    b = c.coverage_for("Polish", "English")
    assert a == b
