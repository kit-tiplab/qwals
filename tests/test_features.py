"""Tests for v0.5 features:
   - per-feature weights
   - nearest()
   - distance_to_many()
   - shared_features() / features_for()
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qwals import QwalsCalculator


# ---------- shared fixture --------------------------------------------------

def _build_calc(tmp_path: Path) -> QwalsCalculator:
    """A 4-language × 3-feature dataset that's easy to reason about."""
    data = tmp_path / "wals-data.csv"
    order = tmp_path / "WALS_feature_order.csv"

    rows = [
        # Polish
        {"Language_ID": "pol", "Language_name": "Polish",  "Parameter_name": "F1", "Value": "Low"},
        {"Language_ID": "pol", "Language_name": "Polish",  "Parameter_name": "F2", "Value": "Yes"},
        {"Language_ID": "pol", "Language_name": "Polish",  "Parameter_name": "F3", "Value": "Type A"},
        # English: F1 differs, F2 same, F3 same
        {"Language_ID": "eng", "Language_name": "English", "Parameter_name": "F1", "Value": "High"},
        {"Language_ID": "eng", "Language_name": "English", "Parameter_name": "F2", "Value": "Yes"},
        {"Language_ID": "eng", "Language_name": "English", "Parameter_name": "F3", "Value": "Type A"},
        # German: F1 mid, F2 same, F3 differs (and only has F1+F3, missing F2)
        {"Language_ID": "ger", "Language_name": "German",  "Parameter_name": "F1", "Value": "Average"},
        {"Language_ID": "ger", "Language_name": "German",  "Parameter_name": "F3", "Value": "Type B"},
        # Russian: only F1, identical to Polish
        {"Language_ID": "rus", "Language_name": "Russian", "Parameter_name": "F1", "Value": "Low"},
    ]
    pd.DataFrame(rows).to_csv(data, index=False)

    pd.DataFrame(
        [
            {"NAME": "F1", "VALUES IN ORDER": "Low,Average,High"},
            {"NAME": "F2", "VALUES IN ORDER": "No,Yes"},
            {"NAME": "F3", "VALUES IN ORDER": "Type A,Type B"},
        ]
    ).to_csv(order, index=False)
    return QwalsCalculator(data, order)


# =====================================================================
# Per-feature weights
# =====================================================================

def test_default_weights_are_one(tmp_path):
    c = _build_calc(tmp_path)
    assert c.weights == {}                          # empty dict = all 1.0
    assert np.allclose(c._weights, 1.0)


def test_weights_via_constructor(tmp_path):
    data = tmp_path / "wals-data.csv"
    order = tmp_path / "WALS_feature_order.csv"
    _build_calc(tmp_path)  # to write CSVs
    c = QwalsCalculator(data, order, weights={"F1": 2.0, "F2": 0.5})
    assert c.weights == {"F1": 2.0, "F2": 0.5}
    assert c._weights[c._feat_idx["F3"]] == 1.0     # untouched stays at 1.0


def test_weight_doubles_feature_contribution(tmp_path):
    """Polish vs English share 3 features (F1, F2, F3). With equal weights:
    F1: |0-2|/2 = 1.0, F2: 0, F3: 0  →  mean = 1/3.
    Doubling F1's weight gives  (2*1 + 1*0 + 1*0) / (2+1+1) = 0.5.
    """
    c = _build_calc(tmp_path)
    assert c.distance("Polish", "English") == pytest.approx(1 / 3, abs=1e-6)
    c.set_weight("F1", 2.0)
    assert c.distance("Polish", "English") == pytest.approx(0.5, abs=1e-6)


def test_zero_weight_drops_feature(tmp_path):
    c = _build_calc(tmp_path)
    # Polish vs German: shared features are F1 and F3.
    # F1: |Low(0)-Average(1)|/2 = 0.5; F3: differs → 1.0.  mean = 0.75.
    assert c.distance("Polish", "German") == pytest.approx(0.75, abs=1e-6)
    # Drop F3 entirely → only F1 remains: distance = 0.5.
    c.set_weight("F3", 0.0)
    assert c.distance("Polish", "German") == pytest.approx(0.5, abs=1e-6)


def test_reset_weights(tmp_path):
    c = _build_calc(tmp_path)
    c.set_weight("F1", 3.0)
    c.reset_weights()
    assert c.weights == {}
    assert np.allclose(c._weights, 1.0)


def test_weights_invalid_raise(tmp_path):
    c = _build_calc(tmp_path)
    with pytest.raises(ValueError):
        c.set_weight("F1", -1.0)                    # negative
    with pytest.raises(ValueError):
        c.set_weight("F1", float("inf"))            # infinite
    with pytest.raises(ValueError):
        c.set_weight("nonexistent_feature", 1.0)    # unknown feature


def test_weights_affect_pairwise(tmp_path):
    c = _build_calc(tmp_path)
    m_default = c.pairwise_matrix(method="ordinal")
    c.set_weight("F1", 2.0)
    m_weighted = c.pairwise_matrix(method="ordinal")
    # Polish-English distance should match the single-pair calculation.
    assert m_weighted.loc["Polish", "English"] == pytest.approx(0.5, abs=1e-6)
    # Default still matches single-pair.
    assert m_default.loc["Polish", "English"] == pytest.approx(1 / 3, abs=1e-6)


def test_weights_in_details_dataframe(tmp_path):
    c = _build_calc(tmp_path)
    c.set_weight("F1", 2.5)
    out = c.distance("Polish", "English", return_details=True)
    df = out["details"]
    assert "Weight" in df.columns
    f1_row = df[df["Feature"] == "F1"].iloc[0]
    assert f1_row["Weight"] == 2.5


def test_weights_not_in_details_when_unused(tmp_path):
    """If all weights are 1.0, details should not have a Weight column (cleaner output)."""
    c = _build_calc(tmp_path)
    out = c.distance("Polish", "English", return_details=True)
    assert "Weight" not in out["details"].columns


# =====================================================================
# nearest()
# =====================================================================

def test_nearest_basic(tmp_path):
    """Tiny 3-feature fixture — we have to pass min_shared=0 because the
    real-WALS-tuned default of 50 would exclude every fixture language."""
    c = _build_calc(tmp_path)
    out = c.nearest("Polish", n=3, min_shared=0)
    names = [n for n, _ in out]
    # Russian shares only F1 and matches → distance 0.0; should be top.
    assert names[0] == "Russian"
    assert out[0][1] == pytest.approx(0.0, abs=1e-6)
    # Polish itself excluded by default.
    assert "Polish" not in names


def test_nearest_includes_self_when_asked(tmp_path):
    c = _build_calc(tmp_path)
    out = c.nearest("Polish", n=2, include_self=True, min_shared=0)
    assert out[0] == ("Polish", 0.0)


def test_nearest_n_clamps_to_available(tmp_path):
    c = _build_calc(tmp_path)
    # We have 4 langs, only 3 are non-self.
    assert len(c.nearest("Polish", n=100, min_shared=0)) == 3


def test_nearest_handles_n_zero(tmp_path):
    c = _build_calc(tmp_path)
    assert c.nearest("Polish", n=0) == []


def test_nearest_method_onehot(tmp_path):
    c = _build_calc(tmp_path)
    out = c.nearest("Polish", n=3, method="onehot", min_shared=0)
    # Russian is identical on shared features → distance 0.
    assert out[0][0] == "Russian"


def test_nearest_honors_weights(tmp_path):
    c = _build_calc(tmp_path)
    # Set F1 weight to zero — Polish/Russian share ONLY F1, so they end up
    # with no weighted shared features → Russian falls out of the result.
    c.set_weight("F1", 0.0)
    out = c.nearest("Polish", n=3, min_shared=0)
    names = [n for n, _ in out]
    assert "Russian" not in names


def test_nearest_default_min_shared_filters_fixture(tmp_path):
    """The fix in 0.7: with the default min_shared (50), no 3-feature
    fixture language survives the threshold, so the result is empty.
    This is intentional — empty is the right answer when no language
    shares enough features to make a defensible nearest-claim."""
    c = _build_calc(tmp_path)
    assert c.nearest("Polish", n=10) == []


def test_nearest_min_shared_threshold_explicit(tmp_path):
    """Russian shares 1 feature with Polish in this fixture. min_shared=1
    keeps it; min_shared=2 drops it."""
    c = _build_calc(tmp_path)
    assert "Russian" in [n for n, _ in c.nearest("Polish", n=5, min_shared=1)]
    assert "Russian" not in [n for n, _ in c.nearest("Polish", n=5, min_shared=2)]


def test_nearest_min_shared_does_not_silently_change_with_weights(tmp_path):
    """Reweighting a feature must not shift the min_shared cutoff —
    it's measured on the unweighted shared-feature count."""
    c = _build_calc(tmp_path)
    out_default = c.nearest("Polish", n=5, min_shared=2)
    c.set_weights({"F1": 0.1, "F2": 0.1, "F3": 0.1})  # weighted total drops 10×
    out_weighted = c.nearest("Polish", n=5, min_shared=2)
    # Same languages survive the filter; only the distance values change.
    assert [n for n, _ in out_default] == [n for n, _ in out_weighted]


def test_nearest_class_default_constant_exposed(tmp_path):
    """The default lives on the class as NEAREST_MIN_SHARED so users can
    monkey-patch or subclass to change it."""
    c = _build_calc(tmp_path)
    assert isinstance(c.NEAREST_MIN_SHARED, int)
    assert c.NEAREST_MIN_SHARED >= 1


def test_nearest_alias_input(tmp_path):
    c = _build_calc(tmp_path)
    a = c.nearest("Polish")
    b = c.nearest("pl")           # ISO code
    c2 = c.nearest("pol")          # WALS Language_ID
    assert a == b == c2


# =====================================================================
# distance_to_many()
# =====================================================================

def test_distance_to_many_default(tmp_path):
    c = _build_calc(tmp_path)
    d = c.distance_to_many("Polish")
    assert "Polish" not in d                        # self excluded by default
    assert set(d.keys()) == {"English", "German", "Russian"}
    assert d["Russian"] == pytest.approx(0.0, abs=1e-6)


def test_distance_to_many_explicit_others(tmp_path):
    c = _build_calc(tmp_path)
    d = c.distance_to_many("Polish", others=["English", "German"])
    assert set(d.keys()) == {"English", "German"}


def test_distance_to_many_aliases(tmp_path):
    c = _build_calc(tmp_path)
    d = c.distance_to_many("pl", others=["en", "de"])
    assert "English" in d and "German" in d         # canonical names returned


def test_distance_to_many_as_series(tmp_path):
    c = _build_calc(tmp_path)
    s = c.distance_to_many("Polish", as_series=True)
    assert isinstance(s, pd.Series)
    assert s.name == "distance_from_Polish"
    assert "Russian" in s.index


def test_distance_to_many_include_self(tmp_path):
    c = _build_calc(tmp_path)
    d = c.distance_to_many("Polish", include_self=True)
    assert d["Polish"] == 0.0


def test_distance_to_many_matches_distance(tmp_path):
    """Sanity: each one-vs-many entry should equal the per-pair distance()."""
    c = _build_calc(tmp_path)
    d = c.distance_to_many("Polish")
    for other, val in d.items():
        assert val == pytest.approx(c.distance("Polish", other), abs=1e-6), other


def test_distance_to_many_min_shared_filter(tmp_path):
    """min_shared on distance_to_many drops below-threshold entries from
    the result dict (rather than reporting them as inf)."""
    c = _build_calc(tmp_path)
    d = c.distance_to_many("Polish", min_shared=2)
    # Russian shares only F1 (1 feature) with Polish — drops out at min_shared=2.
    assert "Russian" not in d
    # English and German share F1+F2+F3 and F1+F3 respectively → still in.
    assert "English" in d and "German" in d


def test_distance_to_many_min_shared_default_is_zero(tmp_path):
    """Backward-compatible: distance_to_many keeps the existing
    'return everything' default; only nearest got the new opinionated default."""
    c = _build_calc(tmp_path)
    d = c.distance_to_many("Polish")
    assert "Russian" in d              # 1 shared, would be cut by any min_shared>=2


# =====================================================================
# shared_features() / features_for()
# =====================================================================

def test_features_for_basic(tmp_path):
    c = _build_calc(tmp_path)
    assert c.features_for("Polish") == ["F1", "F2", "F3"]
    assert c.features_for("German") == ["F1", "F3"]   # F2 missing in fixture
    assert c.features_for("Russian") == ["F1"]


def test_features_for_alias(tmp_path):
    c = _build_calc(tmp_path)
    assert c.features_for("pl") == c.features_for("Polish")


def test_shared_features_basic(tmp_path):
    c = _build_calc(tmp_path)
    assert c.shared_features("Polish", "English") == ["F1", "F2", "F3"]
    assert c.shared_features("Polish", "German") == ["F1", "F3"]
    assert c.shared_features("Polish", "Russian") == ["F1"]


def test_shared_features_symmetric(tmp_path):
    c = _build_calc(tmp_path)
    assert c.shared_features("Polish", "German") == c.shared_features("German", "Polish")


def test_shared_features_aliases(tmp_path):
    c = _build_calc(tmp_path)
    assert c.shared_features("pl", "de") == c.shared_features("Polish", "German")
