"""Tests for v0.8 task-specific feature presets + use_features API."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qwals import QwalsCalculator, TASK_FEATURES, TASKS


def _toy_calc(tmp_path):
    """Tiny fixture with a couple of WALS-canonical feature names so the
    tests can exercise the preset machinery without loading real WALS."""
    data = tmp_path / "wals-data.csv"
    order = tmp_path / "WALS_feature_order.csv"
    pd.DataFrame(
        [
            # A handful of features that appear in multiple presets.
            {"Language_ID": "pol", "Language_name": "Polish",  "Parameter_name": "Number of Genders",            "Value": "Three"},
            {"Language_ID": "pol", "Language_name": "Polish",  "Parameter_name": "Reduplication",                "Value": "No"},
            {"Language_ID": "pol", "Language_name": "Polish",  "Parameter_name": "Tone",                         "Value": "No tones"},
            {"Language_ID": "eng", "Language_name": "English", "Parameter_name": "Number of Genders",            "Value": "None"},
            {"Language_ID": "eng", "Language_name": "English", "Parameter_name": "Reduplication",                "Value": "No"},
            {"Language_ID": "eng", "Language_name": "English", "Parameter_name": "Tone",                         "Value": "No tones"},
            # Filler so each lang has more features than in the preset.
            {"Language_ID": "pol", "Language_name": "Polish",  "Parameter_name": "Filler 1", "Value": "A"},
            {"Language_ID": "eng", "Language_name": "English", "Parameter_name": "Filler 1", "Value": "B"},
        ]
    ).to_csv(data, index=False)
    pd.DataFrame(
        [
            {"NAME": "Number of Genders", "VALUES IN ORDER": "None,Two,Three,Four,Five+"},
            {"NAME": "Reduplication",     "VALUES IN ORDER": "No,Yes"},
            {"NAME": "Tone",              "VALUES IN ORDER": "No tones,Simple,Complex"},
            {"NAME": "Filler 1",          "VALUES IN ORDER": "A,B"},
        ]
    ).to_csv(order, index=False)
    return QwalsCalculator(data, order)


def test_preset_constants_exposed():
    """The four task names + their feature lists are public."""
    assert set(TASKS) == {"abusive", "sentiment", "ner", "dep"}
    assert all(isinstance(TASK_FEATURES[t], tuple) for t in TASKS)
    # Feature counts match the paper's Appendix A totals.
    assert len(TASK_FEATURES["abusive"]) == 53
    assert len(TASK_FEATURES["sentiment"]) == 21
    assert len(TASK_FEATURES["ner"]) == 63
    assert len(TASK_FEATURES["dep"]) == 75


def test_default_feature_mask_is_all_true(tmp_path):
    c = _toy_calc(tmp_path)
    assert c.active_preset is None
    assert sum(c._feature_mask) == len(c.features)
    assert set(c.active_features) == set(c.features)


def test_use_features_preset_applies_intersection(tmp_path):
    c = _toy_calc(tmp_path)
    # 'dep' preset includes 'Number of Genders', 'Reduplication', 'Tone'.
    n = c.use_features("dep")
    assert n == 3                       # only those three are in our fixture
    assert c.active_preset == "dep"
    assert set(c.active_features) == {"Number of Genders", "Reduplication", "Tone"}


def test_use_features_unknown_preset_raises(tmp_path):
    c = _toy_calc(tmp_path)
    with pytest.raises(ValueError, match="Unknown preset"):
        c.use_features("not_a_real_task")


def test_use_features_explicit_list(tmp_path):
    c = _toy_calc(tmp_path)
    n = c.use_features(["Number of Genders", "Tone"])
    assert n == 2
    assert c.active_preset is None       # explicit list ≠ named preset


def test_use_features_silently_skips_missing(tmp_path):
    c = _toy_calc(tmp_path)
    n = c.use_features(["Number of Genders", "TotallyMadeUp Feature"])
    assert n == 1


def test_use_features_all_missing_raises(tmp_path):
    c = _toy_calc(tmp_path)
    with pytest.raises(ValueError, match="active mask would be empty"):
        c.use_features(["TotallyMadeUp 1", "TotallyMadeUp 2"])


def test_reset_features_restores_default(tmp_path):
    c = _toy_calc(tmp_path)
    c.use_features("dep")
    c.reset_features()
    assert sum(c._feature_mask) == len(c.features)
    assert c.active_preset is None


def test_use_features_changes_distance(tmp_path):
    """Restricting to features where Polish/English match gives lower distance."""
    c = _toy_calc(tmp_path)
    d_full = c.distance("Polish", "English")
    # Reduplication and Tone match (both "No"/"No tones"); restrict to them.
    c.use_features(["Reduplication", "Tone"])
    d_matching = c.distance("Polish", "English")
    assert d_matching == 0.0
    assert d_full > 0.0


def test_use_features_preserves_alias_resolution(tmp_path):
    c = _toy_calc(tmp_path)
    c.use_features("dep")
    assert c.distance("pl", "en") == c.distance("Polish", "English")
