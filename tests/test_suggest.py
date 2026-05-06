"""Tests for v0.8 suggest_transfer_source() ranking utility."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qwals import QwalsCalculator


def _calc(tmp_path):
    """4-language fixture; English/German/Russian as candidate sources for Polish."""
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
    ).to_csv(tmp_path / "wals-data.csv", index=False)
    pd.DataFrame(
        [
            {"NAME": "F1", "VALUES IN ORDER": "Low,Average,High"},
            {"NAME": "F2", "VALUES IN ORDER": "No,Yes"},
            {"NAME": "F3", "VALUES IN ORDER": "Type A,Type B"},
        ]
    ).to_csv(tmp_path / "WALS_feature_order.csv", index=False)
    return QwalsCalculator(tmp_path / "wals-data.csv", tmp_path / "WALS_feature_order.csv")


def test_suggest_returns_ranked_dicts(tmp_path):
    c = _calc(tmp_path)
    out = c.suggest_transfer_source("Polish", n=3, min_shared=0)
    assert all({"language", "distance", "n_shared", "coverage", "confidence"}
               <= set(r.keys()) for r in out)
    # Sorted ascending by distance.
    assert [r["distance"] for r in out] == sorted(r["distance"] for r in out)


def test_suggest_excludes_target(tmp_path):
    c = _calc(tmp_path)
    out = c.suggest_transfer_source("Polish", n=10, min_shared=0)
    assert all(r["language"] != "Polish" for r in out)


def test_suggest_with_explicit_candidates(tmp_path):
    c = _calc(tmp_path)
    out = c.suggest_transfer_source(
        "Polish", candidates=["English", "German"], n=10, min_shared=0,
    )
    names = {r["language"] for r in out}
    assert names == {"English", "German"}


def test_suggest_does_not_mutate_active_mask(tmp_path):
    c = _calc(tmp_path)
    c.use_features(["F1"])           # arbitrary preset
    mask_before = c._feature_mask.copy()
    preset_before = c.active_preset
    # task=None means: don't touch the mask. Use min_shared=0 on this fixture.
    c.suggest_transfer_source("Polish", n=3, min_shared=0)
    assert np.array_equal(c._feature_mask, mask_before)
    assert c.active_preset == preset_before


def test_suggest_with_task_restores_mask(tmp_path):
    """Even when task= temporarily applies a preset, the active mask
    must be restored when the call returns."""
    c = _calc(tmp_path)
    c.use_features(["F2"])
    mask_before = c._feature_mask.copy()
    # Use a paper preset; it'll likely match nothing in our fixture but
    # the call must still leave the mask untouched on exit.
    try:
        c.suggest_transfer_source("Polish", task="dep", n=3, min_shared=0)
    except (ValueError, KeyError):
        pass    # may raise if nothing in dep matches our fixture; that's OK
    assert np.array_equal(c._feature_mask, mask_before)


def test_suggest_alias_input(tmp_path):
    c = _calc(tmp_path)
    a = c.suggest_transfer_source("Polish", n=3, min_shared=0)
    b = c.suggest_transfer_source("pl", n=3, min_shared=0)
    assert [r["language"] for r in a] == [r["language"] for r in b]


def test_suggest_confidence_in_unit_interval(tmp_path):
    c = _calc(tmp_path)
    out = c.suggest_transfer_source("Polish", n=3, min_shared=0)
    for r in out:
        assert 0.0 <= r["confidence"] <= 1.0
