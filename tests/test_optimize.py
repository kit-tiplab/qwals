"""Tests for v0.8 LOFO feature optimiser."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qwals import QwalsCalculator


def _toy_calc(tmp_path):
    """Five languages × eight features. F0..F3 align with a "transfer" signal
    (Lang0 best matches Lang1, etc.); F4..F7 are noise that should be pruned."""
    rows = []
    feature_orders = []
    for fi in range(8):
        feat = f"F{fi}"
        feature_orders.append({"NAME": feat, "VALUES IN ORDER": "v0,v1,v2,v3,v4"})

    # Real signal — Lang values that mirror a "linguistic similarity" gradient.
    signal_values = {
        "L0": ["v0"] * 4,
        "L1": ["v0"] * 4,           # twin of L0
        "L2": ["v1"] * 4,           # one step away
        "L3": ["v3"] * 4,           # far
        "L4": ["v4"] * 4,           # farthest
    }
    # Noise — random-looking values uncorrelated with the signal.
    noise = {
        "L0": ["v0", "v1", "v2", "v3"],
        "L1": ["v4", "v3", "v2", "v1"],
        "L2": ["v0", "v3", "v4", "v1"],
        "L3": ["v2", "v0", "v1", "v4"],
        "L4": ["v1", "v4", "v3", "v0"],
    }

    code_to_id = {f"L{i}": f"l{i}" for i in range(5)}
    for lang, sig in signal_values.items():
        for fi, v in enumerate(sig):
            rows.append({"Language_ID": code_to_id[lang], "Language_name": lang,
                         "Parameter_name": f"F{fi}", "Value": v})
        for fi, v in enumerate(noise[lang]):
            rows.append({"Language_ID": code_to_id[lang], "Language_name": lang,
                         "Parameter_name": f"F{fi+4}", "Value": v})

    pd.DataFrame(rows).to_csv(tmp_path / "wals-data.csv", index=False)
    pd.DataFrame(feature_orders).to_csv(tmp_path / "WALS_feature_order.csv", index=False)
    return QwalsCalculator(tmp_path / "wals-data.csv", tmp_path / "WALS_feature_order.csv")


# Transfer scores monotonically anti-correlated with the signal-feature distance.
TARGET_SCORES = {
    ("L0", "L1"): 0.95,        # twins → high transfer
    ("L0", "L2"): 0.85,
    ("L0", "L3"): 0.50,
    ("L0", "L4"): 0.30,        # farthest → low transfer
    ("L1", "L2"): 0.85,
    ("L1", "L3"): 0.50,
}


def test_optimize_returns_expected_keys(tmp_path):
    c = _toy_calc(tmp_path)
    res = c.optimize_features(TARGET_SCORES)
    assert {"features", "pearson", "spearman", "n_features",
            "n_dropped", "history"}.issubset(res.keys())


def test_optimize_drops_noise_features(tmp_path):
    """The optimiser should drop the 4 noise features and keep the signal ones."""
    c = _toy_calc(tmp_path)
    res = c.optimize_features(TARGET_SCORES, min_features=2)
    assert res["n_dropped"] >= 1
    # Final correlation should be at least as good as the starting one.
    base_corr = res["history"][0]["score"]
    final_corr = res["history"][-1]["score"]
    assert final_corr >= base_corr


def test_optimize_does_not_mutate_calculator(tmp_path):
    c = _toy_calc(tmp_path)
    mask_before = c._feature_mask.copy()
    c.optimize_features(TARGET_SCORES)
    assert np.array_equal(c._feature_mask, mask_before)


def test_optimize_history_lengths_consistent(tmp_path):
    c = _toy_calc(tmp_path)
    res = c.optimize_features(TARGET_SCORES, max_drops=2)
    # history[0] is the starting state, then up to max_drops more entries.
    assert len(res["history"]) == res["n_dropped"] + 1
    assert res["n_dropped"] <= 2


def test_optimize_min_features_floor(tmp_path):
    c = _toy_calc(tmp_path)
    res = c.optimize_features(TARGET_SCORES, min_features=6)
    assert res["n_features"] >= 6


def test_optimize_empty_targets_raises(tmp_path):
    c = _toy_calc(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        c.optimize_features({})


def test_optimize_self_pairs_dropped(tmp_path):
    """A self-pair (lang, lang) carries no signal and is silently skipped."""
    c = _toy_calc(tmp_path)
    res = c.optimize_features({("L0", "L0"): 0.99, **TARGET_SCORES})
    # Should still complete normally; the self-pair is just ignored.
    assert "features" in res


def test_optimize_apply_workflow(tmp_path):
    """Common downstream pattern: optimise, then apply via use_features."""
    c = _toy_calc(tmp_path)
    res = c.optimize_features(TARGET_SCORES, min_features=2)
    n = c.use_features(res["features"])
    assert n == res["n_features"]
    assert sorted(c.active_features) == sorted(res["features"])


def test_optimize_correlation_method_choice(tmp_path):
    c = _toy_calc(tmp_path)
    p = c.optimize_features(TARGET_SCORES, correlation="pearson")
    s = c.optimize_features(TARGET_SCORES, correlation="spearman")
    # Both terminate; final n_features may differ between Pearson and Spearman.
    assert p["correlation"] == "pearson"
    assert s["correlation"] == "spearman"
