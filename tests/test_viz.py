"""Tests for v0.8 plot_heatmap / plot_dendrogram (matplotlib + scipy)."""
from __future__ import annotations

import pandas as pd
import pytest

from qwals import QwalsCalculator

# Skip the entire module gracefully if matplotlib isn't installed.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")          # headless — no display required


@pytest.fixture
def calc(tmp_path):
    pd.DataFrame(
        [
            {"Language_ID": "pol", "Language_name": "Polish",  "Parameter_name": "F1", "Value": "Low"},
            {"Language_ID": "pol", "Language_name": "Polish",  "Parameter_name": "F2", "Value": "Yes"},
            {"Language_ID": "eng", "Language_name": "English", "Parameter_name": "F1", "Value": "High"},
            {"Language_ID": "eng", "Language_name": "English", "Parameter_name": "F2", "Value": "Yes"},
            {"Language_ID": "ger", "Language_name": "German",  "Parameter_name": "F1", "Value": "Average"},
            {"Language_ID": "ger", "Language_name": "German",  "Parameter_name": "F2", "Value": "No"},
            {"Language_ID": "rus", "Language_name": "Russian", "Parameter_name": "F1", "Value": "Low"},
            {"Language_ID": "rus", "Language_name": "Russian", "Parameter_name": "F2", "Value": "Yes"},
        ]
    ).to_csv(tmp_path / "wals-data.csv", index=False)
    pd.DataFrame(
        [
            {"NAME": "F1", "VALUES IN ORDER": "Low,Average,High"},
            {"NAME": "F2", "VALUES IN ORDER": "No,Yes"},
        ]
    ).to_csv(tmp_path / "WALS_feature_order.csv", index=False)
    return QwalsCalculator(tmp_path / "wals-data.csv", tmp_path / "WALS_feature_order.csv")


def test_plot_heatmap_returns_axes(calc):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    out = calc.plot_heatmap(["pl", "en", "de", "ru"], ax=ax, annotate=False)
    assert out is ax
    # Ticks labelled with canonical names.
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert set(labels) == {"Polish", "English", "German", "Russian"}
    plt.close(fig)


def test_plot_heatmap_creates_axes_when_none(calc):
    ax = calc.plot_heatmap(["pl", "en"], annotate=False)
    assert ax is not None
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_plot_heatmap_alias_input(calc):
    """ISO codes should resolve via the alias machinery."""
    ax = calc.plot_heatmap(["pl", "ru"], annotate=False)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert set(labels) == {"Polish", "Russian"}
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_plot_dendrogram_returns_axes(calc):
    pytest.importorskip("scipy")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    out = calc.plot_dendrogram(["pl", "en", "de", "ru"], ax=ax)
    assert out is ax
    plt.close(fig)


def test_plot_dendrogram_linkage_methods(calc):
    pytest.importorskip("scipy")
    import matplotlib.pyplot as plt
    for method in ("average", "single", "complete", "ward"):
        ax = calc.plot_dendrogram(["pl", "en", "de", "ru"], linkage_method=method)
        assert ax is not None
        plt.close(ax.figure)


def test_viz_import_error_messages(monkeypatch):
    """When matplotlib is missing, the helpful error mentions [viz]."""
    import builtins
    real_import = builtins.__import__

    def _no_mpl(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_mpl)
    from qwals._viz import _require_matplotlib
    with pytest.raises(ImportError, match=r"\[viz\]"):
        _require_matplotlib()
