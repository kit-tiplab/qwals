import pandas as pd
import pytest

from qwals import QwalsCalculator


def test_comma_cleaning_and_methods(tmp_path):
    data = tmp_path / "wals-data.csv"
    order = tmp_path / "WALS_feature_order.csv"

    pd.DataFrame(
        [
            {"Language_name": "A", "Parameter_name": "Feature 1", "Value": "Low"},
            {"Language_name": "B", "Parameter_name": "Feature 1", "Value": "Average"},
            {"Language_name": "A", "Parameter_name": "Feature 2", "Value": "All nouns, always obligatory"},
            {"Language_name": "B", "Parameter_name": "Feature 2", "Value": "No nouns"},
        ]
    ).to_csv(data, index=False)

    pd.DataFrame(
        [
            {"NAME": "Feature 1", "VALUES IN ORDER": "Low,Average,High"},
            {"NAME": "Feature 2", "VALUES IN ORDER": "No nouns,All nouns always obligatory"},
        ]
    ).to_csv(order, index=False)

    calc = QwalsCalculator(data, order)

    assert calc.feature_distance("Feature 1", "Low", "Average", method="ordinal") == 0.5
    assert calc.feature_distance("Feature 1", "Low", "Average", method="onehot") == 1.0
    assert calc.feature_distance(
        "Feature 2",
        "All nouns, always obligatory",
        "No nouns",
        method="ordinal",
    ) == 1.0


def _alias_calc(tmp_path):
    """Tiny synthetic dataset that includes Language_ID for alias resolution."""
    data = tmp_path / "wals-data.csv"
    order = tmp_path / "WALS_feature_order.csv"
    pd.DataFrame(
        [
            {"Language_ID": "pol", "Language_name": "Polish",  "Parameter_name": "F1", "Value": "Low"},
            {"Language_ID": "eng", "Language_name": "English", "Parameter_name": "F1", "Value": "High"},
            {"Language_ID": "ger", "Language_name": "German",  "Parameter_name": "F1", "Value": "Average"},
        ]
    ).to_csv(data, index=False)
    pd.DataFrame(
        [{"NAME": "F1", "VALUES IN ORDER": "Low,Average,High"}]
    ).to_csv(order, index=False)
    return QwalsCalculator(data, order)


def test_alias_resolution(tmp_path):
    calc = _alias_calc(tmp_path)

    # ISO 639-1 (two-letter)
    assert calc.resolve_language("pl") == "Polish"
    assert calc.resolve_language("en") == "English"
    assert calc.resolve_language("de") == "German"

    # WALS Language_ID (three-letter)
    assert calc.resolve_language("pol") == "Polish"
    assert calc.resolve_language("eng") == "English"
    assert calc.resolve_language("ger") == "German"

    # exact name + case-insensitive name
    assert calc.resolve_language("Polish") == "Polish"
    assert calc.resolve_language("polish") == "Polish"
    assert calc.resolve_language("POLISH") == "Polish"

    # whitespace tolerance
    assert calc.resolve_language("  PL  ") == "Polish"


def test_alias_distance_equivalence(tmp_path):
    calc = _alias_calc(tmp_path)
    # Same answer regardless of input form.
    by_name = calc.distance("Polish", "English", method="ordinal")
    by_iso  = calc.distance("pl", "en", method="ordinal")
    by_code = calc.distance("pol", "eng", method="ordinal")
    assert by_name == by_iso == by_code == 1.0  # Low (idx 0) vs High (idx 2) = 2/2

    # Mixing forms also works
    assert calc.distance("pl", "English") == by_name


def test_pairwise_uses_canonical_labels(tmp_path):
    calc = _alias_calc(tmp_path)
    m = calc.pairwise_matrix(["pl", "en", "de"], method="onehot")
    assert list(m.index) == ["Polish", "English", "German"]
    assert list(m.columns) == ["Polish", "English", "German"]
    assert m.loc["Polish", "Polish"] == 0.0
    assert m.loc["Polish", "English"] == 1.0


def test_add_and_query_aliases(tmp_path):
    calc = _alias_calc(tmp_path)
    calc.add_alias("Lehia", "Polish")
    assert calc.resolve_language("Lehia") == "Polish"
    assert calc.resolve_language("LEHIA") == "Polish"
    assert "lehia" in calc.aliases_for("Polish")
    # Built-in aliases should also surface
    polish_aliases = calc.aliases_for("pl")
    assert "pl" in polish_aliases and "pol" in polish_aliases and "polish" in polish_aliases


def test_unknown_alias_raises(tmp_path):
    calc = _alias_calc(tmp_path)
    with pytest.raises(ValueError):
        calc.resolve_language("zz")  # ISO code not in our table
    with pytest.raises(ValueError):
        calc.add_alias("foo", "Klingon")  # alias target must exist in WALS
