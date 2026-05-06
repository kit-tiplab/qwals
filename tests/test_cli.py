"""Tests for the ``python -m qwals`` CLI."""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
import pytest

from qwals.__main__ import main


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """A 3-language fixture written into tmp_path; returns the folder."""
    pd.DataFrame(
        [
            {"Language_ID": "pol", "Language_name": "Polish",  "Parameter_name": "F1", "Value": "Low"},
            {"Language_ID": "eng", "Language_name": "English", "Parameter_name": "F1", "Value": "High"},
            {"Language_ID": "ger", "Language_name": "German",  "Parameter_name": "F1", "Value": "Average"},
        ]
    ).to_csv(tmp_path / "wals-data.csv", index=False)
    pd.DataFrame(
        [{"NAME": "F1", "VALUES IN ORDER": "Low,Average,High"}]
    ).to_csv(tmp_path / "WALS_feature_order.csv", index=False)
    return tmp_path


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Run main(argv) capturing stdout / stderr separately."""
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


def test_cli_compare(tiny_dataset):
    rc, out, _ = _run(["--data", str(tiny_dataset), "compare", "pl", "en"])
    assert rc == 0
    # Polish (Low=0) vs English (High=2) over a 3-value scale → 1.0
    assert out.strip() == "Polish\tEnglish\t1.000000"


def test_cli_compare_alias_mix(tiny_dataset):
    rc1, out1, _ = _run(["--data", str(tiny_dataset), "compare", "Polish", "en"])
    rc2, out2, _ = _run(["--data", str(tiny_dataset), "compare", "pol", "English"])
    assert rc1 == rc2 == 0
    assert out1 == out2


def test_cli_compare_method_onehot(tiny_dataset):
    rc, out, _ = _run(["--data", str(tiny_dataset), "compare", "pl", "en", "--method", "onehot"])
    assert rc == 0
    # Different values → onehot distance is 1.0
    assert out.strip().endswith("1.000000")


def test_cli_nearest(tiny_dataset):
    """The 1-feature fixture needs --min-shared 0 to show any results,
    since the default of 50 (tuned for real WALS) excludes everything."""
    rc, out, _ = _run([
        "--data", str(tiny_dataset),
        "nearest", "Polish", "--n", "2", "--min-shared", "0",
    ])
    assert rc == 0
    lines = [l for l in out.strip().split("\n") if not l.startswith("#")]
    # Closest to Polish: German (mid) closer than English (far).
    names = [l.split("\t")[0] for l in lines]
    assert names == ["German", "English"]


def test_cli_nearest_default_warns_when_filter_excludes_all(tiny_dataset):
    """Without --min-shared the default of 50 excludes every fixture
    language; the CLI should print a helpful warning to stderr that
    suggests --min-shared, and exit 0."""
    rc, out, err = _run(["--data", str(tiny_dataset), "nearest", "Polish"])
    assert rc == 0
    assert out == ""                          # nothing useful to print
    assert "min-shared" in err.lower()        # tells the user how to fix it


def test_cli_pairwise_to_stdout(tiny_dataset):
    rc, out, _ = _run(["--data", str(tiny_dataset), "pairwise"])
    assert rc == 0
    # Should be a CSV with the three language names as both rows and columns.
    assert "Polish" in out and "English" in out and "German" in out


def test_cli_pairwise_to_file(tiny_dataset, tmp_path):
    out_csv = tmp_path / "out.csv"
    rc, _, _ = _run(["--data", str(tiny_dataset), "pairwise", "--out", str(out_csv)])
    assert rc == 0
    assert out_csv.exists()
    df = pd.read_csv(out_csv, index_col=0)
    assert list(df.index) == ["English", "German", "Polish"]    # alphabetical


def test_cli_shared(tiny_dataset):
    rc, out, _ = _run(["--data", str(tiny_dataset), "shared", "pl", "en"])
    assert rc == 0
    assert "F1" in out


def test_cli_features(tiny_dataset):
    rc, out, _ = _run(["--data", str(tiny_dataset), "features", "Polish"])
    assert rc == 0
    assert "F1" in out


def test_cli_unknown_language_errors(tiny_dataset):
    rc, _, err = _run(["--data", str(tiny_dataset), "compare", "Klingon", "Polish"])
    assert rc != 0
    assert "Klingon" in err or "not found" in err


def test_cli_no_cache_flag(tiny_dataset, tmp_path, monkeypatch):
    """--no-cache should bypass the cache; the call still succeeds."""
    cache_dir = tmp_path / "iso_cache"
    monkeypatch.setenv("QWALS_CACHE_DIR", str(cache_dir))
    rc, _, _ = _run(["--data", str(tiny_dataset), "--no-cache", "compare", "pl", "en"])
    assert rc == 0
    # No cache file should have been written.
    assert not cache_dir.exists() or not list(cache_dir.glob("*.npz"))
