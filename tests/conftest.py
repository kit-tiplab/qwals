"""Shared fixtures.

Keep the on-disk matrix cache out of the user's ``~/.cache`` during the
test run by redirecting it to a per-test temp dir via the env var the
``_cache`` module honours.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("QWALS_CACHE_DIR", str(tmp_path / "_qwals_cache"))
