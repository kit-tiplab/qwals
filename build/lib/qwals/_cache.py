"""Disk cache for the precomputed numpy matrices.

The cache turns repeated ``QwalsCalculator(...)`` constructions on the
same CSVs from a ~190 ms parse into a ~5–10 ms ``np.load``.

Layout
------
A single ``.npz`` per (data, order, options, package-version) combination,
written via :func:`numpy.savez_compressed` and read with ``allow_pickle=False``.
Strings and JSON-encoded mappings travel as 0-d unicode arrays so we never
need pickle, which keeps the cache files trustable across machines and
Python versions.

The default location is ``~/.cache/qwals/`` (overridable with the
``QWALS_CACHE_DIR`` env var). A custom path passed to the constructor
wins over both.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_CACHE_VERSION = "1"  # bump if the on-disk schema changes
_ENV_VAR = "QWALS_CACHE_DIR"


def default_cache_dir() -> Path:
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "qwals"


def _file_fingerprint(path: Path | None) -> str:
    """A cheap, robust identity for a CSV file: resolved path + size + mtime_ns.

    Avoids hashing the whole 9 MB CSV on every init while still being
    sensitive enough to catch in-place edits.
    """
    if path is None:
        return "none"
    st = path.stat()
    return f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"


def cache_key(
    *,
    data_path: Path,
    order_path: Path | None,
    infer_missing_orders: bool,
    inferred_order_method: str,
    package_version: str,
) -> str:
    parts = [
        f"v={_CACHE_VERSION}",
        f"pkg={package_version}",
        f"data={_file_fingerprint(data_path)}",
        f"order={_file_fingerprint(order_path)}",
        f"infer={int(bool(infer_missing_orders))}",
        f"method={inferred_order_method}",
    ]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:32]


def _atomic_savez(path: Path, **arrays: Any) -> None:
    """Write ``path`` atomically — never leave a half-written cache around.

    ``mkstemp`` creates an empty placeholder; we close the fd and overwrite
    that path via ``np.savez_compressed``. The ``.npz`` suffix tells
    NumPy not to append a second one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp.npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save(
    path: Path,
    *,
    key: str,
    package_version: str,
    languages: list[str],
    features: list[str],
    feature_orders: dict[str, list[str]],
    alias: dict[str, str],
    synth_to_val: list[list[str]],
    ord_matrix: np.ndarray,
    val_matrix: np.ndarray,
    n_ord: np.ndarray,
) -> None:
    _atomic_savez(
        path,
        cache_version=np.array(_CACHE_VERSION),
        package_version=np.array(package_version),
        key=np.array(key),
        languages=np.array(languages),
        features=np.array(features),
        feature_orders=np.array(json.dumps(feature_orders)),
        alias=np.array(json.dumps(alias)),
        synth_to_val=np.array(json.dumps(synth_to_val)),
        ord_matrix=ord_matrix,
        val_matrix=val_matrix,
        n_ord=n_ord,
    )


def load(path: Path, *, key: str, package_version: str) -> dict[str, Any] | None:
    """Return the cache contents if valid, else ``None``.

    A mismatched ``key`` (means the source CSV changed under us), a
    different ``package_version`` (schema or behaviour might differ), or
    any I/O / decode error all map to "cache miss" — the caller should
    fall back to building from CSV and write a fresh cache.
    """
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["cache_version"]) != _CACHE_VERSION:
                return None
            if str(z["package_version"]) != package_version:
                return None
            if str(z["key"]) != key:
                return None
            return {
                "languages": [str(s) for s in z["languages"]],
                "features": [str(s) for s in z["features"]],
                "feature_orders": json.loads(str(z["feature_orders"])),
                "alias": json.loads(str(z["alias"])),
                "synth_to_val": json.loads(str(z["synth_to_val"])),
                "ord_matrix": z["ord_matrix"],
                "val_matrix": z["val_matrix"],
                "n_ord": z["n_ord"],
            }
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def resolve_cache_path(
    cache: bool | str | os.PathLike,
    *,
    key: str,
) -> Path | None:
    """Translate the user-facing ``cache`` argument into an actual file path.

    ``True``  → ``<default_cache_dir()>/<key>.npz``
    ``False`` → ``None`` (caching disabled)
    ``str | Path`` → that exact file (must end in ``.npz`` by convention)
    """
    if cache is False:
        return None
    if cache is True:
        return default_cache_dir() / f"{key}.npz"
    return Path(os.fspath(cache)).expanduser()
