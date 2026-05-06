"""WALS-style linguistic distance — minimal, fast, NumPy-vectorised."""
from __future__ import annotations

import csv
from difflib import get_close_matches
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

DistanceMethod = Literal["ordinal", "onehot"]
_REQ_DATA = ("Language_name", "Parameter_name", "Value")
_REQ_ORDER = ("NAME", "VALUES IN ORDER")
# Tokens treated as missing — matches pandas' default ``na_values`` set.
_NA = frozenset((
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None",
    "n/a", "nan", "null",
))


def _clean(s: object) -> str:
    """Collapse whitespace and tabs."""
    return " ".join(str(s).replace("\t", " ").split())


def _norm(s: object) -> str:
    """Cleaned + comma-stripped (used for features and values)."""
    return _clean(s).replace(",", "")


def _read_csv(
    path: Path,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> list[list[str]]:
    """Return required (+ optional) columns of *path* as raw string rows.

    Optional columns missing from the header are filled with empty strings.
    """
    with path.open(newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        try:
            header = next(rdr)
        except StopIteration as exc:
            raise ValueError(f"{path.name} is empty.") from exc
        idx = {n: i for i, n in enumerate(header)}
        missing = [c for c in required if c not in idx]
        if missing:
            raise ValueError(f"{path.name} is missing required column(s): {sorted(missing)}")
        req_cols = [idx[c] for c in required]
        opt_cols = [idx.get(c, -1) for c in optional]
        out = []
        for row in rdr:
            r = [row[i] if i < len(row) else "" for i in req_cols]
            r.extend(row[i] if 0 <= i < len(row) else "" for i in opt_cols)
            out.append(r)
        return out


class QwalsCalculator:
    """Distance between languages from WALS-style CSVs.

    methods: ``ordinal`` = ``|i1 - i2| / (n - 1)`` over feature value order,
    ``onehot`` = ``0`` if equal else ``1``.
    """

    def __init__(
        self,
        data_path: str | Path,
        order_path: str | Path | None = None,
        *,
        infer_missing_orders: bool = True,
        inferred_order_method: Literal["appearance", "sorted"] = "appearance",
        cache: bool | str | Path = True,
        weights: dict[str, float] | None = None,
    ) -> None:
        if inferred_order_method not in ("appearance", "sorted"):
            raise ValueError("inferred_order_method must be 'appearance' or 'sorted'.")

        data_path = Path(data_path)
        order_path = Path(order_path) if order_path is not None else None

        # ---- cache lookup (best-effort: any failure → rebuild from CSV) ----
        from . import __version__ as _pkg_version
        from . import _cache

        key = _cache.cache_key(
            data_path=data_path,
            order_path=order_path,
            infer_missing_orders=infer_missing_orders,
            inferred_order_method=inferred_order_method,
            package_version=_pkg_version,
        )
        cache_path = _cache.resolve_cache_path(cache, key=key)

        loaded_from_cache = False
        if cache_path is not None and cache_path.exists():
            payload = _cache.load(cache_path, key=key, package_version=_pkg_version)
            if payload is not None:
                self._populate_from_payload(payload)
                loaded_from_cache = True

        if not loaded_from_cache:
            # ---- cache miss: parse CSVs and build matrices ----
            self._build_from_csv(
                data_path,
                order_path,
                infer_missing_orders=infer_missing_orders,
                inferred_order_method=inferred_order_method,
            )

            if cache_path is not None:
                try:
                    _cache.save(
                        cache_path,
                        key=key,
                        package_version=_pkg_version,
                        languages=self.languages,
                        features=self.features,
                        feature_orders=self.feature_orders,
                        alias=self._alias,
                        synth_to_val=self._synth_to_val,
                        ord_matrix=self._ord,
                        val_matrix=self._val,
                        n_ord=self._n_ord,
                    )
                except OSError:
                    # Read-only home, full disk, etc. — caching is best-effort.
                    pass

        # Per-feature weights default to 1.0. Weights are runtime state and
        # *not* part of the cache identity, so they're initialised after the
        # cache/CSV path has populated `self.features`.
        self._weights: np.ndarray = np.ones(len(self.features), dtype=np.float32)
        if weights:
            self.set_weights(weights)

        # Active-feature mask: True == use this feature, False == ignore it.
        # Default is "use everything" (back-compat with v0.7 behaviour).
        # Set via :meth:`use_features` (e.g. ``use_features("dep")`` to apply
        # the paper's task-specific preset) and cleared with
        # :meth:`reset_features`.
        self._feature_mask: np.ndarray = np.ones(len(self.features), dtype=bool)
        self._active_preset: str | None = None

    # -- Population paths --

    def _populate_from_payload(self, p: dict) -> None:
        """Initialise all attributes from a deserialised cache payload."""
        self.languages = p["languages"]
        self.features = p["features"]
        self._lang_idx = {n: i for i, n in enumerate(self.languages)}
        self._feat_idx = {n: i for i, n in enumerate(self.features)}
        self._alias = dict(p["alias"])
        self.feature_orders = {k: list(v) for k, v in p["feature_orders"].items()}
        self._synth_to_val = [list(v) for v in p["synth_to_val"]]
        self._ord = np.ascontiguousarray(p["ord_matrix"], dtype=np.int16)
        self._val = np.ascontiguousarray(p["val_matrix"], dtype=np.int16)
        self._n_ord = np.ascontiguousarray(p["n_ord"], dtype=np.int16)

    def _build_from_csv(
        self,
        data_path: Path,
        order_path: Path | None,
        *,
        infer_missing_orders: bool,
        inferred_order_method: str,
    ) -> None:
        triples: list[tuple[str, str, str]] = []
        code_to_name: dict[str, str] = {}
        for lang, feat, val, code in _read_csv(data_path, _REQ_DATA, ("Language_ID",)):
            if lang in _NA or feat in _NA or val in _NA:
                continue
            cl, cf, cv = _clean(lang), _norm(feat), _norm(val)
            if cl and cf and cv:
                triples.append((cl, cf, cv))
                if code and code not in _NA:
                    code_to_name.setdefault(_clean(code), cl)

        self.languages: list[str] = sorted({t[0] for t in triples})
        self.features: list[str] = sorted({t[1] for t in triples})
        self._lang_idx = {n: i for i, n in enumerate(self.languages)}
        self._feat_idx = {n: i for i, n in enumerate(self.features)}

        # Alias map: lower-cased token -> canonical language name. Sources:
        # (1) language name itself (case-insensitive), (2) WALS Language_ID,
        # (3) embedded ISO 639-1 → name table. Custom entries can be added
        # via :meth:`add_alias`.
        self._alias: dict[str, str] = {n.lower(): n for n in self.languages}
        for code, name in code_to_name.items():
            if name in self._lang_idx:
                self._alias.setdefault(code.lower(), name)
        from ._aliases import ISO_639_1
        for code, target in ISO_639_1.items():
            if target in self._lang_idx:
                self._alias.setdefault(code.lower(), target)

        self.feature_orders: dict[str, list[str]] = {}
        if order_path is not None:
            self.feature_orders.update(self._read_orders(order_path))

        if infer_missing_orders:
            seen_set: dict[str, set[str]] = {}
            seen_list: dict[str, list[str]] = {}
            for _, f, v in triples:
                s = seen_set.setdefault(f, set())
                if v not in s:
                    s.add(v)
                    seen_list.setdefault(f, []).append(v)
            transform = sorted if inferred_order_method == "sorted" else (lambda x: x)
            for f, vs in seen_list.items():
                self.feature_orders.setdefault(f, transform(vs))

        n_l, n_f = len(self.languages), len(self.features)
        # int16 is sufficient: WALS features have ≤ 9 unique values; the
        # absolute upper bound for either matrix is 32 767 distinct values
        # per feature.
        self._ord = np.full((n_l, n_f), -1, dtype=np.int16)
        self._val = np.full((n_l, n_f), -1, dtype=np.int16)
        ord_lookup = [
            {v: i for i, v in enumerate(self.feature_orders.get(self.features[fi], []))}
            for fi in range(n_f)
        ]
        self._n_ord = np.fromiter((len(d) for d in ord_lookup), dtype=np.int16, count=n_f)

        synth: list[dict[str, int]] = [{} for _ in range(n_f)]
        for lang, feat, val in triples:
            li, fi = self._lang_idx[lang], self._feat_idx[feat]
            d = synth[fi]
            sid = d.get(val)
            if sid is None:
                sid = len(d)
                d[val] = sid
            self._val[li, fi] = sid
            oi = ord_lookup[fi].get(val, -1)
            if oi >= 0:
                self._ord[li, fi] = oi

        self._synth_to_val: list[list[str]] = [sorted(d, key=d.get) for d in synth]

    # -- I/O --

    @classmethod
    def from_folder(
        cls,
        folder: str | Path,
        *,
        data_filename: str = "wals-data.csv",
        order_filename: str = "WALS_feature_order.csv",
        **kwargs,
    ) -> "QwalsCalculator":
        folder = Path(folder)
        order = folder / order_filename
        return cls(folder / data_filename, order if order.exists() else None, **kwargs)

    @staticmethod
    def _read_orders(path: Path) -> dict[str, list[str]]:
        orders: dict[str, list[str]] = {}
        for name, vals in _read_csv(path, _REQ_ORDER):
            if name in _NA or vals in _NA:
                continue
            cleaned = [c for c in (_norm(p) for p in vals.split(",")) if c]
            if cleaned:
                orders[_norm(name)] = cleaned
        return orders

    # -- Lookups --

    def suggest_languages(self, name: str, limit: int = 5) -> list[str]:
        return get_close_matches(_clean(name), self.languages, n=limit)

    def suggest_features(self, name: str, limit: int = 5) -> list[str]:
        return get_close_matches(_norm(name), self.features, n=limit)

    def possible_values(self, feature: str) -> list[str]:
        f = _norm(feature)
        order = self.feature_orders.get(f)
        if order is None:
            extra = self.suggest_features(f)
            raise ValueError(
                f"No possible values known for feature '{f}'."
                + (f" Suggestions: {extra}" if extra else "")
            )
        return list(order)

    def add_feature_order(self, feature: str, ordered_values: Iterable[str]) -> None:
        f = _norm(feature)
        cleaned = [c for c in (_norm(v) for v in ordered_values) if c]
        if not cleaned:
            raise ValueError("ordered_values must contain at least one value.")
        self.feature_orders[f] = cleaned

        fi = self._feat_idx.get(f)
        if fi is None:
            return
        ord_lookup = {v: i for i, v in enumerate(cleaned)}
        self._n_ord[fi] = len(cleaned)
        new_col = np.full(self._val.shape[0], -1, dtype=self._ord.dtype)
        col = self._val[:, fi]
        for sid, value in enumerate(self._synth_to_val[fi]):
            oi = ord_lookup.get(value, -1)
            if oi >= 0:
                new_col[col == sid] = oi
        self._ord[:, fi] = new_col

    def resolve_language(self, name: str) -> str:
        """Resolve *name* (exact, alias, ISO 639-1, WALS Language_ID, or
        case-insensitive form) to its canonical WALS Language_name. Raises
        ``ValueError`` with close-match suggestions if nothing matches.
        """
        cleaned = _clean(name)
        if cleaned in self._lang_idx:
            return cleaned
        canonical = self._alias.get(cleaned.lower())
        if canonical is not None:
            return canonical
        extra = self.suggest_languages(cleaned)
        raise ValueError(
            f"Language '{cleaned}' not found."
            + (f" Suggestions: {extra}" if extra else "")
        )

    def add_alias(self, alias: str, language: str) -> None:
        """Make *alias* resolve to *language* (which must be loaded)."""
        target = self.resolve_language(language)
        key = _clean(alias).lower()
        if not key:
            raise ValueError("alias must be a non-empty string.")
        self._alias[key] = target

    def aliases_for(self, language: str) -> list[str]:
        """All registered aliases that resolve to *language* (sorted)."""
        target = self.resolve_language(language)
        return sorted(k for k, v in self._alias.items() if v == target)

    def _lang_id(self, name: str) -> int:
        return self._lang_idx[self.resolve_language(name)]

    # -- Per-feature weights --

    @property
    def weights(self) -> dict[str, float]:
        """Current non-default weights as a ``{feature: weight}`` dict.

        Only features whose weight differs from 1.0 are returned, so the
        common "all features equal" case yields an empty dict. Use the
        underlying ``_weights`` array directly (read-only by convention)
        if you need the dense vector.
        """
        return {
            f: float(self._weights[i])
            for i, f in enumerate(self.features)
            if self._weights[i] != 1.0
        }

    def set_weight(self, feature: str, weight: float) -> None:
        """Set the weight applied to *feature* in distance calculations.

        Weights must be non-negative and finite. A weight of ``0`` excludes
        the feature from the distance entirely (equivalent to dropping it).
        """
        f = _norm(feature)
        fi = self._feat_idx.get(f)
        if fi is None:
            extra = self.suggest_features(f)
            raise ValueError(
                f"Feature '{f}' not found."
                + (f" Suggestions: {extra}" if extra else "")
            )
        w = float(weight)
        if not np.isfinite(w) or w < 0:
            raise ValueError("weight must be a non-negative finite number.")
        self._weights[fi] = w

    def set_weights(self, weights: dict[str, float]) -> None:
        """Update many weights at once. Unspecified features keep their current weight."""
        for f, w in weights.items():
            self.set_weight(f, w)

    def reset_weights(self) -> None:
        """Reset every feature's weight to 1.0 (the equal-weight default)."""
        self._weights[:] = 1.0

    # -- Active-feature presets --

    @property
    def active_features(self) -> list[str]:
        """Names of features currently in the active mask, in canonical order."""
        return [self.features[i] for i in np.flatnonzero(self._feature_mask)]

    @property
    def active_preset(self) -> str | None:
        """Name of the preset last applied via :meth:`use_features`, or ``None``."""
        return self._active_preset

    def use_features(self, features: "str | Iterable[str]") -> int:
        """Restrict distance computations to a specific subset of features.

        ``features`` is either a preset name (one of the keys of
        :data:`qwals.TASK_FEATURES` — currently ``"abusive"``,
        ``"sentiment"``, ``"ner"``, ``"dep"`` from the qWALS paper's
        Appendix A) or an explicit iterable of feature names.

        Names absent from the loaded WALS data are silently dropped. If
        the input is a preset, the preset name is recorded on
        :attr:`active_preset` for downstream introspection.

        Returns the number of features successfully activated. Use
        :meth:`reset_features` to go back to "use every feature".
        """
        from ._presets import TASK_FEATURES

        preset_name: str | None = None
        if isinstance(features, str):
            if features not in TASK_FEATURES:
                raise ValueError(
                    f"Unknown preset '{features}'. "
                    f"Valid presets: {sorted(TASK_FEATURES)}, "
                    "or pass an explicit iterable of feature names."
                )
            preset_name = features
            feature_iter = TASK_FEATURES[features]
        else:
            feature_iter = list(features)

        new_mask = np.zeros(len(self.features), dtype=bool)
        for f in feature_iter:
            fi = self._feat_idx.get(_norm(f))
            if fi is not None:
                new_mask[fi] = True
        n = int(new_mask.sum())
        if n == 0:
            raise ValueError(
                "use_features: none of the requested features are present in the "
                "loaded WALS data — the active mask would be empty."
            )
        self._feature_mask = new_mask
        self._active_preset = preset_name
        return n

    def reset_features(self) -> None:
        """Re-enable every loaded feature (back to default behaviour)."""
        self._feature_mask[:] = True
        self._active_preset = None

    # -- Internal: vectorised one-vs-all distance --

    def _distance_vector(
        self,
        target_idx: int,
        method: DistanceMethod,
        min_shared: int = 0,
    ) -> np.ndarray:
        """Distance from one language (by index) to every language, as a float64 vector.

        Languages with no shared features — or fewer than ``min_shared``
        shared features — get ``np.inf`` so callers can sort/exclude them
        cleanly. The threshold is on the **unweighted** count of features
        both languages have a value for, so changing per-feature weights
        does not silently shift it. The target's own slot is always 0.0.

        Honors both ``self._weights`` and the active feature mask
        (set via :meth:`use_features`).
        """
        # Effective per-feature weight: zero out features outside the
        # active mask. Weights and masking compose multiplicatively.
        W = (self._weights * self._feature_mask).astype(np.float32)
        # Unweighted active mask — the "shared" count below ignores both
        # weights and inactive features so it stays interpretable.
        FM = self._feature_mask
        if method == "onehot":
            V = self._val
            mask = (V >= 0) & FM                                  # (L, F)
            v_t = V[target_idx]
            m_t = mask[target_idx]
            joint = mask & m_t                                    # (L, F) bool
            joint_w = joint.astype(np.float32) * W                # (L, F)
            total = joint_w.sum(axis=1, dtype=np.float64)         # (L,)
            neq = (V != v_t) & joint
            num = (neq.astype(np.float32) * W).sum(axis=1, dtype=np.float64)
        elif method == "ordinal":
            O = self._ord
            mask = (O >= 0) & FM
            o_t = O[target_idx]
            m_t = mask[target_idx]
            joint = mask & m_t
            denom_f = np.maximum(self._n_ord - 1, 1).astype(np.float32)
            # |o_i - o_t| / (n-1), masked, weighted; broadcasts across L.
            diffs = np.abs(O.astype(np.float32) - o_t.astype(np.float32)) / denom_f
            jf = joint.astype(np.float32)
            num = (diffs * jf * W).sum(axis=1, dtype=np.float64)
            total = (jf * W).sum(axis=1, dtype=np.float64)
        else:
            raise ValueError("method must be 'ordinal' or 'onehot'.")

        # Unweighted count of shared features per language — used for the
        # ``min_shared`` filter so it has the same meaning regardless of
        # what weights the caller has set or which features are active.
        shared = joint.sum(axis=1)                                # (L,) int

        out = np.full(total.shape, np.inf, dtype=np.float64)
        ok = (total > 0) & (shared >= max(int(min_shared), 0))
        np.divide(num, total, out=out, where=ok)
        # Self-comparison should be 0.0 (matches `distance(L, L)`),
        # regardless of `min_shared` — a language is always "near itself".
        out[target_idx] = 0.0
        return out

    # -- Distance --

    def feature_distance(
        self,
        feature: str,
        value1: str,
        value2: str,
        *,
        method: DistanceMethod = "ordinal",
    ) -> float:
        v1, v2 = _norm(value1), _norm(value2)
        if method == "onehot":
            return 0.0 if v1 == v2 else 1.0
        if method != "ordinal":
            raise ValueError("method must be 'ordinal' or 'onehot'.")
        f = _norm(feature)
        order = self.feature_orders.get(f)
        if order is None:
            raise ValueError(f"No ordered values known for feature '{f}'.")
        if v1 not in order or v2 not in order:
            raise ValueError(
                f"Value not found in ordered values for feature '{f}'. "
                f"Values were '{v1}' and '{v2}'. Ordered values are {order}."
            )
        n = len(order)
        return 0.0 if n <= 1 else abs(order.index(v1) - order.index(v2)) / (n - 1)

    def distance(
        self,
        language1: str,
        language2: str,
        *,
        method: DistanceMethod = "ordinal",
        return_details: bool = False,
        return_coverage: bool = False,
    ):
        """Distance between two languages.

        ``return_coverage=True`` switches the return value to a dict
        ``{"distance", "n_shared", "coverage", "n_total_features"}``,
        which is useful for downstream analyses that need to weigh a
        distance by how many features it was actually computed from.
        Mutually compatible with ``return_details=True`` (the dict gains
        a ``"details"`` DataFrame too).
        """
        if method not in ("ordinal", "onehot"):
            raise ValueError("method must be 'ordinal' or 'onehot'.")

        l1, l2 = self.resolve_language(language1), self.resolve_language(language2)
        if l1 == l2:
            n_active = int(self._feature_mask.sum())
            if return_coverage or return_details:
                out: dict = {
                    "language_1": l1, "language_2": l2, "method": method,
                    "distance": 0.0, "n_shared": n_active,
                    "coverage": 1.0 if n_active else 0.0,
                    "n_total_features": int(self._feature_mask.sum()),
                }
                if return_details:
                    import pandas as pd
                    out["features_used"] = 0
                    out["details"] = pd.DataFrame()
                return out
            return 0.0

        i1, i2 = self._lang_idx[l1], self._lang_idx[l2]
        FM = self._feature_mask
        # Effective weight = base weight × active mask.
        Weff = self._weights * FM

        if method == "onehot":
            v1, v2 = self._val[i1], self._val[i2]
            mask = (v1 >= 0) & (v2 >= 0) & FM
            if not mask.any():
                raise ValueError(f"No shared features between '{l1}' and '{l2}'.")
            mw = mask.astype(np.float32) * Weff
            total_w = float(mw.sum())
            if total_w == 0.0:
                raise ValueError(
                    f"All shared features between '{l1}' and '{l2}' have weight zero."
                )
            dist = float(((v1 != v2).astype(np.float32) * mw).sum() / total_w)
        else:
            o1, o2 = self._ord[i1], self._ord[i2]
            mask = (o1 >= 0) & (o2 >= 0) & FM
            if not mask.any():
                raise ValueError(f"No shared (ordered) features between '{l1}' and '{l2}'.")
            denom = np.maximum(self._n_ord - 1, 1)
            mw = mask.astype(np.float32) * Weff
            total_w = float(mw.sum())
            if total_w == 0.0:
                raise ValueError(
                    f"All shared ordered features between '{l1}' and '{l2}' have weight zero."
                )
            diffs = (np.abs(o1 - o2).astype(np.float32) / denom.astype(np.float32))
            dist = float((diffs * mw).sum() / total_w)

        if not return_details and not return_coverage:
            return dist

        n_shared = int(mask.sum())
        n_active = int(FM.sum())
        coverage = n_shared / n_active if n_active else 0.0
        if return_coverage and not return_details:
            return {
                "language_1": l1, "language_2": l2, "method": method,
                "distance": dist, "n_shared": n_shared, "coverage": coverage,
                "n_total_features": n_active,
            }

        import pandas as pd
        weights_in_use = bool((self._weights != 1.0).any())
        rows = []
        for fi in np.flatnonzero(mask):
            f = self.features[fi]
            v1s = self._synth_to_val[fi][int(self._val[i1, fi])]
            v2s = self._synth_to_val[fi][int(self._val[i2, fi])]
            if method == "onehot":
                d = 0.0 if v1s == v2s else 1.0
            else:
                denom_i = max(int(self._n_ord[fi]) - 1, 1)
                d = abs(int(self._ord[i1, fi]) - int(self._ord[i2, fi])) / denom_i
            row = {"Feature": f, l1: v1s, l2: v2s, "Method": method, "Distance": d}
            if weights_in_use:
                row["Weight"] = float(self._weights[fi])
            if method == "ordinal":
                order = self.feature_orders.get(f, [])
                row.update({
                    "Values in order": ",".join(order),
                    f"{l1}_index": order.index(v1s) if v1s in order else None,
                    f"{l2}_index": order.index(v2s) if v2s in order else None,
                    "Count": len(order),
                })
            rows.append(row)
        result = {"language_1": l1, "language_2": l2, "method": method,
                  "distance": dist, "features_used": len(rows),
                  "details": pd.DataFrame(rows)}
        if return_coverage:
            result["n_shared"] = n_shared
            result["coverage"] = coverage
            result["n_total_features"] = n_active
        return result

    def similarity(self, language1: str, language2: str, *,
                   method: DistanceMethod = "ordinal") -> float:
        return 1.0 - self.distance(language1, language2, method=method)

    # -- One-vs-many --

    def distance_to_many(
        self,
        language: str,
        others: Iterable[str] | None = None,
        *,
        method: DistanceMethod = "ordinal",
        as_series: bool = False,
        include_self: bool = False,
        min_shared: int = 0,
    ):
        """Distance from *language* to each of *others* (defaults to every loaded language).

        Returns a ``dict[str, float]`` keyed on canonical names. Pass
        ``as_series=True`` to get a ``pandas.Series`` instead — handy for
        plotting or sorting in a notebook. Languages with no shared
        (weighted) features get ``inf``. The target itself is omitted by
        default; pass ``include_self=True`` to keep its 0.0 entry.

        ``min_shared`` (default 0 — no filter) excludes languages that
        share fewer than that many *features* with the target. WALS is
        unevenly populated; setting e.g. ``min_shared=20`` drops the
        sparsely-attested languages whose few documented features happen
        to coincide with the target's.

        Honors per-feature weights set via :meth:`set_weight` /
        :meth:`set_weights`.
        """
        target = self.resolve_language(language)
        ti = self._lang_idx[target]
        d = self._distance_vector(ti, method, min_shared=min_shared)

        if others is None:
            it = (
                (self.languages[i], float(d[i]))
                for i in range(len(self.languages))
                if (include_self or i != ti) and np.isfinite(d[i])
            )
        else:
            it_pairs = []
            for o in others:
                n = self.resolve_language(o)
                idx = self._lang_idx[n]
                if not include_self and idx == ti:
                    continue
                if not np.isfinite(d[idx]):
                    continue        # below min_shared or no overlap
                it_pairs.append((n, float(d[idx])))
            it = iter(it_pairs)

        result = dict(it)
        if as_series:
            import pandas as pd
            return pd.Series(result, name=f"distance_from_{target}")
        return result

    # Default minimum shared-feature count for ``nearest``. Tuned on the
    # bundled WALS data (2,673 langs × 192 features): below ~50, results
    # are dominated by sparsely-attested languages whose handful of
    # documented features happens to match the target. Power users who
    # want a more permissive search pass a smaller value (or 0).
    NEAREST_MIN_SHARED: int = 50

    def nearest(
        self,
        language: str,
        n: int = 10,
        *,
        method: DistanceMethod = "ordinal",
        include_self: bool = False,
        min_shared: int | None = None,
        with_coverage: bool = False,
        sort_by: str = "distance",
        n_bootstrap: int = 0,
        ci: float = 0.95,
        rng: "int | np.random.Generator | None" = None,
    ):
        """Return the *n* languages closest to *language*, sorted ascending.

        Uses the same weighted distance as :meth:`distance`, computed as a
        single vectorised pass.

        ``min_shared`` filters out languages that share fewer than that
        many features with the target. The default — ``None``, which
        means :attr:`NEAREST_MIN_SHARED` (currently 50) — keeps results
        meaningful on real WALS data, where many languages have only 1–3
        documented features. When a feature preset is active (e.g. after
        ``use_features("dep")``), the default is automatically scaled
        down to ``round(NEAREST_MIN_SHARED · n_active / n_total)`` so it
        stays sensible (e.g. ~20 for the 75-feature DEP preset). Pass
        ``min_shared=0`` to disable the filter and recover the pre-0.7
        behaviour, or any other integer to tune the threshold yourself.

        ``with_coverage`` returns 3-tuples ``(name, distance, n_shared)``
        instead of 2-tuples, so callers can immediately see how reliable
        each neighbour is.

        ``sort_by="upper_ci"`` sorts by the *upper* end of a bootstrap
        confidence interval rather than the point estimate. Combined
        with ``n_bootstrap > 0`` this surfaces languages whose nominal
        closeness survives sparsity-driven uncertainty — useful when
        comparing well- and poorly-attested neighbours head-to-head. The
        result entries become 4-tuples
        ``(name, distance, n_shared, ci_high)``.

        If fewer than *n* viable neighbours exist after filtering, the
        result is shorter than *n*.
        """
        if n <= 0:
            return []
        if sort_by not in ("distance", "upper_ci"):
            raise ValueError("sort_by must be 'distance' or 'upper_ci'.")
        if sort_by == "upper_ci" and n_bootstrap <= 0:
            raise ValueError("sort_by='upper_ci' requires n_bootstrap > 0.")
        if min_shared is None:
            n_active = int(self._feature_mask.sum())
            n_total = len(self._feature_mask)
            min_shared = (
                self.NEAREST_MIN_SHARED
                if n_active == n_total
                else max(1, round(self.NEAREST_MIN_SHARED * n_active / n_total))
            )
        target = self.resolve_language(language)
        ti = self._lang_idx[target]
        d = self._distance_vector(ti, method, min_shared=min_shared)

        # Per-language shared-count vector — needed for both
        # `with_coverage` and the upper-CI sort.
        FM = self._feature_mask
        m_t = (self._val[ti] >= 0) & FM
        shared = ((self._val >= 0) & FM & m_t).sum(axis=1)

        if not include_self:
            d = d.copy()           # don't mutate the freshly-allocated vector
            d[ti] = np.inf

        # Stable argsort by point distance keeps ties resolved by language
        # index (== alphabetical), matching distance_to_many ordering.
        primary_order = np.argsort(d, kind="stable")
        # Take the first *bigger-than-n* slice of viable candidates. We
        # may need extras for upper-CI re-sorting (it can rearrange).
        candidates: list[int] = []
        for i in primary_order:
            if not np.isfinite(d[i]):
                break
            candidates.append(int(i))
            # When re-sorting by upper_ci, fetch a wider candidate pool
            # (4 × n) so the CI sort has room to reshuffle meaningfully.
            cap = max(n * 4, n) if sort_by == "upper_ci" else n
            if len(candidates) >= cap:
                break

        if not candidates:
            return []

        if sort_by == "upper_ci":
            # Compute bootstrap CI for each candidate, then re-sort.
            ci_highs: list[float] = []
            for ci_idx in candidates:
                ci_res = self.distance_ci(
                    target, self.languages[ci_idx],
                    method=method, n_bootstrap=n_bootstrap, ci=ci, rng=rng,
                )
                ci_highs.append(ci_res["ci_high"])
            order = sorted(range(len(candidates)), key=lambda k: (ci_highs[k], candidates[k]))
            chosen = [(candidates[k], ci_highs[k]) for k in order[:n]]
            return [
                (self.languages[i], float(d[i]), int(shared[i]), float(hi))
                for i, hi in chosen
            ]

        # Default: sort by point distance only.
        chosen = candidates[:n]
        if with_coverage:
            return [(self.languages[i], float(d[i]), int(shared[i])) for i in chosen]
        return [(self.languages[i], float(d[i])) for i in chosen]

    def plot_heatmap(
        self,
        languages: "Iterable[str] | None" = None,
        *,
        method: DistanceMethod = "ordinal",
        ax=None,
        cmap: str = "viridis_r",
        annotate: bool = True,
        annotate_fmt: str = "{:.2f}",
        title: str | None = None,
    ):
        """Heatmap of the pairwise qWALS distance matrix (matplotlib).

        Thin wrapper around :func:`qwals._viz.plot_heatmap`. Requires
        the ``[viz]`` extra (matplotlib). Returns the matplotlib Axes
        for further composition. See the underlying function for full
        kwargs documentation.
        """
        from ._viz import plot_heatmap as _plot_heatmap
        return _plot_heatmap(
            self, languages, method=method, ax=ax,
            cmap=cmap, annotate=annotate, annotate_fmt=annotate_fmt, title=title,
        )

    def plot_dendrogram(
        self,
        languages: "Sequence[str] | None" = None,
        *,
        method: DistanceMethod = "ordinal",
        linkage_method: str = "average",
        ax=None,
        color_threshold: float | None = None,
        title: str | None = None,
    ):
        """Hierarchical-clustering dendrogram from qWALS distance.

        Thin wrapper around :func:`qwals._viz.plot_dendrogram`. Requires
        the ``[viz]`` extra (matplotlib + scipy). Returns the matplotlib
        Axes. See the underlying function for full kwargs documentation.
        """
        from ._viz import plot_dendrogram as _plot_dendrogram
        return _plot_dendrogram(
            self, languages, method=method,
            linkage_method=linkage_method, ax=ax,
            color_threshold=color_threshold, title=title,
        )

    def suggest_transfer_source(
        self,
        target: str,
        *,
        task: str | None = None,
        candidates: "Iterable[str] | None" = None,
        n: int = 5,
        method: DistanceMethod = "ordinal",
        min_shared: int | None = None,
    ) -> list[dict]:
        """Rank source-language candidates for cross-lingual transfer to *target*.

        Wraps the paper's recipe: when ``task`` is given, apply that
        task-specific feature preset (one of ``"abusive"``,
        ``"sentiment"``, ``"ner"``, ``"dep"``); compute one-vs-many
        weighted distance under :meth:`nearest`; return ranked candidates
        with both the distance and the shared-feature count so the user
        can see how reliable each suggestion is.

        Parameters
        ----------
        target :
            Target language (any alias form accepted).
        task :
            Optional task-specific feature preset to apply for the
            ranking. The active feature mask is restored on exit, so the
            calculator's user-visible state is unchanged.
        candidates :
            Optional iterable of candidate source languages to consider.
            ``None`` (the default) considers every loaded language.
        n :
            Maximum number of candidates to return.
        method :
            Distance method, ``"ordinal"`` or ``"onehot"``.
        min_shared :
            Threshold passed through to :meth:`nearest`. ``None`` uses
            the auto-scaled :attr:`NEAREST_MIN_SHARED` default — see
            :meth:`nearest` for details.

        Returns
        -------
        list of dict
            Sorted by distance ascending. Each entry has keys
            ``language``, ``distance``, ``n_shared``, ``coverage``,
            and ``confidence`` — a ``[0, 1]`` score combining distance
            (closer is better) and coverage (more shared features →
            more reliable). The composite is intentionally simple
            (``(1 - distance) * sqrt(coverage)``); the paper showed that
            naïve multiplicative reliability penalties can mislead, so
            we surface the raw components alongside it for inspection.
        """
        prev_mask = self._feature_mask.copy()
        prev_preset = self._active_preset
        try:
            if task is not None:
                self.use_features(task)

            target_canon = self.resolve_language(target)
            if candidates is None:
                neigh = self.nearest(
                    target_canon, n=n, method=method,
                    min_shared=min_shared, with_coverage=True,
                )
            else:
                # Restrict to the candidate pool. Cheaper than a custom
                # path: compute the distance vector once, then look up.
                ti = self._lang_idx[target_canon]
                d = self._distance_vector(ti, method, min_shared=min_shared or 0)
                FM = self._feature_mask
                m_t = (self._val[ti] >= 0) & FM
                shared = ((self._val >= 0) & FM & m_t).sum(axis=1)
                rows = []
                for c in candidates:
                    ci = self._lang_idx[self.resolve_language(c)]
                    if ci == ti:
                        continue
                    if np.isfinite(d[ci]):
                        rows.append((self.languages[ci], float(d[ci]), int(shared[ci])))
                rows.sort(key=lambda r: (r[1], r[0]))
                neigh = rows[:n]

            n_active = int(self._feature_mask.sum())
            results = []
            for name, dist, n_shared in neigh:
                coverage = n_shared / n_active if n_active else 0.0
                # Composite confidence: closer is better, more coverage
                # is better. sqrt(coverage) softens the penalty so
                # well-attested-but-distant pairs aren't crushed by a
                # mid-range coverage value.
                confidence = float(max(0.0, 1.0 - dist) * np.sqrt(coverage))
                results.append({
                    "language": name,
                    "distance": dist,
                    "n_shared": n_shared,
                    "coverage": coverage,
                    "confidence": confidence,
                })
            return results
        finally:
            self._feature_mask = prev_mask
            self._active_preset = prev_preset

    # -- Feature-set introspection --

    def features_for(self, language: str) -> list[str]:
        """List of features that *language* has a (non-missing) value for.

        Useful for filtering before a pairwise call, or for debugging
        coverage gaps. Order matches :attr:`features`.
        """
        target = self.resolve_language(language)
        i = self._lang_idx[target]
        return [self.features[fi] for fi in np.flatnonzero(self._val[i] >= 0)]

    def shared_features(self, language1: str, language2: str) -> list[str]:
        """Features for which *both* languages have a value.

        Returned in the canonical :attr:`features` order, so two calls
        with the same arguments are bit-stable. Honors the active feature
        mask: if a preset is in effect, only mask-active shared features
        are returned.
        """
        l1 = self.resolve_language(language1)
        l2 = self.resolve_language(language2)
        i1, i2 = self._lang_idx[l1], self._lang_idx[l2]
        mask = (self._val[i1] >= 0) & (self._val[i2] >= 0) & self._feature_mask
        return [self.features[fi] for fi in np.flatnonzero(mask)]

    def optimize_features(
        self,
        target_scores,
        *,
        method: DistanceMethod = "ordinal",
        correlation: str = "pearson",
        direction: str = "minimise",
        max_drops: int | None = None,
        min_features: int = 5,
        verbose: bool = False,
    ) -> dict:
        """Greedy leave-one-feature-out optimisation against target scores.

        Thin wrapper around :func:`qwals._optimize.optimize_features` —
        see that module for full docs. The starting feature set is
        whatever is currently active (from :meth:`use_features` or
        :meth:`reset_features`); the calculator's mask is **not**
        mutated. Apply the result with
        ``calc.use_features(result["features"])``.
        """
        from ._optimize import optimize_features as _opt
        return _opt(
            self,
            target_scores,
            method=method,
            correlation=correlation,
            direction=direction,
            max_drops=max_drops,
            min_features=min_features,
            verbose=verbose,
        )

    def distance_ci(
        self,
        language1: str,
        language2: str,
        *,
        method: DistanceMethod = "ordinal",
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        rng: "int | np.random.Generator | None" = None,
    ) -> dict:
        """Bootstrap confidence interval for the distance between two languages.

        Resamples the *shared* feature index with replacement
        ``n_bootstrap`` times and recomputes the weighted distance on
        each resample, giving a non-parametric CI on the point estimate.
        For sparsely-attested pairs the CI will be wide; for richly
        documented pairs it will be tight — directly addressing the
        sparsity-reliability concern raised in the qWALS paper (§4.4).

        Parameters
        ----------
        n_bootstrap : int, default 1000
            Number of bootstrap resamples. ~1k is enough for a 95 % CI;
            bump to 10k for tighter percentiles.
        ci : float, default 0.95
            Two-sided confidence level. ``0.95`` → 2.5/97.5 percentiles.
        rng : int or numpy.random.Generator or None
            Seed or RNG for reproducible CIs (passed to
            :func:`numpy.random.default_rng`). ``None`` uses a fresh RNG.

        Returns
        -------
        dict
            ``{distance, ci_low, ci_high, std, n_shared, n_active,
              n_bootstrap, ci_level, language_1, language_2, method}``.
            ``distance`` is the point estimate (same as :meth:`distance`),
            and ``ci_low/ci_high`` bracket it at the requested level.
        """
        if method not in ("ordinal", "onehot"):
            raise ValueError("method must be 'ordinal' or 'onehot'.")
        if not 0 < ci < 1:
            raise ValueError("ci must be in (0, 1).")
        if n_bootstrap < 1:
            raise ValueError("n_bootstrap must be ≥ 1.")
        rng = np.random.default_rng(rng)

        l1 = self.resolve_language(language1)
        l2 = self.resolve_language(language2)
        i1, i2 = self._lang_idx[l1], self._lang_idx[l2]
        FM = self._feature_mask
        mask = (self._val[i1] >= 0) & (self._val[i2] >= 0) & FM
        n_shared = int(mask.sum())
        if n_shared == 0:
            raise ValueError(f"No shared (active) features between '{l1}' and '{l2}'.")

        idx = np.flatnonzero(mask)             # the universe to resample from
        w_all = self._weights.astype(np.float64)[idx]
        if method == "onehot":
            d_all = (self._val[i1, idx] != self._val[i2, idx]).astype(np.float64)
        else:
            denom = np.maximum(self._n_ord[idx] - 1, 1).astype(np.float64)
            d_all = np.abs(
                self._ord[i1, idx].astype(np.float64) - self._ord[i2, idx].astype(np.float64)
            ) / denom
        wd = w_all * d_all

        # Point estimate (same as `distance(...)`).
        total_w_full = float(w_all.sum())
        point = float(wd.sum() / total_w_full) if total_w_full > 0 else 0.0

        # Bootstrap: each resample picks `n_shared` indices with replacement
        # over `0..n_shared-1`, then computes the weighted mean.
        boot_idx = rng.integers(0, n_shared, size=(n_bootstrap, n_shared))
        wd_b = wd[boot_idx]                    # (n_bootstrap, n_shared)
        w_b = w_all[boot_idx]
        sums_w = w_b.sum(axis=1)
        # If a resample happens to draw all-zero weights (only possible
        # when many features have weight 0), fall back to the point.
        with np.errstate(invalid="ignore", divide="ignore"):
            samples = np.where(sums_w > 0, wd_b.sum(axis=1) / sums_w, point)

        alpha = (1 - ci) / 2
        lo, hi = np.quantile(samples, [alpha, 1 - alpha])
        return {
            "language_1": l1, "language_2": l2, "method": method,
            "distance": point,
            "ci_low": float(lo), "ci_high": float(hi),
            "std": float(samples.std(ddof=1)),
            "n_shared": n_shared,
            "n_active": int(FM.sum()),
            "n_bootstrap": int(n_bootstrap),
            "ci_level": ci,
        }

    def explain_distance(
        self,
        language1: str,
        language2: str,
        *,
        method: DistanceMethod = "ordinal",
        top_k: int | None = 10,
    ):
        """Per-feature breakdown of why two languages get the distance they do.

        Returns a ``pandas.DataFrame`` with one row per shared feature,
        sorted by *contribution* (descending) — i.e. the share of the
        final weighted distance that this feature is responsible for.
        Contributions sum to 1.0 across the table.

        Columns: ``Feature``, ``<lang1>``, ``<lang2>``, ``per_feature_distance``,
        ``Weight``, ``contribution`` (= ``per_feature_distance × weight /
        total_weighted_sum``). For ``method="ordinal"``, additional
        columns ``Values in order``, ``<lang1>_index``, ``<lang2>_index``,
        and ``Count`` describe the ordinal scale.

        Honors per-feature weights and the active feature mask.
        ``top_k`` limits the result to the *k* largest-contribution rows
        (default 10); pass ``None`` for the full table.

        This is the "transparent vs lang2vec" pitch from the qWALS paper
        cashed in: users can inspect exactly which typological features
        drove their distance.
        """
        if method not in ("ordinal", "onehot"):
            raise ValueError("method must be 'ordinal' or 'onehot'.")
        import pandas as pd

        l1 = self.resolve_language(language1)
        l2 = self.resolve_language(language2)
        if l1 == l2:
            return pd.DataFrame()

        i1, i2 = self._lang_idx[l1], self._lang_idx[l2]
        FM = self._feature_mask
        mask = (self._val[i1] >= 0) & (self._val[i2] >= 0) & FM
        if not mask.any():
            raise ValueError(f"No shared (active) features between '{l1}' and '{l2}'.")

        rows = []
        for fi in np.flatnonzero(mask):
            f = self.features[fi]
            v1s = self._synth_to_val[fi][int(self._val[i1, fi])]
            v2s = self._synth_to_val[fi][int(self._val[i2, fi])]
            w = float(self._weights[fi])
            if method == "onehot":
                d = 0.0 if v1s == v2s else 1.0
            else:
                denom_i = max(int(self._n_ord[fi]) - 1, 1)
                d = abs(int(self._ord[i1, fi]) - int(self._ord[i2, fi])) / denom_i
            row = {
                "Feature": f,
                l1: v1s,
                l2: v2s,
                "per_feature_distance": d,
                "Weight": w,
                "_wd": w * d,           # internal: weighted distance
            }
            if method == "ordinal":
                order = self.feature_orders.get(f, [])
                row.update({
                    "Values in order": ",".join(order),
                    f"{l1}_index": order.index(v1s) if v1s in order else None,
                    f"{l2}_index": order.index(v2s) if v2s in order else None,
                    "Count": len(order),
                })
            rows.append(row)

        df = pd.DataFrame(rows)
        total_wd = float(df["_wd"].sum())
        if total_wd == 0.0:
            df["contribution"] = 0.0
        else:
            df["contribution"] = df["_wd"] / total_wd
        df = df.drop(columns="_wd").sort_values("contribution", ascending=False).reset_index(drop=True)
        if top_k is not None and top_k > 0:
            df = df.head(int(top_k))
        return df

    def coverage_for(self, language1: str, language2: str) -> dict:
        """How many features the two languages share, plus per-language totals.

        Returns ``{n_shared, n_active, coverage, lang1_total, lang2_total,
        lang1_active, lang2_active}``. ``coverage`` is ``n_shared / n_active``
        — a proxy for the reliability of any distance computed on this
        pair, as discussed in the qWALS paper (§4.4).

        Honors the active feature mask. Useful as a first-pass diagnostic
        for sparsely-attested language pairs before trusting their
        distance value.
        """
        l1 = self.resolve_language(language1)
        l2 = self.resolve_language(language2)
        i1, i2 = self._lang_idx[l1], self._lang_idx[l2]
        FM = self._feature_mask
        m1 = (self._val[i1] >= 0) & FM
        m2 = (self._val[i2] >= 0) & FM
        joint = m1 & m2
        n_active = int(FM.sum())
        n_shared = int(joint.sum())
        return {
            "n_shared": n_shared,
            "n_active": n_active,
            "coverage": n_shared / n_active if n_active else 0.0,
            "lang1": l1,
            "lang2": l2,
            "lang1_active": int(m1.sum()),
            "lang2_active": int(m2.sum()),
            "lang1_total": int((self._val[i1] >= 0).sum()),
            "lang2_total": int((self._val[i2] >= 0).sum()),
        }

    # -- Pairwise --

    # Block size for the pairwise tile. With F ≈ 192, a (B, B, F) float32
    # buffer at B=64 is ~3 MiB — fits comfortably in L2/L3, amortises
    # NumPy's per-op overhead, and benchmarks ~2.1× faster than the v0.3
    # row loop on the full WALS dataset.
    _PAIRWISE_BLOCK: int = 64

    def pairwise_matrix(
        self,
        languages: list[str] | None = None,
        *,
        method: DistanceMethod = "ordinal",
        block: int | None = None,
    ):
        if method not in ("ordinal", "onehot"):
            raise ValueError("method must be 'ordinal' or 'onehot'.")
        import pandas as pd

        if languages is None:
            ids = np.arange(len(self.languages))
            labels = list(self.languages)
        else:
            labels = [self.resolve_language(l) for l in languages]
            ids = np.asarray([self._lang_idx[n] for n in labels], dtype=np.int64)

        L = len(ids)
        B = block if block is not None else self._PAIRWISE_BLOCK
        out = np.zeros((L, L), dtype=np.float64)
        # Effective per-feature weight folds in the active mask: features
        # outside the mask get weight 0 and drop out of the sums below.
        W = self._weights * self._feature_mask              # float32 (F,)

        # Shared (weighted) "feature mass" matrix:
        # total[i, j] = sum_f w[f] * mask[i,f] * mask[j,f]. One BLAS matmul
        # replaces L row-wise reductions, and folding w[f] into one side
        # makes the weighted version free.
        if method == "onehot":
            V = self._val[ids]                            # (L, F) int16
            mask = V >= 0
            Mf = mask.astype(np.float32)
            MfW = Mf * W                                  # (L, F) weighted
            total = (MfW @ Mf.T).astype(np.float64)       # (L, L)
            denom = np.maximum(total, 1e-12)
            for i0 in range(0, L, B):
                i1 = min(i0 + B, L)
                Vi, Mi = V[i0:i1, None, :], mask[i0:i1, None, :]
                for j0 in range(0, L, B):
                    j1 = min(j0 + B, L)
                    shared = Mi & mask[None, j0:j1, :]
                    diff = (Vi != V[None, j0:j1, :]) & shared
                    s = (diff.astype(np.float32) * W).sum(axis=2, dtype=np.float64)
                    out[i0:i1, j0:j1] = s / denom[i0:i1, j0:j1]
        else:
            O = self._ord[ids]                            # (L, F) int16
            mask = O >= 0
            denom_f = np.maximum(self._n_ord - 1, 1).astype(np.float32)
            # float32 halves bandwidth vs. float64; the final dist is still
            # float64 below since `out` is float64 and accumulates wider.
            Os = (O.astype(np.float32) / denom_f) * mask  # masked-out → 0
            Mf = mask.astype(np.float32)
            MfW = Mf * W
            total = (MfW @ Mf.T).astype(np.float64)
            denom = np.maximum(total, 1e-12)
            for i0 in range(0, L, B):
                i1 = min(i0 + B, L)
                Osi = Os[i0:i1, None, :]                  # (Bi, 1, F)
                Mi = Mf[i0:i1, None, :]
                for j0 in range(0, L, B):
                    j1 = min(j0 + B, L)
                    # (Bi, Bj, F) tile: only allocate at this block.
                    diff = np.abs(Osi - Os[None, j0:j1, :])
                    diff *= Mi
                    diff *= Mf[None, j0:j1, :]
                    diff *= W                              # broadcast (F,)
                    s = diff.sum(axis=2, dtype=np.float64)
                    out[i0:i1, j0:j1] = s / denom[i0:i1, j0:j1]

        # Cells where no shared features exist — or all weights zero — get
        # 0 by virtue of the `denom = max(total, 1e-12)` clamp; the
        # numerator there is 0 too. Diagonal is always 0.0.
        return pd.DataFrame(out, index=labels, columns=labels)

    def save_pairwise_matrix(
        self,
        output_csv: str | Path,
        languages: list[str] | None = None,
        *,
        method: DistanceMethod = "ordinal",
    ):
        m = self.pairwise_matrix(languages, method=method)
        m.to_csv(output_csv)
        return m
