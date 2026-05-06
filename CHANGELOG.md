# Changelog

All notable changes to the `qwals` package are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [0.8.1] — 2026-05-06

### Changed README.md
- Added `Citation` section with information on how to cite this software in research reports.

## [0.8.0] — 2026-05-02

This release operationalises the methodology and headline results of
Eronen et al. (2026), *"Language Models Are Polyglots: Language
Similarity Predicts Cross-Lingual Transfer Learning Performance"*
(Mach. Learn. Knowl. Extr. 8(3):65). Previous releases shipped the
core qWALS metric; 0.8.0 ships the **paper's task-specific feature
sets, optimisation procedure, explainability outputs, sparsity-aware
uncertainty handling, transfer-source ranking, and visualisations.**

### Added — task-specific feature presets (paper Appendix A)
- `qwals.TASK_FEATURES` and `qwals.TASKS`: the four LOFO-optimised
  feature subsets from the paper — `dep` (75), `ner` (63), `abusive`
  (53), `sentiment` (21).
- `QwalsCalculator.use_features(name_or_list)`: restrict the
  calculator to a preset (or an explicit feature list). Returns the
  number of features actually activated.
- `QwalsCalculator.reset_features()`: re-enable all loaded features.
- `QwalsCalculator.active_features` / `.active_preset`: introspection.

### Added — leave-one-feature-out (LOFO) optimisation
- New module `qwals._optimize`. `QwalsCalculator.optimize_features(
  target_scores, ...)` replicates the paper's §4.5 procedure:
  greedily drops one feature at a time until correlation with a target
  signal stops improving. Returns `{features, pearson, spearman,
  n_features, n_dropped, history}`. `direction="minimise" / "maximise"`,
  `correlation="pearson" / "spearman"`, `max_drops`, `min_features`,
  and `verbose=True` knobs available.
- Pure NumPy — no SciPy required for the optimiser itself.

### Added — explainability and coverage
- `QwalsCalculator.explain_distance(L1, L2, top_k=10)`: returns a
  `pandas.DataFrame` of per-feature contributions (sorted by share of
  the final weighted distance), with the values each language has on
  every shared feature. Cashes in on the paper's
  "transparent vs. monolithic lang2vec" pitch.
- `distance(..., return_coverage=True)`: dict result with `n_shared`,
  `coverage`, and `n_total_features` alongside `distance`.
  Compatible with `return_details=True`.
- `coverage_for(L1, L2)`: dedicated coverage helper —
  `{n_shared, n_active, coverage, lang1_total, lang2_total, ...}`.

### Added — bootstrap confidence intervals
- `QwalsCalculator.distance_ci(L1, L2, n_bootstrap=1000, ci=0.95,
  rng=None)`: resamples the shared feature index with replacement and
  returns `{distance, ci_low, ci_high, std, n_shared, n_active, ...}`.
  Directly addresses the sparsity-reliability concern from §4.4 of the
  paper.
- `nearest(..., sort_by="upper_ci", n_bootstrap=N)`: re-rank candidates
  by the *upper* end of a bootstrap CI rather than the point estimate,
  surfacing neighbours whose closeness is robust to feature sparsity.
  Result entries become 4-tuples `(name, distance, n_shared, ci_high)`.
- `nearest(..., with_coverage=True)`: 3-tuples
  `(name, distance, n_shared)`.

### Added — transfer-source ranking
- `QwalsCalculator.suggest_transfer_source(target, task=None,
  candidates=None, n=5, ...)`: ranks candidate source languages with
  the optional task preset applied for the duration of the call (the
  active mask is restored on exit). Each entry exposes `language`,
  `distance`, `n_shared`, `coverage`, and a composite `confidence`
  score `(1 − distance) · √coverage`. Designed for the paper's
  headline use case ("which source should I pick for cross-lingual
  transfer to X?").

### Added — visualisation (`[viz]` extra)
- New module `qwals._viz` with `plot_heatmap` and `plot_dendrogram`.
  Both also exposed as `QwalsCalculator.plot_heatmap` /
  `.plot_dendrogram`. Lazy imports — useful error message if matplotlib
  / scipy isn't installed.
- New `[viz]` optional dependency: `matplotlib>=3.5`, `scipy>=1.8`.
  `[dev]` now includes them too.
- `plot_heatmap` reproduces the per-task panels from Figure 2 of the
  paper. `plot_dendrogram` (UPGMA / Ward / single / complete linkage)
  recovers the paper's family clusters as a visual sanity check.

### Changed
- The active feature mask composes multiplicatively with `_weights` in
  every distance computation (`distance`, `pairwise_matrix`,
  `_distance_vector`, `_distance_vector_for_pair`), so applying a
  preset is equivalent to setting weights to 0 outside the mask but
  cheaper.
- `nearest`'s default `min_shared` now auto-scales to the active
  feature count: when a preset is in effect, the threshold becomes
  `round(NEAREST_MIN_SHARED · n_active / n_total)` — e.g. ~20 for the
  75-feature DEP preset, instead of always 50. Pass an explicit
  `min_shared=` to override.
- `shared_features` now honors the active feature mask (returns the
  intersection within the active set, not over all 192 features).
- Bumped version `0.7.0 → 0.8.0`.
- 57 new tests across `test_presets.py` (10), `test_explain.py` (15),
  `test_ci.py` (11), `test_optimize.py` (9), `test_suggest.py` (7),
  and `test_viz.py` (6). **114 tests pass total** (was 57 in 0.7.0).

### Performance
Unchanged from 0.7.0 for the existing operations. New operations on
the bundled WALS data (2,673 langs × 192 features):

| Operation                                     | Time   |
| --------------------------------------------- | -----: |
| `use_features("dep")`                         | ~0.1 ms |
| `explain_distance` (top 10)                   | ~3 ms  |
| `distance_ci` (n_bootstrap=2000)              | ~6 ms  |
| `nearest(sort_by="upper_ci", n_bootstrap=200)`| ~80 ms |
| `optimize_features` (8 langs, ~30 drops)      | ~250 ms |
| `plot_heatmap` (8 langs)                      | ~120 ms |

## [0.7.0] — 2026-05-02

### Fixed (behaviour change in default)
**`nearest()` no longer surfaces meaningless 1-feature matches by default.**

WALS is unevenly populated — many languages have only 1–4 features
documented. Before 0.7.0, `nearest("Polish")` returned an alphabetical
list of obscure languages with `distance=0.0` because their handful of
documented features happened to coincide with Polish's. This made the
method effectively unusable as documented.

```python
# Before 0.7.0:
calc.nearest("Polish", n=5)
# [('Abenaki (Western)', 0.0), ('Aguaruna', 0.0), ('Algonquin', 0.0), ...]

# 0.7.0 default:
calc.nearest("Polish", n=5)
# [('Lithuanian',     0.131),
#  ('Russian',        0.138),
#  ('Latvian',        0.146),
#  ('Greek (Modern)', 0.163),
#  ('Albanian',       0.173)]
```

### Added
- New `min_shared` keyword on **`nearest()`** (default `None`, which
  means `QwalsCalculator.NEAREST_MIN_SHARED` — currently `50`). Languages
  sharing fewer than that many features with the target are excluded
  from consideration. Pass `min_shared=0` to recover the pre-0.7
  behaviour. Pass any other integer to tune the threshold for your
  dataset (e.g. `20` for permissive results, `80` for strict).
- New `min_shared` keyword on **`distance_to_many()`** (default `0` —
  no filter, preserves the existing dict-of-everything behaviour).
  Filtered entries are dropped from the result dict rather than reported
  as `inf`.
- New CLI flag **`--min-shared N`** on `qwals nearest`. The CLI also
  prints a clearer warning to stderr when the filter excludes
  everything, suggesting the user pass a smaller value.
- New class attribute **`QwalsCalculator.NEAREST_MIN_SHARED`** so users
  can monkey-patch or subclass to change the default for their
  application without passing the kwarg every call.
- Seven new tests covering: the new default returning empty on the
  3-feature fixture (correctly), `min_shared` thresholds at exact
  feature counts, weight-independence of the filter, the class-level
  default constant, and the new CLI flag plus its empty-result warning.

### Changed
- `_distance_vector` now also computes the unweighted shared-feature
  count per language (`joint.sum(axis=1)`) and uses it to apply the
  filter. The weighted `total` is still used for the distance itself,
  so per-feature weights only affect the *value* of the distance, not
  whether a language passes the threshold.
- The four pre-existing tests that used the 3-feature fixture
  (`test_nearest_basic`, `test_nearest_includes_self_when_asked`,
  `test_nearest_n_clamps_to_available`, `test_nearest_method_onehot`)
  now pass `min_shared=0` explicitly, since the new real-WALS-tuned
  default would otherwise correctly report "no defensible neighbours".
- Bumped version `0.6.0 → 0.7.0`.

### Migration
Code that called `nearest()` and depended on the old "any low-overlap
match counts" behaviour (e.g. for testing on small synthetic datasets)
must now pass `min_shared=0` explicitly:

```python
# old
calc.nearest("Polish", n=5)
# new equivalent
calc.nearest("Polish", n=5, min_shared=0)
```

Code that used the method on real WALS data and was confused by the
output should simply update to 0.7.0 — the default now does the right
thing.

### Performance
On the bundled WALS data (2,673 langs × 192 features), `nearest`
remains ~1.5 ms per call. The added unweighted-count reduction is one
extra `axis=1` sum on a `(L, F)` boolean matrix — negligible.

## [0.6.0] — 2026-05-02

### Renamed (no behaviour changes)
This release is the **final pre-ship rename** from `wals-distance` to
`qwals`. The algorithms, public method shapes, return values, file formats,
performance characteristics, and the on-disk cache schema are all
unchanged. Every test from 0.5.0 still passes against the renamed package.

| Old | New |
| --- | --- |
| `pip install wals-distance` | `pip install qwals` |
| `from wals_distance import WalsDistanceCalculator` | `from qwals import QwalsCalculator` |
| `WalsDistanceCalculator(...)` | `QwalsCalculator(...)` |
| `python -m wals_distance ...` | `python -m qwals ...` |
| console script `wals-distance` | console script `qwals` |
| env var `WALS_DISTANCE_CACHE_DIR` | env var `QWALS_CACHE_DIR` |
| cache dir `~/.cache/wals_distance/` | cache dir `~/.cache/qwals/` |
| `src/wals_distance/` | `src/qwals/` |

### Migration

```python
# before (0.5.x)
from wals_distance import WalsDistanceCalculator
calc = WalsDistanceCalculator.from_folder("data/")

# after (0.6.x)
from qwals import QwalsCalculator
calc = QwalsCalculator.from_folder("data/")
```

CLI users replace `wals-distance` (or `python -m wals_distance`) with
`qwals` (or `python -m qwals`); shell aliases and CI configs that pin the
old console script need updating. Anyone setting
`WALS_DISTANCE_CACHE_DIR` should switch to `QWALS_CACHE_DIR`. Cache files
written by 0.5.0 are not reused — the package version is part of the
cache key, so 0.6.0 will silently rebuild them on first use (~180 ms one
time) and write them to `~/.cache/qwals/`. The old
`~/.cache/wals_distance/` directory can be deleted by hand.

External references to the **WALS dataset itself** are unchanged: the
data file is still `wals-data.csv`, the order file is still
`WALS_feature_order.csv`, the optional column is still `Language_ID`, and
the keyword `WALS` remains in the package description, classifiers, and
docstrings wherever it refers to the dataset rather than this package.

### Internal
- Bumped version `0.5.0 → 0.6.0`.
- Class renamed `WalsDistanceCalculator → QwalsCalculator`.
- Module renamed `wals_distance → qwals`.
- Console script renamed `wals-distance → qwals`.
- Cache env var renamed `WALS_DISTANCE_CACHE_DIR → QWALS_CACHE_DIR`.
- Default cache directory renamed `~/.cache/wals_distance → ~/.cache/qwals`.
- All 50 tests updated to the new imports/env-var; **50/50 still pass**.

## [0.5.0] — 2026-05-01

### Added
- **`nearest(language, n=10, method="ordinal", include_self=False)`** — top-N
  closest languages as `[(name, distance), ...]`, sorted ascending. Vectorised
  one-vs-all on the existing matrices (~2 ms for the full WALS dataset). Ties
  are broken alphabetically (stable sort by language index), so two calls
  with the same arguments are bit-stable. Languages with no shared (weighted)
  features are excluded; if fewer than *n* viable neighbours exist the result
  is shorter.
- **`distance_to_many(language, others=None, method="ordinal", as_series=False, include_self=False)`**
  — a `{lang: distance}` dict (or `pandas.Series` when `as_series=True`).
  `others=None` means "every other language". Shares the same vectorised
  kernel as `nearest`, so it's roughly the same cost regardless of how many
  targets you pass.
- **Per-feature weights** for `distance` / `pairwise_matrix` / `nearest` /
  `distance_to_many`. Pass `weights={"Order of Subject and Verb": 2.0}` to
  the constructor, or mutate at runtime via:
  - `set_weight(feature, weight)` — single feature, validated
    (non-negative, finite, must exist).
  - `set_weights({feature: weight, ...})` — batch update; missing features
    keep their current value.
  - `reset_weights()` — back to all-1.0.
  - `weights` property — non-default weights as a `{feature: weight}` dict.
  Weighted formula:
  `Σ w[f]·mask[f]·d_f(L1,L2)  /  Σ w[f]·mask[f]`. A weight of 0 drops the
  feature entirely. Weights are *not* part of the cache identity (they're
  runtime state); `return_details=True` adds a `Weight` column to the
  per-feature DataFrame whenever any weight differs from 1.0.
- **`features_for(language) -> list[str]`** and
  **`shared_features(l1, l2) -> list[str]`** — the features one language has
  a value for, or that two languages both have. Both return names in the
  canonical `calc.features` order so they're stable across calls. Aliases
  accepted for the language argument(s).
- **CLI** at `python -m qwals` (and as a `qwals` console script after
  `pip install`; was `wals-distance` before the 0.6.0 rename). Subcommands:
  `compare`, `nearest`, `pairwise`, `shared`, `features`. Common flags:
  `--data PATH`, `--method {ordinal,onehot}`, `--no-cache`, `--version`.
  Pairwise can write to stdout (default) or `--out FILE.csv`.
- 33 new tests across `tests/test_features.py` and `tests/test_cli.py`
  covering each new method, the weights API, alias-input parity, and every
  CLI subcommand. **50 tests pass total** (was 13 in 0.4.0).

### Changed
- `distance(...)` and `pairwise_matrix(...)` internals reworked to honour
  per-feature weights. The shared-feature-count matmul that landed in 0.4.0
  is now `(Mf * w) @ Mf.T`, which folds in weights at zero extra cost.
  Behaviour is unchanged for the default (all-1.0) case — single-pair
  distance is bit-exact and pairwise drift stays ≤ 4·10⁻⁸ on ordinal.
- New `nearest()` ordering matches
  `sorted(distance_to_many(...).items(), key=lambda kv: kv[1])` for
  consistency, including alphabetical tie-breaking via stable argsort.
- `pyproject.toml` declares a console script (named `wals-distance` in
  0.5.0; renamed to `qwals` in 0.6.0) so the CLI works after a regular
  install.
- Bumped version `0.4.0 → 0.5.0`.

### Performance
On the bundled WALS data (2,673 languages × 192 features):

| New operation              | Time    | Notes                              |
| -------------------------- | ------: | ---------------------------------- |
| `nearest("Polish", n=10)`  | ~3 ms   | One-vs-all + stable argsort        |
| `distance_to_many` (all)   | ~2 ms   | Same kernel as `nearest`           |
| `features_for(lang)`       | <0.1 ms | Single mask read                   |
| `shared_features(l1, l2)`  | <0.1 ms | Two mask reads + AND               |

Unchanged from 0.4.0: cold init ~180 ms, warm init ~7 ms, full ordinal
pairwise ~1.1 s, full onehot pairwise ~0.7 s.

## [0.4.0] — 2026-05-01

### Added
- **Disk-cached precomputed matrices.** First build still parses the CSV
  (~180 ms); every subsequent `QwalsCalculator(...)` on the same data
  drops to **~7 ms** by `np.load`-ing a small `.npz` next to the data.
  New module `src/qwals/_cache.py`.
- New constructor parameter `cache: bool | str | Path = True`:
  - `True` (default) — read/write
    `~/.cache/qwals/<sha256-of-inputs>.npz`.
  - `False` — never read or write the cache.
  - `str | Path` — use that exact `.npz` path.
- New env var `QWALS_CACHE_DIR` overrides the default cache root.
- The cache key includes both file fingerprints (resolved path + size +
  mtime_ns) and the package version, so editing the source CSV, switching
  inference options, or upgrading the package all invalidate transparently.
  Corrupt / malformed cache files are silently treated as misses, never
  surfaced as errors.
- Seven new tests under `tests/test_cache.py` cover round-trip equivalence,
  custom paths, option/data invalidation, corrupt-file recovery, and key
  stability. A new `tests/conftest.py` redirects the cache to `tmp_path`
  during the test run so unit tests never touch the user's `~/.cache`.

### Changed
- **Tiled pairwise inner loop.** `pairwise_matrix` now processes
  `(B, B, F)` blocks (default `B=64`) instead of one row at a time, with
  the shared-feature count matrix precomputed via a single BLAS matmul
  rather than per-row reductions. On the full WALS dataset:
  - ordinal: 1.9 s → **1.1 s** (1.7× faster)
  - onehot:  1.0 s → **0.7 s** (1.4× faster)
  - 30-language ordinal subset: 0.6 ms → **0.2 ms**
  Internal arithmetic uses `float32` for the (B, B, F) tile and `float64`
  for the final accumulator; numerical drift vs the v0.3 row loop is
  bounded by ~4·10⁻⁸ on ordinal pairs (onehot remains bit-exact).
- `pairwise_matrix` gained an optional `block: int | None` argument for
  callers who want to override the default tile size.
- **Matrices are now `int16` instead of `int32`** (`_ord`, `_val`,
  `_n_ord`). WALS features have ≤ 9 unique values, so the int16 ceiling
  of 32 767 is comfortable. Memory of the two precomputed matrices is
  halved (4.1 MiB → 2.05 MiB on the full dataset).
- Constructor split into `_build_from_csv` + `_populate_from_payload`,
  used by the CSV path and the cache path respectively. No public-API
  change.
- Bumped version `0.3.0 → 0.4.0`.

### Cleanup
- Removed stale `build/` directory, `src/qwals/__pycache__/`, and the
  legacy `pytest-cache-files-*` directory.

### Performance
On the bundled WALS data (2,673 languages × 192 features):

| Operation                              | 0.3 (numpy)  | 0.4 (this)  | Speedup vs 0.3 |
| -------------------------------------- | -----------: | ----------: | -------------: |
| `init` (cold, no cache)                | ~190 ms      | ~180 ms     | ~1×            |
| `init` (warm, disk-cached)             | n/a          | **~7 ms**   | ~26×           |
| `distance` (ordinal, single pair)      | 7 µs         | 6 µs        | ~1.2×          |
| `distance` (onehot, single pair)       | 4 µs         | 3 µs        | ~1.3×          |
| `pairwise_matrix` 30 langs (ordinal)   | <1 ms        | 0.2 ms      | ~3.4×          |
| `pairwise_matrix` full ordinal (2,673) | 1.9 s        | **1.1 s**   | ~1.7×          |
| `pairwise_matrix` full onehot (2,673)  | 1.0 s        | **0.7 s**   | ~1.4×          |
| `_ord` + `_val` memory                 | 4.10 MiB     | 2.05 MiB    | 0.5×           |

## [0.3.0] — 2026-05-01

### Added
- **Alias resolution** — `distance`, `similarity`, `pairwise_matrix`,
  `feature_distance`, and `save_pairwise_matrix` now accept any of:
  - exact WALS `Language_name` (e.g. `"Polish"`)
  - case-insensitive form (e.g. `"polish"`, `"POLISH"`)
  - WALS `Language_ID` (3-letter, pulled from the data CSV — e.g. `"pol"`)
  - ISO 639-1 (2-letter, from an embedded table — e.g. `"pl"`)
  - any user-registered alias (see `add_alias`)
- New module `src/qwals/_aliases.py` — embedded ISO 639-1 →
  WALS-friendly English-name table (~165 entries, ~2.5 KB). Names that
  don't appear in the loaded data are silently dropped, so the table is
  safe to extend.
- New public methods:
  - `resolve_language(name) -> str` — canonicalize any input form.
  - `add_alias(alias, language)` — register custom aliases at runtime.
  - `aliases_for(language) -> list[str]` — discover all known aliases.
- `pairwise_matrix` now always emits canonical WALS names in `index` /
  `columns`, regardless of which alias form the caller passed.
- `_read_csv` gained an `optional` parameter for tolerated columns;
  `Language_ID` is read this way.
- Five new tests covering alias resolution, distance equivalence across
  input forms, canonical labels, custom alias add/query, and
  unknown-alias errors. (Six tests total now pass.)
- README sections: **Language identifiers** and updated **Performance**
  table with v0.3 numbers.

### Changed
- Ordinal `pairwise_matrix` inner loop now precomputes
  `O_scaled = O / denom` once and reuses a single `(L, F)` float buffer
  with in-place numpy ops. Halves runtime on the full WALS dataset
  (2.1 s → ~1.9 s), and the onehot path is similarly down to ~1.0 s.
- Bumped version `0.2.0 → 0.3.0`.

### Performance
On the bundled WALS data (2,673 languages × 192 features):

| Operation                              | 0.1 (pandas)  | 0.3 (numpy) | Speedup    |
| -------------------------------------- | ------------: | ----------: | ---------: |
| `distance` (ordinal, single pair)      | 22 ms         | 7 µs        | ~3,000×    |
| `distance` (onehot, single pair)       | 24 ms         | 4 µs        | ~5,500×    |
| `pairwise_matrix` 30 langs (ordinal)   | 18 s          | <1 ms       | ~26,000×   |
| `pairwise_matrix` full ordinal (2,673) | ~39 h (proj.) | 1.9 s       | ~75,000×   |
| `pairwise_matrix` full onehot (2,673)  | ~39 h (proj.) | 1.0 s       | ~140,000×  |

## [0.2.0] — 2026-05-01

### Added
- `from_folder` now silently skips a missing `WALS_feature_order.csv`
  rather than raising.
- `__version__` exported from package root.
- `py.typed` marker for downstream type-checkers.
- README **Performance** section.

### Changed
- **Vectorised numpy core.** Two `int32` matrices of shape
  `(n_languages, n_features)` are precomputed once at construction time:
  - `_ord` — value's ordinal index, or `-1` for missing/unordered.
  - `_val` — synthetic value id (used only for onehot equality), or `-1`
    for missing.
  All hot paths (`distance`, `feature_distance`, `pairwise_matrix`) are
  pure numpy on row slices of these matrices.
- **Pandas is now lazy/optional.** Imported only inside `pairwise_matrix`,
  `save_pairwise_matrix`, and `distance(..., return_details=True)`.
  `pyproject.toml` lists pandas under `[reports]` extras; the minimal
  install is numpy-only.
- CSV reading switched from `pandas.read_csv` to stdlib `csv`. Behaviour
  matches pandas defaults (the `_NA` frozenset reproduces
  `pandas.read_csv`'s default `na_values`), so language and feature
  counts are bit-identical to v0.1.
- `infer_feature_orders` collapsed from a `groupby` plus inner Python
  loop into a single pass over the cleaned triples.

### Fixed
- NA-sentinel handling now matches pandas semantics — strings like
  `"None"`, `"NA"`, `"null"` are dropped rather than treated as values.
  (Previously, switching off pandas changed the language count by 3 and
  the feature counts of several languages.)

### Removed
- Top-level `pandas` import. Users who never call `pairwise_matrix` or
  `return_details=True` no longer pay its import cost.

## [0.1.0] — Original release

Initial pandas-based implementation. The main calculator class (called
`WalsDistanceCalculator` until 0.5.0, then renamed to `QwalsCalculator`
in 0.6.0) provided `distance` / `feature_distance` / `pairwise_matrix`
methods backed by DataFrame filters and Python loops.

[0.8.0]: ./CHANGELOG.md#080--2026-05-02
[0.7.0]: ./CHANGELOG.md#070--2026-05-02
[0.6.0]: ./CHANGELOG.md#060--2026-05-02
[0.5.0]: ./CHANGELOG.md#050--2026-05-01
[0.4.0]: ./CHANGELOG.md#040--2026-05-01
[0.3.0]: ./CHANGELOG.md#030--2026-05-01
[0.2.0]: ./CHANGELOG.md#020--2026-05-01
[0.1.0]: ./CHANGELOG.md#010--original-release
