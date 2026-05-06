# Potential improvements (post-0.8.0)

A menu of ideas for the next session, grouped by category and ordered
within each group by **effort-to-payoff** (best first). Pick whichever
matches your appetite.

## What landed in 0.8.0 (paper-aligned)

Nine new features shipped in 0.8.0, aligned with Eronen et al. (2026):

- ✅ **Task-specific feature presets** — `TASK_FEATURES` dict + `use_features("dep"/...)`,
  with the four paper-Appendix-A subsets pre-shipped (53/21/63/75 features).
- ✅ **Leave-one-feature-out optimiser** — `optimize_features(target_scores, ...)`
  replicates §4.5; greedy LOFO with Pearson/Spearman, history trace, etc.
- ✅ **Per-distance coverage / shared-count return** — `distance(..., return_coverage=True)`,
  `coverage_for(L1, L2)`, and `nearest(..., with_coverage=True)`.
- ✅ **Bootstrap confidence intervals** — `distance_ci(L1, L2, n_bootstrap, ci, rng)`
  for the sparsity-reliability concern from §4.4.
- ✅ **Distance-with-uncertainty for nearest** — `nearest(..., sort_by="upper_ci",
  n_bootstrap=N)` returns `(name, dist, n_shared, ci_high)` 4-tuples.
- ✅ **`explain_distance(L1, L2, top_k)`** — per-feature contribution DataFrame
  ("transparent vs lang2vec").
- ✅ **`plot_heatmap()`** — matplotlib heatmaps mirroring Figure 2 of the paper.
- ✅ **`plot_dendrogram()`** — hierarchical clustering with UPGMA / Ward / etc.
- ✅ **`suggest_transfer_source()`** — paper-cited ranking utility wrapping
  preset + nearest + coverage in a single call.

## Original roadmap — strongest remaining candidates

1. **Optional Numba JIT** — ~2× on full pairwise (item 2 below).
2. **Mixed methods per feature** — the only piece deliberately deferred
   from the 0.5 cycle (item 8 below).
3. **Geographic and/or phylogenetic priors** (items 15–16) for richer
   typology research.

---

## Performance

### 1. ~~Disk-cached precomputed matrices~~ — DONE in 0.4.0
Cached to `~/.cache/qwals/<hash>.npz`. Warm init: ~7 ms (was
~190 ms). See `src/qwals/_cache.py`.

### 2. Optional Numba JIT for the pairwise inner loop
The ordinal pairwise spends most of its time on `np.subtract` /
`np.abs` / `np.multiply` over `(L, F)` arrays. A `@njit(parallel=True)`
function with `prange(L)` typically halves the 1.1 s on a multi-core
machine.

- Effort: ~10 LOC core, ~5 LOC fallback path.
- Payoff: ~2× on full pairwise; first call has a JIT warm-up cost.
- Add as soft-optional: `import numba` inside a try/except, fall back
  to current numpy implementation otherwise.

### 3. ~~Tile the pairwise into blocks~~ — DONE in 0.4.0
`(B=64, B=64, F)` tile + a single BLAS matmul for the shared-mask
matrix replaced the row loop. Ordinal 1.9 s → 1.1 s, onehot 1.0 s →
0.7 s.

### 4. ~~`int16` matrices instead of `int32`~~ — DONE in 0.4.0
`_ord`, `_val`, `_n_ord` are now int16. Matrix memory halved
(4.1 MiB → 2.05 MiB on full WALS).

---

## New API features

### 5. ~~`nearest(language, n=10, method="ordinal")`~~ — DONE in 0.5.0
Vectorised one-vs-all + stable argsort. ~3 ms on full WALS. Honors
weights; ties broken alphabetically. See
`QwalsCalculator.nearest`.

### 6. ~~`distance_to_many(language, others=None, method=...)`~~ — DONE in 0.5.0
Returns `dict[str, float]` (or `pandas.Series` with `as_series=True`).
Same kernel as `nearest`, ~2 ms. Honors weights.

### 7. ~~Per-feature weights~~ — DONE in 0.5.0
Constructor `weights={...}` plus `set_weight` / `set_weights` /
`reset_weights` / `weights` property. Threads through `distance`,
`pairwise_matrix`, `nearest`, `distance_to_many`. Default behaviour
unchanged.

### 8. Mixed methods per feature  ★ next API target
Some WALS features are genuinely categorical (no order makes sense).
Allow `methods={"Feature A": "onehot", "Feature B": "ordinal"}` and
combine. Deliberately skipped in the 0.5 cycle so it stays uncrowded.

- Effort: ~15 LOC.
- Payoff: more linguistically defensible distances.

### 9. ~~`shared_features` and `features_for`~~ — DONE in 0.5.0
Both return canonical-order lists in <0.1 ms. Aliases accepted.

### 10. ~~CLI~~ — DONE in 0.5.0
`python -m qwals` (and `qwals` console script after install).
Subcommands: `compare`, `nearest`, `pairwise`, `shared`, `features`.
See README "Command-line interface" section.

---

## Robustness

### 11. Detect duplicate `(language, feature)` rows on load
Currently the last row silently wins. Either warn or take the first
(make it configurable).

- Effort: ~10 LOC during the existing load pass.

### 12. Fuzzy alias matching as a last resort (opt-in)
When `resolve_language("Polski")` fails, optionally try
`get_close_matches` against names *and* alias keys before raising.

- Effort: ~8 LOC.
- Payoff: nicer DX for polyglot researchers.

### 13. Extra `infer_missing_orders` strategies
Add `"alphabetical"` and `"frequency"` (most-common value first) as
options alongside the existing `"appearance"` and `"sorted"`.

- Effort: ~8 LOC.

### 14. Asymmetric distance variants
Some research uses Hamming-style distance over the intersection size
weighted by total feature count of one language. Currently you
implicitly assume symmetric mean.

- Effort: ~15 LOC (new method, not a flag — keep `distance` clean).

---

## Linguistic capabilities (research extensions)

### 15. Phylogenetic prior
Accept a Newick tree of language families and combine WALS distance
with tree distance: `α · WALS + (1-α) · tree`.

- Effort: ~80 LOC + a Newick parser dep (or hand-roll, ~30 LOC).
- Payoff: powerful for studies of language contact vs. inheritance.

### 16. Geographic prior
Accept lat/lon (from WALS' `languages.csv`), compute great-circle
distance, blend it with WALS distance.

- Effort: ~30 LOC, no new deps (Haversine is one formula).
- Payoff: separates areal influence from typological similarity.

### 17. Multi-source merging
Concatenate WALS with APiCS, PHOIBLE, or Grambank along the feature
axis. Naturally falls out of the `(L, F)` matrix layout.

- Effort: ~40 LOC + per-source ingestion.
- Payoff: bigger and richer feature spaces.

### 18. Correlation-style distance
Instead of mean of |i₁−i₂|/(n−1), compute Pearson or cosine over the
ordinal index vectors. Captures shape patterns rather than absolute
gaps.

- Effort: ~10 LOC as an additional `method`.

---

## Quality / distribution

### 19. CI on GitHub Actions
Matrix of Python 3.9–3.13 × Linux/macOS/Windows; run `pytest`, `ruff`,
and a tiny benchmark with regression alarm. ~30 lines of YAML.

### 20. Publish to PyPI
`pyproject.toml` is already PEP-621-shaped. Need to fill in the real
GitHub URL and a real maintainer name; then `python -m build` and
`twine upload`.

### 21. `mypy --strict` + `ruff` in dev extras
Codebase is small enough that both finish in milliseconds.

### 22. Property-based tests with Hypothesis
Generate random `(language, feature, value)` triples and assert
invariants:
- `distance(L, L) == 0` (self-distance)
- `distance(A, B) == distance(B, A)` (symmetry)
- `0 ≤ distance ≤ 1` (range)
- `onehot ≥ ordinal` for the same pair (always)

### 23. Doctests in docstrings
Keeps the README examples executable; `pytest --doctest-modules` then
catches drift.

---

## Cleanup still pending (carry-overs)

All 0.3.0-era cleanup is **DONE in 0.4.0** — `build/`,
`src/qwals/__pycache__/`, and the legacy
`pytest-cache-files-*/` directories have been removed.
