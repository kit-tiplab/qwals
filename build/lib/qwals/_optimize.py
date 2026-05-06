"""Leave-one-feature-out (LOFO) optimisation of the qWALS feature set.

Replicates the procedure described in §4.5 of the qWALS paper:

    "We omitted one feature at a time and checked the correlation with
     cross-lingual transfer performance. After going through all the
     features, we dropped the feature whose removal caused the greatest
     decrease in the correlation between linguistic similarity and
     transfer performance. The process was continued until no more
     improvement was observed."

That paper got the dependency-parsing correlation from −0.77 (full 169
features) to −0.99 (75 features). This module gives users the same
procedure as a callable, so they can re-run it on any task or transfer
score they have.

The optimiser is greedy and operates on the *active* feature mask of the
provided ``QwalsCalculator``: it begins from whatever set is currently
active (use :meth:`reset_features` first if you want to start from the
full set, or :meth:`use_features` first to start from one of the paper's
presets) and iteratively drops one feature at a time until no further
improvement is possible — or until ``max_drops`` / ``min_features`` is
hit.

Performance: each step recomputes the pairwise weighted distance for
every dropped-feature candidate. With ``L`` study languages and ``F``
currently-active features, one step is ``O(F · L²)``. For a typical
qWALS workload (8 languages, 192 features, 8 pairs evaluated), the
whole optimisation completes in a few hundred milliseconds.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np

if TYPE_CHECKING:
    from .calculator import QwalsCalculator, DistanceMethod


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; pure NumPy so we avoid a SciPy dep."""
    if a.size < 2:
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ma, mb = ra.mean(), rb.mean()
    num = ((ra - ma) * (rb - mb)).sum()
    den = np.sqrt(((ra - ma) ** 2).sum() * ((rb - mb) ** 2).sum())
    return float(num / den) if den != 0 else 0.0


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return 0.0
    ma, mb = a.mean(), b.mean()
    num = ((a - ma) * (b - mb)).sum()
    den = np.sqrt(((a - ma) ** 2).sum() * ((b - mb) ** 2).sum())
    return float(num / den) if den != 0 else 0.0


def _score(
    correlation: str,
    direction: str,
    distances: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Score we want to maximise during the search.

    ``correlation`` is ``"pearson"`` or ``"spearman"``; ``direction`` is
    ``"minimise"`` (we expect distance and transfer-quality to anti-
    correlate, so a more-negative ρ is better, i.e. maximise ``-ρ``) or
    ``"maximise"`` (e.g. correlating distance against an expert
    *dissimilarity* rating — a more-positive ρ is better).
    """
    fn = _pearson if correlation == "pearson" else _spearman
    rho = fn(distances, targets)
    return -rho if direction == "minimise" else rho


def optimize_features(
    calc: "QwalsCalculator",
    target_scores: Mapping[tuple[str, str], float],
    *,
    method: "DistanceMethod" = "ordinal",
    correlation: str = "pearson",
    direction: str = "minimise",
    max_drops: int | None = None,
    min_features: int = 5,
    verbose: bool = False,
) -> dict:
    """Greedy LOFO search for the feature set that best correlates with target_scores.

    Parameters
    ----------
    calc :
        The :class:`QwalsCalculator` to operate on. The active feature
        mask is *not* mutated — call ``calc.use_features(result["features"])``
        afterwards if you want to apply the optimised set.
    target_scores :
        Mapping ``{(lang1, lang2): score}``. Typical scores are
        cross-lingual transfer F1 / LAS values, but any per-pair scalar
        works (expert similarity ratings, lang2vec distances, etc.).
        Aliases are resolved through the calculator.
    method :
        Distance method, ``"ordinal"`` or ``"onehot"`` — same as in
        :meth:`QwalsCalculator.distance`.
    correlation :
        ``"pearson"`` (default) or ``"spearman"``.
    direction :
        ``"minimise"`` (default) when low distance ↔ high target score
        (the typical "more similar → better transfer" case), or
        ``"maximise"`` when distance ↔ target are positively related.
    max_drops :
        Maximum number of features to drop in total. ``None`` means "no
        limit; keep going until improvement stalls".
    min_features :
        Floor on the number of remaining active features. The search
        stops if dropping one more would put the active count below this
        value. Default 5 — empirically the smallest set that produces
        stable correlations.
    verbose :
        If True, print one line per drop step (which feature, new ρ).

    Returns
    -------
    dict
        ``{"features": [...], "pearson": ρ, "spearman": ρ_s, "n_features",
          "n_dropped", "history": [{"drop_step", "feature", "score",
          "n_active"}, ...]}``.
        ``features`` is the final active list after optimisation, in
        canonical order.
    """
    if not target_scores:
        raise ValueError("target_scores is empty.")
    if correlation not in ("pearson", "spearman"):
        raise ValueError("correlation must be 'pearson' or 'spearman'.")
    if direction not in ("minimise", "maximise"):
        raise ValueError("direction must be 'minimise' or 'maximise'.")

    # Resolve language pairs once.
    pair_indices: list[tuple[int, int]] = []
    targets_list: list[float] = []
    for (a, b), score in target_scores.items():
        ai = calc._lang_idx[calc.resolve_language(a)]
        bi = calc._lang_idx[calc.resolve_language(b)]
        if ai == bi:
            continue   # self-pair carries no signal about transfer
        pair_indices.append((ai, bi))
        targets_list.append(float(score))
    if len(pair_indices) < 2:
        raise ValueError(
            "optimize_features needs at least 2 distinct language pairs to "
            "correlate against."
        )
    targets = np.asarray(targets_list, dtype=np.float64)

    # Snapshot then operate on a copy of the active mask — we never
    # mutate the caller's calculator state.
    work_mask = calc._feature_mask.copy()
    weights = calc._weights.astype(np.float64)
    n_total = work_mask.size

    def _distances_for(mask: np.ndarray) -> np.ndarray:
        """Pairwise weighted distances under the given feature mask."""
        Weff = weights * mask
        dists = np.empty(len(pair_indices), dtype=np.float64)
        for k, (ai, bi) in enumerate(pair_indices):
            if method == "onehot":
                v1, v2 = calc._val[ai], calc._val[bi]
                m = (v1 >= 0) & (v2 >= 0) & mask
                if not m.any():
                    dists[k] = np.nan
                    continue
                mw = m.astype(np.float64) * Weff
                tw = mw.sum()
                d = ((v1 != v2).astype(np.float64) * mw).sum() / tw if tw > 0 else 0.0
            else:
                o1, o2 = calc._ord[ai], calc._ord[bi]
                m = (o1 >= 0) & (o2 >= 0) & mask
                if not m.any():
                    dists[k] = np.nan
                    continue
                denom = np.maximum(calc._n_ord - 1, 1).astype(np.float64)
                diffs = np.abs(o1 - o2).astype(np.float64) / denom
                mw = m.astype(np.float64) * Weff
                tw = mw.sum()
                d = (diffs * mw).sum() / tw if tw > 0 else 0.0
            dists[k] = d
        return dists

    def _score_mask(mask: np.ndarray) -> float:
        d = _distances_for(mask)
        if np.isnan(d).any():
            return -np.inf
        return _score(correlation, direction, d, targets)

    base_score = _score_mask(work_mask)
    history: list[dict] = [{
        "drop_step": 0, "feature": None,
        "score": base_score, "n_active": int(work_mask.sum()),
    }]
    if verbose:
        print(f"[step 0] start: n={int(work_mask.sum())} score={base_score:+.4f}")

    n_dropped = 0
    while True:
        if max_drops is not None and n_dropped >= max_drops:
            break
        if int(work_mask.sum()) <= min_features:
            break

        active_idx = np.flatnonzero(work_mask)
        best_score = base_score
        best_feat = -1
        for fi in active_idx:
            work_mask[fi] = False
            s = _score_mask(work_mask)
            work_mask[fi] = True
            if s > best_score:
                best_score = s
                best_feat = int(fi)

        if best_feat < 0:
            break  # no single drop improves the score

        work_mask[best_feat] = False
        n_dropped += 1
        base_score = best_score
        history.append({
            "drop_step": n_dropped,
            "feature": calc.features[best_feat],
            "score": best_score,
            "n_active": int(work_mask.sum()),
        })
        if verbose:
            print(
                f"[step {n_dropped}] dropped {calc.features[best_feat]!r}: "
                f"n={int(work_mask.sum())} score={best_score:+.4f}"
            )

    # Compute both Pearson and Spearman for the final mask, in raw
    # (un-direction-flipped) form, so the report is intuitive.
    final_dists = _distances_for(work_mask)
    pearson = _pearson(final_dists, targets) if not np.isnan(final_dists).any() else float("nan")
    spearman = _spearman(final_dists, targets) if not np.isnan(final_dists).any() else float("nan")

    return {
        "features": [calc.features[i] for i in np.flatnonzero(work_mask)],
        "pearson": pearson,
        "spearman": spearman,
        "n_features": int(work_mask.sum()),
        "n_dropped": n_dropped,
        "history": history,
        "method": method,
        "correlation": correlation,
        "direction": direction,
    }
