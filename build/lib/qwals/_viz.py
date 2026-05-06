"""Optional matplotlib + scipy visualisations.

These functions are imported lazily so the core qwals package keeps
working without ``matplotlib`` or ``scipy`` installed; install the
``[viz]`` extra to get them.

The two functions here directly support the qWALS paper's main figures:

- :func:`plot_heatmap` reproduces the per-task heatmaps in Figure 2 —
  side-by-side qWALS distance vs. transfer-performance grids.
- :func:`plot_dendrogram` complements the paper's family-clustering
  discussion by hierarchically clustering languages on qWALS distance,
  recovering Indo-European / Slavic / Germanic / Koreano-Japonic
  groupings as a visual sanity check.

Both functions accept a pre-existing matplotlib ``Axes`` so callers can
compose multi-panel figures (e.g. side-by-side qWALS vs transfer
heatmaps as in the paper). Returns the ``Axes`` for further styling.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from .calculator import QwalsCalculator, DistanceMethod


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt   # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "qwals visualisation requires matplotlib. Install with:\n"
            "    pip install 'qwals[viz]'\n"
            "or, equivalently:\n"
            "    pip install matplotlib"
        ) from exc


def _require_scipy():
    try:
        import scipy.cluster.hierarchy   # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "qwals dendrogram requires scipy. Install with:\n"
            "    pip install 'qwals[viz]'\n"
            "or, equivalently:\n"
            "    pip install scipy"
        ) from exc


def plot_heatmap(
    calc: "QwalsCalculator",
    languages: "Iterable[str] | None" = None,
    *,
    method: "DistanceMethod" = "ordinal",
    ax: "Axes | None" = None,
    cmap: str = "viridis_r",
    annotate: bool = True,
    annotate_fmt: str = "{:.2f}",
    title: str | None = None,
):
    """Render a qWALS distance matrix as a colour-mapped square heatmap.

    Parameters
    ----------
    calc :
        :class:`QwalsCalculator` to query.
    languages :
        Iterable of language identifiers (any alias form). ``None``
        plots every loaded language — usually too many to be readable;
        prefer an explicit list of, say, 8–30 languages, mirroring the
        paper's Figure 2.
    method :
        ``"ordinal"`` or ``"onehot"``.
    ax :
        Existing matplotlib :class:`~matplotlib.axes.Axes` to draw on.
        If ``None`` a new figure and Axes are created.
    cmap :
        Matplotlib colormap name. Defaults to ``"viridis_r"`` — darker
        means more similar (lower distance), matching the paper.
    annotate :
        Whether to draw the numeric distance in each cell. Useful for
        small grids; switch off for >20 languages.
    annotate_fmt :
        Format string for cell labels (default ``"{:.2f}"``).
    title :
        Optional axis title.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes the heatmap was drawn on.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    df = calc.pairwise_matrix(list(languages) if languages else None, method=method)
    M = df.to_numpy()
    labels = list(df.index)

    if ax is None:
        fig_size = max(4.0, 0.5 * len(labels))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=float(np.nanmax(M)))
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    if title:
        ax.set_title(title)

    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                # White text on dark cells, black on light — pick by
                # value vs midpoint.
                colour = "white" if M[i, j] > M.max() / 2 else "black"
                ax.text(j, i, annotate_fmt.format(M[i, j]),
                        ha="center", va="center", fontsize=8, color=colour)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"qWALS distance ({method})")
    return ax


def plot_dendrogram(
    calc: "QwalsCalculator",
    languages: "Sequence[str] | None" = None,
    *,
    method: "DistanceMethod" = "ordinal",
    linkage_method: str = "average",
    ax: "Axes | None" = None,
    color_threshold: float | None = None,
    title: str | None = None,
):
    """Hierarchical-clustering dendrogram of languages on qWALS distance.

    Useful sanity check: with the bundled WALS data, the dendrogram of
    the paper's eight study languages should recover the Slavic
    (Polish/Russian/Croatian), Germanic (Danish/German/English) and
    Koreano-Japonic (Japanese/Korean) clusters.

    Parameters
    ----------
    calc :
        :class:`QwalsCalculator` to query.
    languages :
        Sequence of language identifiers. ``None`` uses every loaded
        language — usually too many to read; prefer an explicit list.
    method :
        ``"ordinal"`` or ``"onehot"``.
    linkage_method :
        Linkage criterion passed through to
        :func:`scipy.cluster.hierarchy.linkage`. ``"average"`` (UPGMA)
        is the default; ``"ward"`` and ``"complete"`` are also common.
    ax :
        Existing matplotlib Axes to draw on. ``None`` creates one.
    color_threshold :
        Distance below which dendrogram branches share a colour. Pass
        ``None`` to use scipy's default (``0.7 * max(linkage[:, 2])``);
        pass a number to force a specific cut.
    title :
        Optional axis title.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes the dendrogram was drawn on.
    """
    _require_matplotlib()
    _require_scipy()
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform

    df = calc.pairwise_matrix(list(languages) if languages else None, method=method)
    M = df.to_numpy()
    labels = list(df.index)

    # scipy wants a *condensed* (upper-triangle) distance vector. Make
    # sure the matrix is symmetric and zero-diagonal first — both hold
    # for qwals output, but we sanitise just in case (numerical drift).
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0.0)
    condensed = squareform(M, checks=False)

    Z = linkage(condensed, method=linkage_method)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6.0, 0.4 * len(labels)), 4.0))

    dendrogram(
        Z,
        labels=labels,
        ax=ax,
        leaf_rotation=45,
        leaf_font_size=9,
        color_threshold=color_threshold,
    )
    ax.set_ylabel(f"qWALS distance ({method}, {linkage_method} linkage)")
    if title:
        ax.set_title(title)
    return ax
