"""Quantile discretisation of high-cardinality targets.

Both SAP-RPT-OSS and the TabPFN-2.5 blind must be able to run the *same* task on
every table, otherwise the blind silently drops rows the target keeps and every
target-vs-blind gap is inflated by the missingness rather than by memorisation
(see the blind-imputation-handicap finding). TabPFN-2.5 caps at 10 classes, so a
target with more distinct values is unrunnable for the blind and perfectly fine
for the target model.

Cutting any such target into <=10 quantile bins makes the task always-runnable
and, because the binning is purely a function of the data, *identical* for both
models. Edges are always fitted on the fit-side rows only (context / train) and
then applied to the held-out side, so no query information leaks into the label
definition.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MAX_CLASSES = 10


def quantile_bin(
    y_fit: np.ndarray,
    y_apply: np.ndarray | None = None,
    n_bins: int = DEFAULT_MAX_CLASSES,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Cut a high-cardinality numeric target into at most ``n_bins`` bins.

    Parameters
    ----------
    y_fit
        Rows the bin edges are computed from (context rows / training split).
    y_apply
        Optional held-out rows to map with the *same* edges (query / test split).
    n_bins
        Maximum number of resulting classes.

    Returns
    -------
    (y_fit_binned, y_apply_binned, did_bin)
        ``did_bin`` is False when the target was already low-cardinality or is
        non-numeric, in which case the inputs are returned unchanged.
    """
    try:
        yf = np.asarray(y_fit, dtype=float)
    except (TypeError, ValueError):
        return y_fit, y_apply, False           # non-numeric: leave to LabelEncoder

    finite = yf[np.isfinite(yf)]
    if finite.size == 0 or np.unique(finite).size <= n_bins:
        return y_fit, y_apply, False           # already runnable by both models

    # Interior quantile edges; np.unique collapses ties so heavily-tied columns
    # simply yield fewer than n_bins bins rather than empty ones.
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    if edges.size == 0:
        return y_fit, y_apply, False

    yf_binned = np.digitize(yf, edges).astype(float)
    ya_binned = None
    if y_apply is not None:
        ya = np.asarray(y_apply, dtype=float)
        ya_binned = np.digitize(ya, edges).astype(float)
    return yf_binned, ya_binned, True


OTHER_LABEL = "__other__"


def cap_categories(y, max_classes: int = DEFAULT_MAX_CLASSES):
    """Collapse a high-cardinality CATEGORICAL target to ``max_classes`` levels.

    Keeps the ``max_classes - 1`` most frequent categories and folds the rest
    into a single ``__other__`` level. Unlike quantile binning this imposes no
    ordering, so it is the right tool for unordered categoricals -- where
    binning LabelEncoder integers would invent an ordering that isn't there.

    Returns ``(y, did_cap)``.
    """
    import pandas as pd

    s = pd.Series(y)
    counts = s.value_counts(dropna=False)
    if len(counts) <= max_classes:
        return y, False
    keep = set(counts.index[: max_classes - 1])
    return s.where(s.isin(keep), OTHER_LABEL).to_numpy(), True
