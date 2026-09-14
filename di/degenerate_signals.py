"""Signals that are NOT comparable between the TabDPT target and the TabPFN-2.5 blind.

Validated 2026-07-23 (full pipeline validation). Three signal families are either
structurally degenerate or representation-mismatched for the TabPFN-2.5 blind, so
any target-vs-blind contrast on them manufactures a spurious "target beats blind"
gap purely from the blind's inability to vary — NOT from memorization. They are
excluded from every target-vs-blind comparison and stripped from the grid output
so future result files never carry them into one.

1. ``seed_*`` — the seed sweep varies the feature-permutation ``seed`` at PREDICT
   time. TabDPT permutes feature columns per predict call; TabPFN-2.5's
   ``predict_proba(X)`` takes no seed (``random_state`` is a *constructor* arg,
   applied at fit), so every seeded prediction is identical → ``seed_*_std`` /
   ``_range`` / ``seed_rank_corr`` are exactly 0 for the blind (blind AUC pinned at
   0.5). ``seed_consistency`` is additionally broken for regression: it is an
   unnormalized ``1 - mean(var(preds))`` that swings to ~-1e11 on large-scale
   targets.

2. ``temp_{accuracy,chi_squared,wasserstein}_{std,range}`` — temperature scaling
   never changes the argmax, so these argmax-based dispersion signals carry no
   information for either backend (exactly 0 for the blind's post-hoc rescale,
   numerical noise for the target). NB: the confidence-based temperature signals
   (``temp_correct_confidence`` / ``entropy`` / ``kl_divergence`` / ``loss`` /
   ``max_confidence`` / ``prediction_margin`` — ``_std`` / ``_range`` / ``_mean``)
   DO vary for both backends and are kept: TabPFN's post-hoc temperature rescale is
   mathematically identical to native logit-temperature scaling.

3. ``logit_*`` — the blind returns log-probs (≤0) via ``np.log(clip(probs))`` while
   TabDPT returns raw pre-softmax logits. ``logit_magnitude`` / ``logit_kurtosis``
   measure different quantities across backends (``logit_margin`` is comparable
   only for binary tasks). The per-row logit features in ``/grid/row_signals.py``
   are a separate concern (row-level DI) and are unaffected by this filter.
"""

from __future__ import annotations

from typing import Mapping, TypeVar

# Exact-name degenerate signals: the six temperature argmax-dispersion features.
_TEMP_ARGMAX_DEGENERATE = frozenset(
    {
        "temp_accuracy_std",
        "temp_accuracy_range",
        "temp_chi_squared_std",
        "temp_chi_squared_range",
        "temp_wasserstein_std",
        "temp_wasserstein_range",
    }
)


def is_blind_degenerate(name: str) -> bool:
    """True if ``name`` is a signal that is not comparable to the TabPFN-2.5 blind.

    Covers the whole ``seed_*`` sweep, all dataset-level ``logit_*`` geometry
    signals, and the six ``temp_*`` argmax-dispersion signals. Use to exclude
    signals from any target-vs-blind comparison and to strip them from grid output.
    """
    if name.startswith("seed_"):
        return True
    if name.startswith("logit_"):
        return True
    return name in _TEMP_ARGMAX_DEGENERATE


_V = TypeVar("_V")


def filter_degenerate(signals: Mapping[str, _V]) -> dict[str, _V]:
    """Return a new dict with all blind-degenerate signal keys removed."""
    return {k: v for k, v in signals.items() if not is_blind_degenerate(k)}
