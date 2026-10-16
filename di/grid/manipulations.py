"""Modular manipulation pipeline: composable functions over ManipulationState."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import os

import numpy as np

from ..binning import quantile_bin
from .specs import ManipulationSpec

# Bin the task_shuffle target to <=10 classes so the TabPFN-2.5 blind can run the
# same probe as the target model. Opt out to reproduce pre-2026-07-26 extractions.
BIN_SWAPPED_TARGET = os.environ.get("DI_NO_BIN_SWAPPED_TARGET") != "1"

# ---------------------------------------------------------------------------
# Shared state threaded through the pipeline
# ---------------------------------------------------------------------------

@dataclass
class ManipulationState:
    X_context: np.ndarray
    y_context: np.ndarray
    X_query: np.ndarray | None = None
    y_query: np.ndarray | None = None
    col_indices: list[int] | None = None


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ManipulationFn = Callable[[ManipulationState, ManipulationSpec], ManipulationState]
MislabelStrategyFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.random.Generator], np.ndarray
]


# ---------------------------------------------------------------------------
# Individual manipulation functions
# ---------------------------------------------------------------------------

def task_shuffle(state: ManipulationState, spec: ManipulationSpec) -> ManipulationState:
    """Swap a feature column with the target column."""
    if spec.target_column is None:
        return state
    k = spec.target_column

    # Context: build [X | y], select column k as new target
    full_ctx = np.column_stack([state.X_context, state.y_context.reshape(-1, 1)])
    state.y_context = full_ctx[:, k].copy()
    state.X_context = np.delete(full_ctx, k, axis=1)

    # Query: same column swap
    if state.X_query is not None and state.y_query is not None:
        full_q = np.column_stack([state.X_query, state.y_query.reshape(-1, 1)])
        state.y_query = full_q[:, k].copy()
        state.X_query = np.delete(full_q, k, axis=1)

    # The swapped-in column is an arbitrary feature, so its cardinality is
    # arbitrary too — often continuous. A >10-class target is unrunnable for the
    # TabPFN-2.5 blind but fine for the target model, which would leave the blind
    # with fewer valid rows purely as an artifact. Cut it to <=10 quantile bins,
    # edges from the context rows only, so both models get an identical task.
    if BIN_SWAPPED_TARGET:
        y_ctx, y_q, _ = quantile_bin(state.y_context, state.y_query)
        state.y_context = y_ctx
        if y_q is not None:
            state.y_query = y_q

    return state


def feature_transform(state: ManipulationState, spec: ManipulationSpec) -> ManipulationState:
    """Apply a fixed feature transform to both context and query."""
    if spec.feature_transform is None:
        return state

    def _apply(X: np.ndarray, transform: str) -> np.ndarray:
        if transform == "log":
            return np.sign(X) * np.log1p(np.abs(X))
        elif transform == "sqrt":
            return np.sign(X) * np.sqrt(np.abs(X))
        elif transform == "square":
            return X ** 2
        elif transform == "reciprocal":
            return np.sign(X) / (1.0 + np.abs(X))
        return X

    state.X_context = _apply(state.X_context, spec.feature_transform)
    if state.X_query is not None:
        state.X_query = _apply(state.X_query, spec.feature_transform)
    return state


def row_shuffle(state: ManipulationState, spec: ManipulationSpec) -> ManipulationState:
    """Shuffle context row order."""
    if spec.row_shuffle_seed is None:
        return state
    rng = np.random.default_rng(spec.row_shuffle_seed)
    idx = rng.permutation(len(state.X_context))
    state.X_context = state.X_context[idx]
    state.y_context = state.y_context[idx]
    return state


def subsample_rows(state: ManipulationState, spec: ManipulationSpec) -> ManipulationState:
    if spec.context_size is None or spec.context_size >= len(state.X_context):
        return state
    rng = np.random.default_rng(spec.subsample_seed)
    idx = rng.choice(len(state.X_context), size=spec.context_size, replace=False)
    state.X_context = state.X_context[idx]
    state.y_context = state.y_context[idx]
    return state


def subsample_columns(state: ManipulationState, spec: ManipulationSpec) -> ManipulationState:
    if spec.column_count is None or spec.column_count >= state.X_context.shape[1]:
        return state
    col_rng = np.random.default_rng(spec.column_seed)
    col_indices = sorted(
        col_rng.choice(state.X_context.shape[1], size=spec.column_count, replace=False)
    )
    state.X_context = state.X_context[:, col_indices]
    if state.X_query is not None:
        state.X_query = state.X_query[:, col_indices]
    state.col_indices = col_indices
    return state


def inject_noise(state: ManipulationState, spec: ManipulationSpec) -> ManipulationState:
    if spec.noise_level <= 0:
        return state
    noise_rng = np.random.default_rng(spec.noise_seed)
    feature_std = np.std(state.X_context, axis=0) + 1e-10
    noise = noise_rng.standard_normal(state.X_context.shape) * feature_std[np.newaxis, :]
    state.X_context = state.X_context + spec.noise_level * noise
    return state


def label_permutation(state: ManipulationState, spec: ManipulationSpec) -> ManipulationState:
    """Randomly permute the class label encoding (e.g. 0→2, 1→0, 2→1)."""
    if spec.label_permutation_seed is None:
        return state
    rng = np.random.default_rng(spec.label_permutation_seed)
    unique_labels = np.unique(state.y_context)
    if len(unique_labels) < 2:
        return state
    perm = rng.permutation(unique_labels)
    label_map = dict(zip(unique_labels, perm))
    state.y_context = np.array([label_map.get(v, v) for v in state.y_context])
    if state.y_query is not None:
        state.y_query = np.array([label_map.get(v, v) for v in state.y_query])
    return state


def constant_labels(state: ManipulationState, spec: ManipulationSpec) -> ManipulationState:
    """Set all context labels to the most common value."""
    if not spec.constant_label:
        return state
    if len(state.y_context) == 0:
        return state
    # Use the most common label as the constant
    counts = Counter(state.y_context)
    most_common = counts.most_common(1)[0][0]
    state.y_context = np.full_like(state.y_context, most_common)
    return state


def mislabel(state: ManipulationState, spec: ManipulationSpec) -> ManipulationState:
    if spec.mislabel_fraction <= 0:
        return state
    mis_rng = np.random.default_rng(spec.mislabel_seed)
    n_poison = int(len(state.y_context) * spec.mislabel_fraction)
    if n_poison <= 0:
        return state
    poison_idx = mis_rng.choice(len(state.y_context), size=n_poison, replace=False)
    strategy_fn = MISLABEL_STRATEGIES[spec.mislabel_strategy]
    state.y_context = strategy_fn(state.y_context, state.X_context, poison_idx, mis_rng)
    return state


def row_duplication(state: ManipulationState, spec: ManipulationSpec) -> ManipulationState:
    """Duplicate a fraction of context rows."""
    if spec.row_duplication_fraction <= 0:
        return state
    rng = np.random.default_rng(spec.row_duplication_seed)
    n_dup = max(1, int(len(state.X_context) * spec.row_duplication_fraction))
    dup_idx = rng.choice(len(state.X_context), size=n_dup, replace=True)
    state.X_context = np.concatenate([state.X_context, state.X_context[dup_idx]], axis=0)
    state.y_context = np.concatenate([state.y_context, state.y_context[dup_idx]])
    return state


def query_leakage(state: ManipulationState, spec: ManipulationSpec) -> ManipulationState:
    """Copy a fraction of query rows (with correct labels) into context."""
    if spec.query_leakage_fraction <= 0:
        return state
    if state.X_query is None or state.y_query is None:
        return state
    rng = np.random.default_rng(spec.query_leakage_seed)
    # Only leak query rows whose label already exists in the (post-cap) context.
    # A leaked row carrying a class the context lacks would widen the model's
    # prob vector (num_classes = #unique context labels), so the manipulated
    # prediction and the clean reference would have different widths and
    # kl_to_clean would silently return None. Because the cap only bites tables
    # larger than the budget (i.e. members), that NaN was biased toward members.
    ctx_labels = np.unique(state.y_context)
    eligible = np.nonzero(np.isin(state.y_query, ctx_labels))[0]
    if len(eligible) == 0:
        return state
    # Dose is a fraction of the (already-capped) CONTEXT, not the query, so the
    # injected fraction is table-size-independent and matches truth_serum /
    # row_duplication. Capped at the number of eligible query rows.
    n_leak = max(1, int(len(state.X_context) * spec.query_leakage_fraction))
    n_leak = min(n_leak, len(eligible))
    leak_idx = rng.choice(eligible, size=n_leak, replace=False)
    state.X_context = np.concatenate([state.X_context, state.X_query[leak_idx]], axis=0)
    state.y_context = np.concatenate([state.y_context, state.y_query[leak_idx]])
    return state


def truth_serum(state: ManipulationState, spec: ManipulationSpec) -> ManipulationState:
    """Duplicate a fraction of context rows but with perturbed labels."""
    if spec.truth_serum_fraction <= 0:
        return state
    rng = np.random.default_rng(spec.truth_serum_seed)
    n_dup = max(1, int(len(state.X_context) * spec.truth_serum_fraction))
    dup_idx = rng.choice(len(state.X_context), size=n_dup, replace=True)
    X_dup = state.X_context[dup_idx].copy()
    y_dup = state.y_context[dup_idx].copy()
    # Perturb labels: assign random different label
    unique_labels = np.unique(state.y_context)
    if len(unique_labels) > 1:
        for i in range(len(y_dup)):
            others = unique_labels[unique_labels != y_dup[i]]
            y_dup[i] = rng.choice(others)
    state.X_context = np.concatenate([state.X_context, X_dup], axis=0)
    state.y_context = np.concatenate([state.y_context, y_dup])
    return state


def brainwash(state: ManipulationState, spec: ManipulationSpec) -> ManipulationState:
    """Copy a fraction of query rows into context but with perturbed labels."""
    if spec.brainwash_fraction <= 0:
        return state
    if state.X_query is None or state.y_query is None:
        return state
    rng = np.random.default_rng(spec.brainwash_seed)
    # Dose is a fraction of the (already-capped) CONTEXT, not the query, so the
    # injected fraction is table-size-independent and matches truth_serum.
    # Capped at the number of available query rows.
    n_bw = max(1, int(len(state.X_context) * spec.brainwash_fraction))
    n_bw = min(n_bw, len(state.X_query))
    bw_idx = rng.choice(len(state.X_query), size=n_bw, replace=False)
    X_bw = state.X_query[bw_idx].copy()
    y_bw = state.y_query[bw_idx].copy()
    # Perturb labels
    unique_labels = np.unique(state.y_context)
    if len(unique_labels) > 1:
        for i in range(len(y_bw)):
            others = unique_labels[unique_labels != y_bw[i]]
            y_bw[i] = rng.choice(others)
    state.X_context = np.concatenate([state.X_context, X_bw], axis=0)
    state.y_context = np.concatenate([state.y_context, y_bw])
    return state


def context_equals_query(state: ManipulationState, spec: ManipulationSpec) -> ManipulationState:
    """Set query data to be the same as context data."""
    if not spec.context_equals_query:
        return state
    state.X_query = state.X_context.copy()
    state.y_query = state.y_context.copy()
    return state


# ---------------------------------------------------------------------------
# Mislabel sub-strategies
# ---------------------------------------------------------------------------

def _mislabel_random(
    y_ctx: np.ndarray, X_ctx: np.ndarray, poison_idx: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    unique_labels = np.unique(y_ctx)
    for i in poison_idx:
        others = unique_labels[unique_labels != y_ctx[i]]
        if len(others) > 0:
            y_ctx[i] = rng.choice(others)
    return y_ctx


def _mislabel_swap(
    y_ctx: np.ndarray, X_ctx: np.ndarray, poison_idx: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    counts = Counter(y_ctx)
    top2 = [c for c, _ in counts.most_common(2)]
    if len(top2) == 2:
        c1, c2 = top2
        for i in poison_idx:
            if y_ctx[i] == c1:
                y_ctx[i] = c2
            elif y_ctx[i] == c2:
                y_ctx[i] = c1
    return y_ctx


def _mislabel_boundary(
    y_ctx: np.ndarray, X_ctx: np.ndarray, poison_idx: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    unique_labels = np.unique(y_ctx)
    nn = NearestNeighbors(n_neighbors=min(5, len(X_ctx)))
    nn.fit(X_ctx)
    _dists, nbr_idx = nn.kneighbors(X_ctx)
    boundary_scores = np.array(
        [np.mean(y_ctx[nbr_idx[i, 1:]] != y_ctx[i]) for i in range(len(y_ctx))]
    )
    # Override poison_idx: pick top-n by boundary score
    n_poison = len(poison_idx)
    poison_idx = np.argsort(boundary_scores)[-n_poison:]
    for i in poison_idx:
        others = unique_labels[unique_labels != y_ctx[i]]
        if len(others) > 0:
            y_ctx[i] = rng.choice(others)
    return y_ctx


MISLABEL_STRATEGIES: dict[str, MislabelStrategyFn] = {
    "random": _mislabel_random,
    "swap": _mislabel_swap,
    "boundary": _mislabel_boundary,
}


# ---------------------------------------------------------------------------
# Ordered pipeline & executor
# ---------------------------------------------------------------------------

MANIPULATION_PIPELINE: list[tuple[str, ManipulationFn]] = [
    ("task_shuffle", task_shuffle),
    ("feature_transform", feature_transform),
    ("row_shuffle", row_shuffle),
    ("subsample_rows", subsample_rows),
    ("subsample_columns", subsample_columns),
    ("inject_noise", inject_noise),
    ("label_permutation", label_permutation),
    ("constant_labels", constant_labels),
    ("mislabel", mislabel),
    ("row_duplication", row_duplication),
    ("query_leakage", query_leakage),
    ("truth_serum", truth_serum),
    ("brainwash", brainwash),
    ("context_equals_query", context_equals_query),
]


def apply_manipulation(
    X_context: np.ndarray,
    y_context: np.ndarray,
    spec: ManipulationSpec,
    X_query: np.ndarray | None = None,
    y_query: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, list[int] | None]:
    """Apply manipulation pipeline to context data.

    Returns (X_ctx, y_ctx, X_query_manip, y_query_manip, col_indices).
    """
    state = ManipulationState(
        X_context.copy(),
        y_context.copy(),
        X_query.copy() if X_query is not None else None,
        y_query.copy() if y_query is not None else None,
    )
    for _, fn in MANIPULATION_PIPELINE:
        state = fn(state, spec)
    return state.X_context, state.y_context, state.X_query, state.y_query, state.col_indices
