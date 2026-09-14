"""Per-sample signal extraction from in-memory grid predictions.

``extract_row_signals`` turns the ``all_predictions`` dict that
``SignalGrid._collect_measurements`` already builds into a flat mapping
``{column_name -> array(n_query,)}``. No GPU work, no disk I/O — every
array is a simple numpy reduction over data that is already in memory
alongside the existing dataset-level signals.
"""

from __future__ import annotations

import numpy as np

from .predictions import _encode_labels
from .specs import ManipulationSpec, PredictionSpec


# ── Per-prediction feature extraction ────────────────────────────────────


def _row_features(
    arrays: dict[str, np.ndarray],
    y_true: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Per-sample features from one prediction bundle.

    Classification: ``max_confidence``, ``prediction_margin``, ``entropy``,
    plus ``confidence``/``loss``/``correct`` when ``y_true`` is valid.
    Regression: ``prediction`` plus residual stats when ``y_true`` is valid.
    Logit features are added when ``logits`` is present.
    """
    feats: dict[str, np.ndarray] = {}
    probs = arrays.get("probs")
    preds = arrays.get("preds")
    logits = arrays.get("logits")

    if probs is not None and len(probs) > 0:
        n = len(probs)
        n_classes = probs.shape[1]

        max_conf = probs.max(axis=1)
        feats["max_confidence"] = max_conf.astype(np.float32)

        sorted_probs = np.sort(probs, axis=1)
        if n_classes >= 2:
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        else:
            margin = max_conf
        feats["prediction_margin"] = margin.astype(np.float32)

        log_probs = np.log(np.clip(probs, 1e-10, 1.0))
        feats["entropy"] = (-np.sum(probs * log_probs, axis=1)).astype(np.float32)

        if y_true is not None and len(y_true) == n:
            y_int = np.asarray(y_true, dtype=np.int64)
            if y_int.min() >= 0 and y_int.max() < n_classes:
                correct_prob = probs[np.arange(n), y_int]
                feats["confidence"] = correct_prob.astype(np.float32)
                feats["loss"] = (
                    -np.log(np.clip(correct_prob, 1e-10, 1.0))
                ).astype(np.float32)
                feats["correct"] = (
                    np.argmax(probs, axis=1) == y_int
                ).astype(np.float32)

    elif preds is not None and len(preds) > 0:
        preds_f = preds.astype(np.float32)
        feats["prediction"] = preds_f
        if y_true is not None and len(y_true) == len(preds_f):
            residual = preds_f - np.asarray(y_true, dtype=np.float32)
            feats["residual"] = residual
            feats["abs_residual"] = np.abs(residual)
            feats["sq_residual"] = residual * residual

    if logits is not None and len(logits) > 0:
        sorted_logits = np.sort(logits, axis=1)
        if sorted_logits.shape[1] >= 2:
            feats["logit_margin"] = (
                sorted_logits[:, -1] - sorted_logits[:, -2]
            ).astype(np.float32)
        feats["logit_norm"] = np.linalg.norm(logits, axis=1).astype(np.float32)

    return feats


# ── Per-sample aggregation helpers ───────────────────────────────────────


def _nan_array(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=np.float32)


def _valid_stacks(
    stacks: list[np.ndarray | None], n: int
) -> list[np.ndarray]:
    return [s for s in stacks if s is not None and len(s) == n]


def _aggregate_per_sample(
    stacks: list[np.ndarray | None], n: int
) -> dict[str, np.ndarray]:
    """Per-sample ``mean``, ``std``, ``range`` across unordered variants."""
    valid = _valid_stacks(stacks, n)
    if len(valid) < 2:
        return {
            "mean": _nan_array(n),
            "std": _nan_array(n),
            "range": _nan_array(n),
        }
    stacked = np.stack(valid, axis=0)  # (k, n)
    return {
        "mean": np.mean(stacked, axis=0).astype(np.float32),
        "std": np.std(stacked, axis=0).astype(np.float32),
        "range": np.ptp(stacked, axis=0).astype(np.float32),
    }


def _trajectory_per_sample(
    intensities: list[float],
    stacks: list[np.ndarray | None],
    n: int,
) -> dict[str, np.ndarray]:
    """Per-sample ``mean``, ``std``, ``range``, ``slope`` vs sweep intensity.

    Slope is a simple linear regression, log-scaled on x when the sweep
    spans more than 4x and is strictly positive (mirrors
    ``compute_trajectory`` in trajectories.py).
    """
    pairs = [
        (x, s)
        for x, s in zip(intensities, stacks)
        if s is not None and len(s) == n
    ]
    if len(pairs) < 2:
        return {
            "mean": _nan_array(n),
            "std": _nan_array(n),
            "range": _nan_array(n),
            "slope": _nan_array(n),
        }
    xs = np.array([p[0] for p in pairs], dtype=np.float64)
    stacked = np.stack([p[1] for p in pairs], axis=0).astype(np.float64)

    if xs.min() > 0 and xs.max() / xs.min() > 4:
        xs_fit = np.log2(xs)
    else:
        xs_fit = xs

    x_centered = xs_fit - xs_fit.mean()
    x_var = float((x_centered ** 2).sum())
    if x_var > 1e-12:
        y_centered = stacked - stacked.mean(axis=0, keepdims=True)
        slope = (x_centered[:, None] * y_centered).sum(axis=0) / x_var
    else:
        slope = np.full(n, np.nan, dtype=np.float64)

    return {
        "mean": stacked.mean(axis=0).astype(np.float32),
        "std": stacked.std(axis=0).astype(np.float32),
        "range": np.ptp(stacked, axis=0).astype(np.float32),
        "slope": slope.astype(np.float32),
    }


# ── Main entry point ────────────────────────────────────────────────────


def extract_row_signals(
    grid,
    all_predictions: dict[
        tuple[ManipulationSpec, PredictionSpec], dict[str, np.ndarray]
    ],
    y_query: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    """Flatten per-sample predictions into a dict of columns.

    Mirrors ``SignalGrid._derive_signals`` but operates on per-sample arrays
    instead of scalar measurements. Every returned array has length
    ``len(y_query)``. Missing / empty bundles become NaN-filled columns so
    the output schema is stable across datasets.
    """
    if task_type == "classification":
        y_query_enc, _ = _encode_labels(y_query)
    else:
        y_query_enc = np.asarray(y_query)

    n = len(y_query_enc)

    def _feats_for(manip: ManipulationSpec, pred: PredictionSpec) -> dict[str, np.ndarray]:
        arrays = all_predictions.get((manip, pred))
        if not arrays:
            return {}
        # Task-shuffle / context_equals_query inject a per-variant y_query.
        y_q = arrays.get("y_query")
        if y_q is not None and len(y_q) > 0:
            if task_type == "classification":
                y_enc, _ = _encode_labels(y_q)
            else:
                y_enc = np.asarray(y_q)
        else:
            y_enc = y_query_enc
        return _row_features(arrays, y_enc)

    out: dict[str, np.ndarray] = {}

    # ── One-shot bundles ────────────────────────────────────────────────
    for tag, manip, pred in grid._oneshots:
        feats = _feats_for(manip, pred)
        for fname, arr in feats.items():
            col = arr if len(arr) == n else _nan_array(n)
            out[f"{tag}_{fname}"] = col.astype(np.float32)

    # ── Trajectory sweeps: mean/std/range/slope per sample ──────────────
    for sweep_name, entries in grid._trajectory_sweeps.items():
        intensities = [e[0] for e in entries]
        per_entry_feats: list[dict[str, np.ndarray]] = []
        feat_names: set[str] = set()
        for _, manip, preds in entries:
            feats = _feats_for(manip, preds[0])
            per_entry_feats.append(feats)
            feat_names.update(feats.keys())
        for fname in sorted(feat_names):
            stacks = [f.get(fname) for f in per_entry_feats]
            aggs = _trajectory_per_sample(intensities, stacks, n)
            for stat_name, arr in aggs.items():
                out[f"{sweep_name}_{fname}_{stat_name}"] = arr

    # ── Aggregate sweeps: mean/std/range per sample ─────────────────────
    for sweep_name, entries in grid._aggregate_sweeps.items():
        per_entry_feats = []
        feat_names = set()
        for _, manip, preds in entries:
            feats = _feats_for(manip, preds[0])
            per_entry_feats.append(feats)
            feat_names.update(feats.keys())
        for fname in sorted(feat_names):
            stacks = [f.get(fname) for f in per_entry_feats]
            aggs = _aggregate_per_sample(stacks, n)
            for stat_name, arr in aggs.items():
                out[f"{sweep_name}_{fname}_{stat_name}"] = arr

        # Task-shuffle: fraction of variants that agree with the modal class
        if sweep_name == "tshuffle":
            pred_cols: list[np.ndarray] = []
            for _, manip, preds in entries:
                arrays = all_predictions.get((manip, preds[0]))
                if arrays is None:
                    continue
                p = arrays.get("preds")
                if p is not None and len(p) == n:
                    pred_cols.append(np.asarray(p))
            if len(pred_cols) >= 2:
                stacked = np.stack(pred_cols, axis=0)  # (k, n)
                k = stacked.shape[0]
                agreement = np.empty(n, dtype=np.float32)
                for i in range(n):
                    _, counts = np.unique(stacked[:, i], return_counts=True)
                    agreement[i] = counts.max() / k
                out["tshuffle_pred_agreement"] = agreement
            else:
                out["tshuffle_pred_agreement"] = _nan_array(n)

    return out
