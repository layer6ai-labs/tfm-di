"""Trajectory and aggregate computation from measurement sequences.

Trajectories: measurement vs manipulation_intensity -> curve features.
Aggregates: statistic across manipulation variants -> summary features.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def compute_trajectory(
    intensities: list[float],
    values: list[float | None],
    prefix: str,
) -> dict[str, float | None]:
    """Compute trajectory features from (intensity, value) pairs.

    Returns: {prefix}_slope, {prefix}_auc, {prefix}_gain, {prefix}_early, {prefix}_late_gain
    """
    # Filter out None AND non-finite (NaN/Inf) values. Measurements such as
    # residual_kurtosis / reg_chi_squared can return NaN (not None) on degenerate
    # inputs; if such a point survived, np.polyfit / np.trapezoid propagate NaN to
    # EVERY trajectory feature of that measurement, which is then median-imputed at
    # the meta-classifier — silently replacing a real signal with the pooled median.
    pairs = [
        (x, v) for x, v in zip(intensities, values)
        if v is not None and np.isfinite(v)
    ]
    if len(pairs) < 2:
        return {
            f"{prefix}_slope": None,
            f"{prefix}_auc": None,
            f"{prefix}_gain": None,
            f"{prefix}_early": None,
            f"{prefix}_late_gain": None,
        }

    xs = np.array([p[0] for p in pairs])
    vs = np.array([p[1] for p in pairs])

    # Use log-scale for context_size / column_count (positive integer intensities)
    if all(x > 0 for x in xs) and xs.max() / xs.min() > 4:
        xs_fit = np.log2(xs)
    else:
        xs_fit = xs

    # Slope: linear regression
    slope, _ = np.polyfit(xs_fit, vs, 1)

    # AUC: normalized trapezoid
    auc_raw = float(np.trapezoid(vs, xs_fit))
    x_range = xs_fit[-1] - xs_fit[0]
    auc = auc_raw / x_range if x_range > 0 else auc_raw

    # Gain: last - first
    gain = float(vs[-1] - vs[0])

    # Early: value at ~25% point relative to max
    max_val = np.max(np.abs(vs)) if np.max(np.abs(vs)) > 0 else 1.0
    early_idx = max(0, int(np.ceil(0.25 * (len(vs) - 1))))
    early = float(vs[early_idx] / max_val) if max_val > 1e-10 else None

    # Late gain: last - second to last
    late_gain = float(vs[-1] - vs[-2])

    return {
        f"{prefix}_slope": float(slope),
        f"{prefix}_auc": float(auc),
        f"{prefix}_gain": gain,
        f"{prefix}_early": early,
        f"{prefix}_late_gain": late_gain,
    }


def compute_aggregates(
    values: list[float | None],
    prefix: str,
) -> dict[str, float | None]:
    """Compute aggregate statistics across manipulation variants.

    Returns: {prefix}_std, {prefix}_range, {prefix}_mean
    """
    clean = [v for v in values if v is not None and np.isfinite(v)]
    if len(clean) < 2:
        return {
            f"{prefix}_std": None,
            f"{prefix}_range": None,
            f"{prefix}_mean": None,
        }

    arr = np.array(clean)
    return {
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_range": float(np.ptp(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
    }


def compute_consistency(
    prediction_arrays: list[np.ndarray],
    task_type: str,
    prefix: str,
) -> dict[str, float | None]:
    """Compute prediction consistency across variants (seeds, temperatures).

    For classification: pairwise agreement rate.
    For regression: pairwise Spearman correlation.
    """
    if len(prediction_arrays) < 2:
        return {f"{prefix}_consistency": None, f"{prefix}_rank_corr": None}

    if task_type == "classification":
        # Pairwise agreement
        n_pairs = 0
        total_agreement = 0.0
        for i in range(len(prediction_arrays)):
            for j in range(i + 1, len(prediction_arrays)):
                if len(prediction_arrays[i]) > 0 and len(prediction_arrays[j]) > 0:
                    total_agreement += float(
                        np.mean(prediction_arrays[i] == prediction_arrays[j])
                    )
                    n_pairs += 1
        consistency = total_agreement / n_pairs if n_pairs > 0 else None
    else:
        # Mean variance across predictions
        stacked = np.stack(prediction_arrays, axis=0)
        consistency = float(1.0 - np.mean(np.var(stacked, axis=0)))

    # Rank correlation (works for both cls and reg)
    correlations = []
    for i in range(len(prediction_arrays)):
        for j in range(i + 1, len(prediction_arrays)):
            if len(prediction_arrays[i]) > 0 and len(prediction_arrays[j]) > 0:
                rho, _ = spearmanr(prediction_arrays[i], prediction_arrays[j])
                if not np.isnan(rho):
                    correlations.append(rho)
    rank_corr = float(np.mean(correlations)) if correlations else None

    return {
        f"{prefix}_consistency": consistency,
        f"{prefix}_rank_corr": rank_corr,
    }
