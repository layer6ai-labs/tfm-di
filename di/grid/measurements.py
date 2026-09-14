"""Measurement extraction: probs/logits/preds + y_true -> scalars.

Registry pattern: each measurement is a function that takes prediction arrays
and ground truth, returning a float or None.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import chisquare as scipy_chisquare
from scipy.stats import entropy as scipy_entropy
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import wasserstein_distance as scipy_wasserstein


MeasurementFn = Callable[[dict[str, np.ndarray], np.ndarray], float | None]

# ── Classification measurements ─────────────────────────────────────────

def accuracy(arrays: dict[str, np.ndarray], y_true: np.ndarray) -> float | None:
    preds = arrays.get("preds")
    if preds is None or len(preds) == 0:
        return None
    return float(np.mean(preds == y_true))


def max_confidence(arrays: dict[str, np.ndarray], y_true: np.ndarray) -> float | None:
    probs = arrays.get("probs")
    if probs is None or len(probs) == 0:
        return None
    return float(np.mean(np.max(probs, axis=1)))


def correct_confidence(
    arrays: dict[str, np.ndarray], y_true: np.ndarray
) -> float | None:
    probs = arrays.get("probs")
    if probs is None or len(probs) == 0:
        return None
    n_classes = probs.shape[1]
    y_int = y_true.astype(int)
    if y_int.max() < n_classes and y_int.min() >= 0:
        return float(np.mean(probs[np.arange(len(y_int)), y_int]))
    return float(np.mean(np.max(probs, axis=1)))


def prediction_entropy(
    arrays: dict[str, np.ndarray], y_true: np.ndarray
) -> float | None:
    probs = arrays.get("probs")
    if probs is None or len(probs) == 0:
        return None
    eps = 1e-10
    ent = -np.sum(probs * np.log(probs + eps), axis=1)
    return float(np.mean(ent))


def loss(arrays: dict[str, np.ndarray], y_true: np.ndarray) -> float | None:
    probs = arrays.get("probs")
    if probs is None or len(probs) == 0:
        return None
    n_classes = probs.shape[1]
    y_int = y_true.astype(int)
    if y_int.max() >= n_classes or y_int.min() < 0:
        return None
    eps = 1e-10
    correct_probs = probs[np.arange(len(y_int)), y_int]
    return float(-np.mean(np.log(correct_probs + eps)))


def chi_squared(arrays: dict[str, np.ndarray], y_true: np.ndarray) -> float | None:
    """Chi-squared statistic: predicted class counts vs true class counts."""
    preds = arrays.get("preds")
    if preds is None or len(preds) == 0:
        return None
    n = len(y_true)
    n_classes = int(max(y_true.max(), preds.max())) + 1
    observed = np.bincount(preds.astype(int), minlength=n_classes).astype(float)
    expected = np.bincount(y_true.astype(int), minlength=n_classes).astype(float)
    # Union support: include classes the model predicts but that are absent from
    # the truth (dropping them understated the mismatch). Compute the statistic
    # manually (avoids scipy's observed==expected sum constraint once we add the
    # predicted-only bins) and normalize by N so it measures distribution SHAPE
    # rather than acting as a dataset-size / class-count proxy.
    mask = (observed > 0) | (expected > 0)
    if mask.sum() < 2:
        return None
    obs, exp = observed[mask], expected[mask]
    exp = exp * (obs.sum() / exp.sum())
    exp = np.clip(exp, 1e-10, None)
    stat = float(np.sum((obs - exp) ** 2 / exp))
    return stat / n


def kl_divergence(arrays: dict[str, np.ndarray], y_true: np.ndarray) -> float | None:
    """KL(true class proportions || mean predicted probabilities)."""
    probs = arrays.get("probs")
    if probs is None or len(probs) == 0:
        return None
    n_classes = probs.shape[1]
    y_int = y_true.astype(int)
    if y_int.max() >= n_classes or y_int.min() < 0:
        return None
    p_true = np.bincount(y_int, minlength=n_classes).astype(float)
    p_true /= p_true.sum()
    p_pred = probs.mean(axis=0)
    eps = 1e-10
    return float(scipy_entropy(p_true + eps, p_pred + eps))


def wasserstein(arrays: dict[str, np.ndarray], y_true: np.ndarray) -> float | None:
    """Wasserstein distance between predicted and true class distributions."""
    preds = arrays.get("preds")
    if preds is None or len(preds) == 0:
        return None
    return float(scipy_wasserstein(y_true.astype(float), preds.astype(float)))


def prediction_margin(arrays: dict[str, np.ndarray], y_true: np.ndarray) -> float | None:
    """Difference between highest and second-highest predicted probability."""
    probs = arrays.get("probs")
    if probs is None or len(probs) == 0 or probs.shape[1] < 2:
        return None
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
    return float(np.mean(margins))


def logit_margin(arrays: dict[str, np.ndarray], y_true: np.ndarray) -> float | None:
    logits = arrays.get("logits")
    if logits is None or len(logits) == 0:
        return None
    sorted_logits = np.sort(logits, axis=1)[:, ::-1]
    if sorted_logits.shape[1] < 2:
        return None
    margins = sorted_logits[:, 0] - sorted_logits[:, 1]
    return float(np.mean(margins))


def logit_magnitude(
    arrays: dict[str, np.ndarray], y_true: np.ndarray
) -> float | None:
    logits = arrays.get("logits")
    if logits is None or len(logits) == 0:
        return None
    return float(np.mean(np.linalg.norm(logits, axis=1)))


def logit_kurtosis(
    arrays: dict[str, np.ndarray], y_true: np.ndarray
) -> float | None:
    logits = arrays.get("logits")
    if logits is None or len(logits) == 0:
        return None
    return float(scipy_kurtosis(logits.ravel(), fisher=True))


# ── Regression measurements ─────────────────────────────────────────────

def neg_mse(arrays: dict[str, np.ndarray], y_true: np.ndarray) -> float | None:
    preds = arrays.get("preds")
    if preds is None or len(preds) == 0:
        return None
    return float(-np.mean((preds - y_true) ** 2))


def r2(arrays: dict[str, np.ndarray], y_true: np.ndarray) -> float | None:
    preds = arrays.get("preds")
    if preds is None or len(preds) == 0:
        return None
    ss_res = np.sum((y_true - preds) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        return None
    return float(1 - ss_res / ss_tot)


def residual_kurtosis(
    arrays: dict[str, np.ndarray], y_true: np.ndarray
) -> float | None:
    preds = arrays.get("preds")
    if preds is None or len(preds) == 0:
        return None
    residuals = y_true - preds
    return float(scipy_kurtosis(residuals, fisher=True))


def residual_skewness(
    arrays: dict[str, np.ndarray], y_true: np.ndarray
) -> float | None:
    from scipy.stats import skew

    preds = arrays.get("preds")
    if preds is None or len(preds) == 0:
        return None
    residuals = y_true - preds
    return float(skew(residuals))


def pred_spread(arrays: dict[str, np.ndarray], y_true: np.ndarray) -> float | None:
    preds = arrays.get("preds")
    if preds is None or len(preds) == 0:
        return None
    std_pred = np.std(preds)
    std_true = np.std(y_true) + 1e-10
    return float(std_pred / std_true)


def _regression_histograms(
    preds: np.ndarray, y_true: np.ndarray, n_bins: int = 30,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Shared binning for regression distributional metrics.

    Keep bins where EITHER histogram has mass (union), so bins the model
    over-predicts into are not silently dropped.
    """
    combined = np.concatenate([y_true, preds])
    edges = np.histogram_bin_edges(combined, bins=n_bins)
    obs = np.histogram(preds, bins=edges)[0].astype(float)
    exp = np.histogram(y_true, bins=edges)[0].astype(float)
    mask = (obs > 0) | (exp > 0)
    if mask.sum() < 2:
        return None
    return obs[mask], exp[mask]


def reg_chi_squared(arrays: dict[str, np.ndarray], y_true: np.ndarray) -> float | None:
    """Normalized chi-squared statistic on binned predictions vs true values.

    Normalized by N so it measures distribution SHAPE, not dataset size.
    """
    preds = arrays.get("preds")
    if preds is None or len(preds) == 0:
        return None
    result = _regression_histograms(preds, y_true)
    if result is None:
        return None
    obs, exp = result
    exp = exp * (obs.sum() / exp.sum())
    exp = np.clip(exp, 1e-10, None)
    stat = float(np.sum((obs - exp) ** 2 / exp))
    return stat / len(y_true)


def reg_kl_divergence(arrays: dict[str, np.ndarray], y_true: np.ndarray) -> float | None:
    """KL(true || predicted) via kernel density estimation."""
    from scipy.stats import gaussian_kde

    preds = arrays.get("preds")
    if preds is None or len(preds) == 0:
        return None
    if np.std(y_true) < 1e-10 or np.std(preds) < 1e-10:
        return None
    kde_true = gaussian_kde(y_true)
    kde_pred = gaussian_kde(preds.astype(float))
    # Evaluate on a shared grid spanning both distributions
    lo = min(y_true.min(), preds.min())
    hi = max(y_true.max(), preds.max())
    grid = np.linspace(lo, hi, 500)
    p = kde_true(grid)
    q = kde_pred(grid)
    eps = 1e-10
    kl = np.trapz(p * np.log((p + eps) / (q + eps)), grid)
    return float(max(kl, 0.0))


def reg_wasserstein(arrays: dict[str, np.ndarray], y_true: np.ndarray) -> float | None:
    """Wasserstein distance between predictions and true values."""
    preds = arrays.get("preds")
    if preds is None or len(preds) == 0:
        return None
    return float(scipy_wasserstein(y_true.astype(float), preds.astype(float)))


# ── Registries ──────────────────────────────────────────────────────────

CLASSIFICATION_MEASUREMENTS: dict[str, MeasurementFn] = {
    "accuracy": accuracy,
    "max_confidence": max_confidence,
    "correct_confidence": correct_confidence,
    "prediction_margin": prediction_margin,
    "entropy": prediction_entropy,
    "loss": loss,
    "chi_squared": chi_squared,
    "kl_divergence": kl_divergence,
    "wasserstein": wasserstein,
}

LOGIT_MEASUREMENTS: dict[str, MeasurementFn] = {
    "logit_margin": logit_margin,
    "logit_magnitude": logit_magnitude,
    "logit_kurtosis": logit_kurtosis,
}

REGRESSION_MEASUREMENTS: dict[str, MeasurementFn] = {
    "neg_mse": neg_mse,
    "r2": r2,
    "residual_kurtosis": residual_kurtosis,
    "residual_skewness": residual_skewness,
    "pred_spread": pred_spread,
    "chi_squared": reg_chi_squared,
    "kl_divergence": reg_kl_divergence,
    "wasserstein": reg_wasserstein,
}


class MeasurementExtractor:
    """Extract all registered measurements from prediction arrays."""

    def __init__(self, task_type: str):
        self.task_type = task_type
        if task_type == "classification":
            self.measurements = dict(CLASSIFICATION_MEASUREMENTS)
        else:
            self.measurements = dict(REGRESSION_MEASUREMENTS)

    def extract(
        self, arrays: dict[str, np.ndarray], y_true: np.ndarray
    ) -> dict[str, float | None]:
        """Extract all measurements from one prediction output."""
        result = {}
        for name, fn in self.measurements.items():
            try:
                result[name] = fn(arrays, y_true)
            except Exception:
                result[name] = None
        return result

    def extract_logits(
        self, arrays: dict[str, np.ndarray], y_true: np.ndarray
    ) -> dict[str, float | None]:
        """Extract logit-specific measurements."""
        result = {}
        for name, fn in LOGIT_MEASUREMENTS.items():
            try:
                result[name] = fn(arrays, y_true)
            except Exception:
                result[name] = None
        return result


# ── Cross-prediction measurements (manipulated vs clean) ──────────────

def kl_to_clean(
    manip_arrays: dict[str, np.ndarray],
    clean_arrays: dict[str, np.ndarray],
) -> float | None:
    """Mean per-sample KL(manip || clean) for classification probs."""
    p = manip_arrays.get("probs")
    q = clean_arrays.get("probs")
    if p is None or q is None or len(p) == 0 or len(q) == 0:
        return None
    if p.shape != q.shape:
        return None
    eps = 1e-10
    kl = np.sum(p * np.log((p + eps) / (q + eps)), axis=1)
    return float(np.mean(kl))


def reverse_kl_to_clean(
    manip_arrays: dict[str, np.ndarray],
    clean_arrays: dict[str, np.ndarray],
) -> float | None:
    """Mean per-sample KL(clean || manip) for classification probs."""
    p = clean_arrays.get("probs")
    q = manip_arrays.get("probs")
    if p is None or q is None or len(p) == 0 or len(q) == 0:
        return None
    if p.shape != q.shape:
        return None
    eps = 1e-10
    kl = np.sum(p * np.log((p + eps) / (q + eps)), axis=1)
    return float(np.mean(kl))


def regression_kl_to_clean(
    manip_arrays: dict[str, np.ndarray],
    clean_arrays: dict[str, np.ndarray],
) -> float | None:
    """MSE between manipulated and clean predictions (regression proxy for KL)."""
    p = manip_arrays.get("preds")
    q = clean_arrays.get("preds")
    if p is None or q is None or len(p) == 0 or len(q) == 0:
        return None
    if len(p) != len(q):
        return None
    return float(np.mean((p - q) ** 2))
