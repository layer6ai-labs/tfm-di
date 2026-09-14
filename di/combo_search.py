# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "scikit-learn",
#     "scipy>=1.17.0",
# ]
# ///
"""Exhaustive signal combination search with vectorized LOOCV logistic regression.

Key optimizations over naive approach:
1. All n LOO folds run simultaneously via 3D tensor ops (no Python loop over folds)
2. Full Newton (not diagonal) — (d+1)x(d+1) Hessian is tiny for d≤5, converges in ~15 iters
"""

import argparse
import json
import os
import sys
import time
from glob import glob
from itertools import combinations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# Shared blind-degeneracy denylist. This script runs standalone via
# `uv run di/combo_search.py`, so add the repo root to sys.path for the import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from di.degenerate_signals import is_blind_degenerate


def loocv_predict(col_indices, X_loo, y_loo, sw_loo, X_test, C=1.0, n_steps=15):
    """Vectorized LOOCV: all n folds solved simultaneously with full Newton.

    Returns predicted P(member) for each held-out sample.

    Args:
        col_indices: tuple of column indices into X_loo's feature dim
        X_loo: (n, n-1, D+1) — precomputed LOO features with bias column appended
        y_loo: (n, n-1) — precomputed LOO labels
        sw_loo: (n, n-1) — precomputed LOO sample weights
        X_test: (n, D+1) — full feature matrix with bias column
        C: regularization (higher = less reg)
        n_steps: Newton iterations (15 is plenty for full Newton with d≤6)
    """
    n = X_test.shape[0]
    bias_col = X_loo.shape[2] - 1
    use_cols = list(col_indices) + [bias_col]
    d1 = len(use_cols)  # d + 1 (includes bias)

    X_b = X_loo[:, :, use_cols]  # (n, n-1, d+1)
    X_t = X_test[:, use_cols]  # (n, d+1)

    w = np.zeros((n, d1))  # (n, d+1)
    lam = 1.0 / C
    pen = lam * np.eye(d1)
    pen[-1, -1] = 0.0  # don't penalize intercept
    pen_diag = np.diag(pen)  # (d+1,) for gradient

    for _ in range(n_steps):
        # Forward pass — batched across all n folds
        z = (X_b @ w[:, :, None]).squeeze(-1)  # (n, n-1)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        r = sw_loo * p * (1 - p)  # (n, n-1)
        residual = sw_loo * (p - y_loo)  # (n, n-1)

        # Gradient: (n, d+1)
        grad = (X_b.transpose(0, 2, 1) @ residual[:, :, None]).squeeze(
            -1
        ) + pen_diag * w

        # Full Hessian: (n, d+1, d+1)
        X_b_r = X_b * r[:, :, None]  # (n, n-1, d+1)
        H = X_b_r.transpose(0, 2, 1) @ X_b + pen  # (n, d+1, d+1)

        # Newton step
        w -= np.linalg.solve(H, grad[:, :, None]).squeeze(-1)

    # Predict held-out sample for each fold
    z_test = np.sum(X_t * w, axis=1)  # (n,)
    return 1.0 / (1.0 + np.exp(-np.clip(z_test, -30, 30)))


def score_predictions(y, probs, metric):
    """Score LOOCV predictions with the chosen metric.

    metric: 'auc' or 'tpr@X' where X is the target FPR percentage (e.g. 'tpr@1', 'tpr@5').
    """
    if metric == "auc":
        return roc_auc_score(y, probs)
    # tpr@X — interpolate TPR at X% FPR
    target_fpr = float(metric.split("@")[1]) / 100.0
    fpr, tpr, _ = roc_curve(y, probs)
    return float(np.interp(target_fpr, fpr, tpr))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Exhaustive signal combination search")
    parser.add_argument("results", nargs="?", default=None, help="Results JSON file")
    parser.add_argument(
        "--metric",
        default="auc",
        help="Metric to optimize: 'auc' (default), 'tpr@1', 'tpr@5', 'tpr@10'",
    )
    parser.add_argument(
        "--engineer",
        action="store_true",
        help="Add engineered features (log, percentile indicators, rank transform)",
    )
    args = parser.parse_args()

    metric = args.metric.lower()
    if metric != "auc" and not metric.startswith("tpr@"):
        print(f"Unknown metric: {metric}. Use 'auc' or 'tpr@X' (e.g. 'tpr@1')")
        sys.exit(1)
    metric_label = metric.upper()

    path = args.results
    if path is None:
        files = sorted(glob("di_extended_results_*.json"))
        if not files:
            print("No results files found")
            sys.exit(1)
        path = files[-1]

    print(f"Loading {path}")
    print(f"Optimizing for: {metric_label}")
    with open(path) as f:
        data = json.load(f)

    results = [r for r in data["successful_results"] if r.get("success")]
    labels = np.array([r["true_label"] for r in results], dtype=np.float64)
    n = len(labels)
    print(
        f"  {n} datasets ({int(labels.sum())} members, {int((1 - labels).sum())} non-members)"
    )

    # Auto-discover numeric signal fields from results
    METADATA_KEYS = {
        "dataset_id", "dataset_name", "is_member", "true_label",
        "source_is_member", "iid_split", "iid_member_partition",
        "iid_split_seed", "model_backend", "task_type", "n_pool",
        "n_features", "grid_version", "success", "_curves", "_metadata",
    }

    # Collect all numeric signal names present in any result. Exclude the
    # blind-degenerate families (seed_*, the six temp argmax-dispersion signals,
    # and dataset-level logit_* geometry): they are not comparable to the TabPFN
    # blind and would let the search "win" on a structural artifact rather than
    # memorization. See di/degenerate_signals.py.
    all_signal_names = set()
    excluded_degenerate = set()
    for r in results:
        for k, v in r.items():
            if k not in METADATA_KEYS and isinstance(v, (int, float)):
                if is_blind_degenerate(k):
                    excluded_degenerate.add(k)
                else:
                    all_signal_names.add(k)
    all_signal_names = sorted(all_signal_names)
    if excluded_degenerate:
        print(
            f"  Excluded {len(excluded_degenerate)} blind-degenerate signals "
            f"(seed_*/temp argmax-dispersion/logit_*)"
        )

    # Build feature matrix (no direction flipping — let the classifier learn signs)
    feature_matrix = {}
    for sig in all_signal_names:
        vals = []
        for r in results:
            v = r.get(sig)
            vals.append(float(v) if v is not None and not (isinstance(v, float) and (v != v)) else np.nan)
        feature_matrix[sig] = np.array(vals)

    # Per-task availability: many signals are defined for only one task type
    # (e.g. reverse-KL `*_rkl2clean_*` is all-NaN on regression, `*_neg_mse_*`
    # is all-NaN on classification). Pooling them across a mixed cls+reg pool
    # would force cross-task imputation of a structurally-absent signal.
    task_types = np.array([str(r.get("task_type", "classification")) for r in results])
    present_tasks = sorted(set(task_types))

    # Drop signals that are all-NaN/constant, have <5 non-NaN values, or are
    # all-NaN within any task type present in the pool.
    sig_names = []
    dropped = 0
    dropped_pertask = 0
    for sig in all_signal_names:
        vals = feature_matrix[sig]
        finite = ~np.isnan(vals)
        if any(finite[task_types == tt].sum() == 0 for tt in present_tasks):
            dropped += 1
            dropped_pertask += 1
            continue
        if np.sum(finite) > 5 and np.nanstd(vals) > 1e-12:
            sig_names.append(sig)
        else:
            dropped += 1
    extra = f", {dropped_pertask} all-NaN within a task type" if len(present_tasks) > 1 else ""
    print(f"  {len(all_signal_names)} signals found, {dropped} dropped (NaN/constant{extra}), {len(sig_names)} usable")

    # --- Feature engineering ---
    if args.engineer:
        from scipy.stats import kurtosis as _kurtosis, rankdata

        print("\n  Engineering features...")
        engineered = {}
        for sig in list(sig_names):
            vals = feature_matrix[sig]
            valid = vals[~np.isnan(vals)]
            if len(valid) < 10:
                continue

            # Log transform: sign(x) * log(1 + |x|) — useful for heavy-tailed signals
            kurt = float(_kurtosis(valid, fisher=True, nan_policy="omit"))
            if kurt > 5:
                log_vals = np.where(
                    np.isnan(vals), np.nan, np.sign(vals) * np.log1p(np.abs(vals))
                )
                name = f"{sig}__log"
                engineered[name] = log_vals
                print(f"    {name:<36} (kurtosis={kurt:.1f})")

            # Percentile indicator: 1 if above 90th percentile of the signal
            p90 = np.nanpercentile(vals, 90)
            ind_vals = np.where(np.isnan(vals), np.nan, (vals > p90).astype(float))
            # Only add if indicator has variance (not all 0 or all 1)
            if np.nanstd(ind_vals) > 0.05:
                name = f"{sig}__p90"
                engineered[name] = ind_vals
                print(f"    {name:<36} (threshold={p90:.4g})")

            # Rank transform: maps to [0, 1] — robust to any distribution shape
            rank_vals = np.full_like(vals, np.nan)
            mask = ~np.isnan(vals)
            if mask.sum() > 10:
                rank_vals[mask] = rankdata(vals[mask]) / mask.sum()
                name = f"{sig}__rank"
                engineered[name] = rank_vals

        # Add engineered features to the pool
        for name, vals in engineered.items():
            feature_matrix[name] = vals
            sig_names.append(name)
        print(f"    Total features: {len(sig_names)} ({len(engineered)} engineered)")

    # Individual signal scores
    print(f"\n=== Individual Signal Scores ({metric_label}) ===")
    individual = {}
    for sig in sig_names:
        vals = feature_matrix[sig]
        mask = ~np.isnan(vals)
        if mask.sum() > 10 and len(set(labels[mask])) == 2:
            individual[sig] = score_predictions(labels[mask], vals[mask], metric)

    sorted_sigs = sorted(individual.items(), key=lambda x: -x[1])
    for sig, score in sorted_sigs[:20]:
        print(f"  {sig:<36} {metric_label}={score:.4f}")

    # Combo search setup — select top signals by individual AUC
    TOP_N = min(20 if args.engineer else 15, len(sorted_sigs))
    MAX_K = 5
    top_sigs = [s for s, _ in sorted_sigs[:TOP_N]]

    print(
        f"\n=== Exhaustive Combination Search (top {TOP_N}, k=1..{MAX_K}, metric={metric_label}) ==="
    )

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    all_cols = np.column_stack([feature_matrix[s] for s in top_sigs])
    all_cols_imp = SimpleImputer(strategy="median").fit_transform(all_cols)
    all_cols_scaled = StandardScaler().fit_transform(all_cols_imp)

    # Precompute LOO tensors (all folds at once)
    loo_idx = np.array(
        [np.concatenate([np.arange(i), np.arange(i + 1, n)]) for i in range(n)]
    )

    # Append bias column to features
    all_with_bias = np.hstack([all_cols_scaled, np.ones((n, 1))])  # (n, TOP_N+1)

    X_loo = all_with_bias[loo_idx]  # (n, n-1, TOP_N+1)
    y_loo = labels[loo_idx]  # (n, n-1)

    # Balanced sample weights
    n_pos, n_neg = labels.sum(), n - labels.sum()
    sw = np.where(labels == 1, n / (2 * n_pos), n / (2 * n_neg))
    sw_loo = sw[loo_idx]  # (n, n-1)

    # Benchmark
    print("  Benchmarking single LOOCV...", end=" ", flush=True)
    t0 = time.time()
    _probs = loocv_predict((0,), X_loo, y_loo, sw_loo, all_with_bias)
    _ = score_predictions(labels, _probs, metric)
    t_single = time.time() - t0
    total_combos = sum(
        len(list(combinations(range(TOP_N), k))) for k in range(1, MAX_K + 1)
    )
    est = t_single * total_combos
    print(f"{t_single:.4f}s/combo, {total_combos} combos, ETA ~{est:.1f}s")

    # Exhaustive search
    searched = 0
    t_start = time.time()
    best_per_k = {}

    for k in range(1, MAX_K + 1):
        best_score = -1.0
        best_combo = None
        top5 = []

        for col_indices in combinations(range(TOP_N), k):
            try:
                probs = loocv_predict(col_indices, X_loo, y_loo, sw_loo, all_with_bias)
                s = score_predictions(labels, probs, metric)
            except Exception:
                continue
            searched += 1
            combo_names = tuple(top_sigs[i] for i in col_indices)
            top5.append((s, combo_names))
            top5.sort(key=lambda x: -x[0])
            top5 = top5[:5]
            if s > best_score:
                best_score = s
                best_combo = combo_names

        best_per_k[k] = (best_score, best_combo)
        elapsed = time.time() - t_start
        if best_combo:
            print(
                f"\n  k={k}: best {metric_label}={best_score:.4f}  signals={best_combo}  ({elapsed:.1f}s)"
            )
            for rank, (sc, c) in enumerate(top5):
                print(f"    #{rank + 1}: {metric_label}={sc:.4f}  {c}")

    print(f"\n  Searched {searched} combinations in {time.time() - t_start:.1f}s")

    print(f"\n=== Summary: Best Per k ({metric_label}) ===")
    for k in range(1, MAX_K + 1):
        sc, combo = best_per_k[k]
        if combo:
            print(f"  k={k}: {metric_label}={sc:.4f}  {combo}")

    best_k = max(best_per_k, key=lambda kk: best_per_k[kk][0])
    best_score, best_combo = best_per_k[best_k]
    print(f"\n=== Best Overall ===")
    print(f"  k={best_k}: {metric_label}={best_score:.4f}  signals={best_combo}")


if __name__ == "__main__":
    main()
