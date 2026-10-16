# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scikit-learn",
#     "scipy",
#     "seaborn",
# ]
# ///
"""Evaluate multi-model DI: train meta-classifier on default model, apply to all seeds.

Usage:
    uv run eval_multi_model.py results_default.json results_seed42.json results_seed123.json results_seed456.json

The first file is treated as the training set (default model). A logistic regression
is trained on 3 signals (ent_slope, ctx_knn_vs_random_acc, temp_accuracy_std) using
all samples. The fitted classifier is then applied to each model's results to produce
ROC curves and AUC scores.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# ICML figure style (from agent-instructions/figure.md)
# ---------------------------------------------------------------------------
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "pink": "#CC79A7",
    "yellow": "#F0E442",
    "cyan": "#56B4E9",
    "red": "#D55E00",
    "black": "#000000",
}

FIG_DIR = Path("figures")

MODEL_COLORS = {
    "default": OKABE_ITO["blue"],
    "seed42": OKABE_ITO["orange"],
    "seed123": OKABE_ITO["green"],
    "seed456": OKABE_ITO["pink"],
}

MODEL_LINESTYLES = {
    "default": "-",
    "seed42": "--",
    "seed123": "-.",
    "seed456": ":",
}

SIGNAL_COMBOS = {
    "auc_opt": {
        "label": "AUC-optimal (k=3)",
        "signals": ["ent_slope", "ctx_knn_vs_random_acc", "temp_accuracy_std"],
        "suffix": "auc_opt",
        "engineered": False,
    },
    "tpr1_opt": {
        "label": "TPR@1%-optimal raw (k=5)",
        "signals": ["conf_late_gain", "conf_early", "ent_gain", "conf_gain", "logit_kurtosis"],
        "suffix": "tpr1_opt",
        "engineered": False,
    },
    "tpr1_eng": {
        "label": "TPR@1%-optimal engineered (k=3)",
        # Base signals needed; rank + p90 of conf_late_gain computed at extraction time
        "signals": ["conf_late_gain", "conf_early"],
        "suffix": "tpr1_eng",
        "engineered": True,
    },
}

# Default for transfer evaluation
SIGNALS = SIGNAL_COMBOS["auc_opt"]["signals"]


def extract_engineered(results: list[dict]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Extract the engineered feature set: conf_late_gain__rank, conf_late_gain__p90, conf_early.

    rank and p90 are computed from the provided results (dataset-level).
    Returns X (n, 3), y (n,), ids.
    """
    from scipy.stats import rankdata
    # First pass: collect raw values and filter missing
    raw, y_list, ids = [], [], []
    for r in results:
        clg = r.get("conf_late_gain")
        ce = r.get("conf_early")
        if clg is None or ce is None:
            continue
        raw.append((clg, ce))
        y_list.append(1 if r.get("is_member") else 0)
        ids.append(r.get("dataset_id", -1))
    if not raw:
        return np.array([]), np.array([]), []
    clg_arr = np.array([v[0] for v in raw])
    ce_arr = np.array([v[1] for v in raw])
    n = len(clg_arr)
    # Rank transform (higher rank = higher value) normalized to [0, 1]
    clg_rank = rankdata(clg_arr) / n
    # P90 indicator
    p90 = np.percentile(clg_arr, 90)
    clg_p90 = (clg_arr > p90).astype(float)
    X = np.column_stack([clg_rank, clg_p90, ce_arr])
    return X, np.array(y_list), ids


def loocv_probs(X: np.ndarray, y: np.ndarray, engineered: bool = False) -> np.ndarray:
    """LOOCV logistic regression: return predicted P(member) for each held-out sample.

    If engineered=True, recompute rank + p90 features on training fold only
    (columns 0=rank, 1=p90 of conf_late_gain; column 2=conf_early kept as-is).
    """
    from scipy.stats import rankdata
    n = len(y)
    probs = np.zeros(n)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        if engineered:
            # X[:, 0] is raw conf_late_gain rank (placeholder), X[:, 1] is p90 indicator,
            # X[:, 2] is conf_early. But we need the raw conf_late_gain to recompute.
            # We stored rank in col 0 — recover the raw ordering from the full rank.
            # Actually, it's simpler: pass raw values and engineer per fold.
            # But extract_engineered already computed rank/p90 on full data.
            # For proper LOO: recompute rank and p90 on training fold.
            # We need raw conf_late_gain — it's monotonically related to the rank,
            # so we can use the rank values as a proxy for re-ranking.
            # Re-rank the training fold values:
            X_tr = X[train_idx].copy()
            X_te = X[test_idx].copy()
            # Col 0 is rank (from full data) — re-rank within training fold
            train_ranks = rankdata(X_tr[:, 0]) / len(train_idx)
            X_tr[:, 0] = train_ranks
            # Re-rank test point relative to training
            test_val = X_te[0, 0]
            X_te[0, 0] = np.mean(train_ranks[X_tr[:, 0] <= test_val]) if np.any(X_tr[:, 0] <= test_val) else 0
            # Col 1 is p90 — recompute threshold on training fold
            p90_thresh = np.percentile(X_tr[:, 0], 90)
            X_tr[:, 1] = (X_tr[:, 0] > p90_thresh).astype(float)
            X_te[0, 1] = (X_te[0, 0] > p90_thresh).astype(float)
        else:
            X_tr = X[train_idx]
            X_te = X[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr_s, y[train_idx])
        probs[test_idx] = clf.predict_proba(X_te_s)[:, 1]
    return probs


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Compute TPR at a given FPR threshold via interpolation on the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    # Interpolate: find TPR at target_fpr
    return float(np.interp(target_fpr, fpr, tpr))


def setup_icml_style():
    sns.set_theme(style="ticks", font_scale=1.0)
    plt.rcParams.update({
        "figure.figsize": (2.5, 2.5),
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_legend_variants(handles, labels, path_stem):
    n = len(labels)
    ncol_variants = sorted(set([1, 2, min(3, n), n]))
    for ncol in ncol_variants:
        nrow = (n + ncol - 1) // ncol
        fig_leg = plt.figure(figsize=(1.2 * ncol, 0.3 * nrow))
        fig_leg.legend(handles, labels, loc="center", frameon=False, ncol=ncol)
        fig_leg.savefig(f"{path_stem}_legend_{ncol}col.pdf",
                        bbox_inches="tight", pad_inches=0.02)
        plt.close(fig_leg)


def save_icml_figure(fig, path):
    handles, labels = [], []
    seen = set()
    for ax in fig.axes:
        ax.set_title("")
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in seen:
                handles.append(hi)
                labels.append(li)
                seen.add(li)
        if ax.get_legend():
            ax.get_legend().remove()
    sns.despine(fig=fig)
    fig.savefig(path)
    if handles and labels:
        save_legend_variants(handles, labels, str(path).replace(".pdf", ""))
    plt.close(fig)


def _derive_curve_signals(curve_raw, prefix):
    """Compute slope/gain/etc from a cached curve dict."""
    if not curve_raw:
        return {}
    curve = {int(k): v for k, v in curve_raw.items() if v is not None}
    if len(curve) < 2:
        return {}
    sizes = sorted(curve.keys())
    vals = [curve[s] for s in sizes]
    log_sizes = [math.log2(s) for s in sizes]
    n = len(log_sizes)
    sx = sum(log_sizes)
    sy = sum(vals)
    sxx = sum(x * x for x in log_sizes)
    sxy = sum(x * y for x, y in zip(log_sizes, vals))
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom if denom != 0 else 0
    gain = vals[-1] - vals[0]
    late_gain = vals[-1] - vals[-2]
    max_val = max(vals)
    early = None
    if 64 in curve and max_val > 0:
        early = curve[64] / max_val
    early_ratio = None
    if abs(gain) > 0.01:
        early_s = min((s for s in sizes if s >= 64), default=sizes[n // 2])
        early_ratio = (curve.get(early_s, vals[0]) - vals[0]) / gain
    return {
        f'{prefix}_slope': slope,
        f'{prefix}_gain': gain,
        f'{prefix}_late_gain': late_gain,
        f'{prefix}_early': early,
        f'{prefix}_early_ratio': early_ratio,
    }


def load_results(path: str) -> tuple[str, list[dict]]:
    """Load a results JSON and return (model_name, successful_results)."""
    with open(path) as f:
        data = json.load(f)
    model_name = data.get("model", "default")
    results = data["successful_results"]
    # Derive missing adaptation signals from curves
    for r in results:
        if r.get('ent_slope') is None:
            r.update(_derive_curve_signals(r.get('entropy_curve'), 'ent'))
        if r.get('conf_slope') is None:
            r.update(_derive_curve_signals(r.get('confidence_curve'), 'conf'))
        if r.get('adaptation_slope') is None:
            r.update(_derive_curve_signals(r.get('adaptation_curve'), 'adaptation'))
    return model_name, results


def extract_features(results: list[dict], signals: list[str] = None) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Extract feature matrix X, labels y, and dataset_ids for samples with all signals."""
    if signals is None:
        signals = SIGNALS
    X, y, ids = [], [], []
    for r in results:
        vals = [r.get(s) for s in signals]
        if any(v is None for v in vals):
            continue
        X.append(vals)
        y.append(1 if r.get("is_member") else 0)
        ids.append(r.get("dataset_id", -1))
    return np.array(X), np.array(y), ids


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run eval_multi_model.py <results_1.json> [results_2.json ...]")
        sys.exit(1)

    paths = sys.argv[1:]

    # Load all result files
    model_data: list[tuple[str, list[dict]]] = []
    for p in paths:
        name, results = load_results(p)
        model_data.append((name, results))
        print(f"Loaded {p}: model={name}, n={len(results)}")

    # Train classifier on first (training) file using all data
    train_name, train_results = model_data[0]
    X_train, y_train, _ = extract_features(train_results)

    if len(X_train) == 0:
        print("ERROR: No samples with all 3 signals in training file")
        sys.exit(1)

    print(f"\nTraining classifier on {train_name}: {len(X_train)} samples "
          f"({y_train.sum()} members, {len(y_train) - y_train.sum()} non-members)")
    print(f"Signals: {SIGNALS}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    # Print coefficients
    print(f"\nClassifier coefficients:")
    for name, coef in zip(SIGNALS, clf.coef_[0]):
        print(f"  {name:<30} {coef:+.4f}")
    print(f"  {'intercept':<30} {clf.intercept_[0]:+.4f}")

    # Evaluate on each model's results
    setup_icml_style()
    FIG_DIR.mkdir(exist_ok=True)

    # --- Transfer evaluation (train on default, apply to all) ---
    roc_data: list[tuple[str, np.ndarray, np.ndarray, float]] = []

    print(f"\n{'Model':<12} {'AUC':<8} {'N':<6} {'Members':<10} {'Non-members':<12}")
    print("-" * 50)

    for model_name, results in model_data:
        X, y, _ = extract_features(results)
        if len(X) == 0:
            print(f"{model_name:<12} {'N/A':<8} 0")
            continue

        X_scaled = scaler.transform(X)
        probs = clf.predict_proba(X_scaled)[:, 1]

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            print(f"{model_name:<12} {'N/A':<8} {len(y):<6} {n_pos:<10} {n_neg:<12}")
            continue

        auc = roc_auc_score(y, probs)
        fpr, tpr, _ = roc_curve(y, probs)
        roc_data.append((model_name, fpr, tpr, auc))

        print(f"{model_name:<12} {auc:<8.4f} {len(y):<6} {n_pos:<10} {n_neg:<12}")

    # --- Transfer ROC: Linear ---
    fig, ax = plt.subplots()
    for model_name, fpr, tpr, auc in roc_data:
        color = MODEL_COLORS.get(model_name, OKABE_ITO["black"])
        ls = MODEL_LINESTYLES.get(model_name, "-")
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.5,
                label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    save_icml_figure(fig, str(FIG_DIR / "fig_multi_model_roc.pdf"))
    print(f"\nFigure saved to {FIG_DIR / 'fig_multi_model_roc.pdf'}")

    # --- Transfer ROC: Log-log ---
    fig, ax = plt.subplots()
    for model_name, fpr, tpr, auc in roc_data:
        color = MODEL_COLORS.get(model_name, OKABE_ITO["black"])
        ls = MODEL_LINESTYLES.get(model_name, "-")
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.5,
                label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([1e-3, 1], [1e-3, 1], color="gray", linestyle=":", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(1e-2, 1)
    save_icml_figure(fig, str(FIG_DIR / "fig_multi_model_roc_loglog.pdf"))
    print(f"Figure saved to {FIG_DIR / 'fig_multi_model_roc_loglog.pdf'}")

    # ===================================================================
    # LOOCV evaluation: within-model LOOCV for unbiased per-model metrics
    # Run for each signal combo (AUC-optimal and TPR@1%-optimal)
    # ===================================================================
    for combo_key, combo_info in SIGNAL_COMBOS.items():
        combo_signals = combo_info["signals"]
        combo_label = combo_info["label"]
        combo_suffix = combo_info["suffix"]
        is_engineered = combo_info.get("engineered", False)

        print("\n" + "=" * 60)
        print(f"LOOCV Evaluation — {combo_label}")
        if is_engineered:
            print(f"Features: conf_late_gain__rank, conf_late_gain__p90, conf_early")
        else:
            print(f"Signals: {combo_signals}")
        print("=" * 60)

        loocv_roc_data: list[tuple[str, np.ndarray, np.ndarray, float, float]] = []

        print(f"\n{'Model':<12} {'AUC':<8} {'TPR@1%':<10} {'TPR@5%':<10} {'TPR@10%':<10} {'N':<6}")
        print("-" * 60)

        for model_name, results in model_data:
            if is_engineered:
                X, y, _ = extract_engineered(results)
            else:
                X, y, _ = extract_features(results, combo_signals)
            if len(X) == 0:
                continue
            n_pos = int(y.sum())
            n_neg = len(y) - n_pos
            if n_pos < 2 or n_neg < 2:
                print(f"{model_name:<12} {'N/A':<8} (need >=2 of each class)")
                continue

            probs = loocv_probs(X, y, engineered=is_engineered)
            auc = roc_auc_score(y, probs)
            fpr, tpr, _ = roc_curve(y, probs)
            t1 = tpr_at_fpr(y, probs, 0.01)
            t5 = tpr_at_fpr(y, probs, 0.05)
            t10 = tpr_at_fpr(y, probs, 0.10)
            loocv_roc_data.append((model_name, fpr, tpr, auc, t1))

            print(f"{model_name:<12} {auc:<8.4f} {t1:<10.4f} {t5:<10.4f} {t10:<10.4f} {len(y):<6}")

        # --- LOOCV ROC: Linear ---
        fig, ax = plt.subplots()
        for model_name, fpr, tpr, auc, t1 in loocv_roc_data:
            color = MODEL_COLORS.get(model_name, OKABE_ITO["black"])
            ls = MODEL_LINESTYLES.get(model_name, "-")
            ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.5,
                    label=f"{model_name} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        p = FIG_DIR / f"fig_multi_model_loocv_roc_{combo_suffix}.pdf"
        save_icml_figure(fig, str(p))
        print(f"\nFigure saved to {p}")

        # --- LOOCV ROC: Log-log ---
        fig, ax = plt.subplots()
        for model_name, fpr, tpr, auc, t1 in loocv_roc_data:
            color = MODEL_COLORS.get(model_name, OKABE_ITO["black"])
            ls = MODEL_LINESTYLES.get(model_name, "-")
            ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.5,
                    label=f"{model_name} (AUC={auc:.3f})")
        ax.plot([1e-3, 1], [1e-3, 1], color="gray", linestyle=":", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(1e-3, 1)
        ax.set_ylim(1e-2, 1)
        p = FIG_DIR / f"fig_multi_model_loocv_roc_loglog_{combo_suffix}.pdf"
        save_icml_figure(fig, str(p))
        print(f"Figure saved to {p}")

        # --- TPR@FPR bar chart ---
        if loocv_roc_data:
            model_names_bar = [d[0] for d in loocv_roc_data]
            tpr1_vals = [d[4] for d in loocv_roc_data]
            colors_bar = [MODEL_COLORS.get(n, OKABE_ITO["black"]) for n in model_names_bar]

            fig, ax = plt.subplots()
            x_pos = np.arange(len(model_names_bar))
            bars = ax.bar(x_pos, tpr1_vals, color=colors_bar, width=0.6, edgecolor="white")
            for bar, val in zip(bars, tpr1_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names_bar, rotation=30, ha="right")
            ax.set_ylabel("TPR @ 1% FPR")
            ax.set_ylim(0, max(tpr1_vals) * 1.25 if max(tpr1_vals) > 0 else 0.1)
            ax.axhline(y=0.01, color="gray", linestyle=":", linewidth=0.8, label="Random baseline")
            p = FIG_DIR / f"fig_multi_model_tpr_at_1fpr_{combo_suffix}.pdf"
            save_icml_figure(fig, str(p))
            print(f"Figure saved to {p}")


if __name__ == "__main__":
    main()
