#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "scikit-learn",
# ]
# ///

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from di.data_loading import load_openml_dataset
from di.datasets.tabdpt_training_ids import get_all_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blind baseline for dataset membership inference"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="di_extended_results_*.json (default: latest)",
    )
    parser.add_argument("--feature-set", choices=["minimal", "raw"], default="raw")
    parser.add_argument("--train-frac", type=float, default=0.2)
    parser.add_argument("--split-seeds", type=str, default="7,13,42,99,2026")
    parser.add_argument(
        "--id-source", choices=["results", "all_ids"], default="results"
    )
    parser.add_argument("--cache-features", action="store_true", default=True)
    parser.add_argument(
        "--no-cache-features", dest="cache_features", action="store_false"
    )
    parser.add_argument("--output-dir", type=str, default="experiments")
    return parser.parse_args()


def latest_results_path() -> Path:
    files = sorted(Path(".").glob("di_extended_results_*.json"))
    if not files:
        raise FileNotFoundError("No di_extended_results_*.json files found")
    return files[-1]


def safe_float(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts.astype(float)
    total = counts.sum()
    if total <= 0:
        return np.nan
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def nan_skew(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    z = (x - m) / s
    return float(np.mean(z**3))


def nan_kurtosis_excess(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.nan
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    z = (x - m) / s
    return float(np.mean(z**4) - 3.0)


def per_column_unique_stats(X: np.ndarray) -> tuple[float, float, float]:
    n = X.shape[0]
    if n == 0:
        return np.nan, np.nan, np.nan

    unique_counts = []
    for j in range(X.shape[1]):
        col = X[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            unique_counts.append(0)
            continue
        unique_counts.append(np.unique(col).size)

    unique_counts = np.array(unique_counts, dtype=float)
    unique_ratio = unique_counts / max(n, 1)
    return (
        float(np.mean(unique_ratio)),
        float(np.mean(unique_counts <= 10)),
        float(np.mean(unique_counts <= 2)),
    )


def compute_raw_features(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, metadata: dict
) -> dict:
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)

    n_train, n_features = X.shape
    n_test = int(len(X_test))
    n_samples = int(n_train + n_test)

    finite_mask = np.isfinite(X)
    missing_rate = float(1.0 - finite_mask.mean())
    miss_per_col = 1.0 - finite_mask.mean(axis=0)

    col_mean = np.nanmean(X, axis=0)
    col_std = np.nanstd(X, axis=0)

    col_skew = np.array([nan_skew(X[:, j]) for j in range(n_features)], dtype=float)
    col_kurt = np.array(
        [nan_kurtosis_excess(X[:, j]) for j in range(n_features)], dtype=float
    )

    unique_ratio_mean, low_card_frac, binary_like_frac = per_column_unique_stats(X)

    y_valid = y[np.isfinite(y)]
    target_type = metadata.get("target_type", "unknown")
    is_cls = 1.0 if target_type == "classification" else 0.0

    n_classes = np.nan
    class_entropy = np.nan
    majority_frac = np.nan
    minority_frac = np.nan
    y_std = np.nan
    y_skew = np.nan
    y_kurt = np.nan

    if y_valid.size > 0:
        if target_type == "classification":
            y_int = np.round(y_valid).astype(int)
            _, counts = np.unique(y_int, return_counts=True)
            n_classes = float(len(counts))
            class_entropy = entropy_from_counts(counts)
            probs = counts / counts.sum()
            majority_frac = float(np.max(probs))
            minority_frac = float(np.min(probs))
        else:
            y_std = float(np.std(y_valid))
            y_skew = nan_skew(y_valid)
            y_kurt = nan_kurtosis_excess(y_valid)

    return {
        "n_samples": float(n_samples),
        "n_train": float(n_train),
        "n_test": float(n_test),
        "n_features": float(n_features),
        "x_missing_rate": missing_rate,
        "x_missing_col_mean": float(np.nanmean(miss_per_col)),
        "x_missing_col_std": float(np.nanstd(miss_per_col)),
        "x_col_mean_abs_mean": float(np.nanmean(np.abs(col_mean))),
        "x_col_mean_std": float(np.nanstd(col_mean)),
        "x_col_std_mean": float(np.nanmean(col_std)),
        "x_col_std_std": float(np.nanstd(col_std)),
        "x_col_skew_mean": float(np.nanmean(col_skew)),
        "x_col_skew_std": float(np.nanstd(col_skew)),
        "x_col_kurt_mean": float(np.nanmean(col_kurt)),
        "x_col_kurt_std": float(np.nanstd(col_kurt)),
        "x_unique_ratio_mean": unique_ratio_mean,
        "x_low_card_frac": low_card_frac,
        "x_binary_like_frac": binary_like_frac,
        "target_is_classification": is_cls,
        "y_missing_rate": float(1.0 - np.isfinite(y).mean()) if y.size > 0 else np.nan,
        "y_n_classes": n_classes,
        "y_class_entropy": class_entropy,
        "y_majority_frac": majority_frac,
        "y_minority_frac": minority_frac,
        "y_std": y_std,
        "y_skew": y_skew,
        "y_kurt": y_kurt,
    }


def compute_raw_features_from_cache(dataset_id: int, cache_dir: str | Path) -> dict | None:
    """Compute metadata features from cached NPZ + meta JSON (no network needed).

    Args:
        dataset_id: OpenML dataset ID
        cache_dir: path to slurm_cache directory (contains data/{id}_split.npz)

    Returns:
        Feature dict (same as compute_raw_features), or None if cache miss.
    """
    cache_path = Path(cache_dir) / "data"
    npz_path = cache_path / f"{dataset_id}_split.npz"
    meta_path = cache_path / f"{dataset_id}_meta.json"

    if not npz_path.exists():
        return None

    data = np.load(npz_path)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]

    metadata = {}
    if meta_path.exists():
        with meta_path.open() as f:
            metadata = json.load(f)

    return compute_raw_features(X_train, X_test, y_train, metadata)


def load_rows_from_results(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)

    rows = []
    for r in data.get("successful_results", []):
        if not r.get("success"):
            continue
        dataset_id = r.get("dataset_id")
        if dataset_id is None:
            continue
        label = int(r.get("true_label", int(bool(r.get("is_member", False)))))
        is_member = bool(r.get("is_member", label == 1))
        rows.append(
            {
                "dataset_id": int(dataset_id),
                "label": label,
                "is_member": is_member,
                "n_train": safe_float(r.get("n_train")),
                "n_test": safe_float(r.get("n_test")),
                "n_features": safe_float(r.get("n_features")),
            }
        )
    return rows


def build_rows_from_all_ids() -> list[dict]:
    ids = get_all_ids()
    members = set(ids["members"])
    rows = []
    for dataset_id in ids["all_ids"]:
        is_member = dataset_id in members
        rows.append(
            {
                "dataset_id": int(dataset_id),
                "label": 1 if is_member else 0,
                "is_member": is_member,
                "n_train": np.nan,
                "n_test": np.nan,
                "n_features": np.nan,
            }
        )
    return rows


def build_feature_table(
    rows: list[dict], feature_set: str, cache_path: Path | None
) -> tuple[list[dict], list[dict]]:
    if feature_set == "minimal":
        out_rows = []
        failed = []
        for r in rows:
            if (
                np.isnan(r["n_train"])
                or np.isnan(r["n_test"])
                or np.isnan(r["n_features"])
            ):
                failed.append(
                    {
                        "dataset_id": r["dataset_id"],
                        "reason": "missing_minimal_features",
                    }
                )
                continue
            out_rows.append(
                {
                    "dataset_id": r["dataset_id"],
                    "label": r["label"],
                    "is_member": r["is_member"],
                    "features": {
                        "n_train": r["n_train"],
                        "n_test": r["n_test"],
                        "n_features": r["n_features"],
                    },
                }
            )
        return out_rows, failed

    cache = {}
    if cache_path is not None and cache_path.exists():
        with cache_path.open() as f:
            cached = json.load(f)
        for item in cached.get("rows", []):
            cache[int(item["dataset_id"])] = item

    out_rows = []
    failed = []

    for i, r in enumerate(rows, start=1):
        did = r["dataset_id"]
        print(f"[{i}/{len(rows)}] Dataset {did} (member={r['is_member']})")

        if did in cache:
            item = cache[did]
            out_rows.append(
                {
                    "dataset_id": did,
                    "label": r["label"],
                    "is_member": r["is_member"],
                    "features": item["features"],
                }
            )
            continue

        try:
            if r["is_member"]:
                X_train, X_test, y_train, _y_test, metadata = load_openml_dataset(
                    dataset_id=did, test_size=0.3, seed=42
                )
            else:
                X_train, X_test, y_train, _y_test, metadata = load_openml_dataset(
                    task_id=did, test_size=0.3, seed=42
                )
            feats = compute_raw_features(X_train, X_test, y_train, metadata)
            item = {
                "dataset_id": did,
                "features": feats,
            }
            cache[did] = item
            out_rows.append(
                {
                    "dataset_id": did,
                    "label": r["label"],
                    "is_member": r["is_member"],
                    "features": feats,
                }
            )
            if cache_path is not None:
                payload = {
                    "updated_at": datetime.now().isoformat(),
                    "rows": sorted(cache.values(), key=lambda x: x["dataset_id"]),
                }
                with cache_path.open("w") as f:
                    json.dump(payload, f, indent=2)
        except Exception as e:
            failed.append({"dataset_id": did, "reason": str(e)[:500]})
            print(f"  failed: {e}")

    return out_rows, failed


def evaluate_splits(
    table_rows: list[dict], split_seeds: list[int], train_frac: float
) -> list[dict]:
    feature_names = sorted(table_rows[0]["features"].keys())

    X = np.array(
        [[safe_float(r["features"].get(k)) for k in feature_names] for r in table_rows],
        dtype=float,
    )
    y = np.array([int(r["label"]) for r in table_rows], dtype=int)
    ids = np.array([int(r["dataset_id"]) for r in table_rows], dtype=int)

    records = []
    for seed in split_seeds:
        X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
            X, y, ids, train_size=train_frac, stratify=y, random_state=seed
        )

        overlap = len(set(id_tr.tolist()) & set(id_te.tolist()))

        clf = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        class_weight="balanced", max_iter=2000, random_state=seed
                    ),
                ),
            ]
        )
        clf.fit(X_tr, y_tr)

        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y_te, probs)

        tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()

        rec = {
            "seed": seed,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "train_members": int(y_tr.sum()),
            "train_non_members": int((1 - y_tr).sum()),
            "test_members": int(y_te.sum()),
            "test_non_members": int((1 - y_te).sum()),
            "overlap_ids": int(overlap),
            "roc_auc": float(roc_auc_score(y_te, probs)),
            "avg_precision": float(average_precision_score(y_te, probs)),
            "accuracy": float(accuracy_score(y_te, preds)),
            "precision": float(precision_score(y_te, preds, zero_division=0)),
            "recall": float(recall_score(y_te, preds, zero_division=0)),
            "f1": float(f1_score(y_te, preds, zero_division=0)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "tpr_at_1fpr": float(np.interp(0.01, fpr, tpr)),
            "tpr_at_5fpr": float(np.interp(0.05, fpr, tpr)),
            "tpr_at_10fpr": float(np.interp(0.10, fpr, tpr)),
            "feature_names": feature_names,
        }

        lr = clf.named_steps["lr"]
        rec["coef"] = {name: float(c) for name, c in zip(feature_names, lr.coef_[0])}
        rec["intercept"] = float(lr.intercept_[0])

        records.append(rec)

    return records


def summarize(records: list[dict]) -> dict:
    keys = [
        "roc_auc",
        "avg_precision",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tpr_at_1fpr",
        "tpr_at_5fpr",
        "tpr_at_10fpr",
    ]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in records], dtype=float)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out


def main():
    args = parse_args()

    if not (0 < args.train_frac < 1):
        raise ValueError("--train-frac must be in (0, 1)")

    split_seeds = [int(s.strip()) for s in args.split_seeds.split(",") if s.strip()]
    if not split_seeds:
        raise ValueError("No valid split seeds")

    results_path = (
        Path(args.results_file) if args.results_file else latest_results_path()
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.id_source == "results":
        rows = load_rows_from_results(results_path)
    else:
        rows = build_rows_from_all_ids()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_path = None
    if args.cache_features and args.feature_set == "raw":
        source_tag = results_path.stem if args.id_source == "results" else "all_ids"
        cache_path = output_dir / f"blind_raw_features_cache_{source_tag}.json"

    table_rows, failed = build_feature_table(rows, args.feature_set, cache_path)
    if not table_rows:
        raise RuntimeError("No usable rows for training/evaluation")

    records = evaluate_splits(
        table_rows, split_seeds=split_seeds, train_frac=args.train_frac
    )
    summary = summarize(records)

    result = {
        "timestamp": stamp,
        "feature_set": args.feature_set,
        "id_source": args.id_source,
        "results_file": str(results_path),
        "n_total_rows_requested": len(rows),
        "n_rows_used": len(table_rows),
        "n_failed_feature_rows": len(failed),
        "failed_feature_rows": failed,
        "train_fraction": args.train_frac,
        "test_fraction": 1.0 - args.train_frac,
        "split_seeds": split_seeds,
        "records": records,
        "summary": summary,
    }

    out_path = (
        output_dir / f"blind_baseline_{args.feature_set}_{args.id_source}_{stamp}.json"
    )
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"RESULTS_FILE={results_path}")
    print(f"FEATURE_SET={args.feature_set}")
    print(f"ID_SOURCE={args.id_source}")
    print(f"ROWS_USED={len(table_rows)} FAILED={len(failed)}")
    if failed:
        print("FAILED_DATASETS_SAMPLE=")
        for r in failed[:10]:
            print(f"  {r['dataset_id']}: {r['reason']}")
    print("seed\tAUC\tAP\tAcc\tF1\tTPR@1\tTPR@5\tTPR@10\toverlap")
    for r in records:
        print(
            f"{r['seed']}\t{r['roc_auc']:.4f}\t{r['avg_precision']:.4f}\t{r['accuracy']:.4f}\t"
            f"{r['f1']:.4f}\t{r['tpr_at_1fpr']:.4f}\t{r['tpr_at_5fpr']:.4f}\t{r['tpr_at_10fpr']:.4f}\t{r['overlap_ids']}"
        )

    print("mean+-std")
    for k, v in summary.items():
        print(f"{k}={v['mean']:.4f}+-{v['std']:.4f}")
    print(f"OUTPUT_FILE={out_path}")
    if cache_path is not None:
        print(f"FEATURE_CACHE={cache_path}")


if __name__ == "__main__":
    main()
