"""Dataset loading utilities for DIA."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from .binning import OTHER_LABEL, quantile_bin
from .datasets.openml import OpenMLDataset

MAX_CLASSES = 10
MAX_NAN_RATIO = 0.5
MAX_UNIQUE_RATIO = 0.2


def _is_date_column(col: pd.Series) -> bool:
    """True for datetime columns and object columns whose first values parse as ISO dates."""
    if pd.api.types.is_datetime64_any_dtype(col):
        return True
    if col.dtype == object and len(col) > 0:
        sample = col.dropna().head(5)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample, errors="raise", format="ISO8601")
                return True
            except (ValueError, TypeError):
                pass
    return False


def _candidate_columns(df: pd.DataFrame, relax_unique_ratio: bool = False) -> list[str]:
    """Columns eligible as a forced classification target, in frame order.

    Mirrors the T4 target filter: no date columns, numeric columns with at most
    ``MAX_NAN_RATIO`` missing, at most ``MAX_UNIQUE_RATIO`` distinct values per row
    unless ``relax_unique_ratio``, at least two distinct non-null values, and at
    least two rows per level for non-numeric columns and for every column with at
    most ``MAX_CLASSES`` levels (a low-cardinality target is used as is).
    """
    out: list[str] = []
    for col in df.columns:
        s = df[col]
        if _is_date_column(s):
            continue
        numeric = pd.api.types.is_numeric_dtype(s)
        if numeric and s.isna().mean() > MAX_NAN_RATIO:
            continue
        try:
            nunique = s.dropna().nunique()
            min_count = s.value_counts().min() if (not numeric or nunique <= MAX_CLASSES) else 2
        except (TypeError, ValueError):
            continue
        if nunique < 2:
            continue
        if not relax_unique_ratio and nunique / len(s) > MAX_UNIQUE_RATIO:
            continue
        if min_count < 2:
            continue
        out.append(str(col))
    return out


def _choose_target(df: pd.DataFrame, rng: np.random.Generator) -> tuple[str, str]:
    """Pick the forced target column and its rule.

    A column with 2..``MAX_CLASSES`` distinct values is drawn uniformly at random
    when one exists (``"lowcard"``); otherwise the eligible column with the fewest
    distinct values is used (``"fallback"``, discretised later).
    """
    cands = _candidate_columns(df) or _candidate_columns(df, relax_unique_ratio=True)
    if not cands:
        raise ValueError("no column is eligible as a classification target")
    nunique = {c: int(df[c].dropna().nunique()) for c in cands}
    lowcard = [c for c in cands if 2 <= nunique[c] <= MAX_CLASSES]
    if lowcard:
        return str(rng.choice(lowcard)), "lowcard"
    return min(cands, key=lambda c: nunique[c]), "fallback"


def _encode_features(X: pd.DataFrame) -> np.ndarray:
    """Ordinal-encode object and categorical columns and cast to float32."""
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object or isinstance(X[col].dtype, pd.CategoricalDtype):
            X[[col]] = OrdinalEncoder().fit_transform(X[[col]])
    return X.to_numpy().astype(np.float32)


def _cap_levels(y_fit: np.ndarray, y_apply: np.ndarray, max_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep the ``max_classes - 1`` most frequent levels of ``y_fit``; fold the rest,
    on both splits, into ``OTHER_LABEL``."""
    keep = set(pd.Series(y_fit).value_counts().index[: max_classes - 1])
    fit = np.where(np.isin(y_fit, list(keep)), y_fit, OTHER_LABEL)
    apply = np.where(np.isin(y_apply, list(keep)), y_apply, OTHER_LABEL)
    return fit, apply


def _discretise(y_tr: np.ndarray, y_te: np.ndarray, numeric: bool, rule: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Reduce a fallback target to at most ``MAX_CLASSES`` levels with edges or kept
    levels fitted on ``y_tr`` only. Low-cardinality targets pass through."""
    n_distinct = len(np.unique(np.concatenate([y_tr, y_te])))
    if rule == "lowcard" or n_distinct <= MAX_CLASSES:
        return y_tr, y_te, rule
    if numeric:
        y_tr, y_te, did_bin = quantile_bin(y_tr, y_te, n_bins=MAX_CLASSES)
        if not did_bin:
            # At most MAX_CLASSES levels in the training split: those levels are
            # the bins, and held-out values fall into the nearest lower level.
            edges = np.unique(y_tr)[1:]
            y_tr, y_te = np.digitize(y_tr, edges).astype(float), np.digitize(y_te, edges).astype(float)
        return y_tr, y_te, "binned"
    y_tr, y_te = _cap_levels(y_tr, y_te, MAX_CLASSES)
    return y_tr, y_te, "capped"


def _encode_labels(y_tr: np.ndarray, y_te: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Map the levels present in either split to 0..k-1 (sorted level order)."""
    classes = np.unique(np.concatenate([y_tr, y_te]))
    if len(classes) > MAX_CLASSES:
        raise ValueError(f"forced target has {len(classes)} classes (max {MAX_CLASSES})")
    index = {v: i for i, v in enumerate(classes.tolist())}
    y_tr = np.asarray([index[v] for v in y_tr.tolist()], dtype=np.float32)
    y_te = np.asarray([index[v] for v in y_te.tolist()], dtype=np.float32)
    return y_tr, y_te, len(classes)


def _check_size(n: int, id_label: object, min_samples: int, test_size: float) -> None:
    """Raise when ``n`` rows cannot yield the minimum train and test sizes."""
    if n < min_samples:
        raise ValueError(f"Dataset {id_label} has only {n} valid samples (min: {min_samples})")
    min_test = max(10, int(n * test_size))
    min_train = max(10, int(n * (1 - test_size)))
    if n < min_test + min_train:
        raise ValueError(f"Dataset {id_label} has only {n} samples, need at least {min_test + min_train}")


def _force_classification(
    dataset: OpenMLDataset,
    rng_seed: int,
    test_size: float,
    seed: int,
    min_samples: int,
    return_full: bool,
):
    """Re-derive a regression dataset as a classification task.

    The target column is chosen from the raw frame (the designated regression
    target stays a feature), rows missing it are dropped, the split is stratified by
    a low-cardinality target and drawn as in the regression path otherwise, and a
    fallback target is discretised with edges fitted on the training split. Returns the same tuple as
    ``load_openml_dataset``; ``metadata["forced_target"]`` records the choice.
    """
    df, _, _, _ = dataset.openml_dataset.get_data()
    column, rule = _choose_target(df, np.random.default_rng(rng_seed))
    df = df[df[column].notna()].reset_index(drop=True)
    id_label = dataset.metadata.get("openml_task_id") or dataset.metadata.get("openml_dataset_id")
    _check_size(len(df), id_label, min_samples, test_size)

    X = _encode_features(df.drop(columns=[column]))
    X[~np.isfinite(X)] = np.nan
    numeric = pd.api.types.is_numeric_dtype(df[column])
    y = df[column].to_numpy(dtype=float) if numeric else df[column].astype(str).to_numpy()

    if return_full:
        idx_tr, idx_te = np.arange(len(df)), np.arange(0)
    else:
        idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=test_size, random_state=seed,
                                          stratify=y if rule == "lowcard" else None)
    y_tr, y_te, rule = _discretise(y[idx_tr], y[idx_te], numeric, rule)
    y_tr, y_te, n_classes = _encode_labels(y_tr, y_te)

    metadata = dict(dataset.metadata)
    metadata["target_type"] = "classification"
    metadata["forced_target"] = {
        "column": column,
        "rule": rule,
        "n_classes": n_classes,
        "regression_target": dataset.openml_dataset.default_target_attribute,
    }
    print(f"Forced classification target: {column} ({rule}, {n_classes} classes)")
    print(f"Dataset loaded: {len(X)} samples, {X.shape[1]} features")
    if return_full:
        return X, y_tr, metadata
    print(f"Train: {len(idx_tr)}, Test: {len(idx_te)}")
    return X[idx_tr], X[idx_te], y_tr, y_te, metadata


def load_openml_dataset(
    dataset_id: int = None,
    task_id: int = None,
    test_size: float = 0.3,
    seed: int = 42,
    min_samples: int = 100,
    return_full: bool = False,
    force_task_type: str | None = None,
):
    """
    Load an OpenML dataset for dataset inference testing.

    Parameters
    ----------
    dataset_id : int, optional
        OpenML dataset ID (for member datasets)
    task_id : int, optional
        OpenML task ID (for CC18 non-member datasets). The task is resolved
        to its underlying dataset automatically.
    test_size : float
        Proportion of data to use as test set
    seed : int
        Random seed for reproducibility
    min_samples : int
        Minimum number of samples required after cleaning
    return_full : bool
        If True, return (X, y, metadata) without splitting.
    force_task_type : {"classification", None}
        With ``"classification"``, a regression dataset is re-derived as a
        classification task (see ``_force_classification``); classification
        datasets are unaffected.

    Returns
    -------
    X_train, X_test, y_train, y_test, metadata : tuple
        Split dataset (or X, y, metadata if return_full=True)
    """
    if dataset_id is None and task_id is None:
        raise ValueError("Must specify either dataset_id or task_id")

    id_label = dataset_id if dataset_id is not None else f"task_{task_id}"
    print(f"Loading OpenML dataset {id_label}...")
    if dataset_id is not None:
        dataset = OpenMLDataset(name=f"dataset_{dataset_id}", dataset_id=dataset_id)
    else:
        dataset = OpenMLDataset(name=f"task_{task_id}", task_id=task_id)
    dataset.prepare_data(download_dir="./data")

    if force_task_type == "classification" and dataset.metadata.get("target_type") == "regression":
        rng_seed = int(task_id if task_id is not None else dataset_id)
        return _force_classification(dataset, rng_seed, test_size, seed, min_samples, return_full)

    X, y = dataset.all_instances()

    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.values

    try:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Dataset {dataset_id} has non-numeric data that cannot be converted: {e}"
        )

    # Replace Inf with NaN — TFM handles NaN via internal mean imputation.
    X[~np.isfinite(X)] = np.nan
    y[~np.isfinite(y)] = np.nan

    if len(X) < min_samples:
        raise ValueError(
            f"Dataset {dataset_id} has only {len(X)} valid samples (min: {min_samples})"
        )

    min_test = max(10, int(len(X) * test_size))
    min_train = max(10, int(len(X) * (1 - test_size)))
    if len(X) < min_test + min_train:
        raise ValueError(
            f"Dataset {dataset_id} has only {len(X)} samples, need at least {min_test + min_train}"
        )

    if return_full:
        print(f"Dataset loaded: {len(X)} samples, {X.shape[1]} features")
        print(f"Task type: {dataset.metadata.get('target_type', 'unknown')}")
        return X, y, dataset.metadata

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    print(f"Dataset loaded: {len(X)} samples, {X.shape[1]} features")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Task type: {dataset.metadata.get('target_type', 'unknown')}")

    return X_train, X_test, y_train, y_test, dataset.metadata
