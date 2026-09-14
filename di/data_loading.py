"""Dataset loading utilities for DI."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .datasets.openml import OpenMLDataset


def load_openml_dataset(
    dataset_id: int = None,
    task_id: int = None,
    test_size: float = 0.3,
    seed: int = 42,
    min_samples: int = 100,
    return_full: bool = False,
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
