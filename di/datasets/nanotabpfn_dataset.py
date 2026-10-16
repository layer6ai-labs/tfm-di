"""
Dataset loader for NanoTabPFN synthetic training data splits.

Provides functions to load synthetic classification and regression datasets
from the train and holdout H5 files.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Tuple


def load_nanotabpfn_dataset(
    dataset_id: Tuple[str, str, int],
    test_size: float = 0.3,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load a single synthetic NanoTabPFN dataset from the H5 file.

    Parameters:
    -----------
    dataset_id : tuple
        Dataset ID in form (task_type, split, index)
        - task_type: 'classification' or 'regression'
        - split: 'train' or 'holdout'
        - index: 0-based index within the split
    test_size : float, default=0.3
        Proportion of data for test set
    seed : int, default=42
        Random seed

    Returns:
    --------
    X_train, X_test, y_train, y_test, metadata
    """
    task_type, split, index = dataset_id
    
    # Construct H5 file path
    if task_type == 'classification':
        h5_file = Path('tfm_playground') / f'50x3_3_100k_classification_{split}.h5'
    elif task_type == 'regression':
        h5_file = Path('tfm_playground') / f'50x3_1280k_regression_{split}.h5'
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    if not h5_file.exists():
        raise FileNotFoundError(
            f"NanoTabPFN dataset file not found: {h5_file}. "
            f"Please ensure the H5 files have been split into train/holdout."
        )
    
    # Load dataset from H5
    with h5py.File(h5_file, 'r') as f:
        if index >= f['X'].shape[0]:
            raise IndexError(
                f"Dataset index {index} out of range for {task_type} {split} "
                f"({f['X'].shape[0]} datasets available)"
            )
        
        # Get this specific table
        X = f['X'][index]  # (num_rows, num_features)
        y = f['y'][index]  # (num_rows,)
        single_eval_pos = int(f['single_eval_pos'][index])
        
        # Convert to numpy arrays (copy to ensure they're not memory-mapped)
        X = np.asarray(X, dtype=np.float32).copy()
        y = np.asarray(y, dtype=np.float32 if task_type == 'regression' else np.int64).copy()
        
        # Remove NaN rows
        valid_mask = ~np.isnan(y)
        if task_type == 'regression':
            valid_mask = valid_mask & ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        single_eval_pos = min(single_eval_pos, len(y))
    
    # Split into train/test using single_eval_pos as the context split position
    # The convention is: rows 0:single_eval_pos are context, rows single_eval_pos: are evaluation
    if single_eval_pos > 0 and single_eval_pos < len(y):
        # Use the given split position
        n_context = single_eval_pos
    else:
        # Fall back to test_size split
        n_context = int(len(y) * (1 - test_size))
    
    # Random split for more robustness (use local RNG to avoid mutating global state)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    X_perm = X[perm]
    y_perm = y[perm]
    
    # Split
    X_train = X_perm[:n_context]
    X_test = X_perm[n_context:]
    y_train = y_perm[:n_context]
    y_test = y_perm[n_context:]
    
    if len(X_test) == 0:
        # If test set is empty, use a simpler split
        n_test = max(1, int(len(y_perm) * test_size))
        X_train = X_perm[:-n_test]
        X_test = X_perm[-n_test:]
        y_train = y_perm[:-n_test]
        y_test = y_perm[-n_test:]
    
    # Create metadata
    metadata = {
        'name': f'{task_type}_{split}_{index}',
        'source': 'nanotabpfn_synthetic',
        'task_type': task_type,
        'split': split,
        'index': index,
        'target_type': task_type,  # 'classification' or 'regression'
        'n_features': X.shape[1],
        'n_classes': int(np.max(y_train) + 1) if task_type == 'classification' else None,
    }
    
    return X_train, X_test, y_train, y_test, metadata
