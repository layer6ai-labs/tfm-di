#!/usr/bin/env python
"""
Split .h5 prior dump files into 50/50 train/holdout splits.

For each h5 file, creates two new files with "_train" and "_holdout" suffixes,
splitting the dataset dimension 50/50 while preserving all metadata.

If the source files do not exist, they are automatically downloaded from the
TFM-Playground research artifacts server.
"""

import h5py
import os
import urllib.request
from pathlib import Path

DATASET_URLS = {
    "50x3_3_100k_classification.h5": (
        "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle"
        "/TFM-Playground/50x3_3_100k_classification.h5"
    ),
    "50x3_1280k_regression.h5": (
        "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle"
        "/TFM-Playground/50x3_1280k_regression.h5"
    ),
}


def _make_progress_hook(filename: str):
    """Return a urlretrieve reporthook that prints a simple progress line."""
    def hook(block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded / total_size * 100)
            mb_done = downloaded / 1_048_576
            mb_total = total_size / 1_048_576
            print(f"\r  Downloading {filename}: {mb_done:.1f} / {mb_total:.1f} MB ({pct:.1f}%)", end="", flush=True)
        else:
            mb_done = downloaded / 1_048_576
            print(f"\r  Downloading {filename}: {mb_done:.1f} MB", end="", flush=True)
    return hook


def download_if_missing(dest_path: Path) -> None:
    """Download the file at *dest_path* from the known URL if it does not exist."""
    if dest_path.exists():
        return

    filename = dest_path.name
    url = DATASET_URLS.get(filename)
    if url is None:
        raise FileNotFoundError(
            f"{dest_path} not found and no download URL is registered for '{filename}'."
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {filename} not found — downloading from:\n  {url}")
    tmp_path = dest_path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_make_progress_hook(filename))
        tmp_path.rename(dest_path)
        print(f"\n  ✓ Downloaded {filename}")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def split_h5_file(input_path: str, output_dir: str = None):
    """
    Split an h5 file into train and holdout splits (50/50).
    
    Args:
        input_path: Path to the input .h5 file
        output_dir: Directory to save split files (default: same as input)
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_path) or "."
    
    input_path = Path(input_path)
    stem = input_path.stem
    suffix = input_path.suffix
    
    train_path = Path(output_dir) / f"{stem}_train{suffix}"
    holdout_path = Path(output_dir) / f"{stem}_holdout{suffix}"
    
    print(f"Splitting {input_path.name}...")
    print(f"  Train output:   {train_path.name}")
    print(f"  Holdout output: {holdout_path.name}")
    
    with h5py.File(input_path, 'r') as f_in:
        # Get dataset dimensions
        num_tables = f_in['X'].shape[0]
        num_rows = f_in['X'].shape[1]
        num_features_max = f_in['X'].shape[2]
        
        mid = num_tables // 2
        
        print(f"  Total tables: {num_tables}")
        print(f"  Train tables: {mid}")
        print(f"  Holdout tables: {num_tables - mid}")
        
        # Create train file
        with h5py.File(train_path, 'w') as f_train:
            # Copy datasets
            f_train.create_dataset('X', data=f_in['X'][:mid])
            f_train.create_dataset('y', data=f_in['y'][:mid])
            f_train.create_dataset('single_eval_pos', data=f_in['single_eval_pos'][:mid])
            
            # Copy metadata
            for key in f_in.keys():
                if key not in ['X', 'y', 'single_eval_pos']:
                    if isinstance(f_in[key], h5py.Dataset):
                        f_train.create_dataset(key, data=f_in[key])
                    elif isinstance(f_in[key], h5py.Group):
                        # Skip groups, copy only datasets
                        pass
        
        print(f"  ✓ Train split created: {num_rows}x{num_features_max}, {mid} tables")
        
        # Create holdout file
        with h5py.File(holdout_path, 'w') as f_holdout:
            # Copy datasets
            f_holdout.create_dataset('X', data=f_in['X'][mid:])
            f_holdout.create_dataset('y', data=f_in['y'][mid:])
            f_holdout.create_dataset('single_eval_pos', data=f_in['single_eval_pos'][mid:])
            
            # Copy metadata
            for key in f_in.keys():
                if key not in ['X', 'y', 'single_eval_pos']:
                    if isinstance(f_in[key], h5py.Dataset):
                        f_holdout.create_dataset(key, data=f_in[key])
                    elif isinstance(f_in[key], h5py.Group):
                        # Skip groups, copy only datasets
                        pass
        
        print(f"  ✓ Holdout split created: {num_rows}x{num_features_max}, {num_tables - mid} tables")


if __name__ == '__main__':
    tfm_playground_dir = Path(__file__).parent / 'tfm_playground'

    # Download and split classification file
    class_file = tfm_playground_dir / '50x3_3_100k_classification.h5'
    download_if_missing(class_file)
    split_h5_file(class_file)

    print()

    # Download and split regression file
    reg_file = tfm_playground_dir / '50x3_1280k_regression.h5'
    download_if_missing(reg_file)
    split_h5_file(reg_file)

    print("\n✓ Done!")
