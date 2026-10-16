"""Stack a published classification signal table with newly extracted rows.

Writes ``signals/<model>/<pool>/datasets.parquet`` for one model. New rows (one or
more ``datasets.parquet`` written by ``run_di.py``) are reindexed to the existing
table's columns (plus ``EXTRA_COLUMNS`` when present), their ids formatted like the
existing table's, and the union is sorted by ``dataset_id`` as strings. Existing
rows whose id also appears in the new rows are replaced. Payload, JSON-only and
``DROP_COLUMNS`` bookkeeping columns are dropped.

Usage:
    uv run scripts/merge_allcls_pools.py --existing <datasets.parquet> \
        --new '<glob of datasets.parquet>' --out <data_root>/signals/<model>/<pool>/datasets.parquet
"""

from __future__ import annotations

import argparse
import ast
import glob
from pathlib import Path

import pandas as pd

PAYLOAD_PREFIXES = ("_row_", "_curves", "_metadata")
KEY_COLUMNS = ("dataset_id", "true_label", "task_type", "success")
EXTRA_COLUMNS = ("forced_target",)
DROP_COLUMNS = ("iid_member_partition", "iid_split_seed")


def grid_id(value: object) -> str:
    """Id in the TabDPT grid convention, ``"['openml', <id>]"``."""
    text = str(value)
    if text.startswith("["):
        return str(list(ast.literal_eval(text)))
    return str(["openml", int(float(text))])


def plain_id(value: object) -> str:
    """Id as a bare integer string."""
    text = str(value)
    if text.startswith("["):
        return str(int(ast.literal_eval(text)[1]))
    return str(int(float(text)))


def read_new(pattern: str) -> pd.DataFrame:
    """Concatenate the successful rows of every table matching ``pattern``."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files match {pattern}")
    frames = [pd.read_parquet(p) for p in paths]
    new = pd.concat(frames, ignore_index=True, sort=False)
    new = new[new["success"].astype(bool)].reset_index(drop=True)
    print(f"new rows: {len(new)} from {len(paths)} files, {len(new.columns)} columns")
    return new


def merge(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Existing rows (minus replaced ids) and the new rows, restricted to the existing
    table's non-payload columns and sorted by id string."""
    fmt = grid_id if str(existing["dataset_id"].iloc[0]).startswith("[") else plain_id
    existing = existing.copy()
    existing["dataset_id"] = existing["dataset_id"].map(fmt)
    new = new.copy()
    new["dataset_id"] = new["dataset_id"].map(fmt)
    if new["dataset_id"].duplicated().any():
        raise ValueError("duplicate dataset ids among the new rows")

    columns = [c for c in existing.columns if not c.startswith(PAYLOAD_PREFIXES) and c not in DROP_COLUMNS]
    columns += [c for c in EXTRA_COLUMNS if c in new.columns and c not in columns]
    missing = [c for c in KEY_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"existing table lacks {missing}")
    replaced = existing["dataset_id"].isin(set(new["dataset_id"]))
    kept = existing.loc[~replaced].reindex(columns=columns)
    added = new.reindex(columns=columns)

    dropped = [c for c in new.columns if c not in columns and not c.startswith(PAYLOAD_PREFIXES)]
    absent = [c for c in columns if c not in new.columns]
    print(f"existing rows kept: {len(kept)} (replaced {int(replaced.sum())}); "
          f"new-row columns dropped: {len(dropped)}; existing columns absent in new rows: {len(absent)}")
    if absent:
        print("  absent:", ", ".join(absent[:20]), "..." if len(absent) > 20 else "")

    merged = pd.concat([kept, added], ignore_index=True, sort=False)
    merged = merged.sort_values("dataset_id", key=lambda s: s.astype(str), kind="stable").reset_index(drop=True)
    merged["dataset_id"] = merged["dataset_id"].astype(str)
    merged["true_label"] = merged["true_label"].astype(int)
    merged["success"] = merged["success"].astype(bool)
    if merged["dataset_id"].duplicated().any():
        raise ValueError("duplicate dataset ids after merge")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--existing", required=True, help="published classification table (parquet)")
    parser.add_argument("--new", required=True, help="glob of run_di.py datasets.parquet files")
    parser.add_argument("--out", required=True, help="output datasets.parquet")
    args = parser.parse_args()

    existing = pd.read_parquet(args.existing)
    print(f"existing rows: {len(existing)}, {len(existing.columns)} columns")
    merged = merge(existing, read_new(args.new))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    labels = merged["true_label"].value_counts().to_dict()
    tasks = merged["task_type"].value_counts().to_dict()
    print(f"wrote {out}: {len(merged)} rows x {len(merged.columns)} columns; labels {labels}; task_type {tasks}")


if __name__ == "__main__":
    main()
