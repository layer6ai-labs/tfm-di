"""Pool registry: one target model and the blind reference extracted on the same datasets.

Tables follow the layout ``signals/<model>/<pool>/datasets.parquet``: one row per
dataset in audit order, the columns ``dataset_id``, ``true_label``, ``task_type``,
``success``, ``n_pool``, ``n_features`` and one column per signal. ``load`` returns the
aligned target and blind matrices; ``select`` maps a scope string to column indices.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NON_FEATURES = {"true_label", "is_member", "member", "dataset_id", "dataset_name",
                "success", "task_type", "grid_version", "iid_split", "did",
                "n_pool", "n_features", "n_classes", "n_str_cols", "n", "d", "C",
                "elapsed_s", "dataset_id_str", "label", "y", "target"}
META = ("n_pool", "n_features")
FAMILIES = ("ctxeqq", "nnsplit", "lperm", "ftx", "rowshuf", "tshuffle", "tabdpt_sim",
            "tserum", "bwash", "mislbl", "qleak", "rowdup",
            "ctx", "noise", "col", "splitsize", "temp", "base")
BLIND = "tabpfn25"


@dataclass(frozen=True)
class Pool:
    """Target model directory, pool directory, and the alternative target weights
    (``<target>_<variant>`` directories holding the same datasets)."""
    target: str
    pool: str
    variants: tuple[str, ...] = ()


POOLS: dict[str, Pool] = {
    "sap": Pool("sap-rpt-oss", "binary_t4"),
    "tabdpt": Pool("tabdpt", "binary_tabdpt", ("seed42", "seed123", "seed456")),
    "nano": Pool("nanotabpfn", "nanotabpfn_matched20k"),
    "realtabpfn": Pool("realtabpfn25", "binary_realtabpfn25"),
}


@dataclass(frozen=True)
class PoolData:
    """Aligned target (``Xt``) and blind (``Xb``) signal matrices for one pool, with
    the table-shape columns ``META`` in ``meta``."""
    name: str
    ids: list[str]
    y: np.ndarray
    signals: list[str]
    Xt: np.ndarray
    Xb: np.ndarray
    meta: np.ndarray

    @property
    def n_member(self) -> int:
        return int(self.y.sum())


def fam_of(signal: str) -> str:
    """Signal family: the longest registered prefix, else the text before the first '_'."""
    for f in FAMILIES:
        if signal == f or signal.startswith(f + "_"):
            return f
    return signal.split("_")[0]


def select(signals: list[str], scope: str) -> list[int]:
    """Column indices for ``ALL``, a family union ``a+b`` or an exclusion ``no:a+b``."""
    if scope == "ALL":
        return list(range(len(signals)))
    if scope.startswith("no:"):
        drop = set(scope[3:].split("+"))
        return [j for j, s in enumerate(signals) if fam_of(s) not in drop]
    want = set(scope.split("+"))
    return [j for j, s in enumerate(signals) if fam_of(s) in want]


def _clean(a: np.ndarray) -> np.ndarray:
    """Non-finite values and |x| above float32 range / 10 become NaN (overflow sentinels
    in the exported tables)."""
    a = np.where(np.isfinite(a), a, np.nan)
    return np.where(np.abs(a) > np.finfo(np.float32).max / 10, np.nan, a)


def _numeric(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return _clean(frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float))


def table_path(data_root: str | Path, model: str, pool: str) -> Path:
    return Path(data_root) / "signals" / model / pool / "datasets.parquet"


def load(name: str, data_root: str | Path, target: str = "default",
         pool_dir: str | None = None) -> PoolData:
    """Load a registered pool. ``target`` selects alternative target weights
    (``default`` or one of ``Pool.variants``); ``pool_dir`` replaces the registered pool
    directory (ablation tables under the same model directories); the blind is always
    TabPFN-2.5."""
    pool = POOLS[name]
    if target != "default" and target not in pool.variants:
        raise ValueError(f"{name}: target must be default or one of {pool.variants}")
    model = pool.target if target == "default" else f"{pool.target}_{target}"
    pool_dir = pool_dir or pool.pool
    t = pd.read_parquet(table_path(data_root, model, pool_dir))
    b = pd.read_parquet(table_path(data_root, BLIND, pool_dir))
    t = t[t.success.astype(bool)].reset_index(drop=True)
    b = b.set_index("dataset_id").loc[t.dataset_id].reset_index()
    if not (t.true_label.values == b.true_label.values).all():
        raise ValueError(f"target and blind labels disagree in {pool_dir}")
    sigs = sorted(c for c in t.columns if c not in NON_FEATURES and c in b.columns
                  and pd.api.types.is_numeric_dtype(t[c])
                  and not pd.api.types.is_bool_dtype(t[c]))
    return PoolData(name, [str(i) for i in t.dataset_id], t.true_label.astype(int).to_numpy(),
                    sigs, _numeric(t, sigs), _numeric(b, sigs), _numeric(t, list(META)))
