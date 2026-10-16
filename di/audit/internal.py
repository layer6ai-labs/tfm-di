"""Internal audit: labelled meta-classification of target vs blind signals.

``audit_cell`` scores one (column set, meta-classifier) cell; ``run_internal`` runs the
main scope of a pool plus one cell per signal family. The protocol is documented in
README.md.
"""
from __future__ import annotations

import warnings
import zlib
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from di.audit.models import KINDS, elbow, fit_predict, make_tabpfn, rank_treeshap
from di.audit.pools import PoolData, fam_of, select

DEFAULT_KS = (1, 2, 4, 8, 16, 32, 64)
SAP_KS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 64)
Folds = list[tuple[np.ndarray, np.ndarray]]


def _safe(kind: str, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
          tabpfn: ClassifierMixin | None, where: str) -> np.ndarray:
    """fit_predict; a TabPFN fit that fails on a degenerate fold scores a constant 0.5."""
    try:
        return fit_predict(kind, Xtr, ytr, Xte, tabpfn)
    except Exception as e:  # noqa: BLE001
        if kind != "tabpfn":
            raise
        warnings.warn(f"tabpfn fit failed at {where}: {e!r}; scoring 0.5", stacklevel=2)
        return np.full(len(Xte), 0.5)


def _folds(y: np.ndarray, folds: int) -> Folds:
    """Stratified K-fold over datasets, capped at the minority class count."""
    k = min(folds, int(np.bincount(y).min()))
    return list(StratifiedKFold(k, shuffle=True, random_state=0).split(np.zeros(len(y)), y))


def _usable(Xt: np.ndarray, Xb: np.ndarray) -> np.ndarray:
    """Columns that vary in the target or in the blind."""
    return np.array([np.nanstd(Xt[:, j]) > 0 or np.nanstd(Xb[:, j]) > 0
                     for j in range(Xt.shape[1])])


def _bootstrap(y: np.ndarray, pt: np.ndarray, pb: np.ndarray, idx: np.ndarray,
               keep: list[int]) -> dict[str, float]:
    tg, bl, g = np.empty(len(keep)), np.empty(len(keep)), np.empty(len(keep))
    for j, i in enumerate(keep):
        s = idx[i]
        tt, bb = roc_auc_score(y[s], pt[s]), roc_auc_score(y[s], pb[s])
        tg[j], bl[j], g[j] = tt, bb, tt - bb
    return {"gap_sd": round(float(g.std()), 4), "target_sd": round(float(tg.std()), 4),
            "blind_sd": round(float(bl.std()), 4), "p_gap_le_0": round(float(np.mean(g <= 0)), 4)}


def cell_inputs(pool: PoolData, cols: Sequence[int], folds: int = 5
                ) -> tuple[np.ndarray, np.ndarray, Folds, list[np.ndarray]] | None:
    """Usable target/blind columns of ``cols``, the folds and the per-fold TreeSHAP
    ranking on the target; ``None`` when no column is usable."""
    Xt, Xb = pool.Xt[:, cols], pool.Xb[:, cols]
    ok = _usable(Xt, Xb)
    if not ok.any():
        return None
    fold_list = _folds(pool.y, folds)
    ranks = [rank_treeshap(Xt[:, ok][tr], pool.y[tr]) for tr, _ in fold_list]
    return Xt[:, ok], Xb[:, ok], fold_list, ranks


def audit_cell(Xt: np.ndarray, Xb: np.ndarray, y: np.ndarray, kind: str, ks: Sequence[int],
               folds: Folds, ranks: list[np.ndarray], rng: np.random.Generator,
               nboot: int = 2000, tabpfn: ClassifierMixin | None = None,
               scores: bool = False) -> dict:
    """Top-K curve, elbow K* and paired bootstrap for one cell.

    ``Xt``/``Xb`` are the usable columns of the cell, ``ranks`` their per-fold TreeSHAP
    order. Returns K, target/blind AUC, gap, bootstrap sds and the AUC curve; with
    ``scores`` also the pooled out-of-fold target and blind scores at K*.
    """
    cur = {}
    for K in [k for k in ks if k <= Xt.shape[1]]:
        pt, pb = np.zeros(len(y)), np.zeros(len(y))
        for i, (tr, te) in enumerate(folds):
            sel = ranks[i][:K]
            pt[te] = _safe(kind, Xt[tr][:, sel], y[tr], Xt[te][:, sel], tabpfn,
                           f"K={K} fold {i} target")
            pb[te] = _safe(kind, Xb[tr][:, sel], y[tr], Xb[te][:, sel], tabpfn,
                           f"K={K} fold {i} blind")
        cur[K] = (pt, pb, roc_auc_score(y, pt), roc_auc_score(y, pb))
    Ks = sorted(cur)
    K = elbow(Ks, [cur[k][2] for k in Ks])
    pt, pb, t, b = cur[K]
    idx = rng.integers(0, len(y), size=(nboot, len(y)))
    keep = [i for i in range(nboot) if len(np.unique(y[idx[i]])) >= 2]
    out = {"K": int(K), "target": round(float(t), 4), "blind": round(float(b), 4),
           "gap": round(float(t - b), 4)}
    out.update(_bootstrap(y, pt, pb, idx, keep))
    out["nboot_used"] = len(keep)
    out["curve"] = {int(k): [round(float(cur[k][2]), 4), round(float(cur[k][3]), 4)] for k in Ks}
    if scores:
        out["scores"] = {"target": pt.tolist(), "blind": pb.tolist()}
    return out


def run_internal(pool: PoolData, kinds: Sequence[str] = ("lr", "xgb"),
                 ks: Sequence[int] = DEFAULT_KS, scope: str = "ALL", folds: int = 5,
                 nboot: int = 2000, seed: int = 0, device: str = "cpu",
                 families: bool = True, min_family: int = 5,
                 log: Callable[[str], None] | None = None) -> dict:
    """Audit one pool: the ``scope`` cell, then one cell per signal family inside it
    with at least ``min_family`` columns, for every meta-classifier in ``kinds``.

    Returns ``{"meta": {...}, "cells": {scope_name: {"n_cols": int, kind: cell|None}}}``;
    a cell is ``None`` when none of its columns is usable.
    """
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown kinds {sorted(unknown)}; expected a subset of {KINDS}")
    y = pool.y
    base = select(pool.signals, scope)
    tabpfn = make_tabpfn(device) if "tabpfn" in kinds else None
    scopes = [(scope, base)]
    if families:
        fams: dict[str, list[int]] = {}
        for j in base:
            fams.setdefault(fam_of(pool.signals[j]), []).append(j)
        scopes += [(f, c) for f, c in sorted(fams.items()) if len(c) >= min_family]
    cells = {}
    for name, cols in scopes:
        cells[name] = {"n_cols": len(cols)}
        inputs = cell_inputs(pool, cols, folds)
        for kind in kinds:
            if inputs is None:
                cells[name][kind] = None
                continue
            Xt, Xb, fold_list, ranks = inputs
            rng = np.random.default_rng([seed, zlib.crc32(f"{pool.name}|{name}|{kind}".encode())])
            cells[name][kind] = audit_cell(Xt, Xb, y, kind, ks, fold_list, ranks, rng, nboot, tabpfn)
            if log:
                c = cells[name][kind]
                log(f"{pool.name:18s} {name:12s} {kind:6s} K={c['K']:<3d} "
                    f"{c['target']:.4f}/{c['blind']:.4f} gap {c['gap']:+.4f}+-{c['gap_sd']:.4f}")
    meta = {"pool": pool.name, "n": int(len(y)), "n_member": pool.n_member,
            "n_signals": len(pool.signals), "scope": scope, "n_cols": len(base),
            "folds": len(_folds(y, folds)), "nboot": nboot, "ks": list(ks), "kinds": list(kinds),
            "seed": seed}
    return {"meta": meta, "cells": cells}
