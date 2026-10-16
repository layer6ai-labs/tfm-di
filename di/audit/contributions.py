"""Feature contributions (Table 3 / Table 8): the mean |TreeSHAP| of the XGBoost fits that
rank the columns in the internal audit, averaged over its training folds, for the target
and for the blind fit on its own fingerprints over the same columns."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import spearmanr

from di.audit.internal import _folds, _usable
from di.audit.models import treeshap_importance
from di.audit.names import describe
from di.audit.pools import PoolData, select


def _rows(names: list[str], order: np.ndarray, it: np.ndarray, ib: np.ndarray,
          rank_t: np.ndarray, rank_b: np.ndarray, top: int) -> list[dict]:
    return [{"column": names[j], **describe(names[j]), "target": round(float(it[j]), 4),
             "blind": round(float(ib[j]), 4), "target_rank": int(rank_t[j]) + 1,
             "blind_rank": int(rank_b[j]) + 1} for j in order[:top]]


def run_contributions(pool: PoolData, scope: str = "ALL", folds: int = 5, top: int = 10,
                      log: Callable[[str], None] | None = None) -> dict:
    """Mean |TreeSHAP| per usable column of ``scope`` for target and blind, averaged over the
    ``folds`` training folds of the internal audit; the top ``top`` columns of each side,
    the overlap of the two top sets and the Spearman correlation of the two importance
    vectors."""
    y = pool.y
    cols = select(pool.signals, scope)
    Xt, Xb = pool.Xt[:, cols], pool.Xb[:, cols]
    ok = _usable(Xt, Xb)
    names = [pool.signals[j] for j, keep in zip(cols, ok) if keep]
    Xt, Xb = Xt[:, ok], Xb[:, ok]
    fold_list = _folds(y, folds)
    it = np.mean([treeshap_importance(Xt[tr], y[tr]) for tr, _ in fold_list], axis=0)
    ib = np.mean([treeshap_importance(Xb[tr], y[tr]) for tr, _ in fold_list], axis=0)
    order_t, order_b = np.argsort(-it, kind="stable"), np.argsort(-ib, kind="stable")
    rank_t, rank_b = np.empty(len(names), int), np.empty(len(names), int)
    rank_t[order_t], rank_b[order_b] = np.arange(len(names)), np.arange(len(names))
    out = {"meta": {"pool": pool.name, "n": int(len(y)), "n_member": pool.n_member, "scope": scope,
                    "n_cols": len(names), "folds": len(fold_list), "top": top},
           "target_top": _rows(names, order_t, it, ib, rank_t, rank_b, top),
           "blind_top": _rows(names, order_b, it, ib, rank_t, rank_b, top),
           "overlap": {str(k): int(len(set(order_t[:k]) & set(order_b[:k]))) for k in (top, 32)},
           "spearman": round(float(spearmanr(it, ib).statistic), 4),
           "n_blind_zero_in_target_top": int(sum(ib[j] == 0 for j in order_t[:top])),
           "importance": {n: [round(float(a), 6), round(float(b), 6)] for n, a, b in zip(names, it, ib)}}
    if log:
        for r in out["target_top"]:
            log(f"{pool.name:10s} {r['feature']:40s} {r['manipulation']:>4s} {r['signal']:>4s} "
                f"{r['reduction']:>3s} target {r['target']:.4f} blind {r['blind']:.4f} (blind rank {r['blind_rank']})")
        log(f"{pool.name:10s} top-{top} overlap {out['overlap'][str(top)]}, spearman {out['spearman']:+.3f}")
    return out
