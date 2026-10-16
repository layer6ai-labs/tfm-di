"""External audit: label-free anomaly scoring of suspect datasets.

``evaluate`` scores one column set over ``draws`` random known/suspect splits;
``run_external`` runs the ALL arm of a pool plus one arm per signal family, each
without and with the informativeness screen. The protocol is documented in README.md.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from di.audit.pools import PoolData, fam_of

Scheme = Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray]
SCREEN_B = 40
ALPHA = 0.1


def _impute(X: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Fill NaN with column medians of ``ref`` (default: ``X``); an all-NaN median is 0."""
    X = np.array(X, float)
    src = X if ref is None else np.asarray(ref, float)
    med = np.nanmedian(np.where(np.isfinite(src), src, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(X)
    X[bad] = np.take(med, np.nonzero(bad)[1])
    return X


def _iforest(Xk: np.ndarray, Xs: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """IsolationForest fitted on the known pile; anomaly scores for known and suspects."""
    f = IsolationForest(n_estimators=200, contamination="auto", random_state=seed).fit(Xk)
    return -f.score_samples(Xk), -f.score_samples(Xs)


def _conformal_p(cal: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Split-conformal p-values of ``test`` scores against the calibration scores."""
    cal = np.sort(np.asarray(cal))
    return (1.0 + len(cal) - np.searchsorted(cal, np.asarray(test), side="left")) / (len(cal) + 1.0)


def _split(y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Known pile = random half of the non-members (at least 5); suspects = the rest
    plus every member. Returns (known idx, suspect idx, suspect labels)."""
    non, mem = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    p = rng.permutation(non)
    k = max(5, int(round(0.5 * len(p))))
    known, held = p[:k], p[k:]
    return known, np.concatenate([held, mem]), np.concatenate([np.zeros(len(held)), np.ones(len(mem))])


def _sep_cols(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n1, n2 = len(A), len(B)
    R = rankdata(np.vstack([A, B]), axis=0)
    auc = (R[n1:].sum(0) - n2 * (n2 + 1) / 2.0) / (n1 * n2)
    return np.abs(auc - 0.5)


def _floor(Xk: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    null = np.empty((SCREEN_B, Xk.shape[1]))
    for b in range(SCREEN_B):
        q = rng.permutation(len(Xk))
        h = len(q) // 2
        null[b] = _sep_cols(Xk[q[:h]], Xk[q[h:]])
    return np.quantile(null, 0.95, axis=0)


def _screen(Xk: np.ndarray, Xs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Keep column j when its known-vs-suspect |AUC - 0.5| exceeds the 95th percentile of
    the same statistic between random halves of the known pile."""
    return _sep_cols(Xk, Xs) > _floor(Xk, rng)


def _noscreen(Xk: np.ndarray, Xs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Keep every column."""
    return np.ones(Xk.shape[1], bool)


SCHEMES: dict[str, Scheme] = {"noscreen": _noscreen, "screen": _screen}


def _one_model(X: np.ndarray, y: np.ndarray, scheme: Scheme, rng: np.random.Generator,
               columns: np.ndarray | None = None) -> dict | None:
    """One draw on one model's matrix: split, impute from the known pile, run the scheme,
    fit the IsolationForest on the selected columns. ``columns`` replaces the scheme's
    selection after it ran, so the generator state (and the IsolationForest seed) matches
    the target's draw. ``None`` when fewer than two columns are selected."""
    known, sus, yt = _split(y, rng)
    Xk, Xs = _impute(X[known], ref=X[known]), _impute(X[sus], ref=X[known])
    sel = np.asarray(scheme(Xk, Xs, rng), bool)
    if columns is not None:
        sel = np.asarray(columns, bool)
    if sel.sum() < 2:
        return None
    sk, ss = _iforest(Xk[:, sel], Xs[:, sel], int(rng.integers(1 << 30)))
    pv = _conformal_p(sk, ss)
    return {"auc": float(roc_auc_score(yt, ss)), "nsel": int(sel.sum()), "scores": ss,
            "pvals": pv, "yt": yt, "columns": sel,
            "tpr": float((pv <= ALPHA)[yt == 1].mean()),
            "fpr": float((pv <= ALPHA)[yt == 0].mean())}




def _meta_only_auc(meta: np.ndarray, y: np.ndarray, draws: int, seed0: int) -> float:
    """Membership AUC of an IsolationForest on log1p(table metadata) alone, orientation-free."""
    M = np.log1p(np.where(np.isfinite(meta), meta, np.nan))
    keep = np.isfinite(M).any(0)
    if keep.sum() < 1:
        return float("nan")
    M = _impute(M[:, keep])
    a = []
    for i in range(draws):
        known, sus, yt = _split(y, np.random.default_rng(seed0 + i))
        if M.shape[1] < 2:
            sk, ss = M[known, 0], M[sus, 0]
        else:
            sk, ss = _iforest(M[known], M[sus], seed0 + i)
        v = roc_auc_score(yt, ss)
        a.append(max(v, 1 - v))
    return round(float(np.mean(a)), 4)


def evaluate(Xt: np.ndarray, Xb: np.ndarray, y: np.ndarray, scheme: Scheme, draws: int = 10,
             seed0: int = 7) -> dict | None:
    """Mean and sd over draws of target AUC, blind AUC and their paired gap. The blind is
    scored on the columns the scheme selected on the target. Draws on which either side
    selects fewer than two columns are dropped; ``None`` when none survive."""
    rt, rb = [], []
    for i in range(draws):
        a = _one_model(Xt, y, scheme, np.random.default_rng(seed0 + i))
        b = _one_model(Xb, y, scheme, np.random.default_rng(seed0 + i),
                       columns=a["columns"] if a else None)
        if a and b:
            rt.append(a)
            rb.append(b)
    if not rt:
        return None

    def m(rs: list[dict], k: str) -> float:
        return float(np.mean([r[k] for r in rs]))

    def sd(v: list[float]) -> float:
        return round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0

    return {"target_auc": round(m(rt, "auc"), 4), "blind_auc": round(m(rb, "auc"), 4),
            "gap": round(m(rt, "auc") - m(rb, "auc"), 4),
            "n_selected": int(np.mean([r["nsel"] for r in rt])),
            "tpr@0.1": round(m(rt, "tpr"), 4), "fpr@0.1": round(m(rt, "fpr"), 4),
            "target_auc_sd": sd([r["auc"] for r in rt]),
            "blind_auc_sd": sd([r["auc"] for r in rb]),
            "gap_sd": sd([a["auc"] - b["auc"] for a, b in zip(rt, rb)]),
            "draws": len(rt)}


def _arms(signals: list[str], exclude: Sequence[str] = (),
          min_cols: int = 2) -> list[tuple[str, np.ndarray]]:
    """Column selections: ALL (minus excluded families), then each remaining family with
    at least ``min_cols`` columns, sorted by name."""
    drop = set(exclude)
    out = [("ALL", np.array([fam_of(s) not in drop for s in signals]))]
    for f in sorted({fam_of(s) for s in signals} - drop):
        c = np.array([fam_of(s) == f for s in signals])
        if c.sum() >= min_cols:
            out.append((f, c))
    return out


def run_external(pool: PoolData, draws: int = 10, exclude: Sequence[str] = (), seed0: int = 7,
                 families: bool = True, log: Callable[[str], None] | None = None) -> list[dict]:
    """One row per arm with a ``noscreen`` and a ``screen`` cell (``None`` when no draw
    survived), the column count, target/blind coverage and the metadata-only AUC."""
    meta_auc = _meta_only_auc(pool.meta, pool.y, draws, seed0)
    arm_list = _arms(pool.signals, exclude)
    rows = []
    for name, cols in arm_list if families else arm_list[:1]:
        Xt, Xb = pool.Xt[:, cols], pool.Xb[:, cols]
        row = {"pool": pool.name, "arm": name, "n_cols": int(cols.sum()),
               "target_cov": round(float(np.isfinite(Xt).mean()), 4),
               "blind_cov": round(float(np.isfinite(Xb).mean()), 4),
               "meta_only_auc": meta_auc}
        for key, scheme in SCHEMES.items():
            r = evaluate(Xt, Xb, pool.y, scheme, draws, seed0)
            row[key] = r
            if log and r:
                log(f"{pool.name:18s} {name:12s} {key:8s} {r['target_auc']:.4f}/{r['blind_auc']:.4f} "
                    f"gap {r['gap']:+.4f}+-{r['gap_sd']:.4f} kept {r['n_selected']} draws {r['draws']}")
        rows.append(row)
    return rows
