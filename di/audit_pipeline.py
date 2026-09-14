"""The two submission DI audits, as one reusable library.

Both audits take the same input — a per-dataset signal table for a TARGET model,
the same table for the BLIND (TabPFN-2.5), and dataset-level membership labels —
and answer "does the target leak membership *beyond* the public blind".

------------------------------------------------------------------ INTERNAL
Supervised privacy audit (auditor HAS member labels). Nested leave-one-dataset-out:
  * outer LOOCV over datasets;
  * inner 5-fold stratified CV on the outer-train picks K* = the top-K feature
    count (features ranked by XGBoost/TreeSHAP mean|contrib| on the inner-train);
  * held-out predictions from LR / XGB (/TabPFN-2.5) at K*;
  * tie-aware folded AUC (`roc_auc_score` on a shuffled row order, max(a,1-a)) —
    never naive argsort ranks (see the sat90 tie bug).
The auditor NEVER consults the blind: K* = argmax inner TARGET auc, features are
ranked on the target, and the blind is then scored on exactly those columns. The
blind is the control we evaluate the target's selected signals against — it is
never given its own ranking, its own K, or any influence on the selection.
Significance: multi-draw shuffled-label permutation null that re-runs the WHOLE
pipeline (selection included), so the null is selection-aware.

------------------------------------------------------------------ EXTERNAL
Label-free outside auditor (H11 threat model): the auditor holds only a set of
KNOWN NON-MEMBERS, a SUSPECT pile of unknown membership, and query access to the
target and the blind. No member labels anywhere inside the method.
  * per signal, z-score against the known non-members;
  * direction from a PU shift — the mean of the *unlabeled* suspect pile minus
    the known-non mean — never from member labels;
  * score = mean of direction-oriented z over the scope's signals;
  * split-conformal p-value against the known-non scores -> flag at nominal FPR;
  * scored (ONLY for evaluation) with true membership: unfolded AUC on the
    suspect pile + TPR at calibrated FPR.
Blind control = the IDENTICAL auditor run on the blind's own signals (not
blind-subtraction: subtracting removes the size-correlated component the owner
ruling explicitly allows as an indicator).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "XGBOOST_NTHREAD"):
    os.environ.setdefault(_v, "1")

from dataclasses import dataclass, field

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

XGB_PARAMS = dict(max_depth=3, n_estimators=150, learning_rate=0.06, subsample=0.85,
                  colsample_bytree=0.7, reg_lambda=1.0, min_child_weight=2,
                  eval_metric="logloss", n_jobs=1, verbosity=0, tree_method="hist",
                  random_state=0)
KGRID_DEFAULT = [1, 3, 5, 8, 13, 21, 34, 55]
SHUFFLE_SEED = 2026          # row-order shuffle before every AUC (tie-bug rule)


# --------------------------------------------------------------------- metrics
def folded_auc(y, p):
    """Tie-aware AUC on a shuffled row order, folded to max(a, 1-a)."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    perm = np.random.default_rng(SHUFFLE_SEED).permutation(len(y))
    a = float(roc_auc_score(y[perm], p[perm]))
    return max(a, 1.0 - a)


def directed_auc(y, p):
    """Tie-aware AUC, direction NOT folded (the auditor committed to a sign)."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    perm = np.random.default_rng(SHUFFLE_SEED).permutation(len(y))
    return float(roc_auc_score(y[perm], p[perm]))


# ------------------------------------------------------------------ signal set
@dataclass
class SignalSet:
    """Aligned target/blind signal matrices for one experiment."""
    name: str
    ids: list
    y: np.ndarray                    # 1 = member
    sigs: list                       # column names
    Xt: dict                         # target model name -> (n, p) matrix
    Xb: np.ndarray                   # blind matrix, same columns
    scopes: dict = field(default_factory=dict)   # scope name -> column indices
    notes: str = ""

    @property
    def n(self):
        return len(self.y)


def _impute_median(M):
    M = np.array(M, dtype=float)
    for j in range(M.shape[1]):
        c = M[:, j]
        bad = ~np.isfinite(c)
        if bad.any():
            m = np.nanmedian(np.where(np.isfinite(c), c, np.nan))
            c[bad] = m if np.isfinite(m) else 0.0
    return M


def build_signal_set(name, ids, y, sigs, target_mats, blind_mat,
                     min_support=0.5, scope_fn=None, notes="", impute="base"):
    """Equal-support column filter (>=min_support finite in EVERY model, target
    AND blind), drop constants, median-impute. Same rule as
    hyp_estimator_comparison.build_data so the pools stay apples-to-apples.

    impute controls the missing-data handicap. The blind is far patchier than the
    targets (e.g. 9.1% of TabDPT-grid cells vs 0.5%), and median-imputing each
    matrix on its own hands the target strictly more information — which inflates
    every target-blind gap. So report all three:
      "base"     each matrix median-imputed independently (the submission's rule)
      "masked"   a cell missing in ANY model is treated as missing in ALL, then
                 imputed — the target is handicapped down to the blind's coverage
      "complete" keep only columns finite everywhere (may leave nothing)
    """
    mats = list(target_mats.values()) + [blind_mat]
    keep = []
    for j in range(len(sigs)):
        if not all(np.isfinite(m[:, j]).mean() >= min_support for m in mats):
            continue
        if all(np.nanstd(np.where(np.isfinite(m[:, j]), m[:, j], np.nan)) == 0
               for m in mats):
            continue
        keep.append(j)
    sigs = [sigs[j] for j in keep]
    tmats = {k: np.array(m[:, keep], float) for k, m in target_mats.items()}
    bmat = np.array(blind_mat[:, keep], float)

    bad = ~np.isfinite(bmat)
    for m in tmats.values():
        bad |= ~np.isfinite(m)
    if impute == "complete":
        cols = np.flatnonzero(~bad.any(axis=0))
        sigs = [sigs[j] for j in cols]
        tmats = {k: m[:, cols] for k, m in tmats.items()}
        bmat, bad = bmat[:, cols], bad[:, cols]
    elif impute == "masked":
        for m in tmats.values():
            m[bad] = np.nan
        bmat[bad] = np.nan
    elif impute != "base":
        raise ValueError(impute)

    Xt = {k: _impute_median(m) for k, m in tmats.items()}
    Xb = _impute_median(bmat)
    scopes = scope_fn(sigs) if scope_fn else {"full": list(range(len(sigs)))}
    scopes = {k: v for k, v in scopes.items() if len(v) > 0}
    ss = SignalSet(name=name, ids=list(ids), y=np.asarray(y, int), sigs=sigs,
                   Xt=Xt, Xb=Xb, scopes=scopes, notes=notes)
    ss.impute = impute
    ss.missing = {**{f"target:{k}": round(float(np.mean(~np.isfinite(m[:, keep]))), 4)
                     for k, m in target_mats.items()},
                  "blind": round(float(np.mean(~np.isfinite(blind_mat[:, keep]))), 4),
                  "union_masked": round(float(bad.mean()), 4)}
    return ss


# ============================================================ INTERNAL AUDIT
def shap_order(X, y):
    """Feature ranking by XGBoost TreeSHAP mean|contribution| (train rows only)."""
    import xgboost as xgb
    clf = xgb.XGBClassifier(**XGB_PARAMS)
    clf.fit(X, y)
    contribs = clf.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return np.argsort(-np.abs(contribs[:, :-1]).mean(axis=0))


def _fitpred(kind, Xtr, ytr, Xte, tab_clf=None):
    if kind == "lr":
        m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=0))
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    if kind == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    if kind == "tabpfn":
        tab_clf.fit(Xtr, ytr)
        return tab_clf.predict_proba(Xte)[:, 1]
    raise ValueError(kind)


def _outer_splits(X, y, outer, seed=0):
    """LOOCV (the submission default) or repeated stratified K-fold.

    LOOCV is fine at n=160 but pathological at n~15: holding out a member leaves a
    class-balanced training set while holding out a non-member leaves a
    member-heavy one, so any classifier that leans on the base rate produces
    perfectly ANTI-correlated held-out scores — which folding (max(a,1-a)) then
    reports as AUC 1.0. Repeated stratified K-fold keeps the training base rate
    constant and removes the artifact.
    """
    if outer == "loo":
        return list(LeaveOneOut().split(X))
    from sklearn.model_selection import RepeatedStratifiedKFold
    _, ns, nr = outer
    ns = int(min(ns, np.bincount(y, minlength=2).min()))
    return list(RepeatedStratifiedKFold(n_splits=max(2, ns), n_repeats=nr,
                                        random_state=seed).split(X, y))


def nested_audit(Xt, Xb, y, kind="lr", kgrid=None, inner_seed=0, tab_clf=None,
                 outer="loo"):
    """Nested leave-one-dataset-out audit. Returns
    (target_auc, blind_auc, gap, K* list, unfolded_target_auc, unfolded_blind_auc).

    THE AUDITOR NEVER SEES THE BLIND. Feature ranking, K* and every fit are
    derived from the target and the membership labels alone; the blind is then
    scored, as a control, on exactly the columns the auditor chose. It gets no
    ranking of its own and no say in K* — it is what we as paper writers evaluate
    the target's selected signals against, not a second auditor to be optimised.
    """
    n, p = Xt.shape
    kgrid = sorted({min(k, p) for k in (kgrid or KGRID_DEFAULT)})
    pt, pb, cnt, kstars = np.zeros(n), np.zeros(n), np.zeros(n), []
    for tr, te in _outer_splits(Xt, y, outer, seed=inner_seed):
        ytr = y[tr]
        # 5-fold when the fold sizes allow it; the red-team probe tables are tiny
        # (n~15) so the inner CV degrades gracefully rather than crashing.
        nf = int(min(5, np.bincount(ytr, minlength=2).min()))
        if nf < 2:
            raise ValueError("inner CV needs >=2 datasets of each class")
        inner = StratifiedKFold(nf, shuffle=True, random_state=inner_seed)
        it = {K: np.zeros(len(tr)) for K in kgrid}
        for itr, iva in inner.split(Xt[tr], ytr):
            o = shap_order(Xt[tr][itr], ytr[itr])
            for K in kgrid:
                it[K][iva] = _fitpred(kind, Xt[tr][itr][:, o[:K]], ytr[itr],
                                      Xt[tr][iva][:, o[:K]], tab_clf)
        Ks = max(kgrid, key=lambda K: (folded_auc(ytr, it[K]), -K))
        kstars.append(int(Ks))
        sel = shap_order(Xt[tr], ytr)[:Ks]
        pt[te] += _fitpred(kind, Xt[tr][:, sel], ytr, Xt[te][:, sel], tab_clf)
        pb[te] += _fitpred(kind, Xb[tr][:, sel], ytr, Xb[te][:, sel], tab_clf)
        cnt[te] += 1
    cnt = np.where(cnt > 0, cnt, 1.0)
    vt, vb = pt / cnt, pb / cnt
    at, ab = folded_auc(y, vt), folded_auc(y, vb)
    return at, ab, at - ab, kstars, directed_auc(y, vt), directed_auc(y, vb)


def internal_audit(ss: SignalSet, scope="full", kinds=("lr", "xgb"), kgrid=None,
                   nperm=0, njobs=1, perm_seed_base=5000, tab_clf=None,
                   null_only_if_positive=True, verbose=False, outer="loo"):
    """Run the supervised nested-LOOCV audit for every (target model, classifier).

    Reported per cell: held-out target AUC, blind AUC (same selected columns), gap,
    K* histogram, and — when nperm>0 — a selection-aware shuffled-label null of the
    gap (the whole nested pipeline is re-run on permuted labels).

    null_only_if_positive: a gap <= 0 cannot be significant in the one-sided test,
    so the (expensive) null is skipped and the cell is marked accordingly.
    """
    cols = ss.scopes[scope]
    Xb = ss.Xb[:, cols]
    kinds = tuple(kinds)
    par = (lambda jobs: _parallel(jobs, njobs)) if (njobs > 1 and tab_clf is None) \
        else (lambda jobs: [f() for f in jobs])

    obs_jobs, keys = [], []
    for mname, M in ss.Xt.items():
        Xt = M[:, cols]          # carried in `keys` — the null MUST use this exact
        for kind in kinds:       # matrix, not whatever `M` happens to be later
            keys.append((mname, kind, Xt))
            obs_jobs.append(_bind(nested_audit, Xt, Xb, ss.y, kind=kind, kgrid=kgrid,
                                  tab_clf=tab_clf, outer=outer))
    obs = par(obs_jobs)

    out, null_jobs, null_keys = {}, [], []
    for (mname, kind, Xt), (at, ab, gap, ks, rt, rb) in zip(keys, obs):
        cell = {"target_auc": round(float(at), 4), "blind_auc": round(float(ab), 4),
                "gap": round(float(gap), 4),
                "target_auc_unfolded": round(float(rt), 4),
                "blind_auc_unfolded": round(float(rb), 4),
                "kstar_hist": {str(k): int(ks.count(k)) for k in sorted(set(ks))}}
        if min(rt, rb) < 0.5:
            cell["WARN"] = ("held-out scores anti-correlate with membership "
                            "(unfolded AUC < 0.5) — folding inflates this cell; "
                            "typical LOOCV base-rate artifact at small n")
        out[f"{mname}|{kind}"] = cell
        if nperm:
            if null_only_if_positive and gap <= 0:
                cell["null"] = "skipped (gap<=0, p=1 by construction)"
                continue
            for d in range(nperm):
                null_keys.append((mname, kind))
                null_jobs.append(_bind(_one_null, Xt, Xb, ss.y, kind, kgrid, tab_clf,
                                       perm_seed_base + d, outer))
    if null_jobs:
        if verbose:
            print(f"    internal null: {len(null_jobs)} permutation runs", flush=True)
        vals = par(null_jobs)
        acc = {}
        for k, v in zip(null_keys, vals):
            acc.setdefault(k, []).append(v)
        for (mname, kind), lst in acc.items():
            nulls = np.asarray([z for z in lst if np.isfinite(z)])
            cell = out[f"{mname}|{kind}"]
            cell.update(null_mean=round(float(nulls.mean()), 4),
                        null_sd=round(float(nulls.std(ddof=1)), 4),
                        null_q95=round(float(np.quantile(nulls, 0.95)), 4),
                        p_gap=round(float((1 + (nulls >= cell["gap"]).sum()) / (1 + len(nulls))), 4),
                        nperm=int(len(nulls)))
    return {"scope": scope, "n_signals": len(cols),
            "selection": "target-only; blind scored on the target-selected columns",
            "kgrid": sorted({min(k, len(cols)) for k in (kgrid or KGRID_DEFAULT)}),
            "cells": out}


def _bind(fn, *a, **kw):
    return lambda: fn(*a, **kw)


def _parallel(jobs, njobs):
    from joblib import Parallel, delayed
    return list(Parallel(n_jobs=njobs)(delayed(f)() for f in jobs))


def _one_null(Xt, Xb, y, kind, kgrid, tab_clf, seed, outer="loo"):
    rng = np.random.default_rng(seed)
    yp = y[rng.permutation(len(y))]
    return nested_audit(Xt, Xb, yp, kind=kind, kgrid=kgrid,
                        inner_seed=1 + (seed % 1000), tab_clf=tab_clf, outer=outer)[2]


# ============================================================ EXTERNAL AUDIT
def _pu_score(Xknown_non, Xsuspect, robust=False):
    """Label-free score: z vs known-non, oriented by the unlabeled-pile shift.

    Returns (known_non_scores, suspect_scores). The direction of every signal is
    set by where the SUSPECT pile sits relative to the known non-members — the
    suspect pile is member-enriched by construction of the threat model, so its
    shift points member-ward. Member labels are never touched.
    """
    if robust:
        c = np.median(Xknown_non, axis=0)
        s = np.median(np.abs(Xknown_non - c), axis=0) * 1.4826
    else:
        c = Xknown_non.mean(axis=0)
        s = Xknown_non.std(axis=0, ddof=1)
    s = np.where(np.isfinite(s) & (s > 1e-12), s, 1.0)
    zk = (Xknown_non - c) / s
    zs = (Xsuspect - c) / s
    d = zs.mean(axis=0)                       # PU shift (label-free)
    sign = np.where(d >= 0, 1.0, -1.0)
    return (zk * sign).mean(axis=1), (zs * sign).mean(axis=1)


def _conformal_p(cal_scores, test_scores):
    """One-sided split-conformal p: how extreme is the suspect vs the known-non."""
    cal = np.sort(np.asarray(cal_scores))
    ge = len(cal) - np.searchsorted(cal, np.asarray(test_scores), side="left")
    return (1.0 + ge) / (len(cal) + 1.0)


def external_audit_once(X, y, rng, alphas=(0.01, 0.05, 0.10), known_frac=0.5,
                        robust=False):
    """One known-non / suspect draw of the label-free auditor on one model's X."""
    non = np.flatnonzero(y == 0)
    mem = np.flatnonzero(y == 1)
    k = max(2, int(round(known_frac * len(non))))
    perm = rng.permutation(non)
    known, held_non = perm[:k], perm[k:]
    suspects = np.concatenate([held_non, mem])
    sk, ss_ = _pu_score(X[known], X[suspects], robust=robust)
    ytrue = np.concatenate([np.zeros(len(held_non)), np.ones(len(mem))])
    res = {"auc": directed_auc(ytrue, ss_), "n_known_non": int(k)}
    pv = _conformal_p(sk, ss_)
    for a in alphas:
        flag = pv <= a
        res[f"tpr@{a}"] = float(flag[ytrue == 1].mean())
        res[f"fpr@{a}"] = float(flag[ytrue == 0].mean()) if (ytrue == 0).any() else float("nan")
    return res


def external_audit(ss: SignalSet, scope="full", nsplits=50, alphas=(0.01, 0.05, 0.10),
                   known_frac=0.5, seed=7, robust=False, nperm=0):
    """Label-free auditor on the target(s) and on the blind (identical method).

    The blind column is the control: if the public model's own signals let the
    same auditor rank suspects just as well, the ranking is distribution shift,
    not target leakage.
    """
    cols = ss.scopes[scope]
    models = {**{f"target:{k}": v[:, cols] for k, v in ss.Xt.items()},
              "blind": ss.Xb[:, cols]}
    agg = {}
    for mname, X in models.items():
        rows = [external_audit_once(X, ss.y, np.random.default_rng(seed + i),
                                    alphas, known_frac, robust)
                for i in range(nsplits)]
        agg[mname] = _summarize(rows, alphas)
    tgt_aucs = [v["auc_mean"] for k, v in agg.items() if k.startswith("target:")]
    out = {"scope": scope, "n_signals": len(cols), "nsplits": nsplits,
           "models": agg,
           "target_mean_auc": round(float(np.mean(tgt_aucs)), 4),
           "blind_auc": agg["blind"]["auc_mean"],
           "gap": round(float(np.mean(tgt_aucs) - agg["blind"]["auc_mean"]), 4)}
    if nperm:
        out["null"] = _external_null(ss, cols, nsplits, alphas, known_frac, seed,
                                     robust, nperm, out)
    return out


def _summarize(rows, alphas):
    def ms(key):
        v = np.array([r[key] for r in rows], float)
        v = v[np.isfinite(v)]
        if not len(v):
            return float("nan"), float("nan")
        return float(v.mean()), float(v.std(ddof=1))
    m, sd = ms("auc")
    out = {"auc_mean": round(m, 4), "auc_sd": round(sd, 4)}
    for a in alphas:
        out[f"tpr@{a}"] = round(ms(f"tpr@{a}")[0], 4)
        out[f"emp_fpr@{a}"] = round(ms(f"fpr@{a}")[0], 4)
    return out


def _external_null(ss, cols, nsplits, alphas, known_frac, seed, robust, nperm, obs):
    """Shuffled-membership null for (a) the target auditor AUC and (b) the
    target-minus-blind gap. Permuting y re-draws which datasets are known-non,
    so the whole auditor is re-run under the null."""
    Xts = [v[:, cols] for v in ss.Xt.values()]
    Xb = ss.Xb[:, cols]
    ns = max(5, nsplits // 5)
    a_null, g_null = [], []
    for d in range(nperm):
        rng = np.random.default_rng(90000 + d)
        yp = ss.y[rng.permutation(ss.n)]
        ta = []
        for X in Xts:
            ta.append(np.mean([external_audit_once(X, yp, np.random.default_rng(seed + i),
                                                   alphas, known_frac, robust)["auc"]
                               for i in range(ns)]))
        ba = np.mean([external_audit_once(Xb, yp, np.random.default_rng(seed + i),
                                          alphas, known_frac, robust)["auc"]
                      for i in range(ns)])
        a_null.append(float(np.mean(ta)))
        g_null.append(float(np.mean(ta) - ba))
    a_null, g_null = np.array(a_null), np.array(g_null)
    return {
        "nperm": nperm, "nsplits_per_perm": ns,
        "auc_null_mean": round(float(a_null.mean()), 4),
        "auc_null_q95": round(float(np.quantile(a_null, 0.95)), 4),
        "p_auc": round(float((1 + (a_null >= obs["target_mean_auc"]).sum()) / (1 + nperm)), 4),
        "gap_null_mean": round(float(g_null.mean()), 4),
        "gap_null_sd": round(float(g_null.std(ddof=1)), 4),
        "gap_null_q95": round(float(np.quantile(g_null, 0.95)), 4),
        "p_gap": round(float((1 + (g_null >= obs["gap"]).sum()) / (1 + nperm)), 4),
    }


# ------------------------------------------------------------------- per-signal
def per_signal_table(ss: SignalSet, scope="full", top=25):
    """Optimistic single-signal target-vs-blind ranking (diagnostic only — it is
    in-sample selection, so treat the top entries as an upper bound)."""
    cols = ss.scopes[scope]
    rows = []
    Xts = list(ss.Xt.values())
    for j in cols:
        t = float(np.mean([folded_auc(ss.y, X[:, j]) for X in Xts]))
        b = folded_auc(ss.y, ss.Xb[:, j])
        rows.append({"signal": ss.sigs[j], "target": round(t, 4),
                     "blind": round(b, 4), "gap": round(t - b, 4)})
    rows.sort(key=lambda r: -r["gap"])
    return {"n_signals": len(rows), "mean_gap": round(float(np.mean([r["gap"] for r in rows])), 4),
            "frac_positive": round(float(np.mean([r["gap"] > 0 for r in rows])), 4),
            "top": rows[:top], "bottom": rows[-5:]}


def atomic_dump(obj, path):
    import json
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)
