"""Internal audit table: all signals, then each signal family, x LR / XGB / TabPFN-2.5.

Protocol (identical for every cell):
  * stratified K-fold over datasets (5 by default)
  * per fold: fit XGBoost on the training rows -> TreeSHAP ranking (target only;
    the blind never ranks anything)
  * for each K in the grid: fit the classifier on the target's top-K columns,
    predict the held-out rows; score the BLIND on exactly the same columns
  * pool the out-of-fold predictions -> one target AUC and one blind AUC per K
  * K* = elbow of the target-AUC-vs-K curve (max distance above the chord on
    log2 K); the blind is never consulted in choosing it
  * paired bootstrap over datasets -> sd of the gap

Nothing is dropped beyond the label, ids and table-shape/timing metadata (see
baseline_topk.NON_FEATURES). `ctx` is NOT removed -- it appears as its own family
row, which is more informative than deleting it.

  python di/family_audit.py --sets tabdpt,sap,nano,realtabpfn
  python di/family_audit.py --sets sap --kinds lr,xgb --nboot 2000
"""
import os

for _v in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "XGBOOST_NTHREAD"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from di.baseline_topk import (LOADERS, XGB_PARAMS, fam_of,  # noqa: E402
                               fitpred)

# the four model families, and which loader gives target-vs-blind for each
SETS = {
    "tabdpt":     ("tabdpt_fullgrid",   "default", "TabDPT vs TabPFN-2.5"),
    "sap":        ("sap_t4_allcls",     None,      "SAP RPT-OSS vs TabPFN-2.5 (native-cls + forced-cls)"),
    "sap_hybrid": ("sap_t4_allhybrid",  None,      "SAP RPT-OSS vs TabPFN-2.5 (native-cls + hybrid)"),
    "nano":       ("nano_matched",      None,      "NanoTabPFN vs TabPFN-2.5 (matched 20k, n_pool=50)"),
    "realtabpfn": ("realtabpfn",        None,      "RealTabPFN-2.5 vs synthetic TabPFN-2.5"),
}


def rank(X, y):
    c = xgb.XGBClassifier(**XGB_PARAMS)
    c.fit(X, y)
    return np.argsort(-np.abs(c.get_booster().predict(
        xgb.DMatrix(X), pred_contribs=True)[:, :-1]).mean(0))


def elbow(Ks, aucs):
    x = np.log2(np.asarray(Ks, float))
    y = np.asarray(aucs, float)
    if len(x) < 3 or y.max() == y.min():
        return Ks[int(np.argmax(y))]
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    return Ks[int(np.argmax(yn - (xn * (yn[-1] - yn[0]) + yn[0])))]


def _safe(kind, Xtr, ytr, Xte, tab):
    """A constant or all-NaN column set makes TabPFN's internal SVD fail
    (ArpackError: starting vector is zero). Those cells carry no information, so
    fall back to a chance prediction rather than losing the whole run."""
    try:
        return fitpred(kind, Xtr, ytr, Xte, tab)
    except Exception:
        return np.full(len(Xte), 0.5)


def audit(Xt, Xb, y, kind, KS, folds, tab, nboot, rng):
    """One (family, classifier) cell: elbow-selected K + bootstrap sd of the gap.
    Returns None when the family has no usable (non-constant) column."""
    ok = np.array([np.nanstd(Xt[:, j]) > 0 or np.nanstd(Xb[:, j]) > 0
                   for j in range(Xt.shape[1])])
    if not ok.any():
        return None
    Xt, Xb = Xt[:, ok], Xb[:, ok]
    fold_rank = [rank(Xt[tr], y[tr]) for tr, _ in folds]
    cur = {}
    for K in [k for k in KS if k <= Xt.shape[1]]:
        pt, pb = np.zeros(len(y)), np.zeros(len(y))
        for i, (tr, te) in enumerate(folds):
            sel = fold_rank[i][:K]
            pt[te] = _safe(kind, Xt[tr][:, sel], y[tr], Xt[te][:, sel], tab)
            pb[te] = _safe(kind, Xb[tr][:, sel], y[tr], Xb[te][:, sel], tab)
        cur[K] = (pt, pb, roc_auc_score(y, pt), roc_auc_score(y, pb))
    Ks = sorted(cur)
    K = elbow(Ks, [cur[k][2] for k in Ks])
    pt, pb, t, b = cur[K]
    idx = rng.integers(0, len(y), size=(nboot, len(y)))
    g = []
    for i in range(nboot):
        s = idx[i]
        if len(np.unique(y[s])) < 2:
            continue
        g.append(roc_auc_score(y[s], pt[s]) - roc_auc_score(y[s], pb[s]))
    g = np.asarray(g)
    return {"K": int(K), "target": round(float(t), 4), "blind": round(float(b), 4),
            "gap": round(float(t - b), 4), "gap_sd": round(float(g.std()), 4),
            "p_gap_le_0": round(float(np.mean(g <= 0)), 4),
            "curve": {int(k): [round(float(cur[k][2]), 4), round(float(cur[k][3]), 4)]
                      for k in Ks}}


def render(name, title, meta, cells, kinds):
    w = 22
    out = [f"\n{'='*84}", f"{title}",
           f"n = {meta['n']} ({meta['n_member']} member / {meta['n']-meta['n_member']} non), "
           f"{meta['n_signals']} signals, {meta['folds']}-fold, {meta['nboot']} bootstrap draws",
           "="*84,
           f"{'family':16s} {'cols':>5s} | " + " | ".join(f"{k.upper():^{w}s}" for k in kinds)]
    for fam, row in cells.items():
        line = f"{fam:16s} {row['n_cols']:5d} |"
        for k in kinds:
            c = row.get(k)
            line += (f" {c['gap']:+.3f}±{c['gap_sd']:.3f} ({c['target']:.3f}/{c['blind']:.3f}) |"
                     if c else f" {'—':^{w}s} |")
        out.append(line)
    out.append("  gap = target AUC - blind AUC, +- bootstrap sd; K chosen by the elbow of "
               "the target-AUC curve")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="tabdpt,sap,nano,realtabpfn")
    ap.add_argument("--kinds", default="lr,xgb,tabpfn")
    ap.add_argument("--ks", default="1,2,4,8,16,32,64")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--min-family", type=int, default=5,
                    help="skip families with fewer than this many columns")
    ap.add_argument("--device", default="cuda")
    _E = Path(os.environ.get("AUDIT_REPO", "/p/project1/hai_1159/tfm-di")) / "exports"
    ap.add_argument("--out", default=str(_E / "hyp_family_audit.json"))
    ap.add_argument("--txt", default=str(_E / "FAMILY_AUDIT.txt"))
    a = ap.parse_args()

    kinds = tuple(a.kinds.split(","))
    KS = [int(k) for k in a.ks.split(",")]
    tab = None
    if "tabpfn" in kinds:
        from di.new_signal_audits import make_tabpfn
        tab = make_tabpfn(a.device, 4)
    rng = np.random.default_rng(0)
    results, texts = {}, []

    for key in a.sets.split(","):
        loader_name, model, title = SETS[key]
        ids, y, sigs, Xts, Xb = LOADERS[loader_name]()
        if model:
            Xts = {model: Xts[model]}
        Xt_all = list(Xts.values())[0]
        folds = list(StratifiedKFold(min(a.folds, int(np.bincount(y).min())),
                                     shuffle=True, random_state=0).split(Xt_all, y))
        fams = {}
        for j, s in enumerate(sigs):
            fams.setdefault(fam_of(s), []).append(j)
        scopes = [("ALL", list(range(len(sigs))))] + [
            (f, c) for f, c in sorted(fams.items()) if len(c) >= a.min_family]
        meta = {"n": len(y), "n_member": int(y.sum()), "n_signals": len(sigs),
                "folds": len(folds), "nboot": a.nboot, "loader": loader_name,
                "model": model or "target"}
        cells = {}
        for fam, cols in scopes:
            cells[fam] = {"n_cols": len(cols)}
            for kind in kinds:
                t0 = time.time()
                cells[fam][kind] = audit(Xt_all[:, cols], Xb[:, cols], y, kind,
                                         KS, folds, tab, a.nboot, rng)
                c = cells[fam][kind]
                if c is None:
                    print(f"  {key:11s} {fam:14s} {kind:6s} skipped (all columns constant)",
                          flush=True); continue
                print(f"  {key:11s} {fam:14s} {kind:6s} K={c['K']:<3d} "
                      f"{c['target']:.3f}/{c['blind']:.3f} gap {c['gap']:+.3f}"
                      f"±{c['gap_sd']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
            results[key] = {"meta": meta, "title": title, "cells": cells}
            Path(a.out).write_text(json.dumps(results, indent=1))
        texts.append(render(key, title, meta, cells, kinds))
        Path(a.txt).write_text("\n".join(texts))
        print(texts[-1], flush=True)


if __name__ == "__main__":
    main()
