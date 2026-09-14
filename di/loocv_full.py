"""Fully nested leave-one-dataset-out audit. One XGBoost fit per fold supplies
both the ranking and K; nothing about the held-out dataset enters either.

Per outer fold (160 of them):
  fit XGBoost on the 159 training datasets -> TreeSHAP importances
  K = elbow of the sorted importance profile (no inner CV, no AUC curve)
  A   fit on the 159 at the target's top-K, predict the held-out dataset
  B   same K, same columns, on the BLIND's matrix   (blind as passive control)
  C   repeat on the BLIND -- its own importances, its own elbow, its own K --
      and predict the held-out dataset from the blind alone (blind as auditor)

gap_AB = A - B is the usual definition; gap_AC = A - C asks whether the target
beats what the same procedure achieves with only the public model.
"""
import os
for _v in ("OMP_NUM_THREADS","OMP_THREAD_LIMIT","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import argparse, json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from di.baseline_topk import LOADERS, fitpred, XGB_PARAMS

def shap_importance(X, y):
    c = xgb.XGBClassifier(**XGB_PARAMS); c.fit(X, y)
    return np.abs(c.get_booster().predict(
        xgb.DMatrix(X), pred_contribs=True)[:, :-1]).mean(0)

def importance_elbow(imp, kmax=None):
    """K = knee of the sorted |SHAP| profile. The curve falls steeply then flattens;
    the knee is the point furthest above the chord joining its two endpoints."""
    o = np.argsort(-imp); v = imp[o]
    nz = int((v > 0).sum())
    v = v[:max(3, min(nz, kmax or len(v)))]
    if len(v) < 3 or v[0] == v[-1]:
        return o, max(1, len(v))
    x = np.arange(len(v), dtype=float); xn = x / x[-1]
    yn = (v - v[-1]) / (v[0] - v[-1])
    return o, int(np.argmax(yn - (1.0 - xn))) + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="tabdpt_fullgrid")
    ap.add_argument("--models", default="default")
    ap.add_argument("--kinds", default="lr,xgb,tabpfn")
    ap.add_argument("--kmax", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    kinds = tuple(a.kinds.split(","))
    tab = None
    if "tabpfn" in kinds:
        from di.new_signal_audits import make_tabpfn
        tab = make_tabpfn(a.device, 4)
    ids, y, sigs, Xts, Xb = LOADERS[a.set]()
    if a.models: Xts = {k: v for k, v in Xts.items() if k in set(a.models.split(","))}
    splits = list(LeaveOneOut().split(Xb))
    out = {"meta": {"set": a.set, "n": len(y), "n_member": int(y.sum()),
                    "n_signals": len(sigs), "folds": len(splits), "kmax": a.kmax,
                    "selection": "K = elbow of the per-fold SHAP importance profile",
                    "created": time.strftime("%Y-%m-%d %H:%M")}, "rows": []}
    for mname, Xt in Xts.items():
        # one XGBoost fit per fold per model -- shared across classifiers
        print(f"  {mname}: ranking {len(splits)} folds ...", flush=True)
        selT, selB = [], []
        for tr, _ in splits:
            ot, Kt = importance_elbow(shap_importance(Xt[tr], y[tr]), a.kmax)
            ob, Kb = importance_elbow(shap_importance(Xb[tr], y[tr]), a.kmax)
            selT.append((ot[:Kt], Kt)); selB.append((ob[:Kb], Kb))
        kt = Counter(k for _, k in selT); kb = Counter(k for _, k in selB)
        votes = Counter(int(j) for s, _ in selT for j in s[:8])
        for kind in kinds:
            pA, pB, pC = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
            t0 = time.time()
            for i, (tr, te) in enumerate(splits):
                st, _ = selT[i]; sb, _ = selB[i]
                pA[te] = fitpred(kind, Xt[tr][:, st], y[tr], Xt[te][:, st], tab)
                pB[te] = fitpred(kind, Xb[tr][:, st], y[tr], Xb[te][:, st], tab)
                pC[te] = fitpred(kind, Xb[tr][:, sb], y[tr], Xb[te][:, sb], tab)
            A, B, C = (float(roc_auc_score(y, p)) for p in (pA, pB, pC))
            out["rows"].append({"model": mname, "clf": kind,
                "A_target": round(A,4), "B_blind_target_feats": round(B,4),
                "C_blind_self": round(C,4), "gap_AB": round(A-B,4), "gap_AC": round(A-C,4),
                "Kstar_target": dict(kt), "Kstar_blind": dict(kb),
                "top_votes": [{"signal": sigs[j], "folds": c} for j, c in votes.most_common(10)],
                "secs": round(time.time()-t0)})
            print(f"  {mname} {kind:6s} A {A:.3f} | B {B:.3f} (A-B {A-B:+.3f}) "
                  f"| C {C:.3f} (A-C {A-C:+.3f})", flush=True)
            Path(a.out or f"/p/project1/hai_1159/tfm-di/exports/hyp_loocvfull_{a.set}.json"
                 ).write_text(json.dumps(out, indent=1))
        print(f"  K* target {dict(kt)}\n  K* blind  {dict(kb)}", flush=True)

if __name__ == "__main__":
    main()
