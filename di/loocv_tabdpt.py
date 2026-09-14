"""True leave-one-dataset-out audit on the TabDPT grid (n=160), matching the
submission's protocol rather than approximating it with k-fold.

Two ranking modes, same folds and same K grid:
  global   XGBoost/TreeSHAP fitted once on all 160 -> fixed column order
  perfold  refitted on the 159 training datasets inside every fold

The blind is always scored on the columns the TARGET's ranking selected; it never
ranks anything. K is the elbow of the LOOCV target-AUC curve (1-SE is unavailable:
a single held-out dataset has no fold-level AUC). AUCs are reported unfolded so a
base-rate flip would be visible as a value below 0.5.
"""
import os
for _v in ("OMP_NUM_THREADS","OMP_THREAD_LIMIT","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import argparse, json, sys, time
from pathlib import Path
import numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from di.baseline_topk import load_tabdpt, fitpred, XGB_PARAMS

def rank(X, y):
    c = xgb.XGBClassifier(**XGB_PARAMS); c.fit(X, y)
    return np.argsort(-np.abs(c.get_booster().predict(
        xgb.DMatrix(X), pred_contribs=True)[:, :-1]).mean(0))

def elbow(Ks, a):
    x = np.log2(np.asarray(Ks, float)); y = np.asarray(a, float)
    if len(x) < 3 or y.max() == y.min(): return Ks[int(np.argmax(y))]
    xn = (x-x.min())/(x.max()-x.min()); yn = (y-y.min())/(y.max()-y.min())
    return Ks[int(np.argmax(yn - (xn*(yn[-1]-yn[0]) + yn[0])))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="1,2,4,6,8,12,16,24,32,48,64")
    ap.add_argument("--kinds", default="lr,xgb,tabpfn")
    ap.add_argument("--models", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/p/project1/hai_1159/tfm-di/exports/hyp_loocv_tabdpt.json")
    a = ap.parse_args()
    kinds = tuple(a.kinds.split(",")); KS = [int(k) for k in a.ks.split(",")]
    tab = None
    if "tabpfn" in kinds:
        from di.new_signal_audits import make_tabpfn
        tab = make_tabpfn(a.device, 4)
    ids, y, sigs, Xts, Xb = load_tabdpt()
    if a.models: Xts = {k: v for k, v in Xts.items() if k in set(a.models.split(","))}
    splits = list(LeaveOneOut().split(Xb))
    out = {"meta": {"n": len(y), "n_member": int(y.sum()), "n_signals": len(sigs),
                    "folds": len(splits), "created": time.strftime("%Y-%m-%d %H:%M")}, "rows": []}
    for mname, Xt in Xts.items():
        gl = rank(Xt, y)
        fo = [rank(Xt[tr], y[tr]) for tr, _ in splits]      # 160 per-fold rankings
        for mode, orders in (("global", None), ("perfold", fo)):
            for K in [k for k in KS if k <= Xt.shape[1]]:
                for kind in kinds:
                    pt, pb = np.zeros(len(y)), np.zeros(len(y))
                    for i, (tr, te) in enumerate(splits):
                        sel = (gl if orders is None else orders[i])[:K]
                        pt[te] = fitpred(kind, Xt[tr][:, sel], y[tr], Xt[te][:, sel], tab)
                        pb[te] = fitpred(kind, Xb[tr][:, sel], y[tr], Xb[te][:, sel], tab)
                    t, b = float(roc_auc_score(y, pt)), float(roc_auc_score(y, pb))
                    out["rows"].append({"model": mname, "mode": mode, "K": K, "clf": kind,
                                        "target": round(t,4), "blind": round(b,4),
                                        "gap": round(t-b,4)})
                    print(f"  {mname:9s} {mode:8s} K={K:<3d} {kind:6s} "
                          f"target {t:.3f} blind {b:.3f} gap {t-b:+.3f}", flush=True)
        out["rows"].append({"model": mname, "global_top16": [sigs[j] for j in gl[:16]]})
        Path(a.out).write_text(json.dumps(out, indent=1))
    Path(a.out).write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
