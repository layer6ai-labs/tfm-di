"""How much of the target-blind gap is the SIGNAL, and how much is the SELECTION?

Same pool, same folds, same classifiers — only the honesty of the selection changes:

  1 insample_fixedK   ranking fitted on ALL rows; K fixed by hand
  2 perfold_fixedK    ranking refitted inside each training fold; K still fixed
  3 nested            ranking refitted inside each fold AND K chosen by an inner
                      CV on the training rows only  (the fully honest auditor)

Step 1->2 prices the feature ranking; step 2->3 prices the choice of K.
"""
import os
for _v in ("OMP_NUM_THREADS","OMP_THREAD_LIMIT","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys, time
from pathlib import Path
import numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from di.baseline_topk import LOADERS, XGB_PARAMS, fitpred, scope_cols

def rank(X, y):
    c = xgb.XGBClassifier(**XGB_PARAMS); c.fit(X, y)
    return np.argsort(-np.abs(c.get_booster().predict(
        xgb.DMatrix(X), pred_contribs=True)[:, :-1]).mean(0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="tabdpt_fullgrid")
    ap.add_argument("--scope", default="full")
    ap.add_argument("--models", default="")
    ap.add_argument("--kinds", default="lr,xgb,tabpfn")
    ap.add_argument("--fixed-ks", default="8,16,32")
    ap.add_argument("--kgrid", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    kinds = tuple(a.kinds.split(","))
    tab = None
    if "tabpfn" in kinds:
        from di.new_signal_audits import make_tabpfn
        tab = make_tabpfn(a.device, 4)
    KG = [int(k) for k in a.kgrid.split(",")]
    out = {"meta": {"created": time.strftime("%Y-%m-%d %H:%M"), "args": vars(a)}, "rows": []}
    for name in a.sets.split(","):
        ids, y, sigs, Xts, Xb0 = LOADERS[name]()
        if a.models:
            Xts = {k: v for k, v in Xts.items() if k in set(a.models.split(","))}
        cols = scope_cols(sigs, a.scope); Xb = Xb0[:, cols]
        splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=4,
                                              random_state=0).split(Xb, y))
        for mname, M in Xts.items():
            Xt = M[:, cols]
            order_all = rank(Xt, y)
            fold_rank = [rank(Xt[tr], y[tr]) for tr, _ in splits]
            for kind in kinds:
                # ---- modes 1 and 2: fixed K, ranking either global or per-fold
                for K in [int(k) for k in a.fixed_ks.split(",")]:
                    for mode, ords in (("insample_fixedK", None), ("perfold_fixedK", fold_rank)):
                        pt, pb = np.zeros(len(y)), np.zeros(len(y))
                        for i, (tr, te) in enumerate(splits):
                            sel = (order_all if ords is None else ords[i])[:K]
                            pt[te] += fitpred(kind, Xt[tr][:, sel], y[tr], Xt[te][:, sel], tab)
                            pb[te] += fitpred(kind, Xb[tr][:, sel], y[tr], Xb[te][:, sel], tab)
                        t, b = roc_auc_score(y, pt), roc_auc_score(y, pb)
                        out["rows"].append({"set": name, "model": mname, "clf": kind,
                                            "mode": mode, "K": K, "target": round(float(t), 4),
                                            "blind": round(float(b), 4),
                                            "gap": round(float(t - b), 4)})
                        print(f"  {mname:9s} {kind:6s} {mode:16s} K={K:<4d} "
                              f"tgt {t:.3f} blind {b:.3f} gap {t-b:+.3f}", flush=True)
                # ---- mode 3: nested — inner CV on the training rows picks K
                pt, pb, ks = np.zeros(len(y)), np.zeros(len(y)), []
                for i, (tr, te) in enumerate(splits):
                    ytr = y[tr]
                    inner = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xt[tr], ytr))
                    sc = {}
                    for K in [k for k in KG if k <= len(cols)]:
                        p = np.zeros(len(tr))
                        for itr, iva in inner:
                            s = rank(Xt[tr][itr], ytr[itr])[:K]
                            p[iva] = fitpred(kind, Xt[tr][itr][:, s], ytr[itr],
                                             Xt[tr][iva][:, s], tab)
                        sc[K] = roc_auc_score(ytr, p)          # TARGET only
                    Ks = max(sc, key=lambda k: (sc[k], -k)); ks.append(Ks)
                    sel = fold_rank[i][:Ks]
                    pt[te] += fitpred(kind, Xt[tr][:, sel], ytr, Xt[te][:, sel], tab)
                    pb[te] += fitpred(kind, Xb[tr][:, sel], ytr, Xb[te][:, sel], tab)
                t, b = roc_auc_score(y, pt), roc_auc_score(y, pb)
                from collections import Counter
                out["rows"].append({"set": name, "model": mname, "clf": kind, "mode": "nested",
                                    "K": None, "kstar_hist": dict(Counter(ks)),
                                    "target": round(float(t), 4), "blind": round(float(b), 4),
                                    "gap": round(float(t - b), 4)})
                print(f"  {mname:9s} {kind:6s} {'nested':16s} K*={dict(Counter(ks))} "
                      f"tgt {t:.3f} blind {b:.3f} gap {t-b:+.3f}", flush=True)
    p = Path(a.out) if a.out else Path("/p/project1/hai_1159/tfm-di/exports/hyp_ladder.json")
    p.write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
