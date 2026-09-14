"""Repeated global train/test split: rank on train, fit on train, score on test.

One feature list per split (unlike per-fold ranking), and the test half touches
neither the ranking nor the fit. Averaged over many random splits so the estimate
isn't hostage to one draw. Also counts how often each signal survives into the
top-K -- the vote, which doubles as a stability diagnostic.
"""
import os
for _v in ("OMP_NUM_THREADS","OMP_THREAD_LIMIT","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import argparse, json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from di.baseline_topk import LOADERS, fitpred, XGB_PARAMS

def rank(X, y):
    c = xgb.XGBClassifier(**XGB_PARAMS); c.fit(X, y)
    return np.argsort(-np.abs(c.get_booster().predict(
        xgb.DMatrix(X), pred_contribs=True)[:, :-1]).mean(0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="tabdpt_fullgrid")
    ap.add_argument("--models", default="default")
    ap.add_argument("--ks", default="1,2,4,6,8,12,16,24,32,48,64")
    ap.add_argument("--kinds", default="lr,xgb,tabpfn")
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    kinds = tuple(a.kinds.split(",")); KS = [int(k) for k in a.ks.split(",")]
    tab = None
    if "tabpfn" in kinds:
        from di.new_signal_audits import make_tabpfn
        tab = make_tabpfn(a.device, 4)
    ids, y, sigs, Xts, Xb = LOADERS[a.set]()
    if a.models: Xts = {k: v for k, v in Xts.items() if k in set(a.models.split(","))}
    sss = list(StratifiedShuffleSplit(n_splits=a.splits, test_size=a.test_size,
                                      random_state=0).split(Xb, y))
    out = {"meta": {"set": a.set, "n": len(y), "n_member": int(y.sum()),
                    "n_signals": len(sigs), "splits": a.splits,
                    "test_size": a.test_size, "created": time.strftime("%Y-%m-%d %H:%M")},
           "rows": [], "votes": {}}
    for mname, Xt in Xts.items():
        acc = {(K, k): {"t": [], "b": []} for K in KS for k in kinds}
        votes = Counter()
        for si, (tr, te) in enumerate(sss):
            o = rank(Xt[tr], y[tr])                     # ranking: TRAIN ONLY
            votes.update(int(j) for j in o[:8])
            for K in [k for k in KS if k <= Xt.shape[1]]:
                sel = o[:K]
                for kind in kinds:
                    pt = fitpred(kind, Xt[tr][:, sel], y[tr], Xt[te][:, sel], tab)
                    pb = fitpred(kind, Xb[tr][:, sel], y[tr], Xb[te][:, sel], tab)
                    acc[(K, kind)]["t"].append(roc_auc_score(y[te], pt))
                    acc[(K, kind)]["b"].append(roc_auc_score(y[te], pb))
            if (si+1) % 20 == 0: print(f"  {mname}: {si+1}/{a.splits} splits", flush=True)
        for (K, kind), v in acc.items():
            t, b = np.mean(v["t"]), np.mean(v["b"])
            out["rows"].append({"model": mname, "K": K, "clf": kind,
                                "target": round(float(t),4), "blind": round(float(b),4),
                                "gap": round(float(t-b),4),
                                "target_sd": round(float(np.std(v["t"])),4),
                                "gap_sd": round(float(np.std(np.array(v["t"])-np.array(v["b"]))),4)})
            print(f"  {mname:9s} K={K:<3d} {kind:6s} target {t:.3f} blind {b:.3f} "
                  f"gap {t-b:+.3f} (sd {np.std(np.array(v['t'])-np.array(v['b'])):.3f})", flush=True)
        out["votes"][mname] = [{"signal": sigs[j], "in_top8_of_splits": c,
                                "pct": round(100*c/a.splits,1)} for j, c in votes.most_common(20)]
    p = Path(a.out) if a.out else Path(f"/p/project1/hai_1159/tfm-di/exports/hyp_holdout_{a.set}.json")
    p.write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
