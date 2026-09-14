"""Masked-coverage audit: the blind is far patchier than the target on the parquet
extractions, and the cells it lacks are where the discriminative information sits.
NaN the target wherever the blind is NaN so both are asked exactly the same
questions, then re-run the same elbow top-K audit. Reports raw and masked side by
side with a paired dataset bootstrap.

Separate file on purpose -- di/baseline_topk.py is being edited concurrently.
"""
import os
for _v in ("OMP_NUM_THREADS","OMP_THREAD_LIMIT","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v,"8")
import argparse, json, sys, time
from pathlib import Path
import numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from di.baseline_topk import LOADERS, fitpred, XGB_PARAMS, scope_cols

def rank(X, y, nt=8):
    P = dict(XGB_PARAMS); P["n_jobs"] = nt
    c = xgb.XGBClassifier(**P); c.fit(X, y)
    return np.argsort(-np.abs(c.get_booster().predict(
        xgb.DMatrix(X, nthread=nt), pred_contribs=True)[:, :-1]).mean(0))

def elbow(Ks, a):
    x = np.log2(np.asarray(Ks, float)); y = np.asarray(a, float)
    if len(x) < 3 or y.max() == y.min(): return Ks[int(np.argmax(y))]
    xn = (x-x.min())/(x.max()-x.min()); yn = (y-y.min())/(y.max()-y.min())
    return Ks[int(np.argmax(yn - (xn*(yn[-1]-yn[0]) + yn[0])))]

def run(Xt, Xb, y, KS, kinds, tab, nfold=5):
    o = rank(Xt, y)
    sp = list(StratifiedKFold(nfold, shuffle=True, random_state=0).split(Xt, y))
    out = {}
    for kind in kinds:
        curve = {}
        for K in KS:
            sel = o[:K]; pt, pb = np.zeros(len(y)), np.zeros(len(y))
            for tr, te in sp:
                pt[te] = fitpred(kind, Xt[tr][:, sel], y[tr], Xt[te][:, sel], tab)
                pb[te] = fitpred(kind, Xb[tr][:, sel], y[tr], Xb[te][:, sel], tab)
            curve[K] = (roc_auc_score(y, pt), roc_auc_score(y, pb), pt, pb)
        K = elbow(sorted(curve), [curve[k][0] for k in sorted(curve)])
        out[kind] = (K,) + curve[K]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="sap_t4_hybrid,sap_t4_cls_hybrid")
    ap.add_argument("--ks", default="1,2,4,8,16,32,64")
    ap.add_argument("--kinds", default="lr,xgb,tabpfn")
    ap.add_argument("--drop-fams", default="ctx", help="comma list, or empty")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/p/project1/hai_1159/tfm-di/exports/hyp_masked_audit.json")
    a = ap.parse_args()
    KS = [int(k) for k in a.ks.split(",")]; kinds = tuple(a.kinds.split(","))
    tab = None
    if "tabpfn" in kinds:
        from di.new_signal_audits import make_tabpfn
        tab = make_tabpfn(a.device, 4)
    rng = np.random.default_rng(0)
    res = {"meta": {"created": time.strftime("%Y-%m-%d %H:%M"), "args": vars(a)}, "rows": []}
    for name in a.sets.split(","):
        ids, y, sigs, Xts, Xb0 = LOADERS[name]()
        Xt0 = list(Xts.values())[0]
        if a.drop_fams:
            cols = scope_cols(sigs, "no:" + a.drop_fams.replace(",", "+"))
            Xt0, Xb0, sigs = Xt0[:, cols], Xb0[:, cols], [sigs[j] for j in cols]
        cov_t, cov_b = np.mean(~np.isfinite(Xt0)), np.mean(~np.isfinite(Xb0))
        # masked: NaN the target wherever the blind is NaN -> identical coverage
        Xt_m = Xt0.copy(); Xt_m[~np.isfinite(Xb0)] = np.nan
        print(f"\n### {name}  n={len(y)} ({int(y.sum())} member)  {len(sigs)} signals"
              f"  NaN tgt {100*cov_t:.2f}% blind {100*cov_b:.2f}%", flush=True)
        for mode, Xt in (("raw", Xt0), ("masked", Xt_m)):
            r = run(Xt, Xb0, y, KS, kinds, tab)
            for kind, (K, T, B, pt, pb) in r.items():
                bs = []
                for _ in range(a.boot):
                    idx = rng.integers(0, len(y), len(y))
                    if len(np.unique(y[idx])) < 2: continue
                    bs.append(roc_auc_score(y[idx], pt[idx]) - roc_auc_score(y[idx], pb[idx]))
                sd = float(np.std(bs, ddof=1))
                res["rows"].append({"set": name, "mode": mode, "clf": kind, "K": K,
                                    "target": round(T,4), "blind": round(B,4),
                                    "gap": round(T-B,4), "gap_sd": round(sd,4),
                                    "nan_target": round(float(cov_t),4),
                                    "nan_blind": round(float(cov_b),4)})
                print(f"  {mode:7s} {kind.upper():7s} K={K:<3d} {T:.3f}/{B:.3f} "
                      f"{T-B:+.3f} +/-{sd:.3f}", flush=True)
                Path(a.out).write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
