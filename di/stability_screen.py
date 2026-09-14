"""Reject unstable signals using ONLY the known non-members.

Split the known non-members at random into two halves and measure the separation
between them on each signal. Both halves are non-members, so the honest answer is
0.5; anything above that is the signal generating rank structure out of nothing —
heavy tails, outliers, a few extreme datasets carrying the ordering. Repeat B
times to get a per-signal null distribution and REJECT signals whose upper tail
runs hotter than the pool as a whole.

The suspect pile is never consulted, so this cannot select for membership, only
against instability.
"""
import os
for _v in ("OMP_NUM_THREADS","OMP_THREAD_LIMIT","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from di.baseline_topk import LOADERS, REPO, fam_of
from di.external_family_audit import impute, iforest_scores, one_draw

def null_dist(Xk, B, seed=0):
    """Per-signal |AUC-0.5| over B random half-splits of the known non-members."""
    rng = np.random.default_rng(seed); n = len(Xk); h = n//2
    D = np.empty((B, Xk.shape[1]))
    for b in range(B):
        p = rng.permutation(n); A, C = p[:h], p[h:]
        z = np.concatenate([np.zeros(len(A)), np.ones(len(C))])
        for j in range(Xk.shape[1]):
            D[b, j] = abs(roc_auc_score(z, np.concatenate([Xk[A, j], Xk[C, j]])) - 0.5)
    return D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="sap_t4_forcecls,tabdpt_fullgrid,realtabpfn")
    ap.add_argument("--family", default="tshuffle")
    ap.add_argument("--nullboot", type=int, default=200)
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--boot", type=int, default=0, help="bootstrap resamples for a CI on the gap")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = {"meta": vars(a), "rows": []}
    for name in a.sets.split(","):
        ids, y, sigs, Xts, Xb = LOADERS[name]()
        cols = [j for j, s in enumerate(sigs) if (not a.family) or fam_of(s) == a.family]
        keep = np.isfinite(Xb[:, cols]).mean(1) > 0
        T = impute(list(Xts.values())[0][keep][:, cols]); B = impute(Xb[keep][:, cols]); yy = y[keep]
        names = [sigs[j] for j in cols]
        rng = np.random.default_rng(1)
        non = np.flatnonzero(yy == 0); k = len(non)//2
        known = rng.permutation(non)[:k]
        D = null_dist(T[known], a.nullboot)          # target's own signals, known-non only
        q95 = np.quantile(D, 0.95, axis=0); med = np.median(D, axis=0)
        pool_q95 = float(np.median(q95))
        print(f"\n{name}: {k} known-non, {len(cols)} signals")
        print(f"  null half-split |AUC-0.5|: median across signals {float(np.median(med)):.3f}, "
              f"typical q95 {pool_q95:.3f}")
        for mult in (1.5, 1.25):
            drop = q95 > mult*pool_q95
            print(f"  reject q95 > {mult}x pool: drops {int(drop.sum())}/{len(cols)}"
                  + (f"  e.g. {', '.join(np.array(names)[drop][:3])}" if drop.sum() else ""))
        drop = q95 > 1.25*pool_q95; sel = ~drop
        if sel.sum() < 2: print("  nothing survives"); continue
        for sc in ("iforest",):
            for lab, m in (("screened", sel), ("all", np.ones(len(cols), bool))):
                r = {}
                for kk, X in (("t", T), ("b", B)):
                    f = None
                    r[kk] = float(np.mean([one_draw(X[:, m], yy, np.random.default_rng(7+i),
                        (lambda Xk, Xs, i=i: iforest_scores(Xk, Xs, 7+i)) if sc == "iforest" else f,
                        (0.01, 0.05, 0.10))["auc"] for i in range(a.draws)]))
                ci = ""
                lo = hi = None
                if a.boot:
                    memi = np.flatnonzero(yy == 1); noni = np.flatnonzero(yy == 0); gaps = []
                    for bi in range(a.boot):
                        rg = np.random.default_rng(30_000+bi)
                        idx = np.concatenate([rg.choice(memi, len(memi), True),
                                              rg.choice(noni, len(noni), True)])
                        yb = yy[idx]; g2 = {}
                        for kk, X in (("t", T), ("b", B)):
                            f2 = None
                            g2[kk] = float(np.mean([one_draw(X[idx][:, m], yb,
                                np.random.default_rng(7+i),
                                (lambda Xk, Xs, i=i: iforest_scores(Xk, Xs, 7+i)) if sc == "iforest" else f2,
                                (0.01, 0.05, 0.10))["auc"] for i in range(max(3, a.draws//8))]))
                        gaps.append(g2["t"] - g2["b"])
                    sd = float(np.std(gaps, ddof=1))
                    ci = f" +/-{sd:.3f}"
                print(f"    {sc:8s} {lab:9s} ({int(m.sum()):3d} sig): "
                      f"{r['t']:.3f}/{r['b']:.3f} {r['t']-r['b']:+.3f}{ci}", flush=True)
                out["rows"].append({"set": name, "scorer": sc, "arm": lab,
                                    "n_sig": int(m.sum()), "target": round(r['t'],4),
                                    "blind": round(r['b'],4), "gap": round(r['t']-r['b'],4),
                                    "gap_sd": round(sd, 4)})
    (Path(a.out) if a.out else REPO/"exports"/"hyp_stability.json").write_text(json.dumps(out,indent=1))

if __name__ == "__main__":
    main()
