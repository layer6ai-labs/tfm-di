"""Label-free signal selection by self-calibration on the known non-members.

The auditor holds KNOWN NON-MEMBERS (K) and an unlabelled SUSPECT pile (S).
For each signal:

  null   split K at random into two halves and measure the separation between
         them. Both halves are non-members, so any separation is the machinery
         talking to itself. Repeat B times -> a per-signal null distribution,
         centred on 0.5 if the signal is well behaved.
  obs    separation between K and S on the same signal.

A signal is SELECTED when obs sits beyond the (1-alpha) quantile of its own null.
No membership labels are used anywhere; labels only score the result afterwards.

Caveat worth stating in any writeup: this selects signals on which the suspect
pile differs from the known non-members. Membership is one reason for that; any
other systematic difference between the two pools is another.
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

def sep(a, b):
    """|AUC-0.5| between two groups on one signal — direction-free separation."""
    v = np.concatenate([a, b]); z = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    return abs(roc_auc_score(z, v) - 0.5)

def select(Xk, Xs, B=200, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    nk = len(Xk); half = nk // 2
    obs = np.array([sep(Xk[:, j], Xs[:, j]) for j in range(Xk.shape[1])])
    null = np.empty((B, Xk.shape[1]))
    for b in range(B):
        p = rng.permutation(nk)
        A, Bh = p[:half], p[half:]
        for j in range(Xk.shape[1]):
            null[b, j] = sep(Xk[A, j], Xk[Bh, j])
    q = np.quantile(null, 1 - alpha, axis=0)
    return obs > q, obs, q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="sap_t4_forcecls,tabdpt_fullgrid,realtabpfn")
    ap.add_argument("--family", default="")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = {"meta": vars(a), "rows": []}
    for name in a.sets.split(","):
        ids, y, sigs, Xts, Xb = LOADERS[name]()
        cols = [j for j, s in enumerate(sigs) if (not a.family) or fam_of(s) == a.family]
        keep = np.isfinite(Xb[:, cols]).mean(1) > 0
        T = impute(list(Xts.values())[0][keep][:, cols]); B = impute(Xb[keep][:, cols]); yy = y[keep]
        names = [sigs[j] for j in cols]
        # the auditor's own split: half the non-members are "known"
        rng = np.random.default_rng(1)
        non = np.flatnonzero(yy == 0); mem = np.flatnonzero(yy == 1)
        p = rng.permutation(non); k = len(non)//2
        known, held = p[:k], p[k:]
        sus = np.concatenate([held, mem])
        selT, obsT, qT = select(T[known], T[sus], a.boot, a.alpha)
        print(f"\n{name}: n={keep.sum()} ({len(mem)} member), {len(cols)} signals, "
              f"{k} known-non / {len(sus)} suspects")
        print(f"  self-calibrated selection keeps {selT.sum()} signals "
              f"({100*selT.mean():.0f}%)  [expected under a pure null: {100*a.alpha:.0f}%]")
        ysus = np.concatenate([np.zeros(len(held)), np.ones(len(mem))])
        for lab, sel in (("selected", selT), ("all", np.ones(len(cols), bool))):
            if sel.sum() == 0: continue
            for sc in ("iforest",):
                r = {}
                for m, X in (("t", T), ("b", B)):
                    f = None
                    r[m] = float(np.mean([one_draw(
                        X[:, sel], yy, np.random.default_rng(7+i),
                        (lambda Xk, Xs, i=i: iforest_scores(Xk, Xs, 7+i)) if sc == "iforest" else f,
                        (0.01, 0.05, 0.10))["auc"] for i in range(a.draws)]))
                print(f"    {sc:8s} on {lab:9s} ({int(sel.sum()):4d} sig): "
                      f"{r['t']:.3f}/{r['b']:.3f} {r['t']-r['b']:+.3f}")
                out["rows"].append({"set": name, "arm": lab, "scorer": sc,
                                    "n_sig": int(sel.sum()), "target": round(r["t"],4),
                                    "blind": round(r["b"],4), "gap": round(r["t"]-r["b"],4)})
        top = np.argsort(-(obsT - qT))[:8]
        print("    top selected:", ", ".join(names[j] for j in top if selT[j])[:150])
    p = Path(a.out) if a.out else REPO/"exports"/"hyp_selfcalib.json"
    p.write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
