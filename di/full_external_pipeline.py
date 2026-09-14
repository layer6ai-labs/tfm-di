"""The whole label-free auditor, end to end, with bootstrap sds.

  1. rows filtered so the blind actually produced the signals (no placeholders)
  2. SELECTION: self-calibration on the known non-members — split them in half,
     measure per-signal separation between halves (pure null), keep signals whose
     real known-non-vs-suspects separation beats their own null quantile.
     No membership labels.
  3. SCORING: PU and IsolationForest on the selected signals, target and blind.
  4. CI: datasets resampled with replacement, stratified on membership.

Selection is done once on the full data and the CI covers the scoring only, so the
interval understates total procedure variance.
"""
import os
for _v in ("OMP_NUM_THREADS","OMP_THREAD_LIMIT","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from di.baseline_topk import LOADERS, REPO, fam_of
from di.external_family_audit import impute, iforest_scores, one_draw
from di.selfcalib_select import select

def audit(X, y, sc, draws, seed0=7):
    f = None
    return float(np.mean([one_draw(
        X, y, np.random.default_rng(seed0+i),
        (lambda Xk, Xs, i=i: iforest_scores(Xk, Xs, seed0+i)) if sc == "iforest" else f,
        (0.01, 0.05, 0.10))["auc"] for i in range(draws)]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="sap_t4_forcecls,tabdpt_fullgrid,realtabpfn")
    ap.add_argument("--nullboot", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--boot", type=int, default=150)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = {"meta": vars(a), "rows": []}
    for name in a.sets.split(","):
        ids, y, sigs, Xts, Xb = LOADERS[name]()
        keep = np.isfinite(Xb).mean(1) > 0.3          # blind produced something
        T = impute(list(Xts.values())[0][keep]); B = impute(Xb[keep]); yy = y[keep]
        rng = np.random.default_rng(1)
        non = np.flatnonzero(yy == 0); mem = np.flatnonzero(yy == 1)
        p = rng.permutation(non); k = len(non)//2
        known, sus = p[:k], np.concatenate([p[k:], mem])
        sel, obs, q = select(T[known], T[sus], a.nullboot, a.alpha)
        fams = {}
        for j, s in enumerate(sigs):
            if sel[j]: fams[fam_of(s)] = fams.get(fam_of(s), 0) + 1
        print(f"\n{name}: n={keep.sum()} ({len(mem)} mem), {len(sigs)} signals, "
              f"{k} known-non / {len(sus)} suspects")
        print(f"  selected {int(sel.sum())} signals ({100*sel.mean():.0f}%; null rate "
              f"{100*a.alpha:.0f}%)  families: {dict(sorted(fams.items(), key=lambda z:-z[1])[:6])}")
        if sel.sum() < 2:
            print("  too few signals selected — no scoring"); continue
        Ts, Bs = T[:, sel], B[:, sel]
        for sc in ("iforest",):
            pt, pb = audit(Ts, yy, sc, a.draws*3), audit(Bs, yy, sc, a.draws*3)
            gaps = []
            for b in range(a.boot):
                r = np.random.default_rng(20_000+b)
                idx = np.concatenate([r.choice(mem, len(mem), True), r.choice(non, len(non), True)])
                gaps.append(audit(Ts[idx], yy[idx], sc, a.draws) - audit(Bs[idx], yy[idx], sc, a.draws))
            sd = float(np.std(gaps, ddof=1))
            print(f"    {sc:8s}: {pt:.3f}/{pb:.3f} {pt-pb:+.3f} +/-{sd:.3f}", flush=True)
            out["rows"].append({"set": name, "scorer": sc, "n": int(keep.sum()),
                                "n_sig": int(sel.sum()), "target": round(pt,4),
                                "blind": round(pb,4), "gap": round(pt-pb,4),
                                "gap_sd": round(sd, 4)})
    (Path(a.out) if a.out else REPO/"exports"/"hyp_full_external.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
