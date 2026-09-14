"""Bootstrap sd for the external (label-free) audit, one signal family.

Datasets are resampled with replacement, stratified on membership so the
member/non composition is preserved. For each resample the auditor is re-run over
several known-non/suspect draws and the mean gap recorded; the CI is the
percentile interval of those gaps. Rows are pre-filtered per family so the blind
actually produced that family (no median placeholders).
"""
import os
for _v in ("OMP_NUM_THREADS","OMP_THREAD_LIMIT","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from di.baseline_topk import LOADERS, REPO, fam_of
from di.external_family_audit import (impute, iforest_scores,
                                       iforest_oriented_scores, one_draw)

def audit(X, y, scorer, draws, seed0=7):
    f = None
    fn = {"iforest": iforest_scores, "iforest_or": iforest_oriented_scores}.get(scorer)
    vals=[]
    for i in range(draws):
        g = (lambda Xk,Xs,i=i: fn(Xk,Xs,seed0+i)) if fn else f
        vals.append(one_draw(X, y, np.random.default_rng(seed0+i), g, (0.01,0.05,0.10))["auc"])
    return float(np.mean(vals))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sets", default="tabdpt_fullgrid,sap_t4_forcecls,realtabpfn")
    ap.add_argument("--family", default="tshuffle")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--scorers", default="pu,iforest")
    ap.add_argument("--out", default=None)
    a=ap.parse_args()
    out={"meta":{"family":a.family,"boot":a.boot,"draws":a.draws},"rows":[]}
    for name in a.sets.split(","):
        ids,y,sigs,Xts,Xb=LOADERS[name]()
        c=[j for j,s in enumerate(sigs) if fam_of(s)==a.family]
        keep=np.isfinite(Xb[:,c]).mean(1)>0                  # per-family row filter
        T=impute(list(Xts.values())[0][keep][:,c]); B=impute(Xb[keep][:,c]); yy=y[keep]
        mem=np.flatnonzero(yy==1); non=np.flatnonzero(yy==0)
        for sc in a.scorers.split(","):
            pt=audit(T,yy,sc,a.draws*3); pb=audit(B,yy,sc,a.draws*3)
            gaps=[]
            for b in range(a.boot):
                rng=np.random.default_rng(10_000+b)
                idx=np.concatenate([rng.choice(mem,len(mem),True), rng.choice(non,len(non),True)])
                yb=yy[idx]
                gaps.append(audit(T[idx],yb,sc,a.draws) - audit(B[idx],yb,sc,a.draws))
            sd=float(np.std(gaps,ddof=1))
            r={"set":name,"family":a.family,"n":int(keep.sum()),"n_member":int(yy.sum()),
               "scorer":sc,"target":round(pt,4),"blind":round(pb,4),"gap":round(pt-pb,4),
               "boot_mean":round(float(np.mean(gaps)),4),"boot_std":round(sd,4),
               "gaps":[round(float(g),5) for g in gaps]}
            out["rows"].append(r)
            print(f"  {name:18s} {sc:10s} n={r['n']:5d}  {pt:.3f}/{pb:.3f} {pt-pb:+.3f} "
                  f"+/-{sd:.3f}", flush=True)
    p=Path(a.out) if a.out else REPO/"exports"/f"hyp_extboot_{a.family}.json"
    p.write_text(json.dumps(out,indent=1))

if __name__=="__main__":
    main()
