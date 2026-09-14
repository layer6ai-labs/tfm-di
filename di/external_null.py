"""Permutation null for the external auditor: how much gap does each scorer
manufacture when membership is randomised?

Membership labels are shuffled (preserving the member/non counts) and the ENTIRE
auditor is re-run — known-non draw, orientation, scoring. A scorer that is really
reading membership gives a null centred on zero; one that is fitting an incidental
distribution difference between the known-non subset and the suspect pile gives a
null that is wide or offset.
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
    f=None
    fn={"iforest":iforest_scores,"iforest_or":iforest_oriented_scores}.get(scorer)
    return float(np.mean([one_draw(X,y,np.random.default_rng(seed0+i),
        (lambda Xk,Xs,i=i: fn(Xk,Xs,seed0+i)) if fn else f,(0.01,0.05,0.10))["auc"]
        for i in range(draws)]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sets", default="sap_t4_forcecls")
    ap.add_argument("--family", default="tshuffle")
    ap.add_argument("--nperm", type=int, default=100)
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--scorers", default="pu,iforest,iforest_or")
    ap.add_argument("--out", default=None)
    a=ap.parse_args()
    out={"meta":vars(a),"rows":[]}
    for name in a.sets.split(","):
        ids,y,sigs,Xts,Xb=LOADERS[name]()
        c=[j for j,s in enumerate(sigs) if fam_of(s)==a.family]
        keep=np.isfinite(Xb[:,c]).mean(1)>0
        T=impute(list(Xts.values())[0][keep][:,c]); B=impute(Xb[keep][:,c]); yy=y[keep]
        for sc in a.scorers.split(","):
            obs=audit(T,yy,sc,a.draws*3)-audit(B,yy,sc,a.draws*3)
            nulls=[]
            for p in range(a.nperm):
                yp=np.random.default_rng(50_000+p).permutation(yy)
                nulls.append(audit(T,yp,sc,a.draws)-audit(B,yp,sc,a.draws))
            nulls=np.array(nulls)
            pv=(1+int((nulls>=obs).sum()))/(1+len(nulls))
            r={"set":name,"family":a.family,"n":int(keep.sum()),"scorer":sc,
               "observed":round(obs,4),"null_mean":round(float(nulls.mean()),4),
               "null_std":round(float(nulls.std(ddof=1)),4),
               "null_q95":round(float(np.quantile(nulls,.95)),4),"p":round(pv,4),
               "z":round(float((obs-nulls.mean())/max(nulls.std(ddof=1),1e-9)),2)}
            out["rows"].append(r)
            print(f"  {name:18s} {sc:11s} obs {obs:+.3f} | null {nulls.mean():+.3f}"
                  f" +/-{nulls.std(ddof=1):.3f} q95 {np.quantile(nulls,.95):+.3f}"
                  f" | z={r['z']:+.1f} p={pv:.4f}", flush=True)
    p=Path(a.out) if a.out else REPO/"exports"/f"hyp_extnull_{a.family}.json"
    p.write_text(json.dumps(out,indent=1))

if __name__=="__main__":
    main()
