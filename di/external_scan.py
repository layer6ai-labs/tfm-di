"""Scan every signal family with the external auditor, calibrated against a
SELECTION-AWARE null.

Reporting the best of N families against a per-family null is winner's curse: with
18 families the largest gap is inflated by the search. Here each permutation of the
membership labels re-runs ALL families and records the MAXIMUM gap, so the observed
best is compared against the distribution of the best-under-null. Marginal
(per-family) nulls are reported too, for families named in advance.
"""
import os
for _v in ("OMP_NUM_THREADS","OMP_THREAD_LIMIT","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from di.baseline_topk import LOADERS, REPO, fam_of
from di.external_family_audit import impute
from di.external_null import audit

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--set", default="sap_t4_forcecls")
    ap.add_argument("--scorer", default="iforest")
    ap.add_argument("--nperm", type=int, default=50)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--out", default=None)
    a=ap.parse_args()
    ids,y,sigs,Xts,Xb=LOADERS[a.set]()
    fams=sorted({fam_of(s) for s in sigs})
    Tfull=list(Xts.values())[0]
    prep={}
    for f in fams:                       # per-family row filter + imputation, once
        c=[j for j,s in enumerate(sigs) if fam_of(s)==f]
        keep=np.isfinite(Xb[:,c]).mean(1)>0
        if keep.sum()<60 or len(set(y[keep]))<2: continue
        prep[f]=(impute(Tfull[keep][:,c]), impute(Xb[keep][:,c]), y[keep], len(c))
    print(f"{a.set}: {len(prep)} families usable, scorer={a.scorer}\n", flush=True)
    obs={}
    for f,(T,B,yy,nc) in prep.items():
        t=audit(T,yy,a.scorer,a.draws*3); b=audit(B,yy,a.scorer,a.draws*3)
        obs[f]=(t,b,t-b,len(yy),nc)
        print(f"  {f:12s} {nc:3d}c {len(yy):5d}r  target {t:.3f} blind {b:.3f} gap {t-b:+.3f}", flush=True)
    nullmax=[]; nullfam={f:[] for f in prep}
    for p in range(a.nperm):
        g={}
        for f,(T,B,yy,_) in prep.items():
            yp=np.random.default_rng(70_000+p).permutation(yy)
            g[f]=audit(T,yp,a.scorer,a.draws)-audit(B,yp,a.scorer,a.draws)
            nullfam[f].append(g[f])
        nullmax.append(max(g.values()))
        if (p+1)%10==0: print(f"    perm {p+1}/{a.nperm}  null-max so far "
                              f"{np.mean(nullmax):+.3f}+/-{np.std(nullmax):.3f}", flush=True)
    nullmax=np.array(nullmax)
    best=max(obs, key=lambda f: obs[f][2])
    rows=[]
    for f,(t,b,g,n,nc) in sorted(obs.items(), key=lambda kv:-kv[1][2]):
        nf=np.array(nullfam[f])
        rows.append({"family":f,"n":n,"n_cols":nc,"target":round(t,4),"blind":round(b,4),
                     "gap":round(g,4),"marg_null_mean":round(float(nf.mean()),4),
                     "marg_null_sd":round(float(nf.std(ddof=1)),4),
                     "p_marginal":round(float((1+(nf>=g).sum())/(1+len(nf))),4),
                     "p_selection_aware":round(float((1+(nullmax>=g).sum())/(1+len(nullmax))),4)})
    out={"meta":{"set":a.set,"scorer":a.scorer,"nperm":a.nperm,"draws":a.draws,
                 "best_family":best,
                 "nullmax_mean":round(float(nullmax.mean()),4),
                 "nullmax_sd":round(float(nullmax.std(ddof=1)),4),
                 "nullmax_q95":round(float(np.quantile(nullmax,.95)),4)},"rows":rows}
    p=Path(a.out) if a.out else REPO/"exports"/f"hyp_extscan_{a.set}_{a.scorer}.json"
    p.write_text(json.dumps(out,indent=1))
    print(f"\nnull-of-max: mean {nullmax.mean():+.3f} sd {nullmax.std(ddof=1):.3f} "
          f"q95 {np.quantile(nullmax,.95):+.3f}   best family = {best} "
          f"gap {obs[best][2]:+.3f}", flush=True)

if __name__=="__main__":
    main()
