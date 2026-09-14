"""External (label-free) audit restricted to a small set of pre-registered families.

Threat model: the auditor holds a set of KNOWN NON-MEMBERS and a SUSPECT pile of
unknown membership. No member labels anywhere inside the method. Membership
labels are used only to score the method afterwards.

Scorers
  pu        z-score each signal against the known non-members; orient each signal
            by the shift of the *unlabelled* suspect pile (never by labels); score
            = mean oriented z. Split-conformal p against the known-non scores.
  iforest   IsolationForest fitted on the known non-members; suspects scored by
            how anomalous they look. Direction is fixed by construction
            (anomalous = member-like), which is the method's main weakness:
            anomalous is not the same thing as member.

The blind is run through the IDENTICAL auditor on its own signals — that is the
control. Nothing here consults the blind to build the target's score.
"""
import os

for _v in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from di.baseline_topk import LOADERS, REPO, fam_of  # noqa: E402


def impute(X):
    X = np.array(X, float)
    for j in range(X.shape[1]):
        c = X[:, j]
        bad = ~np.isfinite(c)
        if bad.any():
            m = np.nanmedian(np.where(np.isfinite(c), c, np.nan))
            c[bad] = m if np.isfinite(m) else 0.0
    return X


def iforest_scores(Xk, Xs, seed):
    f = IsolationForest(n_estimators=200, random_state=seed, contamination="auto")
    f.fit(Xk)
    return -f.score_samples(Xk), -f.score_samples(Xs)   # higher = more anomalous


def iforest_oriented_scores(Xk, Xs, seed):
    """IsolationForest, but with the direction derived label-free instead of assumed.

    Plain iforest hard-codes "more anomalous = more member-like". That is often
    false: when members form a TIGHTER cloud than the known non-members the forest
    was fitted on, they are isolated less easily and the ranking inverts (blind AUC
    0.33-0.47 across most SAP families). Here the sign is taken from the shift of
    the *unlabelled* suspect pile relative to the known non-members — the same PU
    orientation, no labels touched — so an inverted family is corrected rather than
    reported as a large gap.
    """
    ak, as_ = iforest_scores(Xk, Xs, seed)
    sign = 1.0 if as_.mean() >= ak.mean() else -1.0
    return sign * ak, sign * as_


def conformal_p(cal, test):
    cal = np.sort(np.asarray(cal))
    ge = len(cal) - np.searchsorted(cal, np.asarray(test), side="left")
    return (1.0 + ge) / (len(cal) + 1.0)


def one_draw(X, y, rng, scorer, alphas, known_frac=0.5):
    non, mem = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    perm = rng.permutation(non)
    k = max(5, int(round(known_frac * len(non))))
    known, held = perm[:k], perm[k:]
    sus = np.concatenate([held, mem])
    sk, ss = scorer(X[known], X[sus])
    yt = np.concatenate([np.zeros(len(held)), np.ones(len(mem))])
    out = {"auc": float(roc_auc_score(yt, ss)), "n_known": int(k)}
    p = conformal_p(sk, ss)
    for a in alphas:
        fl = p <= a
        out[f"tpr@{a}"] = float(fl[yt == 1].mean())
        out[f"fpr@{a}"] = float(fl[yt == 0].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="tabdpt_fullgrid,sap_t4_all")
    ap.add_argument("--families", default="splitsize,tshuffle,ctx")
    ap.add_argument("--also-all", action="store_true", help="add an all-families arm")
    ap.add_argument("--exclude", default="", help="families dropped from the ALL arm")
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--impute", default="base", choices=["base", "masked"])
    ap.add_argument("--min-cov", type=float, default=0.0,
                    help="keep only datasets where every model has at least this "
                         "fraction of the arm's columns present (0 = keep all)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    alphas = (0.01, 0.05, 0.10)
    fams = a.families.split(",")
    arms = {a.families: set(fams), **{f: {f} for f in fams}}
    drop = set(x for x in a.exclude.split(",") if x)
    if a.also_all:
        arms["ALL" + (f" minus {a.exclude}" if drop else "")] = ("ALL", drop)
    out = {"meta": {"created": time.strftime("%Y-%m-%d %H:%M"), "args": vars(a)}, "rows": []}

    for name in a.sets.split(","):
        ids, y, sigs, Xts, Xb = LOADERS[name]()
        print(f"\n===== {name}  n={len(y)} ({int(y.sum())} member / {int((y==0).sum())} non)")
        for arm, want in arms.items():
            if isinstance(want, tuple):          # the ALL arm, minus excluded families
                cols = [j for j, s in enumerate(sigs) if fam_of(s) not in want[1]]
            else:
                cols = [j for j, s in enumerate(sigs) if fam_of(s) in want]
            if not cols:
                continue
            tsub = {k: v[:, cols].copy() for k, v in Xts.items()}
            bsub = Xb[:, cols].copy()
            yy = y
            if a.min_cov > 0:
                # keep only datasets where the BLIND actually produced signals in
                # this family — otherwise its rows are median placeholders and the
                # gap measures coverage rather than behaviour
                cov = np.isfinite(bsub).mean(1)
                for v in tsub.values():
                    cov = np.minimum(cov, np.isfinite(v).mean(1))
                keep = cov >= a.min_cov
                if keep.sum() < 40 or len(set(y[keep])) < 2:
                    print(f"  {arm:26s} skipped: only {int(keep.sum())} rows survive", flush=True)
                    continue
                tsub = {k: v[keep] for k, v in tsub.items()}
                bsub = bsub[keep]; yy = y[keep]
            if a.impute == "masked":
                # a cell missing in ANY model is treated as missing in ALL, so the
                # target is handicapped down to the blind's coverage. tshuffle's
                # blind is ~70% missing while its target is ~36%, so without this
                # the "gap" is largely the blind being imputed to a constant.
                bad = ~np.isfinite(bsub)
                for v in tsub.values():
                    bad |= ~np.isfinite(v)
                for v in tsub.values():
                    v[bad] = np.nan
                bsub[bad] = np.nan
            mats = {**{f"target:{k}": impute(v) for k, v in tsub.items()},
                    "blind": impute(bsub)}
            for sname, sfun in (("iforest", iforest_scores),
                                ("iforest_or", iforest_oriented_scores)):
                res = {}
                for mname, X in mats.items():
                    rows = [one_draw(X, yy, np.random.default_rng(7 + i),
                                     (lambda Xk, Xs, i=i: sfun(Xk, Xs, 7 + i))
                                     if sname.startswith("iforest") else sfun, alphas)
                            for i in range(a.draws)]
                    res[mname] = {k: float(np.mean([r[k] for r in rows]))
                                  for k in rows[0] if k != "n_known"}
                tg = np.mean([v["auc"] for k, v in res.items() if k.startswith("target")])
                bl = res["blind"]["auc"]
                t10 = np.mean([v["tpr@0.1"] for k, v in res.items() if k.startswith("target")])
                f10 = np.mean([v["fpr@0.1"] for k, v in res.items() if k.startswith("target")])
                t1 = np.mean([v["tpr@0.01"] for k, v in res.items() if k.startswith("target")])
                f1 = np.mean([v["fpr@0.01"] for k, v in res.items() if k.startswith("target")])
                out["rows"].append({"set": name, "arm": arm, "impute": a.impute,
                                    "min_cov": a.min_cov, "n_rows": int(len(yy)),
                                    "n_member": int(yy.sum()), "n_cols": len(cols),
                                    "scorer": sname, "target": round(tg, 4),
                                    "blind": round(bl, 4), "gap": round(tg - bl, 4),
                                    "tpr@1": round(t1, 4), "fpr@1": round(f1, 4),
                                    "tpr@10": round(t10, 4), "fpr@10": round(f10, 4)})
                print(f"  {arm:26s} {sname:8s} ({len(cols):4d}c/{len(yy):5d}r "
                      f"{100*yy.mean():.0f}%m)  target {tg:.3f} "
                      f"blind {bl:.3f}  gap {tg-bl:+.3f}   TPR@1%={t1:.3f}({f1:.3f}) "
                      f"TPR@10%={t10:.3f}({f10:.3f})", flush=True)
    p = Path(a.out) if a.out else REPO / "exports" / "hyp_external_families.json"
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
