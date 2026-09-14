"""Simple top-K audit: pick the target's top features once with XGBoost/TreeSHAP,
then score LR / XGBoost / TabPFN-2.5 on the target and on the blind using exactly
those features.

No inner CV, no K* search — one ranking, a fixed K, three classifiers. The blind
is only ever a control: it is scored on the target-selected columns and never
ranks or selects anything of its own.

Two selection modes are reported because they answer different questions:
  in-sample  the ranking is fitted on all n datasets, then the same columns are
             cross-validated. This is what "run XGBoost, take the top features,
             then evaluate" literally means, and it is optimistic — the labels
             leaked into the selection.
  per-fold   the ranking is refitted on each CV training split. Slower by the
             number of folds, honest, and the number to quote.

  python di/simple_topk_audit.py --sets tabdpt_fullgrid,sap_fullgrid --ks 4,8,16
"""
import os

for _v in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "XGBOOST_NTHREAD"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from di.audit_pipeline import (_fitpred, atomic_dump, folded_auc,  # noqa: E402
                                shap_order)
import di.new_signal_audits as NSA  # noqa: E402
from di.new_signal_audits import EXPORTS, LOADERS, make_tabpfn  # noqa: E402


def cv_auc(X, y, sel_fn, kind, splits, tab_clf=None):
    """Cross-validated held-out AUC on the columns sel_fn(fold_index) returns."""
    pred, cnt = np.zeros(len(y)), np.zeros(len(y))
    for i, (tr, te) in enumerate(splits):
        sel = sel_fn(i)
        pred[te] += _fitpred(kind, X[0][tr][:, sel], y[tr], X[0][te][:, sel], tab_clf)
        cnt[te] += 1
    return folded_auc(y, pred / np.where(cnt > 0, cnt, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="tabdpt_fullgrid,sap_fullgrid")
    ap.add_argument("--scopes", default="full,new_families,corruption,fam:tserum")
    ap.add_argument("--ks", default="4,8,16")
    ap.add_argument("--kinds", default="lr,xgb,tabpfn")
    ap.add_argument("--models", default="")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--min-support", type=float, default=0.5)
    ap.add_argument("--impute", default="base", choices=["base", "masked", "complete"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    NSA.MIN_SUPPORT = a.min_support
    NSA.IMPUTE = a.impute
    kinds = tuple(a.kinds.split(","))
    KS = [int(k) for k in a.ks.split(",")]
    tab_clf = make_tabpfn(a.device, 4) if "tabpfn" in kinds else None
    out = {"meta": {"args": vars(a), "created": time.strftime("%Y-%m-%d %H:%M"),
                    "procedure": "XGBoost/TreeSHAP ranking on the TARGET; top-K columns "
                                 "applied to target and blind; repeated stratified "
                                 f"{a.splits}-fold x{a.repeats}; tie-aware folded AUC"},
           "sets": {}}

    for name in a.sets.split(","):
        ss = LOADERS[name]()
        if a.models:
            ss.Xt = {k: v for k, v in ss.Xt.items() if k in set(a.models.split(","))}
        rows = []
        for scope in [s for s in a.scopes.split(",") if s in ss.scopes]:
            cols = ss.scopes[scope]
            Xb = ss.Xb[:, cols]
            splits = list(RepeatedStratifiedKFold(
                n_splits=a.splits, n_repeats=a.repeats, random_state=0).split(Xb, ss.y))
            for mname, Mfull in ss.Xt.items():
                Xt = Mfull[:, cols]
                order_all = shap_order(Xt, ss.y)                    # in-sample ranking
                fold_order = {i: shap_order(Xt[tr], ss.y[tr])       # per-fold rankings
                              for i, (tr, _) in enumerate(splits)}
                for K in [k for k in KS if k <= Xt.shape[1]]:
                    for sel_name, sel_of in (
                            ("in-sample", lambda i, K=K: order_all[:K]),
                            ("per-fold", lambda i, K=K: fold_order[i][:K])):
                        for kind in kinds:
                            t = cv_auc((Xt,), ss.y, sel_of, kind, splits, tab_clf)
                            b = cv_auc((Xb,), ss.y, sel_of, kind, splits, tab_clf)
                            rows.append({"scope": scope, "n_signals": len(cols),
                                         "model": mname, "K": K, "selection": sel_name,
                                         "clf": kind, "target": round(float(t), 4),
                                         "blind": round(float(b), 4),
                                         "gap": round(float(t - b), 4)})
                            print(f"  {name:16s} {scope:14s} {mname:9s} K={K:<3d} "
                                  f"{sel_name:9s} {kind:6s} tgt {t:.3f} blind {b:.3f} "
                                  f"gap {t-b:+.3f}", flush=True)
                top = [ss.sigs[cols[j]] for j in order_all[:max(KS)]]
                rows.append({"scope": scope, "model": mname, "top_features": top})
        out["sets"][name] = {"n": ss.n, "n_member": int(ss.y.sum()),
                             "min_support": a.min_support, "impute": a.impute,
                             "missing": ss.missing, "rows": rows}
        atomic_dump(out, Path(a.out) if a.out else EXPORTS / f"hyp_topk_{name}.json")
    return out


if __name__ == "__main__":
    main()
