"""BASELINE: the recipe with nothing added.

    whole table -> XGBoost -> top-K SHAP features -> LR / XGBoost / TabPFN-2.5
    on the target and on the blind with exactly those features.

Deliberately does NOT do any of the things the main pipeline does:
  * no min_support filter        (every numeric column is a candidate)
  * no constant-column drop
  * no META_KEYS exclusion       (n_features and the other shape columns ARE
                                  candidate features; n_pool is the one exception,
                                  excluded everywhere — see NON_FEATURES)
  * no global median imputation  (NaN is passed to the classifiers, which all
                                  handle it: XGBoost natively, TabPFN natively,
                                  LR via a fold-local median imputer)
  * no AUC folding               (raw directed AUC; below 0.5 stays below 0.5)
  * no per-fold re-selection     (ranking is fitted once on all rows, as asked)
  * no scopes                    (the whole table, nothing else)

The only unavoidable choices: rows are restricted to classification datasets and
aligned across models by dataset id, and a repeated stratified 5-fold gives the
held-out estimate. Both are stated in the output.

  python di/baseline_topk.py --sets tabdpt_fullgrid,sap_fullgrid,sap_redteam
"""
import os

for _v in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "XGBOOST_NTHREAD"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(os.environ.get("AUDIT_REPO", "/p/project1/hai_1159/tfm-di")).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

XGB_PARAMS = dict(max_depth=3, n_estimators=150, learning_rate=0.06, subsample=0.85,
                  colsample_bytree=0.7, reg_lambda=1.0, min_child_weight=2,
                  eval_metric="logloss", n_jobs=1, verbosity=0, tree_method="hist",
                  random_state=0)
# Label, id, and — by request — dataset size. n_pool IS membership on SAP (the
# >150-row threshold), so leaving it in pins target and blind at AUC 1.000 and makes
# the gap identically zero; on the TabDPT grid it outranks every behavioural signal.
# Excluded everywhere, so what is measured is model behaviour.
NON_FEATURES = {"true_label", "is_member", "member", "dataset_id", "dataset_name",
                "success", "task_type", "grid_version", "iid_split", "did",
                # all table-shape metadata: identical for target and blind, so it
                # cannot create a gap on its own, but it is available to both and on
                # small tables a tree model leans on it (n_features was rank 2 of 27
                # on the SAP red-team set). Excluded so only behaviour is measured.
                "n_pool", "n_features", "n_classes", "n_str_cols", "n", "d", "C",
                # wall-clock extraction time. Member/non AUC 0.786 on the T4 chunk
                # and, unlike the shape columns, it DIFFERS between target and blind
                # (the two models run at different speeds) — so it can manufacture a
                # target-blind gap out of pure infrastructure. It ranked 5th of 1020.
                "elapsed_s", "dataset_id_str",
                # the nanotabpfn parquet ships a column literally named "label"
                # that is byte-identical to true_label. |SHAP| 6.93, AUC 1.0000 at
                # K=1. Not present in the other exports.
                "label", "y", "target"}


def num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return np.nan
    return float(v) if np.isfinite(v) else np.nan


def matrices(recs_by_model, ids, sigs):
    return {k: np.array([[num(g[i].get(s)) for s in sigs] for i in ids], float)
            for k, g in recs_by_model.items()}


def load_tabdpt():
    def ld(pat):
        f = sorted(glob.glob(str(REPO / pat)))[-1]
        return {str(r["dataset_id"]): r for r in json.load(open(f))["successful_results"]
                if r.get("success")}
    tg = {s: ld(f"di_grid_results_{s}_2026071*_*.json")
          for s in ("default", "seed42", "seed123", "seed456")}
    bl = ld("di_grid_results_tabpfn25_2026071*_*.json")
    ids = sorted(i for i in set.intersection(*[set(g) for g in tg.values()], set(bl))
                 if tg["default"][i].get("task_type") == "classification")
    y = np.array([1 if bl[i].get("true_label", bl[i].get("is_member")) else 0 for i in ids], int)
    sigs = sorted({k for g in list(tg.values()) + [bl] for r in g.values()
                   for k, v in r.items()
                   if k not in NON_FEATURES and isinstance(v, (int, float))
                   and not isinstance(v, bool)})
    return ids, y, sigs, matrices(tg, ids, sigs), matrices({"b": bl}, ids, sigs)["b"]


def load_sap_grid():
    dt = json.load(open(REPO / "exports/sap_fullgrid_signals_sap-rpt-oss.json"))
    db = {r["dataset_id"]: r for r in
          json.load(open(REPO / "exports/sap_fullgrid_signals_tabpfn25.json"))}
    dt = [r for r in dt if r["dataset_id"] in db]
    ids = [r["dataset_id"] for r in dt]
    tg = {r["dataset_id"]: r for r in dt}
    y = np.array([1 if tg[i].get("true_label") in (1, True) else 0 for i in ids], int)
    sigs = sorted({k for r in dt for k, v in r.items()
                   if k not in NON_FEATURES and isinstance(v, (int, float))
                   and not isinstance(v, bool)} & set(db[ids[0]]))
    return ids, y, sigs, matrices({"sap-rpt-oss": tg}, ids, sigs), matrices({"b": db}, ids, sigs)["b"]


def load_sap_redteam():
    recs = json.load(open(REPO / "exports/hyp_sap_redteam_raw.json"))
    ids = [r["dataset_id"] for r in recs]
    by = {r["dataset_id"]: r for r in recs}
    y = np.array([1 if by[i]["is_member"] else 0 for i in ids], int)
    meas = ["acc", "p_true", "loss", "conf"]
    pairs = [(f"sap_{c}_{m}", f"blind_{m}") for c in
             ("semantic", "anonname", "shuffval", "numeric") for m in meas]
    pairs += [(f"sapk_{c}_{m}", f"blindk_{m}") for c in ("semantic", "numeric") for m in meas]
    extra = []   # all shape metadata excluded — see NON_FEATURES
    pairs = [(t, b) for t, b in pairs if t in recs[0]] + [(e, e) for e in extra if e in recs[0]]
    sigs = [t for t, _ in pairs]
    Xt = np.array([[num(by[i].get(t)) for t, _ in pairs] for i in ids], float)
    Xb = np.array([[num(by[i].get(b)) for _, b in pairs] for i in ids], float)
    return ids, y, sigs, {"sap-rpt-oss": Xt}, Xb


def _load_t4_parquet(task, tfile="exports/sap_binary_t4_all_datasets_corrected.parquet",
                     bfile="exports/tabpfn25_binary_t4_all_datasets_corrected.parquet"):
    """SAP extraction as two aligned parquets with identical schema, one row per
    dataset: target (sap-rpt-oss) and blind (tabpfn25)."""
    import pandas as pd
    t = pd.read_parquet(REPO / tfile)
    b = pd.read_parquet(REPO / bfile)
    keep = t.success if task is None else ((t.task_type == task) & t.success)
    t = t[keep].sort_values("dataset_id").reset_index(drop=True)
    b = b.set_index("dataset_id").loc[t.dataset_id].reset_index()
    assert (t.true_label.values == b.true_label.values).all()
    sigs = sorted(c for c in t.columns
                  if c not in NON_FEATURES and c in b.columns
                  and pd.api.types.is_numeric_dtype(t[c])
                  and not pd.api.types.is_bool_dtype(t[c]))
    y = t.true_label.astype(int).to_numpy()
    def to(d):
        a = d[sigs].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        # inf -> NaN, and also anything that overflows float32: XGBoost casts to
        # float32 internally, so a finite float64 like 1e300 becomes inf there.
        # The regression signals carry values of that magnitude.
        a = np.where(np.isfinite(a), a, np.nan)
        return np.where(np.abs(a) > np.finfo(np.float32).max / 10, np.nan, a)
    return list(t.dataset_id), y, sigs, {"sap-rpt-oss": to(t)}, to(b)


def load_sap_t4_all():
    return _load_t4_parquet("classification")


def load_sap_t4_forcecls():
    """The T4 tables whose natural target is continuous, re-extracted with a
    classification target forced. Same tables as the regression arm, so the
    comparison isolates the task type rather than the table population."""
    return _load_t4_parquet(None,
                            "exports/sap_t4_regression_forcecls.parquet",
                            "exports/tabpfn25_t4_regression_forcecls.parquet")


def load_sap_t4_bin10():
    """Regression tables re-extracted with the target quantile-binned to 10 classes.
    This removes TabPFN's 10-class cap, so the blind can finally run the
    target-swapping probes: tshuffle/tabdpt_sim coverage is now symmetric
    (0.0%/0.0% vs the 0%/63-77% of the un-binned extraction)."""
    return _load_t4_parquet(None,
                            "exports/sap_t4_regression_bin10.parquet",
                            "exports/tabpfn25_t4_regression_bin10.parquet")


def _concat_sets(a, b):
    ia, ya, sa, Xta, Xba = a
    ib, yb, sb, Xtb, Xbb = b
    common = sorted(set(sa) & set(sb))
    pa = [sa.index(c) for c in common]; pb = [sb.index(c) for c in common]
    k = next(iter(Xta))
    return (list(ia) + list(ib), np.concatenate([ya, yb]), common,
            {k: np.vstack([list(Xta.values())[0][:, pa], list(Xtb.values())[0][:, pb]])},
            np.vstack([Xba[:, pa], Xbb[:, pb]]))


def load_sap_t4_cls_plus_bin10():
    """Natively-classification tables + the regression tables with their target
    binned to 10 classes. Both halves have symmetric blind coverage, so this is
    the fully-corrected pooled set."""
    return _concat_sets(_load_t4_parquet("classification"), load_sap_t4_bin10())


def load_sap_t4_cls_plus_forcecls():
    """Natively-classification tables + the forced-classification extraction of the
    regression tables. NOTE the forcecls half still has the 63-77% blind coverage
    hole on tshuffle/tabdpt_sim — use the bin10 version unless you specifically
    want the un-corrected comparison."""
    return _concat_sets(_load_t4_parquet("classification"), load_sap_t4_forcecls())


def load_sap_t4_all_both():
    """Both task types pooled. The two vocabularies barely overlap — a
    classification row is NaN in every regression-only column and vice versa —
    so the missingness pattern itself encodes task type. XGBoost and TabPFN take
    NaN natively; LR sees a fold-local median, which is why LR is the one to
    distrust here."""
    return _load_t4_parquet(None)


def load_sap_t4_all_reg():
    return _load_t4_parquet("regression")


def load_sap_t4_bin10():
    """The regression tables with their continuous target quantile-binned into at
    most 10 classes. Full coverage (all 11,197) and, unlike the forced-cls subset,
    target and blind have matched missingness (2.00% vs 1.98%)."""
    return _load_t4_parquet("classification",
                            "exports/sap_t4_regression_bin10.parquet",
                            "exports/tabpfn25_t4_regression_bin10.parquet")


def load_sap_t4_forcecls_clean():
    """forcecls restricted to the tables where the blind also ran the target-swap
    probes (1,716 of 4,594). Both models complete, so target-blind is a fair
    comparison without masking."""
    import numpy as _np
    ids, y, sigs, Xt, Xb = load_sap_t4_forcecls()
    c = [j for j, sg in enumerate(sigs) if fam_of(sg) in ("tshuffle", "tabdpt_sim")]
    keep = _np.isfinite(Xb[:, c]).mean(1) > 0
    return ([i for i, k in zip(ids, keep) if k], y[keep], sigs,
            {k: v[keep] for k, v in Xt.items()}, Xb[keep])


def load_sap_t4_cls_hybrid():
    """cls + HYBRID: the natively-categorical T4 tables plus every regression table
    under the hybrid target-column policy (forcecls rule where a 2..10-unique column
    exists, capped min-unique otherwise). 8,907 + 11,193 = 20,100 — the complete
    chunk under one classification vocabulary, unlike sap_t4_allcls which only takes
    the 4,594 forcecls rows and misses 6,599 regression tables."""
    import numpy as _np
    a = _load_t4_parquet("classification")
    b = load_sap_t4_hybrid()
    assert not (set(a[0]) & set(b[0])), "pools must be disjoint"
    common = sorted(set(a[2]) & set(b[2]))
    pa = [a[2].index(c) for c in common]; pb = [b[2].index(c) for c in common]
    k = next(iter(a[3]))
    return (list(a[0]) + list(b[0]), _np.concatenate([a[1], b[1]]), common,
            {k: _np.vstack([list(a[3].values())[0][:, pa], list(b[3].values())[0][:, pb]])},
            _np.vstack([a[4][:, pa], b[4][:, pb]]))


def load_sap_t4_allbin():
    """The whole T4 chunk under one vocabulary: natively-categorical tables plus
    every regression table binned to <=10 classes. 20,104 datasets."""
    import pandas as pd
    a_ids, a_y, a_sig, a_Xt, a_Xb = _load_t4_parquet("classification")
    b_ids, b_y, b_sig, b_Xt, b_Xb = load_sap_t4_bin10()
    assert not (set(a_ids) & set(b_ids)), "pools must be disjoint"
    sigs = [s for s in a_sig if s in set(b_sig)]
    ia = [a_sig.index(s) for s in sigs]; ib = [b_sig.index(s) for s in sigs]
    return (list(a_ids) + list(b_ids), np.concatenate([a_y, b_y]), sigs,
            {"sap-rpt-oss": np.vstack([a_Xt["sap-rpt-oss"][:, ia], b_Xt["sap-rpt-oss"][:, ib]])},
            np.vstack([a_Xb[:, ia], b_Xb[:, ib]]))


def load_sap_t4_bin10_sub():
    """bin10 restricted to exactly the tables the forced-cls extraction covered.
    Isolates 'which column became the target' from 'which tables were included'."""
    import pandas as pd
    ids, y, sigs, Xt, Xb = load_sap_t4_bin10()
    keep_ids = set(pd.read_parquet(REPO / "exports/sap_t4_regression_forcecls.parquet",
                                   columns=["dataset_id"]).dataset_id)
    m = np.array([i in keep_ids for i in ids])
    return ([i for i, k in zip(ids, m) if k], y[m], sigs,
            {"sap-rpt-oss": Xt["sap-rpt-oss"][m]}, Xb[m])


def load_realtabpfn(task="classification"):
    """RealTabPFN-2.5 (target) vs synthetic TabPFN-2.5 (blind), 70 datasets."""
    return _load_t4_parquet(task,
                            "exports/realtabpfn25_binary_realtabpfn25.parquet",
                            "exports/tabpfn25_binary_realtabpfn25.parquet")


def load_realtabpfn_all():
    return load_realtabpfn(None)


def load_sap_t4_binmin():
    """Regression tables with the continuous target binned to the MINIMUM class
    count (vs bin10's fixed <=10). Same tables as bin10, so the two isolate the
    effect of how coarse the discretisation is."""
    return _load_t4_parquet("classification",
                            "exports/sap_t4_regression_binmin.parquet",
                            "exports/tabpfn25_t4_regression_binmin.parquet")


def load_sap_t4_hybrid():
    """Hybrid target-column policy: the forced-cls draw where the table has a
    low-cardinality column, a capped min-unique binning otherwise. Covers all
    11,193 regression tables and is bit-identical to forced-cls on the 4,594."""
    return _load_t4_parquet("classification",
                            "exports/sap_t4_regression_hybrid.parquet",
                            "exports/tabpfn25_t4_regression_hybrid.parquet")


def load_sap_t4_binswap():
    """binswap extraction: same 11,193 regression tables as hybrid, but target and
    blind have MATCHED coverage (1.47% vs 1.37% missing), so the gap is not exposed
    to the coverage artifact that killed the hybrid result."""
    return _load_t4_parquet("classification",
                            "exports/sap_t4_regression_binswap.parquet",
                            "exports/tabpfn25_t4_regression_binswap.parquet")


def load_nano_matched():
    """NanoTabPFN vs TabPFN-2.5 on 20k matched synthetic tables (10k/10k, n_pool
    fixed at 50 so size cannot separate)."""
    return _load_t4_parquet(None, "exports/nanotabpfn_matched20k.parquet",
                            "exports/tabpfn25_nano_matched20k.parquet")


def load_realtabpfn():
    """RealTabPFN-2.5 vs synthetic TabPFN-2.5 blind."""
    return _load_t4_parquet(None, "exports/realtabpfn25_binary_realtabpfn25.parquet",
                            "exports/tabpfn25_binary_realtabpfn25.parquet")


def load_sap_t4_hybrid():
    return _load_t4_parquet(None, "exports/sap_t4_regression_hybrid.parquet",
                            "exports/tabpfn25_t4_regression_hybrid.parquet")


def load_sap_t4_allhybrid():
    """Whole T4 chunk, one vocabulary: the natively-categorical tables (8,907) plus
    every regression table re-extracted under the hybrid target policy (11,193).
    Disjoint by dataset_id; 20,100 total."""
    import pandas as pd
    a_ids, a_y, a_sig, a_Xt, a_Xb = _load_t4_parquet("classification")
    b_ids, b_y, b_sig, b_Xt, b_Xb = load_sap_t4_hybrid()
    assert not (set(a_ids) & set(b_ids)), "pools must be disjoint"
    sigs = [x for x in a_sig if x in set(b_sig)]
    ia = [a_sig.index(x) for x in sigs]; ib = [b_sig.index(x) for x in sigs]
    return (list(a_ids) + list(b_ids), np.concatenate([a_y, b_y]), sigs,
            {"sap-rpt-oss": np.vstack([a_Xt["sap-rpt-oss"][:, ia], b_Xt["sap-rpt-oss"][:, ib]])},
            np.vstack([a_Xb[:, ia], b_Xb[:, ib]]))


def load_sap_t4_allcls():
    """Everything as a classification task: the T4 tables that natively drew a
    categorical target, plus the ones that drew a continuous target and were
    re-extracted with a categorical one forced. One vocabulary, one pool."""
    import pandas as pd
    a_ids, a_y, a_sig, a_Xt, a_Xb = _load_t4_parquet("classification")
    b_ids, b_y, b_sig, b_Xt, b_Xb = load_sap_t4_forcecls()
    assert not (set(a_ids) & set(b_ids)), "the two pools must be disjoint"
    sigs = [s for s in a_sig if s in set(b_sig)]        # shared columns, a's order
    ia = [a_sig.index(s) for s in sigs]; ib = [b_sig.index(s) for s in sigs]
    Xt = np.vstack([a_Xt["sap-rpt-oss"][:, ia], b_Xt["sap-rpt-oss"][:, ib]])
    Xb = np.vstack([a_Xb[:, ia], b_Xb[:, ib]])
    return (list(a_ids) + list(b_ids), np.concatenate([a_y, b_y]), sigs,
            {"sap-rpt-oss": Xt}, Xb)


def load_sap_t4_forcecls():
    """The T4 tables that originally drew a continuous target, re-extracted with a
    low-cardinality target forced instead. Same tables as the regression pool, so
    this isolates the effect of the task type from the effect of the tables."""
    return _load_t4_parquet("classification",
                            "exports/sap_t4_regression_forcecls.parquet",
                            "exports/tabpfn25_t4_regression_forcecls.parquet")


LOADERS = {"tabdpt_fullgrid": load_tabdpt, "sap_fullgrid": load_sap_grid,
           "sap_redteam": load_sap_redteam, "sap_t4_all": load_sap_t4_all,
           "sap_t4_all_reg": load_sap_t4_all_reg, "sap_t4_all_both": load_sap_t4_all_both,
           "sap_t4_forcecls": load_sap_t4_forcecls,
           "sap_t4_allcls": load_sap_t4_allcls, "nano_matched": load_nano_matched,
           "realtabpfn": load_realtabpfn, "sap_t4_hybrid": load_sap_t4_hybrid, "sap_t4_allhybrid": load_sap_t4_allhybrid,
           "sap_t4_bin10": load_sap_t4_bin10, "sap_t4_binmin": load_sap_t4_binmin, "sap_t4_hybrid": load_sap_t4_hybrid, "sap_t4_allhybrid": load_sap_t4_allhybrid, "sap_t4_binswap": load_sap_t4_binswap, "realtabpfn": load_realtabpfn,
           "realtabpfn_all": load_realtabpfn_all, "sap_t4_bin10_sub": load_sap_t4_bin10_sub,
           "sap_t4_allbin": load_sap_t4_allbin,
           "sap_t4_fc_clean": load_sap_t4_forcecls_clean,
           "sap_t4_cls_hybrid": load_sap_t4_cls_hybrid}

NEW_FAMS = ("ctxeqq", "nnsplit", "lperm", "ftx", "rowshuf", "tshuffle", "tabdpt_sim")
CORRUPTION_FAMS = ("tserum", "bwash", "mislbl", "qleak", "rowdup")
ALL_FAMS = NEW_FAMS + CORRUPTION_FAMS + ("ctx", "noise", "col", "splitsize", "temp", "base")


def fam_of(sig):
    for f in ALL_FAMS:
        if sig == f or sig.startswith(f + "_"):
            return f
    return sig.split("_")[0]


def scope_cols(sigs, scope):
    if scope.startswith("no:"):        # everything EXCEPT the named families
        drop = set(scope[3:].split("+"))
        return [j for j, x in enumerate(sigs) if fam_of(x) not in drop]
    """Restricting to a family also drops n_pool/n_features, since they belong to
    no family — so a family scope is behavioural signals only."""
    if scope == "full":
        return list(range(len(sigs)))
    want = {"new_families": set(NEW_FAMS), "corruption": set(CORRUPTION_FAMS)}.get(
        scope, set(scope.split("+")))
    return [j for j, s in enumerate(sigs) if fam_of(s) in want]


def fitpred(kind, Xtr, ytr, Xte, tab=None):
    if kind == "lr":
        m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
    elif kind == "xgb":
        m = xgb.XGBClassifier(**XGB_PARAMS)          # NaN handled natively
    else:
        tab.fit(Xtr, ytr)
        return tab.predict_proba(Xte)[:, 1]
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="tabdpt_fullgrid,sap_fullgrid,sap_redteam")
    ap.add_argument("--ks", default="4,8,16,32")
    ap.add_argument("--scopes", default="full",
                    help="comma list; 'full', a family name, 'a+b' to union families, "
                         "or the shorthands new_families / corruption")
    ap.add_argument("--kinds", default="lr,xgb,tabpfn")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--models", default="")
    ap.add_argument("--nested-k", action="store_true",
                    help="choose K inside each training fold (inner 5-fold CV on the "
                         "training rows, elbow of the inner target-AUC curve) instead of "
                         "reading it off the reported curve. Implies --rank-per-fold.")
    ap.add_argument("--rank-per-fold", action="store_true",
                    help="refit the TreeSHAP ranking inside every training fold, so the "
                         "held-out rows never influence which columns are used")
    ap.add_argument("--rank-on", default="target", choices=["target", "blind"],
                    help="which model's signal table the TreeSHAP ranking is fitted on. "
                         "'blind' is the mirror-image control: give the blind its best "
                         "possible columns and see whether the target still leads.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-preds", default=None,
                    help="prefix; writes out-of-fold predictions so separate task-type "
                         "runs can be pooled into one AUC afterwards")
    a = ap.parse_args()
    kinds = tuple(a.kinds.split(","))
    tab = None
    if "tabpfn" in kinds:
        from di.new_signal_audits import make_tabpfn
        tab = make_tabpfn(a.device, 4)

    out = {"meta": {"created": time.strftime("%Y-%m-%d %H:%M"), "args": vars(a),
                    "procedure": "whole table (all numeric cols incl. n_pool, no support "
                                 "filter, no constant drop, no pre-imputation) -> XGBoost "
                                 "TreeSHAP ranking on ALL rows of the target -> top-K -> "
                                 "LR/XGB/TabPFN on target and blind, repeated stratified "
                                 "5-fold x4, RAW (unfolded) AUC"},
           "sets": {}}
    for name in a.sets.split(","):
        ids, y, sigs, Xts, Xb = LOADERS[name]()
        if a.models:
            Xts = {k: v for k, v in Xts.items() if k in set(a.models.split(","))}
        splits = list(RepeatedStratifiedKFold(n_splits=a.splits, n_repeats=a.repeats,
                                              random_state=0).split(Xb, y))
        rows = []
        print(f"\n[{name}] n={len(y)} ({int(y.sum())} member)  columns={len(sigs)} "
              f"(NOTHING filtered)  NaN={100*np.mean(~np.isfinite(Xb)):.1f}% blind", flush=True)
        for scope in a.scopes.split(","):
          cols = scope_cols(sigs, scope)
          if not cols:
              print(f"   scope {scope}: no columns, skipped", flush=True); continue
          scope_sigs = [sigs[j] for j in cols]
          for mname, Xt_all in Xts.items():
            Xt, Xb_s = Xt_all[:, cols], Xb[:, cols]
            Xrank = Xt if a.rank_on == "target" else Xb_s
            c = xgb.XGBClassifier(**XGB_PARAMS)
            c.fit(Xrank, y)
            imp = np.abs(c.get_booster().predict(xgb.DMatrix(Xrank),
                                                 pred_contribs=True)[:, :-1]).mean(0)
            order = np.argsort(-imp)
            fold_order = None
            if a.rank_per_fold:
                fold_order = []
                for tr, _ in splits:
                    cf = xgb.XGBClassifier(**XGB_PARAMS); cf.fit(Xrank[tr], y[tr])
                    fi = np.abs(cf.get_booster().predict(
                        xgb.DMatrix(Xrank[tr]), pred_contribs=True)[:, :-1]).mean(0)
                    fold_order.append(np.argsort(-fi))
            # "all" = no selection at all: every column in the scope is used
            Klist = sorted({len(cols) if k.strip() == "all" else int(k)
                            for k in a.ks.split(",")})
            for K in [k for k in Klist if k <= len(cols)]:
                sel = order[:K]
                for kind in kinds:
                    pt, pb = np.zeros(len(y)), np.zeros(len(y))
                    ft, fb = [], []          # per-fold AUCs -> standard error for 1-SE
                    for fi_, (tr, te) in enumerate(splits):
                        sel_f = sel if fold_order is None else fold_order[fi_][:K]
                        qt = fitpred(kind, Xt[tr][:, sel_f], y[tr], Xt[te][:, sel_f], tab)
                        qb = fitpred(kind, Xb_s[tr][:, sel_f], y[tr], Xb_s[te][:, sel_f], tab)
                        pt[te] += qt; pb[te] += qb
                        if len(np.unique(y[te])) > 1:
                            ft.append(float(roc_auc_score(y[te], qt)))
                            fb.append(float(roc_auc_score(y[te], qb)))
                    t = float(roc_auc_score(y, pt)); b = float(roc_auc_score(y, pb))
                    if a.save_preds:   # so separate task-type runs can be pooled later
                        np.savez_compressed(f"{a.save_preds}__{kind}__K{K}.npz",
                                            y=y, pt=pt, pb=pb,
                                            ids=np.array([str(i) for i in ids]))
                    rows.append({"scope": scope, "n_cols": len(cols), "model": mname,
                                 "rank_on": a.rank_on, "rank_per_fold": a.rank_per_fold,
                                 "K": K, "no_selection": K == len(cols), "clf": kind, "target": round(t, 4),
                                 "blind": round(b, 4), "gap": round(t - b, 4),
                                 "fold_target_mean": round(float(np.mean(ft)), 4) if ft else None,
                                 "fold_target_se": round(float(np.std(ft, ddof=1) / np.sqrt(len(ft))), 5) if len(ft) > 1 else None,
                                 "fold_blind_mean": round(float(np.mean(fb)), 4) if fb else None,
                                 "n_folds": len(ft)})
                    print(f"   {scope:14s} {mname:9s} K={K:<3d} {kind:6s} target {t:.3f}  "
                          f"blind {b:.3f}  gap {t-b:+.3f}", flush=True)
            rows.append({"scope": scope, "model": mname,
                         "top_features": [scope_sigs[j] for j in order[:16]]})
        out["sets"][name] = {"n": len(y), "n_member": int(y.sum()),
                             "n_columns": len(sigs), "rows": rows}
        p = Path(a.out) if a.out else REPO / "exports" / f"hyp_baseline_{name}.json"
        p.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
