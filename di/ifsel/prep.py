"""Cache each corpus as one npz so probe agents load in a second, without pandas.

Writes exports/ifsel_cache/<corpus>.npz with:
  Xt   (n, p) target signal matrix, NaN where missing
  Xb   (n, p) blind  signal matrix, NaN where missing   (REFEREE ONLY)
  y    (n,)   membership labels                          (REFEREE ONLY)
  sigs (p,)   signal names
  meta (n, m) observable table metadata  + meta_names
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

import os  # noqa: E402
from di.ifsel.referee import AUDIT_REPO, CACHE  # noqa: E402
REPO = AUDIT_REPO
OUT = CACHE
SNAP = os.environ.get(
    "TFMS_SNAPSHOT",
    ".hf_hub_cache/models--dwahdany--tfms/snapshots/"
    "28723d00fcdf35610efe04dbcc54f1dda3d50d33/signals")
NON_FEATURES = {"true_label", "is_member", "member", "dataset_id", "dataset_name",
                "success", "task_type", "grid_version", "iid_split", "did",
                "n_pool", "n_features", "n_classes", "n_str_cols", "n", "d", "C",
                "elapsed_s", "dataset_id_str", "label", "y", "target"}
# metadata an outside auditor can genuinely observe about their own tables
META = ["n_pool", "n_features", "n_classes", "n_str_cols"]


def _clean(a):
    a = np.where(np.isfinite(a), a, np.nan)
    return np.where(np.abs(a) > np.finfo(np.float32).max / 10, np.nan, a)


def from_parquet(tfile, bfile, task="classification"):
    t = pd.read_parquet(REPO / tfile)
    b = pd.read_parquet(REPO / bfile)
    keep = t.success if task is None else ((t.task_type == task) & t.success)
    t = t[keep].sort_values("dataset_id").reset_index(drop=True)
    b = b.set_index("dataset_id").loc[t.dataset_id].reset_index()
    assert (t.true_label.values == b.true_label.values).all()
    sigs = sorted(c for c in t.columns if c not in NON_FEATURES and c in b.columns
                  and pd.api.types.is_numeric_dtype(t[c])
                  and not pd.api.types.is_bool_dtype(t[c]))
    to = lambda d: _clean(d[sigs].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    mcols = [c for c in META if c in t.columns]
    meta = _clean(t[mcols].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    return to(t), to(b), t.true_label.astype(int).to_numpy(), sigs, meta, mcols


def from_grid_json():
    def ld(pat):
        f = sorted(glob.glob(str(REPO / pat)))[-1]
        return {str(r["dataset_id"]): r for r in json.load(open(f))["successful_results"]
                if r.get("success")}
    tg, bl = ld("di_grid_results_default_2026071*_*.json"), ld("di_grid_results_tabpfn25_2026071*_*.json")
    ids = sorted(i for i in set(tg) & set(bl) if tg[i].get("task_type") == "classification")
    y = np.array([1 if bl[i].get("true_label", bl[i].get("is_member")) else 0 for i in ids], int)
    isnum = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    sigs = sorted({k for g in (tg, bl) for r in g.values() for k, v in r.items()
                   if k not in NON_FEATURES and isnum(v)})
    def mat(d, cols):
        a = np.full((len(ids), len(cols)), np.nan)
        for i, di in enumerate(ids):
            r = d.get(di, {})
            for j, s in enumerate(cols):
                v = r.get(s)
                if isnum(v) and np.isfinite(v):
                    a[i, j] = v
        return a
    return mat(tg, sigs), mat(bl, sigs), y, sigs, mat(tg, META), META


def subsample(c, n, seed=0):
    """Stratified subsample. 100k x 716 would put ~25k rows into every iForest
    fit; 20k keeps the control comfortably powered and the audit tractable."""
    Xt, Xb, y, sigs, meta, mc = c
    rng = np.random.default_rng(seed)
    per = n // 2
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == 1), per, False),
                                  rng.choice(np.flatnonzero(y == 0), per, False)]))
    return Xt[idx], Xb[idx], y[idx], sigs, meta[idx], mc


def union(a, b):
    """Two disjoint pools under one vocabulary, stacked on their shared columns."""
    Xta, Xba, ya, sa, ma, mca = a
    Xtb, Xbb, yb, sb, mb, mcb = b
    sigs = [s for s in sa if s in set(sb)]
    ia = [sa.index(s) for s in sigs]
    ib = [sb.index(s) for s in sigs]
    mc = [c for c in mca if c in set(mcb)]
    ja = [mca.index(c) for c in mc]
    jb = [mcb.index(c) for c in mc]
    return (np.vstack([Xta[:, ia], Xtb[:, ib]]), np.vstack([Xba[:, ia], Xbb[:, ib]]),
            np.concatenate([ya, yb]), sigs, np.vstack([ma[:, ja], mb[:, jb]]), mc)


CORPORA = {
    # the whole T4 chunk under one vocabulary: the tables that natively drew a
    # categorical target, plus the regression tables re-extracted under the
    # hybrid target-column policy.
    "sap_allcls_hybrid": lambda: union(
        from_parquet("exports/sap_binary_t4_all_datasets_corrected.parquet",
                     "exports/tabpfn25_binary_t4_all_datasets_corrected.parquet",
                     "classification"),
        from_parquet("exports/sap_t4_regression_hybrid.parquet",
                     "exports/tabpfn25_t4_regression_hybrid.parquet")),
    # native_cls + FORCED-CLS: the same native-classification pool, unioned with
    # only those regression tables that actually have a low-cardinality column to
    # force as the target (~41% of them), rather than hybrid's fall-back binning.
    "sap_allcls_forcecls": lambda: union(
        from_parquet("exports/sap_binary_t4_all_datasets_corrected.parquet",
                     "exports/tabpfn25_binary_t4_all_datasets_corrected.parquet",
                     "classification"),
        from_parquet("exports/sap_t4_regression_forcecls.parquet",
                     "exports/tabpfn25_t4_regression_forcecls.parquet", None)),
    # discovery corpus. HYBRID target-column policy: the forced-cls draw where the
    # table has a low-cardinality column, capped min-unique binning otherwise.
    # bin10 is NOT usable -- quantile-binning a continuous target into 10 classes
    # destroys the signal (forced-cls +0.12 vs bin10 0.000 on identical tables).
    "sap_hybrid": lambda: from_parquet("exports/sap_t4_regression_hybrid.parquet",
                                       "exports/tabpfn25_t4_regression_hybrid.parquet"),
    # confirmation: the real target model. 224 datasets (160 classification + 64
    # regression). The parquet ships a column literally named `label` that is
    # byte-identical to true_label -- it is in NON_FEATURES, keep it there.
    "tabdpt": lambda: from_parquet(f"{SNAP}/tabdpt/binary_all_tabdpt/datasets.parquet",
                                   f"{SNAP}/tabpfn25/binary_all_tabdpt/datasets.parquet", None),
    "tabdpt_json": from_grid_json,
    # STRONG negative control. nanotabpfn's own synthetic training stream: 100k
    # tables, 50k/50k, n_pool constant at 50 and n_features constant at 3, so no
    # size confound exists at all. Not one of 672 live signals exceeds univariate
    # |AUC| 0.52. Unlike realtabpfn it HAS a blind-missingness deficit (2.5% vs
    # 7.9%), so the imputation artifact has a channel here and this control can
    # actually catch it.
    "nanotabpfn": lambda: subsample(
        from_parquet(f"{SNAP}/nanotabpfn/nanotabpfn_all/datasets.parquet",
                     f"{SNAP}/tabpfn25/nanotabpfn_all/datasets.parquet", None), 20000),
    # negative control: light real-data finetuning, known null
    "realtabpfn": lambda: from_parquet("exports/realtabpfn25_binary_realtabpfn25.parquet",
                                       "exports/tabpfn25_binary_realtabpfn25.parquet", None),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    want = sys.argv[1:] or list(CORPORA)
    for name in want:
        Xt, Xb, y, sigs, meta, mcols = CORPORA[name]()
        np.savez_compressed(OUT / f"{name}.npz", Xt=Xt, Xb=Xb, y=y,
                            sigs=np.array(sigs), meta=meta, meta_names=np.array(mcols))
        print(f"{name:12s} n={len(y):6d} ({int(y.sum())}m/{int((y==0).sum())}n) "
              f"p={len(sigs):5d}  target cov {np.isfinite(Xt).mean():.1%}  "
              f"blind cov {np.isfinite(Xb).mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
