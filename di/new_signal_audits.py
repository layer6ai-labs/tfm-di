"""Run the submission's two DI audits (internal supervised + external label-free)
on the NEW TabDPT and SAP signal sets.

Signal sets covered
  sap_fullgrid     SAP RPT-OSS vs TabPFN-2.5 blind, full 683-key grid incl. the new
                   families (ctxeqq / nnsplit / lperm / ftx / rowshuf / tshuffle /
                   tabdpt_sim). exports/sap_fullgrid_signals_{sap-rpt-oss,tabpfn25}.json
  sap_redteam      H14 semantic-channel probes (semantic / anon-name / shuffled-value /
                   numeric input channels x acc,p_true,loss,conf, full + kNN context).
                   exports/hyp_sap_redteam_raw.json
  tabdpt_redteam   H13 red-team probes: SHARP (surface sharpness), PSTAB (partition
                   stability), DEJAVU (exact-row recognition), ANTIKNN (accuracy under
                   neighbour deletion). exports/hyp_redteam_raw/probe_r{1,2}_*.json
  tabdpt_fullgrid  4 TabDPT seeds vs blind on the July grid — the submission's own
                   pool, used here both as an implementation control (must reproduce
                   the published gaps) and to score the NEW families on TabDPT.

Usage
  python di/new_signal_audits.py --sets sap_redteam,tabdpt_redteam --nperm 200
  python di/new_signal_audits.py --sets sap_fullgrid --nperm 200 --njobs 64
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

REPO = Path(os.environ.get("AUDIT_REPO", "/p/project1/hai_1159/tfm-di")).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from di.audit_pipeline import atomic_dump  # noqa: E402
from di.audit_pipeline import build_signal_set as _build_signal_set  # noqa: E402
from di.audit_pipeline import external_audit, internal_audit, per_signal_table  # noqa: E402

IMPUTE = "base"       # set from --impute; see audit_pipeline.build_signal_set
MIN_SUPPORT = 0.5     # set from --min-support: a signal enters the pool only if it
                      # is >= this fraction finite in EVERY model, blind included


def build_signal_set(*a, **kw):
    kw.setdefault("impute", IMPUTE)
    kw.setdefault("min_support", MIN_SUPPORT)
    return _build_signal_set(*a, **kw)

EXPORTS = REPO / "exports"
META_KEYS = {"dataset_id", "dataset_name", "true_label", "task_type", "n_pool",
             "n_features", "success", "error", "is_member", "member", "source_is_member",
             "iid_split", "model_backend", "split_model", "grid_version", "target",
             "n_str_cols", "n_classes", "did", "n", "d", "C"}

# New signal families added after the submission grid was first extracted.
NEW_FAMS = ("ctxeqq", "nnsplit", "lperm", "ftx", "rowshuf", "tshuffle", "tabdpt_sim")
CORRUPTION_FAMS = ("tserum", "bwash", "mislbl", "qleak", "rowdup")


def _fam(sig):
    for f in NEW_FAMS + CORRUPTION_FAMS + ("ctx", "noise", "col", "splitsize", "temp", "base"):
        if sig == f or sig.startswith(f + "_"):
            return f
    return sig.split("_")[0]


def grid_scopes(sigs):
    """full / each family / the union of the new families / the corruption union."""
    sc = {"full": list(range(len(sigs)))}
    fams = {}
    for j, s in enumerate(sigs):
        fams.setdefault(_fam(s), []).append(j)
    for f, idx in fams.items():
        if len(idx) >= 3:
            sc[f"fam:{f}"] = idx
    sc["new_families"] = [j for j, s in enumerate(sigs) if _fam(s) in NEW_FAMS]
    sc["corruption"] = [j for j, s in enumerate(sigs) if _fam(s) in CORRUPTION_FAMS]
    return sc


# ------------------------------------------------------------------- loaders
def load_sap_fullgrid():
    dt = json.load(open(EXPORTS / "sap_fullgrid_signals_sap-rpt-oss.json"))
    db = {r["dataset_id"]: r for r in json.load(open(EXPORTS / "sap_fullgrid_signals_tabpfn25.json"))}
    dt = [r for r in dt if r["dataset_id"] in db and r.get("success", True)]
    ids = [r["dataset_id"] for r in dt]
    y = np.array([1 if r.get("true_label") in (1, True) else 0 for r in dt], int)
    sigs = sorted({k for r in dt for k, v in r.items()
                   if k not in META_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool)}
                  & {k for k in db[ids[0]]})
    M = lambda recs: np.array([[_num(r.get(s)) for s in sigs] for r in recs], float)
    return build_signal_set("sap_fullgrid", ids, y, sigs,
                            {"sap-rpt-oss": M(dt)}, M([db[i] for i in ids]),
                            scope_fn=grid_scopes,
                            notes="SAP RPT-OSS (member = T4 table with >150 rows) vs "
                                  "TabPFN-2.5 blind, full grid incl. new families.")


def load_tabdpt_fullgrid():
    def load(pat):
        f = sorted(glob.glob(str(REPO / pat)))[-1]
        d = json.load(open(f))
        return {str(r["dataset_id"]): r for r in d["successful_results"] if r.get("success")}
    seeds = {s: load(f"di_grid_results_{s}_2026071*_*.json")
             for s in ("default", "seed42", "seed123", "seed456")}
    bl = load("di_grid_results_tabpfn25_2026071*_*.json")
    common = set.intersection(*[set(g) for g in seeds.values()], set(bl))
    ids = sorted(i for i in common
                 if seeds["default"][i].get("task_type") == "classification"
                 and not seeds["default"][i].get("iid_split"))
    y = np.array([1 if bl[i].get("true_label", bl[i].get("is_member")) else 0 for i in ids], int)
    srcs = list(seeds.values()) + [bl]
    sigs = sorted({k for g in srcs for r in g.values() for k, v in r.items()
                   if k not in META_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool)})
    M = lambda g: np.array([[_num(g[i].get(s)) for s in sigs] for i in ids], float)
    return build_signal_set("tabdpt_fullgrid", ids, y, sigs,
                            {s: M(g) for s, g in seeds.items()}, M(bl),
                            scope_fn=grid_scopes,
                            notes="4 TabDPT seeds vs TabPFN-2.5 blind, July grid "
                                  "(the submission pool; also carries the new families).")


def load_sap_redteam():
    """H14 probes. The blind has no semantic channel, so each target channel is
    compared against the blind's counterpart measure (blind_* for full-context
    probes, blindk_* for the kNN-context probes) — the blind genuinely cannot
    distinguish channels, which is the point of the hypothesis."""
    recs = [r for r in json.load(open(EXPORTS / "hyp_sap_redteam_raw.json")) if r.get("success", True)]
    ids = [r["dataset_id"] for r in recs]
    y = np.array([1 if r["is_member"] else 0 for r in recs], int)
    meas = ["acc", "p_true", "loss", "conf"]
    pairs = []                                   # (target key, blind counterpart)
    for ch in ("semantic", "anonname", "shuffval", "numeric"):
        pairs += [(f"sap_{ch}_{m}", f"blind_{m}") for m in meas]
    for ch in ("semantic", "numeric"):
        pairs += [(f"sapk_{ch}_{m}", f"blindk_{m}") for m in meas]
    pairs = [(t, b) for t, b in pairs if t in recs[0] and b in recs[0]]
    sigs = [t for t, _ in pairs]
    Xt = np.array([[_num(r.get(t)) for t, _ in pairs] for r in recs], float)
    Xb = np.array([[_num(r.get(b)) for _, b in pairs] for r in recs], float)

    def scopes(ss):
        sc = {"full": list(range(len(ss)))}
        for ch in ("semantic", "anonname", "shuffval", "numeric"):
            idx = [j for j, s in enumerate(ss) if f"_{ch}_" in s]
            if idx:
                sc[f"chan:{ch}"] = idx
        sc["matched_numeric"] = [j for j, s in enumerate(ss) if "_numeric_" in s]
        sc["semantic_only"] = [j for j, s in enumerate(ss)
                               if "_semantic_" in s or "_anonname_" in s or "_shuffval_" in s]
        return sc

    return build_signal_set("sap_redteam", ids, y, sigs, {"sap-rpt-oss": Xt}, Xb,
                            scope_fn=scopes,
                            notes="H14 semantic-channel probes. Blind columns are the "
                                  "counterpart measure (the blind sees numeric input only).")


def load_tabdpt_redteam():
    def rd(p):
        d = json.load(open(EXPORTS / "hyp_redteam_raw" / p))
        return {str(r["did"]): r for r in d["results"]}
    r1t, r1b = rd("probe_r1_default.json"), rd("probe_r1_blind.json")
    r2t, r2b = rd("probe_r2_default.json"), rd("probe_r2_blind.json")
    ids = sorted(set(r1t) & set(r1b) & set(r2t) & set(r2b), key=int)
    y = np.array([1 if r1t[i]["member"] else 0 for i in ids], int)
    s1 = ["acc", "loss", "conf", "sharp_full", "sharp_knn", "pstab", "dejavu_full"]
    s2 = ["acc_norm", "loss_norm", "acc_anti", "loss_anti", "acc_drop", "loss_rise"]
    sigs = [f"r1_{s}" for s in s1] + [f"r2_{s}" for s in s2]
    build = lambda a, b: np.array(
        [[_num(a[i].get(s)) for s in s1] + [_num(b[i].get(s)) for s in s2] for i in ids], float)

    def scopes(ss):
        sc = {"full": list(range(len(ss)))}
        sc["baseline"] = [j for j, s in enumerate(ss) if s in ("r1_acc", "r1_loss", "r1_conf")]
        sc["preregistered"] = [j for j, s in enumerate(ss)
                               if any(k in s for k in ("sharp", "pstab", "dejavu", "anti", "drop", "rise"))]
        for k, nm in (("sharp", "SHARP"), ("pstab", "PSTAB"), ("dejavu", "DEJAVU")):
            idx = [j for j, s in enumerate(ss) if k in s]
            if idx:
                sc[f"probe:{nm}"] = idx
        sc["probe:ANTIKNN"] = [j for j, s in enumerate(ss)
                               if any(k in s for k in ("anti", "drop", "rise"))]
        return sc

    return build_signal_set("tabdpt_redteam", ids, y, sigs,
                            {"default": build(r1t, r2t)}, build(r1b, r2b),
                            scope_fn=scopes,
                            notes="H13 red-team probes (SHARP/PSTAB/DEJAVU/ANTIKNN) on "
                                  "TabDPT default vs TabPFN-2.5 blind. n is tiny by design "
                                  "(interactive small-scale screen).")


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return np.nan
    return float(v) if np.isfinite(v) else np.nan


LOADERS = {"sap_fullgrid": load_sap_fullgrid, "sap_redteam": load_sap_redteam,
           "tabdpt_redteam": load_tabdpt_redteam, "tabdpt_fullgrid": load_tabdpt_fullgrid}


def make_tabpfn(device="cuda", n_est=4):
    """TabPFN-2.5 (SYNTHETIC checkpoint) as the meta-classifier — the strongest
    auditor in the submission (+0.095 on the TabDPT grid). Needs the torch venv
    and a GPU; the login-node Arbiter kills it, so run this on SLURM."""
    from tabpfn import TabPFNClassifier
    ckpt = Path.home() / ".cache" / "tabpfn" / "tabpfn-v2.5-classifier-v2.5_default-2.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"synthetic TabPFN-2.5 classifier ckpt missing: {ckpt}")
    try:    # pre-loaded weights (much faster across thousands of fits)
        from di.datasets.tabpfn25_wrapper import _get_cached_specs
        specs = _get_cached_specs(TabPFNClassifier, str(ckpt), device, "classifier")
    except Exception:                       # older wrapper in this checkout
        specs = str(ckpt)
    kw = dict(device=device, ignore_pretraining_limits=True, model_path=specs,
              n_estimators=n_est)
    try:
        return TabPFNClassifier(random_state=0, **kw)
    except TypeError:
        return TabPFNClassifier(**kw)


# ---------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="sap_redteam,tabdpt_redteam")
    ap.add_argument("--scopes", default="", help="comma list; default = per-set choice")
    ap.add_argument("--kinds", default="lr,xgb")
    ap.add_argument("--nperm", type=int, default=0)
    ap.add_argument("--ext-nperm", type=int, default=0)
    ap.add_argument("--njobs", type=int, default=1)
    ap.add_argument("--nsplits", type=int, default=50)
    ap.add_argument("--outer", default="loo",
                    help="comma list of outer CV schemes: loo (submission default) "
                         "and/or skf (repeated stratified 5-fold x10, immune to the "
                         "LOOCV base-rate flip at small n)")
    ap.add_argument("--impute", default="base", choices=["base", "masked", "complete"],
                    help="missing-data handicap control; the blind is much patchier "
                         "than the targets, so 'base' (independent median imputation) "
                         "favours the target")
    ap.add_argument("--models", default="", help="comma list to restrict target models")
    ap.add_argument("--min-support", type=float, default=0.5,
                    help="minimum finite fraction required in EVERY model for a signal "
                         "to enter the pool. Lowering it admits families the blind "
                         "barely covers (e.g. tshuffle on TabDPT, blind ~40%%), which "
                         "then lean heavily on imputation for the blind")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tabpfn-nest", type=int, default=4)
    ap.add_argument("--skip-internal", action="store_true")
    ap.add_argument("--skip-external", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    DEFAULT_SCOPES = {
        "sap_redteam": ["full", "matched_numeric", "semantic_only", "chan:semantic"],
        "tabdpt_redteam": ["full", "preregistered", "baseline"],
        "sap_fullgrid": ["full", "new_families", "corruption", "fam:tserum"],
        "tabdpt_fullgrid": ["full", "new_families", "corruption", "fam:tserum"],
    }
    kinds = tuple(a.kinds.split(","))
    globals()["IMPUTE"] = a.impute
    globals()["MIN_SUPPORT"] = a.min_support
    tab_clf = make_tabpfn(a.device, a.tabpfn_nest) if "tabpfn" in kinds else None
    out = {"meta": {"created": time.strftime("%Y-%m-%d %H:%M"), "args": vars(a)}, "sets": {}}
    for name in a.sets.split(","):
        name = name.strip()
        t0 = time.time()
        ss = LOADERS[name]()
        if a.models:
            want = set(a.models.split(","))
            ss.Xt = {k: v for k, v in ss.Xt.items() if k in want}
        scopes = [s for s in (a.scopes.split(",") if a.scopes else DEFAULT_SCOPES[name])
                  if s in ss.scopes]
        if not scopes:
            print(f"[{name}] no usable scopes under impute={a.impute} "
                  f"(all columns dropped) — skipping", flush=True)
            continue
        rec = {"n": ss.n, "n_member": int(ss.y.sum()), "n_non": int((ss.y == 0).sum()),
               "n_signals": len(ss.sigs), "notes": ss.notes, "impute": a.impute,
               "min_support": a.min_support,
               "missing_fraction": ss.missing,
               "models": list(ss.Xt), "scopes": {k: len(v) for k, v in ss.scopes.items()},
               "internal": {}, "external": {}, "per_signal": {}}
        print(f"[{name}] n={ss.n} ({rec['n_member']}m/{rec['n_non']}n) "
              f"signals={len(ss.sigs)} scopes={scopes}", flush=True)
        for sc in scopes:
            rec["per_signal"][sc] = per_signal_table(ss, sc)
            if not a.skip_internal:
                for ocv in a.outer.split(","):
                    outer = "loo" if ocv == "loo" else ("skf", 5, 4)
                    key = sc if ocv == "loo" else f"{sc}@{ocv}"
                    rec["internal"][key] = internal_audit(
                        ss, scope=sc, kinds=kinds, nperm=a.nperm, njobs=a.njobs,
                        outer=outer, verbose=True, tab_clf=tab_clf)
                    rec["internal"][key]["outer_cv"] = ocv
                    print(f"  internal[{key}] "
                          f"{ {k: v['gap'] for k, v in rec['internal'][key]['cells'].items()} }",
                          flush=True)
            if not a.skip_external:
                rec["external"][sc] = external_audit(ss, scope=sc, nsplits=a.nsplits,
                                                     nperm=a.ext_nperm)
                e = rec["external"][sc]
                print(f"  external[{sc}] target={e['target_mean_auc']} "
                      f"blind={e['blind_auc']} gap={e['gap']}", flush=True)
        rec["secs"] = round(time.time() - t0, 1)
        out["sets"][name] = rec
        path = Path(a.out) if a.out else EXPORTS / f"hyp_newsignal_audit_{name}.json"
        atomic_dump({"meta": out["meta"], "sets": {name: rec}}, path)
        print(f"[{name}] done in {rec['secs']}s -> {path}", flush=True)
    return out


if __name__ == "__main__":
    main()
