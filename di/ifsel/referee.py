"""The referee: the ONLY component that may see membership labels or the blind.

A *scheme* is a label-free function

    select(Xk, Xs, meta_k, meta_s, rng) -> (mask, Xk2, Xs2)

where
  Xk       (nk, p)  the auditor's KNOWN non-members, one row per table
  Xs       (ns, p)  the SUSPECT pile, membership unknown, mixed
  meta_k/s (n, m)   observable table metadata (n_pool, n_features, ...)
  mask     (p,)     bool, which signals to keep
  Xk2/Xs2           optionally transformed matrices (e.g. residualised); return
                    the inputs unchanged if the scheme only selects.

The scheme never receives y and never receives the blind. The referee then runs
the IDENTICAL scheme independently on the blind's own signals -- that is what an
auditor pointing the same procedure at an untrained model would get, and it is
the control we compare against.

Reported per corpus, per scheme:
  target AUC   member-vs-nonmember AUC inside the suspect pile (iForest score)
  blind  AUC   the same, from the blind's matrices, scheme re-derived on them
  gap          target - blind
  TPR@FPR      via split-conformal p-values calibrated on the known pile
  null q95     label-permutation null on the gap (optional, expensive)

ALWAYS read the blind's ABSOLUTE AUC: a blind below ~0.49 is ranked backwards
and its "gap" is an artifact, not a detection.
"""
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

import os  # noqa: E402
# Same convention as di/baseline_topk.py: $AUDIT_REPO is the tree that holds
# exports/ and the di_grid_results_*.json.
AUDIT_REPO = Path(os.environ.get("AUDIT_REPO", "/p/project1/hai_1159/tfm-di")).resolve()
CACHE = Path(os.environ.get("IFSEL_CACHE", AUDIT_REPO / "exports" / "ifsel_cache"))
CACHE.mkdir(parents=True, exist_ok=True)


def load(name):
    d = np.load(CACHE / f"{name}.npz", allow_pickle=True)
    return (d["Xt"], d["Xb"], d["y"].astype(int), [str(s) for s in d["sigs"]],
            d["meta"], [str(s) for s in d["meta_names"]])


def impute(X, ref=None):
    """Median-fill. Medians come from `ref` (the known pile) when given, so the
    suspect pile never donates statistics the auditor could not compute."""
    X = np.array(X, float)
    src = X if ref is None else np.asarray(ref, float)
    med = np.nanmedian(np.where(np.isfinite(src), src, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(X)
    X[bad] = np.take(med, np.nonzero(bad)[1])
    return X


def iforest(Xk, Xs, seed, **kw):
    p = dict(n_estimators=200, contamination="auto", random_state=seed)
    p.update(kw)
    f = IsolationForest(**p).fit(Xk)
    return -f.score_samples(Xk), -f.score_samples(Xs)


def conformal_p(cal, test):
    cal = np.sort(np.asarray(cal))
    return (1.0 + len(cal) - np.searchsorted(cal, np.asarray(test), side="left")) / (len(cal) + 1.0)


def _split(y, rng, known_frac=0.5):
    non, mem = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    p = rng.permutation(non)
    k = max(5, int(round(known_frac * len(p))))
    known, held = p[:k], p[k:]
    return known, np.concatenate([held, mem]), np.concatenate([np.zeros(len(held)), np.ones(len(mem))])


def one_model(X, y, scheme, meta, rng, alphas, ifkw, force_mask=None, scorer=iforest):
    """Run the scheme + iForest on ONE model's matrices. Never sees the other.

    force_mask short-circuits the scheme's own selection and imposes a mask
    computed elsewhere -- used to score the blind on the TARGET-selected columns
    (see `blind_selection` in evaluate()).
    """
    known, sus, yt = _split(y, rng)
    Xk, Xs = impute(X[known], ref=X[known]), impute(X[sus], ref=X[known])
    out = scheme(Xk, Xs, meta[known], meta[sus], rng)
    mask, Xk2, Xs2 = out if isinstance(out, tuple) else (out, Xk, Xs)
    if mask is None:
        mask = np.ones(Xk2.shape[1], bool)
    mask = np.asarray(mask, bool)
    if force_mask is not None and len(force_mask) == Xk2.shape[1]:
        mask = np.asarray(force_mask, bool)
    if mask.sum() < 2:
        return None
    sk, ss = scorer(Xk2[:, mask], Xs2[:, mask], int(rng.integers(1 << 30)) % (2**31), **ifkw)
    r = {"auc": float(roc_auc_score(yt, ss)), "nsel": int(mask.sum()), "scores": ss,
         "yt": yt, "mask": mask}
    pv = conformal_p(sk, ss)
    for a in alphas:
        r[f"tpr@{a}"] = float((pv <= a)[yt == 1].mean())
        r[f"fpr@{a}"] = float((pv <= a)[yt == 0].mean())
    return r


def evaluate(corpus, scheme, draws=10, seed0=7, alphas=(0.01, 0.10), ifkw=None,
             y_override=None, blind_selection="target", scorer="iforest"):
    """Score a scheme on one corpus.

    blind_selection decides WHO PICKS THE BLIND'S SIGNALS:
      "target" (default)  the blind is scored on exactly the columns the scheme
                          chose from the TARGET. The blind selects nothing -- it
                          is only ever our control, never an auditor. This is the
                          same convention as the internal audit, where the
                          XGB/TreeSHAP ranking comes from the target alone.
      "own"               the blind re-derives the scheme on its own signals,
                          i.e. "what would this whole procedure return if pointed
                          at an untrained model?". The blind gets its best shot,
                          so gaps come out SMALLER; it is the conservative read.
    """
    Xt, Xb, y, sigs, meta, _ = load(corpus)
    if y_override is not None:
        y = y_override
    ifkw = ifkw or {}
    from di.ifsel.scorers import SCORERS
    sf = SCORERS[scorer] if isinstance(scorer, str) else scorer
    rt, rb = [], []
    for i in range(draws):
        a = one_model(Xt, y, scheme, meta, np.random.default_rng(seed0 + i), alphas, ifkw,
                      scorer=sf)
        fm = a["mask"] if (a and blind_selection == "target") else None
        b = one_model(Xb, y, scheme, meta, np.random.default_rng(seed0 + i), alphas, ifkw,
                      force_mask=fm, scorer=sf)
        if a and b:
            rt.append(a); rb.append(b)
    if not rt:
        return {"corpus": corpus, "error": "scheme selected <2 signals on every draw"}
    m = lambda rs, k: float(np.mean([r[k] for r in rs]))
    res = {"corpus": corpus, "n": int(len(y)), "n_member": int(y.sum()),
           "draws": len(rt), "n_selected": int(np.mean([r["nsel"] for r in rt])),
           "target_auc": round(m(rt, "auc"), 4), "blind_auc": round(m(rb, "auc"), 4),
           "gap": round(m(rt, "auc") - m(rb, "auc"), 4),
           "target_auc_sd": round(float(np.std([r["auc"] for r in rt], ddof=1)), 4)
           if len(rt) > 1 else 0.0,
           "blind_auc_sd": round(float(np.std([r["auc"] for r in rb], ddof=1)), 4)
           if len(rb) > 1 else 0.0,
           # draws are paired: target and blind share the same rng seed, hence the
           # same known/suspect split, so the per-draw gap is a matched difference
           "gap_sd": round(float(np.std([a["auc"] - b["auc"] for a, b in zip(rt, rb)],
                                        ddof=1)), 4) if len(rt) > 1 else 0.0}
    for a in alphas:
        res[f"tpr@{a}"] = round(m(rt, f"tpr@{a}"), 4)
        res[f"fpr@{a}"] = round(m(rt, f"fpr@{a}"), 4)
    res["blind_below_chance"] = bool(res["blind_auc"] < 0.49)
    res["blind_selection"] = blind_selection
    res["scorer"] = scorer if isinstance(scorer, str) else scorer.__name__
    res["n_selected_blind"] = int(np.mean([r["nsel"] for r in rb]))
    # dominance guard: what an auditor gets from OBSERVABLE METADATA alone,
    # never querying the model. On sap_hybrid membership IS n_pool>150, so this
    # scores 0.980 with TPR@10%=1.000 and no behavioural scheme can be said to
    # "work" unless it beats it.
    res["meta_only_auc"] = _meta_only(corpus, y, seed0, draws)
    res["dominated_by_metadata"] = bool(res["meta_only_auc"] >= res["target_auc"])
    res["draw_spread"] = [round(float(min(r["auc"] for r in rt) - max(r["auc"] for r in rb)), 4),
                          round(float(max(r["auc"] for r in rt) - min(r["auc"] for r in rb)), 4)]
    return res


_META_CACHE = {}


def _meta_only(corpus, y, seed0, draws):
    """iForest run on log-metadata only -- the model is never queried."""
    key = (corpus, y.tobytes(), seed0, draws)
    if key in _META_CACHE:
        return _META_CACHE[key]
    _, _, _, _, meta, _ = load(corpus)
    M = np.log1p(np.where(np.isfinite(meta), meta, np.nan))
    keep = np.isfinite(M).any(0)
    if keep.sum() < 1:
        return float("nan")
    M = impute(M[:, keep])
    a = []
    for i in range(draws):
        rng = np.random.default_rng(seed0 + i)
        known, sus, yt = _split(y, rng)
        if M.shape[1] < 2:
            sk, ss = M[known, 0], M[sus, 0]
        else:
            sk, ss = iforest(M[known], M[sus], seed0 + i)
        v = roc_auc_score(yt, ss)
        a.append(max(v, 1 - v))
    _META_CACHE[key] = round(float(np.mean(a)), 4)
    return _META_CACHE[key]


def permutation_null(corpus, scheme, nulls=100, draws=3, ifkw=None, seed=0):
    """Gap distribution when membership carries no information. Permuting y keeps
    the signal matrices and the class balance identical, so anything the scheme
    can still 'find' is selection noise."""
    _, _, y, _, _, _ = load(corpus)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nulls):
        r = evaluate(corpus, scheme, draws=draws, seed0=101, ifkw=ifkw,
                     y_override=rng.permutation(y))
        if "gap" in r:
            out.append(r["gap"])
    return np.array(out)
