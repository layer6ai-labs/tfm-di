"""Informativeness vs stability -- two screens that are easy to conflate.

INFORMATIVENESS (what schemes.stability_screen actually implements, despite its
name): keep signal j if its known-vs-suspect separation beats the 95th percentile
of separations between random halves of the KNOWN pile.
        obs_j = sep(Xk[:,j], Xs[:,j])   >   q95_j = q95 of sep(Xk_A, Xk_B)
It LOOKS AT THE SUSPECT PILE. That is its power and its danger: it selects
exactly the signals on which K and S differ, and on these corpora the largest
K-vs-S differences are the missing-cell pattern and the row-count confound, not
membership.

STABILITY (a noise-floor filter): reject signal j if its OWN half-split noise
floor q95_j is large relative to the pool.
        keep_j = q95_j <= tau * median_k(q95_k)
It never looks at the suspect pile at all. It cannot select on drift, size or
missingness, because none of those enter the criterion -- so it is immune by
construction to the artifact that killed the design fleet. The cost is that it is
blind to informativeness: it keeps quiet signals whether or not they carry
membership, and drops noisy ones whether or not they do.

The two are complementary rather than rival, so the conjunction is included.
"""
import numpy as np
from scipy.special import ndtr
from scipy.stats import rankdata


def _sep_cols(A, B):
    """|AUC - 0.5| of A-vs-B for EVERY column at once.

    Identical to looping roc_auc_score per column (max abs diff 1e-16, ties
    included, since rankdata uses the same average-rank convention) but ~10x
    faster: the screen calls this B+1 times over ~700-1100 columns, so the
    per-column Python loop was the whole cost of the audit.
    """
    n1, n2 = len(A), len(B)
    R = rankdata(np.vstack([A, B]), axis=0)
    auc = (R[n1:].sum(0) - n2 * (n2 + 1) / 2.0) / (n1 * n2)
    return np.abs(auc - 0.5)


def _sep(a, b):
    return float(_sep_cols(np.asarray(a).reshape(-1, 1), np.asarray(b).reshape(-1, 1))[0])


def _floor(Xk, rng, B=40):
    """Per-signal noise floor: q95 of the separation between random halves of the
    KNOWN pile. Uses Xk only -- the suspect pile is never touched."""
    null = np.empty((B, Xk.shape[1]))
    for b in range(B):
        q = rng.permutation(len(Xk))
        h = len(q) // 2
        null[b] = _sep_cols(Xk[q[:h]], Xk[q[h:]])
    return np.quantile(null, 0.95, axis=0)


def _obs(Xk, Xs):
    return _sep_cols(Xk, Xs)


def informative(Xk, Xs, mk, ms, rng, alpha=0.05, B=40):
    """obs > q95 of the within-known null. Looks at the suspect pile."""
    return _obs(Xk, Xs) > _floor(Xk, rng, B)


def stability_floor(Xk, Xs, mk, ms, rng, tau=1.25, B=40):
    """Keep only the QUIET signals. Never looks at the suspect pile."""
    f = _floor(Xk, rng, B)
    return f <= tau * np.median(f)


def stability_floor_15(Xk, Xs, mk, ms, rng):
    return stability_floor(Xk, Xs, mk, ms, rng, tau=1.5)


def stability_floor_10(Xk, Xs, mk, ms, rng):
    return stability_floor(Xk, Xs, mk, ms, rng, tau=1.0)


def informative_x_floor(Xk, Xs, mk, ms, rng, tau=1.25, B=40):
    """Both: quiet AND separating. The floor is computed once and reused."""
    f = _floor(Xk, rng, B)
    return (_obs(Xk, Xs) > f) & (f <= tau * np.median(f))


def informative_over_floor_top(Xk, Xs, mk, ms, rng, k=40, B=40):
    """Rank by the ratio obs/floor and keep the top k -- a signal-to-noise
    ordering rather than a hypothesis test, so the number kept is fixed and does
    not silently collapse to nothing on a small known pile."""
    f = _floor(Xk, rng, B)
    r = _obs(Xk, Xs) / np.where(f > 1e-9, f, 1e-9)
    m = np.zeros(len(r), bool)
    m[np.argsort(-r)[:min(k, len(r))]] = True
    return m


# ---------------------------------------------------------------- stringency
# The default screen tests each signal at alpha=0.05 against its own null with NO
# multiplicity control: over 678 signals that is ~34 expected false selections,
# and it keeps 503/678 on sap_hybrid -- barely a filter. These tighten it.

def _pvals(Xk, Xs, rng, B=60):
    """Per-signal p for the observed known-vs-suspect separation against the
    within-known null.

    A raw permutation p bottoms out at 1/(B+1) -- with B=60 that is 0.016, so
    alpha=0.01, alpha=0.001 and Bonferroni (0.05/678 = 7e-5) are all UNREACHABLE
    and every signal fails them regardless of the data. So the null's mean and sd
    are estimated from the B draws and the tail is taken from a Gaussian, which
    extrapolates past 1/(B+1) instead of saturating. The approximation only has
    to be decent in the tail ordering, since these thresholds are used to rank
    and cut rather than to make a calibrated claim.
    """
    obs = _obs(Xk, Xs)
    null = np.empty((B, Xk.shape[1]))
    for b in range(B):
        q = rng.permutation(len(Xk))
        h = len(q) // 2
        null[b] = _sep_cols(Xk[q[:h]], Xk[q[h:]])
    mu, sd = null.mean(0), null.std(0, ddof=1)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1e-12)
    z = (obs - mu) / sd
    return ndtr(-z), obs


def informative_alpha(alpha, B=60):
    def f(Xk, Xs, mk, ms, rng):
        p, _ = _pvals(Xk, Xs, rng, B)
        return p <= alpha
    return f


def informative_bonferroni(Xk, Xs, mk, ms, rng, alpha=0.05, B=60):
    p, _ = _pvals(Xk, Xs, rng, B)
    return p <= alpha / len(p)


def informative_bh(Xk, Xs, mk, ms, rng, alpha=0.05, B=60):
    """Benjamini-Hochberg: control the false DISCOVERY rate rather than the
    family-wise error. Less brutal than Bonferroni, still multiplicity-aware."""
    p, _ = _pvals(Xk, Xs, rng, B)
    o = np.argsort(p)
    thr = alpha * (np.arange(1, len(p) + 1) / len(p))
    passed = p[o] <= thr
    k = np.flatnonzero(passed).max() + 1 if passed.any() else 0
    m = np.zeros(len(p), bool)
    m[o[:k]] = True
    return m


def snr_top(k, B=60):
    """Rank by observed separation / own noise floor, keep the top k. Fixes the
    number kept, so it cannot silently collapse to nothing on a small pile."""
    def f(Xk, Xs, mk, ms, rng):
        fl = _floor(Xk, rng, B)
        r = _obs(Xk, Xs) / np.where(fl > 1e-9, fl, 1e-9)
        m = np.zeros(len(r), bool)
        m[np.argsort(-r)[:min(k, len(r))]] = True
        return m
    return f
