"""Label-free scorers for the external audit.

All of these take (Xk, Xs) -- the auditor's KNOWN non-members and the SUSPECT
pile -- and return (known scores, suspect scores), higher = more member-like.
None of them sees a membership label. What they differ in is the assumption
they smuggle in.

  iforest      IsolationForest fitted on Xk; suspects ranked by anomaly.
               ASSUMES anomalous = member-like. That assumption is unverifiable
               and it inverts whenever members form a TIGHTER cloud than the
               known pile, which is exactly what produces below-chance blinds.
               Also ignores correlation between signals entirely.

  mahalanobis  One-class Gaussian: mean and shrunk covariance from Xk, score =
               Mahalanobis distance. Same "far = member" assumption as iforest
               but it accounts for correlation, and it is closed-form -- no
               trees, no randomness, no hyperparameters beyond the shrinkage.

  knn          Mean distance to the k nearest known non-members. The simplest
               possible density score; same directional assumption again.

The PU scorers (pu_z / pu_lr / pu_ridge / pu_elkan_noto) that used to live here
were removed: the audit reports iForest, and a discriminative known-vs-suspect
classifier is a different threat model. `external_family_audit.iforest_oriented_scores`
keeps the one useful piece -- taking the SIGN from the unlabelled suspect pile's
shift instead of assuming "anomalous = member-like" -- which is direction
estimation, not classification.

"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def iforest(Xk, Xs, seed=0, **kw):
    p = dict(n_estimators=200, contamination="auto", random_state=seed)
    p.update(kw)
    f = IsolationForest(**p).fit(Xk)
    return -f.score_samples(Xk), -f.score_samples(Xs)


def mahalanobis(Xk, Xs, seed=0, shrink=0.1):
    mu = Xk.mean(0)
    S = np.cov(Xk, rowvar=False)
    S = np.atleast_2d(S)
    S = (1 - shrink) * S + shrink * np.eye(len(S)) * np.trace(S) / max(len(S), 1)
    P = np.linalg.pinv(S)
    d = lambda X: np.einsum("ij,jk,ik->i", X - mu, P, X - mu)
    return d(Xk), d(Xs)


def knn(Xk, Xs, seed=0, k=10):
    k = int(min(k, max(1, len(Xk) - 1)))
    nn = NearestNeighbors(n_neighbors=k).fit(Xk)
    sk = nn.kneighbors(Xk, n_neighbors=min(k + 1, len(Xk)))[0][:, 1:].mean(1)
    return sk, nn.kneighbors(Xs)[0].mean(1)


SCORERS = {"iforest": iforest, "mahalanobis": mahalanobis, "knn": knn}
