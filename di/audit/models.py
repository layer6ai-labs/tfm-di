"""Meta-classifiers, the TreeSHAP ranking and the elbow rule of the internal audit."""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.base import ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

XGB_PARAMS = dict(max_depth=3, n_estimators=150, learning_rate=0.06, subsample=0.85,
                  colsample_bytree=0.7, reg_lambda=1.0, min_child_weight=2,
                  eval_metric="logloss", n_jobs=1, verbosity=0, tree_method="hist",
                  random_state=0)
KINDS = ("lr", "xgb", "tabpfn")
TABPFN_CKPT = (Path(os.environ.get("TABPFN_MODEL_CACHE_DIR", Path.home() / ".cache" / "tabpfn"))
               / "tabpfn-v2.5-classifier-v2.5_default-2.ckpt")


def make_tabpfn(device: str = "cuda", n_estimators: int = 4) -> ClassifierMixin:
    """TabPFN-2.5 classifier from the local synthetic checkpoint, weights loaded once."""
    from tabpfn import TabPFNClassifier

    from di.datasets.tabpfn25_wrapper import _get_cached_specs
    if not TABPFN_CKPT.exists():
        raise FileNotFoundError(f"TabPFN-2.5 classifier checkpoint missing: {TABPFN_CKPT}")
    specs = _get_cached_specs(TabPFNClassifier, str(TABPFN_CKPT), device, "classifier")
    return TabPFNClassifier(random_state=0, device=device, ignore_pretraining_limits=True,
                            model_path=specs, n_estimators=n_estimators)


def fit_predict(kind: str, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                tabpfn: ClassifierMixin | None = None) -> np.ndarray:
    """Fit ``kind`` on (Xtr, ytr) and return P(member) for Xte. NaN is passed through:
    the LR pipeline imputes fold-local medians, XGBoost and TabPFN handle it natively."""
    if kind == "lr":
        m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
    elif kind == "xgb":
        m = xgb.XGBClassifier(**XGB_PARAMS)
    elif kind == "tabpfn":
        if tabpfn is None:
            raise ValueError("kind 'tabpfn' needs a model from make_tabpfn")
        m = tabpfn
    else:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def treeshap_importance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean |TreeSHAP| per column of an XGBoost fit on (X, y), in log-odds units."""
    c = xgb.XGBClassifier(**XGB_PARAMS)
    c.fit(X, y)
    return np.abs(c.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]).mean(0)


def rank_treeshap(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Column indices sorted by decreasing mean |TreeSHAP| of an XGBoost fit on (X, y)."""
    return np.argsort(-treeshap_importance(X, y))


def elbow(ks: Sequence[int], aucs: Sequence[float]) -> int:
    """The K on the AUC-vs-log2(K) curve farthest above the chord between its endpoints;
    the argmax when the curve has fewer than three points or is flat."""
    x = np.log2(np.asarray(ks, float))
    y = np.asarray(aucs, float)
    if len(x) < 3 or y.max() == y.min():
        return ks[int(np.argmax(y))]
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    return ks[int(np.argmax(yn - (xn * (yn[-1] - yn[0]) + yn[0])))]
