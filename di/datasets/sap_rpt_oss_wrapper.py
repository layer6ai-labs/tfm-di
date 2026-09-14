"""
Wrapper classes for SAP-RPT-OSS model to make it compatible with DI framework.

The SAP-RPT-OSS model has a slightly different API than TabDPT, so we wrap it
to ensure compatibility with the DI interface.
"""

import os
from threading import Lock
from typing import Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder

try:
    from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor
except ImportError:
    raise ImportError(
        "sap-rpt-oss is not installed. Install with: pip install .[sap-rpt-oss]"
    )


# ---------------------------------------------------------------------------
# Weight-load cache (safe) — skip re-reading the checkpoint, share nothing else
# ---------------------------------------------------------------------------
# RPT.load_weights() does `torch.load(ckpt, map_location=device)`: a 62 MB disk
# read + deserialize + host->device transfer. The DI grid constructs a new
# model for EVERY manipulation (~70 fits per dataset, di/grid/predictions.py),
# so that reload happened ~70x per dataset and dominated wall-clock.
#
# We cache ONLY the loaded state_dict, keyed by (checkpoint, device). Every fit
# still gets a completely fresh estimator: new RPT graph, new tokenizer, new
# fit state — so there is no shared mutable state and results are unchanged.
# (Caching the whole estimator instead makes signals history-dependent; see the
# note on _get_sap_model below.)
#
# `copy_last_layer_weights_to_all` mutates the dict it is given, so each call
# gets a shallow copy of the cached mapping; the tensors themselves are only
# read (load_state_dict copies data into the fresh module's parameters).
_state_dict_cache: dict[tuple, dict] = {}
_state_dict_lock = Lock()
_weight_cache_installed = False


def install_weight_cache() -> None:
    """Patch RPT.load_weights to read the checkpoint from disk at most once."""
    global _weight_cache_installed
    if _weight_cache_installed or os.environ.get("SAP_DISABLE_WEIGHT_CACHE") == "1":
        return
    import torch
    from sap_rpt_oss.model.torch_model import RPT

    def _cached_load_weights(self, checkpoint_path, device, is_copy_last_layer=True):
        key = (str(checkpoint_path), str(device))
        with _state_dict_lock:
            sd = _state_dict_cache.get(key)
        if sd is None:
            sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
            with _state_dict_lock:
                sd = _state_dict_cache.setdefault(key, sd)
        sd = dict(sd)  # never let the transform mutate the cached mapping
        if is_copy_last_layer:
            sd = self.copy_last_layer_weights_to_all(sd)
        self.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})

    RPT.load_weights = _cached_load_weights
    _weight_cache_installed = True


install_weight_cache()


# ---------------------------------------------------------------------------
# Full-estimator cache (UNSAFE — opt-in only)
# ---------------------------------------------------------------------------
# Constructing a SAP estimator is expensive: sap_rpt_oss/rpt.py:65-104 resolves
# the checkpoint via hf_hub_download, builds the RPT graph, casts it to
# bf16/fp16, reads the 62 MB weight file off disk and moves the whole model onto
# the GPU. The DI grid builds a brand-new model for EVERY manipulation
# (di/grid/predictions.py — ~70 fits per dataset), so that setup, not
# inference, dominated wall-clock: measured ~164 s/dataset with T4 and L4 within
# 1% of each other, i.e. the run was not GPU-bound at all.
#
# Caching the loaded estimator per (kind, max_context_size, bagging) collapses
# those ~70 loads into one. This mirrors the `_model_specs_cache` the TabPFN-2.5
# wrapper already uses for exactly the same reason.
#
# Safety: the RPT network is eval-mode and never mutated (SAP is an in-context
# learner — no gradient updates), and `fit()` overwrites all per-fit state
# (X_, y_, bagging_config, task_specific_fit) on every call. The grid uses a
# model strictly sequentially (construct -> fit -> predict -> discard), so a
# shared instance cannot interleave. Set SAP_DISABLE_MODEL_CACHE=1 to opt out.
_sap_model_cache: dict[tuple, object] = {}
_sap_cache_lock = Lock()


def _get_sap_model(kind: str, max_context_size: int, bagging):
    """Return a SAP estimator; shares one instance only if explicitly enabled.

    !! OPT-IN (SAP_ENABLE_MODEL_CACHE=1) — NOT SAFE BY DEFAULT !!
    Sharing one estimator across fits is a ~6x speedup (measured 163.8 ->
    27.6 s/dataset on T4, i.e. ~85% of runtime is construction), but it CHANGES
    RESULTS. Measured against a fresh-instance baseline on 6 identical datasets:
    each regime is perfectly deterministic on its own (0/3856 values differ
    between two cached runs), yet they disagree with each other in a
    history-dependent way — 29/681 and 41/681 signals changed on two
    classification tables but 462/681 and 470/681 on two others, while
    regression was untouched (0/566). So the shared estimator carries state
    between fits (tokenizer / label mappings), making a dataset's signals depend
    on what the container processed before it — i.e. on shard composition and
    preemption points. Unusable for a scientific run until the per-fit state is
    properly isolated (cache the loaded weights, not the fitted estimator).
    """
    import copy

    cls = SAP_RPT_OSS_Classifier if kind == "classifier" else SAP_RPT_OSS_Regressor

    # --- SAFE variant: deepcopy a cached prototype -----------------------
    # Sharing live networks (below) corrupts signals because fit() mutates
    # shared module state. A DEEP copy shares nothing: the result is an
    # independent estimator whose weights came from the same checkpoint, i.e.
    # semantically identical to constructing one from scratch -- but it skips
    # the random init, the disk read and the 62 MB host->device transfer,
    # replacing them with one GPU->GPU clone.
    if os.environ.get("SAP_DEEPCOPY_MODEL_CACHE") == "1":
        import torch
        key = ("deep", kind, max_context_size, bagging)
        with _sap_cache_lock:
            proto = _sap_model_cache.get(key)
        if proto is None:
            rng = torch.random.get_rng_state()
            built = cls(max_context_size=max_context_size, bagging=bagging)
            with _sap_cache_lock:
                proto = _sap_model_cache.setdefault(key, built)
                # Replay the RNG consumption a fresh construction would cause,
                # so downstream sampling sees the same stream either way.
                _sap_model_cache[("rng", kind)] = (rng, torch.random.get_rng_state())
        est = copy.deepcopy(proto)
        for attr in ("X_", "y_", "classes_"):
            est.__dict__.pop(attr, None)
        if hasattr(est, "tokenizer") and hasattr(est.tokenizer, "cache"):
            est.tokenizer.cache = type(est.tokenizer.cache)(
                max_size=int(os.getenv("LRU_CACHE_SIZE", 1_000_000)))
        return est

    if os.environ.get("SAP_ENABLE_MODEL_CACHE") != "1":
        return cls(max_context_size=max_context_size, bagging=bagging)

    # --- prototype: pay full construction exactly once -------------------
    key = (kind, max_context_size, bagging)
    with _sap_cache_lock:
        proto = _sap_model_cache.get(key)
    if proto is None:
        built = cls(max_context_size=max_context_size, bagging=bagging)
        with _sap_cache_lock:
            proto = _sap_model_cache.setdefault(key, built)

    # --- per fit: share the two NETWORKS, reset everything mutable -------
    # Construction cost is dominated by building/moving two models: the RPT
    # graph and the Tokenizer's SentenceEmbedder (MiniLM). Both are eval-mode
    # and stateless w.r.t. fitting, so they are safe to share. Everything that
    # accumulates state gets rebuilt: the tokenizer's LRU_Cache (memoises
    # per-table column/value embeddings — sharing it is what made signals
    # history-dependent) and the sklearn fitted attributes (X_, y_, classes_).
    est = copy.copy(proto)                 # new object, shares .model (RPT)
    tok = copy.copy(proto.tokenizer)       # shares .sentence_embedder
    # The tokenizer's LRU_Cache memoises embeddings of column names / string
    # values. Sharing it across the ~70 fits of ONE table is pure win (same
    # table => same strings => same embeddings) and is where essentially all of
    # the speedup comes from. Sharing it ACROSS tables is what made signals
    # history-dependent, so callers must call reset_tokenizer_caches() at every
    # dataset boundary.
    tok.cache = _get_shared_tok_cache(kind, proto)
    est.tokenizer = tok
    for attr in ("X_", "y_", "classes_"):
        est.__dict__.pop(attr, None)
    return est


_shared_tok_caches: dict = {}


def _get_shared_tok_cache(kind: str, proto):
    cache = _shared_tok_caches.get(kind)
    if cache is None:
        cache = type(proto.tokenizer.cache)(
            max_size=int(os.getenv("LRU_CACHE_SIZE", 1_000_000))
        )
        _shared_tok_caches[kind] = cache
    return cache


def reset_tokenizer_caches() -> None:
    """Drop memoised embeddings. MUST be called between datasets."""
    _shared_tok_caches.clear()


class SAP_RPT_OSS_ClassifierWrapper:
    """
    Wrapper around SAP_RPT_OSS_Classifier to match TabDPT interface.
    
    The wrapper ensures compatibility with the DI framework
    by providing the same interface as TabDPT.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        verbose: bool = False,
        model_weight_path: Optional[str] = None,
        max_context_size: int = 8192,
        bagging: int = 8,
        **kwargs
    ):
        """
        Initialize the classifier wrapper.

        Parameters:
        -----------
        device : str, optional
            Device to use ('cuda' or 'cpu'). Currently not used by sap-rpt-oss.
        verbose : bool, default=False
            Whether to print debug information.
        model_weight_path : str, optional
            Path to model weights (not used for sap-rpt-oss).
        max_context_size : int, default=8192
            Maximum context size for the model.
        bagging : int, default=8
            Number of bagging iterations.
        """
        self.device = device
        self.verbose = verbose
        self.model_weight_path = model_weight_path
        self.max_context_size = max_context_size
        self.bagging = bagging
        
        # Initialize the underlying SAP-RPT-OSS classifier
        self.model = _get_sap_model("classifier", max_context_size, bagging)
        
        # Store label information
        self.label_encoder_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SAP_RPT_OSS_ClassifierWrapper":
        """
        Fit the classifier on training data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels.

        Returns:
        --------
        self
        """
        # Encode labels if necessary
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        # Store training data for context_size / seed support
        self.X_train_ = np.asarray(X)
        self.y_train_ = y_encoded.copy()

        # Fit the underlying model
        self.model.fit(X, y_encoded)
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.

        Returns:
        --------
        np.ndarray of shape (n_samples,)
            Predicted class labels (in original label space).
        """
        predictions = self.model.predict(X)
        
        # If we encoded labels, decode the predictions
        if self.label_encoder_ is not None:
            predictions = self.label_encoder_.inverse_transform(predictions)
        
        return predictions

    def predict_proba(
        self,
        X: np.ndarray,
        temperature: float = 1.0,
        context_size: Optional[int] = None,
        seed: Optional[int] = None,
        return_logits: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict class probabilities or logits.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.
        temperature : float, default=1.0
            Temperature for softmax scaling. Higher values make predictions
            more uniform.
        context_size : int, optional
            Number of training examples to use as context. If None, uses all
            available training data.
        seed : int, optional
            Random seed for feature column permutation.
        return_logits : bool, default=False
            If True, return raw logits instead of probabilities.

        Returns:
        --------
        np.ndarray of shape (n_samples, n_classes)
            Predicted class probabilities or logits.
        """
        X = np.asarray(X)

        # Handle context_size / seed by refitting with modified context
        need_refit = False
        ctx_X = self.X_train_
        ctx_y = self.y_train_

        if context_size is not None and context_size < len(self.X_train_):
            ctx_X = ctx_X[:context_size]
            ctx_y = ctx_y[:context_size]
            need_refit = True

        if seed is not None:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(X.shape[1])
            ctx_X = ctx_X[:, perm]
            X = X[:, perm]
            need_refit = True

        if need_refit:
            self.model.fit(ctx_X, ctx_y)

        probs = self.model.predict_proba(X)

        # Restore original fit state if we refitted
        if need_refit:
            self.model.fit(self.X_train_, self.y_train_)

        # Ensure 2D
        probs = np.atleast_1d(probs)
        if probs.ndim == 1:
            if len(probs) == len(X):
                probs = np.column_stack([1 - probs, probs])
            else:
                probs = probs.reshape(1, -1)
        elif probs.ndim == 2 and probs.shape[1] == 1:
            probs_positive = probs.ravel()
            probs = np.column_stack([1 - probs_positive, probs_positive])

        # Convert to logits
        eps = 1e-15
        log_probs = np.log(np.clip(probs, eps, 1.0))

        if return_logits:
            return log_probs.astype(np.float32)

        # Apply temperature scaling
        if temperature != 1.0 and probs.shape[1] > 1:
            scaled = log_probs / temperature
            max_scaled = np.max(scaled, axis=1, keepdims=True)
            exp_scaled = np.exp(scaled - max_scaled)
            probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

        return probs

    def ensemble_predict_proba(
        self,
        X: np.ndarray,
        n_ensembles: int = 1,
        temperature: float = 1.0,
        context_size: Optional[int] = None,
        permute_classes: bool = False,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Predict class probabilities using ensemble averaging.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.
        n_ensembles : int, default=1
            Number of ensemble iterations (multiple forward passes).
        temperature : float, default=1.0
            Temperature for softmax scaling.
        context_size : int, optional
            Context size for predictions.
        permute_classes : bool, default=False
            Whether to permute classes (ignored for sap-rpt-oss).
        seed : int, optional
            Random seed for reproducibility.

        Returns:
        --------
        np.ndarray of shape (n_samples, n_classes)
            Averaged ensemble probabilities.
        """
        probabilities_list = []
        
        for i in range(n_ensembles):
            current_seed = seed + i if seed is not None else None
            probs = self.predict_proba(
                X,
                temperature=temperature,
                context_size=context_size,
                seed=current_seed,
            )
            probabilities_list.append(probs)
        
        return np.mean(probabilities_list, axis=0)

    @staticmethod
    def download_weights():
        """
        Download model weights if needed.

        SAP-RPT-OSS doesn't require pre-downloaded weights, so this is a no-op.

        Returns:
        --------
        None
        """
        return None


class SAP_RPT_OSS_RegressorWrapper:
    """
    Wrapper around SAP_RPT_OSS_Regressor to match TabDPT interface.
    
    The wrapper ensures compatibility with the DI framework
    by providing the same interface as TabDPT.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        verbose: bool = False,
        model_weight_path: Optional[str] = None,
        max_context_size: int = 8192,
        bagging: int = 8,
        **kwargs
    ):
        """
        Initialize the regressor wrapper.

        Parameters:
        -----------
        device : str, optional
            Device to use ('cuda' or 'cpu'). Currently not used by sap-rpt-oss.
        verbose : bool, default=False
            Whether to print debug information.
        model_weight_path : str, optional
            Path to model weights (not used for sap-rpt-oss).
        max_context_size : int, default=8192
            Maximum context size for the model.
        bagging : int, default=8
            Number of bagging iterations.
        """
        self.device = device
        self.verbose = verbose
        self.model_weight_path = model_weight_path
        self.max_context_size = max_context_size
        self.bagging = bagging
        
        # Initialize the underlying SAP-RPT-OSS regressor
        self.model = _get_sap_model("regressor", max_context_size, bagging)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SAP_RPT_OSS_RegressorWrapper":
        """
        Fit the regressor on training data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.

        Returns:
        --------
        self
        """
        self.X_train_ = np.asarray(X)
        self.y_train_ = np.asarray(y)
        self.model.fit(X, y)
        return self

    def predict(
        self,
        X: np.ndarray,
        n_ensembles: int = 1,
        context_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Predict continuous values.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.
        n_ensembles : int, default=1
            Number of ensemble iterations (multiple forward passes).
        context_size : int, optional
            Number of training examples to use as context.
        seed : int, optional
            Random seed for feature column permutation.

        Returns:
        --------
        np.ndarray of shape (n_samples,)
            Predicted values.
        """
        X = np.asarray(X)

        need_refit = False
        ctx_X = self.X_train_
        ctx_y = self.y_train_

        if context_size is not None and context_size < len(self.X_train_):
            ctx_X = ctx_X[:context_size]
            ctx_y = ctx_y[:context_size]
            need_refit = True

        if seed is not None:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(X.shape[1])
            ctx_X = ctx_X[:, perm]
            X = X[:, perm]
            need_refit = True

        if need_refit:
            self.model.fit(ctx_X, ctx_y)

        predictions_list = []
        for _ in range(n_ensembles):
            preds = self.model.predict(X)
            predictions_list.append(preds)

        if need_refit:
            self.model.fit(self.X_train_, self.y_train_)

        return np.mean(predictions_list, axis=0)

    @staticmethod
    def download_weights():
        """
        Download model weights if needed.

        SAP-RPT-OSS doesn't require pre-downloaded weights, so this is a no-op.

        Returns:
        --------
        None
        """
        return None