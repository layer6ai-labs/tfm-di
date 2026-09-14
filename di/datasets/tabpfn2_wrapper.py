"""Thin TabPFN-2 adapter exposing the TabDPT-style API expected by the
grid runner. Also supports RealTabPFN for classification.

TabPFN-2 is the older v2 model from Prior-Labs. Checkpoints should be
downloaded from https://huggingface.co/Prior-Labs/TabPFN-v2-clf
and from https://huggingface.co/Prior-Labs/TabPFN-v2-reg"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import numpy as np


# Default location for TabPFN-2 and RealTabPFN checkpoints. Users can download from:
# https://huggingface.co/Prior-Labs/TabPFN-v2-clf
_DEFAULT_V2_CACHE = Path.home() / ".cache" / "tabpfn"
_SYNTHETIC_V2_CLASSIFIER_CKPT = _DEFAULT_V2_CACHE / "tabpfn-v2-classifier-v2_default.ckpt"
_REAL_V2_CLASSIFIER_CKPT = _DEFAULT_V2_CACHE / "tabpfn-v2-classifier-finetuned-zk73skhh.cpkt"
# https://huggingface.co/Prior-Labs/TabPFN-v2-reg
_SYNTHETIC_V2_REGRESSOR_CKPT = _DEFAULT_V2_CACHE / "tabpfn-v2-regressor.ckpt"
# RealTabPFN does not support regression tasks, so no real regressor checkpoint.

# Module-level cache: load model weights from disk once, reuse across all
# TabPFN2* wrapper instances.  Keyed by (ckpt_path, device).
_model_specs_cache: dict[tuple[str, str], object] = {}
_cache_lock = Lock()


def _resolve_v2_path(explicit: str | None, default: Path) -> str:
    """Return an explicit override, else the cached v2 ckpt, else 'auto'."""
    if explicit:
        return explicit
    if default.exists():
        return str(default)
    # Last resort — let tabpfn try to download. Will fail on compute nodes
    # without internet access.
    return "auto"


def _get_cached_specs(backend_cls, model_path: str, device: str):
    """Load model weights once and return a ClassifierModelSpecs / equivalent.

    TabPFN's ``__init__`` + ``fit`` calls ``_initialize_model_variables`` which
    runs ``load_model_criterion_config`` every time — reading the checkpoint
    from disk, rebuilding the model graph, and moving it to GPU.  With 34-36
    fits per dataset that dominates wall-clock time.

    By pre-loading the model into a ``ClassifierModelSpecs`` and passing it as
    ``model_path``, TabPFN hits the fast path in ``initialize_tabpfn_model``
    and skips the disk I/O entirely.
    """
    key = (model_path, device)
    with _cache_lock:
        if key in _model_specs_cache:
            return _model_specs_cache[key]

    # First call: load from disk (outside lock — may take a moment).
    from tabpfn.base import ClassifierModelSpecs, load_model_criterion_config
    from tabpfn.model_loading import resolve_model_version

    version = resolve_model_version(model_path)
    models, _, arch_configs, inference_config = load_model_criterion_config(
        model_path=model_path,
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="classifier",
        version=version.value,
        download_if_not_exists=False,
    )
    specs = ClassifierModelSpecs(
        model=models[0],
        architecture_config=arch_configs[0],
        inference_config=inference_config,
    )

    with _cache_lock:
        _model_specs_cache[key] = specs
    return specs


class _BaseTabPFN2Adapter:
    """Common init / fit shared by classifier and regressor wrappers.

    TabPFN v2 does not natively support the ``temperature``,
    ``context_size``, or ``seed`` predict-time knobs that TabDPT exposes.

    * **temperature** — applied post-hoc: divide log-probs by *T*, re-softmax.
    * **context_size** — ignored here; already handled by
      ``ManipulationSpec.context_size`` → ``subsample_rows()`` in the grid's
      manipulation pipeline, which subsamples *before* ``model.fit()``.
    * **seed** — ignored; TabDPT uses this internally for column permutation,
      but TabPFN has no equivalent predict-time feature shuffle.

    Model weights are loaded from disk **once** per (checkpoint, device) pair
    and cached at module level.  Subsequent ``__init__`` calls reuse the
    already-loaded weights via TabPFN's ``ClassifierModelSpecs`` fast path,
    avoiding repeated disk I/O and model graph construction.
    """

    _backend_cls: type | None = None  # set by subclasses
    _default_ckpt: Path | None = None  # set by subclasses

    def __init__(
        self,
        device: str | None = None,
        verbose: bool = False,
        model_weight_path: str | None = None,
        normalizer: str = "standard",
        **kwargs,
    ) -> None:
        if self._backend_cls is None:
            raise RuntimeError("_backend_cls must be set on the subclass")
        backend_device = "cuda" if device == "cuda" else "cpu"
        model_path = _resolve_v2_path(
            model_weight_path, self._default_ckpt or Path("/dev/null")
        )
        # Use cached model specs to avoid reloading weights from disk.
        specs = _get_cached_specs(self._backend_cls, model_path, backend_device)
        self.model = self._backend_cls(
            device=backend_device,
            ignore_pretraining_limits=True,
            model_path=specs,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X, **kwargs):
        return self.model.predict(X)


class TabPFN2Classifier(_BaseTabPFN2Adapter):
    """Classifier wrapper around ``tabpfn.TabPFNClassifier`` for v2."""

    _default_ckpt = _SYNTHETIC_V2_CLASSIFIER_CKPT

    def __init__(self, *args, **kwargs):
        from tabpfn import TabPFNClassifier as _TabPFNCls

        type(self)._backend_cls = _TabPFNCls
        super().__init__(*args, **kwargs)

    def predict_proba(
        self,
        X,
        temperature: float | None = None,
        context_size: int | None = None,
        return_logits: bool = False,
        seed: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        probs = self.model.predict_proba(X)

        # Temperature scaling: convert to log-probs, divide by T, re-softmax.
        if temperature is not None and temperature != 0.8:
            log_probs = np.log(np.clip(probs, 1e-12, 1.0))
            scaled = log_probs / temperature
            scaled -= scaled.max(axis=1, keepdims=True)  # numerical stability
            exp_scaled = np.exp(scaled)
            probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

        if return_logits:
            return np.log(np.clip(probs, 1e-12, 1.0))
        return probs


class RealTabPFNClassifier(TabPFN2Classifier):
    """Sets alternate model checkpoint for RealTabPFN."""

    _default_ckpt = _REAL_V2_CLASSIFIER_CKPT


class TabPFN2Regressor(_BaseTabPFN2Adapter):
    """Regressor wrapper around ``tabpfn.TabPFNRegressor`` for v2."""

    _default_ckpt = _SYNTHETIC_V2_REGRESSOR_CKPT

    def __init__(self, *args, **kwargs):
        from tabpfn import TabPFNRegressor as _TabPFNReg

        type(self)._backend_cls = _TabPFNReg
        super().__init__(*args, **kwargs)

    def predict_proba(
        self,
        X,
        temperature: float | None = None,
        context_size: int | None = None,
        return_logits: bool = False,
        seed: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        return self.model.predict(X)


class RealTabPFNRegressor(TabPFN2Regressor):
    """RealTabPFN not available for regression tasks."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("RealTabPFN does not support regression tasks.")
