"""Thin TabPFN-2.5 adapter exposing the TabDPT-style API expected by the
grid runner. Mirrors the inline classes in ``slurm/grid_worker.py:280-301``
so they can be referenced from a Hydra model config."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import numpy as np


# Default location tabpfn's own downloader drops v2.5 checkpoints. Pinning
# this path side-steps tabpfn>=7.1's v2.6 auto-download, which now requires
# PriorLabs license acceptance via an interactive browser flow — fatal on
# JURECA compute nodes that have no internet and no browser. The v2.5 file
# is the one already on disk from earlier runs and is the weight set the
# rest of this repo was validated against.
_DEFAULT_V25_CACHE = Path.home() / ".cache" / "tabpfn"
_SYNTHETIC_V25_CLASSIFIER_CKPT = _DEFAULT_V25_CACHE / "tabpfn-v2.5-classifier-v2.5_default-2.ckpt"
_REAL_V25_CLASSIFIER_CKPT = _DEFAULT_V25_CACHE / "tabpfn-v2.5-classifier-v2.5_default.ckpt"
_REAL_ALT_V25_CLASSIFIER_CKPT = _DEFAULT_V25_CACHE / "tabpfn-v2.5-classifier-v2.5_real.ckpt"
_SYNTHETIC_V25_REGRESSOR_CKPT = _DEFAULT_V25_CACHE / "tabpfn-v2.5-regressor-v2.5_default.ckpt"
_REAL_V25_REGRESSOR_CKPT = _DEFAULT_V25_CACHE / "tabpfn-v2.5-regressor-v2.5_real.ckpt"
_REAL_ALT_V25_REGRESSOR_CKPT = _DEFAULT_V25_CACHE / "tabpfn-v2.5-regressor-v2.5_real-variant.ckpt"

# Module-level cache: load model weights from disk once, reuse across all
# TabPFN25* wrapper instances.  Keyed by (ckpt_path, device).
_model_specs_cache: dict[tuple[str, str], object] = {}
_cache_lock = Lock()


def _resolve_v25_path(explicit: str | None, default: Path) -> str:
    """Return an explicit override, else the cached v2.5 ckpt, else 'auto'."""
    if explicit:
        return explicit
    if default.exists():
        return str(default)
    # Last resort — let tabpfn try to download. Will fail on compute nodes
    # without a license, but the error message will be informative.
    return "auto"


def _get_cached_specs(backend_cls, model_path: str, device: str, which: str = "classifier"):
    """Load model weights once and return a Classifier/RegressorModelSpecs.

    TabPFN's ``__init__`` + ``fit`` calls ``_initialize_model_variables`` which
    runs ``load_model_criterion_config`` every time — reading the checkpoint
    from disk, rebuilding the model graph, and moving it to GPU.  With 34-36
    fits per dataset that dominates wall-clock time.

    By pre-loading the model into a ModelSpecs object and passing it as
    ``model_path``, TabPFN hits the fast path in ``initialize_tabpfn_model``
    and skips the disk I/O entirely.

    ``which`` MUST match the backend: a regressor needs a ``RegressorModelSpecs``
    carrying the ``norm_criterion`` (FullSupportBarDistribution). Previously this
    always built a ``ClassifierModelSpecs``, so every ``TabPFNRegressor`` fit
    raised ``TypeError`` (ModelSpecs/which mismatch) — the crash was swallowed by
    the prediction engine and every regression BLIND signal became all-NaN.
    """
    key = (model_path, device, which)
    with _cache_lock:
        if key in _model_specs_cache:
            return _model_specs_cache[key]

    # First call: load from disk (outside lock — may take a moment).
    from tabpfn.base import (
        ClassifierModelSpecs,
        RegressorModelSpecs,
        load_model_criterion_config,
    )
    from tabpfn.model_loading import resolve_model_version

    version = resolve_model_version(model_path)
    models, criterion, arch_configs, inference_config = load_model_criterion_config(
        model_path=model_path,
        check_bar_distribution_criterion=(which == "regressor"),
        cache_trainset_representation=False,
        which=which,
        version=version.value,
        download_if_not_exists=False,
    )
    if which == "regressor":
        specs = RegressorModelSpecs(
            model=models[0],
            architecture_config=arch_configs[0],
            inference_config=inference_config,
            norm_criterion=criterion,
        )
    else:
        specs = ClassifierModelSpecs(
            model=models[0],
            architecture_config=arch_configs[0],
            inference_config=inference_config,
        )

    with _cache_lock:
        _model_specs_cache[key] = specs
    return specs


class _BaseTabPFNAdapter:
    """Common init / fit shared by classifier and regressor wrappers.

    TabPFN v2.5 does not natively support the ``temperature``,
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
    _which: str = "classifier"  # "classifier" | "regressor" — set by subclasses

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
        model_path = _resolve_v25_path(
            model_weight_path, self._default_ckpt or Path("/dev/null")
        )
        # Use cached model specs to avoid reloading weights from disk.
        specs = _get_cached_specs(
            self._backend_cls, model_path, backend_device, self._which
        )
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


class TabPFN25Classifier(_BaseTabPFNAdapter):
    """Classifier wrapper around ``tabpfn.TabPFNClassifier``."""

    _default_ckpt = _SYNTHETIC_V25_CLASSIFIER_CKPT

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


class RealTabPFN25Classifier(TabPFN25Classifier):
    """Sets model checkpoint for best RealTabPFN-2.5 classifier."""

    _default_ckpt = _REAL_V25_CLASSIFIER_CKPT


class RealTabPFN25AltClassifier(TabPFN25Classifier):
    """Sets model checkpoint for alternate RealTabPFN-2.5 classifier."""

    _default_ckpt = _REAL_ALT_V25_CLASSIFIER_CKPT


class TabPFN25Regressor(_BaseTabPFNAdapter):
    """Regressor wrapper around ``tabpfn.TabPFNRegressor``."""

    _default_ckpt = _SYNTHETIC_V25_REGRESSOR_CKPT
    _which = "regressor"

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


class RealTabPFN25Regressor(TabPFN25Regressor):
    """Sets model checkpoint for best RealTabPFN-2.5 regressor."""

    _default_ckpt = _REAL_V25_REGRESSOR_CKPT


class RealTabPFN25AltRegressor(TabPFN25Regressor):
    """Sets model checkpoint for alternate RealTabPFN-2.5 regressor.

    Previously this class was ALSO named ``RealTabPFN25Regressor`` and therefore
    shadowed the one above, silently pointing the 'best' real regressor at the
    ``real-variant`` checkpoint (which is not on disk → 'auto' → load failure).
    """

    _default_ckpt = _REAL_ALT_V25_REGRESSOR_CKPT
