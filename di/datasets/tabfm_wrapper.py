"""Thin TabFM adapter exposing the TabDPT-style API expected by the grid runner.

TabFM (google-research/tabfm, v1.0.0, released 2026-06-30) is a second *blind*
baseline alongside TabPFN-2.5.  Like TabPFN's synthetic checkpoint it is
pre-trained exclusively on synthetic tables generated from structural causal
models — it has never seen an OpenML table — so it is a valid stand-in for
"what a model that did not train on this dataset would do".

Why bother with a second blind at all: a single blind that happens to be *wrong*
on some dataset family manufactures a positive target-minus-blind gap without
any memorisation (the anti-predictive-blind failure mode).  TabFM is
architecturally unrelated to TabPFN — alternating row/column attention with row
compression, feeding a 24-block ICL transformer, 1.64 B parameters — so its
errors are far less correlated with TabPFN's than a second TabPFN checkpoint
would be.

Benchmarked over the whole DI pool (194 datasets: 72 CC18 non-members + 122
TabDPT members, 4096 context rows, 1000 query rows).  On the 134 datasets every
model completed, TabFM at ``n_estimators=4`` beats TabPFN-2.5 on **112, ties 8,
loses 14** (mean ROC-AUC 0.9094 vs 0.8980, mean delta +0.0114, median +0.0047),
and beats the TabDPT target too (0.8913).  A stronger blind is the conservative
choice: a gap that survives it is memorisation rather than the target merely
being the better model.

It is also the cheaper blind.  With ``cache_context`` the repeat-predict cost at
4096 context rows is 0.03 s (n_estimators=1) / 0.13 s (n_estimators=4) against
TabPFN-2.5's 0.35 s, because TabPFN re-encodes its context on every predict
while TabFM encodes once per fit — which is exactly how the grid drives it.  The
32-member upstream default costs 6.78 s per predict for ~0.0004 mean AUC over 4
members, so ``n_estimators`` defaults to 4.

Three upstream quirks this wrapper papers over:

* **NaN**: ``TabFMClassifier.fit`` raises ``ValueError: Input X contains NaN``
  when handed a float ndarray — the internal ``SimpleImputer`` only sits on the
  pandas/mixed-dtype code path.  Roughly 17 % of the DI pool carries some
  missing values, so every input is converted to a ``DataFrame`` first.
* **weights**: the checkpoint is 6.1 GB on disk (3.3 GB resident, bf16) and
  takes 30-70 s to load.  The grid builds a fresh model object per
  ``ManipulationSpec``, so the loaded module is cached at module level.
* **safetensors**: ``load()`` dies with ``NameError: name 'safetensors' is not
  defined`` if the optional ``safetensors`` package is absent — it is an
  undeclared dependency of huggingface_hub's PyTorch mixin path.  The import
  error is re-raised with an actionable message.
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Any

import numpy as np

# Ensemble members. The upstream default is 32, which costs 6.78 s per predict
# at 4096 context rows against 0.85 s for 4 members while winning on 18/18 of
# the same datasets either way. 4 is the accuracy/cost knee.
_DEFAULT_N_ESTIMATORS = int(os.environ.get("TFM_DI_TABFM_N_ESTIMATORS", "4"))

# Encode the context once per fit and reuse it across the many predicts the
# grid issues per manipulation. Quantisation is disabled: upstream documents the
# unquantised cached path as numerically identical to the uncached one, and int8
# rounding has no place in a baseline used to compute fine-grained signals.
_DEFAULT_CACHE_CONTEXT = os.environ.get("TFM_DI_TABFM_CACHE_CONTEXT", "1") != "0"

# TabDPT's predict-time default; matches ``tabpfn25_wrapper`` so that
# "unmanipulated" means the same thing for both blinds.
_TABDPT_DEFAULT_TEMPERATURE = 0.8

# ``normalizer`` names used by the grid -> TabFM's ``norm_methods`` vocabulary.
# TabFM always applies its own standardisation on top, so "standard" and
# "minmax" both collapse to "none" (i.e. no *extra* transform).
_NORMALIZER_MAP = {
    "standard": "none",
    "minmax": "none",
    "robust": "robust",
    "power": "power",
    "quantile": "quantile",
}

# Module-level cache: the 6.1 GB checkpoint is loaded once per (model_type,
# device) and shared by every wrapper instance.
_model_cache: dict[tuple[str, str], Any] = {}
_cache_lock = Lock()


def _load_backbone(model_type: str, device: str):
    """Load (and memoise) the TabFM backbone for ``model_type`` on ``device``."""
    key = (model_type, device)
    with _cache_lock:
        if key in _model_cache:
            return _model_cache[key]

    try:
        import safetensors  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "tabfm's PyTorch loader needs `safetensors`, which tabfm does not "
            "declare as a dependency. Install it with `uv pip install safetensors`."
        ) from exc

    from tabfm import tabfm_v1_0_0_pytorch as backbone

    model = backbone.load(model_type=model_type)
    try:
        model = model.to(device)
    except Exception:
        # Some backbone builds are already pinned to a device; not fatal.
        pass

    with _cache_lock:
        _model_cache[key] = model
    return model


def _as_frame(X):
    """Return ``X`` as a DataFrame so TabFM's imputing preprocessor runs.

    Passing a bare ndarray skips ``TransformToNumerical`` and lands in
    ``EnsembleGenerator``, whose ``check_array`` rejects NaN outright.
    """
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])


class _BaseTabFMAdapter:
    """Common init / fit shared by the classifier and regressor wrappers.

    TabFM does not expose the ``temperature`` / ``context_size`` / ``seed``
    predict-time knobs that TabDPT does:

    * **temperature** — applied post-hoc (divide log-probs by *T*, re-softmax),
      identically to ``tabpfn25_wrapper``, so the temperature family stays
      comparable across the two blinds.
    * **context_size** — ignored. ``ManipulationSpec.context_size`` already
      subsamples rows before ``fit()``.
    * **seed** — ignored. TabDPT uses it for a predict-time column permutation;
      TabFM's equivalent (``random_state``, driving the ensemble's feature
      shuffle) is bound at construction and cannot be varied per predict without
      refitting. ``tabpfn25_wrapper`` ignores it for the same reason, which is
      why the ``seed_*`` signal family is on the must-exclude list for blinds.

    Unlike ``tabpfn25_wrapper``, ``normalizer`` *is* honoured — TabFM exposes
    ``norm_methods`` — so the normaliser-robustness family is non-degenerate for
    this blind even though it is degenerate for the TabPFN-2.5 one.
    """

    _model_type: str = "classification"
    _estimator_attr: str = "TabFMClassifier"

    def __init__(
        self,
        device: str | None = None,
        verbose: bool = False,
        model_weight_path: str | None = None,  # noqa: ARG002 - HF-hosted weights
        normalizer: str = "standard",
        n_estimators: int | None = None,
        cache_context: bool | None = None,
        **kwargs,
    ) -> None:
        import tabfm

        backend_device = "cuda" if device == "cuda" else "cpu"
        backbone = _load_backbone(self._model_type, backend_device)

        self._n_estimators = (
            _DEFAULT_N_ESTIMATORS if n_estimators is None else n_estimators
        )
        self._cache_context = (
            _DEFAULT_CACHE_CONTEXT if cache_context is None else cache_context
        )
        self._norm_method = _NORMALIZER_MAP.get(normalizer, "none")
        self._estimator_cls = getattr(tabfm, self._estimator_attr)
        self._backbone = backbone
        self._verbose = verbose
        self.model = None
        self._cached_active = False
        self._X = None
        self._y = None

    def _build(self, cache_context: bool):
        return self._estimator_cls(
            model=self._backbone,
            n_estimators=self._n_estimators,
            norm_methods=[self._norm_method],
            random_state=42,
            verbose=self._verbose,
            cache_context=cache_context,
            maybe_quantize_kv_cache=False,
        )

    @staticmethod
    def _free_cuda():
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def _refit_uncached(self) -> bool:
        """Rebuild without context caching and refit. False if already uncached."""
        if not self._cached_active:
            return False
        self._cached_active = False
        self.model = None
        self._free_cuda()
        self.model = self._build(cache_context=False)
        self.model.fit(self._X, self._y)
        return True

    def fit(self, X, y):
        self._X = _as_frame(X)
        self._y = np.asarray(y)
        # Context caching materialises per-member K/V for the whole context,
        # which on a wide table with a large context can exhaust the device.
        # Retry uncached rather than null out the manipulation.
        #
        # Scale note: at 4096 context rows, 12 of 194 benchmark datasets OOM'd on
        # a 24 GB L4 (>=561 features) — and the uncached path OOM'd too, so this
        # fallback does not rescue that regime. All 12 run clean at context 1024,
        # which is what SignalGrid actually uses, so the wide-table OOM is an
        # artifact of that benchmark's 4096 cap rather than a production limit.
        # Deliberately NOT falling back to a reduced ``max_num_rows``: silently
        # shrinking the context on wide datasets would weaken the blind exactly
        # where tables are widest and inflate the target-minus-blind gap there.
        # A loud failure that nulls one manipulation is the safer error.
        if self._cache_context:
            self._cached_active = True
            try:
                self.model = self._build(cache_context=True)
                self.model.fit(self._X, self._y)
                return self
            except (RuntimeError, MemoryError, NotImplementedError):
                pass
        self._cached_active = False
        self.model = None
        self._free_cuda()
        self.model = self._build(cache_context=False)
        self.model.fit(self._X, self._y)
        return self

    def _call_with_oom_retry(self, fn_name: str, X):
        """Run ``model.<fn_name>(X)``, refitting uncached once if it OOMs.

        The cached-context path can survive ``fit`` and only blow up inside
        ``predict``, so the fallback has to cover both halves.
        """
        frame = _as_frame(X)
        try:
            return getattr(self.model, fn_name)(frame)
        except (RuntimeError, MemoryError) as exc:
            self._free_cuda()
            if not self._refit_uncached():
                raise exc
            return getattr(self.model, fn_name)(frame)

    def predict(self, X, **kwargs):
        return np.asarray(self._call_with_oom_retry("predict", X))


class TabFMClassifier(_BaseTabFMAdapter):
    """Classifier wrapper around ``tabfm.TabFMClassifier``.

    Raises on targets with more than 10 classes — an architectural cap of the
    checkpoint. TabPFN-2.5 has the same limit, so this is not a regression
    relative to the incumbent blind; the grid catches the failure per dataset.
    """

    _model_type = "classification"
    _estimator_attr = "TabFMClassifier"

    def predict_proba(
        self,
        X,
        temperature: float | None = None,
        context_size: int | None = None,  # noqa: ARG002 - applied before fit()
        return_logits: bool = False,
        seed: int | None = None,  # noqa: ARG002 - no predict-time equivalent
        **kwargs,
    ) -> np.ndarray:
        probs = np.asarray(self._call_with_oom_retry("predict_proba", X), dtype=np.float64)

        if temperature is not None and temperature != _TABDPT_DEFAULT_TEMPERATURE:
            log_probs = np.log(np.clip(probs, 1e-12, 1.0))
            scaled = log_probs / temperature
            scaled -= scaled.max(axis=1, keepdims=True)  # numerical stability
            exp_scaled = np.exp(scaled)
            probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

        if return_logits:
            return np.log(np.clip(probs, 1e-12, 1.0))
        return probs


class TabFMRegressor(_BaseTabFMAdapter):
    """Regressor wrapper around ``tabfm.TabFMRegressor``.

    Loads the separate regression checkpoint; ``predict_proba`` is aliased to
    ``predict`` because the grid's regression path calls whichever is wired up.
    """

    _model_type = "regression"
    _estimator_attr = "TabFMRegressor"

    def predict_proba(
        self,
        X,
        temperature: float | None = None,  # noqa: ARG002 - meaningless for regression
        context_size: int | None = None,  # noqa: ARG002
        return_logits: bool = False,  # noqa: ARG002
        seed: int | None = None,  # noqa: ARG002
        **kwargs,
    ) -> np.ndarray:
        return self.predict(X)
