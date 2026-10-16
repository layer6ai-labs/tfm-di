"""Prediction engine: groups by fit-state, runs predict batch."""

from __future__ import annotations

from typing import Any

import numpy as np

from .specs import ManipulationSpec, PredictionSpec
from .cache import PredictionCache
from .manipulations import apply_manipulation


def _encode_labels(y: np.ndarray) -> tuple[np.ndarray, dict]:
    """Encode labels to contiguous 0-indexed integers."""
    unique = np.unique(y[~np.isnan(y)])
    label_map = {v: i for i, v in enumerate(unique)}
    y_enc = np.array([label_map.get(v, 0) for v in y], dtype=np.int64)
    return y_enc, label_map


def _reset_cuda():
    import torch

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except RuntimeError:
            pass


class PredictionEngine:
    """Runs predictions grouped by fit-state for efficiency.

    Usage:
        engine = PredictionEngine(model_class, task_type, device=device, weight_path=path)
        results = engine.run_group(
            X_context, y_context, X_query,
            manip_spec, [pred_spec_1, pred_spec_2, ...],
        )
        # results: dict[PredictionSpec, dict[str, np.ndarray]]
    """

    def __init__(
        self,
        model_class: type,
        task_type: str,
        *,
        device: str | None = None,
        weight_path: str | None = None,
        verbose: bool = False,
        normalizer: str = "standard",
    ):
        self.model_class = model_class
        self.task_type = task_type
        self.device = device
        self.weight_path = weight_path
        self.verbose = verbose
        self.normalizer = normalizer

    def run_group(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        y_query: np.ndarray,
        manip: ManipulationSpec,
        pred_specs: list[PredictionSpec],
        cache: PredictionCache | None = None,
        model_name: str = "",
        dataset_id: int = 0,
        partition: str = "full",
    ) -> dict[PredictionSpec, dict[str, np.ndarray]]:
        """Run one fit + multiple predicts. Returns arrays per PredictionSpec."""
        # Check cache for all specs
        results: dict[PredictionSpec, dict[str, np.ndarray]] = {}
        missing_specs: list[PredictionSpec] = []

        if cache is not None:
            for ps in pred_specs:
                cached = cache.load(model_name, dataset_id, partition, manip, ps)
                if cached is not None:
                    results[ps] = cached
                else:
                    missing_specs.append(ps)
        else:
            missing_specs = list(pred_specs)

        # Apply manipulation (needed for fit/predict and for y_query override)
        needs_manipulation = (
            bool(missing_specs)
            or manip.target_column is not None
            or manip.context_equals_query
        )
        y_q_manip = None
        X_ctx = X_q = None
        if needs_manipulation:
            X_ctx, y_ctx, X_q, y_q_manip, col_indices = apply_manipulation(
                X_context, y_context, manip, X_query, y_query
            )

        if not missing_specs:
            # All cached — but still inject y_query for task shuffle / context_equals_query
            if (manip.target_column is not None or manip.context_equals_query) and y_q_manip is not None:
                for ps in pred_specs:
                    if ps in results and len(results[ps].get("probs", results[ps].get("preds", []))) > 0:
                        results[ps]["y_query"] = y_q_manip.astype(np.float32)
            return results

        # Encode labels for classification
        if self.task_type == "classification":
            y_ctx_enc, _ = _encode_labels(y_ctx)
            if len(np.unique(y_ctx_enc)) < 2:
                # Can't fit with <2 classes
                for ps in missing_specs:
                    results[ps] = {"probs": np.array([]), "preds": np.array([])}
                return results
        else:
            y_ctx_enc = y_ctx

        # Determine normalizer: manipulation can override
        normalizer = manip.normalizer if manip.normalizer != "standard" else self.normalizer

        # Fit once
        try:
            # Seed torch before every fit so any randomized op inside the backend
            # is reproducible across runs / SLURM array elements / merged result
            # files. In particular TabDPT.fit uses torch.pca_lowrank (randomized
            # SVD) for datasets with more features than the model supports; without
            # a fixed seed the PCA basis — and therefore every signal on a wide
            # dataset, incl. the clean-vs-corruption KL-to-clean comparison — is
            # non-deterministic.
            import torch
            torch.manual_seed(0)
            model = self.model_class(
                device=self.device,
                verbose=False,
                model_weight_path=self.weight_path,
                normalizer=normalizer,
            )
            model.fit(X_ctx, y_ctx_enc)
        except Exception as e:
            if self.verbose:
                print(f"  Fit failed for manip={manip.cache_key()}: {e}")
            _reset_cuda()
            for ps in missing_specs:
                results[ps] = {"probs": np.array([]), "preds": np.array([])}
            return results

        # Predict for each missing spec
        query_data = X_q if X_q is not None else X_query
        for ps in missing_specs:
            try:
                arrays = self._predict(model, query_data, y_query, ps)
                results[ps] = arrays
                if cache is not None:
                    cache.save(model_name, dataset_id, partition, manip, ps, arrays)
            except Exception as e:
                if self.verbose:
                    print(
                        f"  Predict failed for manip={manip.cache_key()} "
                        f"pred={ps.cache_key()}: {e}"
                    )
                _reset_cuda()
                results[ps] = {"probs": np.array([]), "preds": np.array([])}

        # Store modified y_query for measurement extraction (task shuffle, context_equals_query)
        if (manip.target_column is not None or manip.context_equals_query) and y_q_manip is not None:
            for ps in pred_specs:
                if ps in results and len(results[ps].get("probs", results[ps].get("preds", []))) > 0:
                    results[ps]["y_query"] = y_q_manip.astype(np.float32)

        return results

    def _predict(
        self,
        model: Any,
        X_query: np.ndarray,
        y_query: np.ndarray,
        spec: PredictionSpec,
    ) -> dict[str, np.ndarray]:
        """Run a single prediction and return arrays."""
        predict_kwargs: dict[str, Any] = {}
        # Always pass context_size, INCLUDING when it is None. ``None`` must mean
        # "use the entire fitted context" uniformly across backends:
        #   * TabDPT   → None → np.inf → no FAISS retrieval, whole context used
        #   * NanoTabPFN / SAP → None → head-slice skipped, whole context used
        #   * TabPFN25 → context_size ignored, whole context used anyway
        # If we only passed it when non-None (the old behaviour), TabDPT would
        # silently fall back to its signature default of 2048, so an intended
        # "no-truncation" prediction would still k-NN-truncate — which is exactly
        # what let the appended truth-serum / brainwash poison be dropped on large
        # tables. Passing None explicitly makes the corruption manipulations
        # apply at their true fraction regardless of table size.
        predict_kwargs["context_size"] = spec.context_size
        if spec.seed is not None:
            predict_kwargs["seed"] = spec.seed

        if self.task_type == "classification":
            if spec.return_logits:
                logits = model.predict_proba(
                    X_query,
                    temperature=spec.temperature,
                    return_logits=True,
                    **predict_kwargs,
                )
                logits = np.asarray(logits, dtype=np.float32)
                # Slice off padding-class columns: TabDPT returns a fixed
                # max_num_classes-wide logit vector, so without this the logit
                # geometry signals (logit_magnitude / logit_kurtosis) would be
                # dominated by padding logits and become a class-count proxy —
                # and would not be comparable to the blind backend, which emits
                # exactly num_classes columns.
                nc = getattr(model, "num_classes", None)
                if nc and logits.ndim == 2 and logits.shape[1] > nc:
                    logits = logits[:, :nc]
                return {"logits": logits}
            else:
                probs = model.predict_proba(
                    X_query,
                    temperature=spec.temperature,
                    **predict_kwargs,
                )
                preds = np.argmax(probs, axis=1)
                return {
                    "probs": probs.astype(np.float32),
                    "preds": preds.astype(np.int32),
                }
        else:
            # Regression: force a single deterministic pass. TabDPT's default
            # predict() runs an 8-member ensemble with a random (seed=None)
            # feature permutation, which makes every regression signal
            # non-deterministic and variance-smoothed (and the disk cache would
            # freeze whichever random realization was computed first). n_ensembles
            # is a no-op kwarg for the nano/tabpfn25 regressors.
            preds = model.predict(X_query, n_ensembles=1, **predict_kwargs)
            return {"preds": np.asarray(preds, dtype=np.float32)}
