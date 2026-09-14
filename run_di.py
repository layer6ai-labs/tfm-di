"""
Dataset Inference Attack for Tabular Foundation Models

Runs grid-based signal extraction on member and non-member datasets,
then evaluates with LOOCV logistic regression.

Quick Start:
    uv run run_di.py attack=fast dataset=binary_test
    uv run run_di.py attack=fast dataset=binary_small model=tabdpt
    uv run run_di.py attack=fast dataset=binary_t4_test model=sap-rpt-oss
"""

import importlib
import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from di.data_loading import load_openml_dataset
from di.datasets.t4 import T4Dataset
from di import dataset_selection
from di.blind_baseline import compute_raw_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def _resolve_model_class(import_path: str):
    """Resolve a model class from 'module.Class' or 'module:Class'."""
    if not import_path:
        return None
    if ":" in import_path:
        module_path, class_name = import_path.split(":", 1)
    else:
        module_path, class_name = import_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        log.warning(f"Failed to import {import_path}: {e}")
        return None


def _load_dataset(dataset_id, model_name: str, dataset_family: str = None,
                  test_size: float = 0.3, seed: int = 42,
                  is_task_id: bool = False, force_task_type: str = None,
                  bin_target_to: int = None,
                  select_min_unique: bool = False):
    """Load a dataset appropriate for the given model.

    For the OpenML family, ``is_task_id=True`` means the numeric id is an
    OpenML task id (CC18 / CTR23 non-members) rather than a bare dataset id
    (TabDPT training members). The two API paths cache to different
    directories so this flag must match how the id was registered upstream
    in ``di/datasets/tabdpt_training_ids.py``.
    """
    model_lower = model_name.lower()
    if dataset_family is None:
        if model_lower == "tabdpt":
            dataset_family = "openml"
        elif model_lower == "sap-rpt-oss":
            dataset_family = "t4"
        elif model_lower == "nanotabpfn":
            dataset_family = "synthetic"
        elif model_lower == "realtabpfn2":
            dataset_family = "openml"
        elif model_lower == "realtabpfn25":
            dataset_family = "openml"
        else:
            raise ValueError(f"Unknown model: {model_name}")

    if dataset_family.lower() == "openml":
        if is_task_id:
            return load_openml_dataset(
                task_id=int(dataset_id), test_size=test_size, seed=seed
            )
        return load_openml_dataset(
            dataset_id=int(dataset_id), test_size=test_size, seed=seed
        )
    elif dataset_family.lower() == "t4":
        return T4Dataset.load_dataset(
            dataset_id=str(dataset_id), test_size=test_size, seed=seed,
            force_task_type=force_task_type, bin_target_to=bin_target_to,
            select_min_unique=select_min_unique,
        )
    elif dataset_family.lower() == "synthetic":
        from di.datasets.nanotabpfn_dataset import load_nanotabpfn_dataset
        return load_nanotabpfn_dataset(
            dataset_id=dataset_id, test_size=test_size, seed=seed
        )
    else:
        raise ValueError(f"Unknown dataset family: {dataset_family}")


def _loocv_logistic_regression(X: np.ndarray, y: np.ndarray, C: float = 1.0,
                                n_steps: int = 15) -> np.ndarray:
    """Vectorized LOOCV logistic regression. Returns predicted P(member) for each sample."""
    n, d = X.shape
    # Add bias column
    X_b = np.hstack([X, np.ones((n, 1))])
    d1 = d + 1

    # Build LOO arrays: (n, n-1, d+1) and (n, n-1)
    X_loo = np.zeros((n, n - 1, d1))
    y_loo = np.zeros((n, n - 1))
    for i in range(n):
        X_loo[i] = np.delete(X_b, i, axis=0)
        y_loo[i] = np.delete(y, i)

    # Newton's method for logistic regression (all folds in parallel)
    w = np.zeros((n, d1))
    lam = 1.0 / C
    pen = lam * np.eye(d1)
    pen[-1, -1] = 0.0  # don't penalize intercept
    pen_diag = np.diag(pen)

    for _ in range(n_steps):
        z = (X_loo @ w[:, :, None]).squeeze(-1)  # (n, n-1)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        r = p * (1 - p)
        residual = p - y_loo

        grad = (X_loo.transpose(0, 2, 1) @ residual[:, :, None]).squeeze(-1) + pen_diag * w
        X_loo_r = X_loo * r[:, :, None]
        H = X_loo_r.transpose(0, 2, 1) @ X_loo + pen
        w -= np.linalg.solve(H, grad[:, :, None]).squeeze(-1)

    # Predict held-out sample
    z_test = np.sum(X_b * w, axis=1)
    return 1.0 / (1.0 + np.exp(-np.clip(z_test, -30, 30)))


def run_grid_attack(dataset_id, true_label: int, model_name: str,
                    model_config, dataset_family: str = None,
                    device: str = None, n_ensembles: int = 4,
                    context_size: int = 1024, test_size: float = 0.3,
                    seed: int = 42, cache_dir: str = None,
                    is_task_id: bool = False,
                    force_task_type: str = None,
                    bin_target_to: int = None,
                    select_min_unique: bool = False) -> dict:
    """Run grid-based DI on a single dataset."""
    try:
        X_train, X_test, y_train, y_test, metadata = _load_dataset(
            dataset_id=dataset_id, model_name=model_name,
            dataset_family=dataset_family, test_size=test_size, seed=seed,
            is_task_id=is_task_id, force_task_type=force_task_type,
            bin_target_to=bin_target_to, select_min_unique=select_min_unique,
        )
        task_type = metadata.get("target_type", "classification")

        # Resolve model class
        if model_config:
            if task_type == "classification":
                import_path = model_config.classifier.get("import_path")
                weight_path = model_config.classifier.get("default_weight_path")
                hf_path = model_config.classifier.get("hf_path")
                hf_repo = model_config.classifier.get("hf_repo")
            else:
                import_path = model_config.regressor.get("import_path")
                weight_path = model_config.regressor.get("default_weight_path")
                hf_path = model_config.regressor.get("hf_path")
                hf_repo = model_config.regressor.get("hf_repo")
            model_class = _resolve_model_class(import_path)
        else:
            from tabdpt import TabDPTClassifier, TabDPTRegressor
            model_class = TabDPTClassifier if task_type == "classification" else TabDPTRegressor
            weight_path = None
            hf_path = None
            hf_repo = None

        # Resolve relative weight paths against the project root (Hydra chdirs
        # into the run output dir, so relative paths in the model YAML would
        # otherwise miss the local safetensors).
        if weight_path and not os.path.isabs(weight_path):
            try:
                from hydra.utils import get_original_cwd
                weight_path = os.path.join(get_original_cwd(), weight_path)
            except Exception:
                pass

        # Auto-download from HF if the local checkpoint is missing and the
        # model config declares an hf_path (e.g. TabDPT seed/split variants).
        if hf_path:
            from di.model_weights import ensure_local_weights
            weight_path = ensure_local_weights(
                local_path=weight_path,
                hf_path=hf_path,
                hf_repo=hf_repo or "dwahdany/tfms",
            )

        if model_class is None:
            raise RuntimeError(f"Could not resolve model class for {model_name}")

        # Subsample large datasets
        max_train, max_test = 5000, 2000
        if len(X_train) > max_train:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train), max_train, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
        if len(X_test) > max_test:
            rng = np.random.RandomState(43)
            idx = rng.choice(len(X_test), max_test, replace=False)
            X_test, y_test = X_test[idx], y_test[idx]

        # Pool for grid
        X_pool = np.concatenate([X_train, X_test], axis=0)
        y_pool = np.concatenate([y_train, y_test], axis=0)

        # Encode labels for classification
        if task_type != "regression":
            unique_labels = np.unique(y_pool[~np.isnan(y_pool)])
            label_map = {v: i for i, v in enumerate(unique_labels)}
            y_pool = np.array([label_map.get(v, 0) for v in y_pool], dtype=np.int64)

        n_features = X_pool.shape[1]
        if n_features > 10000:
            raise ValueError(f"Skipping dataset with {n_features} features (>10000)")

        # Build and run SignalGrid
        from di.grid import SignalGrid, PredictionCache

        pred_cache = None
        if cache_dir:
            pred_cache_dir = str(Path(cache_dir) / "predictions")
            pred_cache = PredictionCache(pred_cache_dir)

        grid = SignalGrid(
            model_class=model_class,
            task_type=task_type,
            device=device or "cuda",
            weight_path=weight_path,
            verbose=True,
            n_ensembles=n_ensembles,
            default_context_size=context_size,
            cache=pred_cache,
            model_name=model_name,
            dataset_id=dataset_id if isinstance(dataset_id, int) else hash(dataset_id) % 100000,
            partition="full",
        )

        # Register sweeps
        grid.add_base_signals()
        grid.add_context_size_sweep(sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        if task_type == "classification":
            grid.add_temperature_sweep()
        grid.add_noise_sweep()
        grid.add_mislabel_sweep(strategy="random")
        # No seed sweep: the feature-permutation `seed` is a TabDPT-only
        # predict-time knob; TabPFN-2.5 (the blind) ignores it, so every seed_*
        # signal is degenerate for the blind (and seed_consistency is numerically
        # broken on regression). See di/degenerate_signals.py.
        if n_features > 2:
            grid.add_column_sweep(n_features=n_features)
        if n_features > 1:
            grid.add_task_shuffle_sweep(n_features=n_features, n_targets=8)
        grid.add_context_equals_query()
        grid.add_query_only()
        grid.add_constant_label_signals()
        grid.add_label_permutation_sweep(n_perms=5)
        grid.add_row_duplication_sweep()
        grid.add_query_leakage_sweep()
        grid.add_truth_serum_sweep()
        grid.add_brainwash_sweep()
        grid.add_feature_transform_sweep()
        grid.add_row_shuffle_sweep(n_shuffles=5)
        grid.add_split_size_sweep(n_total=len(X_pool))
        if n_features > 1:
            grid.add_tabdpt_simulation_sweep(n_features=n_features, n_targets=5)
        grid.add_nn_split_signals()

        grid_result = grid.compute(X_pool, y_pool)
        grid_dict = grid_result.to_dict()

        # Compute metadata features for blind baseline comparison
        try:
            meta_features = compute_raw_features(X_train, X_test, y_train, metadata)
        except Exception as meta_err:
            log.warning(f"Dataset {dataset_id}: metadata features failed: {meta_err}")
            meta_features = {}

        result = {
            "dataset_id": dataset_id,
            "dataset_name": metadata.get("name", f"dataset_{dataset_id}"),
            "true_label": true_label,
            "task_type": task_type,
            "n_pool": len(X_pool),
            "n_features": n_features,
            "success": True,
            "_metadata_features": meta_features,
            "_row_signals": grid_result.row_signals,
            "_row_labels": grid_result.row_labels,
        }
        result.update(grid_dict)

        log.info(f"Dataset {dataset_id}: {len(grid_result.signals)} signals computed")
        return result

    except Exception as e:
        log.error(f"Dataset {dataset_id} failed: {e}")
        log.error(traceback.format_exc())
        return {
            "dataset_id": dataset_id,
            "true_label": true_label,
            "success": False,
            "error": str(e),
        }


def evaluate_loocv(results: list[dict]) -> dict:
    """Run LOOCV logistic regression meta-classifier over all grid signals."""
    from sklearn.metrics import roc_auc_score

    successful = [r for r in results if r.get("success", False)]
    if len(successful) < 5:
        log.error(f"Only {len(successful)} successful results, need at least 5")
        return {}

    labels = np.array([r["true_label"] for r in successful], dtype=np.float64)
    task_types = np.array([str(r.get("task_type", "classification")) for r in successful])
    present_tasks = sorted(set(task_types))
    n = len(labels)

    # Auto-discover numeric signal fields
    METADATA_KEYS = {
        "dataset_id", "dataset_name", "true_label", "task_type",
        "n_pool", "n_features", "success", "error", "_curves", "_metadata",
    }
    all_signal_names = set()
    for r in successful:
        for k, v in r.items():
            if k not in METADATA_KEYS and isinstance(v, (int, float)):
                all_signal_names.add(k)

    # Build feature matrix, drop NaN/constant signals
    sig_names = []
    feature_vectors = {}
    dropped_pertask = 0
    for sig in sorted(all_signal_names):
        vals = np.array([
            float(r.get(sig, np.nan)) if r.get(sig) is not None else np.nan
            for r in successful
        ])
        finite = ~np.isnan(vals)
        # Drop signals that are all-NaN within any task type present in the pool.
        # Many signals are defined for only one task (e.g. reverse-KL
        # `*_rkl2clean_*` is all-NaN on regression, `*_neg_mse_*` is all-NaN on
        # classification). Pooling them across a mixed cls+reg pool would force
        # cross-task median imputation of a structurally-absent signal.
        if any(finite[task_types == tt].sum() == 0 for tt in present_tasks):
            dropped_pertask += 1
            continue
        if np.sum(finite) > 5 and np.nanstd(vals) > 1e-12:
            # Impute NaN with median
            median = np.nanmedian(vals)
            vals = np.where(np.isnan(vals), median, vals)
            feature_vectors[sig] = vals
            sig_names.append(sig)
    if len(present_tasks) > 1 and dropped_pertask:
        log.info(f"  dropped {dropped_pertask} signals all-NaN within a task type "
                 f"(tasks present: {', '.join(present_tasks)})")

    log.info(f"LOOCV evaluation: {n} datasets, {len(sig_names)} signals")
    log.info(f"  Members: {int(labels.sum())}, Non-members: {int((1 - labels).sum())}")

    if len(sig_names) == 0:
        log.error("No usable signals found")
        return {}

    # Build feature matrix
    X = np.column_stack([feature_vectors[s] for s in sig_names])

    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0
    X_std = (X - mean) / std

    # LOOCV logistic regression
    probs = _loocv_logistic_regression(X_std, labels)
    auc = float(roc_auc_score(labels, probs))

    # Also try best single signal
    best_single_auc = 0.5
    best_single_name = ""
    for sig in sig_names:
        vals = feature_vectors[sig]
        try:
            single_auc = float(roc_auc_score(labels, vals))
            single_auc = max(single_auc, 1 - single_auc)  # best direction
            if single_auc > best_single_auc:
                best_single_auc = single_auc
                best_single_name = sig
        except ValueError:
            continue

    metrics = {
        "n_datasets": n,
        "n_members": int(labels.sum()),
        "n_non_members": int((1 - labels).sum()),
        "n_signals": len(sig_names),
        "loocv_auc": auc,
        "best_single_signal": best_single_name,
        "best_single_auc": best_single_auc,
    }

    return metrics


def _write_parquet_tables(
    successful: list[dict],
    output_dir: Path,
    model_name: str,
) -> None:
    """Emit datasets.parquet (one row per dataset) and samples.parquet
    (one row per query sample) from the successful grid results.

    The sample table concatenates per-dataset fragments built from
    ``grid_result.row_signals`` (dict of ``np.ndarray`` stashed under
    ``_row_signals`` on each result dict). Columns present in some datasets
    but missing in others are NaN-filled by ``pd.concat``.
    """
    import pandas as pd

    # ── samples.parquet ─────────────────────────────────────────────────
    sample_frags: list[pd.DataFrame] = []
    for r in successful:
        row_signals: dict = r.get("_row_signals") or {}
        row_labels = r.get("_row_labels")
        if not row_signals:
            continue
        # Use the first array's length as the source of truth for n_query.
        first_arr = next(iter(row_signals.values()))
        n = len(first_arr)
        if n == 0:
            continue
        frag = pd.DataFrame(
            {k: np.asarray(v) for k, v in row_signals.items() if len(v) == n}
        )
        frag.insert(0, "sample_idx", np.arange(n, dtype=np.int32))
        frag.insert(0, "dataset_name", str(r.get("dataset_name", "")))
        # dataset_id can be a tuple (e.g. nanotabpfn uses
        # ("classification", "holdout", 9)); coerce to string so pandas
        # does not try to unpack it into multiple rows.
        frag.insert(0, "dataset_id", str(r.get("dataset_id")))
        frag["label"] = int(r.get("true_label", 0))
        if row_labels is not None and len(row_labels) == n:
            frag["y_query"] = np.asarray(row_labels)
        sample_frags.append(frag)

    if sample_frags:
        samples = pd.concat(sample_frags, ignore_index=True, sort=False)
        samples_path = output_dir / "samples.parquet"
        samples.to_parquet(samples_path, index=False)
        log.info(
            f"Wrote {samples_path} ({len(samples)} rows, "
            f"{len(samples.columns)} columns)"
        )
    else:
        log.warning("No per-sample rows to write — samples.parquet skipped")

    # ── datasets.parquet ────────────────────────────────────────────────
    # Flatten each successful result to the scalar signals + identifying
    # metadata. Drop internal bookkeeping fields, JSON-only structures, and
    # the numpy-array fields that only belong on samples.parquet.
    SKIP_KEYS = {
        "_curves",
        "_metadata",
        "_metadata_features",
        "_row_signals",
        "_row_labels",
        "error",
        "traceback",
    }

    dataset_rows: list[dict] = []
    for r in successful:
        row = {}
        for k, v in r.items():
            if k in SKIP_KEYS:
                continue
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[k] = v
            elif isinstance(v, (tuple, list)):
                # e.g. nanotabpfn dataset_id = ("classification", "holdout", 9)
                row[k] = str(v)
        if "true_label" in row and "label" not in row:
            row["label"] = int(row["true_label"])
        dataset_rows.append(row)

    if dataset_rows:
        datasets = pd.DataFrame(dataset_rows)
        datasets_path = output_dir / "datasets.parquet"
        datasets.to_parquet(datasets_path, index=False)
        log.info(
            f"Wrote {datasets_path} ({len(datasets)} rows, "
            f"{len(datasets.columns)} columns)"
        )
    else:
        log.warning("No dataset rows to write — datasets.parquet skipped")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for grid-based DI evaluation."""
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)

    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # Get datasets
    model_name = cfg.model["model"]
    dataset_cfg = cfg.dataset
    dataset_family_name = dataset_cfg.get("data", None)
    # `dataset_selection_model` lets a model borrow another model's
    # member/non-member dataset list. Resolution order:
    #   1. CLI override `+dataset_selection_model=...`
    #   2. `dataset_selection_model:` field on the model YAML
    #      (e.g. tabdpt_seed42 → tabdpt)
    #   3. fall back to the model's own name
    selection_model = (
        cfg.get("dataset_selection_model", None)
        or cfg.model.get("dataset_selection_model", None)
        or model_name
    )
    ids = dataset_selection.get_model_datasets(selection_model, dataset_family_name)

    num_members = dataset_cfg.get("num_members", 10)
    num_non_members = dataset_cfg.get("num_non_members", 10)
    member_ids = ids["members"][:num_members]
    non_member_ids = ids["non_members"][:num_non_members]

    log.info(f"Selected {len(member_ids)} members, {len(non_member_ids)} non-members")

    # Attack config
    attack_cfg = cfg.attack
    n_ensembles = attack_cfg.model.n_ensembles
    context_size = attack_cfg.model.context_size
    seed = cfg.get("seed", 42)
    device = cfg.get("device", None)
    test_size = dataset_cfg.get("test_size", 0.3)

    # Cache dir for prediction caching.
    # Disable with +no_cache=true for large dataset configs (100K+ datasets)
    # where the ~90 npz files per dataset would exhaust filesystem quotas.
    no_cache = cfg.get("no_cache", False)
    if no_cache:
        cache_dir = None
    else:
        _default_cache = Path(__file__).resolve().parent / "slurm_cache"
        if (_default_cache / "predictions").exists() or (_default_cache / "predictions").is_symlink():
            cache_dir = str(_default_cache)
        else:
            cache_dir = str(output_dir / "cache")

    # Run attacks
    all_datasets = [(family, did, 1) for family, did in member_ids] + \
                   [(family, did, 0) for family, did in non_member_ids]

    # Optional SLURM-style sharding: a launcher can submit N array elements
    # with chunk_idx in [0, n_chunks) so each element processes a stride of
    # the dataset list. Round-robin (stride) slicing keeps the member /
    # non-member ratio roughly balanced within every chunk, which matters
    # when a chunk times out mid-run.
    chunk_idx = int(cfg.get("chunk_idx", 0))
    n_chunks = int(cfg.get("n_chunks", 1))
    if n_chunks > 1:
        if not (0 <= chunk_idx < n_chunks):
            raise ValueError(
                f"chunk_idx={chunk_idx} out of range for n_chunks={n_chunks}"
            )
        all_datasets = all_datasets[chunk_idx::n_chunks]
        log.info(
            f"Chunk {chunk_idx}/{n_chunks}: {len(all_datasets)} datasets"
        )
    total = len(all_datasets)

    results = []
    for idx, (family, dataset_id, true_label) in enumerate(all_datasets, 1):
        log.info(f"[{idx}/{total}] Dataset {dataset_id} (family={family}, label={true_label})")
        # OpenML non-members come from CC18 / CTR23 task id lists, not bare
        # dataset ids. The two API paths cache to different directories and
        # some CC18 task ids have no public OpenML dataset of the same id.
        is_task_id = (
            family == "openml" and true_label == 0
        )
        result = run_grid_attack(
            dataset_id=dataset_id,
            true_label=true_label,
            model_name=model_name,
            model_config=cfg.model,
            dataset_family=family,
            device=device,
            n_ensembles=n_ensembles,
            context_size=context_size,
            test_size=test_size,
            seed=seed,
            cache_dir=cache_dir,
            is_task_id=is_task_id,
        )
        results.append(result)

    successful = [r for r in results if r.get("success", False)]
    log.info(f"Completed {len(successful)}/{total} datasets successfully")

    # Evaluate
    metrics = evaluate_loocv(results)

    if metrics:
        print(f"\n{'='*60}")
        print(" EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"  Datasets: {metrics['n_datasets']} ({metrics['n_members']} members, {metrics['n_non_members']} non-members)")
        print(f"  Signals:  {metrics['n_signals']}")
        print(f"  LOOCV LR AUC: {metrics['loocv_auc']:.4f}")
        print(f"  Best single signal: {metrics['best_single_signal']} (AUC={metrics['best_single_auc']:.4f})")
        print(f"{'='*60}")

    # Build & write parquet tables (datasets.parquet + samples.parquet).
    # This runs before JSON serialization because the per-sample numpy arrays
    # live on the result dicts under `_row_signals` / `_row_labels` and must
    # be stripped before json.dump.
    try:
        _write_parquet_tables(successful, output_dir, model_name)
    except Exception as e:
        log.warning(f"Failed to write parquet tables: {e}")

    # Drop non-JSON-serializable fields from result dicts before saving JSON
    for r in successful:
        r.pop("_row_signals", None)
        r.pop("_row_labels", None)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{dataset_cfg.name}_{attack_cfg.name}_{timestamp}"

    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "attack_config": attack_cfg.name,
            "n_ensembles": n_ensembles,
            "context_size": context_size,
            "seed": seed,
        },
        "metrics": metrics,
        "successful_results": successful,
    }

    output_path = output_dir / f"{output_name}.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
