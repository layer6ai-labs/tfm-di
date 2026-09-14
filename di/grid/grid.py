"""SignalGrid: orchestrator that composes manipulations x measurements.

Usage:
    grid = SignalGrid(
        model_class=TabDPTClassifier,
        task_type="classification",
        device="cuda",
        weight_path="/path/to/weights.safetensors",
    )
    grid.add_context_size_sweep()
    grid.add_temperature_sweep()
    grid.add_noise_sweep()
    grid.add_mislabel_sweep()
    grid.add_seed_sweep()
    grid.add_normalizer_sweep()
    grid.add_column_sweep(n_features=50)
    result = grid.compute(X_all, y_all)
    # result.signals: dict[str, float | None]
    # result.curves: dict[str, dict[float, float]]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .specs import ManipulationSpec, PredictionSpec, ModelSpec
from .predictions import PredictionEngine, _encode_labels
from .measurements import (
    MeasurementExtractor, LOGIT_MEASUREMENTS,
    kl_to_clean, reverse_kl_to_clean, regression_kl_to_clean,
)
from .trajectories import compute_trajectory, compute_aggregates, compute_consistency
from .cache import PredictionCache
from .row_signals import extract_row_signals
from ..degenerate_signals import is_blind_degenerate


@dataclass
class GridResult:
    """Output of SignalGrid.compute()."""

    signals: dict[str, float | None] = field(default_factory=dict)
    curves: dict[str, dict[float, float | None]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    row_signals: dict[str, np.ndarray] = field(default_factory=dict)
    row_labels: np.ndarray | None = None

    def to_dict(self) -> dict:
        """Flat dict for JSON serialization.

        ``row_signals`` / ``row_labels`` are intentionally **not** included —
        they are numpy arrays, best kept out of the JSON blob and instead
        consumed via the dataclass fields by callers that want parquet output.

        Blind-degenerate signals are never *derived* (see ``_derive_signals``),
        so no filtering is needed here.
        """
        out: dict[str, Any] = {}
        out.update(self.signals)
        out["_curves"] = {
            k: {str(kk): vv for kk, vv in v.items()} for k, v in self.curves.items()
        }
        out["_metadata"] = self.metadata
        return out

    @classmethod
    def from_dict(cls, d: dict) -> GridResult:
        curves_raw = d.pop("_curves", {})
        metadata = d.pop("_metadata", {})
        curves = {
            k: {float(kk): vv for kk, vv in v.items()}
            for k, v in curves_raw.items()
        }
        return cls(signals=d, curves=curves, metadata=metadata)


# Type alias for a sweep: list of (intensity, ManipulationSpec, [PredictionSpec])
SweepEntry = tuple[float, ManipulationSpec, list[PredictionSpec]]

# Sweeps whose manipulation APPENDS poisoned / duplicated rows to the tail of
# the context (as opposed to modifying existing rows in place, like mislabel or
# noise). For these the appended rows are the whole point of the manipulation,
# so they must never be silently dropped by the model's context-selection step
# (head-slicing in the NanoTabPFN / SAP wrappers, or k-NN retrieval in TabDPT).
# They are handled specially in two places:
#   * ``_corruption_specs`` caps the clean base context to a fixed budget and
#     predicts on the ENTIRE manipulated context (context_size=None), so the
#     poison fraction is applied in full and is independent of table size.
#   * ``_compute_kl_to_clean`` measures divergence against each sweep's own
#     zero-intensity (frac=0) prediction rather than the global base, because
#     the corruption sweeps run under this different (capped, full-context)
#     regime and the global base uses k-NN-retrieved context_size instead.
CORRUPTION_SWEEPS = frozenset({"rowdup", "qleak", "tserum", "bwash"})


class SignalGrid:
    """Orchestrates manipulation x measurement grid computation."""

    def __init__(
        self,
        model_class: type,
        task_type: str = "classification",
        *,
        device: str | None = None,
        weight_path: str | None = None,
        verbose: bool = True,
        n_ensembles: int = 4,
        default_temperature: float = 0.8,
        default_context_size: int = 1024,
        cache: PredictionCache | None = None,
        model_name: str = "tabdpt_default",
        dataset_id: int = 0,
        partition: str = "full",
        n_splits: int = 1,
        test_size: float = 0.3,
        split_base_seed: int = 42,
    ):
        self.task_type = task_type
        self.verbose = verbose
        self.n_ensembles = n_ensembles
        self.default_temperature = default_temperature
        self.default_context_size = default_context_size
        self.cache = cache
        self.model_name = model_name
        self.dataset_id = dataset_id
        self.partition = partition
        self.n_splits = n_splits
        self.test_size = test_size
        self.split_base_seed = split_base_seed

        self.engine = PredictionEngine(
            model_class=model_class,
            task_type=task_type,
            device=device,
            weight_path=weight_path,
            verbose=verbose,
        )
        self.extractor = MeasurementExtractor(task_type)

        # Registered sweeps: name -> list of SweepEntry
        self._trajectory_sweeps: dict[str, list[SweepEntry]] = {}
        # Aggregate sweeps: name -> list of (variant_label, ManipulationSpec, [PredictionSpec])
        self._aggregate_sweeps: dict[str, list[SweepEntry]] = {}
        # Consistency sweeps: name -> (ManipulationSpec, [PredictionSpec])
        self._consistency_sweeps: dict[str, tuple[ManipulationSpec, list[PredictionSpec]]] = {}
        # One-shot predictions (base signals, logits)
        self._oneshots: list[tuple[str, ManipulationSpec, PredictionSpec]] = []
        # NN split flag
        self._nn_split: bool = False

    def _default_pred(self, **overrides) -> PredictionSpec:
        kw: dict[str, Any] = {
            "temperature": self.default_temperature,
            "context_size": self.default_context_size,
            "n_ensembles": self.n_ensembles,
        }
        kw.update(overrides)
        return PredictionSpec(**kw)

    def _default_manip(self, **overrides) -> ManipulationSpec:
        return ManipulationSpec(**overrides)

    def _corruption_specs(
        self, **corruption_kwargs
    ) -> tuple[ManipulationSpec, PredictionSpec]:
        """Build (manip, pred) for an append-corruption sweep entry.

        The base context is capped to ``default_context_size`` (subsample_rows
        runs *before* the append step in the manipulation pipeline), and the
        prediction uses the entire manipulated context (``context_size=None``).
        Together these make the poison fraction exact and table-size-independent
        and stop the appended rows from being truncated away. See
        ``CORRUPTION_SWEEPS``. A frac=0 entry reduces to (cap-only manip,
        full-context pred) — the clean reference used by ``_compute_kl_to_clean``.
        """
        manip = self._default_manip(
            context_size=self.default_context_size, **corruption_kwargs
        )
        pred = self._default_pred(context_size=None)
        return manip, pred

    # ── Sweep registration ──────────────────────────────────────────────

    def add_base_signals(self) -> None:
        """Add base prediction at default settings + logit prediction."""
        manip = self._default_manip()
        pred = self._default_pred()
        self._oneshots.append(("base", manip, pred))
        # Logits
        if self.task_type == "classification":
            pred_logit = self._default_pred(return_logits=True)
            self._oneshots.append(("logit", manip, pred_logit))

    def add_context_size_sweep(
        self,
        sizes: list[int] | None = None,
    ) -> None:
        """Trajectory sweep over context sizes."""
        if sizes is None:
            sizes = [16, 32, 64, 128, 256, 512, 1024]
        entries: list[SweepEntry] = []
        for size in sizes:
            manip = self._default_manip(context_size=size)
            pred = self._default_pred()
            entries.append((float(size), manip, [pred]))
        self._trajectory_sweeps["ctx"] = entries

    def add_temperature_sweep(
        self,
        temperatures: list[float] | None = None,
    ) -> None:
        """Aggregate sweep over temperatures (shares one fit)."""
        if temperatures is None:
            temperatures = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
        manip = self._default_manip()
        pred_specs = [self._default_pred(temperature=T) for T in temperatures]
        entries: list[SweepEntry] = []
        for T, ps in zip(temperatures, pred_specs):
            entries.append((T, manip, [ps]))
        self._aggregate_sweeps["temp"] = entries

    def add_noise_sweep(
        self,
        noise_levels: list[float] | None = None,
    ) -> None:
        """Trajectory sweep over noise levels."""
        if noise_levels is None:
            noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
        entries: list[SweepEntry] = []
        for alpha in noise_levels:
            manip = self._default_manip(noise_level=alpha, noise_seed=42)
            pred = self._default_pred()
            entries.append((alpha, manip, [pred]))
        self._trajectory_sweeps["noise"] = entries

    def add_mislabel_sweep(
        self,
        fractions: list[float] | None = None,
        strategy: str = "random",
    ) -> None:
        """Trajectory sweep over mislabel fractions."""
        if fractions is None:
            fractions = [0.0, 0.05, 0.1, 0.2, 0.5]
        tag = f"mislbl_{strategy}"
        entries: list[SweepEntry] = []
        for frac in fractions:
            manip = self._default_manip(
                mislabel_fraction=frac,
                mislabel_strategy=strategy,
                mislabel_seed=42,
            )
            pred = self._default_pred()
            entries.append((frac, manip, [pred]))
        self._trajectory_sweeps[tag] = entries

    def add_seed_sweep(
        self,
        n_seeds: int = 10,
        base_seed: int = 42,
    ) -> None:
        """Aggregate + consistency sweep over feature permutation seeds."""
        manip = self._default_manip()
        entries: list[SweepEntry] = []
        all_pred_specs: list[PredictionSpec] = []
        for i in range(n_seeds):
            ps = self._default_pred(seed=base_seed + i)
            entries.append((float(i), manip, [ps]))
            all_pred_specs.append(ps)
        self._aggregate_sweeps["seed"] = entries
        self._consistency_sweeps["seed"] = (manip, all_pred_specs)

    def add_normalizer_sweep(
        self,
        normalizers: list[str] | None = None,
    ) -> None:
        """Aggregate sweep over normalizer choices (separate fits per normalizer)."""
        if normalizers is None:
            normalizers = ["standard", "minmax", "robust", "power", "quantile-normal"]
        entries: list[SweepEntry] = []
        for i, norm in enumerate(normalizers):
            manip = self._default_manip(normalizer=norm)
            pred = self._default_pred()
            entries.append((float(i), manip, [pred]))
        self._aggregate_sweeps["norm"] = entries

    def add_column_sweep(
        self,
        n_features: int,
        n_points: int = 9,
        min_cols: int = 2,
    ) -> None:
        """Trajectory sweep over number of features."""
        if n_features <= min_cols:
            return
        raw = np.geomspace(min_cols, n_features, num=max(2, n_points))
        sizes = sorted({int(round(v)) for v in raw if 1 <= int(round(v)) <= n_features})
        if sizes[0] != min_cols:
            sizes = [min_cols] + sizes
        if sizes[-1] != n_features:
            sizes.append(n_features)
        sizes = sorted(set(sizes))

        entries: list[SweepEntry] = []
        for ncol in sizes:
            manip = self._default_manip(column_count=ncol, column_seed=42)
            pred = self._default_pred()
            entries.append((float(ncol), manip, [pred]))
        self._trajectory_sweeps["col"] = entries

    def add_context_composition(
        self,
        random_context_size: int = 500,
    ) -> None:
        """Compare k-NN vs sequential context (two-point aggregate)."""
        entries: list[SweepEntry] = []
        # k-NN (default)
        manip_knn = self._default_manip(context_selection="knn")
        pred_knn = self._default_pred()
        entries.append((0.0, manip_knn, [pred_knn]))
        # Random
        manip_rand = self._default_manip(
            context_size=random_context_size, context_selection="random"
        )
        pred_rand = self._default_pred(context_size=None)  # no k-NN retrieval at predict
        entries.append((1.0, manip_rand, [pred_rand]))
        self._aggregate_sweeps["ctx_comp"] = entries

    def add_task_shuffle_sweep(
        self,
        n_features: int,
        n_targets: int = 5,
        seed: int = 42,
    ) -> None:
        """Aggregate sweep over target columns (predict different columns as Y)."""
        rng = np.random.default_rng(seed)
        # Select from X columns only (0..n_features-1); n_features index = original Y
        n = min(n_targets, n_features)
        target_cols = sorted(rng.choice(n_features, size=n, replace=False))

        entries: list[SweepEntry] = []
        for i, col in enumerate(target_cols):
            manip = self._default_manip(target_column=int(col))
            pred = self._default_pred()
            entries.append((float(i), manip, [pred]))
        self._aggregate_sweeps["tshuffle"] = entries

    def add_context_equals_query(self) -> None:
        """One-shot: predict on context data itself (measures overfitting to context)."""
        manip = self._default_manip(context_equals_query=True)
        pred = self._default_pred()
        self._oneshots.append(("ctxeqq", manip, pred))

    def add_query_only(self) -> None:
        """One-shot: minimal context (1 row), measures model prior."""
        manip = self._default_manip(context_size=1)
        pred = self._default_pred()
        self._oneshots.append(("qonly", manip, pred))

    def add_label_permutation_sweep(
        self,
        n_perms: int = 5,
        base_seed: int = 100,
    ) -> None:
        """Aggregate sweep over label permutation seeds."""
        entries: list[SweepEntry] = []
        for i in range(n_perms):
            manip = self._default_manip(label_permutation_seed=base_seed + i)
            pred = self._default_pred()
            entries.append((float(i), manip, [pred]))
        self._aggregate_sweeps["lperm"] = entries

    def add_constant_label_signals(self) -> None:
        """One-shot: all context labels set to the most common value."""
        manip = self._default_manip(constant_label=True)
        pred = self._default_pred()
        self._oneshots.append(("constlbl", manip, pred))

    def add_row_duplication_sweep(
        self,
        fractions: list[float] | None = None,
    ) -> None:
        """Trajectory sweep over row duplication fractions."""
        if fractions is None:
            fractions = [0.0, 0.1, 0.25, 0.5, 1.0]
        entries: list[SweepEntry] = []
        for frac in fractions:
            manip, pred = self._corruption_specs(
                row_duplication_fraction=frac, row_duplication_seed=42
            )
            entries.append((frac, manip, [pred]))
        self._trajectory_sweeps["rowdup"] = entries

    def add_query_leakage_sweep(
        self,
        fractions: list[float] | None = None,
    ) -> None:
        """Trajectory sweep over query leakage fractions."""
        if fractions is None:
            fractions = [0.0, 0.05, 0.1, 0.2, 0.5]
        entries: list[SweepEntry] = []
        for frac in fractions:
            manip, pred = self._corruption_specs(
                query_leakage_fraction=frac, query_leakage_seed=42
            )
            entries.append((frac, manip, [pred]))
        self._trajectory_sweeps["qleak"] = entries

    def add_truth_serum_sweep(
        self,
        fractions: list[float] | None = None,
    ) -> None:
        """Trajectory sweep over truth serum fractions."""
        if fractions is None:
            fractions = [0.0, 0.1, 0.25, 0.5]
        entries: list[SweepEntry] = []
        for frac in fractions:
            manip, pred = self._corruption_specs(
                truth_serum_fraction=frac, truth_serum_seed=42
            )
            entries.append((frac, manip, [pred]))
        self._trajectory_sweeps["tserum"] = entries

    def add_brainwash_sweep(
        self,
        fractions: list[float] | None = None,
    ) -> None:
        """Trajectory sweep over brainwash fractions."""
        if fractions is None:
            fractions = [0.0, 0.05, 0.1, 0.2, 0.5]
        entries: list[SweepEntry] = []
        for frac in fractions:
            manip, pred = self._corruption_specs(
                brainwash_fraction=frac, brainwash_seed=42
            )
            entries.append((frac, manip, [pred]))
        self._trajectory_sweeps["bwash"] = entries

    def add_feature_transform_sweep(
        self,
        transforms: list[str] | None = None,
    ) -> None:
        """Aggregate sweep over feature transforms."""
        if transforms is None:
            transforms = ["log", "sqrt", "square", "reciprocal"]
        entries: list[SweepEntry] = []
        for i, tx in enumerate(transforms):
            manip = self._default_manip(feature_transform=tx)
            pred = self._default_pred()
            entries.append((float(i), manip, [pred]))
        self._aggregate_sweeps["ftx"] = entries

    def add_row_shuffle_sweep(
        self,
        n_shuffles: int = 5,
        base_seed: int = 200,
    ) -> None:
        """Aggregate sweep over row shuffle seeds (tests row-order sensitivity)."""
        entries: list[SweepEntry] = []
        for i in range(n_shuffles):
            manip = self._default_manip(row_shuffle_seed=base_seed + i)
            pred = self._default_pred()
            entries.append((float(i), manip, [pred]))
        self._aggregate_sweeps["rowshuf"] = entries

    def add_split_size_sweep(
        self,
        context_fractions: list[float] | None = None,
        n_total: int = 0,
    ) -> None:
        """Trajectory sweep over train/test split ratio via context_size.

        context_fractions are fractions of the total pool used as context.
        """
        if context_fractions is None:
            context_fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        if n_total <= 0:
            return
        entries: list[SweepEntry] = []
        for frac in context_fractions:
            ctx_size = max(2, int(n_total * frac))
            manip = self._default_manip(context_size=ctx_size)
            pred = self._default_pred()
            entries.append((frac, manip, [pred]))
        self._trajectory_sweeps["splitsize"] = entries

    def add_tabdpt_simulation_sweep(
        self,
        n_features: int,
        n_targets: int = 5,
        seed: int = 42,
    ) -> None:
        """Combined kNN retrieval + task shuffle (simulates TabDPT training)."""
        rng = np.random.default_rng(seed)
        n = min(n_targets, n_features)
        target_cols = sorted(rng.choice(n_features, size=n, replace=False))

        entries: list[SweepEntry] = []
        for i, col in enumerate(target_cols):
            manip = self._default_manip(
                target_column=int(col),
                context_selection="knn",
            )
            pred = self._default_pred()
            entries.append((float(i), manip, [pred]))
        self._aggregate_sweeps["tabdpt_sim"] = entries

    def add_nn_split_signals(self) -> None:
        """Flag to compute signals using nearest-neighbour train/test split."""
        self._nn_split = True

    # ── Compute ─────────────────────────────────────────────────────────

    def compute(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
    ) -> GridResult:
        """Run all registered sweeps, averaged over n_splits random context/query partitions.

        Each split produces a full signal vector; the final result is the mean
        across splits, plus ``*_split_std`` signals when n_splits > 1.
        """
        from sklearn.model_selection import train_test_split

        t0 = time.time()

        # Stratify the context/query split by class (classification only, when
        # every class has >=2 samples). Without this, a non-stratified split can
        # drop a class entirely from the context or the query; the model is then
        # fit / the labels encoded over MISMATCHED class sets, so predictions
        # live in a different label space than the ground truth (corrupting
        # accuracy/loss/confidence/margin). The effect is asymmetric — it hits
        # small, high-class-count tables (the CC18 non-members) hardest — so it
        # is a member-vs-non confounder, not just noise.
        strat_all = None
        if self.task_type == "classification":
            _classes, _counts = np.unique(y_all, return_counts=True)
            if len(_classes) >= 2 and _counts.min() >= 2:
                strat_all = y_all

        split_results: list[GridResult] = []
        for i in range(self.n_splits):
            seed = self.split_base_seed + i
            X_ctx, X_q, y_ctx, y_q = train_test_split(
                X_all, y_all, test_size=self.test_size, random_state=seed,
                stratify=strat_all,
            )
            if self.verbose and self.n_splits > 1:
                print(f"  Split {i + 1}/{self.n_splits} (seed={seed})")

            # Update partition tag so prediction cache distinguishes splits
            orig_partition = self.partition
            if self.n_splits > 1:
                self.partition = f"{orig_partition}_sp{seed}"

            sr = self._compute_single_split(X_ctx, y_ctx, X_q, y_q)
            split_results.append(sr)
            self.partition = orig_partition

        # Single split — return as-is
        if self.n_splits == 1:
            result = split_results[0]
            result.metadata["n_splits"] = 1
            elapsed = time.time() - t0
            result.metadata["compute_time_s"] = round(elapsed, 2)
            return result

        # Average signals across splits + compute split_std
        result = GridResult()
        result.metadata = dict(split_results[0].metadata)
        result.metadata["n_splits"] = self.n_splits

        all_signal_names: set[str] = set()
        for sr in split_results:
            all_signal_names.update(sr.signals.keys())

        for name in sorted(all_signal_names):
            vals = [sr.signals.get(name) for sr in split_results]
            valid = [v for v in vals if v is not None]
            if valid:
                result.signals[name] = float(np.mean(valid))
                result.signals[f"{name}_split_std"] = float(np.std(valid))
            else:
                result.signals[name] = None

        # Average curves across splits
        all_curve_names: set[str] = set()
        for sr in split_results:
            all_curve_names.update(sr.curves.keys())

        for cname in sorted(all_curve_names):
            all_intensities: set[float] = set()
            for sr in split_results:
                if cname in sr.curves:
                    all_intensities.update(sr.curves[cname].keys())
            avg_curve: dict[float, float | None] = {}
            for intensity in sorted(all_intensities):
                vals = [
                    sr.curves[cname].get(intensity)
                    for sr in split_results
                    if cname in sr.curves
                ]
                valid = [v for v in vals if v is not None]
                avg_curve[intensity] = float(np.mean(valid)) if valid else None
            result.curves[cname] = avg_curve

        elapsed = time.time() - t0
        result.metadata["compute_time_s"] = round(elapsed, 2)
        if self.verbose:
            print(
                f"  Grid done: {len(result.signals)} signals, "
                f"{len(result.curves)} curves across {self.n_splits} splits "
                f"in {elapsed:.1f}s"
            )

        return result

    def _compute_single_split(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        y_query: np.ndarray,
    ) -> GridResult:
        """Run all registered sweeps on one context/query split."""
        all_predictions, all_measurements = self._collect_measurements(
            X_context, y_context, X_query, y_query
        )
        result = self._derive_signals(all_predictions, all_measurements)
        result.metadata["task_type"] = self.task_type
        result.metadata["n_query"] = len(X_query)
        result.metadata["n_context"] = len(X_context)
        result.metadata["n_features"] = X_context.shape[1]

        # NN split: run base signals with NN-based train/test split
        if self._nn_split:
            nn_result = self._compute_nn_split_signals(
                np.concatenate([X_context, X_query], axis=0),
                np.concatenate([y_context, y_query], axis=0),
            )
            result.signals.update(nn_result)

        # KL-to-clean: compare each manipulated prediction to the base prediction
        self._compute_kl_to_clean(all_predictions, result)

        # Per-sample row signals — extracted from the same in-memory prediction
        # dict that produced the scalar signals above. No extra GPU work.
        try:
            result.row_signals = extract_row_signals(
                self, all_predictions, y_query, self.task_type
            )
            result.row_labels = np.asarray(y_query)
        except Exception as e:
            if self.verbose:
                print(f"  row_signals extraction failed: {e}")
            result.row_signals = {}
            result.row_labels = np.asarray(y_query)

        return result

    def _compute_nn_split_signals(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
    ) -> dict[str, float | None]:
        """Compute base signals using nearest-neighbour train/test split.

        For each point, its nearest neighbour goes to the opposite set
        (one to context, one to query).
        """
        from sklearn.neighbors import NearestNeighbors

        n = len(X_all)
        if n < 10:
            return {}

        # Handle NaN values: impute with column median for NN computation
        if np.any(np.isnan(X_all)):
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy="median")
            X_nn = imputer.fit_transform(X_all)
        else:
            X_nn = X_all

        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X_nn)
        _, indices = nn.kneighbors(X_nn)

        # Greedy assignment: alternate NN pairs between context and query
        assigned = np.full(n, -1)  # -1 = unassigned, 0 = context, 1 = query
        for i in range(n):
            if assigned[i] != -1:
                continue
            neighbor = indices[i, 1]  # nearest neighbor (index 0 is self)
            assigned[i] = 0  # context
            if assigned[neighbor] == -1:
                assigned[neighbor] = 1  # query

        # Assign any remaining unassigned points
        for i in range(n):
            if assigned[i] == -1:
                assigned[i] = 1 if np.sum(assigned == 1) < n // 3 else 0

        ctx_idx = np.where(assigned == 0)[0]
        q_idx = np.where(assigned == 1)[0]

        if len(ctx_idx) < 5 or len(q_idx) < 5:
            return {}

        X_ctx_nn, y_ctx_nn = X_all[ctx_idx], y_all[ctx_idx]
        X_q_nn, y_q_nn = X_all[q_idx], y_all[q_idx]

        # Run base prediction on NN split
        manip = self._default_manip()
        pred = self._default_pred()

        group_results = self.engine.run_group(
            X_ctx_nn, y_ctx_nn, X_q_nn, y_q_nn,
            manip, [pred],
            cache=None,  # don't cache NN splits
        )

        if self.task_type == "classification":
            y_q_enc, _ = _encode_labels(y_q_nn)
        else:
            y_q_enc = y_q_nn

        signals: dict[str, float | None] = {}
        for ps, arrays in group_results.items():
            meas = self.extractor.extract(arrays, y_q_enc)
            for mname, val in meas.items():
                signals[f"nnsplit_{mname}"] = val

        return signals

    def _compute_kl_to_clean(
        self,
        all_predictions: dict[tuple[ManipulationSpec, PredictionSpec], dict[str, np.ndarray]],
        result: GridResult,
    ) -> None:
        """Compute KL-to-clean for each trajectory sweep entry vs a clean ref.

        The clean reference is the global base prediction (default manip +
        default pred, k-NN context) for most sweeps. For the append-corruption
        sweeps (``CORRUPTION_SWEEPS``) it is instead that sweep's own
        zero-intensity entry, because those sweeps run under a capped,
        full-context regime that differs from the global base — comparing them
        to the global base would fold the regime difference into the KL and make
        the trajectory not start at zero.
        """
        # Find the global base prediction (default manip + default pred)
        base_manip = self._default_manip()
        base_pred = self._default_pred()
        base_key = (base_manip, base_pred)
        global_base_arrays = all_predictions.get(base_key)

        # For each trajectory sweep, compute KL-to-clean at each intensity
        for sweep_name, entries in self._trajectory_sweeps.items():
            if sweep_name in CORRUPTION_SWEEPS:
                # Reference = this sweep's zero-intensity (clean) prediction.
                ref_arrays = None
                for intensity, manip, preds in entries:
                    if intensity == 0.0:
                        ref_arrays = all_predictions.get((manip, preds[0]))
                        break
            else:
                ref_arrays = global_base_arrays
            if ref_arrays is None:
                continue
            base_arrays = ref_arrays

            kl_values: list[float | None] = []
            rkl_values: list[float | None] = []
            intensities = [e[0] for e in entries]
            for _, manip, preds in entries:
                ps = preds[0]
                key = (manip, ps)
                manip_arrays = all_predictions.get(key)
                if manip_arrays is None:
                    kl_values.append(None)
                    rkl_values.append(None)
                    continue
                if self.task_type == "classification":
                    kl_values.append(kl_to_clean(manip_arrays, base_arrays))
                    rkl_values.append(reverse_kl_to_clean(manip_arrays, base_arrays))
                else:
                    kl_values.append(regression_kl_to_clean(manip_arrays, base_arrays))
                    rkl_values.append(None)

            # Compute trajectory features for KL-to-clean
            prefix = f"{sweep_name}_kl2clean"
            traj = compute_trajectory(intensities, kl_values, prefix)
            result.signals.update(traj)
            result.curves[prefix] = dict(zip(intensities, kl_values))

            if self.task_type == "classification":
                prefix_r = f"{sweep_name}_rkl2clean"
                traj_r = compute_trajectory(intensities, rkl_values, prefix_r)
                result.signals.update(traj_r)
                result.curves[prefix_r] = dict(zip(intensities, rkl_values))

    def _collect_measurements(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        y_query: np.ndarray,
    ) -> tuple[
        dict[tuple[ManipulationSpec, PredictionSpec], dict[str, np.ndarray]],
        dict[tuple[ManipulationSpec, PredictionSpec], dict[str, float | None]],
    ]:
        """Run predictions and extract scalar measurements.

        Returns (all_predictions, all_measurements).
        """
        # Encode labels for measurement extraction
        if self.task_type == "classification":
            y_query_enc, _ = _encode_labels(y_query)
        else:
            y_query_enc = y_query

        # Group all ManipulationSpecs to minimize fits
        all_groups = self._collect_fit_groups()

        if self.verbose:
            n_fits = len(all_groups)
            n_preds = sum(len(ps) for ps in all_groups.values())
            print(f"  Grid: {n_fits} fits, {n_preds} predictions")

        # Run all predictions
        all_predictions: dict[
            tuple[ManipulationSpec, PredictionSpec], dict[str, np.ndarray]
        ] = {}

        for manip, pred_specs in all_groups.items():
            group_results = self.engine.run_group(
                X_context,
                y_context,
                X_query,
                y_query,
                manip,
                pred_specs,
                cache=self.cache,
                model_name=self.model_name,
                dataset_id=self.dataset_id,
                partition=self.partition,
            )
            for ps, arrays in group_results.items():
                all_predictions[(manip, ps)] = arrays

        # Extract measurements from all predictions
        all_measurements: dict[
            tuple[ManipulationSpec, PredictionSpec], dict[str, float | None]
        ] = {}
        for key, arrays in all_predictions.items():
            # Use per-prediction y_query when available (task shuffle)
            if "y_query" in arrays:
                y_q = arrays["y_query"]
                if self.task_type == "classification":
                    y_q_enc, _ = _encode_labels(y_q)
                else:
                    y_q_enc = y_q
            else:
                y_q_enc = y_query_enc

            meas = self.extractor.extract(arrays, y_q_enc)
            # Also extract logit measurements if logits are present
            if "logits" in arrays and len(arrays.get("logits", [])) > 0:
                meas.update(self.extractor.extract_logits(arrays, y_q_enc))
            all_measurements[key] = meas

        return all_predictions, all_measurements

    def _derive_signals(
        self,
        all_predictions: dict[tuple[ManipulationSpec, PredictionSpec], dict[str, np.ndarray]],
        all_measurements: dict[tuple[ManipulationSpec, PredictionSpec], dict[str, float | None]],
    ) -> GridResult:
        """Derive signals (one-shot, trajectories, aggregates, consistency) from measurements.

        Pure function over the measurement/prediction dicts.
        """
        result = GridResult()

        # ── One-shot signals ────────────────────────────────────────────
        for tag, manip, pred in self._oneshots:
            key = (manip, pred)
            if key in all_measurements:
                meas = all_measurements[key]
                for mname, val in meas.items():
                    name = f"{tag}_{mname}"
                    # Never derive blind-degenerate signals (the dataset-level
                    # logit_* geometry): the logit forward pass is still made and
                    # kept for row_signals, but its scalar signals are not
                    # comparable to the TabPFN blind. See di/degenerate_signals.py.
                    if is_blind_degenerate(name):
                        continue
                    result.signals[name] = val

        # ── Trajectory sweeps ───────────────────────────────────────────
        for sweep_name, entries in self._trajectory_sweeps.items():
            intensities = [e[0] for e in entries]

            # Collect measurement names from first valid entry
            meas_names = set()
            for _, manip, preds in entries:
                for ps in preds:
                    key = (manip, ps)
                    if key in all_measurements:
                        meas_names.update(all_measurements[key].keys())
                        break
                if meas_names:
                    break

            # For each measurement, compute trajectory
            for mname in sorted(meas_names):
                values: list[float | None] = []
                for intensity, manip, preds in entries:
                    ps = preds[0]  # trajectory sweeps have one pred per entry
                    key = (manip, ps)
                    meas = all_measurements.get(key, {})
                    values.append(meas.get(mname))

                prefix = f"{sweep_name}_{mname}"

                # Store raw curve
                curve = {}
                for intensity, val in zip(intensities, values):
                    curve[intensity] = val
                result.curves[prefix] = curve

                # Compute trajectory features
                traj = compute_trajectory(intensities, values, prefix)
                result.signals.update(traj)

        # ── Aggregate sweeps ────────────────────────────────────────────
        for sweep_name, entries in self._aggregate_sweeps.items():
            # Collect measurement names
            meas_names = set()
            for _, manip, preds in entries:
                for ps in preds:
                    key = (manip, ps)
                    if key in all_measurements:
                        meas_names.update(all_measurements[key].keys())
                        break
                if meas_names:
                    break

            for mname in sorted(meas_names):
                values: list[float | None] = []
                for _, manip, preds in entries:
                    ps = preds[0]
                    key = (manip, ps)
                    meas = all_measurements.get(key, {})
                    values.append(meas.get(mname))

                prefix = f"{sweep_name}_{mname}"
                aggs = compute_aggregates(values, prefix)
                # Never derive the blind-degenerate temperature argmax-dispersion
                # signals (temp_{accuracy,chi_squared,wasserstein}_{std,range});
                # the confidence-based temp signals and all *_mean are kept.
                result.signals.update(
                    {k: v for k, v in aggs.items() if not is_blind_degenerate(k)}
                )

        # ── Consistency sweeps ──────────────────────────────────────────
        for sweep_name, (manip, pred_specs) in self._consistency_sweeps.items():
            pred_arrays: list[np.ndarray] = []
            for ps in pred_specs:
                key = (manip, ps)
                arrays = all_predictions.get(key, {})
                preds = arrays.get("preds", np.array([]))
                if len(preds) > 0:
                    pred_arrays.append(preds)

            cons = compute_consistency(
                pred_arrays, self.task_type, f"{sweep_name}"
            )
            result.signals.update(cons)

        return result

    def _collect_fit_groups(
        self,
    ) -> dict[ManipulationSpec, list[PredictionSpec]]:
        """Collect all unique (ManipulationSpec -> [PredictionSpec]) groups."""
        groups: dict[ManipulationSpec, list[PredictionSpec]] = {}

        def add(manip: ManipulationSpec, ps: PredictionSpec):
            if manip not in groups:
                groups[manip] = []
            if ps not in groups[manip]:
                groups[manip].append(ps)

        for _, manip, pred in self._oneshots:
            add(manip, pred)

        for entries in self._trajectory_sweeps.values():
            for _, manip, preds in entries:
                for ps in preds:
                    add(manip, ps)

        for entries in self._aggregate_sweeps.values():
            for _, manip, preds in entries:
                for ps in preds:
                    add(manip, ps)

        for manip, pred_specs in self._consistency_sweeps.values():
            for ps in pred_specs:
                add(manip, ps)

        return groups
