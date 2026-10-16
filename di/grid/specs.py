"""Frozen, hashable specification dataclasses for the  grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ManipulationSpec:
    """Specifies how to manipulate context data before fitting.

    Each unique ManipulationSpec requires a separate model.fit() call.
    """

    context_size: int | None = None
    column_count: int | None = None  # number of features to keep
    column_seed: int = 0  # seed for column subsampling
    noise_level: float = 0.0
    noise_seed: int = 0
    mislabel_fraction: float = 0.0
    mislabel_strategy: Literal["random", "swap", "boundary"] = "random"
    mislabel_seed: int = 0
    normalizer: str = "standard"
    context_selection: Literal["sequential", "random", "knn"] = "random"
    subsample_seed: int = 42
    target_column: int | None = None  # index into [X | y] full matrix; swaps this column with y
    label_permutation_seed: int | None = None  # permute class label encoding
    constant_label: bool = False  # set all context labels to single value
    row_duplication_fraction: float = 0.0  # fraction of context rows to duplicate
    row_duplication_seed: int = 0
    query_leakage_fraction: float = 0.0  # fraction of query rows to copy into context
    query_leakage_seed: int = 0
    truth_serum_fraction: float = 0.0  # fraction of context rows to duplicate with perturbed y
    truth_serum_seed: int = 0
    brainwash_fraction: float = 0.0  # fraction of query rows to copy into context with perturbed y
    brainwash_seed: int = 0
    feature_transform: str | None = None  # "log", "sqrt", "square", "reciprocal"
    context_equals_query: bool = False  # predict on context data itself
    row_shuffle_seed: int | None = None  # shuffle context row order

    def cache_key(self) -> str:
        """Deterministic string key for caching."""
        parts = []
        if self.target_column is not None:
            parts.append(f"tgt{self.target_column}")
        if self.context_size is not None:
            parts.append(f"ctx{self.context_size}")
        if self.column_count is not None:
            parts.append(f"col{self.column_count}_s{self.column_seed}")
        if self.noise_level > 0:
            parts.append(f"noise{self.noise_level}_s{self.noise_seed}")
        if self.mislabel_fraction > 0:
            parts.append(
                f"mislbl{self.mislabel_fraction}_{self.mislabel_strategy}_s{self.mislabel_seed}"
            )
        if self.normalizer != "standard":
            parts.append(f"norm_{self.normalizer}")
        if self.context_selection != "random":
            parts.append(f"sel_{self.context_selection}")
        if self.subsample_seed != 42:
            parts.append(f"ss{self.subsample_seed}")
        if self.label_permutation_seed is not None:
            parts.append(f"lperm_s{self.label_permutation_seed}")
        if self.constant_label:
            parts.append("constlbl")
        if self.row_duplication_fraction > 0:
            parts.append(f"rowdup{self.row_duplication_fraction}_s{self.row_duplication_seed}")
        if self.query_leakage_fraction > 0:
            parts.append(f"qleak{self.query_leakage_fraction}_s{self.query_leakage_seed}")
        if self.truth_serum_fraction > 0:
            parts.append(f"tserum{self.truth_serum_fraction}_s{self.truth_serum_seed}")
        if self.brainwash_fraction > 0:
            parts.append(f"bwash{self.brainwash_fraction}_s{self.brainwash_seed}")
        if self.feature_transform is not None:
            parts.append(f"ftx_{self.feature_transform}")
        if self.context_equals_query:
            parts.append("ctxeqq")
        if self.row_shuffle_seed is not None:
            parts.append(f"rowshuf_s{self.row_shuffle_seed}")
        return "_".join(parts) if parts else "default"


@dataclass(frozen=True)
class PredictionSpec:
    """Specifies how to run predict (shares an existing model.fit()).

    Different PredictionSpecs with the same ManipulationSpec share one fit.
    """

    temperature: float = 0.8
    seed: int | None = None  # feature permutation seed
    return_logits: bool = False
    context_size: int | None = None  # predict-time context_size (k-NN retrieval)
    n_ensembles: int = 4

    def cache_key(self) -> str:
        parts = [f"T{self.temperature}"]
        if self.seed is not None:
            parts.append(f"seed{self.seed}")
        if self.return_logits:
            parts.append("logits")
        if self.context_size is not None:
            parts.append(f"pctx{self.context_size}")
        if self.n_ensembles != 4:
            parts.append(f"ens{self.n_ensembles}")
        return "_".join(parts)


@dataclass(frozen=True)
class ModelSpec:
    """Identifies a model checkpoint and its ground-truth membership set."""

    name: str  # e.g. "tabdpt_default", "tabdpt_split1"
    backend: str = "tabdpt"  # "tabdpt" or "tabpfn25"
    weight_file: str | None = None
    member_ids: frozenset[int] = field(default_factory=frozenset)

    def is_member(self, dataset_id: int) -> bool:
        return dataset_id in self.member_ids


# Default model specs
def default_model_spec() -> ModelSpec:
    from ..datasets.tabdpt_training_ids import MEMBER_IDS_TABDPT

    return ModelSpec(
        name="tabdpt_default",
        backend="tabdpt",
        weight_file=None,
        member_ids=frozenset(MEMBER_IDS_TABDPT),
    )
