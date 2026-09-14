"""DI Grid: Systematic Manipulation x Measurement signal computation."""

from .specs import ManipulationSpec, PredictionSpec, ModelSpec, default_model_spec
from .cache import PredictionCache
from .predictions import PredictionEngine
from .manipulations import ManipulationState
from .measurements import MeasurementExtractor
from .trajectories import compute_trajectory, compute_aggregates, compute_consistency
from .row_signals import extract_row_signals
from .grid import SignalGrid, GridResult

__all__ = [
    "ManipulationSpec",
    "ManipulationState",
    "PredictionSpec",
    "ModelSpec",
    "default_model_spec",
    "PredictionCache",
    "PredictionEngine",
    "MeasurementExtractor",
    "SignalGrid",
    "GridResult",
    "compute_trajectory",
    "compute_aggregates",
    "compute_consistency",
    "extract_row_signals",
]
