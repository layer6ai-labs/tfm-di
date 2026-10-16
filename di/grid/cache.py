"""Prediction cache: npz on disk keyed by spec hierarchy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .specs import ManipulationSpec, PredictionSpec


class PredictionCache:
    """Disk cache for prediction outputs.

    Layout: {base_dir}/{model_name}/{dataset_id}_{partition}/{manip_key}/{pred_key}.npz
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def _path(
        self,
        model_name: str,
        dataset_id: int,
        partition: str,
        manip: ManipulationSpec,
        pred: PredictionSpec,
    ) -> Path:
        return (
            self.base_dir
            / model_name
            / f"{dataset_id}_{partition}"
            / manip.cache_key()
            / f"{pred.cache_key()}.npz"
        )

    def has(
        self,
        model_name: str,
        dataset_id: int,
        partition: str,
        manip: ManipulationSpec,
        pred: PredictionSpec,
    ) -> bool:
        return self._path(model_name, dataset_id, partition, manip, pred).exists()

    def load(
        self,
        model_name: str,
        dataset_id: int,
        partition: str,
        manip: ManipulationSpec,
        pred: PredictionSpec,
    ) -> dict[str, np.ndarray] | None:
        p = self._path(model_name, dataset_id, partition, manip, pred)
        if not p.exists():
            return None
        data = np.load(p, allow_pickle=False)
        return dict(data)

    def save(
        self,
        model_name: str,
        dataset_id: int,
        partition: str,
        manip: ManipulationSpec,
        pred: PredictionSpec,
        arrays: dict[str, np.ndarray],
    ) -> Path:
        p = self._path(model_name, dataset_id, partition, manip, pred)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **arrays)
        return p
