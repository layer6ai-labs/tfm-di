"""Every LR/XGB internal cell and every external cell of the published tables, recomputed
with ``di.audit`` and compared to the reference result files on the point estimates
(``tables.POINT_KEYS`` / ``tables.EXTERNAL_KEYS``), and the feature-contribution tables
(``shap_<model>.json``) compared on the top rows.

``TFM_DATA_ROOT`` (default ``data_root``) holds the signal tables and ``TFM_REFERENCE_DIR``
(default ``exports``) the reference files; tests skip when either is missing.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from di.audit import tables
from di.audit.contributions import run_contributions
from di.audit.pools import load

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("TFM_DATA_ROOT", REPO / "data_root"))
REFERENCE_DIR = Path(os.environ.get("TFM_REFERENCE_DIR", REPO / "exports"))
CPU_KINDS = ("lr", "xgb")
MODELS = list(tables.PUBLISHED)
SHAP_MODELS = ("sap", "tabdpt")
SHAP_KEYS = ("column", "feature", "manipulation", "signal", "reduction", "target", "blind")


def _reference_files() -> set[str]:
    return {f for p in tables.PUBLISHED.values() for f in (p.internal_reference, p.external_reference)}


def _have_references() -> bool:
    return all((REFERENCE_DIR / f).exists() for f in _reference_files())


def _family_scopes() -> list[tuple[str, str]]:
    if not _have_references():
        return []
    ref = tables.reference_internal(REFERENCE_DIR)
    return sorted((m, s) for m, p in tables.PUBLISHED.items()
                  for s in ref[p.pool]["cells"] if s != p.scope)


def _family_arms() -> list[tuple[str, str]]:
    if not _have_references():
        return []
    ref = tables.reference_external(REFERENCE_DIR)
    return [(m, r["arm"]) for m, p in tables.PUBLISHED.items()
            for r in ref[p.pool] if r["arm"] != "ALL"]


@pytest.fixture(scope="session")
def reference_dir() -> Path:
    missing = [f for f in _reference_files() if not (REFERENCE_DIR / f).exists()]
    if missing:
        pytest.skip(f"reference files missing under {REFERENCE_DIR}: {missing}")
    return REFERENCE_DIR


@pytest.fixture(scope="session")
def data_root() -> Path:
    if not (DATA_ROOT / "signals").is_dir():
        pytest.skip(f"data root missing: {DATA_ROOT}")
    return DATA_ROOT


@pytest.fixture(scope="session")
def internal(data_root: Path, fast: bool) -> Callable[[str], dict]:
    cache: dict[str, dict] = {}

    def run(model: str) -> dict:
        pool = tables.PUBLISHED[model].pool
        if pool not in cache:
            cache[pool] = tables.published_internal(model, data_root, CPU_KINDS, families=not fast)
        return cache[pool]

    return run


@pytest.fixture(scope="session")
def external(data_root: Path, fast: bool) -> Callable[[str], list[dict]]:
    cache: dict[str, list[dict]] = {}

    def run(model: str) -> list[dict]:
        pool = tables.PUBLISHED[model].pool
        if pool not in cache:
            cache[pool] = tables.published_external(model, data_root, families=not fast)
        return cache[pool]

    return run


def _assert_same(got: dict | None, ref: dict | None, keys: tuple[str, ...]) -> None:
    flags = tables.compare(got, ref, keys)
    assert all(flags.values()), {k: (got and got.get(k), ref and ref.get(k)) for k, ok in flags.items() if not ok}


@pytest.mark.parametrize("model,kind", [(m, k) for m in MODELS for k in CPU_KINDS])
def test_internal_cell(internal: Callable[[str], dict], reference_dir: Path, model: str, kind: str) -> None:
    ref = tables.internal_cells(tables.reference_internal(reference_dir))[(model, kind)]
    got = tables.internal_cells({tables.PUBLISHED[model].pool: internal(model)})[(model, kind)]
    _assert_same(got, ref, tables.POINT_KEYS)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("scheme", ("noscreen", "screen"))
def test_external_cell(external: Callable[[str], list[dict]], reference_dir: Path, model: str, scheme: str) -> None:
    p = tables.PUBLISHED[model]
    ref = next(r for r in tables.reference_external(reference_dir)[p.pool] if r["arm"] == "ALL")
    got = external(model)[0]
    assert (got["n_cols"], got["target_cov"], got["blind_cov"]) == (ref["n_cols"], ref["target_cov"], ref["blind_cov"])
    _assert_same(got[scheme], ref[scheme], tables.EXTERNAL_KEYS)


@pytest.mark.parametrize("kind", CPU_KINDS)
@pytest.mark.parametrize("model,scope", _family_scopes())
def test_internal_family_cell(internal: Callable[[str], dict], reference_dir: Path, fast: bool, model: str, scope: str,
                              kind: str) -> None:
    if fast:
        pytest.skip("family cells skipped under --fast")
    p = tables.PUBLISHED[model]
    ref = tables.reference_internal(reference_dir)[p.pool]["cells"][scope]
    got = internal(model)["cells"][scope]
    assert got["n_cols"] == ref["n_cols"]
    _assert_same(got[kind], ref[kind], tables.POINT_KEYS)


@pytest.mark.parametrize("scheme", ("noscreen", "screen"))
@pytest.mark.parametrize("model,arm", _family_arms())
def test_external_family_arm(external: Callable[[str], list[dict]], reference_dir: Path, fast: bool, model: str, arm: str,
                             scheme: str) -> None:
    if fast:
        pytest.skip("family arms skipped under --fast")
    p = tables.PUBLISHED[model]
    ref = next(r for r in tables.reference_external(reference_dir)[p.pool] if r["arm"] == arm)
    got = next(r for r in external(model) if r["arm"] == arm)
    assert (got["n_cols"], got["target_cov"], got["blind_cov"]) == (ref["n_cols"], ref["target_cov"], ref["blind_cov"])
    _assert_same(got[scheme], ref[scheme], tables.EXTERNAL_KEYS)


def test_tables_render(reference_dir: Path) -> None:
    """Both tables render from the reference files in both formats with one row per model."""
    cells = tables.internal_cells(tables.reference_internal(reference_dir))
    ext = tables.external_cells(tables.reference_external(reference_dir))
    for fmt in ("md", "tex"):
        assert tables.render_internal(cells, fmt).count("\n") >= 3 * len(MODELS)
        assert tables.render_external(ext, fmt).count("\n") >= len(MODELS)
    assert "\\pmcell{" in tables.render_external(ext, "tex")
    assert "$\\pm$" in tables.render_internal(cells, "tex")


@pytest.mark.parametrize("model", SHAP_MODELS)
def test_contributions(data_root: Path, model: str) -> None:
    ref_path = REFERENCE_DIR / f"shap_{model}.json"
    if not ref_path.exists():
        pytest.skip(f"reference file missing: {ref_path}")
    p = tables.PUBLISHED[model]
    ref = json.loads(ref_path.read_text())[p.pool]
    got = run_contributions(load(p.pool, data_root), scope=p.scope, top=ref["meta"]["top"])
    assert got["meta"]["n_cols"] == ref["meta"]["n_cols"]
    for g, r in zip(got["target_top"], ref["target_top"]):
        assert {k: g[k] for k in SHAP_KEYS} == {k: r[k] for k in SHAP_KEYS}
    assert got["overlap"] == ref["overlap"] and got["spearman"] == ref["spearman"]
