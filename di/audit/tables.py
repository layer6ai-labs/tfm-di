"""The published cells (one row per audited model), the runs behind them, table
rendering and reference comparison. Reference files are read into the layout
``run_internal``/``run_external`` produce, so every accessor works on either."""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from di.audit.external import run_external
from di.audit.internal import DEFAULT_KS, SAP_KS, run_internal
from di.audit.pools import load

KIND_LABELS = (("lr", "LR"), ("xgb", "XGB"), ("tabpfn", "TabPFN-2.5"))
POINT_KEYS = ("K", "target", "blind", "gap")
SD_KEYS = ("gap_sd", "target_sd", "blind_sd", "p_gap_le_0")
EXTERNAL_KEYS = ("target_auc", "blind_auc", "gap", "n_selected", "draws",
                 "target_auc_sd", "blind_auc_sd", "gap_sd")


@dataclass(frozen=True)
class Published:
    """One audited model: its pool, the internal (Table 2) and external (Table 1) run
    parameters, the external cell shown (``screen`` or ``noscreen``) and the reference
    result files (``run_internal`` / ``run_external`` output keyed by pool)."""
    label: str
    pool: str
    scope: str
    ks: tuple[int, ...]
    exclude: tuple[str, ...]
    external_cell: str
    internal_reference: str
    external_reference: str
    draws: int = 10


PUBLISHED: dict[str, Published] = {
    "sap": Published(label="SAP-RPT-1-OSS", pool="sap", scope="no:ctx", ks=SAP_KS, exclude=("ctx",),
                     external_cell="screen", internal_reference="internal_sap.json",
                     external_reference="external_sap.json"),
    "tabdpt": Published(label="TabDPT", pool="tabdpt", scope="ALL", ks=DEFAULT_KS, exclude=(),
                        external_cell="screen", internal_reference="internal_tabdpt.json",
                        external_reference="external_tabdpt.json"),
    "nano": Published(label="nanoTabPFN", pool="nano", scope="ALL", ks=DEFAULT_KS, exclude=(),
                      external_cell="screen", internal_reference="internal_nano.json",
                      external_reference="external_nano.json"),
    "realtabpfn": Published(label="Real-TabPFN-2.5", pool="realtabpfn", scope="ALL", ks=DEFAULT_KS,
                            exclude=(), external_cell="screen",
                            internal_reference="internal_realtabpfn.json",
                            external_reference="external_realtabpfn.json"),
}


def published_internal(model: str, data_root: str | Path, kinds: Sequence[str] = ("lr", "xgb"),
                       families: bool = True, log: Callable[[str], None] | None = None) -> dict:
    """``run_internal`` with the published parameters of ``model``."""
    p = PUBLISHED[model]
    return run_internal(load(p.pool, data_root), kinds=kinds, ks=p.ks, scope=p.scope,
                        families=families, log=log)


def published_external(model: str, data_root: str | Path, families: bool = True,
                       log: Callable[[str], None] | None = None) -> list[dict]:
    """``run_external`` with the published parameters of ``model``."""
    p = PUBLISHED[model]
    return run_external(load(p.pool, data_root), draws=p.draws, exclude=p.exclude,
                        families=families, log=log)


def load_json(paths: Sequence[str | Path]) -> dict:
    """Merge the top-level dicts of several result files."""
    out = {}
    for p in paths:
        out.update(json.loads(Path(p).read_text()))
    return out


def internal_cells(results: dict) -> dict[tuple[str, str], dict | None]:
    """Main-scope cell per (model, kind) from ``run_internal`` outputs keyed by pool."""
    out = {}
    for m, p in PUBLISHED.items():
        r = results.get(p.pool)
        if r is None:
            continue
        cells = r["cells"][p.scope]
        for k, _ in KIND_LABELS:
            if k in cells:
                out[(m, k)] = cells[k]
    return out


def external_cells(results: dict) -> dict[str, dict | None]:
    """Published cell (ALL arm, ``external_cell`` scheme) per model from ``run_external``
    outputs keyed by pool."""
    out = {}
    for m, p in PUBLISHED.items():
        rows = results.get(p.pool)
        if rows is None:
            continue
        row = next((x for x in rows if x["arm"] == "ALL"), None)
        if row is not None:
            out[m] = row[p.external_cell]
    return out


def _pm(v: float | None, sd: float, signed: bool = False, tex: bool = False) -> str:
    if v is None:
        return "---" if tex else "—"
    s = f"{abs(v):.3f}"
    sign = ("$-$" if tex else "-") if v < 0 else (("$+$" if tex else "+") if signed else "")
    return f"{sign}{s} $\\pm$ {sd:.3f}" if tex else f"{sign}{s} ± {sd:.3f}"


def _pmcell(v: float, sd: float, signed: bool = False) -> str:
    sign = "$-$" if v < 0 else ("$+$" if signed else "")
    return "\\pmcell{" + sign + f"{abs(v):.3f}" + "}{" + f"{sd:.3f}" + "}"


def render_internal(cells: dict, fmt: str = "md") -> str:
    """Table 2 (model x meta-classifier: target, blind, excess with bootstrap sd)."""
    rows = [(p.label, klabel, cells.get((m, k)))
            for m, p in PUBLISHED.items() for k, klabel in KIND_LABELS]
    if fmt == "md":
        out = ["| Target TFM | Classifier | K* | Target TFM | Blind TFM | Excess |",
               "|---|---|---:|---|---|---|"]
        for label, klabel, c in rows:
            if c is None:
                out.append(f"| {label} | {klabel} | — | — | — | — |")
                continue
            out.append(f"| {label} | {klabel} | {c['K']} | {_pm(c['target'], c['target_sd'])} | "
                       f"{_pm(c['blind'], c['blind_sd'])} | {_pm(c['gap'], c['gap_sd'], signed=True)} |")
        return "\n".join(out)
    n = len(KIND_LABELS)
    out = ["\\begin{tabular}{lllll}", "\\toprule",
           "Target TFM $f_\\theta$ & Classifier $g_\\varphi$ & \\multicolumn{1}{c}{Target TFM} & "
           "\\multicolumn{1}{c}{Blind TFM} & \\multicolumn{1}{c}{Excess}  \\\\ \\midrule"]
    for i, (label, klabel, c) in enumerate(rows):
        head = f"\\multirow{{{n}}}{{*}}{{{label}}}" if i % n == 0 else ""
        vals = ("--- & --- & ---" if c is None else
                f"{_pm(c['target'], c['target_sd'], tex=True)} & {_pm(c['blind'], c['blind_sd'], tex=True)} & "
                f"{_pm(c['gap'], c['gap_sd'], signed=True, tex=True)}")
        end = " \\\\ \\midrule" if (i % n == n - 1 and i < len(rows) - 1) else " \\\\"
        out.append(f"{head} & {klabel} & {vals}{end}")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def render_external(cells: dict, fmt: str = "md") -> str:
    """Table 1 (model: target, blind, excess with sd over draws)."""
    if fmt == "md":
        out = ["| Target TFM | Target | Blind | Excess |", "|---|---|---|---|"]
        for m, p in PUBLISHED.items():
            c = cells.get(m)
            if c is None:
                out.append(f"| {p.label} | — | — | — |")
                continue
            out.append(f"| {p.label} | {_pm(c['target_auc'], c['target_auc_sd'])} | "
                       f"{_pm(c['blind_auc'], c['blind_auc_sd'])} | {_pm(c['gap'], c['gap_sd'], signed=True)} |")
        return "\n".join(out)
    out = ["\\begin{tabular}{lccc}", "\\toprule", "Target TFM & Target & Blind & Excess \\\\", "\\midrule"]
    for m, p in PUBLISHED.items():
        c = cells.get(m)
        vals = ("--- & --- & ---" if c is None else
                f"{_pmcell(c['target_auc'], c['target_auc_sd'])} & {_pmcell(c['blind_auc'], c['blind_auc_sd'])} & "
                f"{_pmcell(c['gap'], c['gap_sd'], signed=True)}")
        out.append(f"{p.label} & {vals} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def compare(result: dict | None, reference: dict | None, keys: Sequence[str]) -> dict[str, bool]:
    """Per key, whether ``result[key] == reference[key]``; a missing cell on either side
    compares ``None`` against ``None``."""
    if result is None or reference is None:
        return {k: result is None and reference is None for k in keys}
    return {k: result.get(k) == reference.get(k) for k in keys}


def reference_internal(reference_dir: str | Path) -> dict[str, dict]:
    """Reference ``run_internal`` outputs keyed by pool."""
    d = Path(reference_dir)
    return {p.pool: json.loads((d / p.internal_reference).read_text())[p.pool]
            for p in PUBLISHED.values()}


def reference_external(reference_dir: str | Path) -> dict[str, list[dict]]:
    """Reference ``run_external`` outputs keyed by pool."""
    d = Path(reference_dir)
    return {p.pool: json.loads((d / p.external_reference).read_text())[p.pool]
            for p in PUBLISHED.values()}


def render_contributions(result: dict, fmt: str = "md") -> str:
    """Table 3 / Table 8 (top features of the target: paper name, M/S/R codes, target and
    blind mean |TreeSHAP|) from one ``run_contributions`` output."""
    rows = result["target_top"]
    if fmt == "md":
        out = ["| Fingerprint Feature | Manipulation | Signal | Reduction | Target | Blind |",
               "|---|---|---|---|---:|---:|"]
        out += [f"| {r['feature']} | {r['manipulation']} | {r['signal']} | {r['reduction']} | "
                f"{r['target']:.4f} | {r['blind']:.4f} |" for r in rows]
        return "\n".join(out)
    out = ["\\begin{tabular}{lccccc}", "\\toprule",
           "Fingerprint Feature & Manipulation & Signal & Reduction & Target & Blind \\\\", "\\midrule"]
    out += [f"{r['feature'].replace('_', chr(92) + '_')} & {r['manipulation']} & {r['signal']} & "
            f"{r['reduction']} & {r['target']:.4f} & {r['blind']:.4f} \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)
