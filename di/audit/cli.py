"""Command line: ``python -m di.audit {internal,external,shap,tables,reproduce,figures}``."""
from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Sequence
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

from di.audit import tables
from di.audit.contributions import run_contributions
from di.audit.external import run_external
from di.audit.figures import make_all
from di.audit.internal import DEFAULT_KS, run_internal
from di.audit.pools import POOLS, load
from di.audit.tables import PUBLISHED

DATA_ROOT_DEFAULT = "data_root"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write(path: str | None, obj: object) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj, indent=1))
        _log(f"wrote {path}")


def cmd_internal(a: argparse.Namespace) -> None:
    out = {}
    for name in a.pool.split(","):
        pool = load(name, a.data_root, a.target)
        out[name] = run_internal(pool, kinds=tuple(a.kinds.split(",")),
                                 ks=[int(k) for k in a.ks.split(",")], scope=a.scope,
                                 folds=a.folds, nboot=a.nboot, seed=a.seed, device=a.device,
                                 families=not a.no_families, min_family=a.min_family, log=_log)
    _write(a.out, out)


def cmd_external(a: argparse.Namespace) -> None:
    out = {}
    for name in a.pool.split(","):
        pool = load(name, a.data_root)
        out[name] = run_external(pool, draws=a.draws, exclude=[x for x in a.exclude.split(",") if x],
                                 seed0=a.seed, families=not a.no_families, log=_log)
    _write(a.out, out)


def cmd_shap(a: argparse.Namespace) -> None:
    out = {}
    for name in a.pool.split(","):
        pool = load(name, a.data_root, a.target)
        out[name] = run_contributions(pool, scope=a.scope, folds=a.folds, top=a.top, log=_log)
    _write(a.out, out)
    if a.format:
        for name, r in out.items():
            print(tables.render_contributions(r, a.format))


def cmd_figures(a: argparse.Namespace) -> None:
    make_all(a.data_root, a.reference_dir, a.out_dir, log=_log)
    _log(f"wrote figures to {a.out_dir}")


def cmd_tables(a: argparse.Namespace) -> None:
    parts = []
    if a.internal:
        parts.append(tables.render_internal(tables.internal_cells(tables.load_json(a.internal)), a.format))
    if a.external:
        parts.append(tables.render_external(tables.external_cells(tables.load_json(a.external)), a.format))
    for r in tables.load_json(a.shap).values():
        parts.append(tables.render_contributions(r, a.format))
    text = "\n\n".join(parts)
    if a.out:
        Path(a.out).write_text(text + "\n")
        _log(f"wrote {a.out}")
    else:
        print(text)


def _verdict(flags: dict[str, bool]) -> str:
    bad = [k for k, ok in flags.items() if not ok]
    return "identical" if not bad else "DIFFERS on " + ",".join(bad)


def cmd_reproduce(a: argparse.Namespace) -> None:
    models = a.models.split(",")
    if a.which in ("internal", "both"):
        ref = tables.internal_cells(tables.reference_internal(a.reference_dir))
        out = {}
        for m in models:
            p = PUBLISHED[m]
            out[p.pool] = tables.published_internal(m, a.data_root, families=False, log=_log)
            for (model, kind), cell in tables.internal_cells(out).items():
                if model == m:
                    _log(f"internal {m:11s} {kind:4s} point estimates: "
                         f"{_verdict(tables.compare(cell, ref[(model, kind)], tables.POINT_KEYS))}; "
                         f"bootstrap sd: {_verdict(tables.compare(cell, ref[(model, kind)], tables.SD_KEYS))}")
        _write(a.out_internal, out)
    if a.which in ("external", "both"):
        ref = tables.reference_external(a.reference_dir)
        out = {}
        for m in models:
            p = PUBLISHED[m]
            rows = tables.published_external(m, a.data_root, families=False, log=_log)
            out[p.pool] = rows
            ref_row = next(x for x in ref[p.pool] if x["arm"] == "ALL")
            for key in ("noscreen", "screen"):
                _log(f"external {m:11s} {key:8s}: "
                     f"{_verdict(tables.compare(rows[0][key], ref_row[key], tables.EXTERNAL_KEYS))}")
        _write(a.out_external, out)


def main(argv: Sequence[str] | None = None) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    ap = argparse.ArgumentParser(prog="python -m di.audit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("internal", help="labelled top-K audit of one or more pools")
    p.add_argument("--pool", required=True, help=f"comma list from {','.join(POOLS)}")
    p.add_argument("--kinds", default="lr,xgb", help="comma list from lr,xgb,tabpfn")
    p.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)))
    p.add_argument("--scope", default="ALL", help="ALL, a family union a+b, or no:a+b")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--nboot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-family", type=int, default=5)
    p.add_argument("--no-families", action="store_true", help="only the --scope cell")
    p.add_argument("--target", default="default", help="target weights variant (tabdpt: seed42, seed123, seed456)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_internal)

    p = sub.add_parser("external", help="label-free IsolationForest audit of one or more pools")
    p.add_argument("--pool", required=True, help=f"comma list from {','.join(POOLS)}")
    p.add_argument("--draws", type=int, default=10)
    p.add_argument("--exclude", default="", help="comma list of families removed from every arm")
    p.add_argument("--seed", type=int, default=7, help="draw i uses default_rng(seed + i)")
    p.add_argument("--no-families", action="store_true", help="only the ALL arm")
    p.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_external)

    p = sub.add_parser("shap", help="feature contributions (Table 3 / Table 8): fold-averaged "
                                    "mean |TreeSHAP| of the ranking fits, target and blind")
    p.add_argument("--pool", required=True, help=f"comma list from {','.join(POOLS)}")
    p.add_argument("--scope", default="ALL", help="ALL, a family union a+b, or no:a+b")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--target", default="default", help="target weights variant (tabdpt: seed42, seed123, seed456)")
    p.add_argument("--format", choices=("md", "tex"), default=None, help="also print the top table")
    p.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_shap)

    p = sub.add_parser("tables", help="render Table 1 / Table 2 / Table 3 from result files")
    p.add_argument("--internal", nargs="*", default=[], help="outputs of `internal`")
    p.add_argument("--external", nargs="*", default=[], help="outputs of `external`")
    p.add_argument("--shap", nargs="*", default=[], help="outputs of `shap`")
    p.add_argument("--format", choices=("md", "tex"), default="md")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_tables)

    p = sub.add_parser("figures", help="the result figures of the paper (PDF + separate legends)")
    p.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--reference-dir", default="tests/reference", help="internal reference files for the AUC bars")
    p.add_argument("--out-dir", default="figures")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("reproduce", help="re-run the LR/XGB and external cells of the published "
                                         "table and compare them to the reference files")
    p.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--which", choices=("internal", "external", "both"), default="both")
    p.add_argument("--models", default=",".join(PUBLISHED), help=f"comma list from {','.join(PUBLISHED)}")
    p.add_argument("--out-internal", default=None, help="result file in the layout of `internal`")
    p.add_argument("--out-external", default=None, help="result file in the layout of `external`")
    p.set_defaults(func=cmd_reproduce)

    a = ap.parse_args(argv)
    a.func(a)
