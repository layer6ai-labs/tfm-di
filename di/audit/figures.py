"""The result figures of the paper, drawn from the same runs as the tables: the
single-signal and multi-signal separation histograms (Figures 1, 3 and the TabDPT
appendix figure), the internal-audit AUC bars (Figure 4) and the external-audit p-value
histograms (Figure 2). Every file is a despined PDF without title; legends are written to
separate ``*_legend_<n>col.pdf`` files."""
from __future__ import annotations

import json
import zlib
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

from di.audit.external import SCHEMES, _one_model
from di.audit.internal import audit_cell, cell_inputs
from di.audit.pools import PoolData, fam_of, load, select
from di.audit.tables import PUBLISHED, internal_cells, reference_internal

MEMBER, NON_MEMBER = "#0072B2", "#D55E00"
MODEL_COLORS = {"sap": "#009E73", "tabdpt": "#0072B2", "nano": "#D55E00", "realtabpfn": "#CC79A7"}
RC = {"figure.figsize": (2.5, 2.5), "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
      "ytick.labelsize": 8, "legend.fontsize": 8, "savefig.format": "pdf", "savefig.bbox": "tight",
      "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": False}


def _legend_variants(handles: list, labels: list[str], stem: Path) -> None:
    n = len(labels)
    for ncol in sorted({1, 2, min(3, n), n}):
        fig = plt.figure(figsize=(1.2 * ncol, 0.3 * ((n + ncol - 1) // ncol)))
        fig.legend(handles, labels, loc="center", frameon=False, ncol=ncol)
        fig.savefig(f"{stem}_legend_{ncol}col.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def _save(fig: matplotlib.figure.Figure, path: Path, legend: bool = True) -> None:
    """Despine, drop in-figure legends and titles, save; export the legend separately."""
    handles, labels = [], []
    for ax in fig.axes:
        ax.set_title("")
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
        if ax.get_legend():
            ax.get_legend().remove()
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path)
    if legend and handles:
        _legend_variants(handles, labels, path.with_suffix(""))
    plt.close(fig)


def _auc(member: np.ndarray, non: np.ndarray) -> float:
    """Orientation-free AUC of one signal."""
    m, n = member[np.isfinite(member)], non[np.isfinite(non)]
    a = roc_auc_score(np.r_[np.ones(len(m)), np.zeros(len(n))], np.r_[m, n])
    return max(a, 1 - a)


def hist_pair(ax: matplotlib.axes.Axes, member: np.ndarray, non: np.ndarray, xlabel: str,
              log_x: bool = False, bins: int = 40, clip: tuple[float, float] = (0.5, 99.5),
              edges: np.ndarray | None = None, auc_box: bool = True) -> float:
    """Overlaid per-class-normalised histograms of a per-dataset value; returns the AUC."""
    vals = np.r_[member, non]
    vals = vals[np.isfinite(vals)]
    if edges is None:
        lo, hi = np.percentile(vals, clip)
        if log_x:
            lo = max(lo, 1e-8)
            edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
        else:
            edges = np.linspace(lo, hi, bins + 1)
    lo, hi = edges[0], edges[-1]
    for v, color, label in ((non, NON_MEMBER, "non-member"), (member, MEMBER, "member")):
        v = np.clip(v[np.isfinite(v)], lo, hi)
        ax.hist(v, bins=edges, weights=np.full(len(v), 1.0 / len(v)), alpha=0.55, color=color,
                label=label, edgecolor="none")
    auc = _auc(member, non)
    if auc_box:
        ax.text(0.97, 0.97, f"AUC={auc:.3f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                      edgecolor="0.6", linewidth=0.6))
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of datasets")
    return auc


def internal_scores(pool: PoolData, model: str, kind: str = "lr", folds: int = 5,
                    seed: int = 0) -> dict:
    """The published internal cell of ``model`` for ``kind`` with its pooled out-of-fold
    target and blind scores at K*."""
    p = PUBLISHED[model]
    inputs = cell_inputs(pool, select(pool.signals, p.scope), folds)
    if inputs is None:
        raise ValueError(f"{model}: no usable column in scope {p.scope}")
    Xt, Xb, fold_list, ranks = inputs
    rng = np.random.default_rng([seed, zlib.crc32(f"{pool.name}|{p.scope}|{kind}".encode())])
    return audit_cell(Xt, Xb, pool.y, kind, p.ks, fold_list, ranks, rng, scores=True)


def external_pvalues(pool: PoolData, model: str, seed: int = 7) -> tuple[dict, dict]:
    """Conformal p-values of the suspects on one draw of the published external audit
    (``seed`` = the first of the published draws): the target, and the blind on the
    target's screened columns."""
    drop = set(PUBLISHED[model].exclude)
    cols = np.array([fam_of(s) not in drop for s in pool.signals])
    a = _one_model(pool.Xt[:, cols], pool.y, SCHEMES["screen"], np.random.default_rng(seed))
    if a is None:
        raise ValueError(f"{model}: the screen kept fewer than two columns on draw {seed}")
    b = _one_model(pool.Xb[:, cols], pool.y, SCHEMES["screen"], np.random.default_rng(seed),
                   columns=a["columns"])
    return a, b


def plot_signal_separation(pool: PoolData, cell: dict, out_dir: Path, stem: str) -> dict:
    """``<stem>_loss``, ``<stem>_confidence`` (single signals of the unmanipulated probe)
    and ``<stem>_multisig`` (meta-classifier P(member) at K*)."""
    y = pool.y == 1
    col = {s: j for j, s in enumerate(pool.signals)}
    loss, conf = pool.Xt[:, col["base_loss"]], pool.Xt[:, col["base_correct_confidence"]]
    scores = np.asarray(cell["scores"]["target"])
    out = {}
    fig, ax = plt.subplots()
    out["loss_auc"] = hist_pair(ax, loss[y], loss[~y], "avg. loss", log_x=True)
    _save(fig, out_dir / f"{stem}_loss.pdf")
    fig, ax = plt.subplots()
    out["confidence_auc"] = hist_pair(ax, conf[y], conf[~y], "avg. correct confidence")
    _save(fig, out_dir / f"{stem}_confidence.pdf", legend=False)
    fig, ax = plt.subplots()
    out["multisig_auc"] = hist_pair(ax, scores[y], scores[~y], "LR $P$(member)",
                                    edges=np.linspace(0, 1, 41), auc_box=False)
    _save(fig, out_dir / f"{stem}_multisig.pdf", legend=False)
    out["K"], out["target"], out["blind"] = cell["K"], cell["target"], cell["blind"]
    return out


def plot_pvalues(target: dict, blind: dict, out_dir: Path, stem: str, bins: int = 24) -> dict:
    """``<stem>_target_log`` and ``<stem>_blind_log``: p-value histograms of member and
    non-member suspects on a log axis with the uniform null."""
    out = {}
    for side, r in (("target", target), ("blind", blind)):
        pv, yt = np.asarray(r["pvals"]), np.asarray(r["yt"]) == 1
        floor = max(float(pv.min()), 1e-4)
        edges = np.logspace(np.log10(floor), 0, bins + 1)
        fig, ax = plt.subplots(figsize=(2.5, 2.3))
        hist_pair(ax, pv[yt], pv[~yt], f"empirical p-value ({side}, log scale)", log_x=True,
                  edges=edges, auc_box=False)
        ax.plot(0.5 * (edges[1:] + edges[:-1]), np.diff(edges), color="grey", linestyle=":",
                linewidth=0.7, label="uniform null")
        ax.set_xlim(floor, 1.0)
        _save(fig, out_dir / f"{stem}_{side}_log.pdf", legend=side == "target")
        out[side] = {"auc": round(float(r["auc"]), 4), "n_selected": int(r["nsel"]),
                     "n_member": int(yt.sum()), "n_non_member": int((~yt).sum())}
    return out


def plot_auc_bars(reference_dir: Path, out_dir: Path, stem: str, kind: str = "tabpfn") -> dict:
    """Target (solid) and blind (hatched) internal-audit AUC of every published model for
    one meta-classifier, read from the reference files."""
    cells = internal_cells(reference_internal(reference_dir))
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    out = {}
    for i, (m, p) in enumerate(PUBLISHED.items()):
        c = cells[(m, kind)]
        color = MODEL_COLORS[m]
        ax.bar(i - 0.2, c["target"], 0.4, color=color)
        ax.bar(i + 0.2, c["blind"], 0.4, facecolor="white", edgecolor=color, hatch="//", linewidth=1.0)
        out[m] = {"target": c["target"], "blind": c["blind"]}
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xticks(range(len(PUBLISHED)))
    ax.set_xticklabels([p.label for p in PUBLISHED.values()], rotation=20, ha="right")
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Dataset-level AUC")
    handles = [Patch(facecolor=MODEL_COLORS["sap"], label="Target signals"),
               Patch(facecolor="white", edgecolor=MODEL_COLORS["sap"], hatch="//", label="Blind (TabPFN-2.5)")]
    for a in fig.axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_dir / f"{stem}.pdf")
    _legend_variants(handles, [h.get_label() for h in handles], out_dir / stem)
    plt.close(fig)
    return out


def make_all(data_root: str | Path, reference_dir: str | Path, out_dir: str | Path,
             log: Callable[[str], None] | None = None) -> dict:
    """Every result figure of the paper into ``out_dir`` plus ``figure_data.json`` with the
    AUCs behind each panel."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update(RC)
    data = {"auc_bars": plot_auc_bars(Path(reference_dir), out_dir, "fig_attack_dia_auc_2")}
    if log:
        log(f"auc bars: {data['auc_bars']}")
    for model, stem in (("sap", "fig_signal_separation"), ("tabdpt", "fig_signal_separation_tabdpt")):
        pool = load(PUBLISHED[model].pool, data_root)
        cell = internal_scores(pool, model, "lr")
        data[stem] = plot_signal_separation(pool, cell, out_dir, stem)
        if log:
            log(f"{stem}: {data[stem]}")
        if model == "sap":
            t, b = external_pvalues(pool, model)
            data["fig_pvalue_sap"] = plot_pvalues(t, b, out_dir, "fig_pvalue_sap")
            if log:
                log(f"fig_pvalue_sap: {data['fig_pvalue_sap']}")
    (out_dir / "figure_data.json").write_text(json.dumps(data, indent=1))
    return data
