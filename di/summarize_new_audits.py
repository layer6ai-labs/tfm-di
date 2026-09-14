"""Render the new-signal audit JSONs into one markdown report.

  python di/summarize_new_audits.py [--out exports/NEW_SIGNAL_AUDITS.md]
"""
import argparse
import glob
import json
import os
from pathlib import Path

REPO = Path(os.environ.get("AUDIT_REPO", "/p/project1/hai_1159/tfm-di")).resolve()
EXPORTS = REPO / "exports"

TITLES = {
    "tabdpt_redteam": "TabDPT — H13 red-team probes (SHARP / PSTAB / DEJAVU / ANTIKNN)",
    "sap_redteam": "SAP RPT-OSS — H14 semantic-channel probes",
    "sap_fullgrid": "SAP RPT-OSS — full grid (incl. new families)",
    "tabdpt_fullgrid": "TabDPT — full grid (4 seeds; control + new families)",
}


def fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    return f"{v:+.{nd}f}" if isinstance(v, float) and abs(v) < 1 and v < 0 else f"{v:.{nd}f}"


def internal_table(scope_key, blk):
    rows = ["| auditor | selection | target AUC | blind AUC | gap | null q95 | p | flag |",
            "|---|---|---|---|---|---|---|---|"]
    for key, c in sorted(blk["cells"].items()):
        model, kind = key.split("|")
        sel = "target-only"
        p = c.get("p_gap")
        note = c.get("null") if isinstance(c.get("null"), str) else ""
        warn = "⚠ fold-flip" if "WARN" in c else ""
        rows.append(f"| {model} / {kind.upper()} | {sel} | {c['target_auc']:.3f} | "
                    f"{c['blind_auc']:.3f} | {c['gap']:+.3f} | "
                    f"{fmt(c.get('null_q95'))} | {fmt(p) if p is not None else note or '—'} | "
                    f"{warn} |")
    return "\n".join(rows)


def external_table(blk):
    rows = ["| model | AUC (±sd) | TPR@5%FPR (emp) | TPR@10%FPR (emp) |",
            "|---|---|---|---|"]
    for m, v in blk["models"].items():
        rows.append(f"| {m} | {v['auc_mean']:.3f} ± {v['auc_sd']:.3f} | "
                    f"{v.get('tpr@0.05', float('nan')):.2f} ({v.get('emp_fpr@0.05', float('nan')):.2f}) | "
                    f"{v.get('tpr@0.1', float('nan')):.2f} ({v.get('emp_fpr@0.1', float('nan')):.2f}) |")
    out = "\n".join(rows)
    nul = blk.get("null")
    out += (f"\n\ngap (target mean − blind) = **{blk['gap']:+.3f}**")
    if nul:
        out += (f"; shuffled-membership null: gap mean {nul['gap_null_mean']:+.3f}, "
                f"q95 {nul['gap_null_q95']:+.3f} → **p = {nul['p_gap']:.3f}** "
                f"(auditor AUC p = {nul['p_auc']:.3f}, {nul['nperm']} draws)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EXPORTS / "NEW_SIGNAL_AUDITS.md"))
    ap.add_argument("--glob", default="hyp_newsignal_*.json")
    a = ap.parse_args()

    files = sorted(glob.glob(str(EXPORTS / a.glob)))
    # One signal set can be split over several files (LR/XGB run, TabPFN run,
    # extra scopes) — merge them back into a single section per set.
    merged, sources = {}, {}
    for f in files:
        for set_name, rec in json.load(open(f))["sets"].items():
            # base / masked / complete are different pools — never merge them
            imp = rec.get("impute", "base")
            name = set_name if imp == "base" else f"{set_name} [{imp}]"
            sources.setdefault(name, []).append(Path(f).name)
            tgt = merged.setdefault(name, {k: v for k, v in rec.items()
                                           if k not in ("internal", "external", "per_signal")})
            for section in ("internal", "external", "per_signal"):
                dst = tgt.setdefault(section, {})
                for sc, blk in rec.get(section, {}).items():
                    if section == "internal" and sc in dst:
                        dst[sc]["cells"].update(blk["cells"])
                    else:
                        dst.setdefault(sc, blk)
    doc = ["# Submission DI pipeline on the new TabDPT and SAP signals", "",
           "Two audits, both run per signal set:", "",
           "* **Internal audit** — supervised nested-LOOCV auditor (inner 5-fold CV picks "
           "top-K by TreeSHAP; LR / XGBoost / TabPFN-2.5 meta-classifier; tie-aware folded "
           "AUC). The auditor never consults the blind: the ranking, K* and every fit come "
           "from the target and the membership labels alone, and the blind is then scored "
           "on exactly the columns the auditor selected. Significance from a "
           "selection-aware shuffled-label permutation null.",
           "* **External audit** — label-free outside auditor (H11): only known non-members "
           "+ a suspect pile; direction from the PU shift of the unlabeled pile; split-conformal "
           "p-values for calibrated detection. The blind control is the identical auditor run "
           "on the public model's own signals.", "",
           "`gap = target AUC − blind AUC`. Size (`n_pool`) is an allowed indicator and is "
           "never residualised; it is excluded from the signal pools themselves (`META_KEYS`), "
           "matching the submission.", ""]

    # ------------------------------------------------ headline: does it survive?
    def best(rec, section):
        """Best meta-classifier per scope. Multi-seed targets are AVERAGED over
        seeds first — taking the max over seeds would be winner's curse."""
        out = {}
        for sc, blk in rec.get(section, {}).items():
            if section != "internal":
                out[sc] = (blk["gap"], (blk.get("null") or {}).get("p_gap"), "")
                continue
            byclf = {}
            for k, c in blk["cells"].items():
                byclf.setdefault(k.split("|")[1], []).append(c)
            for kind, cs in byclf.items():
                gap = sum(c["gap"] for c in cs) / len(cs)
                ps = [c["p_gap"] for c in cs if isinstance(c.get("p_gap"), float)]
                # per-seed p-values: report the median, not the luckiest one
                p = sorted(ps)[len(ps) // 2] if ps else None
                if sc not in out or gap > out[sc][0]:
                    out[sc] = (gap, p, kind)
        return out

    doc += ["## Headline — does any gap survive the missing-data control?", "",
            "The blind is much patchier than the targets, so `base` (each matrix median-"
            "imputed on its own) systematically favours the target. `masked` handicaps the "
            "target down to the blind's coverage. A gap that does not survive is not "
            "established.", "",
            "| set | scope | audit | base gap (p) | masked gap (p) |", "|---|---|---|---|---|"]
    for name, rec in merged.items():
        if "[" in name:
            continue
        mrec = merged.get(f"{name} [masked]")
        if not mrec:
            continue
        for section, label in (("internal", "internal"), ("external", "external")):
            b, m = best(rec, section), best(mrec, section)
            for sc in sorted(set(b) & set(m)):
                pb = f"{b[sc][1]:.3f}" if isinstance(b[sc][1], float) else "—"
                pm = f"{m[sc][1]:.3f}" if isinstance(m[sc][1], float) else "—"
                doc.append(f"| {name} | `{sc}` | {label} | {b[sc][0]:+.3f} ({pb}) | "
                           f"{m[sc][0]:+.3f} ({pm}) |")
    doc += ["", "Internal cells report the best-performing meta-classifier for that scope.", ""]

    for name, rec in merged.items():
        if True:
            doc += [f"## {TITLES.get(name, name)}", "",
                    f"n = {rec['n']} ({rec['n_member']} member / "
                    f"{rec['n_non']} non-member), {rec['n_signals']} signals, "
                    f"models: {', '.join(rec['models'])}  \n"
                    f"sources: {', '.join('`' + s + '`' for s in sources[name])}",
                    "", rec["notes"], ""]
            for sc, blk in rec.get("internal", {}).items():
                doc += [f"### Internal audit — scope `{sc}` ({blk['n_signals']} signals, "
                        f"outer CV: {blk.get('outer_cv', 'loo')})", "",
                        internal_table(sc, blk), ""]
            for sc, blk in rec.get("external", {}).items():
                doc += [f"### External audit — scope `{sc}` ({blk['n_signals']} signals, "
                        f"{blk['nsplits']} known-non/suspect draws)", "",
                        external_table(blk), ""]
            for sc, ps in rec.get("per_signal", {}).items():
                top = ps["top"][:8]
                doc += [f"<details><summary>Per-signal target−blind, scope <code>{sc}</code> "
                        f"(optimistic, in-sample selection): mean gap {ps['mean_gap']:+.3f}, "
                        f"{100*ps['frac_positive']:.0f}% positive</summary>", "",
                        "| signal | target | blind | gap |", "|---|---|---|---|"]
                doc += [f"| `{r['signal']}` | {r['target']:.3f} | {r['blind']:.3f} | "
                        f"{r['gap']:+.3f} |" for r in top]
                doc += ["", "</details>", ""]
    Path(a.out).write_text("\n".join(doc))
    print(f"wrote {a.out} ({len(files)} files)")


if __name__ == "__main__":
    main()
