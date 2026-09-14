"""External (label-free) iForest audit -> the tables.

For each corpus, one row per arm (ALL signals, then each signal family), scored
twice: without the informativeness screen and with it.

  uv run di/ifsel/audit_table.py                       # SAP union, TabDPT, control
  uv run di/ifsel/audit_table.py --corpora tabdpt --draws 20

Threat model. The auditor holds a pile of tables they KNOW are non-members and a
SUSPECT pile of unknown membership. IsolationForest is fitted on the known pile;
suspects are ranked by anomaly. Half the true non-members are dealt to the known
pile each draw, the rest join the members as suspects. The blind (TabPFN-2.5) is
put through the identical procedure on its own signals; the gap is target minus
blind.

The informativeness screen keeps signal j only when its known-vs-suspect
separation beats the 95th percentile of separations between random halves of the
KNOWN pile. It never sees labels or the blind, and it is re-derived from scratch
on the blind's own signals.

Reading the tables
  blind < 0.49  the blind is ranked BACKWARDS; its "gap" is an artifact.
  meta          informational only: what log(n_pool)+log(n_features) score with
                the model never queried. On SAP this is ~0.97 because membership
                there IS a row-count threshold -- that is the ground truth, not a
                confound, and it does NOT disqualify an arm. The standard is
                target > blind.
"""
import argparse
import os
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from di.ifsel.referee import evaluate, load, CACHE          # noqa: E402
from di.ifsel.probes.noise_floor import informative         # noqa: E402

FAMS = ("base", "bwash", "col", "constlbl", "ctx", "ctxeqq", "ftx", "lperm",
        "logit", "mislbl", "nnsplit", "noise", "qleak", "qonly", "rowdup",
        "rowshuf", "seed", "splitsize", "tabdpt_sim", "temp", "tserum", "tshuffle")


def fam_of(s):
    for f in sorted(FAMS, key=len, reverse=True):
        if s == f or s.startswith(f + "_"):
            return f
    return s.split("_")[0]


def all_signals(Xk, Xs, mk, ms, rng):
    return np.ones(Xk.shape[1], bool)


def arm_matrices(corpus, cols, tag):
    """Slice a corpus down to `cols` and cache it so the referee can load it.

    The tag carries the pid: two of these run concurrently (as-is and matched)
    and a shared scratch name means one process reads a slice the other is still
    writing -- which shows up either as an EOFError on a half-written zip or,
    worse, as silently mismatched numbers.
    """
    tag = f"{tag}_{os.getpid()}"
    Xt, Xb, y, sigs, meta, mn = load(corpus)
    np.savez_compressed(CACHE / f"{tag}.npz", Xt=Xt[:, cols], Xb=Xb[:, cols], y=y,
                        sigs=np.array([s for s, k in zip(sigs, cols) if k]),
                        meta=meta, meta_names=np.array(mn))
    return tag, np.isfinite(Xt[:, cols]).mean(), np.isfinite(Xb[:, cols]).mean()


def run(corpus, draws, screen_B, blind_sel="target", drop=()):
    Xt, Xb, y, sigs, meta, mn = load(corpus)
    drop = set(drop)
    live = np.array([fam_of(s) not in drop for s in sigs])
    arms = [("ALL", live)]
    for f in sorted({fam_of(s) for s in sigs} - drop):
        c = np.array([fam_of(s) == f for s in sigs])
        if c.sum() >= 2:
            arms.append((f, c))

    print(f"\n{'='*112}")
    print(f"{corpus}   n={len(y)} ({int(y.sum())} member / {int((y==0).sum())} non-member)   "
          f"{len(sigs)} signals   iForest   draws={draws}   "
          f"{('excluding ' + ','.join(sorted(drop)) + '   ') if drop else ''}"
          f"blind scored on {'TARGET-selected columns' if blind_sel=='target' else 'its OWN selection'}")
    print(f"{'':30s}|{'  NO SCREEN':^32s}|{'  INFORMATIVENESS SCREEN':^40s}")
    print(f"{'family':12s} {'cols':>5s} {'bcov':>5s} |"
          f"{'target+-sd':>15s} {'blind+-sd':>15s} {'gap+-sd':>15s} |"
          f"{'kept':>5s} {'target+-sd':>15s} {'blind+-sd':>15s} {'gap+-sd':>15s}")
    print("-" * 150)

    rows = []
    for name, cols in arms:
        tag, tcov, bcov = arm_matrices(corpus, cols, "_arm")
        out = {"corpus": corpus, "arm": name, "n_cols": int(cols.sum()),
               "target_cov": round(float(tcov), 4), "blind_cov": round(float(bcov), 4)}
        line = f"{name:12s} {int(cols.sum()):5d} {bcov:5.0%} |"
        for key, sch in (("noscreen", all_signals), ("screen", informative)):
            r = evaluate(tag, sch, draws=draws, blind_selection=blind_sel)
            if "gap" not in r:
                out[key] = None
                line += f"{'  --':>15s}{'':>16s}{'':>16s} |" if key == "noscreen" else \
                         f"{'':>5s}{'  <2 selected':>15s}"
                continue
            out[key] = {k: r[k] for k in ("target_auc", "blind_auc", "gap",
                                          "n_selected", "tpr@0.1", "fpr@0.1",
                                          "meta_only_auc", "blind_below_chance",
                                          "target_auc_sd", "blind_auc_sd", "gap_sd",
                                          "draws")}
            mark = "*" if r["blind_below_chance"] else " "
            nd = "" if r["draws"] == draws else f"!{r['draws']}d"
            cell = (f"{r['target_auc']:6.3f}+-{r['target_auc_sd']:.3f} "
                    f"{r['blind_auc']:6.3f}{mark}+-{r['blind_auc_sd']:.3f} "
                    f"{r['gap']:+6.3f}+-{r['gap_sd']:.3f}")
            if key == "noscreen":
                line += cell + f"{nd} |"
            else:
                line += f"{r['n_selected']:5d} " + cell + nd
        print(line, flush=True)
        rows.append(out)

    meta_auc = next((r["noscreen"]["meta_only_auc"] for r in rows
                     if r.get("noscreen")), float("nan"))
    print("-" * 150)
    print(f"  +-sd = standard deviation across the {draws} draws; the gap sd is the "
          f"paired per-draw difference (target and blind share each draw's split).")
    print(f"  * = blind below 0.49, i.e. ranked backwards.")
    print(f"  !Nd = only N of {draws} draws produced >=2 selected signals; the rest were "
          f"dropped, so that row rests on N draws and its sd is meaningless at N=1.")
    print(f"  metadata-only auditor (model never queried): AUC {meta_auc:.3f}  "
          f"[informational; membership on SAP IS a row-count threshold]")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="sap_allcls_hybrid,tabdpt,realtabpfn",
                    help="sap_allcls_hybrid = SAP hybrid + natively-classification union")
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--screen-B", type=int, default=40)
    ap.add_argument("--exclude", default="",
                    help="comma-separated families dropped entirely: their own row "
                         "is skipped AND their columns leave the ALL arm")
    ap.add_argument("--blind-selection", default="target", choices=["target", "own"],
                    help="target: blind scored on the columns the scheme chose from "
                         "the TARGET (the blind never selects; our control). "
                         "own: blind re-derives the scheme on its own signals.")
    ap.add_argument("--out", default="exports/hyp_ifsel_audit_tables.json")
    a = ap.parse_args()
    allrows = []
    for c in a.corpora.split(","):
        allrows += run(c, a.draws, a.screen_B, a.blind_selection,
                       [x for x in a.exclude.split(',') if x])
    p = Path(os.environ.get("AUDIT_REPO", "/p/project1/hai_1159/tfm-di")) / a.out
    tag = "blind" + a.blind_selection + (("_no" + a.exclude.replace(",", "")) if a.exclude else "")
    p = p.with_name(p.stem + f"_{tag}" + p.suffix)
    p.write_text(json.dumps({"draws": a.draws, "rows": allrows}, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
