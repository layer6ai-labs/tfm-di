# di.audit

Audits of a tabular foundation model's signal tables against the blind TabPFN-2.5
reference. Input is the HuggingFace layout `signals/<model>/<pool>/datasets.parquet`:
one table per model and pool, one row per dataset in audit order, the columns
`dataset_id`, `true_label`, `task_type`, `success`, `n_pool`, `n_features` and one
column per signal; the blind's table sits under `signals/tabpfn25/<pool>/`. Output is
JSON, rendered to Table 1 (external audit) and Table 2 (internal audit). Both tables
read the same four pools, and every dataset is probed as a classification task: a natively
categorical target is kept; for a dataset whose target is continuous a column with 2 to 10
distinct values is drawn at random (seeded by the dataset id) as the target when one exists,
otherwise the column with the fewest distinct values is used, binned into at most 10 quantile
classes with edges from the training split. The rule is a function of the data alone, so
target and blind receive the identical task (`di/data_loading.py`, `force_task_type`).

## Command line

```bash
# Table 2 cells: top-K meta-classification, LR and XGBoost on CPU, TabPFN-2.5 on GPU
uv run python -m di.audit internal --pool tabdpt,nano,realtabpfn --kinds lr,xgb,tabpfn --device cuda --data-root data_root --out results/internal.json
uv run python -m di.audit internal --pool sap --scope no:ctx --ks 1,2,4,6,8,12,16,24,32,64 --kinds lr,xgb,tabpfn --device cuda --no-families --out results/internal_sap.json

# Table 1 cells: label-free IsolationForest audit (CPU)
uv run python -m di.audit external --pool tabdpt,nano,realtabpfn --draws 10 --out results/external.json
uv run python -m di.audit external --pool sap --draws 10 --exclude ctx --out results/external_sap.json

# alternative TabDPT weights (same datasets): --target seed42 | seed123 | seed456

# Table 3 / Table 8: feature contributions of the internal audit's ranking fits (CPU)
uv run python -m di.audit shap --pool sap --scope no:ctx --out results/shap_sap.json
uv run python -m di.audit shap --pool tabdpt --out results/shap_tabdpt.json

# the result figures (Figures 1-4 and the TabDPT appendix figure) from the same runs
uv run python -m di.audit figures --data-root data_root --reference-dir tests/reference --out-dir figures

# render
uv run python -m di.audit tables --internal results/internal*.json --external results/external*.json --shap results/shap*.json --format tex

# re-run the LR/XGB and external cells of the published tables and compare to the reference files
uv run python -m di.audit reproduce --data-root data_root --reference-dir tests/reference

# the same comparison as a test suite (--fast: published cells only)
TFM_DATA_ROOT=data_root TFM_REFERENCE_DIR=tests/reference uv run pytest --fast
```

## Modules

| module | contents |
|---|---|
| `pools.py` | `POOLS` registry, `load(name, data_root)`, `select(signals, scope)`, `fam_of`, `NON_FEATURES` |
| `models.py` | `fit_predict` (lr / xgb / tabpfn), `treeshap_importance`, `rank_treeshap`, `elbow`, `XGB_PARAMS`; `make_tabpfn` reuses the checkpoint cache `di.datasets.tabpfn25_wrapper._get_cached_specs` and reads the checkpoint from `$TABPFN_MODEL_CACHE_DIR` (default `~/.cache/tabpfn`) |
| `internal.py` | `audit_cell`, `run_internal`, the K grids `DEFAULT_KS` and `SAP_KS` |
| `external.py` | `evaluate`, `run_external`, the informativeness screen |
| `contributions.py` | `run_contributions`: fold-averaged mean \|TreeSHAP\| per column for target and blind |
| `names.py` | `describe`: paper label `<Manipulation>_<Signal>_<Reduction>` and M/S/R codes of a signal column |
| `figures.py` | `make_all` and the per-figure functions: separation histograms, internal-audit AUC bars, external-audit p-value histograms |
| `tables.py` | `PUBLISHED` (the four audited models and their pools, parameters and reference files), `published_internal` / `published_external` (the runs behind the tables), `render_internal`, `render_external`, `render_contributions`, `compare`, `reference_internal`, `reference_external` |
| `cli.py` | the six subcommands |

## Internal audit

Datasets are the rows, membership is the label. Columns that are constant or all-NaN in
both the target and the blind are dropped first; K is capped at the number of columns
that remain. Folds are `StratifiedKFold(min(5, minority count), shuffle=True,
random_state=0)`. On each training fold an XGBoost (`XGB_PARAMS`) is fit on the target
columns and they are ranked by mean |TreeSHAP|. For each K in the grid the
meta-classifier (`lr`: fold-local median imputer, `StandardScaler`,
`LogisticRegression(max_iter=2000)`; `xgb`: `XGB_PARAMS`; `tabpfn`: TabPFN-2.5, 4
estimators) is fit on the top-K target columns of the training fold and scores the
held-out fold; the blind is fit and scored on exactly the same columns. Out-of-fold
scores are pooled into one ROC-AUC per side. K* is the point of the target
AUC-vs-log2 K curve farthest above the chord between its end points (the argmax when the
curve has fewer than three points or is flat). An LR or XGBoost failure propagates; a
TabPFN fit that raises (its decomposition on a degenerate fold) scores 0.5 and emits a
warning.

Family cells repeat the procedure on the columns of one signal family (families with at
least five columns). The SAP row uses the scope `no:ctx` (every family except `ctx`)
and the K grid `SAP_KS = 1,2,4,6,8,12,16,24,32,64`; the other rows use `ALL` and
`DEFAULT_KS = 1,2,4,8,16,32,64`.

Reported per cell: K*, target and blind AUC and their gap at K*, the bootstrap sds and
`p_gap_le_0` (share of resamples with gap <= 0, four decimals), `nboot_used`, and the
curve of target/blind AUC at every K. The curve carries point estimates only; the
bootstrap is run at K* alone.

## External audit

Draw i (i = 0..9) uses `default_rng(7 + i)`: the known pile is a random half of the
non-members (at least 5), the suspects are the remaining non-members plus every member.
NaN is filled with known-pile column medians (0 when a column is all NaN). The screen
keeps column j when its known-vs-suspect |AUC - 0.5| exceeds the 95th percentile of the
same statistic over 40 random half-splits of the known pile; `noscreen` keeps every
column. An `IsolationForest(n_estimators=200, contamination="auto")` fit on the known pile
scores the suspects; the blind runs the same draw on its own matrix and is scored on the
columns selected on the target. Draws on which either side selects fewer than two columns
are dropped. Reported per cell: mean and sd (ddof=1) over the surviving draws of target
AUC, blind AUC and their paired gap, the number of selected columns, conformal TPR/FPR at
level 0.1. Reported once per arm: the
number of columns, target and blind coverage, and the AUC of the same scorer on
`log1p(n_pool, n_features)` alone.

Arms: `ALL` (minus the excluded families), then every remaining family with at least two
columns. The published cell is the `screen` cell of `ALL`; draws on which the screen keeps
fewer than two columns are dropped (TabDPT: 6 of 10 draws survive).

## Pools

| pool | tables under `signals/` | datasets | members | non-members | signals | target coverage | blind coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| `sap` | `sap-rpt-oss/binary_t4`, blind `tabpfn25/binary_t4`: every table of one T4 chunk, the 8,907 natively-categorical tables with their own target and the 11,193 tables with a continuous target re-targeted by the rule | 20,100 | 16,179 | 3,921 | 678 | 0.988 | 0.848 |
| `tabdpt` | `tabdpt/binary_tabdpt`, blind `tabpfn25/binary_tabdpt`: TabDPT's pre-training corpus vs. OpenML-CC18/CTR23 datasets it did not use, 160 classification datasets and 64 re-targeted regression datasets; `tabdpt_seed{42,123,456}/binary_tabdpt` hold the re-trained seeds on the same datasets | 224 | 117 | 107 | 678 | 0.995 | 0.904 |
| `nano` | `nanotabpfn/nanotabpfn_matched20k`, blind `tabpfn25/nanotabpfn_matched20k`: 10,000 training and 10,000 held-out tables of the synthetic prior | 20,000 | 10,000 | 10,000 | 678 | 1.000 | 1.000 |
| `realtabpfn` | `realtabpfn25/binary_realtabpfn25`, blind `tabpfn25/binary_realtabpfn25`: the OpenML datasets among Real-TabPFN-2.5's real fine-tuning data vs. TabArena datasets it did not use, 55 classification and 15 re-targeted regression datasets | 70 | 19 | 51 | 678 | 0.996 | 0.996 |

Coverage is the share of finite cells in the signal matrix (the SAP `no:ctx` scope has
623 signals; coverage 0.988 / 0.839). Signals are the numeric, non-boolean columns shared by
target and blind minus `pools.NON_FEATURES`, in sorted name order; rows are taken in file
order and the blind is aligned to the target by `dataset_id`. Re-targeted rows carry the
chosen column and rule in `forced_target`.

## Feature contributions

`shap` reports, for the usable columns of a scope, the mean |TreeSHAP| (log-odds units) of
the XGBoost ranking fit of each training fold of the internal audit, averaged over the folds:
on the target columns and, in a separate fit on the same columns, on the blind's. The table
lists the target's top columns with the blind's value at the same column; the result file also
holds the blind's own top columns, the overlap of the two top sets, the Spearman correlation
of the two importance vectors and the full importance vectors. Scope and folds are those of
the internal audit (SAP: `--scope no:ctx`), so the ranking averaged here is the one the audit
selects top-K from. A column's paper label is `<Manipulation>_<Signal>_<Reduction>` with the
codes of the paper's manipulation, signal and reduction tables (`names.py`).

## Figures

`figures` draws the paper's result figures from the runs behind the tables, never from a
separate fit. The single-signal panels show the unmanipulated probe's `base_loss` and
`base_correct_confidence` per dataset; the multi-signal panel shows the pooled out-of-fold
P(member) of the published LR cell at its K* (`audit_cell(..., scores=True)`); the AUC bars
are the TabPFN-2.5 cells of the reference files; the p-value panels are the conformal
p-values of the first published external draw (`default_rng(7)`) on the screened target
columns, target and blind. Histograms are normalised per class; legends are written to
`*_legend_<n>col.pdf`; `figure_data.json` records the AUC behind every panel.

## Task-type ablation

Three pools per model were audited to motivate the rule: A, every dataset with its natural
task type and both measurement vocabularies; C, the natively-classification datasets only;
B, every dataset as a classification task (the rule). The A and C tables, and a variant of B
that keeps the designated continuous target and bins it, are published under `ablation/` in
the HuggingFace repository in the same layout; load one with `load(name, "<ablation dir>")`.

## What the ± means

Internal (Table 2): the sd over 2000 paired bootstrap resamples of the datasets at K*,
computed from the pooled out-of-fold scores (resamples with one class are dropped;
`nboot_used` counts the rest). Each cell draws its resamples from
`default_rng([seed, crc32("pool|scope|kind")])`, so a cell's ± does not depend on which
other cells run. The published tables drew every cell's resamples from one shared
`default_rng(0)` per process, so a regenerated ± can differ from the published one in
the third decimal. Point estimates do not depend on the bootstrap seed.

External (Table 1): the sd over the surviving draws; the gap sd is the sd of the per-draw
paired difference.

## Reproducing the published tables

`data_root` is a download of the HuggingFace repository `SprintML/tfms` (gated; request
access, then `hf auth login`):

```bash
hf download SprintML/tfms --include "signals/**" --local-dir data_root
```

`uv run python -m di.audit reproduce --data-root data_root --reference-dir tests/reference`
recomputes every LR/XGB internal cell and every external cell on CPU and reports, per
cell, whether K, target, blind and gap (internal) or `tables.EXTERNAL_KEYS` (external)
equal the reference file; the TabPFN-2.5 cells need a GPU and are reported but not
compared. The reference files under `tests/reference/` are `run_internal` / `run_external`
outputs keyed by pool.

Run the internal audit single-threaded (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1`) when comparing against another machine: logistic-regression point
estimates on cells with one or two near-chance columns can move in the fourth decimal
with the BLAS thread count and platform; the published cells do not.
