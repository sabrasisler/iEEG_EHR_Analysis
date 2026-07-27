# CLAUDE.md — iEEG_EHR_Analysis

Operating rules for Claude Code in this repo (pain iEEG: sensory vs affective
pain encoding in ~250 EMU patients; Sisler / Keller Lab). These rules always
apply. For full detail, READ the relevant reference doc BEFORE working on that
area:

- `docs/pain_ieeg_analysis_architecture.md` — data/layer model, directory layout,
  cache format, feature-level QC, cohorts. Read before touching caching, QC,
  features, or directory structure.
- `docs/analysis_kickoff_plan.md` — ordered task plan, repo org, IO, background
  jobs, git workflow. Read before starting a new phase/task or writing sbatch.
- `docs/view_registry.md` — the seven view axes and their order. Read before
  writing or changing any view/normalization/averaging/binning code.

Do not re-derive these decisions from scratch; they are settled. If a task seems
to require violating a rule below, stop and ask.

---

## CODE / DATA BOUNDARY (read first — never violate)

**This repository holds CODE ONLY. All data, derivatives, caches, features,
epoch definitions, QC outputs, cohort files, analysis outputs, plots, models,
logs, and results live on Oak, NOT in the repo.**

- The repo is git-tracked and pushed to GitHub. Writing data into it risks
  committing PHI-adjacent artifacts to a remote and bloating the repo.
- The directory trees shown in the architecture doc (`preprocessed/`, `qc/`,
  `features/`, `analysis/`, `cohorts/`) are rooted at the Oak derivatives base
  `/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/`, NOT in the
  repo. Never create these folders inside the repo.
- ALL output paths must resolve to Oak (or the configured derivatives base in
  the config module). Never write an output, cache, plot, model, or results file
  to a repo-relative path. If a script has no explicit output base, use the
  config module's derivatives base — do not default to `./`.
- The ONLY things in the repo: source code, sbatch templates, config, docs,
  tests, `CLAUDE.md`, and gitignored scratch (`logs/`, `notebooks/` scratch).
  Even scratch DATA does not belong in the repo — throwaway plots go to a
  scratch location on Oak, not into the repo tree.
- `.gitignore` is a backstop, not the mechanism: the correct behavior is that
  nothing writes data into the repo in the first place.

---

## Layers (the core model)

- **Feature families** (PSD/PAC/connectivity): continuous, per-window, expensive,
  stored under `preprocessed/`. Each is a SEPARATE extraction. PAC and
  connectivity come from the TIME-DOMAIN bipolar signal (Hilbert), NOT from FFT
  data. Only PSD-derived things (bandpower, HFA, 1/f-via-polyfit) come from `bipolar_fft`.
- **Cache**: a feature family sliced to event windows — per-2s-window, per-channel,
  per-freq-bin, **log-power**, QC-masked, **PRE-normalization, PRE-averaging**.
  Format: **Parquet, ONE file per subject/session** (subject/session in the
  FILENAME; epochs stacked inside via an `epoch_id` column). NOT NWB. NOT
  one-file-per-epoch. Immutable. float32 (after round-trip validation).
- **Epoch definitions**: tiny Parquet index (run + window indices + pain label +
  mask ref) beside the cache.
- **Views**: the chain domain → baseline → normalize → epoch-average → freq-agg →
  region-agg → binarize. FUNCTIONS in the repo (`views/`), recomputed at load by
  default, NOT saved by default.

## Decision: view vs analysis vs stored feature (stop at first match)

1. **Terminal** (a human looks at it, or a model consumes it as input)? →
   **ANALYSIS**. Save under `analysis/pain/<question>/<output_type>/<scheme>/<run>_<timestamp>/`.
2. **Non-terminal + cheap transform of stored data?** → **VIEW**. Recompute at
   load; do NOT save by default. (Holds even if multi-step or derived from another view.)
3. **Non-terminal + expensive** (new extraction, or expensive intermediate many
   depend on)? → **STORED FEATURE / CACHE**. Save under `features/` / `preprocessed/`.

"Terminal" = nothing in the pipeline consumes it further; the consumer is a human
eye or a model. A view is a step; an analysis is a stop.

## Cache rules

- Store per-2s-window LOG-power. NEVER epoch-averaged, NEVER normalized in the
  cache (normalization is per-window, before averaging — Jensen's inequality).
- New cache (expensive re-run) ONLY for a new epoch length or a new QC mask.
- `build_pain_epoch_power.py`: emit epoch definitions + the per-window Parquet
  cache. Do NOT average over the epoch, do NOT normalize (views do that).

## View functions

- Every view function takes optional `save_path=None`. Default = recompute (don't save).
- If `save_path` is set, ALSO write a provenance+staleness sidecar (cache manifest
  hash, view git commit, view config, date). On load, if current cache hash or git
  commit differ from the sidecar → WARN or refuse. Never a bare `to_csv`/`to_parquet`
  without a sidecar.
- Materialize a view's output ONLY when recompute is measured slow AND something
  depends on it. Do not save cheap views by default.

## Feature-level QC

- Choice-independent: run ONCE on pre-normalization cached power, stored in
  `qc/feature_level/`, inherited by all views. Recording-wide per-channel
  median+MAD baseline. Cascade: window → epoch → channel → epoch-across-channels.
  Thresholds (K, X, Y, Z) are TODO — set on structural grounds BEFORE looking at
  pain relationships. Metric/threshold split (store metrics once, threshold cheaply).

## analysis/ organization (5 levels)

`1 <event>/  2 <question>/  3 <output_type>/  4 <view_scheme>/(optional)  5 <run_name>_<timestamp>/`

- Levels 1-2 are opened DELIBERATELY (new domain / a named question that exists in
  the exploration log). Levels 3-5 are created freely per run. Never a
  folder-per-plot at levels 1-2.
- All combinatorial sweep pressure goes into ROWS in a `sweeps/` `results.parquet`,
  NEVER into folders.
- Discovery vs confirmation is a COHORT REFERENCE in config, not a folder level.
- Which subjects are in a run: read `provenance.json` `subjects[]`, NEVER the folder name.

## Cohorts

- Age is PHI and NOT on Sherlock. Demographic matching happens OFFLINE; only the
  anonymized `subject_id → cohort` assignment + SAFE matching axes cross to Sherlock.
- Current discovery subjects are locked as discovery PERMANENTLY.
- The hold-out cohort is UNREACHABLE by default in exploratory runs; gated by a
  `--split` / cohort-file flag. Never read/plot/model hold-out during exploration.

## Sweeps / exploration

- Sweep output is NOMINATIONS, not findings. Report per-subject effect sizes +
  sign-consistency across subjects + contributing n. Do NOT compute a pooled
  p-value that ignores per-subject structure. FDR within the discovery set only.
- Run cheapest/most-aggregated tiers first; show the results table; gate finer
  tiers on review.

## Git / provenance

- Commit AND push BEFORE any definitive/array run, so the recorded commit hash
  matches the code that ran. Warn if the tree is dirty.
- Every stored artifact writes provenance: git commit (+dirty flag), timestamp,
  parent artifact reference, and `subjects[]` for runs.

## IO / naming

- Parquet for tables; joblib (NOT raw pickle) for fitted models / FOOOF objects;
  JSON for manifests and sidecars. Never pickle tabular data.
- New artifacts only — do NOT bulk-convert existing CSVs; convert one when next touched.
- Do NOT fingerprint runs or plots (use human label + timestamp). Fingerprint ONLY
  materialized-view folders, and only if recompute is measured slow.

## Compute

- Never run Python on the Sherlock login node. Use Slurm (`sbatch`, or `srun -p dev`
  for quick tests). Load env: `module load python/3.12` then activate the venv.
- Parallelize per-subject work as Slurm ARRAY jobs (one task per subject), not via
  multiple agents. Long arrays on `normal`; keep `ckeller1 --qos=high_p` (4-job cap)
  free for interactive work.
- All `.sbatch` files live in `sbatch/`. Scratch notebooks in `notebooks/` (never
  imported by pipeline code). Superseded code in `outdated/`.