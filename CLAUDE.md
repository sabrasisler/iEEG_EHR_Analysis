# CLAUDE.md — iEEG_EHR_Analysis

Operating rules for Claude Code in this repo (pain iEEG: sensory vs affective
pain encoding in ~250 EMU patients; Sisler / Keller Lab). These rules always
apply. For full detail, READ the relevant reference doc BEFORE working on that
area:

- `docs/architecture.md` — data/layer model, directory layout,
  cache format, feature-level QC, cohorts. Read before touching caching, QC,
  features, or directory structure.
- `PLANNING.md` — the phases, milestones, and sequencing rationale. Read before
  starting a new phase or task.
- `docs/io_conventions.md` — the artifact contract: the `io` helper API, the
  sidecar envelope, staleness rules, where outputs go, what stays CSV, and the
  pyarrow install recipe. Read before writing ANY script that produces output.
- `docs/kickoff_plan.md` — repo org, background jobs, git workflow. Read before
  writing sbatch. (The forward-looking half of this doc now lives in
  `PLANNING.md` / `TASKS.md`; its IO section is superseded by
  `docs/io_conventions.md`; what remains is reference material.)
- `docs/WORKFLOW.md` — where every kind of record goes, and the commands that
  write them. Read before logging anything.
- `docs/view_registry.md` — the seven view axes and their order. Read before
  writing or changing any view/normalization/averaging/binning code.

Background / historical detail (descriptive, not normative — the three docs
above win on any conflict):

- `docs/qc_context.md` — how the raw-voltage QC pipeline actually works:
  the four detectors, the metric/threshold split, mask labels, and a running
  log of threshold sweeps and case studies.
- `docs/pain_analysis_context.md` — the pain epoch-power pipeline as it stood
  before the Phase 1 refactor.

Do not re-derive these decisions from scratch; they are settled. If a task seems
to require violating a rule below, stop and ask.

---

## TRACKING FILES — where records go

Every record has exactly one home, determined by what the thing *is*, which is
the same question as how long it lives. `docs/WORKFLOW.md` has the full routing
table and a worked example; the short version:

| The thing is… | Home |
|---|---|
| A standing rule (permanent) | this file |
| A phase / milestone / the project's shape | `PLANNING.md` |
| A thing to **DO** — checkable, has a done-state | `TASKS.md` |
| A thing being **THOUGHT** — question/hunch, no done-state | `SCRATCHPAD.md` |
| A settled call **and its reasons** | `DECISIONS.md` (append-only) |
| A thing that **happened** — ran X, saw Y, Z broke | `docs/labnotebook/YYYY-MM-DD.md` (append-only) |
| A note on one specific figure | `<figure>.png.notes.md` beside it on Oak |
| "I ran this" — terse index line | `docs/analyses_run.md` (machine-appended) |

Rules that apply whenever you touch these files:

- **Tasks vs scratch: can you check it off?** Yes → task. No → scratch. One
  realization often spawns both; split at the seam between the noticing and the
  doing.
- **Links, never copies.** Cross-reference by date, path, commit hash, and
  task/phase ID. Never paste a decision into the notebook or restate a phase in
  `TASKS.md`.
- **Append-only means append-only.** `DECISIONS.md`, the notebook, and
  `analyses_run.md` are never edited in place or reflowed. Corrections are
  appended, so the record of what was believed at the time survives.
- **A sweep result is a NOMINATION, not a finding.** Observations accumulate in
  the notebook until they earn a `DECISIONS.md` entry at P2.6 FREEZE. Do not
  promote one yourself.
- **Deidentified prose only** — anonymized subject IDs, 2001-anchored timeline.
  These files are committed to a GitHub remote.
- **`TASKS.md` and `SCRATCHPAD.md` stay small.** If `TASKS.md` feels
  overwhelming, far-future work has leaked in; push it down to a `PLANNING.md`
  phase. Resolved scratch items are deleted once committed.
- Commands: `/lognote` (write up what happened), `/annotate` (figure sidecar),
  `/addtask`, `/addscratch`, `/standup` (read-only continuity briefing). Every
  `/lognote` prompt is skippable — a bare entry is a success.

Any script that produces analysis output calls `log_analysis()`
(`src/ieeg_ehr/io/analysis_log.py`) beside its provenance sidecar write, passing
the RUN directory. Add it as you next touch a script; no sweep-and-add pass.
This is the ONE sanctioned write into the repo — a text index, never data.

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
  tests, `CLAUDE.md`, and gitignored `logs/`.
  Even scratch DATA does not belong in the repo — throwaway plots go to a
  scratch location on Oak (`analysis/scratch/`), not into the repo tree.
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
  one-file-per-epoch. Immutable. **float32** — validated by the P0.6 audit
  (bit-exact round-trip through Parquet and HDF5; epoch averages agree with a
  float64 pipeline to 8.1 sig figs). `config.CACHE_FLOAT_DTYPE`.
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

- **Store narrow, compute wide (P0.6).** The cache is float32; views must UPCAST
  before they reduce. Any epoch average/sum over windows goes through
  `config.CACHE_ACCUMULATE_DTYPE` (float64) — numpy does NOT do this for you, so
  a bare `arr.mean(axis=0)` on float32 input accumulates in float32 and holds
  only ~6 sig figs. Exponentiating log-power to linear (axis 1, `domain`) uses
  `config.CACHE_LINEAR_DOMAIN_DTYPE` (float64): the worst stored log-power is
  ~-36.8, barely a decade above float32's smallest normal, so a later baseline
  division can underflow to exactly zero.
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

Full contract + API: `docs/io_conventions.md`. Read it before writing any script
that produces output. The rules:

- Parquet for tables; joblib (NOT raw pickle) for fitted models / FOOOF objects;
  JSON for manifests and sidecars. Never pickle tabular data.
- **Go through `ieeg_ehr.io`, never a bare writer.** `io.write_table` /
  `io.save_model` / `io.write_manifest` / `io.write_run_provenance` emit the
  provenance sidecar in the same call; `io.read_table` / `io.load_model` /
  `io.assert_fresh` check staleness on the way back in (`on_stale='refuse'` for
  anything a reported number comes out of). A bare `df.to_parquet` / `to_csv` /
  `joblib.dump` in new code is a bug.
- **Output paths come from `config/paths.py` builders**, never hand-assembled:
  `pain_epoch_{unit_dir,cache_path,defs_path,manifest_path,views_dir}` for the
  Phase-1 cache, `analysis_run_dir` / `sweep_run_dir` for the 5-level analysis
  scheme. All resolve to Oak.
- The **QC tree stays CSV** (existing artifacts, working metric/threshold split);
  `io.save_table` and `io.append_table` remain for it. Everything new is Parquet.
- New artifacts only — do NOT bulk-convert existing CSVs; convert one when next
  touched, and give it a sidecar when you do.
- Do NOT fingerprint runs or plots (use human label + timestamp). Fingerprint ONLY
  materialized-view folders, and only if recompute is measured slow —
  `io.config_hash` is the one sanctioned fingerprint.

## Compute

- Never run Python on the Sherlock login node. Use Slurm (`sbatch`, or `srun -p dev`
  for quick tests). Load env: `module load python/3.12` then activate the venv.
- Parallelize per-subject work as Slurm ARRAY jobs (one task per subject), not via
  multiple agents. Long arrays on `normal`; keep `ckeller1 --qos=high_p` (4-job cap)
  free for interactive work.
- All `.sbatch` files live in `sbatch/`. All code lives in the `src/ieeg_ehr/`
  package and is invoked as `python -m ieeg_ehr.<subpackage>.<module>` — never
  `cd` into the repo first. Superseded code, one-off sbatch, and the retired
  notebooks live in `outdated/` and are never imported. Notebooks are NOT part
  of the workflow; do not add new ones.