# Pain iEEG — Feature & Analysis Architecture

One of three companion docs handed to Claude Code in the Sherlock repo
`/home/groups/ckeller1/sisler/iEEG_EHR_Analysis`:
- **this** — the data/layer model + directory layout + rules
- `kickoff_plan.md` — the ordered task plan, repo org, IO, background jobs
- `view_registry.md` — the enumerated view axes

Project: shared neural signatures, individual variability, and opioid modulation
of pain encoding in ~250 EMU iEEG patients (Sisler / Keller Lab; advisors Yu
Zhang & Corey Keller). Central question: disentangle **sensory-discriminative**
vs **affective-evaluative** contributions to pain — first generalizable
population signatures (Aim 1), then individual deviations / phenotypes (Aim 2),
then opioid modulation (Aim 3).

---

## PART 0 — Current pipeline state (2026-07)

Established upstream, do not redesign:

- **Continuous PSD (stored FEATURE FAMILY)**: `run_pipeline_bipolar.py` reads each
  raw NWB run once, bipolar re-references (trace never persisted, ~40TB avoided),
  Welch PSD in **2s windows, 50% overlap (1s hop), 50 log-spaced bins 1-250 Hz**,
  stored as **log-power** in DecompositionSeries NWB at
  `derivatives/sisler/preprocessed/bipolar_fft/sub-XXX/ses-XXX/`. Line-noise bins
  flagged. Per-subject sidecar has FFT params + git provenance.
- **Canonical bands NOT precomputed** — `bipolar_bands.py` aggregates the 50 log-bins
  into bands on demand (delta 1-4, theta 4-8, alpha 8-12, beta 15-25, gamma 25-70,
  high_gamma 70-170 Hz), linear-then-log to avoid Jensen bias.

  > **DISCREPANCY — unresolved, flagged 2026-07-27 during the structural refactor.**
  > The band edges above are NOT what the code uses. `CANONICAL_BANDS_HZ` (now
  > `ieeg_ehr.config.psd_params`) is:
  > `delta 1-4, theta 4-8, alpha 8-12, beta 13-30, low_gamma 30-58,
  > high_gamma1 65-115, high_gamma2 125-175, high_gamma3 185-235`.
  > The code splits gamma finely and specifically to fall between 60 Hz line-noise
  > harmonics, which this doc's `gamma 25-70 / high_gamma 70-170` edges would
  > straddle. Deliberately NOT reconciled in the refactor — which set is correct
  > is a science decision. Resolve before the P2.2 sweep, since band choice is a
  > sweep axis. See also `VIOLIN_BANDS_HZ`, a third, coarser grouping used only
  > by the violin plots.
- **QC (metric/threshold split)**: raw-voltage detectors (saturation, flatline,
  square-wave, gross-artifact) store continuous *metrics* once; cheap *exclusions*
  (per threshold) → *masks* (OR'd union). Mask candidates: `gross-std3_satmargin15_sw`
  (82 subj), `gross-std3_satmargin15_sw_logz4` (85 subj). Bipolar per-2s variance
  is a tunable-threshold metric too.
- **Pain epoching**: `build_pain_epoch_power.py` currently slices the 5-min pre-event
  window, applies the QC mask, AND averages over the epoch. **The averaging must
  move OUT into the view layer** (PART 4); the script's new job is emitting the
  per-window cache + epoch definitions. Pain scores are **spot ratings** (~1-2h).
- **Pain bin schemes** implemented (absolute, subject_relative) — recomputed at plot
  time. These are VIEWS.
- **Cohort**: ~56-67 currently explored; ~150 more coming. An informal hold-out
  exists now. Formal matched hold-out to be built later (PART 6).

---

## PART 1 — The layer model (read first)

Three kinds of stored things + one kind of function.

| Layer | Category | What | Stored? | Cost | Changes when |
|-------|----------|------|---------|------|--------------|
| **Feature family** | stored data (continuous) | per-window neural quantity over whole run: PSD (`bipolar_fft`), later PAC, connectivity | YES, tree per family | expensive (reads raw) | new family extracted |
| **Cache** | stored data (epoched) | a family SLICED to event windows: per-2s-window, per-channel, per-bin **log-power**, QC-masked, PRE-normalization, PRE-averaging. Immutable. | YES (materialized; slice-on-load measured too slow) | moderate | epoch def (window length/anchor) or QC mask changes |
| **Epoch definitions** | stored metadata (tiny) | index table: run + window indices + pain label + mask ref per epoch | YES (beside cache) | trivial | same as cache |
| **View** | FUNCTION (repo code) | domain→baseline→normalize→epoch-avg→freq-agg→region-agg→binarize | NO by default (recompute); optional `save_path` | ~free | never persisted |

### Cache format (decided)

**Parquet, ONE file per subject/session**, subject/session in the FILENAME (not in
rows). Epochs stacked inside under an `epoch_id` column. Columns:
`epoch_id, window_idx, channel, bin, log_power` (+ an `included` boolean or mask
ref). NOT NWB (its contiguous-run time model fights a stack of discontiguous
epochs), NOT one-file-per-epoch (file explosion). Store **log-power** (matches
`bipolar_fft`, float32-friendly range, Gaussianizes for stats); linear domain is
an exponentiate-in-view option. Epoch definitions are a separate tiny Parquet so
"what epochs exist" is inspectable without opening the big cache.

If the P1.2 storage check shows Parquet too big, fall back to HDF5/Zarr dense
`(epoch × window × channel × bin)` array per subject/session (epochs are
rectangular fixed 5-min windows, so dense arrays are efficient). Decide
empirically by building one subject both ways.

### Why the cache is per-window (the crucial subtlety)

Normalization order is: (1) baseline = mean power over 0-pain windows, (2) z-score
/ baseline-correct **each 2s window**, (3) THEN average over the epoch, (4) THEN
average over channels. Step 2 precedes step 3, and for nonlinear steps
normalize-then-average ≠ average-then-normalize (Jensen). So the cache retains
per-window granularity → z-score-vs-baseline, log-vs-linear averaging, binning,
and epoch-length all stay FREE recomputable views. (Note: purely *linear*
normalizations with a fixed per-channel baseline DO commute with averaging, so in
principle epoch-means + baseline stats could suffice — but per-window is kept
deliberately because (a) recompute over 56 subjects is measured at 1-2h, too slow
to repeat per variant, and (b) PAC and nonlinear sensitivity analyses need it.)

### Feature families are siblings, not a hierarchy

FOOOF/PAC/connectivity are NOT "derived from" the PSD cache — different inputs:

| Feature | Input | Where | Cost |
|---------|-------|-------|------|
| bandpower, HFA/high_gamma | PSD (aggregate bins) | view of PSD cache | cheap |
| 1/f slope (`polyfit`) | PSD (fit over stored log-bins) | cheap derived quantity of PSD cache | cheap |
| FOOOF exponent/offset | PSD at epoch scale (per-epoch fit) | own derivative of PSD family | moderate |
| PAC | bipolar **time-domain** signal (Hilbert) | OWN family off the signal | expensive |
| coherence/PLV | bipolar **time-domain** signal (pairs) | OWN family off the signal | expensive |

Each family: continuous extraction → epoch (slice) → cache (per-window) → views.
PSD is the first built; PAC/connectivity get `preprocessed/bipolar_pac/`,
`preprocessed/bipolar_connectivity/` and their own caches.

---

## PART 2 — Views (see view_registry.md for the full axis list)

A view is a cheap deterministic transform of the cache, a function in the repo,
composed at load time. The seven ordered axes: domain (log/linear) → baseline →
normalization (none/zscore/baseline-subtract) → epoch aggregation
(linear-then-log/log-direct) → frequency aggregation (bins/bands) → region
aggregation (theory sets/DK/global) → pain binarization (absolute/relative/
tertile/graded). Order matters: normalization is per-window BEFORE epoch-averaging.

### save_path rule (universal capability, guarded default)

Every view function takes optional `save_path=None`. Capability is universal;
DEFAULT is recompute (don't save) — a recomputed view can't go stale. When
`save_path` IS set, ALSO write a provenance+staleness sidecar (cache manifest
hash, view git commit, view config, date); on load, if current cache hash or git
commit differ from the sidecar → WARN/refuse. A bare `to_csv` with no sidecar is
forbidden. Materialize ONLY when recompute is measured slow AND something depends
on it. Two destinations: disposable performance cache → `features/.../views/`
(delete freely); terminal deliverable (model input, committee figure's table) →
that's an ANALYSIS output under `analysis/.../<run>/`.

---

## PART 3 — VIEW vs ANALYSIS vs STORED FEATURE (decision procedure)

Apply in order, stop at first match:

1. **TERMINAL?** A human looks at it, or a model consumes it as input? → **ANALYSIS**,
   save under `analysis/pain/<question>/<output_type>/<scheme>/<run>_<timestamp>/`.
2. **NON-TERMINAL + CHEAP transform of stored data?** → **VIEW**, recompute, don't
   save by default. (Holds even if multi-step / derived from another view.)
3. **NON-TERMINAL + EXPENSIVE** (new extraction, or expensive intermediate many
   depend on) → **STORED FEATURE / CACHE**, save under `features/`/`preprocessed/`.

"Terminal" = nothing in the pipeline consumes it further; the consumer is a human
eye or a model. Same computation (z-score) is a VIEW mid-chain, part of an
ANALYSIS when it's the last step before a human/model. **A view is a step, an
analysis is a stop.**

---

## PART 4 — Directory layout

**This tree lives on OAK, not in the repo.** It is rooted at
`/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/`. The repo
holds CODE ONLY — no data, caches, features, cohorts, analysis outputs, plots,
models, or results ever go in the repo. All output paths resolve to this Oak
base (via the config module), never to a repo-relative path.

```
derivatives/sisler/                          # ROOT = /oak/.../iEEG_EHR/derivatives/sisler/ (ON OAK, NOT THE REPO)
  preprocessed/                              # STORED FEATURE FAMILIES (continuous, per-window)
    bipolar_fft/       sub-XXX/ses-YYY/       # PSD (exists, log-power NWB)
    bipolar_pac/                              # PAC (future, own extraction off signal)
    bipolar_connectivity/                     # coherence/PLV (future)
  qc/                                        # MOVED OUT of analysis/ — QC is a data property
    raw_voltage/  {metrics,exclusions,masks,validation}
    bipolar/      {metrics,exclusions,masks}
    feature_level/                            # NEW: choice-independent channel-quality facts (PART 7)
  features/
    pain/
      psd_epochs/
        epoch-5min-pre_mask-<label>/          # BASE UNIT = epoch def + QC mask
          cache/        sub-XXX_ses-YY_epochs.parquet   # per-window log-power, masked, PRE-norm
          epoch_defs/   sub-XXX_ses-YY_defs.parquet     # index: run+window indices+pain label+mask ref
          manifest.json                       # window length, anchor, mask label, bin edges, dtype, git, date
          views/                              # OPTIONAL materialized views (only if recompute measured slow)
            <label>_<confighash>/table.parquet + staleness_sidecar.json
      pac_epochs/  ...                         # parallel per family (future)
  analysis/
    pain/                                     # organized by SCIENTIFIC QUESTION / RUN
      <question>/                             # exploratory-sweep | replicate-huang | phenotype-clustering | ...
        <output_type>/                        # heatmap | violin | effect_size_dist | glmm_binary | ...
          <view_scheme>/                      # optional (absolute|subject_relative | row-order | ...)
            <run_name>_<timestamp>/
              config.yaml                     # cache ref + view config + cohort ref + model spec
              provenance.json                 # resolved subjects[], git commit, parent cache manifest, date
              <outputs>                       # *.parquet, *.png (+spec sidecars), model.joblib, metrics.json, log.md
      sweeps/                                 # tiered nomination runs; grid = ROWS in results.parquet
      scratch/                                # throwaway exploration plots; timestamped, untracked, deletable
  cohorts/
    <name>_<date>.json                        # SAFE cohort files: IDs + cohort + SAFE matching axes (NO age; PART 6)
```

`<run_name>` = label + always a timestamp (never overwrite). Cohort membership
lives ONLY in `cohorts/*.json`, resolved into `provenance.json.subjects[]` —
never in folder names. NEW artifacts are Parquet (tables) / joblib (models); do
not bulk-convert old CSVs.

---

## PART 5 — analysis/ organization (5 levels)

1. `<event>/` — pain | mood | opioid | seizure.
2. `<question>/` — MUST match a named question in the exploration log / freeze doc.
   Do NOT open one without a named question — else `sweeps/` or `scratch/`.
   Discovery vs confirmation is NOT a level; it's a cohort ref in config.
3. `<output_type>/` — plot type OR model OR stats table; one per script+layout.
4. `<view_scheme>/` — optional small enumerable variant slot; omit if none.
5. `<run_name>_<timestamp>/` — one run: config.yaml + provenance.json + outputs.

**Asymmetry rule:** levels 1-2 opened deliberately; 3-5 created freely per run.
No folder-per-plot at 1-2. All combinatorial sweep pressure → ROWS in a `sweeps/`
results.parquet, never folders. Which subjects: read provenance.json subjects[],
never the folder name.

---

## PART 6 — Cohorts & the discovery/hold-out split

**Age is PHI and NOT on Sherlock**, so demographic matching happens OFFLINE
(PHI-side, where the master lives). Only the anonymized `subject_id → cohort`
assignment + SAFE matching axes cross to Sherlock. This mirrors the existing
PHI/SAFE catalog split. (Open: whether the offline matching script lives in this
same repo or a separate PHI-side repo — undecided.)

**Matching axes** (the confounds that must be balanced): pain-score range >4
(also an inclusion criterion for within-subject high/low analysis), sEEG vs ECoG,
age, sex.

**Staging:**
- **Now**: lock current discovery subjects into `cohorts/discovery-core-<date>.json`.
  These stay discovery permanently, even after new subjects arrive. The current
  informal hold-out is provisional (rough sanity check only, NOT the official
  confirmation set).
- **After the ~150 land**: build `cohorts/heldout-matched-<date>.json` offline by
  matching hold-out to discovery on {pain-range>4, sEEG/ECoG, age, sex}.
  Provisional hold-out subjects fold into either set as matching requires.

SAFE cohort JSON carries: subject IDs, cohort label, and SAFE matching axes
(pain range, electrode type) for auditability — NOT age. Hold-out is UNREACHABLE
by default in exploratory runs (a `--split`/cohort-file flag gates it).

---

## PART 7 — Feature-level outlier correction (structure; thresholds TODO)

**Choice-independent fact**: runs ONCE on the pre-normalization cached power (log
or raw, not on any view's normalized output), stored in `qc/feature_level/`,
inherited by ALL views. Same metric/threshold split as raw-voltage QC: store the
continuous flag-fraction metrics once, apply thresholds cheaply, sweep thresholds
without recompute.

Baseline for outlier detection: per-channel **recording-wide median + MAD**
(robust to seizures/pain events), computed across the whole recording — NOT
per-epoch (too few samples; would mask real effects).

Cascade (each threshold is a tunable parameter, values TODO — set on structural
grounds BEFORE looking at pain relationships):
- **window flag**: power > K×MAD above channel's recording-wide median  [K = TODO]
- **epoch flag** (per channel×epoch): > X% of the epoch's windows flagged  [X = TODO]
- **channel exclude** (global): > Y% of the channel's epochs flagged → drop channel everywhere  [Y = TODO]
- **epoch exclude** (across channels): after channel exclusions, > Z% of surviving channels flagged → drop epoch  [Z = TODO; ref EPOCH_MAX_EXCLUDED_FRAC, currently 0.5]

Scope split: global channel-quality facts → `qc/feature_level/` (inherited);
epoch-across-channel exclusion is also global here since it's on raw power.
Store as flags/exclusions parallel to raw-voltage QC. (For features that
genuinely require normalized values, a per-analysis exception — but FFT power is
the clean choice-independent case.)

---

## PART 8 — Rules for Claude Code (CLAUDE.md block)

```
LAYERS
- Feature families (PSD/PAC/connectivity): continuous, per-window, expensive, under
  preprocessed/. PAC/connectivity come from the TIME-DOMAIN signal (Hilbert), NOT FFT.
- Cache: a family sliced to event windows; per-2s-window, per-channel, per-bin
  LOG-power, QC-masked, PRE-normalization, PRE-averaging. Parquet, ONE file per
  subject/session (subject/session in filename; epochs stacked via epoch_id column).
  NOT NWB, NOT one-file-per-epoch. Immutable.
- Epoch definitions: tiny Parquet index beside the cache.
- Views: domain->baseline->normalize->epoch-avg->freq-agg->region-agg->binarize.
  FUNCTIONS in the repo, recomputed at load by default, NOT saved by default.

DECISION (stop at first match)
1. Terminal (human looks / model consumes)? -> ANALYSIS, save under analysis/.
2. Non-terminal + cheap transform of stored data? -> VIEW, recompute, don't save.
3. Non-terminal + expensive? -> STORED FEATURE/CACHE, save under features/.

VIEW FUNCTIONS
- Optional save_path=None; default recompute. If set, also write provenance+staleness
  sidecar (cache hash, view git commit, view config, date); warn/refuse if stale.
  Never bare to_csv. Materialize only when recompute measured slow AND depended on.

CACHE RULES
- Per-2s-window LOG-power, never epoch-averaged (Jensen; normalization is per-window).
- New cache (expensive re-run) only for new epoch length or new QC mask.
- build_pain_epoch_power.py: emit epoch defs + per-window Parquet cache; do NOT
  average, do NOT normalize (views do that). Store float32 after round-trip check.

analysis/ ORGANIZATION (5 levels)
1 <event>/  2 <question>/(named only)  3 <output_type>/  4 <view_scheme>/(optional)  5 <run>_<timestamp>/
- Levels 1-2 deliberate; 3-5 free per run. No folder-per-plot at 1-2.
- Sweep combinatorics = ROWS in sweeps/ results.parquet, never folders.
- Discovery vs confirmation = cohort ref in config, not a folder level.
- Which subjects: provenance.json subjects[], never the folder name.

COHORTS
- Age is PHI; demographic matching is OFFLINE. Only anonymized id->cohort + SAFE
  axes cross to Sherlock. Current discovery subjects locked as discovery permanently.
  Hold-out unreachable by default; gated by a --split/cohort flag.

FEATURE-LEVEL QC
- Choice-independent: on pre-normalization cached power, stored qc/feature_level/,
  inherited by all views. Recording-wide per-channel median+MAD baseline. Cascade
  window->epoch->channel->epoch-across-channels; thresholds TODO, metric/threshold split.

GIT / PROVENANCE
- Commit + push BEFORE any definitive/array run so the recorded hash matches the code
  that ran. Warn if tree dirty. Every stored artifact writes provenance (git+dirty,
  timestamp, parent ref, subjects[] for runs).

NAMING / IO
- Parquet for tables, joblib for models, JSON for manifests/sidecars. New artifacts
  only; don't bulk-convert old CSVs. Do NOT fingerprint runs/plots (label+timestamp);
  fingerprint ONLY materialized-view folders, only if recompute measured slow.
```

---

## Open decisions (resolve in Claude Science / with Claude Code)

1. Pin the QC mask before the big cache build (baked in → change = re-run).
2. P1.2 storage check: Parquet vs HDF5/Zarr for the per-window cache; confirm fit.
3. dtype audit: standardize cache on float32 after round-trip validation.
4. Offline cohort-matching code: same repo as Sherlock analysis, or separate PHI repo?
5. Feature-level QC thresholds (K, X, Y, Z) — set on structural grounds.
6. GLMM feature domain: likely log-power (for normalization); per-window cache defers it.
7. Huang-logic replication / GLMM design — its own future conversation, AFTER the
   exploratory sweep produces a frozen feature set. Not near-term.