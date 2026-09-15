# DECISIONS.md — iEEG_EHR_Analysis

**What lives here:** settled calls **and their reasons**. **Lifespan:**
permanent. **Append-only — never pruned, never reworded in place.** If a decision
is later reversed, append the reversal with its own reasons; do not edit the
original. The value of this file is that it records what you believed *at the
time*, which is what makes a later reversal legible.

Two flavors, both welcome:

- **Settled conventions** (fast) — "pinned mask X because Y." These are usually
  *also* configuration, so they live in `CLAUDE.md` or the config module too; the
  entry here records the **why**, which config can't hold.
- **Earned conclusions** (slow) — e.g. "high-gamma is the primary pain feature."
  These cannot be reached from one plot. They emerge from accumulated
  `docs/labnotebook/` narrative across many analyses, and land at P2.6 FREEZE.
  **The notebook is the path; the decision is the endpoint.**

A sweep result is a **nomination**, not a decision. Do not promote a nomination
here until it has survived the robustness reading (P2.3).

Hand-edited in v0. Newest entries at the bottom. Format: `## YYYY-MM-DD — <call>`
then **Why** and, if relevant, **Where it lives** / **What would reverse it**.

---

## 2026-07-27 — This repo holds CODE ONLY; all data lives on Oak

Every output path resolves under the Oak derivatives base
(`/oak/.../derivatives/sisler/`). No data, derivative, cache, plot, model, or
result file is ever written to a repo-relative path — including throwaway scratch
plots, which go to `analysis/scratch/` on Oak.

**Why:** the repo is git-tracked and pushed to GitHub, so writing data into it
risks committing PHI-adjacent artifacts to a remote, and bloats the repo besides.

**Where it lives:** `CLAUDE.md` (CODE/DATA BOUNDARY), `config/paths.py`.
`.gitignore` is a *backstop*, not the mechanism — the correct behavior is that
nothing writes data into the repo in the first place.

## 2026-07-27 — One installable package at `src/ieeg_ehr`, invoked as `python -m`

Restructured from loose top-level directories into a single installable package;
all `.sbatch` in `sbatch/`; jobs never `cd` into the repo.

**Why:** an editable install makes imports work identically from a login node, a
compute node, and a test, without `sys.path` surgery or cwd assumptions — which
is what was breaking sbatch jobs.

**Deviation from the original plan, deliberately kept:** `config/` sits *inside*
the package rather than at the repo root, because a root-level `config/` is not
importable from an installed package.

## 2026-07-27 — Notebooks are retired, not kept as live scratch

Existing notebooks moved to `outdated/notebooks/`; no new ones.

**Why:** the original plan kept `notebooks/` as a live scratch area. In practice
notebooks hid state, escaped provenance (no commit hash on a cell that produced a
figure), and duplicated logic that belonged in the package. Superseded code is
kept but never imported.

## 2026-07-27 — Parquet for tables, joblib for models, JSON for sidecars

Never pickle tabular data.

**Why:** Parquet gives partial column reads, which the view layer needs (it slices
by frequency bin). Raw pickle is version-fragile, non-portable, and unsafe from
untrusted sources.

**Note:** as of this date `io/tables.py:save_table` still writes CSV because
pyarrow isn't in the venv yet (P0.3). New artifacts only — existing CSVs are not
bulk-converted; convert one when it is next touched.

## 2026-07-27 — The cache stores per-window LOG-power, pre-normalization

Never epoch-averaged and never normalized *in the cache*; normalization is
per-window and happens before averaging, in the view layer.

**Why: Jensen's inequality.** Averaging then normalizing is not the same
operation as normalizing then averaging, and only the latter is what the analysis
means. Baking either step into the cache would silently commit every downstream
view to one choice — and the cache is expensive to rebuild.

**What would reverse it:** nothing short of a new epoch length or a new QC mask,
which are the only two reasons to rebuild the cache at all.

## 2026-07-27 — Current discovery subjects are locked as discovery PERMANENTLY

The hold-out cohort is unreachable by default in exploratory runs, gated behind an
explicit `--split` / cohort-file flag.

**Why:** these subjects have already been looked at. That cannot be undone, so
they can never serve as a hold-out. Making the hold-out *unreachable by default*
rather than merely discouraged is the only version of this rule that survives a
tired afternoon.

**Corollary:** which subjects were in a run is read from `provenance.json`
`subjects[]`, never inferred from a folder name.

## 2026-07-27 — Age is PHI; demographic matching happens offline

Only the anonymized `subject_id → cohort` assignment plus SAFE matching axes cross
to Sherlock.

**Why:** age cannot come to Sherlock at all, so the matching computation has to run
where the PHI master lives. Nothing reachable on Oak or Sherlock contains PHI —
the allowlist is enforced by `phi_manifest.py`, which lives on the PHI side,
outside this repo, upstream of anything that crosses over.

**Extended to prose 2026-07-27:** the same discipline governs the tracking files.
Notebook entries, scratchpad items, tasks, and decisions use deidentified
references only — anonymized subject IDs, and the 2001-anchored timeline rather
than real clinical dates. That is what makes these files safe to commit to a
GitHub remote.

## 2026-07-27 — Adopt the lab-notebook / project-tracking system (v0)

Five cockpit files at the repo root (`CLAUDE.md`, `PLANNING.md`, `TASKS.md`,
`SCRATCHPAD.md`, `DECISIONS.md`), a flight log under `docs/labnotebook/` plus
`docs/analyses_run.md`, per-figure `.notes.md` sidecars on Oak, and five thin
commands (`/lognote`, `/annotate`, `/addtask`, `/addscratch`, `/standup`).

**Why:** every record gets exactly one home, chosen by *what the thing is* — which
is the same question as *how long it lives*. The failure mode being avoided is a
single undifferentiated log that mixes permanent rules with today's half-thought,
which makes both unreadable. The governing design constraint is that **the lazy
path must be fully functional**: every `/lognote` prompt is skippable, because a
tool that demands five prose answers gets avoided and then the whole system rots.

**Deferred on purpose:** `/logdecision`, `/updateplan`, the HTML figure viewer,
Slack sharing. `DECISIONS.md` and `PLANNING.md` are hand-edited until the manual
version becomes annoying.

**What would reverse it:** real usage. Nothing here is expensive to change — it is
plain-text files and thin commands. Reshape before adding anything deferred.

## 2026-07-27 — The cache stores float32; views COMPUTE in float64 (P0.6)

Two rules, settled together because the audit that produced one produced the
other:

1. **Storage is float32.** The per-window cache stores log-power as float32.
2. **Views upcast to float64** before any epoch average/reduction, and before
   exponentiating log-power back to linear.

**Why (1) — measured, not assumed.** A full float64 recompute of one run's PSD,
compared against the production float32 path's epoch averages, agreed to **8.1
significant figures**: worst-case relative error 8.3e-09, which is a fractional
error of **2.5e-07 in linear power**, or 0.14 float32 half-ulps. float32 halves
the cache against float64 for an error four orders of magnitude below anything an
effect size could resolve. float32 also round-trips **bit-exactly** through both
Parquet and HDF5 (both carry IEEE-754 binary32 natively) — verified rather than
inferred.

The end-to-end error is *better* than float32's own ~7.2 digits because
per-window rounding is independent and **averages down** over ~300 windows. That
is the same fact that forces rule (2): accumulator error **grows** with the
number of terms instead.

**Why (2).** A float32 accumulator over a ~5-minute epoch holds only **6.0
significant figures** — at/just below the 6-sig-fig bar this task set, and the
largest precision loss anywhere in the chain. It is not an argument for storing
float64; it is an argument for upcasting at the point of the reduction, which is
free. **numpy does not do this for you**: for float32 input it accumulates in
float32, so the naive `arr.mean(axis=0)` is the lossy version. Separately, the
worst stored log-power observed was **-36.8** (a near-dead channel), leaving only
~1.1 decades above float32's smallest normal — so `10**log_power` in float32 sits
close to underflow, and a later baseline division could silently produce an exact
zero.

**The trap this avoids:** reading the 6.0-sig-fig accumulator result as "float32
is too narrow for the cache." Storage precision and accumulator precision are
different questions with opposite scaling in the number of windows, and
conflating them would have bought a 2x larger cache and still left the real
error — the accumulator — in place.

**Where it lives:** `config/cache_params.py` (`CACHE_FLOAT_DTYPE`,
`CACHE_ACCUMULATE_DTYPE`, `CACHE_LINEAR_DOMAIN_DTYPE`), `CLAUDE.md` (cache +
view rules). The audit is `ieeg_ehr/features/dtype_audit.py`, re-runnable;
output at `$DERIV/qc/feature_level/validation/dtype_audit/p0.6_2026-07-27T160009`.

**What would reverse it:** a feature family whose stored values are NOT
log-scaled and span a much wider dynamic range (float32's exponent range is what
makes log-power comfortable), or a downstream method that genuinely needs more
than ~7 digits of a *stored* value — neither of which is in view. Note the
audit's own scope: leg D compared 8 bipolar pairs of one run of one subject.
It is a precision claim about the arithmetic, which does not vary across
subjects, not a survey.

## 2026-07-27 — Every artifact write goes through `ieeg_ehr.io` and carries a sidecar (P0.3)

`io.write_table` / `io.save_model` / `io.write_manifest` / `io.write_run_provenance`
write the artifact and its provenance JSON in the same call; `io.read_table` /
`io.load_model` / `io.assert_fresh` check staleness on the way back in. One
envelope shape (`schema_version, kind, created, script, git, params, config_hash,
parents[], subjects[]`) in three homes: `<file>.provenance.json`,
`<dir>/manifest.json`, `<run_dir>/provenance.json`.

**Why one writer instead of a documented convention:** the rule "never a bare
`to_parquet`" was already written down and already being broken — nine existing
writers emit a table with nothing beside it. A rule that requires remembering an
extra call gets skipped under time pressure; making the sidecar impossible to
omit (it is in the same function call) is the version that survives.

**Why parents are fingerprinted, not content-hashed:** a per-window cache file is
hundreds of MB to GB. sha256-ing it on every write, and again on every staleness
check, would cost more than recomputing the view the check exists to guard. So a
parent reference is `(path, bytes, mtime)` plus a real digest only for small
files — and view staleness is defined against the **cache manifest's** digest,
which is cheap by construction. `io.file_digest` refuses files over 64 MB so that
guarantee cannot quietly erode.

**Why staleness warns rather than refuses by default:** the safe fallback is
always "recompute," and a recomputed view cannot be stale — which is why views
default to not saving at all. A hard failure on every commit-drift would make an
exploratory session unusable; `on_stale='refuse'` is there for anything a
reported number comes out of, and models/views default to comparing the commit
because for those the code *is* the numbers.

**Sidecar naming:** the suffix is APPENDED (`x.parquet.provenance.json`), not
replaced. Replacing collapses `x.parquet` and `x.csv` onto one sidecar name —
exactly the collision this repo's "convert one CSV when you next touch it" policy
walks into. Readers still resolve the pre-P0.3 replaced form, which is what the
legacy pain caches have on disk.

**The QC tree stays CSV.** ~85 subject-sessions of per-window metrics,
exclusions, and masks with a working metric/threshold split; converting them
would invalidate on-disk artifacts for no analytical benefit. `save_table` now
dispatches on the file extension, so existing `.csv` call sites are untouched
while new code gets Parquet. `append_table` stays CSV by nature — Parquet has no
append-a-few-rows mode, and the streaming metrics writers need one. This narrows
the original P0.3 task ("switch `save_table` to Parquet") on purpose.

**Deps:** pyarrow 20.0.0 + joblib 1.5.3 into the shared venv, `--no-deps
--only-binary=:all:` so numpy 2.4.2 / pandas 2.3.3 / pynwb are untouched. Sherlock
is CentOS 7 (**glibc 2.17**) and modern pyarrow wheels are `manylinux_2_28`, so a
plain `pip install pyarrow` tries a source build and dies on a missing Rust
toolchain; `--only-binary=:all:` makes pip back off to the newest version that
still ships a `manylinux2014` wheel. `io.tables`/`io.models` raise that exact
recipe if the import fails.

**Where it lives:** `docs/io_conventions.md` (the contract + API),
`src/ieeg_ehr/io/{sidecar,tables,models}.py`, `CLAUDE.md` (IO / naming),
`config/paths.py` (`pain_epoch_*` cache paths, `analysis_run_dir` /
`sweep_run_dir` for the 5-level scheme). Tests: `tests/test_io_conventions.py`.

**What would reverse it:** the P1.2 storage check choosing HDF5/Zarr over Parquet
for the cache — that changes `write_table`'s backend for the cache only, not the
sidecar contract, which is format-agnostic on purpose.

---

## 2026-07-28 — Exclude the 60s-hop PSD runs from analysis; re-run their PSD

**Decision:** the runs whose stored `psd_log_bins` has `rate = 1/60` (a 60-second
hop, from the superseded 60s outer-window design) are **excluded from analysis**.
Their PSD is to be **re-run** under the current 2 s window / 50% overlap scheme
(`config/psd_params.py`), and until that lands their epochs do not enter any view,
sweep, or figure.

**Scope as measured** (backfill audit, array `36197924`, 2937 runs across all 83
subject-sessions — `docs/labnotebook/2026-07-28.md` 12:55):

| | 1 s hop | 60 s hop (excluded) |
|---|---|---|
| sub-247 ses-01 | 39 epochs, 300 windows, 33 runs | **19 epochs, 5 windows, 13 runs** |
| sub-257 ses-01 | 37 epochs, 300 windows, 25 runs | **14 epochs, 5 windows, 11 runs** |

The other 81 subject-sessions are uniformly `starting_time=0.0, rate=1.0`.

**Why exclude rather than keep and annotate:** a 5-window epoch mean and a
300-window epoch mean are not the same feature. They share a column name
(`value`), a units label, and a `pain_bin`, but differ by ~sqrt(60) in the noise of
each estimate, and the 5-window epochs sample a 5-minute window at 60x coarser
time resolution. Pooling them means a region's average silently mixes two feature
definitions, and any per-subject effect size for sub-247/sub-257 is a blend of the
two. Down-weighting instead of excluding was rejected because the correct weight
depends on the very noise structure the mixture obscures.

**Why re-run rather than drop the runs permanently:** the raw voltage is intact —
only the derived PSD is stale — so this is a recompute, not lost data. 24 runs is
cheap next to a 33-epoch loss across two subjects, and sub-247/sub-257 otherwise
have healthy 1s-hop coverage (39 and 37 epochs) that would be weakened by dropping
the subjects wholesale.

**Why this was findable at all:** `epoch_start_sec`/`hop_sec` are stored PER RUN in
`epoch_defs` and audited against the expected `(0.0, 1.0)` rather than assumed. The
shortcut on the table was hardcoding a 1 s hop from the manifest's window/overlap
params, which would have (a) misaligned the 60 s QC mask join by 60x for these runs,
silently, and (b) left the feature mixture invisible. Recording this because it is a
concrete case where "store the observed value and check it" beat "derive it from
config".

**Where it lives:** `TASKS.md` (the re-run + the exclusion gate),
`docs/labnotebook/2026-07-28.md` (12:55, the audit that found it),
`src/ieeg_ehr/features/backfill_epoch_defs_timing.py` (the audit itself, re-runnable).

**What would reverse it:** the PSD re-run completing for those 24 runs, at which
point the epochs become ordinary 1s-hop epochs and the exclusion gate is deleted
rather than kept. If a future analysis deliberately wants coarse-time-resolution
features, that is a NEW feature family with its own epoch definition, not a
re-admission of these rows.

---

## 2026-07-28 (addendum) — CORRECTION to the 60s-hop rationale above

The entry above states the 5-window epochs have "~sqrt(60) more noise per
estimate." **That is wrong**, and the record of it stays because this file is
append-only. The decision it justified — exclude, then re-run the PSD — is
unchanged; only the reason is.

**What the files actually are.** Read off the NWB `DecompositionSeries`
descriptions (sub-247 has one of each, which is how this was settled):

    superseded  {"outer_window_sec": 60.0, "inner_segment_sec": 2.0, "overlap_frac": 0.5}
                two-level: a 60 s outer window of ~59 overlapping 2 s inner
                segments, Welch-AVERAGED into one spectrum per minute
    current     {"window_sec": 2.0, "overlap_frac": 0.5, ...}
                single-level: each 2 s window is its own periodogram, stepped 1 s

So each 60 s value is an average of ~59 segments and is therefore *less* noisy per
value, not more; and a 5-minute epoch under the old design covers ~295 inner
segments — comparable raw data to 300 windows of the new design. The sqrt(60)
claim inverted this.

**The real reasons to exclude:**

1. **Different estimator, with Jensen frozen into storage.** The old files hold
   `log(linear-mean of ~59 segments)`; the new hold `log(single 2 s segment)`. An
   epoch mean is then approximately `log(arithmetic mean)` versus a geometric mean
   of per-second values. That is exactly the AXIS 4 log-vs-linear choice — except
   baked into the file, where no view can undo it. The whole point of the
   per-window cache is that this choice stays a free recompute; these runs remove
   that freedom.
2. **QC granularity.** A 60 s window maps 1:1 onto a 60 s mask bin, so masking is
   all-or-nothing per minute, and `EPOCH_MAX_EXCLUDED_FRAC` operates over 5 values
   rather than 300.
3. **Feature-level QC** computes its per-window z metrics on 60 s windows instead
   of 2 s — a different distribution feeding one threshold.

**Also corrected: the cascade is narrower than claimed.** The entry above implies a
PSD re-run invalidates "bipolar variance → std10 → bipolar mask." It does not. The
bipolar variance metric is computed on the **time-domain** signal, and sub-247's
metric CSV is on a 2 s grid (`window_start_time` = 0, 2, 4, 6, 8 …) even though its
PSD is 60 s. The real cascade is PSD → epoch cache → views, plus
`qc/feature_level/` power metrics. So the re-run should pass
`--skip-variance-metrics`.

**Mechanism, for the record:** an incomplete reprocessing pass, not corruption.
`run_pipeline_bipolar.py` has no skip-if-exists and no `--runs` flag, so a partial
re-run leaves both designs on disk with no complaint — which is why the audit
(`qc/psd_timing/`) is derived per RUN rather than per subject.

**Where it lives:** `src/ieeg_ehr/qc/psd_timing.py` (the check + `assert_subject_ok`),
`src/ieeg_ehr/qc/audit_psd_timing.py` (the cohort sweep + re-run list),
`docs/labnotebook/2026-07-28.md`.

---

## 2026-07-28 — Discovery cohort LOCKED at the documented 65 (P0.2)

**Decision:** `cohorts/discovery-core-2026-07-28.json` holds the permanent
discovery set = the 65 subjects of `cohorts/legacy/subjects_65.txt`. Everything
else is **`unassigned`**, NOT hold-out. `--split {discovery,unassigned,all}` gates
analysis, default `discovery`; `--split heldout` RAISES.

**Why the 65 and not the 60.** Only 60 of them have legacy analysis output; five
(`122 138 212 235 259`) were drawn into the cohort but never produced any, so on a
strict "has been seen" test they could have remained hold-out-eligible. They are
discovery anyway: the cohort was DEFINED by a documented random draw (15 forced +
50 sampled, seed 20260723, from an 82-subject mask pool —
`cohorts/legacy/selection_provenance.json`), and withholding the members that
happened to fail processing would make discovery a survivorship-filtered subset of
its own sampling frame. They are unprocessed discovery subjects, and are recorded
as `selected_not_analysed` so the distinction survives.

**Why the rest are `unassigned`.** The matched hold-out is built OFFLINE on the PHI
side, matching on {pain-range, sEEG/ECoG, age, sex}; age is PHI and is not on
Sherlock (PLANNING P4). So no code here may assert hold-out membership.
`--split heldout` raises rather than returning the leftovers, because silently
equating "not discovery" with "matched hold-out" would redefine the comparison set
as whatever happened to be left over.

**Splits gate ANALYSIS, not preprocessing.** QC, masks and PSD extraction
legitimately run over every subject on disk. Views, sweeps, models and figures do
not.

**An explicit `--subjects` list is still checked** against the split
(`assert_split_allowed`). Without that the flag would be advisory, and hand-naming
a hold-out-eligible subject would work — which cannot be undone.

**Consequence already incurred, recorded here because it bears on the cohort:**
two `unassigned` subjects, **sub-222 and sub-231**, were included in the P1.3
timing runs and appear in the group and per-subject heatmaps of 2026-07-28
(`analysis/scratch/view_heatmap/subject_relative/p13_std10_*`). The gate did not
exist yet, and the sample was drawn from "subjects with cache+mask" rather than
from a cohort. No analytic choice was made from those figures — they were a
plumbing/timing validation — but the data has been looked at. Their status needs an
explicit call (see `TASKS.md`); it is not resolved by this entry.

**Where it lives:** `src/ieeg_ehr/config/cohorts.py`,
`cohorts/discovery-core-2026-07-28.json`, `views/build_pain_epoch_view.py`
(`--split`).

**What would reverse it:** nothing reverses the discovery lock — that is the point.
A *different* cohort must be a NEW dated file, never an edit of this one, so any
artifact citing this filename always means the same 65 subjects.

---

## 2026-09-03 — Analgesic medication analysis: drug set, dose units, day 0, denominator

Six calls made while building `src/ieeg_ehr/med_analysis/` (level-1 event `meds`,
question `administration_patterns`), adapted from a colleague's benzodiazepine
analysis at `/home/groups/ckeller1/sisler/iEEG-EHR_Code/med_admin/`.

**1. Analgesics only; anesthetics excluded.** The MAR export does not capture
procedural medication. Across all 98 sessions there is 1 propofol administration,
3 rocuronium, 21 lidocaine (mostly topical/uro-jet), no ketamine, no
dexmedetomidine, no remifentanil, and **not one row with a populated
`infusion_rate`**. There is no anesthetic exposure to analyze. The classes stay in
the taxonomy so the exclusion is a visible predicate
(`med_taxonomy.ANESTHETIC_SUBCLASSES`), not a missing row.
*Reverses if:* an anesthesia record export lands separately.

**2. Doses stay in native units and are never pooled.** 516 of 1,754 analgesic
administrations (every combination product) are dosed in `tablet` or `Film`;
fentanyl is in `mcg`; the rest in `mg`. Product strength lives in the drug NAME,
not in a column, so mg for a combination product is only recoverable by parsing
"5-325" out of the product string. No MME conversion. Every dose axis is per
(drug × route); `load.assert_single_unit` refuses a mixed-unit pool rather than
trusting the caller. Fig 3's fraction-of-personal-max normalization is unit-free,
which is what makes that panel legitimate at all.

**3. Hospital day 0 = midnight of the session's own `session_start` date.** The
colleague's code hardcodes `EPOCH_DATE = 2000-01-01`, on the grounds that
de-identification shifts every admission onto that date. That holds for 95 of 98
sessions; two start 2000-01-05 and one starts 1999-12-31, and those get shifted or
negative day indices under a global constant. Per-session anchoring is identical
wherever the assumption holds. Day 0 is therefore *the calendar day the iEEG
session began* — captions must not call it "admission".

**4. Cohort = every subject with a MAR export** (96 subjects / 98 sessions),
defined by the glob, not by `TFR/incl_subjects.csv` or the discovery lock. The
question is what was administered in this dataset; a subject whose recording
failed QC still received the same drugs. This is a DESCRIPTIVE EHR
characterization with no neural data in it, so the discovery/hold-out split does
not apply — that gate exists to stop hold-out neural data being looked at.

**5. Recorded-hours denominator is registry-where-available, session-span
otherwise.** `sherlock_file_registry.csv` only populates `start_datetime` for runs
that have a PREPROCESSED file — all 2,136 null-timestamp rows have
`has_preprocessed == False` — so registry timing measures *preprocessed* coverage,
not *recorded* coverage. Of the 98 MAR sessions, 41 are fully timestamped, 41
partial, 16 have none at all, and an untimestamped run cannot be placed on a
hospital day. Pure gap-aware would give 16 sessions a zero denominator. So:
gap-aware union where a session has ≥2 timestamped runs covering ≥50% of its runs,
MAR `session_start`→`session_end` span otherwise, with the method recorded per
session and the split logged. Where the registry IS complete, coverage is a median
0.965 of span — so span-based sessions overstate monitoring by ~3.5%.
**Rates from Fig 3 are accurate to a few percent, not exact.**
*Reverses if:* the raw-NWB span extraction in TASKS.md is built — `session_start_time`,
`starting_time`, `rate` and data shape are all present in the raw files (checked),
so true gap-aware coverage for every session is available for the cost of one array job.

**6. Tables under `analysis/` are CSV, not Parquet** (repo-wide, not just here).
Small, terminal, read by eye; Parquet costs a pyarrow round-trip to open a 20-row
table and buys nothing. The cache, views, `features/` and `preprocessed/` stay
Parquet — large, column-sliced, dtype-critical. `io.write_table` already
dispatched on the path extension and emits the sidecar either way, so no IO code
changed. Sidecars stay JSON. See `CLAUDE.md` "IO / naming" and
`docs/io_conventions.md` "Which format, and why".

**Where it lives:** `src/ieeg_ehr/config/med_taxonomy.py`,
`src/ieeg_ehr/med_analysis/`, `sbatch/med_figures.sbatch`,
`tests/test_med_analysis.py`.

**Validation that these choices did not break the port:** the loader reproduces
the colleague's independently published corpus totals exactly — 98 files, 7,340
MAR rows, 421 multi-product rows collapsed, **6,919 unique administrations**, and
**380 benzodiazepine administrations**, with zero unmatched drug names
(`tests/test_med_analysis.py::test_loader_reproduces_published_corpus_totals`).

---

## 2026-09-03 — Linking a dose to the pain score before it: window and ties

Two calls, both made on measurements rather than on plausibility, for the new
level-2 question `pain_coupling` (`med_analysis/pain_link.py`).

**1. The lookback is 30 minutes, and it is an INCLUSION CRITERION.** Matching an
administration to "the most recent prior score" with no cap is not a small
approximation: across the four most-administered analgesics the median gap from
dose back to assessment is 1.3 h, but p90 is 4.9 h and the tail reaches 19 h, so
an uncapped join attributes doses to day-old assessments. 30 minutes was set by
request. The consequence is that an administration with no assessment in the
preceding 30 min is **dropped, not imputed** — 396 of 1,509 (26%) — so
`link_to_prior_score` returns a frame with those rows already removed rather
than a `pain_score` column full of NaNs for a caller to remember about.
Consequently Fig 5's totals do NOT reconcile with Fig 1's, and the excluded
count is printed on the figure so the gap is not mistaken for a data loss.
*Reverses if:* the window changes — it is a `--window-minutes` flag, and the
figure title and footnote both read from it.

**2. A pain score stamped in the same minute as the dose counts as prior.** 45%
of administrations (679 of 1,509) are exactly this case. Charting is
minute-resolution and the nursing sequence is assess -> administer -> chart both,
so a same-minute score is the assessment that prompted the dose, and a gap of
zero is "within 30 minutes prior". This was measured both ways before choosing:
excluding exact matches drops the linked sample from 1,113 to 496 and leaves
every per-drug distribution AND their ordering unchanged (medians stay
acetaminophen 4, hydrocodone-acetaminophen 6, oxycodone 6-7, hydromorphone 8).
It therefore buys sample size, not a conclusion. Kept reachable as
`--strict-prior`, and `n_linked_if_strictly_prior` goes into every run's
provenance so the choice is auditable from the artifact alone.
*Reverses if:* evidence appears that same-minute scores are post-dose
reassessments rather than pre-dose ones — the median gap being exactly 0.0 min
for all four drugs is consistent with paired charting but does not prove
ordering within the minute.

**3. This question is NOT causal, and the code says so in three places**
(module docstring, figure footnote, run provenance). A score preceding a dose
does not make it the reason for the dose: scheduled drugs are given on a clock
whatever the assessment says, and an assessment is often charted precisely
because a PRN dose was requested — the arrow can point either way and this
table cannot separate them. Per CLAUDE.md the ordering it shows is a
NOMINATION, not a finding.

**Where it lives:** `src/ieeg_ehr/med_analysis/pain_link.py`,
`src/ieeg_ehr/med_analysis/plot_pain_score_bars.py`,
`sbatch/med_pain_coupling.sbatch`, `tests/test_med_pain_link.py` (15 tests),
`config.pain_score_files()`, `config.MED_PAIN_QUESTION`.

---

## 2026-09-04 — Attributing a dose to one assessment (Fig 7)

**Each administration belongs to its NEAREST PRECEDING assessment.** Fig 7 asks
the forward question — given an assessment, was an analgesic given within 30
min — so two assessments inside one window could both claim the same dose and
the percentages would stop being a partition. The rule chosen is provably
equivalent to truncating each assessment's window at the next assessment
("the dose whose closest earlier assessment is this one" and "a dose inside
(t_i, min(t_i+30, t_i+1))" select the same rows), so nothing is double-counted.
The cost is explicit and stated on the figure: an assessment followed five
minutes later by another and then a dose reads as "no analgesic", because the
dose answered the later assessment.

Measured before choosing: only 7.8% of assessments have a neighbour inside 30
min, median gap to the next assessment is 120 min, and `--exclude-clustered`
(dropping those assessments outright) moves the overall responded share 24.7% →
24.2%, with per-score deltas under 2 points except score 10, which moves 9
points on n=57 → 45. The rule does not drive the result.
*Reverses if:* a finer-grained charting timestamp appears, making within-minute
ordering recoverable.

**Observability comes from the MAR export EXISTING, not from it containing
analgesic rows.** Three of the four sessions with no analgesic rows do have an
export; those assessments are genuine "no analgesic given" and belong in the
denominator. Only 1 assessment is truly unobservable. Assessments whose window
runs past `session_end` are dropped as right-censored (22, 0.5%) — absence
there is unearned rather than false.

**"No analgesic" is computed against EVERY analgesic**, not the seven coloured
drugs, with the remainder pooled as "Other analgesic"; otherwise a morphine dose
would read as nothing having happened. "Two analgesics" is its own segment
because 4.8% of responded-to assessments have two distinct drugs and a stacked
bar has to assign each assessment exactly one segment for the bar to mean 100%.

**Where it lives:** `pain_link.session_bounds`,
`pain_link.response_by_assessment`, `plot_pain_score_response.py`,
`tests/test_med_pain_response.py` (12 tests).

---

## 2026-09-08 — A `<scope>` level in the analysis tree

**`analysis_run_dir` gains an OPTIONAL sixth level between the question and the
output type, used only by `decoding`.** The five-level scheme assumes each
question has one taxonomic axis below it. `decoding` has two that are genuinely
independent: the scope a model is fitted at (`individual_subject` now,
`generalizable` anticipated) and the model family (`regression` / `ordinal` /
`classification`). Encoding both in one folder name — `individual_regression` —
is exactly what `view_registry.md` argues against for view schemes: unreadable,
and still not a complete description.

Verified safe before changing it: **nothing in the repo reads this tree by
depth.** Every consumer builds paths through `analysis_run_dir` and passes them
around; `med_state.py` hand-assembles one path but constructs rather than
parses; and CLAUDE.md already forbids inferring run membership from a folder
name ("read `provenance.json` `subjects[]`, NEVER the folder name"). The only
thing pinning five levels was a test on the builder's own contract. So this is a
deliberate contract change, not a breakage.

Default is `None`, so every pre-existing path is byte-identical — pinned by
`test_optional_scope_adds_a_sixth_level_and_defaults_off`.

*Reverses if:* a second question wants a scope level for a reason that is really
a sweep axis. A level is justified only when the two axes produce different
output SCHEMAS; when they share one, they are rows (which is why `cv_scheme` and
`outcome_scaling` stay inside the results table).

**Where it lives:** `config/paths.py:analysis_run_dir` (`scope=`),
`config.decoding_run_dir` + `DECODING_ARMS`, `tests/test_io_conventions.py`
(2 tests), CLAUDE.md "analysis/ organization", `docs/architecture.md` PART 5.

---

## 2026-09-08 — `paper_bands_6`, and the settling of the band DISCREPANCY

**The paper's six bands become their own frequency-axis value rather than
replacing `CANONICAL_BANDS_HZ`.** `psd_params.py` has carried a "DISCREPANCY —
unresolved" note since 2026-07-27: `docs/architecture.md` and
`docs/view_registry.md` described the canonical bands as delta 1-4 / theta 4-8 /
alpha 8-12 / **beta 15-25 / gamma 25-70 / high_gamma 70-170**, while the code
used an eight-band set splitting gamma to avoid the 60 Hz harmonics. The docs
were describing Prasad et al. 2025's edges; the code was describing this
project's revision of them. Neither was wrong — they were two different band
sets under one name.

Both now exist under their own names, selected by `axes.bands_for()`, and the
note is settled. **Which set is better is an empirical P2.2 sweep axis, not a
documentation bug.**

**`paper_bands_6` is only correct with `drop_line_noise_bins=True`.** Its gamma
(25-70) and high_gamma (70-170) straddle 60 and 120 Hz. And the two band
aggregators differ: `preprocessing.bipolar_bands.aggregate_to_bands` masks
flagged bins inline, but `views.axes.aggregate_bands` — the one the view layer
uses — does NOT, relying instead on the flag having already NaN'd them and on
`nanmean` skipping them. Measured on synthetic input, a single un-dropped 60 Hz
bin moves the gamma estimate by more than a full log unit. The 12-15 Hz gap
between the paper's alpha and beta is reproduced deliberately, not closed.

*Reverses if:* the P2.2 comparison shows one band set dominating, in which case
the loser becomes historical rather than an option.

**Where it lives:** `config/psd_params.py:PAPER_BANDS_6_HZ`,
`views/axes.py:bands_for`, `views/view_config.py:FREQ_AGGS`,
`views/build_pain_epoch_view.py`, `tests/test_views.py` (5 tests),
`docs/view_registry.md` AXIS 5.

---

## 2026-09-08 — Channel/epoch eligibility for the decoder: Y=0.2, Z=0.5, impute the residue

**Set on STRUCTURAL grounds from the missingness distribution, before looking at
any pain relationship** (CLAUDE.md; `architecture.md` PART 7 left K/X/Y/Z as
TODO). Measured over the 45 discovery subject-sessions with >=30 epochs under
mask `std10_rv-gross-std3_satmargin15_sw_logz4`: a cohort-mean 2.4% of
channel-epoch cells are mask-excluded at `max_excluded_frac=0.5` (worst subject
9.3%).

**Z = 0.5 because the epoch distribution has a GAP and the threshold therefore
does no work.** Per-epoch bad-channel fraction: median 0, p95 = 0.067,
p99 = 0.689. Essentially nothing lives between 7% and 69% — epochs are either
nearly clean or almost entirely destroyed. Z anywhere in [0.2, 0.7] drops the
same ~1% of epochs (1.45% -> 1.02%), so 0.5 sits in the middle of the empty gap
and is maximally insensitive to the exact value. It also happens to be the
existing `EPOCH_MAX_EXCLUDED_FRAC`, which is now structurally justified rather
than merely conventional.

**Y = 0.2 because channels are cheap and epochs are not.** The channel tail is
smooth (p95 = 0.086, p99 = 0.283), so Y does real work: 0.2 drops 1.57% of
channels, 0.5 drops 0.21%. For a per-subject decoder a dropped channel costs 6
features of ~918 — the penalty barely notices — while a dropped epoch costs ~2%
of a ~48-observation training set, and observation count is what decides whether
the model is fittable. So buy cleanliness with channels. This asymmetry is
specific to the decoder and is the opposite of what a group-level heatmap wants.

Channels are judged FIRST, then epochs against the survivors, because a bad
channel is a persistent property of the electrode and a bad epoch is a transient
event; judging transients before removing persistent faults makes every epoch
look bad. Result: 98.9% of epochs and 97.2% of channels retained, and **no unit
falls below 30 epochs** (minimum 33), so the cohort survives QC intact.

**Residual cells are IMPUTED, not dropped, and the numbers are lopsided enough
that this is forced.** After both rules, 0.82% of cells remain bad — but they
are scattered across 23.6% of epochs (up to 50% for one unit). Dropping rows to
purge them would cost a quarter of the training data to remove under 1% of
cells. So: channel median across retained epochs, **fitted inside the training
fold only**, with the imputed count logged per unit. No unit has zero residual
cells, so there is no clean-subset alternative.

**Coverage is not artifact.** 2 of 45 units have montages that differ across
runs (189: 120 of 230 channels absent from some runs; 167: 26 of 169) and their
apparent channel loss is entirely that, not badness. Those get restricted to
channels present in the runs contributing retained epochs, BEFORE Y applies — a
channel that was never recorded must never be imputed.

*Reverses if:* P0.1 pins a different raw-voltage mask. Every number above is
conditional on `_sw_logz4`; re-measuring is a ~10-minute job.

**Where it lives:** measured by scratch probes (not committed);
`docs/labnotebook/2026-09-08.md`. Thresholds to be consumed by
`src/ieeg_ehr/decoding/`.

---

## 2026-09-15 — CORRECTION: the target paper is Huang et al., not "Prasad et al."

**Every earlier entry in this file that cites the decoding replication's source
paper as "Prasad et al. 2025" is WRONG. The correct citation is Huang et al.
2025**, *Naturalistic acute pain states decoded from neural and facial
dynamics*, Nat Commun 16:4371, doi 10.1038/s41467-025-59756-5 — Yuhao Huang
first author, Corey Keller last author. It is a lab paper.

Appended rather than corrected in place, per this file's append-only rule: the
earlier entries record what was believed at the time, and the DOI in them was
correct throughout, so the analysis they describe is unaffected. The error
appears to have come from misreading "Persad, Amit" in the author list. The wrong
name was fixed in place in code, sbatch headers, TASKS.md, PLANNING.md and
docs/view_registry.md, where there is no append-only rule and a stale name would
simply mislead.

**Also settled while checking this: the paper does NOT specify its inner-fold
count, and does not use leave-one-out anywhere.** Its methods say only "as a part
of the inner-fold CV scheme, we optimized the regularization strength"; the
k = 5-or-10 rule is attached explicitly to the OUTER loop ("repeated k times as
a part of the outer k-fold CV scheme"). So `arms.INNER_CV = 3` (regression) /
`INNER_CV_LOGISTIC = 2` are OUR choices filling a gap in their methods, not a
match to them, and should be described that way. The reasoning stands on its own
— an inner split taken from a ~38-epoch training fold leaves ~7 observations per
fold at k=5, so the selected alpha would be noise on top of noise — but it is
inference, not replication.

**And a find that strengthens the blocked-CV addition.** The paper DOES use a
time-blocked scheme, just not for the neural decoder: for the facial/momentary-
pain analysis it used "a conservative sequential 5-fold cross-validation scheme
to account for the temporal dependency of behavioral timepoints occurring close
together, which could artificially increase model performance." That is exactly
the concern our `blocked` scheme addresses, stated by the authors themselves and
applied elsewhere in the same paper but NOT to the self-report neural decoder.
So running it is applying their own standard consistently rather than
second-guessing their design — worth saying that way in any write-up, given the
blocked scheme roughly halves every arm.

**Where it lives:** `src/ieeg_ehr/decoding/`, `src/ieeg_ehr/preprocessing/laplacian.py`,
`sbatch/build_laplacian_bandpass_array.sbatch`, `docs/view_registry.md`.

---

## 2026-09-15 — Store the NATIVE FFT resolution for pain epochs; make all frequency binning a view

New base unit `features/pain/psd_epochs_fullres/epoch-5min-pre/`: the same
epochs, the same 2 s / 1 s window grid and the same bipolar pairs as
`psd_epochs`, but the frequency axis is the native FFT grid — **0.5 Hz, 1–250 Hz,
499 bins** — instead of 50 log-spaced bins. Extracted from the RAW signal, on
pain epochs only. The 50-bin continuous family is unchanged and stays on disk.

**Why — the log-bin reduction is irreversible and wrong at BOTH ends.**

1. **Too coarse where it matters most.** A log bin near 60 Hz spans 53.3–66.4 Hz
   (`log_bin_edges(50, 1, 250)`, bins 36+37). Excluding line noise therefore
   costs **13.2 Hz of real spectrum to remove ~4 Hz of contamination**, and bin
   49 alone spans 26 Hz (223.9–250.0) to catch a ±2 Hz guard at 240. Six of 50
   bins carry `contains_line_noise`, so ~12 % of the axis is discarded to remove
   ~1.6 %. On the native grid the same notch is 8 bins of 499 and costs 4.0 Hz —
   pinned by `test_notch_costs_four_hz_per_harmonic_not_thirteen`.
2. **Degenerate at low frequency.** Below ~4.7 Hz a log bin is *narrower* than
   the 2 s window's own 0.5 Hz resolution, so `_band_average_linear`'s
   nearest-frequency fallback fills it with a copy of a neighbour. Bins
   {1,2,4,5,7,10} are exact duplicates: **the 44 non-line-noise bins carry only
   38 distinct values** (`views.cache_reader.unresolvable_bins`). Delta and theta
   were built from fewer independent measurements than their bin counts implied.
3. **It blocked specparam.** ~2 samples per oscillatory peak is unfittable.
   specparam refuses a `peak_width_limits` lower bound below 2× frequency
   resolution, and the log axis's effective resolution is ~1.2 Hz at alpha and
   ~2.5 Hz at beta — a floor of ~5 Hz, wider than an alpha peak. This is why
   `psd_params.py`'s SLOPE_* block had to concede its exponent was a broadband
   tilt rather than an aperiodic one (BG.3).

**Why it is cheap enough to just do.** The full-resolution PSD is *already
computed* inside `bipolar_reref._welch_one_channel` — `nfft` is never passed, so
`nfft = nperseg` — and then thrown away by the binning step. It exists nowhere on
disk and cannot be recovered from the stored bins, so this needs a raw re-read;
but restricting to pain epochs makes that 3,710 × 301 s instead of 7,902 whole
runs, and V1 raw NWB is chunked `(10000, n_channels)` so a short full-width read
is the FAST direction. Measured: the 26 × 24 h continuous extraction becomes
minutes per subject. It also DROPS `_band_average_linear`, the old pipeline's hot
loop.

**Why the native grid rather than a finer-but-still-reduced one.** 1 Hz linear
bins would halve the ~300 GB, and a piecewise scheme would too. Both were
rejected: storage is ~1 % of Oak's free space, and the entire failure being
corrected here is that *an irreversible reduction got baked into the expensive
layer*. Storing the native grid makes every scheme — `log_bins_50`,
`canonical_bands`, `paper_bands_6`, 1 Hz linear, any notch half-width — a free
recompute, permanently. Nothing is baked in that a view could have decided.

**Why 2 s windows were kept.** df = sfreq/nperseg = 1/`PSD_WINDOW_SEC`, so 2 s
gives 0.5 Hz at 500, 1000 **and** 2000 Hz sampling — the cohort lands on an
identical frequency axis with no resampling, which is the only reason a fixed
axis is possible at all. Keeping it also preserved the existing epoch
definitions, QC masks and feature-level QC flags unchanged, and made the
correctness gate below possible.

**The gate, and why it is worth the design constraint it imposed.** The extractor
deliberately calls `signal.spectrogram` per channel on the float32 bipolar trace,
matching `_welch_one_channel` exactly, rather than batching over an axis or
upcasting to float64. Both alternatives would be slightly more accurate and both
would make the gate approximate. Instead, `fullres_psd_audit.py` check A re-bins
the new output through the real `log_bin_edges` + `_band_average_linear` and
**reproduces the on-disk 50-bin cache BIT-EXACTLY** — measured 0 ulp on sub-019,
3/3 epochs, 2026-09-15. That one comparison proves the read length, the epoch
offset, the pair ordering, `detrend`, `scaling` and the log transform
simultaneously. An approximate gate could not distinguish a 1-ulp difference from
a one-window time misalignment.

**Two measured numbers worth not misreading.** Check B (re-binning from the
*stored* float32 log values) came in at 2.25e-06 in log10 / 5.2e-06 fractional in
linear power per window. That is **not** in tension with P0.6's 2.5e-07: P0.6
measured epoch AVERAGES, and independent per-window rounding averages down —
5.2e-06/√300 = 3.0e-07, reproducing P0.6 almost exactly. The per-window floor is
set by float32's ~7 significant digits against a stored log10 magnitude of ~20
(iEEG log power in V²/Hz is very negative), so 2.25e-06 is *at* the
representation floor, not above it.

**One accepted loss, recorded so it is not rediscovered as a bug.** View AXIS 2's
`whole_session` baseline is UNAVAILABLE in this unit — an epoch-only cache has no
non-epoch windows. The default `zero_pain_epochs` baseline is unaffected, since
0-pain epochs *are* epochs. Use `psd_epochs` for `whole_session`.

**Where it lives:** `src/ieeg_ehr/features/build_pain_epoch_fullres_psd.py`,
`src/ieeg_ehr/features/fullres_psd_audit.py`,
`src/ieeg_ehr/views/fullres_reader.py`,
`sbatch/build_pain_epoch_fullres_array.sbatch`,
`config/psd_params.py` (`PSD_FULLRES_*`, `PSD_NOTCH_HALF_WIDTH_HZ`),
`config/paths.py` (`fullres_epoch_*`), `tests/test_fullres_psd.py`.

**What would reverse it:** a measured storage problem (the estimate is ~300 GB
for 83 subject-sessions; Oak was at 64 % on 2026-09-15, up 14 TB in 11 days, so
this is worth re-checking before the ~250-subject cohort lands), or a decision to
change `PSD_WINDOW_SEC`, which would change the resolution and force a new unit.

---

## 2026-09-15 — `window` and `detrend` are now passed explicitly in the PSD path (numerically inert)

`bipolar_reref._welch_one_channel` hardcoded `window='hann'` while
`config.PSD_WINDOW_FN` was what got RECORDED into provenance, and relied on
scipy's default `detrend='constant'`, which was recorded nowhere. Both are now
explicit parameters (`DEFAULT_PSD_WINDOW_FN`, `DEFAULT_PSD_DETREND`), threaded
through `compute_welch_log_bins`, with `detrend` added to `welch_params`.

**Why:** neither changes a single stored value — the constant *is* `'hann'` and
the default *is* `'constant'` — but a default is not provenance. Changing
`PSD_WINDOW_FN` would have silently produced wrong provenance for unchanged
output, and **every spectrum this project has ever stored has had its
per-2 s-segment mean removed with nothing on disk saying so**: a reader
reconstructing the method from a sidecar could not recover it, and a future scipy
default change would have silently altered the pipeline.

Kept as parameters rather than a config import because `bipolar_reref` is
deliberately config-free — `_welch_one_channel` is pickled to
`ProcessPoolExecutor` workers, and the caller is where config belongs. Nothing
existing was re-run; only newly written sidecars gain the `detrend` key, and
`qc/psd_timing.classify_design` tests key membership only, so the addition is safe.

**Where it lives:** `src/ieeg_ehr/preprocessing/bipolar_reref.py`,
`src/ieeg_ehr/preprocessing/run_pipeline_bipolar.py`.

---

## 2026-09-15 — The full-res cache stays PARQUET, not NWB or HDF5 (measured, not argued)

Asked directly whether NWB would make sense. Measured HDF5 against Parquet on
real sub-256 data (6 epochs, 358,200 rows, 715 MB raw float32 payload):

| Format | Size | vs raw payload |
|---|---|---|
| Parquet, zstd + BYTE_STREAM_SPLIT | 490.3 MB | **0.69x** |
| HDF5, shuffle + gzip-4 | 495.4 MB | **0.69x** |

**A dead tie on size** — unsurprising in hindsight, since HDF5's `shuffle` filter
and Parquet's `BYTE_STREAM_SPLIT` are the same idea (transpose the bytes of each
float so the repetitive sign/exponent bytes group together and the compressor has
something to work with). So there is no storage argument for either, and the
decision falls to everything else — where Parquet wins:

- **Column pruning.** `read_epoch(columns=[...])` reads 9 frequencies in 2 ms
  instead of 499. `io_conventions.md` §7 names this as the reason the cache is
  Parquet at all. HDF5 can hyperslab but decompresses whole chunks.
- **The artifact contract already fits.** `io.write_sidecar` / `config_hash` /
  `parents` / `assert_fresh` are built for files with JSON sidecars. NWB wants
  provenance embedded in a description string, which is exactly how
  `bipolar_fft` ended up with stale per-run JSONs contradicting the in-NWB blob
  (`data_sop.md` §14.1).
- **Format follows the tree** (`io_conventions.md` §7): everything under
  `features/` and `preprocessed/` is Parquet. A third format for one unit costs a
  reader in every consumer.

**And NWB specifically is a worse fit than plain HDF5**, for the reason
`architecture.md` PART 1 already gave when rejecting it for the 50-bin sibling:
*its contiguous-run time model fights a stack of discontiguous epochs*. A
`DecompositionSeries` is `rate` + `starting_time` over one continuous recording;
this cache is 122 discontiguous 5-minute epochs drawn from different runs, keyed
by an `epoch_id` that joins to a pain score. Expressing that needs either a
fabricated time axis or one series object per epoch, and `epoch_id` has no
natural slot either way.

`architecture.md` PART 1 named HDF5/Zarr-dense as the sanctioned fallback "if the
storage check shows Parquet too big." The check ran; it is not.

**Where it lives:** `features/build_pain_epoch_fullres_psd.py`
(`CACHE_COMPRESSION` carries the full measurement table).

---

## 2026-09-15 — Parquet DICTIONARY encoding was inflating the cache 49% AND driving a 16 GB RSS

pyarrow's default `use_dictionary=True` applies to every column. For a float32
column of ~10,000 near-unique log-power values, the dictionary is about as large
as the data *plus* the indices, so it **inflates**. Measured on real sub-019
data, as a fraction of the raw float32 payload, with the cohort projection at the
exact 156,905,430-row total:

| Encoding | vs raw | Cohort |
|---|---|---|
| snappy + dictionary (**pyarrow default**) | 1.49x | **465 GB** |
| snappy, no dictionary | 1.00x | 313 GB |
| zstd, no dictionary | 0.82x | 256 GB |
| snappy + BYTE_STREAM_SPLIT | 0.80x | 249 GB |
| **zstd + BYTE_STREAM_SPLIT** | **0.72x** | **226 GB** |

Chosen: **zstd + BYTE_STREAM_SPLIT on the frequency columns, dictionary on
`channel` only** (~200 repeated short strings per epoch, where a dictionary is
genuinely right), pyarrow's default RLE on the two integer index columns.

zstd over snappy costs ~2x decode (0.138 s vs 0.068 s per epoch; 6.9 vs 3.4 min
for a full-cohort pass) and saves 23 GB now, ~70 GB at the ~250-subject cohort.
Taken because sub-second-per-epoch reads are not the bottleneck for anything
downstream, and Oak was at 64% on 2026-09-15 having grown 14 TB in 11 days.

**Verified LOSSLESS bitwise, not assumed.** BYTE_STREAM_SPLIT only reorders
bytes, but P0.6 validated float32 round-trip through *default* Parquet, so a new
encoding gets the same check — including `-inf` (an exactly-zero bin), `NaN`, and
the ~-36.8 extreme `cache_params.py` cites. Pinned by
`test_chosen_parquet_encoding_is_bitwise_lossless`.

**The same flag was also the memory bug.** sub-256 (199 pairs, 2000 Hz) sat at
**16.00 GB of a 16.00 GB** cgroup limit for a run whose live data is ~1.5 GB.
Instrumenting per-epoch RSS showed it flat at **1.76 GB** under the new encoding,
with pyarrow's pool at 0.00 GB — so the balloon was pyarrow holding ~499
dictionary hash-table builders simultaneously while assembling each row group,
not the spectrogram, not an accumulation, and not the writer's row-group buffer.
One flag, two problems.

Two further reductions applied alongside, both sound on their own: the frequency
slice is resolved from `np.fft.rfftfreq` BEFORE the spectrogram and applied per
channel (at 2000 Hz the full grid is 2001 bins against the 499 stored, so this
avoids allocating 4x for data immediately discarded), and `log10` runs in place.

**What would reverse it:** a pyarrow version that picks a sane encoding by
default, or a measured read-speed problem that makes snappy's 2x decode worth
23 GB.
