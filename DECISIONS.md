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
