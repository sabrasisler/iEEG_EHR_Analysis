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
