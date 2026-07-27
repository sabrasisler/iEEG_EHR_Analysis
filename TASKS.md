# TASKS.md — iEEG_EHR_Analysis

**What lives here:** things I will **DO** — every line has a done-state and is
checkable. **Lifespan:** until done.

**The operational test:** *can you check it off?* Yes → task. No, it's a
question / hunch / maybe → `SCRATCHPAD.md`.

**Must stay small — near-term only** (this week / next). The far future lives in
`PLANNING.md`'s phases, not here. If this file starts to feel overwhelming, that
is the signal that far-future work has leaked in: push it back down to a planning
phase. Tasks may originate here freely without a planning entry — most will.

Add with `/addtask "<action>"`. Check off in place; delete a `[x]` once it has
been committed (git history preserves it), or promote it to `DECISIONS.md` if
finishing it settled something.

Optional trailing `(→ ...)` on a line is its origin — the analysis, notebook
date, or figure that prompted it.

---

## This week

- [ ] **P0.1 Pin the QC mask.** Decide between `gross-std3_satmargin15_sw` and
      `gross-std3_satmargin15_sw_logz4`; record the choice in
      `config/paths.py:CANONICAL_MASK_LABEL` (currently `_sw_logz4` with a
      `TODO(P0.1)`) and log the *why* in `DECISIONS.md`. Review the
      flagged/random example plots once more first.
- [ ] **P0.3 Install pyarrow** into the shared venv
      (`$GROUP_HOME/venvs/ieeg_ehr_analysis`) and switch `io/tables.py:save_table`
      from CSV to Parquet. Do NOT bulk-convert existing CSVs.
- [ ] **P0.3 Add the provenance/staleness sidecar writer** so every artifact
      write goes through it (`io/provenance.py` already has the git+timestamp
      half; the sidecar-emitting writer is what's missing).
- [x] **P0.4 Repo cleanup** — done 2026-07-27, own commit, no logic changes mixed
      in. Delete this line at the next commit.
- [ ] **P0.2 Lock the discovery cohort** into `cohorts/discovery-core-<date>.json`
      (anonymized IDs only) and thread a `--split {discovery,heldout,all}` flag
      through subject resolution. Hold-out UNREACHABLE by default.
- [ ] **BG.1 Submit the PSD array** for the remaining subjects (`normal`
      partition). Highest priority — it is the primary data.
- [ ] **BG.2 Submit the 1/f-slope (polyfit) array** across all subjects — cheapest
      new feature, reads the stored PSD bins.
- [ ] **P0.6 dtype audit** — check the current NWB dtype, validate the float32
      round-trip (epoch averages agree to ~6+ sig figs), standardize the cache on
      float32.

## Next

- [ ] **P1.1 Refactor `build_pain_epoch_power.py`** into the epoch-definitions +
      per-window Parquet cache builder. Blocked on P0.1, P0.2, P0.3, P0.6.
- [ ] **P1.2 Storage sanity check** on one subject before building at scale.

## Notebook-system follow-ups

- [ ] Call `log_analysis()` from each analysis script that produces output as you
      next touch it — one line, beside the existing sidecar write. Do NOT do a
      sweep-and-add pass; add it when the file is open anyway.
      (→ docs/labnotebook/2026-07-27.md)
