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
- [x] **P0.3 Install pyarrow** — done 2026-07-27. pyarrow 20.0.0 + joblib 1.5.3 in
      the shared venv; numpy/pandas untouched. Needed
      `--only-binary=:all:` (Sherlock is glibc 2.17, modern pyarrow wheels are
      manylinux_2_28 → pip fell through to a source build). `save_table` now
      dispatches on extension, so the QC tree stays CSV by design and new code
      uses `io.write_table` (Parquet). Delete this line at the next commit.
- [x] **P0.3 Add the provenance/staleness sidecar writer** — done 2026-07-27.
      `io/sidecar.py` + `io/models.py`; `ieeg_ehr.io` is now a flat namespace where
      every writer emits a sidecar and every reader checks staleness. Contract in
      `docs/io_conventions.md`; 18 tests in `tests/test_io_conventions.py`.
      Delete this line at the next commit. (→ DECISIONS 2026-07-27)
- [x] **P0.4 Repo cleanup** — done 2026-07-27, own commit, no logic changes mixed
      in. Delete this line at the next commit.
- [ ] **P0.2 Lock the discovery cohort** into `cohorts/discovery-core-<date>.json`
      (anonymized IDs only) and thread a `--split {discovery,heldout,all}` flag
      through subject resolution. Hold-out UNREACHABLE by default.
- [ ] **BG.1 Submit the PSD array** for the remaining subjects (`normal`
      partition). Highest priority — it is the primary data.
- [ ] **BG.2 Submit the 1/f-slope (polyfit) array** across all subjects — cheapest
      new feature, reads the stored PSD bins.
- [x] **P0.6 dtype audit** — done 2026-07-27. Cache standardized on float32
      (`config/cache_params.py`); epoch averages agree with a float64 pipeline to
      8.1 sig figs, bit-exact round-trip through Parquet and HDF5. Also found:
      views MUST upcast to float64 to average (a float32 accumulator holds only
      6.0 sig figs) and to exponentiate to linear. Delete this line at the next
      commit. (→ DECISIONS 2026-07-27, docs/labnotebook/2026-07-27.md)
- [x] **Roll the 5 OOM-retry subjects into the pinned mask** — done 2026-07-27
      16:42–16:57 but never ticked; confirmed 2026-07-28 by inspection (all 89
      mask CSVs present at the pinned label, coverage exactly the 89 sessions /
      87 subjects predicted below). Delete this line at the next commit. Original
      note kept for the record: `250, 251, 255,
      256, 257` have raw_voltage metrics on disk but no mask. They are exactly
      `remaining86` minus `new81`: they were in the metrics cohort but dropped
      from the rollup cohort because their metrics were still OOM-retrying when
      it launched, and the rollup was never re-run once those finished. Only the
      cheap CSV-only steps are missing. NOT `--artifact-type all` — that uses
      config defaults (gross std5, saturation default, flatline var5e-13), which
      are not this mask's thresholds. The real recipe is a 4-stage chain
      (see `sbatch/`): gross std3 + saturation marginfrac0.15 + square_wave
      frac0.9 → intermediate mask `gross-std3_satmargin15_sw` → flatline
      `--std-thresh 4 --mask-from-label` that intermediate → final mask +
      summarize. sub-255 has 2 sessions, so this is 6 subject-sessions: coverage
      goes 83 → 89 sessions, 82 → 87 subjects. Do it before P0.1 so the pinning
      decision is made on the fuller cohort. (NOT sub-236 — different, non-OOM
      gap; see SCRATCHPAD.)

- [ ] **Finish the defs-timing backfill across the cohort.** ~71 of 83
      subject-sessions still lack `epoch_start_sec`/`hop_sec`; array `36197924` is
      queued but 12 were done inline. No full-cohort view run until this lands —
      `views/cache_reader.load_defs` refuses without them, by design.
      (→ docs/labnotebook/2026-07-28.md 12:45)
- [ ] **Roll the bipolar mask out to the full cohort** once
      `bipolar_excl` (`36181325`) finishes adding std10 for `250 251 255 256 257`.
      `python -m ieeg_ehr.qc.build_bipolar_mask` is re-runnable and skips subjects
      missing either input; currently 7 subject-sessions exist.
- [ ] **Decide what to do with the 4 subject-sessions that have no DK labels**
      (`093 154 159 240` — no `Desikan_Killiany_anode` column in the NWB electrodes
      table at all). They cannot enter a region-level view under any ROI scheme and
      currently produce an empty table plus a loud error. Options: exclude from
      region analyses via the cohort file, or run them `--region none` per channel.
      Either way the region-level n is 79, not 83, and that belongs in the coverage
      denominator. (→ docs/labnotebook/2026-07-28.md 12:45)
- [ ] **Give `build_bipolar_exclusions.py` a NaN-safe baseline count.**
      `.agg(n='size')` counts NaN `metric_value` rows while `sum` skips them, so any
      NaN metric biases the baseline mean low. Not hit today (no NaNs in the two
      subjects audited) but it is a real latent bug.
      (→ docs/labnotebook/2026-07-28.md 12:45)
- [ ] **Convert `qc/bipolar/exclusions/` to Parquet** once array `36181325`
      finishes writing CSV. The reader already accepts either extension; only the
      writer needs flipping, plus a sidecar. Deliberately deferred to avoid a
      half-converted tree mid-array. (→ docs/io_conventions.md §7)
- [ ] **Add `PYTHONPATH=${SLURM_SUBMIT_DIR}/src` to the other sbatch files.**
      Only `backfill_epoch_defs_timing.sbatch` has it. Without it a job submitted
      from a worktree silently runs the main checkout's code under a commit hash
      that claims otherwise. (→ docs/labnotebook/2026-07-28.md 12:45)

- [ ] **Re-run the PSD for the 24 60s-hop runs** (sub-247: 13 runs, sub-257: 11
      runs) under the current 2s/50% scheme, then delete the exclusion gate below.
      Raw voltage is intact — only the derived PSD is stale — so this is a recompute.
      Decided 2026-07-28: these are EXCLUDED from analysis until then, not
      down-weighted. (→ DECISIONS.md 2026-07-28, notebook 2026-07-28 12:55)
- [ ] **Gate the 60s-hop epochs out of the view layer** so the exclusion cannot be
      forgotten: refuse (or drop with a loud count) any epoch whose `hop_sec` is not
      the expected 1.0, rather than relying on whoever runs the sweep to remember.
      33 epochs across sub-247/sub-257 average 5 windows where every other epoch
      averages 300. (→ DECISIONS.md 2026-07-28)

## Next

- [ ] **P1.1 Refactor `build_pain_epoch_power.py`** into the epoch-definitions +
      per-window Parquet cache builder. Blocked on P0.1, P0.2 only now (P0.3 +
      P0.6 are settled). Writes via `io.write_table` +
      `io.write_manifest`; paths from `config.pain_epoch_*`; worked example in
      `docs/io_conventions.md` §9. Its legacy `_write_provenance()` goes away —
      `io.write_sidecar` replaces it. Cast log-power to
      `config.CACHE_FLOAT_DTYPE` (float32) on write, and record the dtype in the
      manifest.
- [ ] **P1.2 Storage sanity check** on one subject before building at scale.
- [ ] **Make a missing mask impossible to miss.** Three paths silently fall back
      to an unmasked baseline, and the failure is invisible in the output: an
      unmasked baseline keeps artifact windows → inflates the std → deflates z →
      the detector gets *less* sensitive for exactly the subjects whose QC inputs
      were incomplete. Sites: `feature_level/detect_power_outlier.py:195-200`
      (no mask file), `build_exclusions.py:153-159` (flatline
      `--mask-from-label` with no mask file), and
      `mask_projection.project_to_pairs` (an absent run, contact, or 60s bin all
      read as "not excluded"). Escalate from `logger.warning` to a banner block
      plus an explicit `--allow-unmasked` opt-in, so an array task cannot quietly
      emit a degraded baseline. **Also: stop a label from claiming a mask it did
      not use** — sub-236's flatline sat at
      `logz4_masked-gross-std3_satmargin15_sw` with an unmasked baseline for
      months, because the intermediate mask never contained it; if the fallback
      fires, the output should not carry a `masked-<x>` label. **And record
      per-run mask coverage in run_info**, so
      `qc/report_mask_coverage.py`'s tier 2 does not have to reconstruct it by
      joining against the mask CSVs.
      (→ docs/labnotebook/2026-07-28.md)
- [ ] **Revisit `_EXCLUDE_FROM_EXPLORATORY = {'236'}`** in `config/paths.py` now
      that sub-236 has a real pinned mask (built 2026-07-28: its only gap was
      `square_wave/frac0.9`; metrics were complete all along — 107 of 107
      *readable* runs, the 2 absent registry runs being unparseable NWBs). It was
      excluded because its rollups were incomplete, which is no longer true. Then
      it needs bipolar `std10` and feature-level metrics to join the cohort, and
      `cohorts/subjects_pain_epoch_cache.txt` (87) would go to 88.
      (→ SCRATCHPAD "sub-236", docs/labnotebook/2026-07-28.md)

## Notebook-system follow-ups

- [ ] Call `log_analysis()` from each analysis script that produces output as you
      next touch it — one line, beside the existing sidecar write. Do NOT do a
      sweep-and-add pass; add it when the file is open anyway.
      (→ docs/labnotebook/2026-07-27.md)
- [ ] **Give these writers a sidecar as you next touch them** — same
      no-sweep-and-add rule. Audited 2026-07-27; each writes a table with no
      provenance beside it: `qc/summarize_exclusions.py`,
      `qc/build_rail_summary.py`, `qc/build_plot_targets.py`,
      `qc/build_run_start_times.py`, `qc/pad_exclusions.py`,
      `qc/processing_status.py`, `qc/diagnostics/threshold_summary.py`,
      `preprocessing/bipolar_bands.py`, `io/build_file_registry.py`. One line:
      `io.write_sidecar(path, params={...}, parents=[...])`. They stay CSV.
      (The detector/exclusion/mask writers already emit `params.json`, and the
      plot scripts already write `provenance.json` — those are fine.)
