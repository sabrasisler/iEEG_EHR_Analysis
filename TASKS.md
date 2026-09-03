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
- [x] **BG.2 Submit the 1/f-slope (polyfit) array** across all subjects — cheapest
      new feature, reads the stored PSD bins. Done 2026-08-05 for the DISCOVERY
      split: `views/aperiodic.py` + `views/build_pain_epoch_slope.py` +
      `sbatch/slope_discovery_array.sbatch`, figure by
      `analysis/plot_slope_violin.py`. Fit is 1–250 Hz minus the six line-noise
      bins, per channel, then averaged into ROIs. Delete this line at the next
      commit. (→ docs/labnotebook/2026-08-05.md)
- [ ] **Adopt a minimum-0-pain-epoch criterion at the COHORT/VIEW layer, not per
      figure.** Ten of 56 discovery subjects have <5 zero-pain epochs and a 0-pain
      reference whose SEM (0.083) exceeds the effect being measured (0.052):
      039, 067, 071, 109, 124, 183, 206, 209, 210, 230. The slope figures now
      filter them at plot time via
      `analysis/view_tables.exclude_thin_baseline_subjects`, but every
      baseline-normalized artifact on Oak — the `zscore` and `delta` power views,
      the heatmaps, the spectra, the cluster test — was built with them in and
      inherits the same noisy denominator. The real fix belongs in
      `views/build_pain_epoch_view.py` (which already raises when a subject has NO
      0-pain epochs — this is the same check with a sane threshold) and/or
      `config/cohorts.py` as a runnable-subject criterion. Needs a view rebuild, so
      pair it with the split-half baseline task below rather than rebuilding twice.
      (→ docs/labnotebook/2026-08-06.md 2026-08-07, SCRATCHPAD)
- [ ] **Regress the 1/f slope out of the canonical-band values.** The regional
      dissociation already argues the band and slope results are two effects — the
      slope effect is deep/limbic and absent in M1 (0.2x the noise floor) and S1
      (0.5x), while the 2026-08-05 band block was strongest in M1/S1/S2-PO — but
      that is an anatomical argument, not a statistical one. Per (subject, region,
      epoch), residualize band power on the fitted slope and re-run the band
      contrast; if the sensorimotor beta effect survives, it is genuinely
      narrowband. Cheap: both quantities are already in sibling view tables.
      (→ docs/labnotebook/2026-08-06.md 2026-08-07)
- [ ] **Re-run the slope array for the remaining splits** once BG.1's PSD backlog
      lands — the discovery build is 60 subjects, and the fit range is part of the
      config hash so a second range is a separate directory, not a rebuild.
- [ ] **Unify the within-subject standardization.** `plot_band_violin_view.py`
      still carries its own copy of `within_subject_z` / `subject_level`; the
      shared versions now live in `analysis/view_tables.py` and
      `plot_slope_violin.py` uses those. The copy was left alone on 2026-08-05
      only because that file had uncommitted in-flight changes. They agree
      line-for-line today, which is exactly the state that quietly stops being
      true — point the band violin at `view_tables` and delete the duplicate.
- [ ] **Decide whether the broadband tilt needs a knee-free companion.** The
      1–250 Hz fit folds the low-frequency knee and the alpha/beta peaks into one
      number by design (config/psd_params.py says so). If the slope violins show
      anything, re-run at a knee-free range and/or escalate to BG.3 (FOOOF)
      before the result is described as an "aperiodic exponent" anywhere.
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

- [ ] **Decide the status of sub-222 and sub-231.** Both are `unassigned`, and both
      appear in the 2026-07-28 P1.3 heatmaps (group + per-subject) because the split
      gate did not exist when that timing sample was drawn. No analytic choice came
      from those figures, but the data has been seen. Either fold them into discovery
      (honest, costs 2 hold-out candidates) or record them as viewed-but-excluded and
      keep them out of the hold-out anyway. Do NOT leave them silently unassigned.
      (→ DECISIONS.md 2026-07-28 cohort lock)
- [ ] **Build epoch caches for the 5 unprocessed discovery subjects**
      (`122 138 212 235 259`) so the discovery n goes 60 → 65. All have PSD, std10 and
      a bipolar mask; none has a cache. sub-259 needs its PSD re-extraction (job
      36216669) to land first. Note sub-138 has only 1 PSD run on disk.
      (→ DECISIONS.md 2026-07-28 cohort lock)

- [ ] **Make the cache builder emit `channel_meta`, so no view ever reads an NWB.**
      `build_pain_epoch_power.py` already loads the electrodes table per run
      (`_load_run_arrays` reads `Desikan_Killiany_anode` and throws it away), so this
      is writing a table it already has in hand, not new I/O. Today
      `views/channel_meta.py` rebuilds it lazily on first view — one NWB metadata pass
      per subject, cached thereafter — purely because the 34 GB cache predates that
      module. Emitting it at cache-build time makes the view layer pure
      table-reads-only and removes pynwb from its import path.
      Also backfill the existing 83 subject-sessions (cheap, metadata-only, same shape
      as `backfill_epoch_defs_timing`) so the lazy path can then be deleted rather
      than left as a permanent fallback.
      (→ views/channel_meta.py, features/build_pain_epoch_power.py:_load_run_arrays)

- [ ] **Decide whether sub-071 can stay in region-level analyses.** Its MNI
      coordinates are impossible (x -382..371 vs an MNI extent of ~+-90; found
      2026-08-05 by the glass brain, which excluded it), and its DK labels come from
      the same localization. Check whether the registration failed outright — if so
      its parcel assignments are wrong and it must leave region-level work, taking
      the region-level n from 56 to 55 and requiring a re-run of the roi_v2 figures.
      Cheap first check: does its `LEPTO_coord_*` / `fsaverageINF_coord_*` look
      sane, or is every coordinate space broken for this subject? It also has two
      all-`-inf` PSD runs, so consider it as a whole rather than defect by defect.
      (→ SCRATCHPAD "sub-071", docs/labnotebook/2026-08-05.md)
- [ ] **Add a coordinate sanity check upstream, where it belongs.** The glass
      brain caught this only because it plots coordinates; nothing in the QC tree
      looks at them. A per-subject check that MNI coordinates fall inside the
      MNI152 extent is trivial and would have flagged sub-071 before it entered any
      analysis. Natural home is the feature-level QC tree
      (`qc/feature_level/`) or a small standalone audit beside
      `qc/audit_psd_timing.py`, which is the same shape of problem: a
      choice-independent fact about a subject that every downstream view inherits.
      (→ docs/labnotebook/2026-08-05.md)
- [ ] **Build a NON-CIRCULAR negative control for the cluster test: a held-out
      baseline.** Split each subject's 0-pain epochs in half, form the baseline from
      one half, and test the other half against it. The current `none` control is
      circular — the same windows define the baseline and are tested against it — so
      it can only ever reveal bookkeeping asymmetries, never real leakage. A
      split-half control would actually test for leakage, and its floor would be the
      honest one to report. View-layer change (`views/build_pain_epoch_view.py` pass
      1), needs a full rebuild before any figure using it is valid.
      (→ docs/labnotebook/2026-07-29.md, plan `sparkling-coalescing-teacup`)
- [ ] **Feature-QC: drop an epoch that retains too little of itself.** Add an epoch
      exclusion on the fraction of an epoch's windows (and/or bins) surviving the
      mask, tightening or replacing `EPOCH_MAX_EXCLUDED_FRAC = 0.5`. This is not
      only hygiene: epochs with very unequal surviving-window counts are the
      mechanical cause of the `none`-bin offset measured on 2026-07-29
      (`corr(none deviation, n_channel_epochs_dropped_coverage) = +0.651`), because
      the baseline pools WINDOWS while the reported value equal-weights EPOCHS.
      Dropping the most-depleted epochs shrinks that inequality directly. Threshold
      X is an open question — see SCRATCHPAD. Belongs in the P2.1 cascade
      (`architecture.md` PART 7), whose epoch-flag step this is.
      (→ docs/labnotebook/2026-07-29.md, PLANNING P2.1)

## Next

- [ ] **Build a raw-NWB span manifest, so "recorded iEEG hours" means recorded.**
      `sherlock_file_registry.csv` only timestamps runs that have a PREPROCESSED
      file (every one of its 2,136 null-`start_datetime` rows has
      `has_preprocessed == False`), so registry timing measures preprocessed
      coverage. Of the 98 med-admin sessions only 41 are fully timestamped, 41 are
      partial and 16 have none — and an untimestamped run cannot be placed on a
      hospital day at all. `med_analysis/recording_hours.py` currently falls back
      to MAR session span for those, which overstates monitoring by ~3.5%
      (DECISIONS.md 2026-09-03, call 5). The fix: a per-subject Slurm array that
      reads `session_start_time` + `acquisition/*/starting_time` + `rate` + data
      shape straight from the raw NWB (all four confirmed present, metadata-only —
      no data load), writing one run-span row per run. Then
      `recording_hours.session_coverage` uses one method for every session and
      Fig 3's rates become exact. Also unblocks a real coverage denominator for
      anything else that needs one.
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
