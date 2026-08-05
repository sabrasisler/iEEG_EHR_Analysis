# SCRATCHPAD.md — iEEG_EHR_Analysis

**What lives here:** things I am **THINKING** — open questions, hunches, things
I'm chewing on. No done-state. **Lifespan:** until it resolves or graduates.

**The operational test:** *can you check it off?* No, it's a question / hunch /
maybe → here. Yes, it's an action → `TASKS.md`.

**This file is allowed to be half-formed — that is its job.** It is the waiting
room. A scratch item has exactly four fates:

1. resolves on its own → mark `[x]`, delete after committing;
2. dissolves (stops mattering) → delete;
3. **graduates to a task** when it becomes DOable → `/addtask`, delete here;
4. **graduates to a decision** when it becomes a settled call → `DECISIONS.md`
   (with the reasons), delete here.

**Kept small deliberately** so it fits in context. Solved items are *deleted*
after the file is committed — git history is the archive, this file is the
working set.

Add with `/addscratch "<thought>"`. A trailing `(→ ...)` is its origin.

---

## Open

- [ ] **Why does sub-085 look off — and would feature-level artifact detection
      just make it go away?** Hunch is that whatever is wrong is an artifact the
      raw-voltage mask doesn't catch, in which case P2.1 (feature-level QC on
      pre-normalization cached power) resolves it for free and no sub-085-specific
      work is needed. Worth checking that hunch BEFORE spending time on a targeted
      investigation — if it's right, this dissolves; if sub-085 still looks off
      after feature-level QC, that's a much more interesting problem and says the
      cascade is missing a detector. (→ PLANNING P2.1)
- [ ] **Mask choice (P0.1): is `_logz4` actually the right default, or just the
      one with plots?** `_sw_logz4` is the stricter of the two full-cohort
      candidates and the only one with `summary/` and
      `plots/{flagged,random}_examples/` built — which means it may be the
      incumbent by *convenience* rather than by evidence. What would distinguish
      them: how much data does the extra `logz4` detector remove, and is what it
      removes actually artifactual? (→ config/paths.py TODO(P0.1))
- [ ] **Feature-QC thresholds K, X, Y, Z (P2.1) — on what structural grounds?**
      They must be set before looking at pain relationships, which rules out
      tuning them for signal. Open question is what the *structural* criterion is:
      fraction-of-data-retained targets? A break in the metric distribution?
      Agreement with the raw-voltage mask? (→ PLANNING P2.1)
- [ ] **Does Parquet actually win for the cache, or will the dense-array shape
      push toward HDF5/Zarr?** P1.2 answers this empirically (size + read speed on
      one subject). Hunch: Parquet's per-column reads matter more than density
      here because views slice by frequency bin, but that is a guess until
      measured. (→ PLANNING P1.2)
- [ ] **sub-236: is the exclusion-rollup gap fixable, or is it permanently out?**
      It's currently hard-excluded from the exploratory set
      (`_EXCLUDE_FROM_EXPLORATORY`) because its raw_voltage rollups are incomplete
      at the newer sweep labels. If it's just a rerun, it should be a task
      instead. (→ config/paths.py, docs/qc_context.md "sub-236 gap")
- [ ] **PHI-side matching (P4): same repo or a separate PHI repo?** Age can't come
      to Sherlock, so the matching code has to live somewhere the PHI master is.
      Splitting repos is cleaner for the boundary but duplicates the cohort
      schema. Not urgent until the ~150 land. (→ PLANNING P4)

- [ ] **`build_bipolar_exclusions.py` output path doesn't encode which raw-voltage mask
      produced it — re-running the same `--label` (e.g. `std10`) against a different
      `--raw-voltage-mask` silently overwrites prior output (and the shared `params.json`)
      for any overlapping subject.** Just happened for real (2026-07-27): an 82-subject
      `std10` run against `gross-std3_satmargin15_sw_logz4` clobbered the earlier 17-subject
      `std10` run's output against `gross-std3_satmargin10_logz3` for all 17 overlapping
      subjects. Open question is the right fix — namespace the output path by mask label
      too, or make `params.json` per-subject instead of shared — vs. just a documented
      convention (always pick a `--label` that encodes the mask, e.g. `std10_satmargin15sw`).
      (→ qc_scripts/build_bipolar_exclusions.py, qc_scripts/CONTEXT.md "GOTCHA" 2026-07-27)

- [ ] **Should `warn_if_dirty()` also refuse when provenance is unavailable?**
      Inside a worktree on a compute node, `git_provenance()` returns
      `available=False` and the recorded hash becomes `no-git` — the login node's
      system git (1.8.3.1) predates worktree gitfiles. Today that only prints a
      warning, so a definitive run launched from a worktree would produce
      artifacts with NO commit recorded, which is exactly the failure the
      commit-before-you-run rule exists to prevent. Cheap fix would be `ml system
      git` in the sbatch preamble; the open question is whether a missing hash
      should be a hard stop for a *definitive* run rather than a warning.
      (→ docs/labnotebook/2026-07-27.md, io/provenance.py)

- [ ] **Some stored PSD runs are entirely non-finite — is that expected, and
      should the mask or the PSD writer catch it?** Found incidentally by the P0.6
      dtype audit: two of sub-071's runs (the two smallest, ~2 MB and ~4 MB raw)
      have a stored `psd_log_bins` that is 100% `-inf`/NaN, because the raw
      voltage is a single constant value repeated — so PSD == 0 and log10(0) ==
      -inf for every window/channel/bin. sub-071's first *usable* run is still
      18.4% non-finite. Open questions: are these dead runs a known
      recording-side artifact (amp disconnected?), and should something refuse to
      write an all-`-inf` PSD rather than storing it? Note the raw-voltage QC mask
      operates on voltage, so a *constant* trace may pass flatline detection
      depending on the variance threshold. Cheap to check — the audit already
      lists the offending runs. (→ P0.6 audit summary.txt, PLANNING P2.1)
- [ ] **A stored log-power of -36.8 is physically odd — near-dead channel, or a
      units/scaling problem?** sub-088's minimum stored log10-power is -36.8
      (i.e. ~1.6e-37 V²/Hz), against a cohort-typical floor of about -19. It is
      what forced the "exponentiate in float64" rule in P0.6, so it is handled
      numerically — but the *physical* question is untouched. If it is a dead
      channel, feature-level QC (P2.1) should catch it on structural grounds and
      this dissolves. If a whole subject sits orders of magnitude below the rest,
      that is a scaling/`conversion` bug worth finding before the sweep.
      (→ P0.6 audit summary.txt, PLANNING P2.1)
- [ ] **`io/nwb.py`'s float32 cast of the raw voltage is NOT bit-exact — does it
      matter anywhere?** The raw `ElectricalSeries` is stored float64, and the
      P0.6 audit measured the cast: sub-085 and sub-088 are *not* exactly
      representable in float32 (sub-088 has exactly 65536 distinct values = 16-bit
      ADC, scaled by a float64 `conversion`, and the product is not a float32).
      The induced relative error is ~6e-8 on voltage — far below ADC quantisation,
      so almost certainly irrelevant, and it is upstream of P0.6's scope
      (cache dtype) rather than part of it. Recording it because it is the one
      place in the chain where a dtype choice is measurably lossy and nobody has
      written down that it's acceptable. (→ P0.6 audit dtype_audit.json,
      `io/nwb.py`)

- [ ] **`config.PLOTS_ROOT` is `analysis/scratch/`, but `architecture.md` PART 4
      draws scratch as `analysis/pain/scratch/`.** Noticed during P0.3 while adding
      the `analysis_run_dir` builders, deliberately not "fixed": runs already exist
      at the current path and it is throwaway output either way. Open question is
      whether scratch should be per-event at all — if the answer is no, the doc is
      what's wrong, not the code. (→ docs/labnotebook/2026-07-27.md, config/paths.py)

- [ ] **Should a low-n region cell be plotted at all?** The first group z-score
      heatmap was dominated by one S2 (supramarginal) cell at +5.2 that came from a
      single subject, 2 channels, 4 epochs — and specifically from line-noise
      frequency bins. Dropping line-noise bins fixes this instance, but the general
      question stands: a region row backed by one subject is not comparable to one
      backed by 77, yet they share a colour scale. Options are a minimum-n mask on
      display, a robust (percentile) colour scale, or leaving it and reading the n
      annotations. Do NOT settle this by tuning until P2.1 feature-level QC has run —
      it may simply remove the offending channels.
      (→ docs/labnotebook/2026-07-28.md 12:45)
- [ ] **Is `drop_line_noise_bins` the right DEFAULT for the view?** Currently False,
      matching the cache's deliberate choice to store them. But the harmonic bins
      produced every extreme value in the first figure, which argues the useful
      default is True and the cache's choice only means "recoverable", not
      "recommended". Interacts with the canonical band edges, which are already drawn
      to fall BETWEEN harmonics — so for `freq: canonical_bands` this may be moot and
      only matter for `log_bins_50`. (→ config/psd_params.py, view_registry AXIS 5)
- [ ] **Why does sub-159 (and 093, 154, 240) have no DK column while 79 others do?**
      Is this a FreeSurfer/localization pipeline gap that could be filled upstream, or
      are these subjects genuinely un-localizable (no imaging)? If fillable, that is 4
      subjects recovered for every region-level analysis. Note MNI_coord_* may still
      be present even where the DK column is not — worth checking before writing them
      off. (→ docs/labnotebook/2026-07-28.md 12:45)

- [ ] **sub-071's MNI coordinates are physically impossible — so can its DK
      labels be trusted, given the analysis uses them?** Found 2026-08-05 by the
      electrode glass brain, which plotted 55 of the 56 analysis subjects.
      sub-071 has 35 channels, all with coordinates AND DK labels, but the ranges
      are x -382..371, y -784..836, z -216..89 against an MNI152 extent of roughly
      +-90 / -130..90 / -80..110 — off by a factor of ~4-9, i.e. a units or
      registration failure, not noise. It is the ONLY affected subject: the entire
      "15 contacts out of bounds" drop was sub-071's ROI-mapped remainder.
      The question that matters is not the figure (which correctly excluded it) but
      the ANALYSIS: sub-071 currently contributes region rows to all the roi_v2
      heatmaps and spectra, and its DK labels come from the same localization that
      produced these coordinates. If the registration failed, the parcel
      assignments may be wrong too — in which case its channels are attributed to
      the wrong ROIs and it should be excluded from region-level work, taking the
      region-level n from 56 to 55.
      Note sub-071 ALREADY has an independent problem: two of its runs store a
      100% -inf/NaN psd_log_bins (see the non-finite-PSD item below). Two unrelated
      defects in one subject is itself a reason to look at it as a whole.
      (→ docs/labnotebook/2026-08-05.md, analysis/plot_electrode_locations.py)
- [ ] **Is equal-weighting EPOCHS right, given that it makes the 0-pain bin
      unable to be zero?** The z-score baseline pools 0-pain WINDOWS; the reported
      value averages EPOCHS. When epochs have unequal surviving-window counts —
      which is exactly what QC masking produces — those two differ, so the `none`
      bin cannot come back at 0. Measured 2026-07-29: max |group mean| 0.0201 z,
      median 0.0037, and it tracks masking
      (`corr(none dev, n_channel_epochs_dropped_coverage) = +0.651`).
      The tension: equal-weighting epochs is the same principle CLAUDE.md applies to
      subjects (the unit of replication is equal-weighted so a 200-contact subject
      cannot outvote a 30-contact one), and window-weighting would let a subject's
      longest cleanest epoch dominate their own mean. So the current behaviour may be
      correct and the offset simply the price of it. Open question is whether the
      epoch or the window is the unit here. NOT the same question as the epoch-drop
      threshold, which is a task. (→ docs/labnotebook/2026-07-29.md)
- [ ] **What should the epoch-retention threshold X be, on structural grounds?**
      Same constraint as the other P2.1 thresholds: set before looking at pain
      relationships. Candidate criteria — a break in the distribution of surviving-
      window fraction, a target for how much the epoch-weight inequality shrinks
      (measurable directly against the `none` offset above), or simple
      fraction-of-data-retained. Note it trades against n: the most-masked epochs
      are presumably concentrated in a few subjects. (→ TASKS.md, PLANNING P2.1)
- [ ] **Why is the `none` offset POSITIVE (+0.0038 overall)?** The unequal-weighting
      mechanism explains that an offset exists and why it scales with masking, but
      not its SIGN. Hunch is that heavily-masked epochs retain somewhat higher
      residual power, so up-weighting them pushes positive — untested, and it should
      not be asserted anywhere until it is. Cheap to check: regress a subject's
      per-epoch mean z on that epoch's surviving-window count.
      (→ docs/labnotebook/2026-07-29.md)
- [ ] **Does a cluster-forming threshold on t alone let scientifically empty
      effects through?** Demonstrated yes on 2026-07-29: the circular `none` bin
      produced 6 significant two-stage clusters at a mean of ~0.004 z, because a
      tiny mean over a tinier SE is overwhelmingly significant. Currently handled by
      REPORTING effect size (`mean_abs_z`, `floor_ratio`) rather than by gating on
      it, since any multiplier of the floor would be arbitrary. Open question is
      whether a gate should eventually be adopted and what would justify its value.
      (→ analysis/cluster_permutation.py, docs/cluster_permutation.md)

## Next steps (session-end dump — `/standup` reads this tomorrow)

- [ ] Lab-notebook system is built but unexercised. The real test is whether
      `/lognote` gets used on the *next* analysis run without being prompted; if
      it doesn't, the friction is in the wrong place and the command should shrink,
      not the habit. (→ docs/labnotebook/2026-07-27.md)
