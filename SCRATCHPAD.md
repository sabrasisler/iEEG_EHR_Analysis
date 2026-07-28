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

## Next steps (session-end dump — `/standup` reads this tomorrow)

- [ ] Lab-notebook system is built but unexercised. The real test is whether
      `/lognote` gets used on the *next* analysis run without being prompted; if
      it doesn't, the friction is in the wrong place and the command should shrink,
      not the habit. (→ docs/labnotebook/2026-07-27.md)
