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

## Next steps (session-end dump — `/standup` reads this tomorrow)

- [ ] Lab-notebook system is built but unexercised. The real test is whether
      `/lognote` gets used on the *next* analysis run without being prompted; if
      it doesn't, the friction is in the wrong place and the command should shrink,
      not the habit. (→ docs/labnotebook/2026-07-27.md)
