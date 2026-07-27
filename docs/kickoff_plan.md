# Pain iEEG — Analysis Kickoff Plan & Task Orchestration

Companion to `architecture.md` and `view_registry.md`.
Ordered task plan from "preprocessed PSD exists" to "running exploratory
analysis," plus repo org, IO conventions, and background jobs. Target repo:
`/home/groups/ckeller1/sisler/iEEG_EHR_Analysis` (Sherlock).

Sequencing rule: anything BAKED INTO the cache (QC mask, epoch definition) or
that prevents CONTAMINATION (cohort split) is locked BEFORE building cache at
scale — cheap now, expensive to retrofit.

Scope note: Phase 3 (confirmation GLMM, Aims 2/3) is FAR AWAY and deliberately
not detailed. The near-term center of gravity is Phase 2, which is much larger
than a single "signal existence" check — it's the joint exploration of feature
signal AND methodological choices (see Phase 2).

---

## PHASE 0 — Locks & foundations (do first)

- [ ] **P0.1 Pin the QC mask.** Decide canonical mask (`gross-std3_satmargin15_sw`
      vs `_logz4`); record in a config constant. Baked into the cache → switching
      later = full re-run. Review flagged/random example plots once more, commit.
- [ ] **P0.2 Lock current discovery subjects** into `cohorts/discovery-core-<date>.json`
      (IDs only; no age needed). These stay discovery permanently. Add a
      `--split {discovery,heldout,all}` / cohort-file flag threaded through subject
      resolution; hold-out UNREACHABLE by default. (Matched hold-out built later,
      offline — see P4.)
- [ ] **P0.3 IO conventions + deps.** `pip install pyarrow --break-system-packages`.
      Add an `io/` helper: every writer emits artifact + provenance/staleness
      sidecar; readers check staleness. Parquet for tables, joblib for models,
      JSON for manifests.
- [ ] **P0.4 Repo cleanup** — one dedicated commit, no logic changes mixed in
      (see Repo section). Includes: create `outdated/`, move `qc/` out of `analysis/`.
- [ ] **P0.5 Write CLAUDE.md** at repo root from the architecture doc's rules block.
- [ ] **P0.6 dtype audit** — check current NWB dtype; validate float32 round-trip
      (epoch averages agree to ~6+ sig figs); standardize the cache on float32.

---

## PHASE 1 — Build the cache

Depends on P0.1 (mask) + P0.2 (cohort) + P0.3 (IO) + P0.6 (dtype).

- [ ] **P1.1 Refactor `build_pain_epoch_power.py`.** New job: emit (a) epoch
      definitions (tiny Parquet index) and (b) the per-2s-window, per-channel,
      per-bin, QC-masked, LOG-power epoched slice = the CACHE, as **one Parquet per
      subject/session** (subject/session in filename; epochs stacked via epoch_id).
      Must NOT average over the epoch, NOT normalize (those are views). Store log
      (matches bipolar_fft); linear is an exponentiate-in-view option.
- [ ] **P1.2 Storage sanity check.** Build the cache for ONE subject, `du -sh`,
      extrapolate to ~67 then ~250, confirm Oak quota. Build the SAME subject as
      HDF5/Zarr dense array too; compare size + read speed. Pick Parquet unless the
      density of HDF5/Zarr is needed. (~500GB acceptable per Sabra.)
- [ ] **P1.3 Implement the view layer** (`views.py`) — all seven axes from
      view_registry.md, each a function; optional `save_path` + staleness sidecar;
      default recompute.
- [ ] **P1.4 Build cache across the discovery cohort** via Slurm array (one task
      per subject). Manifests carry mask label + git hash + dtype.
- [ ] **P1.5 Reproduce one prior figure** through the new cache+view path to
      confirm numerical fidelity before trusting the refactor.

---

## PHASE 2 — Exploratory analysis (discovery cohort ONLY)

Depends on Phase 1. This is the near-term bulk of the work. Signal existence and
methodological choice are NOT separate — they're the SAME sweep, because
robustness of a signal across reasonable methodological variations is itself
evidence. The sweep's axes therefore include BOTH the feature grid AND the
view-config grid (view_registry.md).

- [ ] **P2.1 Feature-level QC** (structure in architecture PART 7): choice-independent,
      on pre-normalization cached power, stored `qc/feature_level/`, cascade
      window→epoch→channel→epoch-across-channels, metric/threshold split. Set
      thresholds (K, X, Y, Z) on structural grounds BEFORE looking at pain effects.
- [ ] **P2.2 Signal + methodology sweep** (the core exploratory activity). Axes:
      - feature: {high_gamma (primary), canonical bands, 1/f slope}
      - region grouping: {theory_sensory, theory_affective, individual DK, global}
      - view axes: {log vs linear}, {absolute vs subject_relative vs tertile},
        {zscore vs baseline_subtract vs none}, {log_direct vs linear_then_log avg},
        {baseline = zero_pain vs whole_session}
      For each cell: per-subject effect size (SMD high vs low, or correlation for
      graded) + sign-consistency across subjects + contributing n. One `sweeps/`
      run; grid as ROWS in results.parquet; FDR within discovery only. Output =
      ranked NOMINATIONS, not findings. Start cheapest/most-aggregated (global HFA)
      and gate finer tiers on review.
- [ ] **P2.3 Robustness reading**: which nominated signals survive across
      methodological variations (grouping/averaging/normalization/binning). A signal
      robust to reasonable view choices is a strong nomination; one that appears only
      under a specific normalization is weak.
- [ ] **P2.4 Subject-level clustering** on per-subject effect vectors across theory
      regions → candidate phenotypes (Aim 2 groundwork).
- [ ] **P2.5 Maintain the exploration log** throughout (what was looked at).
- [ ] **P2.6 FREEZE**: dated freeze doc — ranked directional hypotheses, frozen
      feature set + frozen view config, model spec, correction plan + n_tests,
      predicted directions. Ends exploration.

---

## PHASE 3 — FAR FUTURE (not detailed)

Model development (Huang-logic GLMM replication) is its own design conversation,
AFTER P2.6 produces a frozen feature set. Confirmation on the matched hold-out,
Aim 2 (BDI/mood), Aim 3 (opioid modulation) follow. Not planned in detail here.

---

## PHASE 4 — Matched hold-out (after the ~150 land; offline)

- [ ] Build `cohorts/heldout-matched-<date>.json` OFFLINE (PHI-side) matching
      hold-out to discovery on {pain-range>4, sEEG/ECoG, age, sex}. Age is PHI, so
      matching runs where the PHI master lives; only the anonymized id→cohort + SAFE
      axes cross to Sherlock. (Open: same repo vs separate PHI repo.)

---

## BACKGROUND JOBS (run continuously on Sherlock while doing interactive work)

Independent, embarrassingly parallel (per-subject Slurm arrays), don't block PSD
work. Start the slow ones early. All use the PINNED mask (P0.1) for cross-family
consistency.

- [ ] **BG.1 Finish PSD for remaining subjects** (~150 incoming). Highest priority —
      primary data. `normal` partition array.
- [ ] **BG.2 1/f slope (polyfit)** across all subjects — cheapest new feature, from
      stored PSD bins; separates band effects from aperiodic shifts. Sibling table.
- [ ] **BG.3 FOOOF / specparam** — bigger lift (per-epoch fits, quality checks);
      runs in background, not needed until finer sweep tiers.
- [ ] **BG.4 PAC extraction** — OWN family off the time-domain signal (Hilbert), at
      epoch scale. Expensive; start early because it's slow.
- [ ] **BG.5 Connectivity / coherence / PLV** — own family off the signal; most
      expensive + coverage-sensitive; lowest priority, ROI pairs first.
- [ ] **BG.6 EHR/confound tables** joined to epoch definitions: time-of-day,
      medication state at score time, seizure proximity, recording day. Cheap; makes
      confound controls possible.

Orchestration: these are Slurm ARRAY jobs (one task/subject), NOT multi-agent
work. One Claude Code session writes each array sbatch; submit + monitor with
squeue/sacct. Long arrays on `normal`; keep `ckeller1 --qos=high_p` (4-job cap)
free for interactive work. Revisit multi-agent only after the single-session +
array workflow is smooth.

---

## REPO ORGANIZATION (one dedicated cleanup commit)

> **DONE 2026-07-27.** The tree below is the as-built state, which differs from
> the original target in two ways, both noted inline: `config/` is inside the
> package rather than at the repo root, and `notebooks/` was retired rather than
> kept as live scratch.

```
iEEG_EHR_Analysis/
  CLAUDE.md
  README.md
  pyproject.toml                 # makes src/ieeg_ehr an installable package
  docs/
  src/ieeg_ehr/                  # ONE package; `python -m ieeg_ehr.<pkg>.<mod>`
    config/                      # DEVIATION: inside the package, not repo-root config/ —
                                 #   a root-level config/ is not importable from an
                                 #   installed package. Same job: single source of
                                 #   paths, pinned mask, band defs, cohort paths.
    io/                          # provenance/sidecar writer, table + NWB helpers
    preprocessing/               # bipolar_reref, run_pipeline_bipolar, bipolar_bands
    qc/                          # detectors, build_exclusions, build_mask, diagnostics
    features/                    # build_pain_epoch_power (cache builder); pac/fooof/slope1f later
    views/                       # EMPTY — the seven view-axis functions land here (P1.3)
    analysis/                    # plotting now; sweep runner, glmm, clustering later
  sbatch/                        # ALL .sbatch here
  tests/
  outdated/                      # superseded, never imported
    notebooks/                   #   DEVIATION: notebooks retired 2026-07-27, not live scratch
    scripts/  sbatch/  preprocessing/
  logs/                          # gitignored
```

Rules:
- All `.sbatch` in `sbatch/`. Submit from the repo root — Slurm `-o` paths are
  relative. Jobs never `cd`; the package is installed editable.
- `outdated/` for dead-but-kept code. **Notebooks are retired** — they live in
  `outdated/notebooks/`, are never imported, and no new ones should be added.
- Move `qc/` OUT of `analysis/` to the top-level derivatives `qc/` (data property,
  not analysis); code uses the existing `--level-root` param so it's mostly config.
- One config module = single source of paths, pinned mask, band defs, cohort paths.
- Commit + push before any definitive run (hash → provenance).
- `.gitignore`: logs, `__pycache__`, `*.out/*.err`, and all data extensions.
- Do the move as its OWN commit BEFORE the Phase 1 refactor.

---

## IO CONVENTIONS

- **Parquet** for all tables (cache, epoch defs, feature tables, sweep results).
  `pip install pyarrow --break-system-packages`. `df.to_parquet` /
  `pd.read_parquet(path, columns=[...])` for partial reads.
- **joblib** (not raw pickle) for fitted models (GLMM/sklearn) and FOOOF fit
  objects. Never pickle tabular data. Pickle is version-fragile / not portable /
  unsafe from untrusted sources.
- **HDF5/Zarr** only if the storage check (P1.2) says the dense-array cache beats
  Parquet on size/speed.
- **JSON** for manifests / provenance / staleness sidecars.
- New artifacts only; don't bulk-convert old CSVs — convert one when you next touch it.

---

## GIT PERMISSION FOR CLAUDE CODE

Goal: Claude Code commits+pushes BEFORE running a definitive/array script so the
provenance hash matches the code that ran. Mechanism (verify against current
Claude Code docs — settings may have changed):
- Grant git permission in the project's Claude Code settings (approve or
  auto-approve git operations for this repo).
- Standing CLAUDE.md instruction: "before any definitive/array run, stage +
  commit + push; warn if the tree is dirty." Your provenance code already reads
  the commit hash, so this closes the loop.

---

## IMMEDIATE NEXT ACTIONS (this week)

1. P0.1 pin the mask (a decision + one review session).
2. P0.3 install pyarrow; add the io/ helper + sidecar writer.
3. P0.4 repo cleanup commit (create outdated/, move qc/ out of analysis/).
4. P0.2 lock discovery cohort + add --split flag.
5. Kick off BG.1 (finish PSD) + BG.2 (1/f slope) as background arrays.
6. P0.6 dtype audit, then P1.1 refactor the cache builder → P1.2 storage check.

Rationale: 1-4 are cheap locks; 5 fills the cluster with useful work immediately;
6 is the first real build and depends on the locks.