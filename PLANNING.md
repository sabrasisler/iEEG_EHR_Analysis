# PLANNING.md — iEEG_EHR_Analysis

**What lives here:** phases, milestones, and the *sequencing rationale* — the
project's shape. **Lifespan:** slow and deliberate. This file changes only when
the shape changes (a phase completes, a milestone moves, a new phase becomes
necessary).

**Stays coarse.** If you catch yourself adding fine-grained sub-task detail here,
stop — that belongs in `TASKS.md`. Tasks do NOT have to descend from planning;
planning is the skeleton tasks hang near, not a parent of every task.

Routing (see `docs/WORKFLOW.md` for the full table): standing rule → `CLAUDE.md`
· phase/milestone → here · a thing to DO → `TASKS.md` · a thing you're THINKING
→ `SCRATCHPAD.md` · a settled call + why → `DECISIONS.md` · a thing that
happened → `docs/labnotebook/YYYY-MM-DD.md`.

Companions: `docs/architecture.md` (data/layer model), `docs/view_registry.md`
(the seven view axes), `docs/kickoff_plan.md` (the source this file was split
from; still holds repo-org and IO detail).

**Sequencing rule:** anything BAKED INTO the cache (QC mask, epoch definition)
or that prevents CONTAMINATION (cohort split) is locked BEFORE building cache at
scale — cheap now, expensive to retrofit.

**Scope note:** Phase 3 (confirmation GLMM, Aims 2/3) is FAR AWAY and
deliberately not detailed. The near-term center of gravity is Phase 2, which is
much larger than a single "signal existence" check — it is the joint exploration
of feature signal AND methodological choices.

---

## PHASE 0 — Locks & foundations (in progress)

Cheap locks that are expensive to retrofit. Everything in Phase 1 depends on
these.

| ID | Milestone | Status |
|---|---|---|
| P0.1 | Pin the canonical QC mask (`_satmargin15_sw` vs `_sw_logz4`) in config | open |
| P0.2 | Lock discovery subjects to a cohort file; add `--split` gate so hold-out is unreachable by default | open |
| P0.3 | IO conventions + deps: pyarrow installed, every writer emits a provenance/staleness sidecar | **done 2026-07-27** |
| P0.4 | Repo cleanup as its own commit (`outdated/`, `qc/` out of `analysis/`) | **done 2026-07-27** |
| P0.5 | `CLAUDE.md` at repo root | **done 2026-07-27** |
| P0.6 | dtype audit — validate float32 round-trip, standardize the cache on float32 | **done 2026-07-27** |

Why this order: the mask and the cohort split are *baked in* downstream — the
mask into every cache file, the split into what you are allowed to have looked
at. Both are one-line config values now and a full re-run (or a compromised
hold-out) later.

## PHASE 1 — Build the cache

Depends on P0.1 (mask) + P0.2 (cohort). Two prerequisites are now **settled**:
P0.3 (IO) — pyarrow/joblib installed, `ieeg_ehr.io` writes every artifact with a
provenance/staleness sidecar, cache + analysis paths built (`docs/io_conventions.md`).
P0.6 (dtype) — cache stores float32, views compute in float64
(`config/cache_params.py`). Both: DECISIONS 2026-07-27.

| ID | Milestone |
|---|---|
| P1.1 | Refactor `build_pain_epoch_power.py` → epoch definitions (tiny Parquet index) + the per-2s-window, per-channel, per-bin, QC-masked, LOG-power cache, one Parquet per subject/session. No epoch-averaging, no normalization (those are views). |
| P1.2 | Storage sanity check — build one subject, `du -sh`, extrapolate to ~67 then ~250, confirm Oak quota. Same subject as HDF5/Zarr for size/read-speed comparison. |
| P1.3 | Implement the view layer (`views/`) — all seven axes as functions, optional `save_path` + staleness sidecar, default recompute. |
| P1.4 | Build the cache across the discovery cohort via Slurm array (one task per subject). |
| P1.5 | Reproduce one prior figure through the new cache+view path to confirm numerical fidelity. |

P1.5 is the gate: do not trust the refactor until an old number comes back out of
the new path.

## PHASE 2 — Exploratory analysis (discovery cohort ONLY)

Depends on Phase 1. **The near-term bulk of the work.** Signal existence and
methodological choice are NOT separate questions — they are the SAME sweep,
because robustness of a signal across reasonable methodological variations is
itself evidence. The sweep's axes therefore include both the feature grid and the
view-config grid.

| ID | Milestone |
|---|---|
| P2.1 | Feature-level QC — choice-independent, on pre-normalization cached power, stored under `qc/feature_level/`. Thresholds set on STRUCTURAL grounds before looking at pain relationships. |
| P2.2 | Signal + methodology sweep (the core exploratory activity). Grid as ROWS in one `sweeps/results.parquet`. Output = ranked NOMINATIONS, not findings. |
| P2.3 | Robustness reading — which nominations survive across methodological variations. |
| P2.4 | Subject-level clustering on per-subject effect vectors → candidate phenotypes (Aim 2 groundwork). |
| P2.5 | Maintain the exploration log throughout (now: `docs/labnotebook/` + `docs/analyses_run.md`). |

### Named questions (the level-2 folders under `analysis/pain/`)

A level-2 folder is opened DELIBERATELY and only for a question named here
(`architecture.md` PART 5). Everything else goes to `sweeps/` or `scratch/`.

| Question | Folder | What it asks |
|---|---|---|
| PSD physiology | `psd_physiology` | Descriptive, pre-model: across the discovery cohort, what does the pain-related change in the power spectrum LOOK like per region — where is it (heatmaps) and what shape does it have (spectra)? Opened 2026-07-29. Deliberately not a hypothesis test: its output feeds P2.2's sweep axes and P2.6's frozen feature set, and nothing from it is a finding. |

### Level-1 event `meds` — medication administration patterns

A SECOND level-1 event beside `pain`, opened 2026-09-03. The unit of analysis is
a drug administration from the EHR MAR export, not a pain epoch; nothing in it
reads the PSD cache, the QC masks, or the view chain. Level-1 vocabulary in
`architecture.md` PART 5 anticipated `opioid`; `meds` is used instead because the
drug set is all analgesics, not opioids alone.

| Question | Folder | What it asks |
|---|---|---|
| Administration patterns | `administration_patterns` | Descriptive characterization of analgesic administration across the ~96 subjects with a MAR export: which drugs carry enough data to analyze, when they are given (clock time, inter-dose interval), how administration rate and dose evolve across the hospitalization, and what they are co-prescribed with. Opened 2026-09-03. Purely descriptive — no neural data, no hypothesis test. Code: `src/ieeg_ehr/med_analysis/`. |
| Pain coupling | `pain_coupling` | Where on the 0-10 charted pain scale each drug is actually used: administrations of the four most-administered analgesics binned by the pain score assessed in the 30 min before the dose. Opened 2026-09-03. A different question from the one above rather than another view of it — that one asks what was given, this asks what the chart said first. Still no neural data, still descriptive. Explicitly NOT causal: scheduled drugs are given on a clock whatever the assessment says, and an assessment is often charted precisely because a PRN dose was requested. Code: `med_analysis/pain_link.py` + `plot_pain_score_bars.py`. |

Downstream, this is the descriptive groundwork for **Aim 3 (opioid modulation)**
in Phase 3, and for `BG.6` (EHR/confound tables joined to epoch definitions) — the
tidy administration table is the join key for a medication-state covariate. It is
NOT itself either of those.

| P2.6 | **FREEZE** — dated freeze doc: ranked directional hypotheses, frozen feature set + view config, model spec, correction plan + n_tests, predicted directions. Ends exploration. |

P2.6 is the hinge where accumulated notebook narrative becomes `DECISIONS.md`
entries. Nothing before it is a finding.

## PHASE 3 — FAR FUTURE (not detailed)

Model development (Huang-logic GLMM replication) is its own design conversation,
AFTER P2.6 produces a frozen feature set. Then confirmation on the matched
hold-out, Aim 2 (BDI/mood), Aim 3 (opioid modulation). Deliberately not planned
in detail — planning it now would be planning against an unfrozen feature set.

## PHASE 4 — Matched hold-out (after the ~150 land; offline)

Build the matched hold-out cohort file OFFLINE, on the PHI side, matching
hold-out to discovery on {pain-range > 4, sEEG/ECoG, age, sex}. Age is PHI, so
matching runs where the PHI master lives; only the anonymized `subject_id →
cohort` assignment plus SAFE axes cross to Sherlock.

---

## BACKGROUND JOBS (continuous, parallel to the phases above)

Independent and embarrassingly parallel (per-subject Slurm arrays); they don't
block PSD work. Start the slow ones early. All use the pinned mask (P0.1) for
cross-family consistency.

| ID | Job | Priority |
|---|---|---|
| BG.1 | Finish PSD for the remaining subjects (~150 incoming) | highest — primary data |
| BG.2 | 1/f slope (polyfit) across all subjects | high — cheapest new feature |
| BG.3 | FOOOF / specparam | medium — not needed until finer sweep tiers |
| BG.4 | PAC extraction (own family, time-domain Hilbert) | start early, slow |
| BG.5 | Connectivity / coherence / PLV | lowest; ROI pairs first |
| BG.6 | EHR/confound tables joined to epoch definitions | cheap, enables confound controls |

Orchestration: Slurm ARRAY jobs (one task per subject), NOT multi-agent work.
Long arrays on `normal`; keep `ckeller1 --qos=high_p` (4-job cap) free for
interactive work.
