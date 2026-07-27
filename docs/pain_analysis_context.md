# pain_analysis/ — context & handoff

Context for picking this work up in a fresh session. Covers the pain-score
epoch featurization + plotting pipeline in `pain_analysis/`, its output
naming convention, current cohort status, and what's next.

## Pipeline shape (for orientation)

- `build_pain_epoch_power.py` — the only script that touches NWB. Per
  subject/session: finds each pain-score event's 5-minute pre-event PSD
  epoch, applies the raw_voltage QC mask, averages log-power per
  channel/freq-bin over the epoch. Writes
  `cache/sub-XXX_ses-YY_epoch_channel_power.csv` + a sidecar
  `*.provenance.json` (mask label, epoch params, git commit). Expensive —
  run on `normal`/`ckeller1` with real memory (16GB was NOT enough for one
  subject with a long session and OOM'd; 32GB has been reliable so far).
- Five plotting scripts, all pandas/matplotlib-only (no NWB, so cheap and
  freely re-runnable): `plot_pain_heatmaps.py` (delta log-power),
  `plot_pain_zscore_heatmaps.py` (z-score vs. own none-bin baseline),
  `plot_pain_epoch_scatter.py` (PSD vs. freq, mean ± SEM band),
  `plot_epoch_count_violin.py` (raw pain-event counts per subject/bin),
  `plot_band_violin.py` (canonical-band z-scores, 3 layouts: by_band,
  by_region, grid).
- `common.py` holds everything shared: cache loading, region grouping
  (`config.categorize_desikan_killiany`/`region_for_dk_label`), subject/epoch
  aggregation, z-score computation, band aggregation, the two pain-bin
  schemes, and the plot-output naming helpers (`make_run_dir`,
  `write_run_provenance`) documented below.

## Output naming convention (as of 2026-07-21)

```
plots/<plot_type>[_<variant>]/<pain_bin_scheme>/<run_name>_<timestamp>/
```

- **`plot_type`** — one folder per script+layout: `delta_heatmap`,
  `zscore_heatmap`, `epoch_scatter`, `epoch_count_violin`,
  `band_violin_by_band` / `band_violin_by_region` / `band_violin_grid`.
  Always the top-level folder, passed as the first segment of
  `common.make_run_dir(..., category=...)`.
- **`pain_bin_scheme`** — `absolute` or `subject_relative` (see below).
  Omitted entirely for scripts that don't support the split
  (`epoch_scatter` has no pain-bin-scheme concept). `plot_pain_zscore_heatmaps.py`
  uses its `--row-order` (`default`/`cluster`/`effect_size`) in this slot
  instead, since that's its analogous "variant" axis, not a pain-bin scheme.
- **`run_name`** — a LABEL ONLY. Never re-encodes plot type or scheme (those
  are already parent folders) — e.g. `full_cohort`, `pilot_5subj`, not
  `full_cohort_zscore_heatmaps_v3_seaborn`. **A timestamp is always
  appended** (`{run_name}_{timestamp}`, or `{timestamp}_n{k}subj` if no
  `--run-name` given) — reruns never collide or silently overwrite a prior
  run's `provenance.json`. This was a real bug we hit: two scripts sharing
  an identical category+run_name overwrote each other's provenance before
  this fix.
- `provenance.json` in every run folder now includes an explicit `subjects`
  list (not just cache-file paths, which only implicitly encode `sub-XXX` in
  filenames) — check this field, not the folder name, to know exactly which
  subjects a given figure reflects. Folder names like the historical
  `full_cohort_*` ones are NOT reliable for this (cohort size varies run to
  run and older folders predate the `subjects` field).

**Pre-2026-07-21 output folders were NOT migrated** — e.g.
`plots/frequency_heatmaps/`, `plots/band_power_analysis/pain_bins_*/full_cohort_v2_nonone/`
etc. still exist under the old ad-hoc scheme and are left as historical
snapshots. Only code going forward uses the convention above.

## Two pain-bin schemes

- **absolute** (default): fixed cutpoints shared across all subjects
  (`config.PAIN_BIN_EDGES`) — none=0, low=1-3, medium=4-6, high=7-10.
- **subject_relative**: none is still score==0, but low/high split at each
  subject's own mean pain score among their non-zero events (no medium) —
  see `common.assign_relative_pain_bins`. Recomputed at plot time from the
  cache's raw `pain_score` column; switching schemes does NOT require
  re-running `build_pain_epoch_power.py`.

## Cohort status (as of 2026-07-21)

- **15 subjects currently in `cache/`**: 071, 085, 088, 099, 150, 176, 191,
  193, 198, 205, 207, 211, 227, 244, 248 (all under the original 19-subject
  exploratory list in `qc_scripts/subjects_qc_raw_voltage_normal.txt` minus
  4 non-sEEG subjects that produce 0 rows: 116, 156, 162, 171).
- **A much larger, separate mask sweep now exists**: the raw_voltage mask
  directory (`qc/raw_voltage/masks/`) currently has labels with far more
  subjects than the pipeline's current default mask
  (`config.DEFAULT_MASK_LABEL = 'gross-std3_satmargin5_logz4'`, only 20
  subjects):
  - `gross-std3_satmargin15_sw` — **82 subjects** (this is almost certainly
    what "83 subjects I currently have masked" referred to; the 83rd file
    listed in that directory is a stray `params.json` sidecar, not a
    subject).
  - `gross-std3_satmargin15_sw_logz4` — 85 subjects.
  - All 15 currently-cached subjects are a subset of the 82-subject
    `gross-std3_satmargin15_sw` set.
  - **Before treating either of these as "the" new cohort**, confirm with
    the user which mask label is actually the intended current-best QC
    candidate (this may have changed since `docs/qc_context.md` was last
    updated — check there first) and whether `config.DEFAULT_MASK_LABEL` /
    `config.exploratory_subjects()` should be updated to match, or whether
    `--mask-label`/`--subjects` should just be passed explicitly per-run
    instead of changing the default.

## Not yet done — pick up here

1. **Train/test subject split.** Flagged early in `docs/featurization_plan.md`
   ("Plan to reserve a subset of subjects purely for exploration... before
   touching the subjects that will go into any confirmatory model — to
   avoid p-hacking/confound risk") but never implemented. User wants this
   done *before* any further tuning on the expanded cohort, specifically to
   avoid peeking at held-out data while iterating on region/band/scheme
   choices. No config constant or CLI convention for this exists yet in
   `pain_analysis/` — needs designing (e.g. a fixed held-out subject list in
   `config.py`, or a `--split {train,test,all}` flag threaded through
   `exploratory_subjects()`/`--subjects` resolution).
2. **Re-run `build_pain_epoch_power.py`** for whichever subjects in the
   82-subject `gross-std3_satmargin15_sw` set aren't yet in `cache/` (67
   subjects as of this writing) — the expensive step, do this on
   `normal`/`ckeller1` with `--mem=32GB` (see notes above), `bash -lc
   "module load python/3.12 && source .../activate && ..."` so module loads
   survive the `srun` node allocation.
3. **Re-run `plot_pain_zscore_heatmaps.py`** (explicitly requested) and any
   other plots against the expanded, train-split cohort, using the naming
   convention above.
4. **1/f slope** — raised as a possible follow-on analysis. Not implemented.
   All the data needed is already in the cache (`mean_log_power` per
   log-spaced freq bin); a simple slope is `np.polyfit(log10(freq_bin_centers),
   mean_log_power, deg=1)` per epoch/region, reusable with all the existing
   z-score/violin/heatmap machinery. A more rigorous aperiodic/periodic
   decomposition (FOOOF/specparam-style) is a bigger lift (new dependency,
   per-epoch fit-quality checks) — start with the simple slope first.
