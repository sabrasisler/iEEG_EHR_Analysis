# iEEG QC pipeline — context & handoff

Context for picking this work up in a fresh session. Covers the raw-voltage QC /
artifact-rejection pipeline in `qc_scripts/`, its file organization, the design
principles behind it, and how to run each step.

## Current state (as of 2026-07-10)

Pipeline code is complete, committed, and pushed (`master`). A first cohort has
been processed into `analysis/qc/raw_voltage/`:

- **Metrics** exist for **17 subjects**: `039, 071, 085, 088, 099, 150, 176, 191,
  193, 198, 205, 207, 211, 217, 227, 244, 248`. `sub-236` was still running its
  metrics pass at last check (large, 109 runs) — once done, run its rollup
  (`build_exclusions --subjects 236`) and rebuild the masks to make it 18.
  Random-draw non-sEEG subjects (`116, 156, 162, 171`, and `034`) produced 0 rows.
- **Per-type exclusions** exist at the config-default labels
  (`saturation/pct0`, `flatline/var5e-13`, `square_wave/frac0.9`,
  `gross_artifact/std5`) plus swept `gross_artifact/std4` and `gross_artifact/std3`.
- **Masks**: `masks/baseline/` (all defaults, gross std5), `masks/gross-std4/`
  (gross at std4, others default), and `masks/gross-std3/` (gross at std3, others
  default). All have `summary/` (5 `exclusion_rates_*.csv` + `flagged_for_review.csv`).
  `baseline` has ~117 flagged example plots; `gross-std4` has example plots +
  a std4-vs-5 diff (`validation/threshold_sweeps/gross_artifact_std4_vs_std5/`).
  `gross-std3` has flagged + 20 random example plots
  (`masks/gross-std3/plots/{flagged_examples,random_examples}/`) plus a
  std3-vs-4 diff (`validation/threshold_sweeps/gross_artifact_std3_vs_std4/`).
- **Baseline exclusion rates** (per channel, % of 60s bins, n=2306 channels):
  saturation mean 0.70% / flatline 0.57% / square_wave 0.66% / gross 0.19% /
  **any 1.46%** (medians ~0.1–0.4%; heavy right tails — a few channels lose ~⅓).
  gross std5→std4 raised gross mean 0.19%→0.22% (max 0.68%→1.00%), flagged rows
  101→116 — a gentle change. gross std4→std3 raised gross mean further to 0.26%
  (max 2.03%), any-mean to 1.47%, flagged rows to 118; the std3-vs-4 diff found
  4810 newly-added bins across 2630 channels (vs. 3413 bins / 2169 channels for
  std4-vs-5) — std3 is noticeably looser than std4, concentrated in a few noisy
  channels (e.g. sub-085 RMH1-8 pick up dozens of extra bins each).

- **Reusable threshold-sweep tooling** (`qc_scripts/diagnostics/`, tracked —
  NOT `tmp_*` scratch): `threshold_summary.py` (per-label exclusion-rate table,
  no `build_mask` needed — reads `exclusions/<type>/<label>/` directly) and
  `threshold_diff.py` (green/red trace diff plots between two labels of the
  same type, direction-agnostic). Both take `--artifact-type` +
  label(s)/`--baseline-label`/`--compare-label`, so any future sweep (any
  artifact type, any threshold pair) reuses the same two scripts instead of a
  new hardcoded one. Output root moved from `_validation/` to `validation/`
  (see `config.validation_dir()`), with sweep-specific output under
  `validation/threshold_sweeps/` (see `config.threshold_sweep_dir()`).

- **Flatline sweep** (`var5e-13` default → looser `var1e-12`, `var1e-11`):
  mean exclusion 0.57%→0.93%→3.78%, channels flagged 19→49→58,
  total bins excluded 45212→49320→383303. The var5e-13→var1e-12 step is gentle;
  var1e-12→var1e-11 is a big jump (mean triples, some channels hit 100%
  excluded) — var1e-11 is likely too loose as a default without per-channel
  review. Diff plots: `validation/threshold_sweeps/flatline_var1e-12_vs_var5e-13/`,
  `flatline_var1e-11_vs_var1e-12/`. Table: `flatline_threshold_summary.csv`.

- **Flatline per-channel-relative (log-variance z-score) mode — IN PROGRESS,
  NOT YET VALIDATED.** The absolute `var_thresh` cutoff was found to
  misclassify some channels' genuinely-quiet-but-real signal as flatline (a
  channel's own normal baseline can sit close to the global cutoff — e.g.
  `sub-085 LAH6`, whose one run averaged ~1.25µV RMS, right at the `var1e-12`
  line, despite no visibly dead period in the raw trace). New mode added to
  `build_exclusions.py`/`config.py` (`--std-thresh` on `flatline` →
  `flatline_channel_log_stats()` + z-score on `log10(variance)`, one-sided
  low, OR'd with the absolute floor as a backstop for genuinely fully-dead
  channels) — see `build_exclusions.py`'s module docstring for the full
  rationale (log-scale chosen because variance is lognormal-shaped/spans many
  orders of magnitude; a raw-scale z would be dominated by the right tail and
  blind at the low end that flatline cares about).
  `logz5`, `logz4`, and `logz3` (label = `logz<std_thresh>`) are now all built
  (`exclusions/flatline/logz{3,4,5}/`). A `logz4_vs_logz5` diff
  (`validation/threshold_sweeps/flatline_logz4_vs_logz5/`) has been run — but
  **results still not reviewed**: no `threshold_summary` table yet, no
  decision on whether this mode should become the new default. Same
  investigation found `gross_artifact`'s raw-variance z-score has an even
  more extreme version of this skew (one channel's 60s-window variance:
  mean=4.84e-9 vs median=3.15e-11, a 153x ratio) but that's a **known,
  accepted tradeoff** already named in `detect_gross_artifact.py`'s docstring
  (masks smaller bursts behind bigger ones, doesn't cause false-positives the
  way flatline's did) — decided NOT to touch `gross_artifact` for now.

- **Combined test mask `gross-std3_satmargin5_logz4`** (gross_artifact=std3,
  saturation=pct0_marginfrac0.05, flatline=logz4, square_wave=default):
  built + summarized (`masks/gross-std3_satmargin5_logz4/summary/`, 116
  flagged rows, any-mean 1.74%/max 40.0%, up from baseline's 1.46%). Has 20
  top-flagged + 20 random example plots
  (`masks/gross-std3_satmargin5_logz4/plots/{flagged_examples,random_examples}/`)
  plus 3 specifically-requested plots in `plots/requested_examples/`:
  `sub-248 run-FA6152DU LAMY6`, `sub-207 run-EA189782 LHPC4`,
  `sub-198 run-SA3332TR LMDT1` (channel picked automatically for the
  sub-248 case since none was specified — many channels tied at 26 excluded
  bins in that run, LAMY6 used as representative). NOTE: `sub-236` was
  excluded from this mask's build (not shared by all 4 exclusion types at
  these specific labels — hasn't been root-caused yet).

- **Unidentified artifact pattern — flagged for follow-up, not yet
  characterized.** All three of the specifically-requested example runs
  above (`sub-248/FA6152DU`, `sub-207/EA189782`, `sub-198/SA3332TR` — which
  also all happen to be each subject's **2nd run**, starting at exactly
  120.0min/7200s into the session) show a distinctive shape: a flatline-like
  quiet stretch with more noise mixed in than a true flatline, bounded on
  both sides by two larger artifactual bursts. Visual impression is a
  lead/electrode being unplugged and replugged (quiet gap while
  disconnected, transient artifacts at disconnect/reconnect). Not yet
  confirmed against ground truth or given a detector. Multi-channel example
  plots for these three runs requested in `validation/` for a closer look
  (see whichever `validation/*artifact*` or similarly-named subfolder was
  most recently added — check `validation/` listing for the exact path, this
  entry predates that job completing).
  Related but distinct: `sub-198 LPCN2` at 4000-4100s in the same
  `SA3332TR` run looks visually flat but is NOT a statistical outlier even
  under the relative logz method — z only reaches ~-0.61 relative to that
  channel's own baseline (would need `std_thresh≈0.6` to catch it, which is
  far too loose to use globally, since z=0.6 is only the ~27th percentile).
  Likely another instance of a channel whose own normal signal just runs
  quieter (same pattern as `sub-085 LAH6` above), not a threshold-tuning gap.

- **Saturation rail-margin sweep** (`pct0` default/margin-off → looser
  `pct0_marginfrac0.05`, `pct0_marginfrac0.1`): mean exclusion
  0.70%→1.10%→1.14%, total bins excluded 98631→189455→198996. Margin 0→0.05 is
  the big jump (roughly doubles excluded bins); 0.05→0.1 adds comparatively
  little more — diminishing returns past 0.05. Concentrated in a few channels
  (e.g. sub-085 RMH*, sub-088 LAHP1). Diff plots:
  `validation/threshold_sweeps/saturation_pct0_marginfrac0.05_vs_pct0/`,
  `saturation_pct0_marginfrac0.1_vs_pct0_marginfrac0.05/`. Table:
  `saturation_threshold_summary.csv`.
- **Threshold-label meaning:** `pct0` is **saturation** (`sat_frac_thresh=0` →
  flag a window if **>0**, i.e. ≥1 sample, hits the rail — the old MIN_SAMPLES=1).
  It is NOT "flag nothing". `var5e-13`=flatline variance floor, `frac0.9`=square
  bimodal fraction, `std5`/`std4`=gross z-threshold.
- **Plotting gotcha:** `plot_flagged_runs --review-csv` over ~100+ flagged
  channels × 2 runs each ≈ 3h of NWB reads and can hit walltime. Give it ≥5h, or
  parallelize with `build_plot_targets` + `plot_targets_array.sbatch`.

## Environment (Sherlock HPC)

- Repo: `/home/groups/ckeller1/sisler/iEEG_EHR_Analysis` (git; remote
  `github.com/sabrasisler/iEEG_EHR_Analysis`, branch `master`).
- **Never run Python on the login node.** Its system Python is 2.7/3.6 and lacks
  the deps. Use Slurm (`sbatch`, or `srun -p dev` for quick interactive tests).
- Environment for every job:
  ```
  module load python/3.12
  source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # pipeline is single-threaded numpy/h5py
  ```
- Partitions: `normal` (shared, keep big/long array jobs here), `ckeller1` (lab
  owner partition, add `--qos=high_p`; `high_p` caps you at **4 concurrent jobs**;
  keep it free for interactive lab work when asked), `dev` (interactive `srun`
  only, ≤2h, small).
- Memory: a single run loads all channels into RAM (a ~2hr run ≈ 10–11 GB in
  float64), so `run_pipeline` tasks want **≥24–32 GB**. The cheap CSV steps
  (`build_exclusions`/`build_mask`/`summarize`) need ~8 GB.
- `git` on compute nodes is old (`/usr/bin/git`, no `-C`); provenance code uses
  `cwd=REPO_DIR` instead.
- Raw data: NWB files under `$RAW_DIR`
  (`/oak/.../iEEG_EHR/iEEG_NWB`); a file registry CSV drives subject/run
  discovery (`config.FILE_REGISTRY_CSV`). Signal is in
  `nwb.acquisition['ElectricalSeries_sEEG']`, unit volts. Some subjects have **no
  sEEG series** (`KeyError('ElectricalSeries_sEEG')`) → produce 0 rows; handled
  gracefully, but random subject draws will hit them.

## Core design principle: metric/threshold split

The **expensive** raw-NWB pass stores only *continuous metrics* (never an
`excluded` flag). **Cheap, CSV-only** steps own thresholding + 60s bucketing.
Retuning any threshold then re-runs only the cheap step (minutes over CSVs),
never the raw pass (hours re-reading NWB).

```
RAW NWB ──(run_pipeline, expensive, once)──▶ metrics (continuous per-window values)
metrics ──(build_exclusions, cheap, per artifact type, sweepable)──▶ per-type 60s exclusion tables
per-type exclusions ──(build_mask, cheap)──▶ ONE combined 60s mask  ──▶ feeds bipolar re-referencing
```

## Output layout: `analysis/qc/<level>/`

Root: `/oak/.../derivatives/sisler/analysis/qc/<level>/`. A *level* is a
processing stage; current level = **`raw_voltage`** (future: `bipolar`,
`features`). Code is level-agnostic (`--level-root`).

```
analysis/qc/raw_voltage/
  metrics/                          # EXPENSIVE, produced once by run_pipeline
    per_window/ sub-XXX_{saturation,flatline,square_wave,gross_artifact}.csv
    run_info/   sub-XXX.json         # per-subject provenance: detection params + git + run_timestamp
  exclusions/<type>/<label>/         # CHEAP per-artifact-type 60s exclusion (build_exclusions)
    sub-XXX.csv                      #   subject,session,run,channel,bin_start,bin_end,excluded
    params.json                      #   thresholds + git + run_timestamp
  masks/<mask_label>/                # CHEAP combined mask (build_mask) — the downstream artifact
    sub-XXX.csv                      #   per-type excluded_<type> columns + OR'd `excluded`
    params.json                      #   which per-type <type>/<label> fed each + provenance
    summary/  exclusion_rates_*.csv, flagged_for_review.csv   (summarize_exclusions)
    plots/    flagged_examples/*.png                          (plot_flagged_runs)
  validation/                        # diagnostic scratch (e.g. square-wave tuning), NOT canonical
    threshold_sweeps/                 # threshold_summary.py / threshold_diff.py outputs
```

**Label naming:** exclusion `<label>`s are self-documenting from the threshold —
`saturation/pct0`, `flatline/var5e-13`, `square_wave/frac0.9`, `gross_artifact/std5`
(auto via `build_exclusions.label_for`). A threshold sweep makes sibling folders,
e.g. `gross_artifact/std4`, `std5`. Mask labels are human-chosen (e.g. `baseline`).

**Prior eras (superseded, left on disk for reference):** `analysis/qc_session_rail/`,
`analysis/qc_variance/`, `analysis/qc_variance_gross_thresh4/`,
`analysis/qc_variance_padded30/`. These used an older schema (had `excluded`
columns, DC-offset gross metric, no square_wave, ±30s absolute-time padding).
The current `qc/raw_voltage/` is the canonical one.

## The four detectors (all per-channel, 2s windows unless noted)

Config lives in `qc_scripts/config.py`. All detectors are **independent** (no
detector's output feeds another's baseline — matters for permutation analyses).

1. **saturation** (`detect_saturation.py`) — amplifier rail clipping. The rail is
   inferred per channel **pooled over the whole session** (all runs), then
   confirmed by **cross-channel agreement**: if ≥`SAT_AGREEMENT_THRESHOLD` (0.25)
   of a session's channels share the same repeated extreme value, that value is
   the rail for *every* channel (so channels that never saturate still get one).
   Rails are very stable (~3200 µV; a few subjects ~50 µV gain). Stores
   `metric_value` = fraction of samples at/beyond the rail, plus `window_max_abs`,
   `rail_value`, `rail_source`. Threshold knob: `sat_frac_thresh` (default 0 →
   any sample at rail flags the window).
2. **flatline** (`detect_flatline.py`) — dead channel. `metric_value` = per-window
   variance (V²). Threshold: `var_thresh` (`FLATLINE_VAR_THRESH = 5e-13`); excluded
   when variance **below** it.
3. **square_wave** (`detect_square_wave.py`) — digital/relay two-level artifact
   (e.g. 0–50 µV square wave) that flatline (high variance), saturation (not at
   rail), and gross_artifact (mean-neutral) all miss. Metric = **bimodal-extremes
   fraction**: fraction of samples pinned within `SQUARE_EPS_FRAC` (0.05) of the
   window's own min/max. **Dimensionless → amplitude- AND frequency-independent**,
   so no per-subject tuning. Stores `metric_value` (fraction) + `range`. Threshold:
   `frac_thresh` (0.9) AND `range > SQUARE_MIN_RANGE_V` (derived from the flatline
   threshold ≈1.4 µV, to not re-flag flat windows). Fast square waves flag directly;
   slow ones get their plateaus caught by flatline + transitions by square_wave,
   unioned by the 60s rollup. Validated on `sub-093/LOF9` (fast) and
   `sub-244/LDMB2` (slow).
4. **gross_artifact** (`detect_gross_artifact.py`) — high-variance/amplitude bursts
   (unplug/replug), on **60s** windows. Metric = raw per-window variance;
   `session_mean`/`session_std` of per-subbin variance also stored. Threshold:
   `std_thresh` (5.0), one-sided high: excluded when
   `z = (var − session_mean)/session_std > std_thresh`. (This replaced an earlier
   DC-offset/mean-based metric that missed mean-neutral bursts.)

### Exclusion granularity (important)
Detection is at 2s; **exclusion is rolled up to the enclosing 60s bin** (a 60s bin
is excluded if ANY 2s window in it is flagged). This replaced an earlier ±30s
absolute-time padding step (which was slow — per-channel time-sorting). Tradeoff
(explicitly chosen): coarser margin (0–60s around an event vs. exactly ±30s) and
no cross-run bridging. `pad_exclusions.py` / `build_run_start_times.py` are
superseded by this.

## Scripts (`qc_scripts/`)

| Script | Role | Cost |
|---|---|---|
| `config.py` | paths, thresholds, `git_provenance()`, `run_timestamp()`, `ARTIFACT_TYPES` | — |
| `io_utils.py` | NWB loading (`load_all_channels`, `load_channels_subset`), subject/run discovery | — |
| `detect_{saturation,flatline,square_wave,gross_artifact}.py` | per-detector classifiers (metric only) | — |
| `run_pipeline.py` | **expensive** single-read-per-run pass → `metrics/` + `run_info/` | hours |
| `build_exclusions.py` | per-type threshold + 2s→60s bucketing → `exclusions/<type>/<label>/`; `--subjects`, `--artifact-type all`, threshold overrides | minutes |
| `build_mask.py` | OR chosen per-type exclusions → `masks/<label>/` | minutes |
| `summarize_exclusions.py` | chunked population stats off a mask → `summary/` (memory-safe, no giant concat) | minutes |
| `plot_distributions.py` | streaming metric-value histograms | minutes |
| `plot_flagged_runs.py` | raw-trace example plots with shaded exclusions (computes `excluded` on the fly at config thresholds); `--review-csv`, `--targets`, `--random-any`, `--exact-csv` (array) | fast |
| `build_plot_targets.py` | pick plot targets (flagged + random) into a CSV for a plot array | fast |
| Array sbatches | `run_pipeline_qc_raw_voltage_normal.sbatch`, `build_exclusions_array.sbatch`, `plot_targets_array.sbatch` | — |

Shading colors (seaborn "deep"): saturation `#4c72b0`, flatline `#dd8452`,
square_wave `#c44e52`, gross_artifact `#55a868`.

## How to run (end to end)

```bash
LEVEL=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc/raw_voltage

# 1. Metrics (expensive) — per-subject array on normal
sbatch qc_scripts/run_pipeline_qc_raw_voltage_normal.sbatch          # reads subjects_qc_raw_voltage_normal.txt
#   or one subject:  python -m qc_scripts.run_pipeline --subjects 217 --level-root $LEVEL

# 2. Per-type exclusions (cheap) — all types, all present subjects (or --subjects / array)
python -m qc_scripts.build_exclusions --level-root $LEVEL --artifact-type all
#   sweep one type:  python -m qc_scripts.build_exclusions --level-root $LEVEL --artifact-type gross_artifact --std-thresh 4   # -> gross_artifact/std4
#   per-subject array:  sbatch --array=0-N%8 --export=ALL,SUBJECTS_FILE=... qc_scripts/build_exclusions_array.sbatch

# 3. Combined mask (cheap)
python -m qc_scripts.build_mask --level-root $LEVEL --label baseline   # picks config-default per-type labels

# 4. Summary + example plots
python -m qc_scripts.summarize_exclusions --mask-dir $LEVEL/masks/baseline
python -m qc_scripts.plot_flagged_runs --level-root $LEVEL --review-csv $LEVEL/masks/baseline/summary/flagged_for_review.csv --n-runs 2 --plots-dir $LEVEL/masks/baseline/plots
python -m qc_scripts.plot_flagged_runs --level-root $LEVEL --random-any 15 --plots-dir $LEVEL/masks/baseline/plots
```

## Provenance

Every sidecar (`metrics/run_info/sub-XXX.json`, `exclusions/.../params.json`,
`masks/.../params.json`) records `git_provenance()` (commit hash + dirty flag +
modified files) and `run_timestamp()`. Scripts **warn when the git tree is
dirty** — a hash is only faithful if committed. Workflow: commit+push before a
definitive run. `.gitignore` excludes logs (`logs/`, `**/logs/`, `*.out`,
`*.err`), `__pycache__`, `qc_scripts/tmp_*`, `qc_scripts/*.csv`.

## Gotchas / lessons

- **Non-sEEG subjects** (`034, 116, 156, 160, 162, 171, …`) → 0-row outputs; expected.
- **Never re-read raw NWB just to retune a threshold** — that's the whole point of
  the split; use `build_exclusions`.
- `summarize`/`plot_distributions` previously OOM'd (64–150 GB) by concatenating
  all subjects; now stream/chunk — keep it that way as the cohort grows to ~250.
- Manifests/params are **per-subject / per-(type,label)** and written atomically
  to survive parallel Slurm array tasks (don't reintroduce a shared merged file).
- A subject whose pipeline task is still running has **partial** metric CSVs (and
  no `gross_artifact.csv` yet — gross needs all runs); exclude it from
  `build_exclusions` via `--subjects` until done.

## Likely next steps (discussed, not yet built)

- **Whole-run / whole-channel exclusion**: flag an entire run or channel when its
  artifact fraction is high enough (before bipolar re-referencing).

## Bipolar re-reference + PSD (built)

The bipolar level is now implemented, split across three folders by concern —
preprocessing (re-reference + FFT), QC (exclusion), and archive:

```
preprocessing/                     # NEW active reref+FFT code (this is NOT qc_scripts/)
  bipolar_reref.py                   pairs, re-referencing, per-2s variance, Welch->50 log bins
  run_pipeline_bipolar.py             fused pass: single read/run -> variance metrics + PSD NWB
  bipolar_bands.py                    downstream: aggregate stored log bins into canonical bands
  run_pipeline_bipolar_normal.sbatch
outdated/preprocessing/             # ARCHIVED: the old preprocess_ieeg*.py pipeline (pre-metric/
                                     # threshold-split, hard-coded canonical bands, no QC-mask
                                     # integration). Not imported by new code.
qc_scripts/build_bipolar_exclusions.py   # QC ONLY: mask-aware z-score exclusion on the variance metric
```

Design, mirroring the raw_voltage split:
- `preprocessing/run_pipeline_bipolar.py` reads each run's raw NWB exactly once. While the
  bipolar-referenced trace is in memory (transient, never persisted — cheap enough to
  recompute later), it computes BOTH a continuous per-2s-window variance metric (written to
  `qc/bipolar/metrics/per_window/sub-XXX_bipolar_variance.csv`, same metric/threshold split as
  raw_voltage — no thresholding here) AND a PSD band-averaged into 50 log-spaced frequency bins
  (1-250 Hz), written to an NWB file under
  `derivatives/sisler/preprocessed/bipolar_fft/sub-XXX/ses-XXX/` (deliberately outside
  `analysis/`, BIDS-like, namespaced under `sisler/` matching `derivatives/sisler/analysis/`'s
  existing convention, to keep large NWB derivatives away from the CSV-oriented QC tree).
- **PSD windowing (revised 2026-07-13, superseding an earlier 60s-outer-window design)**:
  single-level windowing, no coarser outer window — each `--window-sec` window (default 2.0s)
  is its own periodogram-style PSD estimate (single segment, hann-windowed), stepped by
  `--overlap` (default 0.5 → 1s hop). This was a deliberate lab decision to match the PSD's time
  granularity much more closely to the variance metric's 2s grid, trading away the averaging
  benefit a longer outer window gave (noisier per-window spectral estimate, much finer time
  resolution). Implemented via one `scipy.signal.spectrogram` call per channel across the whole
  run (`bipolar_reref._welch_one_channel`) rather than a manual two-level loop — returns every
  window's PSD at once. `rate` in the NWB = `1/hop_sec` (not `1/outer_sec` as before).
- Each PSD bin is flagged `contains_line_noise` if it overlaps a 60 Hz harmonic (60/120/180/240
  Hz) ± a guard band — log-spaced bins are naturally wide enough at higher harmonics to contain
  the notch in one bin. Canonical bands are NOT computed by the fused pass; a separate
  `preprocessing/bipolar_bands.py` aggregates the stored bins into caller-specified bands later
  (edges chosen to avoid the harmonics, `config.CANONICAL_BANDS_HZ`), so retuning band
  definitions never re-reads raw NWB.
- HDF5 chunking on the PSD arrays: one chunk = one channel's entire run (no time
  sub-chunking). Even at the new ~1s-hop density this stays small — measured on real data
  (sub-039, 2.5hr run): 8999 rows/channel, ~1.8MB/channel; a 24hr run would be ~17MB/channel —
  still comfortably below where sub-chunking would help. `--psd-chunk-max-hours` caps this only
  for unusually long recordings.
- **No per-run sidecar JSON** (removed 2026-07-13) — everything that used to live in a separate
  `..._bipolar_psd.json` next to each NWB (git commit, run_timestamp, window/overlap params, bin
  edges, line-noise config, source_nwb, pairs_diverged, hdf5_chunk_shape) is now embedded
  directly in the `DecompositionSeries.description` field instead, so nothing is lost but no
  per-run file clutter accumulates (git/timestamp/params are identical across every run of a
  subject anyway, and already recorded once per subject in
  `qc/bipolar/metrics/run_info/sub-XXX.json`).
- **`--skip-variance-metrics`**: the bipolar variance CSV is independent of the PSD windowing
  scheme (always 2s non-overlapping, unaffected by `--window-sec`/`--overlap` changes) — use this
  flag (or the sbatch's `SKIP_VARIANCE=1` env var) to recompute PSD-only when only PSD parameters
  changed, without redoing the already-correct variance metrics.
- **`qc_scripts/build_bipolar_exclusions.py`** is the ONLY QC piece for this level — it reads
  exclusively the variance-metric CSVs (never the PSD/NWB output; no QC currently runs on FFT
  output, and that should stay true) and applies a z-score threshold
  (`z = (var - session_mean) / session_std > std_thresh`), same convention as `gross_artifact`.
  The one deliberate difference from `gross_artifact`: its session baseline is **mask-aware** —
  it takes an existing `qc/raw_voltage/masks/<label>/` and excludes any bipolar window whose
  monopolar anode OR cathode is already flagged in that mask from the baseline computation, so a
  known raw-voltage artifact doesn't inflate this detector's idea of "normal" variance. No
  combined bipolar mask yet (standalone detector for now).

**Validated on real Sherlock data (2026-07-10)** — smoke test + 3 real subjects
(sub-039, sub-071, sub-085/85 runs):
- `DecompositionSeries.bands` must be a `FrequencyBandsTable`, not a generic `DynamicTable`
  (`TypeError` otherwise); its `add_band(..., **extra)` DOES forward arbitrary kwargs (confirmed
  `contains_line_noise` round-trips correctly), and `H5DataIO` chunking on `.data` takes effect
  exactly as requested (`h5py` `.chunks` matches).
- `DecompositionSeries.source_channels` does **NOT** survive an NWB write/read round-trip in the
  installed pynwb version (reads back `None`) — `bipolar_bands.py` reads channel names from
  `nwb.electrodes` directly instead, which does survive and is in the same row order.
- Found and fixed a real, **pre-existing** bug in `config.git_provenance()`: `_git()` used
  `.strip()` on `git status --porcelain`'s output, which ate the leading space of the FIRST
  modified file's status code (` M` → `M`), truncating that one file's path by one character in
  every provenance record ever written (e.g. `"reprocessing/x.py"` instead of
  `"preprocessing/x.py"`). Fixed to `.rstrip('\n')` — only the trailing newline should be
  stripped, not meaningful leading whitespace. Affects every prior sidecar JSON's
  `modified_files[0]` when that file had an unstaged (leading-space) status; not a data-
  correctness issue, just cosmetic/provenance, and only for historical records (not worth
  re-running past jobs over).
- **Multiprocessing across channels** (`compute_welch_log_bins(..., n_workers=N)`,
  `ProcessPoolExecutor`) produces bitwise-identical output to the sequential path (verified via
  `np.allclose` with 0.0 max diff) — safe to use for the real run.
- **Measured timing/memory (under the OLD 60s-outer-window scheme)** (sub-085, 83 usable runs —
  2 of 85 registry rows have `n_channels` NaN even in the file registry itself, i.e. pre-existing
  corrupt/unparseable NWBs, skipped gracefully by design): single-threaded ≈1.8 min/run;
  **8-worker ≈40 sec/run** (~2.7x speedup, not linear — process-pool overhead). **MaxRSS ≈48GB
  with 8 workers** vs ≈16-17GB single-threaded — the process pool has real fixed overhead (each
  worker loads the full scipy/numpy/BLAS stack). `run_pipeline_bipolar_normal.sbatch` bumped to
  `--mem=64GB` accordingly (was 48GB, too close to the measured peak). **NOT yet re-measured under
  the new 2s/50%-overlap scheme** — expect meaningfully slower per-subject (sub-039, small/2 runs,
  took 7m53s single-threaded vs ~2min under the old scheme, since ~60x more PSD windows are now
  computed/written per run) — watch the first real array tasks of the 2026-07-13 full recompute
  for grounded numbers on a large subject before assuming a multiplier.
- An accidental `srun` test without explicit `-n 1` defaulted to 3 concurrent tasks on this
  cluster, which **tripled every CSV row** via concurrent `config.append_table` writes (no header
  corruption, just literal 3x duplicate rows) — always pass `--ntasks=1` explicitly; also now set
  in the sbatch as a defense-in-depth guard.
- Two crash bugs found in the real 104-subject submission (both fixed): (1) runs producing 0 PSD
  windows (shorter than one window) crashed NWB writing (H5DataIO always requested >=1 chunk row,
  HDF5 rejects that against 0 data rows) — now skipped with a warning. (2) a crash in one subject
  used to take down the rest of its batch (BATCH_SIZE subjects run sequentially in one process) —
  each subject's processing is now wrapped in try/except so one failure logs and moves on;
  `run_info.json`'s absence for a subject is the marker that it still needs a (re)run.
- **Derivatives path moved** (2026-07-13): PSD NWB output moved from
  `derivatives/preprocessed/bipolar_fft/` to `derivatives/sisler/preprocessed/bipolar_fft/`
  (namespaced under `sisler/`, matching `derivatives/sisler/analysis/`'s existing convention).
  Existing output was migrated (`mv`/`rsync -a` merge for subjects touched by more than one
  submission attempt); `config.BIPOLAR_PSD_DERIV_ROOT` and the sbatch's `--psd-out-root` both
  point at the new location now.
- **As of 2026-07-13**: a full recompute of all 104 subjects' PSD (job `33957325`,
  `SKIP_VARIANCE=1` since the variance CSVs are untouched by this parameter change) is running
  under the new 2s/50%-overlap scheme on the `normal` partition (`--array=0-25%6`,
  `BATCH_SIZE=4`, `--cpus-per-task=8`/`--n-workers=8`). Check `qc_scripts.processing_status` or
  `sacct`/`squeue` for current progress when picking this up.
