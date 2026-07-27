# Featurization planning — pain score analysis

Starting point for thinking through how to go from the QC'd, bipolar-referenced,
PSD-decomposed signal (see `qc_scripts/CONTEXT.md` for full pipeline detail) to
features + labels for a pain-score model. This doc summarizes where the pipeline
currently stands, then lays out open decisions to work through.

## 1. Outlier rejection — current state

Raw-voltage QC (`qc_scripts/`), four independent per-channel detectors, all
2s-detection rolled up to a 60s excluded/not-excluded bin:

| Detector | What it catches | Metric | Current default threshold |
|---|---|---|---|
| Saturation | amplifier rail clipping | fraction of samples at/beyond session rail | `sat_frac_thresh=0` (any sample at rail) |
| Flatline | dead channel | per-window variance (V²) | `var_thresh=5e-13` (absolute floor); a per-channel-relative log-variance z-score mode (`logz3/4/5`) also exists but is **not yet validated/decided** |
| Square wave | digital/relay two-level artifact | bimodal-extremes fraction | `frac_thresh=0.9` |
| Gross artifact | high-variance bursts (unplug/replug) | per-60s-window variance z-score vs. session mean/std | `std_thresh=5` (sweeps at 4, 3 also exist) |

Per-type exclusions are combined (OR'd) into a named **mask**
(`masks/<label>/`, e.g. `baseline`) — this is a swappable, cheap re-derivable
layer (metric/threshold split: raw NWB is read once; thresholds/masks are
cheap CSV operations). Current best combined candidate:
`gross-std3_satmargin5_logz4` (gross=std3, saturation=marginfrac0.05,
flatline=logz4, square_wave=default).

**Known still-present artifacts** (not yet caught by any detector):
- transient high-amplitude spikes
- noisy/artifactual stretches likely from an unplugged electrode (distinct
  flatline-like quiet gap bounded by two artifact bursts — seen in
  `sub-248/FA6152DU`, `sub-207/EA189782`, `sub-198/SA3332TR`, all each
  subject's 2nd run starting at exactly 120 min into the session)
- super noisy channels (a channel's own high baseline variance can self-mask
  in the gross_artifact z-score, since it's baked into that channel's own
  `session_mean`/`session_std`)
- drifting events

Bipolar re-referencing (`preprocessing/bipolar_reref.py` +
`qc_scripts/build_bipolar_exclusions.py`) has its own mask-aware z-score
exclusion on bipolar-pair variance, but per the user it currently seems to
flag more non-artifactual events than artifactual ones — not trusted as a
primary filter yet.

**Design implication for this project**: because exclusion is just a mask
applied on top of stored continuous metrics, outlier correction can be revised
at any point without re-deriving the underlying data (raw voltage metrics,
bipolar variance, or PSD). Epoch-level (5-min pre-pain-score) outlier handling
can be layered on independently — see open questions below.

## 2. FFT / PSD parameters — current state

`preprocessing/run_pipeline_bipolar.py`, one read per raw NWB run, computes
(while the bipolar-referenced trace is transiently in memory):

- **Windowing**: single-level, no coarser outer window. Each PSD estimate is
  its own periodogram (`scipy.signal.spectrogram`, one segment, Hann window)
  over a `--window-sec` window, **default 2.0s**, stepped by `--overlap`
  (**default 0.5 → 1s hop**). Revised 2026-07-13 from an earlier 60s-outer-window
  design specifically to match PSD time resolution to the 2s variance-metric grid.
- **Frequency range**: 1–250 Hz.
- **Binning**: **50 log-spaced frequency bins** across that range — stored as
  raw log-bin values, not canonical bands, in the PSD output NWB
  (`derivatives/sisler/preprocessed/bipolar_fft/sub-XXX/ses-XXX/`).
- **Line noise**: each bin flagged `contains_line_noise` if it overlaps a
  60 Hz harmonic (60/120/180/240 Hz) ± guard band.
- **Canonical bands are NOT precomputed** — `preprocessing/bipolar_bands.py`
  aggregates the stored 50 log-bins into caller-specified bands
  (`config.CANONICAL_BANDS_HZ`, edges chosen to avoid line-noise harmonics) on
  demand, so band definitions can be retuned without re-reading raw NWB.
- Also computed in the same pass: per-2s bipolar-pair variance (independent of
  PSD windowing, always 2s non-overlapping) → feeds `build_bipolar_exclusions.py`.

Pairing/re-referencing scheme, electrode-to-region mapping, and the exact
contents of `config.CANONICAL_BANDS_HZ` should be pulled in here once
finalized — not yet captured in this doc.

## 3. Where things stand for this analysis

- Pain scores are administered at discrete timepoints; the plan is to featurize
  the **5 minutes prior to each pain score** as one epoch.
  `brain_region_pain_scores_group.ipynb` / `preliminary_pain_analysis_plots.ipynb`
  / `results/pain_score_regional_analysis` already have some prior exploratory
  work in this direction — worth reconciling with this plan rather than
  starting fully fresh.
- Plan to reserve a subset of subjects purely for exploration (plotting,
  feature/threshold tuning) before touching the subjects that will go into any
  confirmatory model — to avoid p-hacking / confound risk from re-tuning
  thresholds after seeing outcome-linked patterns.

## 4. Open questions / next steps to work through

### Epoch-level outlier handling
- Need an epoch-level rule (distinct from the continuous 60s mask) for
  deciding when a 5-minute pre-pain-score window is usable at all — e.g. what
  fraction of the window's 60s bins can be masked before dropping the epoch
  entirely, vs. averaging over just the surviving bins.
- Whether/how to handle a region average when only some electrodes in that
  region are masked for a given epoch (drop the region-epoch, or average over
  fewer channels).

### Pain score label construction
Several options on the table, not yet decided:
1. High vs. low, split by each subject's own mean/median (binary, subject-relative).
2. High vs. low vs. none, subject-relative mean/median.
3. Fixed bins across everyone (none=0, low=1-3, medium=4-6, high=7-10),
   not subject-relative.
- Motivating hypothesis: pain is likely non-linear (some prior literature
  support), which is the main argument for classification over regression —
  but the specific binning scheme changes both the modeling approach and the
  class-imbalance profile, and interacts with the subject-relative-vs-absolute
  choice (subject-relative binning changes meaning of "high" across subjects
  with very different reporting ranges/baselines).
- Worth considering ordinal-regression approaches as a middle ground between
  fully linear regression and unordered classification, given pain scores are
  ordered categories even if non-linear.

### Modeling approach
- Original plan: linear mixed-effects models, band power (averaged within
  region, within subject) as fixed-effect predictors, subject as random
  intercept, exploring random slopes per subject too. Primary goal is
  inference (group-level effects + how individual subjects deviate), not
  pure prediction.
- Open question raised: given the non-linearity hypothesis, whether to move
  away from continuous regression toward mixed-effects **classification**
  (e.g. multinomial/ordinal mixed models) instead, and whether that's
  reconcilable with the "understand subject deviation from group effect"
  goal (random slopes are less standard/well-supported outside GLMMs).
- Suggested to first do simple descriptive plotting before committing to any
  model class — but the 50 log-spaced PSD bins are hard to plot directly
  without aggregating to bands first.

### Feature set
- Starting point: average band power across all electrodes within a region,
  within subject, for a small set of a priori regions of interest, to keep the
  fixed-effect count small (deliberately avoiding a large feature sweep at the
  outset).
- Which canonical bands, which regions, and how channels map to regions still
  need to be pinned down and documented here.
- Longer-term (explicitly flagged as later, not now): once group-level effects
  are characterized, follow up on *why*/*how* individual subjects deviate from
  the group average — likely a second-stage analysis, not part of the initial
  model.

### Follow-ups from the pain_analysis/ bipolar PSD heatmap first pass
- Direct log-averaging (channel-to-region, epoch time-average, and
  pain-bin/subject averaging all average `log10(V^2/Hz)` directly) vs.
  linear-then-log (rest of the pipeline's convention, e.g.
  `preprocessing/bipolar_bands.py`'s `aggregate_to_bands`, chosen there to
  avoid Jensen's-inequality bias) — flagged to revisit/compare, not an
  oversight.
- Anode-only region assignment (`Desikan_Killiany_anode`) as a temporary
  stand-in for a future virtual-electrode-coordinate-based region lookup.
- `EPOCH_MAX_EXCLUDED_FRAC` (`pain_analysis/config.py`) as a real, tunable
  config knob now, not just a docs mention — default `0.5`, not yet validated
  against real exclusion rates for this cohort.
