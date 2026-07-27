# Pain iEEG — View Registry

> One of three companion docs: `architecture.md` (the model),
> `kickoff_plan.md` (the task plan), and this registry (the view axes).
> Cache format decision: **Parquet, one file per subject/session** (see architecture
> PART 1). GLMM features expected in **log-power** (log is the default domain).

The canonical list of VIEW axes: the cheap, deterministic transforms applied to
the per-window cache at load time to produce analysis-ready features. A "view
config" is one choice per axis. All are functions in the repo (views.py / common.py),
recomputed by default, with optional save_path (+ provenance/staleness sidecar).

Cache stores: per-2s-window, per-channel, per-freq-bin **log-power**, float32,
QC-masked (raw-voltage mask + feature-level mask applied/flagged). One columnar
file per subject/session (Parquet or HDF5 — NOT NWB, NOT one-file-per-epoch).

The view chain runs in this ORDER (order matters; some steps must precede others):

```
cache (per-window log-power, masked)
  1. domain            (log vs linear)            -- exponentiate here if linear wanted
  2. baseline estimate (which windows define baseline)
  3. normalization     (none / zscore / baseline-subtract)   -- per window, BEFORE averaging
  4. epoch aggregation (average over the 5-min window)       -- linear-then-log vs log-direct
  5. frequency agg     (50 log-bins -> canonical bands)      -- linear-then-log (Jensen)
  6. region aggregation(channels -> region)                  -- theory sets / DK / global
  7. pain binarization (how high/low/none defined)
  -> analysis-ready feature table
```

---

## AXIS 1 — Power domain
- `log` (DEFAULT; matches stored cache, Gaussianizes for stats)
- `linear` (exponentiate the stored log-power in-view)
Notes: choice of domain interacts with AXIS 4 (averaging). Store log; linear is
the exponentiate-in-view exception. Stats/GLMM default to log-power features.

## AXIS 2 — Baseline definition (what counts as the reference for normalization)
- `zero_pain_epochs` (DEFAULT; mean power over the subject's 0-pain epoch windows)
- `whole_session` (mean/SD over all of the subject's windows)
- (future) `pre_event_window` variants
Notes: a view-time computation over the chosen windows' per-window values.
Changing it is a free recompute. Baseline stat = per-channel-per-bin mean (+SD
if z-scoring).

## AXIS 3 — Normalization (applied PER WINDOW, before epoch-averaging)
- `none` (raw log-power)
- `zscore_vs_baseline` ((x - mu)/sigma using AXIS-2 baseline)
- `baseline_subtract` (x - mu using AXIS-2 baseline)
Notes: MUST precede AXIS 4. Linear normalizations (both above, in log-space)
commute with averaging IF baseline is a fixed per-channel scalar -- relevant to
whether per-window storage is strictly required (see architecture doc PART 1).

## AXIS 4 — Epoch aggregation (average over the 5-min window)
- `mean` in the current domain (DEFAULT)
- `linear_then_log` vs `log_direct` -- the Jensen choice. If AXIS 1 = linear,
  averaging then re-logging = linear_then_log; averaging stored log directly =
  log_direct (geometric-mean-like, robust, downweights power outliers).
Notes: this is the log-vs-linear averaging decision, made a VIEW precisely so
both are free recomputes from the per-window cache.

## AXIS 5 — Frequency aggregation
- `log_bins_50` (the stored 50 log-spaced bins, no aggregation)
- `canonical_bands` (delta 1-4, theta 4-8, alpha 8-12, beta 15-25, gamma 25-70,
  high_gamma 70-170) via bipolar_bands.aggregate_to_bands
Notes: bands use linear-then-log aggregation (existing convention, avoids Jensen).

## AXIS 6 — Region aggregation (channels -> region)
- `none` (per-channel)
- `theory_sensory` {S1, S2, posterior insula}
- `theory_affective` {ACC, anterior insula, OFC, amygdala}
- `individual_dk` (Desikan-Killiany regions, anode-based for now)
- `global` (all channels)
Notes: region-average uses linear-then-log if in log domain (Jensen again).
Report contributing channel/subject n (coverage confound).

## AXIS 7 — Pain binarization (the outcome definition)
- `absolute` (none=0, low=1-3, med=4-6, high=7-10; fixed across subjects)
- `subject_relative` (DEFAULT for within-subject; none=0, low/high at subject's
  own mean of nonzero events)
- `tertile_extremes` (high vs low tertile, drop middle)
- `graded` (keep continuous / ordinal -- no binarization)
Notes: subject_relative changes meaning of "high" across subjects with different
reporting ranges; include absolute as a contrast. Inclusion requires >4-point
pain range for within-subject analysis.

---

## Non-axis view-time computations (also recomputed, not stored)
- per-subject effect size (standardized mean diff high vs low, or correlation for graded)
- sign-consistency fraction across subjects
- 1/f slope via polyfit over the log-bins (a cheap derived quantity of the PSD cache)

## How a view config is recorded
In each analysis run's config.yaml, e.g.:
```yaml
view:
  domain: log
  baseline: zero_pain_epochs
  normalization: zscore_vs_baseline
  epoch_agg: log_direct
  freq: canonical_bands
  region: theory_affective
  pain_bins: subject_relative
```
Two runs differing in one line are two runs on the SAME cache -- not two caches,
not two feature folders.

## Sensitivity-analysis axes (planned; each a view unless it changes rows)
- epoch length 1/2/5/10 min -> NEW CACHE (changes windows), not a view
- everything else above -> view (free recompute)