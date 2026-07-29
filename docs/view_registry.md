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

**Precision rule for every view below (P0.6): store narrow, compute wide.** The
cache is float32; views UPCAST to `config.CACHE_ACCUMULATE_DTYPE` (float64)
before any reduction. This bites two axes specifically:

- **Axis 1 (domain)** — exponentiating to linear runs in float64. The worst
  stored log-power seen is ~-36.8, barely a decade above float32's smallest
  normal, so `10**log_power` in float32 is a step away from underflowing to
  exactly zero once anything divides it.
- **Axis 4 (epoch aggregation)** — a float32 accumulator over ~300 windows holds
  only ~6 sig figs, the largest precision loss in the whole chain. numpy does NOT
  upcast for you: `arr.mean(axis=0)` on float32 input accumulates in float32.

Axes 5 and 6 (frequency and region aggregation) are reductions too, so the same
rule applies. See DECISIONS 2026-07-27.

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

"Anode-based for now" is a deliberate TEMPORARY stand-in: a bipolar pair is
assigned the DK parcel of its anode (`Desikan_Killiany_anode`), which is wrong
whenever the two contacts of a pair straddle a boundary. The intended
replacement is a lookup on the pair's virtual electrode coordinate (the midpoint
between contacts), not the anode. Until then, treat region assignment near
parcel boundaries as approximate.

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

## scheme_code — the two axes that appear in a FOLDER NAME

`ViewConfig.scheme_code` (`views/view_config.py`) composes a short human label
from AXIS 3 and AXIS 7 only, e.g. `delta-relpain`. It is the level-4
`<view_scheme>` folder in the analysis tree AND the human half of a materialized
view's directory name, from ONE definition, so the two cannot drift:

```
features/pain/psd_epochs/epoch-5min-pre/views/delta-relpain_2735c1062131/
analysis/pain/psd_physiology/region_spectrum/delta-relpain/<run>_<timestamp>/
```

| Code | Axis | Means |
|---|---|---|
| `delta` | AXIS 3 | `baseline_subtract` -- per-window minus the channel's 0-pain mean. In the log domain this IS delta log power |
| `zscore` | AXIS 3 | `zscore_vs_baseline` -- as above, then divided by the 0-pain SD |
| `raw` | AXIS 3 | `normalization: none` -- no baseline applied |
| `relpain` | AXIS 7 | `subject_relative` -- low/high split at the subject's own mean of nonzero events |
| `abspain` | AXIS 7 | `absolute` -- fixed 0 / 1-3 / 4-6 / 7-10 across subjects |

**Only these two axes**, because a folder name spelling out all seven would be
unreadable and *still* not a complete description. The complete description is
the sidecar's `config_hash` -- the trailing hex above -- which covers every axis
plus the mask, the ROI scheme and the cohort split. Read `provenance.json` for
the full view, never the folder name.

AXIS 3 is in the name and AXIS 7 is not enough on its own: the level-4 folder was
briefly `subject_relative` alone, which left the normalization -- the axis that
most changes the numbers -- invisible in the path.

## Sensitivity-analysis axes (planned; each a view unless it changes rows)
- epoch length 1/2/5/10 min -> NEW CACHE (changes windows), not a view
- everything else above -> view (free recompute)