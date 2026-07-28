"""
Feature-level QC thresholds — the std-based power-outlier detector (P2.1).

Same metric/threshold split as raw-voltage QC (`qc_params.py`): the expensive
pass stores continuous *metrics* once, thresholds turn those into *exclusions*
cheaply, and exclusions are OR'd into *masks*. So every value below can be swept
without recomputing anything that reads an NWB.

WHERE THIS DETECTOR SITS
------------------------
Raw-voltage QC works on the monopolar voltage trace and catches saturation,
flatline, square-wave, and gross-variance artifacts. It cannot see an artifact
that only shows up in the *spectrum* — a channel whose broadband power jumps
without its voltage clipping or its variance blowing past a session-relative
threshold. This detector works on the stored bipolar PSD (`preprocessed/
bipolar_fft`, log-power) and is the feature-level half of that pair.

THE RULE (2026-07-27)
---------------------
Per (channel, freq bin), a session-wide baseline mean/std of log-power. Then::

    z[window, channel, bin] = (log_power - mean[channel, bin]) / std[channel, bin]
    window is flagged  <=>  frac(z > FEATURE_Z_THRESH) > FEATURE_BIN_FRAC

i.e. a window is bad if MORE THAN 20% of its frequency bins sit above 5 SD of
that channel-bin's own session baseline.

Mean/std rather than the median/MAD that `architecture.md` PART 7 originally
specified. That was a deliberate call for consistency with the rest of the QC
tree (`GROSS_STD_THRESH`, flatline's relative mode) and with view axis 3's
z-scoring, all of which are mean/std. The cost is that mean/std is not robust:
a large artifact inflates the std and so partially hides itself. Two things
defuse that here, which is why the tradeoff was accepted:

  1. The baseline EXCLUDES windows already flagged by the pinned raw-voltage
     mask (see FEATURE_BASELINE_EXCLUDES_RAW_VOLTAGE), so the grossest
     contamination is gone before the mean is taken. This mirrors what
     `build_exclusions.py`'s flatline relative mode already does with
     --mask-from-label.
  2. Degenerate baselines (std == 0, or too few usable windows) are flagged
     rather than silently dividing by ~0 — see FEATURE_MIN_BASELINE_WINDOWS.

A median/MAD comparison on a subset of subjects is a follow-up, not a blocker:
the robust variant changes only the baseline table, and the per-window metric
can be recomputed from it without re-reading NWBs.

THE CASCADE
-----------
`architecture.md` PART 7 wrote this as four thresholds (K, X, Y, Z). The rule
above splits the first one in two, because flagging a *window* now requires both
a per-bin threshold and a how-many-bins threshold. So there are five levels:

  K  FEATURE_Z_THRESH      per (channel, window, bin): z > K
  B  FEATURE_BIN_FRAC      per (channel, window): > B of bins flagged  -> window bad
  X  (view-time)           per (channel, epoch): > X of windows flagged -> channel-epoch bad
  Y  (view-time)           per channel: > Y of epochs flagged -> drop channel everywhere
  Z  (view-time)           per epoch: > Z of surviving channels flagged -> drop epoch

K and B live here because they are computable from the continuous PSD alone.
X, Y, and Z are NOT here: they are defined over *epochs*, which only exist once
an epoch definition does, so they belong to the view layer and are set against
the epoch cache. Their starting values are deliberately unset — P2.1 sets them
on structural grounds (retained-data fractions, distribution shape, incremental
yield over the raw-voltage mask), and inventing numbers here would pre-empt that.
"""

# ============================================================================
# THE TWO THRESHOLDS THIS LEVEL OWNS
# ============================================================================

# K: how many SDs above a channel-bin's own session baseline counts as "high".
# One-sided (high only) by default: excess broadband power is the artifact this
# detector exists for, and a suspiciously QUIET channel is flatline's job over
# in raw-voltage QC. FEATURE_Z_SIDE flips it if the low tail is wanted too --
# free at threshold time, since only mean/std are stored.
FEATURE_Z_THRESH = 5.0
FEATURE_Z_SIDE = 'high'          # 'high' -> z > K;  'both' -> |z| > K

# B: what fraction of a window's usable frequency bins must exceed K before the
# whole (channel, window) is flagged.
#
# NOTE ON INTERPRETATION: adjacent log-spaced bins are strongly correlated and a
# broadband artifact trips many at once, so "20% of bins" is NOT 20% of
# independent tests. Expect this to behave closer to a near-binary broadband
# detector than the fraction suggests, i.e. B is a much less sensitive knob than
# K. That is a reason to sweep K first, not a reason to distrust B.
FEATURE_BIN_FRAC = 0.20

# ============================================================================
# HOW THE METRIC IS STORED (so K and B stay sweepable)
# ============================================================================
# The rule is a fraction-above-threshold test, and
#
#     frac(z > K) > B   <=>   sorted_desc(z)[floor(B * n)] > K
#
# so storing that ORDER STATISTIC per (channel, window) makes any K a free
# comparison later. Storing one statistic per B in this grid makes those B values
# free too. A B outside this grid needs the (NWB-reading) metric pass re-run --
# hence a grid rather than the single configured value.
#
# floor(B * n) is exact, not an interpolated quantile: with n usable bins,
# sorted_desc[floor(B*n)] > K implies floor(B*n)+1 bins exceed K, and
# (floor(B*n)+1)/n > B for every n. Verified in tests/test_feature_qc.py.
FEATURE_BIN_FRAC_GRID = (0.05, 0.10, 0.20, 0.50)

# Per-window rows are stored ONLY where the LARGEST order statistic in
# FEATURE_BIN_FRAC_GRID (i.e. the one for the smallest B) exceeds this floor. A
# dense per-(window, channel) table across the cohort would rival the epoch cache
# itself in size for no gain, and the floor sits 3 SD below FEATURE_Z_THRESH so K
# can still be swept well down without a re-run.
#
# NOT z_max, which was the first attempt: z_max is the maximum over ~44 bins, so
# it clears a 2 SD floor for most windows by construction (MEASURED: 32% of
# sub-039's channel-windows), which barely sparsifies anything. Flooring on the
# smallest-B statistic also guarantees NO CENSORING across the grid, since the
# statistics are monotone in B.
#
# THIS IS A REAL CAP AND IT IS NOT SILENT: the per-run/per-channel `summary`
# table carries n_windows / n_stored / n_nonfinite / n_rv_excluded, so every
# denominator survives, and the `zhist` table carries the full z distribution on
# a fixed grid so distribution SHAPE (the knee P2.1 is looking for) is not lost
# either. Sweeping K below this floor is the one thing that needs a re-run.
FEATURE_METRIC_STORE_FLOOR = 2.0

# Fixed grid for the per-(run, channel) z histogram, which is what makes
# threshold-setting-on-structural-grounds possible without a dense table:
# (min, max, n_bins) over the B-th order statistic of z.
FEATURE_ZHIST_RANGE = (-6.0, 14.0)
FEATURE_ZHIST_BINS = 100

# ============================================================================
# BASELINE CONSTRUCTION
# ============================================================================

# Exclude windows already flagged by the pinned raw-voltage mask from the
# baseline mean/std. This is the whole reason the metric is scoped by mask label
# on disk (see config.feature_qc_baseline_dir): the baseline is a function of
# WHICH mask you subtract first, so the label has to travel with the artifact.
#
# Precedent: build_exclusions.py's flatline relative mode does exactly this via
# --mask-from-label, for exactly the same reason (other types' known-bad windows
# would otherwise skew a channel's own baseline).
FEATURE_BASELINE_EXCLUDES_RAW_VOLTAGE = True

# A channel-bin needs at least this many usable (finite, un-masked) windows
# before its mean/std is trusted. Below it the baseline is marked degenerate and
# every window of that channel-bin is treated as flagged -- same convention as
# gross_artifact's "degenerate std -> excluded" (build_exclusions.py:236), so a
# channel with no usable baseline fails loudly into the cascade instead of
# silently producing NaN z-scores that comparisons quietly drop.
FEATURE_MIN_BASELINE_WINDOWS = 100

# Line-noise bins are excluded from BOTH the baseline and the bin-fraction
# denominator. They are contaminated by construction (the PSD writer flags
# +/- PSD_LINE_NOISE_GUARD_HZ around each 60 Hz harmonic), so letting them into
# an artifact detector would mean thresholding on known noise. Deliberately
# independent of whether the epoch cache stores those bins -- as of 2026-07-27
# it does store them, and filtering them is a view-time choice there.
FEATURE_EXCLUDE_LINE_NOISE_BINS = True

# ============================================================================
# LABELS
# ============================================================================

# The one artifact type this level defines so far. Parallel to
# qc_params.ARTIFACT_TYPES; a mask at this level ORs across these, which is why
# the rollup step exists even with a single entry (a `nonfinite` or a
# spectral-shape detector would slot in beside it).
FEATURE_ARTIFACT_TYPES = ['power_outlier']


def feature_exclusion_label(z_thresh=None, bin_frac=None, side=None):
    """A self-documenting exclusion label, e.g. 'z5_binfrac20'.

    Same convention as build_exclusions.label_for: read the thresholds off the
    path instead of an opaque 'default'.
    """
    z_thresh = FEATURE_Z_THRESH if z_thresh is None else z_thresh
    bin_frac = FEATURE_BIN_FRAC if bin_frac is None else bin_frac
    side = FEATURE_Z_SIDE if side is None else side
    label = f'z{z_thresh:g}_binfrac{bin_frac * 100:g}'
    if side != 'high':
        label += f'_{side}'
    return label
