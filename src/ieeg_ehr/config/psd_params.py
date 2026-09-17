"""
Bipolar re-reference + Welch PSD parameters, and the canonical frequency bands.

The PSD is the first stored FEATURE FAMILY (architecture.md PART 1): continuous,
per-window, expensive, written once per run to preprocessed/bipolar_fft/.
"""

# PSD params are given in SECONDS (resolved per-run against each run's own
# sfreq, not a fixed sample count) because sfreq varies across subjects in this
# dataset — mostly 1000/2000 Hz, occasionally 500 Hz.
#
# Single-level windowing (per lab discussion, superseding an earlier 60s
# outer-window design): each PSD_WINDOW_SEC window is its own periodogram-style
# estimate — no multi-segment Welch averaging within a coarser window — stepped
# by PSD_OVERLAP_FRAC. Default 2s window / 50% overlap gives a PSD estimate
# every 1s. Matches the variance metric's 2s granularity far more closely than
# the old 60s scheme, at the cost of a noisier per-window spectral estimate
# (accepted tradeoff for the much finer time resolution).

PSD_WINDOW_SEC = 2.0           # sets frequency resolution (sfreq/nperseg = 0.5 Hz)
                               # AND the time granularity of the PSD output
PSD_OVERLAP_FRAC = 0.5         # 50% overlap -> 1s hop for the default 2s window
PSD_WINDOW_FN = 'hann'
PSD_N_LOG_BINS = 50
PSD_FREQ_MIN_HZ = 1.0
PSD_FREQ_MAX_HZ = 250.0        # Nyquist-safe ceiling given rare 500 Hz-sampled subjects
                               # (Nyquist=250Hz there); can restrict further downstream,
                               # can't recover truncated data later

PSD_LINE_NOISE_FREQS_HZ = (60.0, 120.0, 180.0, 240.0)
PSD_LINE_NOISE_GUARD_HZ = 2.0  # +/- band around each harmonic flagged contains_line_noise

# ============================================================================
# FULL-RESOLUTION EPOCH PSD  —  features/pain/psd_epochs_fullres/
# ============================================================================
# The NATIVE FFT grid, stored instead of the 50 log bins above. Same window, same
# hop, same range; the only difference is that no reduction happens.
#
# WHY. The log-bin reduction is irreversible and wrong at both ends of the
# spectrum. At high frequency a bin is far wider than the line noise it must
# exclude (bins 36+37 span 53.3-66.4 Hz, so dropping 60 Hz costs 13.2 Hz where
# the contamination is ~4 Hz). Below ~4.7 Hz a bin is NARROWER than the window's
# own resolution, so `_band_average_linear`'s fallback fills it with a copy of a
# neighbour and the 44 usable bins carry only 38 distinct values. And ~2 samples
# per oscillatory peak is unfittable, which is what blocked specparam (BG.3).
# Storing the native grid makes every binning scheme a view. DECISIONS 2026-09-15.
#
# THE RESOLUTION IS NOT A FREE PARAMETER. df = sfreq/nperseg = 1/PSD_WINDOW_SEC,
# so 2s windows give 0.5 Hz at 500, 1000 AND 2000 Hz sampling — the grid is
# identical across the cohort without resampling anything, which is the whole
# reason a fixed frequency axis is possible at all. The constant below is an
# ASSERTION target, not an input: the extractor computes df from the run's own
# sfreq and refuses if it disagrees. Changing PSD_WINDOW_SEC changes this.
PSD_FULLRES_DF_HZ = 1.0 / PSD_WINDOW_SEC        # 0.5 Hz — asserted, never assumed
PSD_FULLRES_FREQ_MIN_HZ = 1.0
PSD_FULLRES_FREQ_MAX_HZ = 250.0                 # inclusive; same Nyquist-safe ceiling
                                                # as PSD_FREQ_MAX_HZ, and chosen to keep
                                                # CANONICAL_BANDS_HZ's high_gamma3 (185-235)
                                                # and SLOPE_FIT_HI_HZ (250) valid unchanged
# 1.0, 1.5, ... 250.0 inclusive. Derived here so a reader can size a table
# without opening a manifest, but the MANIFEST's freqs_hz is the source of truth
# for any stored artifact — see config.fullres_epoch_manifest_path.
PSD_FULLRES_N_FREQS = int(round((PSD_FULLRES_FREQ_MAX_HZ - PSD_FULLRES_FREQ_MIN_HZ)
                                / PSD_FULLRES_DF_HZ)) + 1        # 499

# The view-time notch half-width, applied to PSD_LINE_NOISE_FREQS_HZ. SEPARATE
# from PSD_LINE_NOISE_GUARD_HZ even though both are 2.0 Hz today, because they
# are different kinds of thing: the guard is baked into a stored artifact's
# contains_line_noise flags and cannot be changed without a rebuild, whereas this
# is a view parameter and is free to sweep. Collapsing them into one constant
# would make a sweepable choice look like a frozen one.
#
# 2.0 Hz is a structural floor, not a tuned value: a 2s Hann window's main lobe
# is 2/T = 1.0 Hz wide (+/- 0.5 Hz), and mains frequency itself drifts by a few
# tenths, so +/-2.0 Hz clears the lobe with margin. The selection is INCLUSIVE at
# both ends (|f - harmonic| <= half), so the cost is 2*half/df + 1 = 9 bins per
# harmonic — 58.0-62.0 Hz, 4.5 Hz of the axis — and 36 of 499 over the four
# harmonics, versus 6 bins of 50 on the log axis, where 60 Hz alone cost 13.2 Hz.
# (Read "8 bins of 499" here until 2026-09-16; that was an off-by-one per
# harmonic, not a different notch.)
PSD_NOTCH_HALF_WIDTH_HZ = 2.0

# HDF5 chunking: default is uncapped (whole run's time axis in one chunk per
# channel). PSD rows are spaced by the hop (~1s by default) — ~60x denser than
# the old 60s scheme, but a channel's entire run is still only single-digit MB
# even for long recordings (2hr run: ~1.4MB/channel; 24hr: ~17MB/channel),
# comfortably one chunk. This differs from raw-voltage chunking (dense samples),
# which DOES need small time-chunks. Only set a cap for unusually long recordings.
PSD_HDF5_CHUNK_MAX_HOURS = None   # e.g. 4.0

# ============================================================================
# APERIODIC (1/f) SLOPE  — BG.2, docs/view_registry.md "non-axis computations"
# ============================================================================
# A straight line through log10(power) vs log10(frequency), fit per channel per
# epoch by views/aperiodic.py. Not a stored feature family: it is a cheap derived
# quantity of the PSD cache (architecture.md PART 1 table).
#
# The DEFAULT RANGE IS THE WHOLE STORED SPECTRUM, 1–250 Hz, minus the six bins
# flagged contains_line_noise. That is the maximum-leverage choice and it is a
# deliberate one, not an oversight: it buys 44 of 50 bins spanning 2.4 decades,
# at the cost of folding the low-frequency knee, the alpha/beta peaks and any
# high-frequency amplifier noise floor into a single number. So the exponent
# here is a BROADBAND TILT, not a knee-free aperiodic exponent in the
# specparam sense — BG.3 (FOOOF) is what separates those. Both edges are CLI
# flags precisely so a narrower range is one run away, and both are hashed into
# the view's directory so two ranges cannot land in the same folder.
SLOPE_FIT_LO_HZ = 1.0
SLOPE_FIT_HI_HZ = 250.0

# A fit needs enough surviving bins, spread over enough frequency, before its
# slope means anything. Below either floor the channel-epoch gets NaN rather than
# a number — a slope from one corner of the spectrum is not the spectrum's slope,
# and it is the one that most easily reaches an absurd value.
#
# 30 of the 44 available bins (~68%) and 1.0 decade of the 2.37 available. Both
# are STRUCTURAL choices made before looking at any pain relationship, which is
# the same rule the feature-level QC thresholds are held to (CLAUDE.md).
SLOPE_MIN_FIT_BINS = 30
SLOPE_MIN_SPAN_DECADES = 1.0

# r² is STORED PER FIT AND NOT THRESHOLDED here. Metric/threshold split: compute
# the metric once in the expensive pass, apply a cutoff downstream where it costs
# nothing and can be varied. `--min-r2` on the plot script is where a cutoff goes.

# ============================================================================
# CANONICAL BANDS
# ============================================================================
# NOT precomputed — ieeg_ehr/preprocessing/bipolar_bands.py aggregates the stored 50 log
# bins into these on demand, linear-then-log to avoid Jensen bias.
#
# Edges fall strictly BETWEEN 60 Hz line-noise harmonics by construction, so no
# canonical band straddles a notch. That is why gamma is split finely rather
# than being one wide band.
#
# DISCREPANCY (flagged 2026-07-27, unresolved): docs/architecture.md states
# beta 15-25 / gamma 25-70 / high_gamma 70-170, which is NOT this. Band choice
# is a P2.2 sweep axis, so resolve before that sweep.
CANONICAL_BANDS_HZ = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (13, 30),
    'low_gamma': (30, 58), 'high_gamma1': (65, 115), 'high_gamma2': (125, 175),
    'high_gamma3': (185, 235),
}

# ----------------------------------------------------------------------------
# THE DECODING REPLICATION'S BANDS  (Huang et al. 2025, doi 10/nat.s41467-025-59756-5)
# ----------------------------------------------------------------------------
# The six bands the per-subject pain decoder replicates. A SEPARATE constant
# rather than a replacement for CANONICAL_BANDS_HZ, because the two answer
# different questions and neither supersedes the other: these are the target
# paper's edges, exactly as published, and CANONICAL_BANDS_HZ is this project's
# own revision of them. Both are selectable on the view's frequency axis, so the
# DISCREPANCY noted above becomes an empirical comparison instead of a conflict.
#
# TWO PROPERTIES OF THESE EDGES THAT MATTER, and are the paper's, not ours:
#
# 1. `gamma` and `high_gamma` STRADDLE the 60/120 Hz line-noise harmonics, which
#    is precisely what CANONICAL_BANDS_HZ's finer gamma split exists to avoid. So
#    a view using these bands MUST also set drop_line_noise_bins=True. Unlike
#    preprocessing.bipolar_bands.aggregate_to_bands, views.axes.aggregate_bands
#    does NOT exclude flagged bins itself -- it relies on them already being NaN,
#    and nanmean skipping them. Without the flag, gamma silently absorbs the
#    58-62 Hz notch residue and high_gamma the 118-122 Hz residue.
#
# 2. There is a GAP at 12-15 Hz: alpha ends at 12 and beta starts at 15. That is
#    how the paper defines them, so it is reproduced rather than closed. Nothing
#    above 170 Hz is covered either, though the stored spectrum reaches 250 Hz.
#
# Low-frequency caveat (ours, not theirs): below ~4.7 Hz a stored log bin is
# narrower than the 2 s window's 0.5 Hz resolution, so bins {1,2,4,5,7,10} are
# exact DUPLICATES of a neighbour (views.cache_reader.unresolvable_bins). Those
# all fall in delta and theta, which are therefore built from fewer independent
# measurements than their bin counts suggest -- relevant if either turns up as an
# important decoder feature.
PAPER_BANDS_6_HZ = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12),
    'beta': (15, 25), 'gamma': (25, 70), 'high_gamma': (70, 170),
}

# ----------------------------------------------------------------------------
# THE SAME SIX BANDS WITH high_gamma EXTENDED TO 200 Hz  (2026-09-17)
# ----------------------------------------------------------------------------
# A SEPARATE constant, never an edit of PAPER_BANDS_6_HZ, because that dict is
# the published set and the decoding replication's claim to be a replication
# rests on its edges being exactly the paper's. Changing it in place would
# silently redefine a completed analysis.
#
# WHY EXTEND IT. 170 Hz was the paper's ceiling, not this dataset's: the stored
# spectrum reaches 250 Hz, and the native-resolution map shows a pain-related
# high-frequency increase in Thalamus running from ~124 Hz to the top of the
# range — structure that 70-170 truncates and therefore dilutes with a wide band
# of nothing. 200 Hz keeps a margin below the 250 Hz ceiling, where the rare
# 500 Hz-sampled subject sits at Nyquist.
#
# 70-200 Hz CONTAINS TWO HARMONICS (120 and 180 Hz), where 70-170 contained one.
# Both are removed by the view-time notch before aggregation — mandatory here,
# not optional, exactly as for the parent set. Derived from the parent so an edit
# to those five shared edges cannot leave this variant behind.
PAPER_BANDS_6_HG200_HZ = {**PAPER_BANDS_6_HZ, 'high_gamma': (70, 200)}
