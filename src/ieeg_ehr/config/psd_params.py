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
