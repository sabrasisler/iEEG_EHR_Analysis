"""
Raw-voltage QC detector thresholds.

Metric/threshold split: detectors write continuous *metrics* once (expensive),
thresholds turn those into *exclusions* cheaply, and exclusions are OR'd into
*masks*. So these values can be swept without recomputing anything expensive.
"""

# ============================================================================
# SUBJECT SUBSET (used when SUBJECT_LIST is None)
# ============================================================================

N_SUBJECTS = 20
SUBJECT_LIST = None    # e.g. ['217', '222'] to run an explicit set instead
RANDOM_SEED = 42

ARTIFACT_TYPES = ['saturation', 'flatline', 'square_wave', 'gross_artifact']

# ============================================================================
# STEP 1: AMPLIFIER SATURATION
# ============================================================================

SAT_WINDOW_SEC = 2.0

# Rail inference: a channel's saturation threshold is inferred from its own
# data, pooled across the ENTIRE SESSION (all runs), rather than assumed or
# computed per-run — amplifier gain/rail voltage varies across subjects but is
# physically shared by every channel/run within one session. Digital clipping
# shows up as the exact same extreme value repeating many times; a real
# (non-clipped) signal essentially never hits the identical floating-point
# value more than once or twice by chance.
SAT_MIN_REPEATS = 5             # session-wide occurrences of a channel's own abs_max
                                # needed to call it a rail, for per-channel inference
SAT_AGREEMENT_THRESHOLD = 0.25  # if this fraction of a session's channels independently
                                # agree on the same abs_max, use it as the rail for EVERY
                                # channel in the session (including ones that never
                                # saturate and so can't infer a rail on their own)
SAT_MIN_SAMPLES = 1             # samples at/beyond the rail needed to flag a window

# Last-resort override ONLY — used if infer_rail finds no repeated extreme
# (i.e. no evidence of clipping for that channel). NOT the primary detection
# path; kept for cases where a hard cutoff is wanted regardless.
SAT_FALLBACK_THRESHOLD_UV = None   # None = don't flag saturation for channels with no rail

# ============================================================================
# STEP 2: FLATLINED CHANNELS
# ============================================================================

FLATLINE_WINDOW_SEC = 2.0
FLATLINE_VAR_THRESH = 0.5e-12   # V^2

# ============================================================================
# STEP 2b: SQUARE-WAVE / TWO-LEVEL ARTIFACT
# ============================================================================
# A digital/relay-style artifact where nearly all samples in a window sit at
# two discrete levels (e.g. a 0-50uV square wave): flatline misses it (high
# variance), saturation misses it (not at the rail), gross_artifact misses it
# (mean-neutral). Metric = fraction of samples pinned within EPS_FRAC of the
# window's own min/max — dimensionless, so amplitude- AND frequency-independent
# (no per-case tuning). Only two shape knobs; the range guard is derived from
# FLATLINE_VAR_THRESH, not a free parameter.

SQUARE_WINDOW_SEC = 2.0         # shared 2s granularity with flatline/saturation
SQUARE_EPS_FRAC = 0.05          # band around each level, as a fraction of the window range
SQUARE_FRAC_THRESH = 0.9        # exclude if >= this fraction of samples sit at the two levels

# Range floor: below this peak-to-peak swing a window is effectively flat
# (flatline's job) and every sample is trivially "near" both extremes. Tied to
# the flatline threshold — the p2p range whose implied variance ==
# FLATLINE_VAR_THRESH.
SQUARE_MIN_RANGE_V = 2 * (FLATLINE_VAR_THRESH ** 0.5)   # ~1.41e-6 V (1.4uV)

# ============================================================================
# STEP 3: NON-NEURAL GROSS ARTIFACT
# ============================================================================
# Session-relative high-variance/amplitude bursts, e.g. unplug/replug — NOT
# DC-offset/drift.

GROSS_WINDOW_SEC = 60.0
GROSS_STD_THRESH = 5.0   # one-sided: only anomalously HIGH variance is excluded

# ============================================================================
# BIPOLAR VARIANCE
# ============================================================================

# WHY 2.0s: matches raw_voltage's SAT/FLATLINE window so the bipolar variance
# detector's bins are directly alignable against raw_voltage masks (60s bins,
# which are just 30x this) without resampling.
BIPOLAR_VARIANCE_WINDOW_SEC = 2.0

# Same one-sided high-variance convention as GROSS_STD_THRESH, applied
# post-bipolar-derivation instead of on raw monopolar channels.
BIPOLAR_VARIANCE_STD_THRESH = 5.0

# ============================================================================
# SUMMARY / REVIEW FLAGGING
# ============================================================================

FLAG_REVIEW_STD_THRESH = 3.0    # flag subject/channel if pct_windows_excluded is this
                                # many cross-subject stds above the mean, per artifact type
