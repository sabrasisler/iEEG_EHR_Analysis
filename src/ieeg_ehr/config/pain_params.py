"""
Pain-event epoch definition, outcome binning, and channel->ROI mapping.

Epoch/bin choices here are the *stored* ones; the alternative schemes the view
layer recomputes at load time are enumerated in docs/view_registry.md.
"""

import pandas as pd

from ieeg_ehr.config.psd_params import CANONICAL_BANDS_HZ  # noqa: F401  re-exported

# ============================================================================
# EPOCH DEFINITION
# ============================================================================

EPOCH_MINUTES_BEFORE = 5.0

# Fixed, absolute, everyone-shares-the-same-cutpoints pain bins (matches the
# planned ordinal-model label scheme (docs/architecture.md, Open decision 7) — NOT
# subject-relative, so a region's fixed-effect relationship to pain level
# stays interpretable independent of the subject random effect).
PAIN_BIN_EDGES = {
    'none': (0, 0),
    'low': (1, 3),
    'medium': (4, 6),
    'high': (7, 10),
}
PAIN_BIN_ORDER = ['none', 'low', 'medium', 'high']


def pain_bin_for_score(score):
    """None if score is NaN or outside all bins (shouldn't happen for a valid 0-10 score)."""
    if pd.isna(score):
        return None
    for name, (lo, hi) in PAIN_BIN_EDGES.items():
        if lo <= score <= hi:
            return name
    return None


# Alternative, subject-relative scheme: 'none' still means score == 0, but
# 'low' vs. 'high' splits at that SAME subject's own mean pain score among
# their non-zero events (score == 0 excluded from the mean) -- no 'medium'
# bin. Unlike PAIN_BIN_EDGES above, this does NOT keep the region/pain-level
# relationship independent of the subject random effect (that was the whole
# reason for the fixed scheme) -- it's an alternative lens, not a
# replacement, per user instruction. Computed at plot time from the cache's
# raw `pain_score` column, not baked into the cache CSVs (no need to re-run
# build_pain_epoch_power.py to switch schemes).
PAIN_BIN_ORDER_SUBJECT_RELATIVE = ['none', 'low', 'high']


def pain_bin_order(scheme):
    return PAIN_BIN_ORDER_SUBJECT_RELATIVE if scheme == 'subject_relative' else PAIN_BIN_ORDER


# One colour per pain level, for every figure that draws pain bins as lines or
# marks rather than as a value on a colour scale.
#
# DISTINCT HUES on a cool -> warm progression, so the palette does two jobs at
# once: the hues are far enough apart to tell apart at a glance (which a
# light-to-dark ramp in ONE hue is bad at, and which was the reason this replaced
# such a ramp on 2026-07-29), while cool->warm still reads as ORDERED, which pain
# level is. Lines are drawn SOLID -- colour and the legend carry identity.
#
# MEASURED, not eyeballed (dataviz validate_palette.js, light surface #fcfcfb).
# All three slots, i.e. the `absolute` scheme:
#   chroma floor        PASS  all >= 0.1
#   CVD separation      PASS  worst adjacent dE 12.7 protan / 22.1 tritan (>= 8)
#   normal-vision floor PASS  worst adjacent dE 24.9 (>= 15)
#   lightness band      PASS  all inside L 0.43-0.77
#   contrast vs surface WARN  orange 2.77:1, below 3:1
# The two slots `subject_relative` actually draws (teal vs crimson) pass every
# check including contrast. Re-run before changing any value:
#   node scripts/validate_palette.js "#0d9488,#e08214,#a11d33" --mode light
#
# THE ORANGE CONTRAST WARN IS ACCEPTED, NOT IGNORED. That check obligates relief
# -- a visible label or a table view -- and both exist: every figure carries a
# legend, and the spectra runs write spectra_table.parquet beside the PNG. It also
# only applies to `absolute`, which is not the default scheme.
#
# WHY NOT BLUE -> RED, which validates better (2-slot CVD dE 21.1 vs 12.7): those
# are exactly RdBu_r's two poles, and the companion region x frequency heatmaps
# use RdBu_r for the SIGN of the change. On one page the same two colours would
# mean two unrelated things. Teal is not RdBu_r's blue; crimson does sit near its
# red, so the collision is reduced rather than eliminated.
#
# WHY NOT GREEN -> RED, which was tested rather than assumed away: it fails
# deuteranopia outright, adjacent dE 2.5 against a >= 8 target -- and the pair it
# fails on is exactly the two-line subject_relative view, the default.
#
# 'none' is grey: under any baseline normalization it IS the reference and sits
# at 0 by construction, so it is drawn as a reference line if at all, never as a
# peer of the other levels.
PAIN_BIN_COLORS = {
    'none': '#9e9e9e',      # grey
    'low': '#0d9488',       # teal
    'medium': '#e08214',    # orange
    'high': '#a11d33',      # crimson
}


# Drop a channel's epoch entirely if more than this fraction of its epoch time
# (post-mask) is excluded; otherwise average over whatever time survives.
# Proposed default, not yet validated against real exclusion rates for this
# cohort -- tune after looking at how many channel-epochs it drops.
EPOCH_MAX_EXCLUDED_FRAC = 0.5

# ============================================================================
# REGION GROUPING
# ============================================================================
# The Desikan-Killiany -> ROI mapping now lives in config/roi_schemes.py as an
# ORDERED DICT of category -> substring patterns, replacing the if/elif ladder
# that used to sit here. Reason: the ladder's branch order was load-bearing
# (specific parcels before catch-alls) and ROI_REGIONS was maintained separately,
# so changing one ROI meant two edits plus reasoning about control flow. With a
# dict, insertion order IS precedence, and a scheme can be swapped by name or
# loaded from a JSON file on Oak without a commit.
#
# Behaviour is unchanged, and that is TESTED, not asserted: tests/
# test_roi_schemes.py holds a verbatim copy of the original chain and checks both
# agree on every DK label in the cohort.
#
# These names are re-exported so every existing `config.ROI_REGIONS` /
# `config.region_for_dk_label(...)` call site keeps working untouched. Both
# functions now take an optional `scheme=`.

from ieeg_ehr.config.roi_schemes import (   # noqa: F401  (re-exported)
    DEFAULT_ROI_SCHEME,
    NON_ROI_CATEGORIES,
    ROI_SCHEMES,
    categorize_desikan_killiany,
    region_for_dk_label,
    resolve_roi_scheme,
    roi_regions,
    scheme_provenance,
)

# The default scheme's display rows, as a plain list -- kept as a module constant
# because dozens of call sites read `config.ROI_REGIONS` directly.
ROI_REGIONS = roi_regions()


# ============================================================================
# Z-SCORE (plot_pain_zscore_heatmaps.py)
# ============================================================================

# Minimum number of a subject's own 'none'-bin epochs required, for a given
# region/freq_bin, before trusting that baseline's std enough to z-score
# against it -- too few baseline epochs gives a noisy/exploding SD estimate.
# Not validated against this cohort's real none-bin counts yet; tune after
# looking at how many (subject, region, freq_bin) cells it excludes.
ZSCORE_MIN_BASELINE_EPOCHS = 5

# ============================================================================
# CANONICAL FREQUENCY BANDS (real aggregation, e.g. plot_band_violin.py)
# ============================================================================

# Single source of truth is ieeg_ehr.qc.config -- same band edges used
# elsewhere in the pipeline (ieeg_ehr/preprocessing/bipolar_bands.py's
# aggregate_to_bands), re-exported here (see the import at the top of this
# module) so pain code reaches it through `ieeg_ehr.config` like everything
# else, without duplicating the dict.

# Coarser grouping used only by plot_band_violin.py: merges CANONICAL_BANDS_HZ's
# 'low_gamma' into 'gamma' (same range, renamed) and merges 'high_gamma1/2/3'
# into a single 'high_gamma' spanning their combined range -- per user
# instruction, fewer/simpler gamma categories for the violin grid instead of
# 4 separate gamma sub-bands. Distinct from CANONICAL_BANDS_HZ (used for
# actual pipeline-wide band aggregation elsewhere, split finer specifically
# to dodge individual line-noise harmonics one at a time) -- that finer
# splitting is moot here since line-noise bins are already absent from the
# cache regardless of how the surrounding band is drawn, so merging the
# three high-gamma ranges into one is safe.
VIOLIN_BANDS_HZ = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (13, 30),
    'gamma': (30, 58),
    'high_gamma': (65, 235),
}

# ============================================================================
# FREQUENCY BAND REFERENCE LINES (plotting only, not used for aggregation)
# ============================================================================

# Approximate low edge (Hz) of each classically-named EEG band, for vertical
# reference lines on the heatmap/scatter plots -- "vaguely where the bands
# are", not a precise redefinition of config.CANONICAL_BANDS_HZ (which is
# used for actual band-power aggregation elsewhere in the pipeline and is
# split more finely, e.g. three high-gamma sub-bands to dodge line-noise
# harmonics). Delta's low edge is PSD_FREQ_MIN_HZ itself, so no line is drawn
# for it.
FREQ_BAND_BOUNDARIES_HZ = {
    'theta': 4,
    'alpha': 8,
    'beta': 12,
    'gamma': 30,
    'high gamma': 65,
}
