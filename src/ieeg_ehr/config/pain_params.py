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


# Drop a channel's epoch entirely if more than this fraction of its epoch time
# (post-mask) is excluded; otherwise average over whatever time survives.
# Proposed default, not yet validated against real exclusion rates for this
# cohort -- tune after looking at how many channel-epochs it drops.
EPOCH_MAX_EXCLUDED_FRAC = 0.5

# ============================================================================
# REGION GROUPING
# ============================================================================

# Fine-grained, pain-circuitry-motivated Desikan-Killiany parcel -> ROI
# category mapping (per user instruction, replacing the earlier coarse-lobe
# placeholder). Order matters -- more specific checks (e.g. medialorbitofrontal
# -> vmPFC) must come before general catch-alls (e.g. 'frontal' -> Frontal
# (other)). Categories not in ROI_CATEGORIES below (Exclude, White Matter,
# CSF/Ventricles, Occipital, Cerebellum, Other, Unlabeled) are dropped from
# the region-level plots -- not a final decision on what to do with them,
# just where things start per user instruction.
def categorize_desikan_killiany(dk_label):
    if pd.isna(dk_label):
        return 'Unlabeled'
    dk_label = str(dk_label).lower().strip("'\"")

    # -- Exclude / artifacts -------------------------------------------------
    if any(x in dk_label for x in ['empty', 'unknown', 'undefined']):
        return 'Exclude'

    # -- Non-neural tissue ----------------------------------------------------
    if any(x in dk_label for x in ['white-matter', 'ventraldc', 'cc_', 'wm-']):
        return 'White Matter'
    if any(x in dk_label for x in ['ventricle', 'csf', 'choroid-plexus', 'hypointensities']):
        return 'CSF/Ventricles'

    # -- Subcortical ------------------------------------------------------------
    if 'hippocampus' in dk_label:
        return 'Hippocampus'
    if 'amygdala' in dk_label:
        return 'Amygdala'
    if 'thalamus' in dk_label:
        return 'Thalamus'

    # Basal Ganglia (grouped)
    if any(x in dk_label for x in ['caudate', 'putamen', 'pallidum', 'accumbens']):
        return 'Basal Ganglia'

    if 'thalamus' in dk_label:
        return 'Thalamus'

    # -- Insula -----------------------------------------------------------------
    if 'insula' in dk_label:
        return 'Insula'

    # -- Cingulate ----------------------------------------------------------------
    if any(x in dk_label for x in ['caudalanteriorcingulate', 'rostralanteriorcingulate']):
        return 'ACC'
    if any(x in dk_label for x in ['posteriorcingulate', 'isthmuscingulate']):
        return 'PCC'

    # -- Prefrontal -- specific regions BEFORE general frontal catch-all --------
    # vmPFC: medial OFC + frontal pole
    if any(x in dk_label for x in ['medialorbitofrontal', 'frontalpole']):
        return 'vmPFC'

    # OFC: lateral orbital frontal
    if 'lateralorbitofrontal' in dk_label:
        return 'OFC'

    # dlPFC: middle frontal gyrus (BA9/46)
    if any(x in dk_label for x in ['rostralmiddlefrontal', 'caudalmiddlefrontal']):
        return 'dlPFC'

    # -- Somatosensory ------------------------------------------------------------
    # S1: primary somatosensory (postcentral gyrus)
    if 'postcentral' in dk_label:
        return 'S1'

    # S2: parietal operculum -- not a named DK label, captured under Parietal
    # below (true S2 sits in the parietal operculum which DK doesn't isolate
    # cleanly; flag here as a reminder -- supramarginal is the closest proxy)
    if 'supramarginal' in dk_label:
        return 'S2 (supramarginal)'

    # -- Remaining frontal (SFG, etc.) --------------------------------------------
    if any(x in dk_label for x in ['frontal', 'frontalpole', 'precentral', 'paracentral',
                                   'parsopercularis', 'parsorbitalis', 'parstriangularis']):
        return 'Frontal (other)'

    # -- Temporal -----------------------------------------------------------------
    if any(x in dk_label for x in [
        'temporal', 'fusiform', 'entorhinal',
        'parahippocampal', 'bankssts', 'transversetemporal', 'temporalpole',
    ]):
        return 'Temporal'

    # -- Parietal -----------------------------------------------------------------
    if any(x in dk_label for x in ['parietal', 'precuneus']):
        return 'Parietal'

    # -- Occipital ----------------------------------------------------------------
    if any(x in dk_label for x in ['occipital', 'cuneus', 'pericalcarine', 'lingual']):
        return 'Occipital'

    # -- Cerebellum ---------------------------------------------------------------
    if 'cerebellum' in dk_label:
        return 'Cerebellum'

    return 'Other'


# Categories actually shown as rows in the region x freq-bin heatmaps, in
# plot order. Everything else categorize_desikan_killiany() can return
# (Exclude, White Matter, CSF/Ventricles, Occipital, Cerebellum, Other,
# Unlabeled) is dropped -- not a final decision on what to do with those,
# just where this analysis starts per user instruction.
ROI_REGIONS = [
    'Hippocampus', 'Amygdala', 'Thalamus', 'Basal Ganglia', 'Insula', 'ACC', 'PCC',
    'vmPFC', 'OFC', 'dlPFC', 'S1', 'S2 (supramarginal)', 'Frontal (other)',
    'Temporal', 'Parietal',
]


def region_for_dk_label(dk_label):
    """Map one Desikan_Killiany_anode string to an ROI_REGIONS category, or
    None if it falls outside the current ROI set (e.g. occipital, white
    matter, unknown) -- callers must drop None and log how many channels
    were dropped, not silently exclude."""
    category = categorize_desikan_killiany(dk_label)
    return category if category in ROI_REGIONS else None


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
