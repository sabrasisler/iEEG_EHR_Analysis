"""
Tests for the roi_v2 ROI scheme and for reading a region order OFF THE VIEW.

The scheme is data, so what needs testing is not arithmetic but the two ways it
can be silently wrong: a precedence collision between substring patterns, and a
figure that filters a view's regions against the DEFAULT scheme's list.
"""

import pytest

from ieeg_ehr.analysis import view_tables
from ieeg_ehr.config import roi_schemes

V2 = 'roi_v2'


def test_roi_v2_has_21_display_regions_all_with_patterns():
    scheme = roi_schemes.resolve_roi_scheme(V2)
    assert len(scheme['display']) == 21
    for region in scheme['display']:
        assert scheme['patterns'][region], region


def test_frontopolar_and_cerebellum_are_absent():
    """Measured 2026-07-29: frontalpole had 2 subjects and cerebellum 0 of 60, so
    both would only ever be a blank or untestable row."""
    display = roi_schemes.roi_regions(V2)
    assert 'Frontopolar' not in display
    assert 'Cerebellum' not in display
    assert roi_schemes.region_for_dk_label('ctx-lh-frontalpole', V2) is None
    assert roi_schemes.region_for_dk_label('Left-Cerebellum-Cortex', V2) is None


def test_precuneus_is_parietal_not_occipital():
    """THE precedence trap: 'precuneus' CONTAINS 'cuneus'. If Occipital were
    ordered first, every precuneus contact would be relabelled occipital."""
    assert roi_schemes.region_for_dk_label('ctx-lh-precuneus', V2) == 'Parietal (other)'
    assert roi_schemes.region_for_dk_label('ctx-lh-cuneus', V2) == 'Occipital'


def test_parahippocampal_is_mtl_not_hippocampus():
    assert roi_schemes.region_for_dk_label('ctx-lh-parahippocampal', V2) == 'MTL (other)'
    assert roi_schemes.region_for_dk_label('Left-Hippocampus', V2) == 'Hippocampus'


def test_temporalpole_is_mtl_not_lateral_temporal():
    assert roi_schemes.region_for_dk_label('ctx-rh-temporalpole', V2) == 'MTL (other)'
    assert (roi_schemes.region_for_dk_label('ctx-rh-superiortemporal', V2)
            == 'Lateral Temporal')


def test_precentral_and_postcentral_do_not_collide():
    assert roi_schemes.region_for_dk_label('ctx-lh-precentral', V2) == 'M1'
    assert roi_schemes.region_for_dk_label('ctx-lh-postcentral', V2) == 'S1'
    assert roi_schemes.region_for_dk_label('ctx-lh-paracentral', V2) == 'S1'


def test_acc_and_ofc_are_split():
    assert roi_schemes.region_for_dk_label('ctx-lh-rostralanteriorcingulate', V2) == 'rACC'
    assert roi_schemes.region_for_dk_label('ctx-lh-caudalanteriorcingulate', V2) == 'dACC'
    assert roi_schemes.region_for_dk_label('ctx-lh-medialorbitofrontal', V2) == 'mOFC'
    assert roi_schemes.region_for_dk_label('ctx-lh-lateralorbitofrontal', V2) == 'lOFC'


def test_non_neural_tissue_is_excluded_before_anatomy():
    for label in ('Left-Cerebral-White-Matter', 'wm-lh-insula',
                  'Left-Lateral-Ventricle', 'unknown'):
        assert roi_schemes.region_for_dk_label(label, V2) is None, label


def test_default_scheme_is_untouched():
    """roi_v2 is ADDITIVE. If default changed, every existing run's provenance would
    describe a region set that no longer exists."""
    default = roi_schemes.roi_regions('default')
    assert len(default) == 15
    assert default[0] == 'Hippocampus' and 'S2 (supramarginal)' in default
    assert roi_schemes.region_for_dk_label('ctx-lh-precuneus', 'default') == 'Parietal'


# ---------------------------------------------------------------------------
# The region order must come from the VIEW, not from config.ROI_REGIONS
# ---------------------------------------------------------------------------

def test_roi_regions_for_reads_the_views_own_scheme():
    assert len(view_tables.roi_regions_for({'roi_scheme': V2})) == 21
    assert len(view_tables.roi_regions_for({'roi_scheme': 'default'})) == 15
    assert len(view_tables.roi_regions_for({})) == 15          # absent -> default


def test_regions_are_not_silently_filtered_against_the_default_15():
    """THE regression this exists to catch. Under roi_v2, 13 of the 21 region names
    are absent from the default list; filtering against config.ROI_REGIONS would
    return 8 rows and a figure that looks completely normal."""
    import pandas as pd
    regions = roi_schemes.roi_regions(V2)
    stats = pd.DataFrame([
        {'pain_bin': b, 'region': r, 'freq_bin_index': f, 'n_subjects': 30}
        for r in regions for b in ('low', 'high') for f in range(3)
    ])
    kept, counts = view_tables.regions_with_min_subjects(
        stats, ['low', 'high'], min_subjects=8, regions=regions)
    assert kept == regions, 'a roi_v2 region was dropped'
    assert len(kept) == 21

    from ieeg_ehr import config
    default_kept, _ = view_tables.regions_with_min_subjects(
        stats, ['low', 'high'], min_subjects=8, regions=config.ROI_REGIONS)
    assert len(default_kept) < 21, 'the bug this guards against should be reproducible'


def test_scheme_code_puts_the_roi_scheme_in_the_path():
    from ieeg_ehr.views.view_config import ViewConfig
    base = dict(mask_label='m', pain_bins='subject_relative')
    assert ViewConfig(normalization='zscore_vs_baseline', roi_scheme='default',
                      **base).scheme_code == 'zscore-relpain'
    assert ViewConfig(normalization='zscore_vs_baseline', roi_scheme=V2,
                      **base).scheme_code == 'zscore-relpain-roiv2'
    assert ViewConfig(normalization='none', roi_scheme=V2,
                      **base).scheme_code == 'raw-relpain-roiv2'


def test_unnormalized_view_is_not_a_difference():
    """Drives whether 'none' is a line, whether y=0 is drawn, and whether the
    cluster test is allowed to run at all."""
    from ieeg_ehr.views.view_config import ViewConfig
    base = dict(mask_label='m', pain_bins='subject_relative', roi_scheme=V2)
    assert not ViewConfig(normalization='none', **base).is_difference
    assert ViewConfig(normalization='baseline_subtract', **base).is_difference
    assert ViewConfig(normalization='none', **base).value_label.startswith('Mean log10')
