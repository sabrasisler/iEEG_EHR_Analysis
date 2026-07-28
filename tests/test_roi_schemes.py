"""
Behaviour-preservation proof for the ROI-mapping refactor (P1.3 Step 3b).

`_ORIGINAL_categorize_desikan_killiany` below is a VERBATIM copy of the if/elif
chain that lived in config/pain_params.py before the mapping became data. It is
frozen on purpose: it is the oracle. If the data-driven scheme and this function
ever disagree, the refactor changed behaviour, and the test says exactly where.

Do NOT "fix" this copy to match a new scheme. If you intend to change the
mapping, change the scheme and update the expectations deliberately -- the point
of the oracle is that an *accidental* change cannot pass.
"""

import pandas as pd
import pytest

from ieeg_ehr.config import roi_schemes


# ---------------------------------------------------------------------------
# The oracle: the pre-refactor implementation, unchanged.
# ---------------------------------------------------------------------------

def _ORIGINAL_categorize_desikan_killiany(dk_label):
    if pd.isna(dk_label):
        return 'Unlabeled'
    dk_label = str(dk_label).lower().strip("'\"")

    if any(x in dk_label for x in ['empty', 'unknown', 'undefined']):
        return 'Exclude'

    if any(x in dk_label for x in ['white-matter', 'ventraldc', 'cc_', 'wm-']):
        return 'White Matter'
    if any(x in dk_label for x in ['ventricle', 'csf', 'choroid-plexus', 'hypointensities']):
        return 'CSF/Ventricles'

    if 'hippocampus' in dk_label:
        return 'Hippocampus'
    if 'amygdala' in dk_label:
        return 'Amygdala'
    if 'thalamus' in dk_label:
        return 'Thalamus'

    if any(x in dk_label for x in ['caudate', 'putamen', 'pallidum', 'accumbens']):
        return 'Basal Ganglia'

    if 'thalamus' in dk_label:
        return 'Thalamus'

    if 'insula' in dk_label:
        return 'Insula'

    if any(x in dk_label for x in ['caudalanteriorcingulate', 'rostralanteriorcingulate']):
        return 'ACC'
    if any(x in dk_label for x in ['posteriorcingulate', 'isthmuscingulate']):
        return 'PCC'

    if any(x in dk_label for x in ['medialorbitofrontal', 'frontalpole']):
        return 'vmPFC'

    if 'lateralorbitofrontal' in dk_label:
        return 'OFC'

    if any(x in dk_label for x in ['rostralmiddlefrontal', 'caudalmiddlefrontal']):
        return 'dlPFC'

    if 'postcentral' in dk_label:
        return 'S1'

    if 'supramarginal' in dk_label:
        return 'S2 (supramarginal)'

    if any(x in dk_label for x in ['frontal', 'frontalpole', 'precentral', 'paracentral',
                                   'parsopercularis', 'parsorbitalis', 'parstriangularis']):
        return 'Frontal (other)'

    if any(x in dk_label for x in [
        'temporal', 'fusiform', 'entorhinal',
        'parahippocampal', 'bankssts', 'transversetemporal', 'temporalpole',
    ]):
        return 'Temporal'

    if any(x in dk_label for x in ['parietal', 'precuneus']):
        return 'Parietal'

    if any(x in dk_label for x in ['occipital', 'cuneus', 'pericalcarine', 'lingual']):
        return 'Occipital'

    if 'cerebellum' in dk_label:
        return 'Cerebellum'

    return 'Other'


_ORIGINAL_ROI_REGIONS = [
    'Hippocampus', 'Amygdala', 'Thalamus', 'Basal Ganglia', 'Insula', 'ACC', 'PCC',
    'vmPFC', 'OFC', 'dlPFC', 'S1', 'S2 (supramarginal)', 'Frontal (other)',
    'Temporal', 'Parietal',
]


# ---------------------------------------------------------------------------
# Labels to compare on.
# ---------------------------------------------------------------------------
# The real FreeSurfer/DK label vocabulary this dataset produces, plus the
# ordering traps that the refactor could plausibly have broken. Each trap is
# named so a failure is self-explaining.

_ORDERING_TRAPS = [
    # 'precuneus' contains 'cuneus' -> must be Parietal, NOT Occipital.
    'ctx-lh-precuneus', 'ctx-rh-precuneus',
    # 'frontalpole' is listed under BOTH vmPFC and Frontal (other) -> vmPFC wins.
    'ctx-lh-frontalpole', 'ctx-rh-frontalpole',
    # 'parahippocampal' must NOT be caught by the 'hippocampus' test.
    'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal',
    # 'temporalpole' is in Temporal but also contains 'temporal'.
    'ctx-lh-temporalpole',
    # medial vs lateral orbitofrontal -> vmPFC vs OFC.
    'ctx-lh-medialorbitofrontal', 'ctx-lh-lateralorbitofrontal',
    # middle frontal -> dlPFC, must beat the 'frontal' catch-all.
    'ctx-lh-rostralmiddlefrontal', 'ctx-rh-caudalmiddlefrontal',
    # superior frontal has no specific home -> Frontal (other).
    'ctx-lh-superiorfrontal',
    # cingulate splits
    'ctx-lh-caudalanteriorcingulate', 'ctx-rh-rostralanteriorcingulate',
    'ctx-lh-posteriorcingulate', 'ctx-rh-isthmuscingulate',
    # supramarginal -> the S2 proxy, must beat 'parietal'? (it is not parietal-named,
    # but it must beat nothing -- included to pin the category name exactly)
    'ctx-lh-supramarginal',
    # postcentral -> S1, must beat 'central' patterns
    'ctx-lh-postcentral', 'ctx-rh-precentral', 'ctx-lh-paracentral',
]

_TISSUE_AND_EDGE = [
    'Left-Cerebral-White-Matter', 'wm-lh-superiorfrontal', 'cc_Anterior',
    'Left-VentralDC', 'Left-Lateral-Ventricle', 'CSF', 'Left-choroid-plexus',
    'Right-WM-hypointensities', 'Unknown', 'undefined', 'empty',
    'Left-Hippocampus', 'Right-Amygdala', 'Left-Thalamus-Proper', 'Left-Caudate',
    'Right-Putamen', 'Left-Pallidum', 'Right-Accumbens-area',
    'ctx-lh-insula', 'ctx-rh-insula',
    'ctx-lh-fusiform', 'ctx-lh-entorhinal', 'ctx-lh-bankssts',
    'ctx-lh-transversetemporal', 'ctx-lh-superiortemporal',
    'ctx-lh-inferiorparietal', 'ctx-lh-superiorparietal',
    'ctx-lh-lateraloccipital', 'ctx-lh-cuneus', 'ctx-lh-pericalcarine',
    'ctx-lh-lingual', 'Left-Cerebellum-Cortex',
    'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-parstriangularis',
    # quoting / case variants the original explicitly stripped
    "'Left-Hippocampus'", '"ctx-lh-insula"', 'LEFT-HIPPOCAMPUS',
    # things with no home at all
    'Brain-Stem', 'Left-vessel', 'optic-chiasm', '',
]

ALL_LABELS = _ORDERING_TRAPS + _TISSUE_AND_EDGE


@pytest.mark.parametrize('label', ALL_LABELS)
def test_categorize_matches_original(label):
    assert roi_schemes.categorize_desikan_killiany(label) == \
        _ORIGINAL_categorize_desikan_killiany(label), (
            f'category changed for {label!r}: the data-driven scheme and the frozen '
            'pre-refactor chain disagree'
        )


@pytest.mark.parametrize('label', ALL_LABELS)
def test_region_for_dk_label_matches_original(label):
    original = _ORIGINAL_categorize_desikan_killiany(label)
    expected = original if original in _ORIGINAL_ROI_REGIONS else None
    assert roi_schemes.region_for_dk_label(label) == expected


def test_null_labels_are_unlabeled():
    for null in (None, float('nan'), pd.NA, pd.NaT):
        assert roi_schemes.categorize_desikan_killiany(null) == 'Unlabeled'
        assert roi_schemes.region_for_dk_label(null) is None


def test_display_order_unchanged():
    assert roi_schemes.roi_regions() == _ORIGINAL_ROI_REGIONS


def test_non_roi_categories_are_dropped_not_displayed():
    # Every category the scheme can return is either displayed or explicitly a
    # NON_ROI category -- no third, silently-ignored bucket.
    scheme = roi_schemes.resolve_roi_scheme()
    producible = set(scheme['patterns']) | {roi_schemes.UNLABELED, roi_schemes.FALLBACK}
    unaccounted = producible - set(scheme['display']) - set(roi_schemes.NON_ROI_CATEGORIES)
    assert not unaccounted, f'categories neither displayed nor declared non-ROI: {unaccounted}'


# ---------------------------------------------------------------------------
# Swappability
# ---------------------------------------------------------------------------

def test_json_scheme_round_trip(tmp_path):
    import json
    custom = {'patterns': {'Insula': ['insula'], 'Everything Else': ['ctx-']},
              'display': ['Insula']}
    path = tmp_path / 'scheme.json'
    path.write_text(json.dumps(custom))

    assert roi_schemes.roi_regions(str(path)) == ['Insula']
    assert roi_schemes.region_for_dk_label('ctx-lh-insula', str(path)) == 'Insula'
    # matched a pattern, but not displayed -> dropped, same as a NON_ROI category
    assert roi_schemes.region_for_dk_label('ctx-lh-superiorfrontal', str(path)) is None
    prov = roi_schemes.scheme_provenance(str(path))
    assert prov['patterns'] == custom['patterns'] and str(path) in prov['origin']


def test_json_key_order_is_precedence(tmp_path):
    """The whole refactor rests on dict order == precedence, so pin it."""
    import json
    parietal_first = {'patterns': {'Parietal': ['precuneus'], 'Occipital': ['cuneus']},
                      'display': ['Parietal', 'Occipital']}
    occipital_first = {'patterns': {'Occipital': ['cuneus'], 'Parietal': ['precuneus']},
                       'display': ['Parietal', 'Occipital']}
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(json.dumps(parietal_first))
    b.write_text(json.dumps(occipital_first))

    assert roi_schemes.region_for_dk_label('ctx-lh-precuneus', str(a)) == 'Parietal'
    assert roi_schemes.region_for_dk_label('ctx-lh-precuneus', str(b)) == 'Occipital'


def test_display_name_without_patterns_is_rejected():
    with pytest.raises(ValueError, match='have no patterns'):
        roi_schemes.resolve_roi_scheme({'patterns': {'Insula': ['insula']},
                                        'display': ['Insula', 'Typo']})


def test_unknown_scheme_name_is_rejected():
    with pytest.raises(ValueError, match='Unknown ROI scheme'):
        roi_schemes.resolve_roi_scheme('not-a-scheme')
