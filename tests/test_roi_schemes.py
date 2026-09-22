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


# ---------------------------------------------------------------------------
# Coordinate-derived regions and the domain registry (2026-09-21)
# ---------------------------------------------------------------------------
# The invariant worth pinning here is NOT "aIns exists". It is that the label
# layer cannot assign an insular contact to anything, so a caller that forgets
# the coordinate step loses insula rather than mislabelling it -- the failure
# mode this design was chosen for.

@pytest.mark.parametrize('scheme', ['roi_v2_ins', 'roi_v2_ofc_ins'])
def test_split_scheme_displays_the_parts_and_not_the_parent(scheme):
    display = roi_schemes.roi_regions(scheme)
    assert 'aIns' in display and 'pIns' in display
    assert 'Insula' not in display
    # Displayed in the parent's slot, so the figure row order is unchanged
    # apart from one row becoming two.
    unsplit = 'roi_v2' if scheme == 'roi_v2_ins' else 'roi_v2_ofc'
    base = roi_schemes.roi_regions(unsplit)
    expected = []
    for r in base:
        expected.extend(('aIns', 'pIns') if r == 'Insula' else (r,))
    assert display == expected


@pytest.mark.parametrize('scheme', ['roi_v2_ins', 'roi_v2_ofc_ins'])
def test_insular_label_reaches_no_region_without_the_coordinate_step(scheme):
    assert roi_schemes.region_for_dk_label('ctx-lh-insula', scheme) is None
    assert roi_schemes.region_for_dk_label(
        'ctx-lh-insula', scheme, include_coordinate_parents=True) == 'Insula'


def test_coordinate_regions_are_reported_and_empty_for_label_only_schemes():
    assert roi_schemes.coordinate_regions('roi_v2_ins') == {'Insula': ('aIns', 'pIns')}
    assert roi_schemes.coordinate_regions('roi_v2') == {}
    assert roi_schemes.coordinate_regions('pain_domains_v2') == {}


def test_a_coordinate_region_with_no_parent_is_rejected():
    with pytest.raises(ValueError, match='coordinate_regions parents'):
        roi_schemes.resolve_roi_scheme(
            {'patterns': {'S1': ['postcentral']}, 'display': ['S1', 'aIns'],
             'coordinate_regions': {'Insula': ['aIns']}})


def test_splitting_a_scheme_does_not_change_any_other_region():
    for label in ('ctx-lh-postcentral', 'Left-Thalamus', 'ctx-rh-precuneus',
                  'ctx-lh-rostralanteriorcingulate', 'Left-Cerebral-White-Matter'):
        assert (roi_schemes.region_for_dk_label(label, 'roi_v2_ins')
                == roi_schemes.region_for_dk_label(label, 'roi_v2'))


# -- the domain registry ----------------------------------------------------

def test_domain_registry_reproduces_the_fused_schemes_assignment():
    """label -> domain must be the same whether it goes via the ROI or not.

    The fused scheme and the ROI membership are two spellings of one mapping.
    They are built from different data, so if an edit to one is not mirrored in
    the other this is what says so.
    """
    spec = roi_schemes.domain_scheme('pain_domains_v2')
    labels = [p for pats in
              roi_schemes.resolve_roi_scheme(spec['base'])['patterns'].values()
              for p in pats]
    for label in labels:
        fused = roi_schemes.region_for_dk_label(label, 'pain_domains_v2')
        roi = roi_schemes.region_for_dk_label(label, spec['base'])
        assert fused == spec['roi_to_domain'].get(roi), label


def test_pain_domains_v3_places_the_insula_halves_in_opposite_domains():
    spec = roi_schemes.domain_scheme('pain_domains_v3')
    assert spec['roi_to_domain']['aIns'] == 'Affective'
    assert spec['roi_to_domain']['pIns'] == 'Sensory'
    assert spec['base'] == 'roi_v2_ins'
    # v3 is v2 plus the two insula halves and nothing else.
    v2 = roi_schemes.domain_scheme('pain_domains_v2')
    assert (set(spec['roi_to_domain']) - {'aIns', 'pIns'}
            == set(v2['roi_to_domain']))


def test_a_domain_member_the_base_scheme_lacks_is_rejected():
    roi_schemes.DOMAIN_SCHEMES['_bad'] = {
        'base': 'roi_v2', 'members': {'Sensory': ('S2/P0',)},
        'display': ['Sensory']}
    try:
        with pytest.raises(ValueError, match='not displayed regions'):
            roi_schemes.domain_scheme('_bad')
    finally:
        del roi_schemes.DOMAIN_SCHEMES['_bad']


def test_domain_scheme_provenance_names_what_it_dropped():
    prov = roi_schemes.domain_scheme_provenance('pain_domains_v3')
    assert prov['base_scheme'] == 'roi_v2_ins'
    # These are real regions of the base scheme that no domain claims. Their
    # absence from the model is a result, so it is recorded, not inferred.
    assert 'PCC' in prov['unassigned_rois']
    assert 'Hippocampus' in prov['unassigned_rois']
    assert 'aIns' not in prov['unassigned_rois']


# ---------------------------------------------------------------------------
# insula_ap.apply_split -- the coordinate step itself
# ---------------------------------------------------------------------------

def _maps():
    return {'sub-001': {'A1-A2': 'Insula', 'B1-B2': 'S1', 'C1-C2': 'Insula'},
            'sub-002': {'D1-D2': 'Insula'}}


def test_apply_split_replaces_parents_and_leaves_everything_else():
    from ieeg_ehr.analysis import insula_ap
    maps = _maps()
    contacts = pd.DataFrame({
        'subject_id': ['sub-001', 'sub-001', 'sub-002'],
        'channel': ['A1-A2', 'C1-C2', 'D1-D2'],
        'ins_part': ['aIns', 'pIns', 'aIns']})
    report = insula_ap.apply_split(maps, contacts, 'roi_v2_ins')
    assert maps == {'sub-001': {'A1-A2': 'aIns', 'B1-B2': 'S1', 'C1-C2': 'pIns'},
                    'sub-002': {'D1-D2': 'aIns'}}
    assert report['n_by_part'] == {'aIns': 2, 'pIns': 1}
    assert report['n_dropped_no_coordinate'] == 0


def test_an_insular_channel_with_no_coordinate_is_dropped_not_guessed():
    """The one behaviour that must never become a default.

    An insular contact with no usable MNI coordinate cannot be placed on either
    side of the cut. Defaulting it would put a contact in a DOMAIN on the
    strength of nothing at all, so it leaves -- counted.
    """
    from ieeg_ehr.analysis import insula_ap
    maps = _maps()
    contacts = pd.DataFrame({'subject_id': ['sub-001'], 'channel': ['A1-A2'],
                             'ins_part': ['aIns']})
    report = insula_ap.apply_split(maps, contacts, 'roi_v2_ins')
    assert maps['sub-001'] == {'A1-A2': 'aIns', 'B1-B2': 'S1'}
    assert maps['sub-002'] == {}
    assert report['n_dropped_no_coordinate'] == 2


def test_apply_split_is_a_no_op_for_a_label_only_scheme():
    from ieeg_ehr.analysis import insula_ap
    maps = _maps()
    before = {k: dict(v) for k, v in maps.items()}
    assert insula_ap.apply_split(maps, None, 'roi_v2') == {'applied': False}
    assert maps == before


def test_median_split_tie_goes_posterior():
    """Pinned because it is arbitrary: at odd n the median IS a contact."""
    import numpy as np
    from ieeg_ehr.analysis import insula_ap
    contacts = pd.DataFrame({
        'subject_id': ['s'] * 3, 'channel': list('abc'),
        'mni_x': [-35.0, -35.0, -35.0], 'mni_y': [-10.0, 0.0, 10.0],
        'mni_z': [0.0, 0.0, 0.0], 'hemisphere': ['L'] * 3})
    out, desc = insula_ap.median_split(contacts)
    assert desc['thresholds'] == {'all': 0.0}
    assert list(out['ins_part']) == ['pIns', 'pIns', 'aIns']
    assert np.isclose(desc['n_posterior'], 2)


def test_median_split_can_be_pinned_to_an_earlier_runs_threshold():
    from ieeg_ehr.analysis import insula_ap
    contacts = pd.DataFrame({
        'subject_id': ['s'] * 3, 'channel': list('abc'),
        'mni_x': [-35.0] * 3, 'mni_y': [-10.0, 0.0, 10.0],
        'mni_z': [0.0] * 3, 'hemisphere': ['L'] * 3})
    out, desc = insula_ap.median_split(contacts, threshold=-20.0)
    assert desc['threshold_source'] == 'pinned'
    assert list(out['ins_part']) == ['aIns', 'aIns', 'aIns']


# ---------------------------------------------------------------------------
# DerSimonian-Laird tau / I2 (domain-level consistency)
# ---------------------------------------------------------------------------
# The whole reason tau exists here rather than slope.std() is that it subtracts
# the sampling component. These pin that it actually does.

def test_tau_is_zero_when_the_spread_is_pure_sampling_noise():
    """k subjects drawn around ONE true slope, spread matching their SEs."""
    import numpy as np
    from ieeg_ehr.analysis.plot_domain_consistency import dersimonian_laird
    rng = np.random.default_rng(0)
    se = np.full(60, 0.01)
    slope = 0.02 + rng.normal(0, 0.01, 60)      # spread == the sampling SE
    tau, i2, _, k = dersimonian_laird(slope, se)
    assert k == 60
    assert tau < 0.004, tau          # ~0: nothing left after the subtraction
    assert i2 < 0.25, i2


def test_tau_recovers_real_between_subject_spread():
    import numpy as np
    from ieeg_ehr.analysis.plot_domain_consistency import dersimonian_laird
    rng = np.random.default_rng(1)
    se = np.full(200, 0.002)
    true = rng.normal(0.02, 0.01, 200)          # real tau = 0.01
    slope = true + rng.normal(0, 0.002, 200)
    tau, i2, _, _ = dersimonian_laird(slope, se)
    assert 0.008 < tau < 0.012, tau
    assert i2 > 0.9, i2


def test_raw_sd_and_tau_diverge_when_precision_is_uneven():
    """The reason the raw SD is not the figure.

    Half the subjects measured 10x worse than the other half, but NO real
    between-subject variation. The plain SD reports a large spread; tau, which
    knows the SEs, reports almost none.
    """
    import numpy as np
    from ieeg_ehr.analysis.plot_domain_consistency import dersimonian_laird
    rng = np.random.default_rng(2)
    se = np.concatenate([np.full(30, 0.002), np.full(30, 0.02)])
    slope = 0.01 + rng.normal(0, 1, 60) * se     # spread is exactly sampling
    tau, i2, _, _ = dersimonian_laird(slope, se)
    raw_sd = float(np.std(slope, ddof=1))
    assert raw_sd > 0.008, raw_sd                # the SD looks heterogeneous
    assert tau < 0.004, tau                      # tau knows it is not
    assert tau < raw_sd / 2


def test_too_few_subjects_gives_nan_rather_than_a_number():
    import numpy as np
    from ieeg_ehr.analysis.plot_domain_consistency import dersimonian_laird
    tau, i2, _, k = dersimonian_laird(np.array([0.01, 0.02]),
                                      np.array([0.001, 0.001]))
    assert k == 2 and np.isnan(tau) and np.isnan(i2)


def test_non_finite_and_zero_se_subjects_are_excluded_not_crashed_on():
    import numpy as np
    from ieeg_ehr.analysis.plot_domain_consistency import dersimonian_laird
    slope = np.array([0.01, 0.02, np.nan, 0.015, 0.012])
    se = np.array([0.001, 0.001, 0.001, 0.0, 0.001])
    _, _, _, k = dersimonian_laird(slope, se)
    assert k == 3          # the NaN slope and the zero-SE subject both leave
