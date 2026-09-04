"""Tests for the per-drug dose table behind Fig 6.

The unit guard is the point: a mixed-unit panel is a wrong figure, not a
slightly-off one, so it has to raise rather than render.
"""

import pandas as pd
import pytest

from ieeg_ehr.med_analysis import plot_dose_distribution as pdd


def frame(rows):
    return pd.DataFrame([
        {'subject': r.get('subject', '001'), 'drug': r['drug'],
         'dose': r['dose'], 'dose_unit': r['unit'],
         'route': r.get('route', 'Oral')}
        for r in rows])


def test_counts_split_by_dose_and_route():
    admin = frame([
        {'drug': 'HYDROMORPHONE', 'dose': 1.0, 'unit': 'mg',
         'route': 'Intravenous'},
        {'drug': 'HYDROMORPHONE', 'dose': 1.0, 'unit': 'mg',
         'route': 'Intravenous', 'subject': '002'},
        {'drug': 'HYDROMORPHONE', 'dose': 1.0, 'unit': 'mg', 'route': 'Oral'},
        {'drug': 'HYDROMORPHONE', 'dose': 4.0, 'unit': 'mg', 'route': 'Oral'},
    ])
    counts, missing = pdd.dose_counts(admin, ['HYDROMORPHONE'])
    assert missing == 0
    iv = counts[(counts['dose'] == 1.0) & (counts['route'] == 'Intravenous')]
    assert iv['n_admin'].iloc[0] == 2
    assert iv['n_subjects'].iloc[0] == 2
    # The same dose at a different route stays a separate row -- that split is
    # the whole reason route is drawn.
    assert len(counts[counts['dose'] == 1.0]) == 2


def test_mixed_units_for_one_drug_raise():
    admin = frame([
        {'drug': 'OXYCODONE', 'dose': 5.0, 'unit': 'mg'},
        {'drug': 'OXYCODONE', 'dose': 1.0, 'unit': 'tablet'},
    ])
    with pytest.raises(ValueError):
        pdd.dose_counts(admin, ['OXYCODONE'])


def test_missing_doses_are_counted_and_excluded():
    admin = frame([
        {'drug': 'OXYCODONE', 'dose': 5.0, 'unit': 'mg'},
        {'drug': 'OXYCODONE', 'dose': float('nan'), 'unit': 'mg'},
    ])
    counts, missing = pdd.dose_counts(admin, ['OXYCODONE'])
    assert missing == 1
    assert counts['n_admin'].sum() == 1


def test_a_drug_with_no_rows_is_skipped_not_fatal():
    admin = frame([{'drug': 'OXYCODONE', 'dose': 5.0, 'unit': 'mg'}])
    counts, _ = pdd.dose_counts(admin, ['OXYCODONE', 'MORPHINE'])
    assert set(counts['drug']) == {'OXYCODONE'}


def test_tablet_unit_is_carried_through_not_converted():
    """1 tablet must never be silently treated as 1 mg; see the module docstring."""
    admin = frame([
        {'drug': 'HYDROCODONE-ACETAMINOPHEN', 'dose': 1.0, 'unit': 'tablet'},
        {'drug': 'HYDROCODONE-ACETAMINOPHEN', 'dose': 2.0, 'unit': 'tablet'},
    ])
    counts, _ = pdd.dose_counts(admin, ['HYDROCODONE-ACETAMINOPHEN'])
    assert set(counts['dose_unit']) == {'tablet'}
    assert sorted(counts['dose']) == [1.0, 2.0]
