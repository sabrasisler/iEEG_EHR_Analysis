"""Tests for the opioid MME curves.

The claims under test: every unit-dose curve integrates to sum(c_i / rate_i),
the oral curves peak where the label says, combination products convert by
per-tablet strength, curves add, and an unmapped opioid raises instead of
disappearing.
"""

import numpy as np
import pandas as pd
import pytest

from ieeg_ehr.config import opioid_mme
from ieeg_ehr.med_analysis import mme_curve

START = pd.Timestamp('2000-01-01 08:00:00')
END = pd.Timestamp('2000-01-03 08:00:00')


def admin(rows):
    return pd.DataFrame([
        {'subject': '001', 'session': '01', 'level2': 'Opioids',
         'session_start': START, 'session_end': END,
         'taken_dt': START + pd.Timedelta(hours=r.get('h', 1.0)),
         'drug': r['drug'], 'medication': r['medication'], 'route': r['route'],
         'dose': r['dose'], 'dose_unit': r['unit']}
        for r in rows])


def test_unit_dose_areas_match_analytic():
    checks = mme_curve.unit_dose_checks()
    assert checks['auc_ok'].all()


def test_one_compartment_area_is_inverse_ke():
    curve = opioid_mme.UNIT_DOSE[('HYDROMORPHONE', 'IV')]
    checks = mme_curve.unit_dose_checks().set_index(['ingredient', 'route_class'])
    assert checks.loc[('HYDROMORPHONE', 'IV'), 'auc_analytic_h'] == pytest.approx(
        1.0 / curve.rates[0])


def test_oral_tmax_matches_label_where_given():
    checks = mme_curve.unit_dose_checks().dropna(subset=['label_tmax_h'])
    assert len(checks)
    assert np.allclose(checks['tmax_h'], checks['label_tmax_h'], atol=0.05)


def test_iv_bolus_starts_at_one():
    for (_, route), curve in opioid_mme.UNIT_DOSE.items():
        if route == 'IV':
            assert mme_curve.unit_dose(curve, 0.0) == pytest.approx(1.0)


def test_combination_tablets_convert_by_strength():
    doses = mme_curve.opioid_doses(admin([
        {'drug': 'HYDROCODONE-ACETAMINOPHEN',
         'medication': 'HYDROCODONE-ACETAMINOPHEN 10-325 MG PO TABS',
         'route': 'Oral', 'dose': 2.0, 'unit': 'tablet'}]))
    assert doses['amount'].iloc[0] == 20.0
    assert doses['mme'].iloc[0] == 20.0


def test_fentanyl_converts_per_mcg():
    doses = mme_curve.opioid_doses(admin([
        {'drug': 'FENTANYL', 'medication': 'FENTANYL (PF) 50 MCG/ML INJECTION',
         'route': 'Intravenous', 'dose': 50.0, 'unit': 'mcg'}]))
    assert doses['mme'].iloc[0] == pytest.approx(50.0 * 0.2)


def test_curves_add():
    rows = [{'drug': 'OXYCODONE', 'medication': 'OXYCODONE 5 MG PO TABS',
             'route': 'Oral', 'dose': 5.0, 'unit': 'mg', 'h': h}
            for h in (1.0, 5.0)]
    both = mme_curve.opioid_doses(admin(rows))
    t = mme_curve.time_grid(START, END, both)
    total = mme_curve.exposure_curve(both, t)
    parts = sum(mme_curve.exposure_curve(both.iloc[[i]], t) for i in range(2))
    assert np.allclose(total, parts)
    area, expected = mme_curve.integral_check(both, START, END)
    assert area == pytest.approx(expected, rel=1e-3)


def test_unmapped_route_raises():
    with pytest.raises(KeyError, match='OXYCODONE'):
        mme_curve.opioid_doses(admin([
            {'drug': 'OXYCODONE', 'medication': 'OXYCODONE 5 MG PO TABS',
             'route': 'Rectal', 'dose': 5.0, 'unit': 'mg'}]))


def test_wrong_unit_raises():
    with pytest.raises(KeyError):
        mme_curve.opioid_doses(admin([
            {'drug': 'FENTANYL', 'medication': 'FENTANYL (PF) 50 MCG/ML INJECTION',
             'route': 'Intravenous', 'dose': 0.05, 'unit': 'mg'}]))


def test_buprenorphine_is_excluded():
    rows = [{'drug': 'BUPRENORPHINE-NALOXONE',
             'medication': 'BUPRENORPHINE-NALOXONE 2-0.5 MG SL FILM',
             'route': 'Sublingual', 'dose': 1.0, 'unit': 'Film'},
            {'drug': 'OXYCODONE', 'medication': 'OXYCODONE 5 MG PO TABS',
             'route': 'Oral', 'dose': 5.0, 'unit': 'mg'}]
    doses = mme_curve.opioid_doses(admin(rows))
    assert list(doses['ingredient']) == ['OXYCODONE']


def test_fentanyl_between_grid_points_keeps_its_peak():
    """A dose 30 s off the minute grid still starts at full height, and its
    area still matches. Fentanyl halves in under 2 min, so a 1-min miss loses
    a large share of both."""
    doses = mme_curve.opioid_doses(admin([
        {'drug': 'FENTANYL', 'medication': 'FENTANYL (PF) 50 MCG/ML INJECTION',
         'route': 'Intravenous', 'dose': 50.0, 'unit': 'mcg',
         'h': 1.0 + 30 / 3600}]))
    t = mme_curve.time_grid(START, END, doses)
    assert mme_curve.exposure_curve(doses, t).max() == pytest.approx(10.0)
    area, expected = mme_curve.integral_check(doses, START, END)
    assert area == pytest.approx(expected, rel=1e-3)
