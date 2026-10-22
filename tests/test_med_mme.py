"""Tests for the opioid MME curves.

The claims under test: every unit-dose curve integrates to sum(c_i / rate_i),
starts at 0 and peaks at 1, the oral plasma curves peak where the label says,
the effect-site curves peak at the published time and keep the plasma area,
combination products convert by per-tablet strength, curves add, and an
unmapped opioid raises instead of disappearing.
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


def area(curve):
    return sum(c / r for c, r in zip(curve.coefs, curve.rates))


def test_one_compartment_area_is_inverse_ke():
    curve = opioid_mme.PLASMA[('HYDROMORPHONE', 'IV')]
    assert area(curve) == pytest.approx(1.0 / curve.rates[0])


def test_oral_plasma_tmax_matches_label_where_given():
    labelled = [c for c in opioid_mme.PLASMA.values() if c.label_tmax_h]
    assert len(labelled)
    for curve in labelled:
        assert opioid_mme.peak_time(curve) == pytest.approx(curve.label_tmax_h,
                                                            abs=0.05)


def test_every_curve_starts_at_zero_and_peaks_at_one():
    checks = mme_curve.unit_dose_checks()
    assert checks['peak_ok'].all()
    for curve in opioid_mme.UNIT_DOSE.values():
        assert curve(0.0) == pytest.approx(0.0, abs=1e-12)


def test_effect_site_peaks_at_published_time():
    checks = mme_curve.unit_dose_checks().dropna(subset=['effect_tpeak_h'])
    assert set(checks['ingredient']) == set(opioid_mme.EFFECT_TPEAK_MIN)
    assert np.allclose(checks['tmax_h'], checks['effect_tpeak_h'], atol=1e-3)


def test_effect_site_keeps_the_plasma_area():
    for ingredient, ke0 in opioid_mme.KE0_PER_H.items():
        plasma = opioid_mme.PLASMA[(ingredient, 'IV')]
        assert area(opioid_mme.effect_site(plasma, ke0)) == pytest.approx(area(plasma))


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
    """A dose 30 s off the minute grid peaks at its MME on the 1-min grid, and
    its area matches. Fentanyl has the fastest effect site (peak at 3.6 min)."""
    doses = mme_curve.opioid_doses(admin([
        {'drug': 'FENTANYL', 'medication': 'FENTANYL (PF) 50 MCG/ML INJECTION',
         'route': 'Intravenous', 'dose': 50.0, 'unit': 'mcg',
         'h': 1.0 + 30 / 3600}]))
    t = mme_curve.time_grid(START, END, doses)
    assert mme_curve.exposure_curve(doses, t).max() == pytest.approx(10.0, rel=1e-2)
    area, expected = mme_curve.integral_check(doses, START, END)
    assert area == pytest.approx(expected, rel=1e-3)


def _oxy_and_fentanyl():
    return mme_curve.opioid_doses(admin([
        {'drug': 'OXYCODONE', 'medication': 'OXYCODONE 5 MG PO TABS',
         'route': 'Oral', 'dose': 5.0, 'unit': 'mg', 'h': 1.0},
        {'drug': 'FENTANYL', 'medication': 'FENTANYL (PF) 50 MCG/ML INJECTION',
         'route': 'Intravenous', 'dose': 50.0, 'unit': 'mcg', 'h': 2.0}]))


def test_window_mean_matches_numeric_average():
    doses = _oxy_and_fentanyl()
    lo, hi = np.array([0.5, 1.9, 2.0, 3.0]), np.array([1.5, 2.1, 2.5, 8.0])
    exact = mme_curve.window_mean(doses, lo, hi)
    for value, a, b in zip(exact, lo, hi):
        t = np.linspace(a, b, 200001)
        numeric = np.trapezoid(mme_curve.exposure_curve(doses, t), t) / (b - a)
        assert value == pytest.approx(numeric, rel=1e-4)


def test_window_mean_takes_a_dose_at_its_start_not_at_its_end():
    """[lo, hi): a dose charted at hi belongs to the next window."""
    doses = _oxy_and_fentanyl().iloc[[1]]            # fentanyl at 2.0 h
    assert mme_curve.window_mean(doses, [1.9], [2.0])[0] == 0.0
    assert mme_curve.window_mean(doses, [2.0], [2.1])[0] > 0.0


def test_window_mean_refuses_an_empty_window():
    with pytest.raises(ValueError):
        mme_curve.window_mean(_oxy_and_fentanyl(), [1.0], [1.0])
