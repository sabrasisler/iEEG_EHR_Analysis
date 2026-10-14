"""
Opioid exposure in morphine milligram equivalents (MME), one curve per session.

Each opioid dose contributes `dose x MME factor x g(t - t_dose)`, and the curves
add (linear kinetics, superposition). Every unit-dose curve g is a sum of
exponentials, `g(t) = sum_i c_i exp(-rate_i t)`, so one-compartment oral
absorption, an IV bolus and the three-compartment fentanyl bolus all go through
the same code. Factors, routes and PK parameters live in `config/opioid_mme.py`.

g follows the shape of PLASMA concentration, scaled so an IV bolus starts at 1.
For a one-compartment drug that is also the amount in the body. For fentanyl it
is not: most of a bolus redistributes into fat and muscle within minutes, where
it does little. The curve is therefore "active MME", and the axis says so.

Times are hours since `session_start`, within one (subject, session) only:
`taken_date` sits on a shared de-identified epoch (load.py, note 3). Doses given
before `session_start` (the OR, before the EMU admission) are not in the MAR
export, so the curve near t=0 is a lower bound for any patient who arrives with
opioid on board. `early_opioid_sessions` flags them.
"""

import logging

import numpy as np
import pandas as pd

from ieeg_ehr.config import med_taxonomy, opioid_mme
from ieeg_ehr.med_analysis import load

logger = logging.getLogger(__name__)

#: Grid step, from the spec.
GRID_STEP_H = 1.0 / 60.0

#: A first opioid dose this soon after `session_start` suggests drug carried in
#: from before the session, which the MAR export does not record.
EARLY_DOSE_H = 2.0


def opioid_doses(admin):
    """The opioid rows of the administration table, converted to MME.

    Returns one row per dose with columns subject, session, taken_dt, t_h (hours
    since session_start), drug, medication, ingredient, route, route_class,
    amount, amount_unit, mme.

    Raises on any opioid (drug, product, route) the config does not map, after
    logging every one of them. Excluded opioids (buprenorphine) are counted and
    logged, never silently dropped.
    """
    opioids = admin[admin['level2'] == 'Opioids'].copy()

    excluded = opioids['drug'].isin(opioid_mme.EXCLUDED_OPIOIDS)
    for drug, n in opioids.loc[excluded, 'drug'].value_counts().items():
        logger.info('excluded %d %s dose(s): %s', n, drug,
                    opioid_mme.EXCLUDED_OPIOIDS[drug])
    opioids = opioids[~excluded]

    rows, unmapped = [], set()
    for r in opioids.itertuples():
        mapped = _map_dose(r)
        if mapped is None:
            unmapped.add((r.drug, r.medication, r.route, r.dose_unit))
            continue
        rows.append(mapped)

    if unmapped:
        for combo in sorted(unmapped):
            logger.error('unmapped opioid (drug, product, route, unit): %s', combo)
        raise KeyError(
            f'{len(unmapped)} opioid (drug, product, route, unit) combination(s) '
            f'have no MME factor or PK curve in config/opioid_mme.py: '
            f'{sorted(unmapped)}')

    doses = pd.DataFrame(rows)
    doses['t_h'] = ((doses['taken_dt'] - doses['session_start'])
                    .dt.total_seconds() / 3600.0)
    return doses.sort_values(['subject', 'session', 'taken_dt']).reset_index(
        drop=True)


def _map_dose(r):
    """One administration -> a dict with amount and MME, or None if unmapped."""
    route_class = opioid_mme.ROUTE_CLASS.get(r.route)
    if r.medication in opioid_mme.COMBINATION_OPIOID_MG:
        ingredient, mg_per_unit = opioid_mme.COMBINATION_OPIOID_MG[r.medication]
        if r.dose_unit != 'tablet':
            return None
        amount, amount_unit = r.dose * mg_per_unit, 'mg'
    elif r.drug in opioid_mme.SINGLE_INGREDIENT_OPIOIDS:
        ingredient = opioid_mme.SINGLE_INGREDIENT_OPIOIDS[r.drug]
        amount, amount_unit = r.dose, r.dose_unit
    else:
        return None

    key = (ingredient, route_class)
    if (key not in opioid_mme.MME_FACTOR or key not in opioid_mme.UNIT_DOSE
            or amount_unit != opioid_mme.DOSE_UNIT[ingredient]):
        return None
    return dict(subject=r.subject, session=r.session, taken_dt=r.taken_dt,
                session_start=r.session_start, session_end=r.session_end,
                drug=r.drug, medication=r.medication, ingredient=ingredient,
                route=r.route, route_class=route_class, amount=amount,
                amount_unit=amount_unit,
                mme=amount * opioid_mme.MME_FACTOR[key].factor)


def unit_dose(curve, t_h):
    """g(t) on `t_h` (hours, >= 0)."""
    t_h = np.asarray(t_h, dtype=np.float64)
    return sum(c * np.exp(-rate * t_h)
               for c, rate in zip(curve.coefs, curve.rates))


def time_grid(session_start, session_end, doses, step_h=GRID_STEP_H):
    """Hours since session_start, from the earlier of session start and the
    first dose to session end, in `step_h` steps plus every dose time.

    The dose times are on the grid so each curve starts at its true peak: an IV
    fentanyl curve halves in under 2 min, so a dose that fell between two grid
    points would lose a large share of its height and area. Doses before
    session start do not exist in the export yet; the grid already admits them.
    """
    t0 = min(0.0, float(doses['t_h'].min())) if len(doses) else 0.0
    t1 = (session_end - session_start).total_seconds() / 3600.0
    return np.union1d(np.arange(t0, t1 + step_h / 2, step_h), doses['t_h'])


def exposure_curve(doses, t_h, left_limit=False):
    """Total active MME on `t_h`, the sum of every dose's curve (superposition).

    `left_limit=True` leaves out a dose at exactly its own time, which gives
    the value just before an IV bolus jumps.
    """
    total = np.zeros_like(t_h, dtype=np.float64)
    for d in doses.itertuples():
        after = t_h > d.t_h if left_limit else t_h >= d.t_h
        curve = opioid_mme.UNIT_DOSE[(d.ingredient, d.route_class)]
        total[after] += d.mme * unit_dose(curve, t_h[after] - d.t_h)
    return total


def early_opioid_sessions(doses, within_h=EARLY_DOSE_H):
    """(subject, session) -> hours to the first opioid dose, for sessions whose
    first dose comes within `within_h` of session start."""
    first = doses.groupby(['subject', 'session'])['t_h'].min()
    return first[first < within_h].to_dict()


# ============================================================================
# CHECKS (spec section 5)
# ============================================================================

def unit_dose_checks():
    """One row per (ingredient, route): the unit-dose curve's shape, against
    the label where the config carries a label Tmax.

    `auc_numeric_h` integrates g on a fine grid and must equal the analytic
    sum(c_i / rate_i). For a one-compartment drug that is 1/ke.
    """
    rows = []
    for (ingredient, route_class), curve in sorted(opioid_mme.UNIT_DOSE.items()):
        slowest = min(curve.rates)
        t = np.arange(0.0, 30.0 / slowest, 1e-4)
        g = unit_dose(curve, t)
        i_peak = int(np.argmax(g))
        below = np.nonzero(g[i_peak:] <= g[i_peak] / 2)[0]
        auc_analytic = sum(c / rate for c, rate in zip(curve.coefs, curve.rates))
        auc_numeric = float(np.trapezoid(g, t))
        rows.append(dict(
            ingredient=ingredient, route_class=route_class,
            tmax_h=float(t[i_peak]), label_tmax_h=curve.label_tmax_h,
            g_peak=float(g[i_peak]),
            peak_to_half_h=float(t[i_peak + below[0]] - t[i_peak]),
            terminal_half_life_h=float(np.log(2) / slowest),
            auc_analytic_h=auc_analytic, auc_numeric_h=auc_numeric,
            auc_ok=bool(abs(auc_numeric - auc_analytic) < 1e-3 * auc_analytic),
            source=curve.source))
    return pd.DataFrame(rows)


def raw_mme_tally(paths):
    """Total MME per session straight from the raw MAR files.

    Reads the CSVs itself and keeps `Given` rows, before load.py's
    multi-product dedup. Products map through the same config tables. What this
    tally skips is everything load.py and `opioid_doses` do in between.
    """
    raw = pd.concat([pd.read_csv(p, encoding='utf-8-sig', dtype=str,
                                 keep_default_na=False) for p in paths],
                    ignore_index=True)
    raw = raw[raw['mar_action'].str.strip() == load.GIVEN]
    rows = []
    for r in raw.itertuples():
        drug = load.extract_medication_name(r.medication)
        if med_taxonomy.DRUG_CLASS.get(drug, ('', ''))[1] != 'Opioids':
            continue
        if drug in opioid_mme.EXCLUDED_OPIOIDS:
            continue
        route_class = opioid_mme.ROUTE_CLASS[r.route.strip()]
        sig = float(r.sig)
        if r.medication in opioid_mme.COMBINATION_OPIOID_MG:
            ingredient, mg = opioid_mme.COMBINATION_OPIOID_MG[r.medication]
            sig *= mg
        else:
            ingredient = opioid_mme.SINGLE_INGREDIENT_OPIOIDS[drug]
        factor = opioid_mme.MME_FACTOR[(ingredient, route_class)].factor
        rows.append(dict(subject=r.sub_id.replace('sub-', ''),
                         session=r.ses_id.replace('ses-', ''),
                         mme=sig * factor))
    return (pd.DataFrame(rows).groupby(['subject', 'session'])['mme'].agg(
        ['sum', 'size']).rename(columns={'sum': 'mme_raw', 'size': 'n_raw'}))


def totals_check(doses, paths):
    """Pipeline total MME per session against the raw tally. A difference is a
    multi-product duplicate row (load.py note 2) or a bug, and the table says
    which sessions to look at."""
    pipeline = doses.groupby(['subject', 'session'])['mme'].agg(
        ['sum', 'size']).rename(columns={'sum': 'mme_pipeline',
                                         'size': 'n_pipeline'})
    out = pipeline.join(raw_mme_tally(paths), how='outer').fillna(0.0)
    out['mme_diff'] = out['mme_raw'] - out['mme_pipeline']
    out['n_diff'] = out['n_raw'] - out['n_pipeline']
    return out.reset_index()


#: Step for the area check. Trapezoid error on fentanyl's fastest phase is
#: ~(rate x step)^2 / 12, about 1% at the 1-min plot grid and under 0.05% here.
CHECK_STEP_H = 10.0 / 3600.0


def integral_check(doses, session_start, session_end):
    """Numerical area under one session's curve against sum(mme x area of g),
    with each dose's area truncated at session end. Confirms the
    superposition, not the PK.

    Each interval uses the curve's left limit at its right end. Plain
    trapezoids would ramp across every IV bolus jump and add half a step of
    full-height area per dose.
    """
    t_h = time_grid(session_start, session_end, doses, step_h=CHECK_STEP_H)
    right = exposure_curve(doses, t_h)
    left = exposure_curve(doses, t_h, left_limit=True)
    area = float(np.sum(0.5 * (right[:-1] + left[1:]) * np.diff(t_h)))
    expected = 0.0
    for d in doses.itertuples():
        g = opioid_mme.UNIT_DOSE[(d.ingredient, d.route_class)]
        span = t_h[-1] - d.t_h
        expected += d.mme * sum(c / rate * (1 - np.exp(-rate * span))
                                for c, rate in zip(g.coefs, g.rates))
    return area, expected
