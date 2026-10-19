"""
Opioid MME factors and unit-dose PK curves, held as data in config.

Same shape as `med_taxonomy.py`: hand-curated, version-controlled, with a source
on every number. `med_analysis/mme_curve.py` reads it. Decisions behind it
(2026-10-14, Sabra):

- Oral factors are CDC 2022 throughout (Dowell et al., MMWR Recomm Rep
  2022;71(3), Table). CDC 2022 lists oral and transdermal opioids only, so the
  IV factors come from Nielsen et al. 2014, NDARC Technical Report 329.
- Buprenorphine is excluded: a partial agonist, with no CDC 2022 factor.
- IV fentanyl uses a three-compartment plasma curve. Every other drug and
  route uses one compartment (first-order absorption for PO, bolus for IV).
- Feeding-tube doses count as PO.

UNIT-DOSE CURVES
----------------
Every curve is a sum of exponentials, g(t) = sum_i coefs[i] * exp(-rates[i] t),
t in hours since the dose. The area under g is sum(coefs[i] / rates[i]).

- PO, one compartment:  g = ka/(ka-ke) * (exp(-ke t) - exp(-ka t))
- IV bolus:             g = exp(-ke t)
- IV fentanyl:          plasma after a bolus from a three-compartment model,
                        normalized to g(0) = 1

The oral ka values were back-solved from each label's Tmax and t1/2:
Tmax = ln(ka/ke) / (ka - ke).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class Factor:
    factor: float
    source: str


@dataclass(frozen=True)
class UnitDose:
    coefs: tuple
    rates: tuple            # 1/h
    source: str
    label_tmax_h: float | None = None


def oral(ka, ke, source, label_tmax_h=None):
    a = ka / (ka - ke)
    return UnitDose((a, -a), (ke, ka), source, label_tmax_h)


def iv_bolus(ke, source):
    return UnitDose((1.0,), (ke,), source)


def oral_from_label(tmax_h, t_half_h, source):
    """One-compartment PO curve whose peak falls at the label Tmax."""
    ke = np.log(2) / t_half_h
    ka = brentq(lambda ka: np.log(ka / ke) / (ka - ke) - tmax_h,
                ke * (1 + 1e-9), 1e3)
    return oral(ka, ke, source, label_tmax_h=tmax_h)


def three_compartment_bolus(v1, v2, v3, cl1, cl2, cl3, source):
    """Central-compartment curve after a unit IV bolus, from volumes (L) and
    clearances (L/min). Returns rates in 1/h and coefs that sum to 1."""
    k10, k12, k13 = cl1 / v1, cl2 / v1, cl3 / v1
    k21, k31 = cl2 / v2, cl3 / v3
    k = np.array([[-(k10 + k12 + k13), k21, k31],
                  [k12, -k21, 0.0],
                  [k13, 0.0, -k31]])
    eigvals, vecs = np.linalg.eig(k)
    coefs = vecs[0, :] * np.linalg.inv(vecs)[:, 0]
    order = np.argsort(-eigvals.real)
    return UnitDose(tuple(float(c) for c in coefs.real[order]),
                    tuple(float(-r * 60.0) for r in eigvals.real[order]),
                    source)


# ============================================================================
# ROUTES AND PRODUCTS
# ============================================================================

#: MAR route -> route class. A route missing here makes its doses unmapped.
ROUTE_CLASS = {'Oral': 'PO', 'Feeding Tube': 'PO', 'Intravenous': 'IV'}

#: Canonical drug (load.extract_medication_name) -> ingredient. `sig` already
#: carries the administered amount in `DOSE_UNIT`.
SINGLE_INGREDIENT_OPIOIDS = {
    'FENTANYL': 'FENTANYL',
    'HYDROMORPHONE': 'HYDROMORPHONE',
    'MORPHINE': 'MORPHINE',
    'OXYCODONE': 'OXYCODONE',
    'TRAMADOL': 'TRAMADOL',
}

#: Combination product (exact MAR `medication` string) -> (opioid ingredient,
#: mg per tablet). `sig` counts tablets. Salt weights, as on the label.
COMBINATION_OPIOID_MG = {
    'HYDROCODONE-ACETAMINOPHEN 10-325 MG PO TABS': ('HYDROCODONE', 10.0),
    'HYDROCODONE-ACETAMINOPHEN 5-325 MG PO TABS': ('HYDROCODONE', 5.0),
    'HYDROCODONE-ACETAMINOPHEN 5-500 MG PO TABS': ('HYDROCODONE', 5.0),
    'OXYCODONE-ACETAMINOPHEN 5-325 MG PO TABS': ('OXYCODONE', 5.0),
    'ACETAMINOPHEN-CODEINE 300-30 MG PO TABS': ('CODEINE', 30.0),
}

#: Opioids left out of the curve, with the reason logged on every run.
EXCLUDED_OPIOIDS = {
    'BUPRENORPHINE-NALOXONE': ('partial agonist with no CDC 2022 MME factor; '
                               'excluded (decision 2026-10-14)'),
}

#: Unit the amount must be in before the factor applies.
DOSE_UNIT = {'CODEINE': 'mg', 'FENTANYL': 'mcg', 'HYDROCODONE': 'mg',
             'HYDROMORPHONE': 'mg', 'MORPHINE': 'mg', 'OXYCODONE': 'mg',
             'TRAMADOL': 'mg'}

#: Legend and color order.
INGREDIENT_ORDER = ('HYDROCODONE', 'OXYCODONE', 'HYDROMORPHONE', 'FENTANYL',
                    'MORPHINE', 'TRAMADOL', 'CODEINE')


# ============================================================================
# MME FACTORS  (mg MME per mg, or per mcg for fentanyl)
# ============================================================================

CDC_2022 = 'CDC 2022 (Dowell et al., MMWR Recomm Rep 2022;71(3))'
NDARC = 'Nielsen et al. 2014, NDARC Technical Report 329'

MME_FACTOR = {
    ('CODEINE', 'PO'): Factor(0.15, CDC_2022),
    ('HYDROCODONE', 'PO'): Factor(1.0, CDC_2022),
    ('HYDROMORPHONE', 'PO'): Factor(5.0, CDC_2022),
    ('OXYCODONE', 'PO'): Factor(1.5, CDC_2022),
    ('TRAMADOL', 'PO'): Factor(0.2, CDC_2022),
    ('FENTANYL', 'IV'): Factor(0.2, f'{NDARC} (sensitivity 0.15-0.3)'),
    ('HYDROMORPHONE', 'IV'): Factor(15.0, f'{NDARC} (sensitivity 10-20)'),
    ('MORPHINE', 'IV'): Factor(3.0, f'{NDARC} (sensitivity 2.5-3)'),
}


# ============================================================================
# UNIT-DOSE PK CURVES
# ============================================================================

UNIT_DOSE = {
    ('HYDROCODONE', 'PO'): oral(2.04, 0.182, 'Norco label (t1/2 3.8 h)',
                                label_tmax_h=1.3),
    ('OXYCODONE', 'PO'): oral(1.82, 0.185, 'Roxicodone label (t1/2 3.75 h)'),
    ('HYDROMORPHONE', 'PO'): oral(3.89, 0.267, 'Dilaudid label (t1/2 2.6 h)',
                                  label_tmax_h=0.74),
    ('HYDROMORPHONE', 'IV'): iv_bolus(0.301, 'Dilaudid injection label '
                                             '(t1/2 2.3 h)'),
    # The label's "accepted elimination half-life in normal subjects is 1.5 to
    # 2 hours"; the midpoint. Stanski et al. 1978 report 2.9 h.
    ('MORPHINE', 'IV'): iv_bolus(np.log(2) / 1.75, 'Morphine Sulfate Injection '
                                                   'label, 12.3 (t1/2 1.5-2 h)'),
    # Shaped on M1, the metabolite that carries the opioid effect (Tmax 3 h,
    # t1/2 7.4 h). Parent tramadol: Tmax 2 h, t1/2 6.3 h.
    ('TRAMADOL', 'PO'): oral_from_label(3.0, 7.4, 'Ultram label (M1: Tmax 3 h, '
                                                  't1/2 7.4 h)'),
    ('CODEINE', 'PO'): oral_from_label(1.0, 3.0, 'Codeine Sulfate Tablets label, '
                                                 '12.3 (Tmax 1 h, t1/2 3 h)'),
    # stanpumpR drugs_fentanyl.R (Shafer), 70 kg reference values. Pools
    # McClain & Hug 1980, Scott & Stanski 1987, Hudson 1986, Varvel 1989 and
    # Shafer 1990. No weight scaling: weights are not in the export.
    ('FENTANYL', 'IV'): three_compartment_bolus(
        v1=12.1, v2=35.7, v3=224.0, cl1=0.632, cl2=2.8, cl3=1.55,
        source='stanpumpR drugs_fentanyl.R, 70 kg (Shafer)'),
}


# ---------------------------------------------------------------------------
# EXPOSURE VERSION: the pair covariates of the `pain_change` opioid model
# ---------------------------------------------------------------------------
# One name for one complete recipe of dC and Cbar (docs/run_tokens.md): the dose
# source, the drug set, the MME factors and unit-dose curves above, the typical
# dose below, and the windows the curve is averaged over. ANY change to the
# recipe, a fix included, takes a new version, so a version always names one
# set of curves.
#
# mme1 (2026-10-15): MAR doses only, so doses before session_start (the OR) are
# missing and early curves are lower bounds. Opioids only, buprenorphine
# excluded. C at an assessment = mean curve over the 5-min epoch before it;
# Cbar = mean curve over [t1, t2). Units: typical doses.
EXPOSURE_VERSION = 'mme1'
#
# mme2 (2026-10-19): mme1, but C at an assessment = mean curve over the neural
# epoch the pair's d_z uses for it, so a second epoch moved to
# [t2 - shift - 5, t2 - shift) (`pain_change --second-epoch-shift-min`) moves C_2
# with it. The window itself is the run's data token (`e2pre10to5`). Equal to mme1
# when no epoch is shifted, so an unshifted run keeps the mme1 name.
SHIFTED_EXPOSURE_VERSION = 'mme2'


def exposure_version(shift_min):
    """The version a pain_change run's exposure follows for its second-epoch shift."""
    return SHIFTED_EXPOSURE_VERSION if shift_min else EXPOSURE_VERSION

#: One typical dose, the divisor that puts exposure in typical-dose units: the
#: median MME over every opioid dose of the 63 discovery subjects (624 doses,
#: 54 subjects), measured 2026-10-15. The median of subject medians is 7.5.
TYPICAL_DOSE_MME = 10.0
