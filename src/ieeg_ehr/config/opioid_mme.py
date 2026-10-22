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

Revised 2026-10-21 (Sabra, DECISIONS 2026-10-21): an effect-site compartment
for every drug with a published time to peak effect, and every curve scaled to
peak at 1 per MME.

UNIT-DOSE CURVES
----------------
Every curve is a sum of exponentials, g(t) = sum_i coefs[i] * exp(-rates[i] t),
t in hours since the dose. The area under g is sum(coefs[i] / rates[i]).

`PLASMA` holds the plasma shapes:

- PO, one compartment:  g = ka/(ka-ke) * (exp(-ke t) - exp(-ka t))
- IV bolus:             g = exp(-ke t)
- IV fentanyl:          plasma after a bolus from a three-compartment model,
                        normalized to g(0) = 1

The oral ka values were back-solved from each label's Tmax and t1/2:
Tmax = ln(ka/ke) / (ka - ke).

`UNIT_DOSE`, the curves the exposure uses, builds on them in two steps:

1. EFFECT SITE. A plasma curve peaks before the drug acts: an IV bolus is at
   full height the moment it is given, and fentanyl's plasma curve halves in
   2 min. The biophase lags plasma, dCe/dt = ke0 (Cp - Ce) with Ce(0) = 0
   (Sheiner et al. 1979), so each plasma term c exp(-r t) becomes
   c ke0 / (ke0 - r) (exp(-r t) - exp(-ke0 t)), with the plasma curve's area.
   ke0 is a property of the drug. It is solved once, so that the effect site
   peaks at the drug's published `EFFECT_TPEAK_MIN` after an IV bolus of the
   `PLASMA` curve, and applied to every route of that drug. Drugs with no
   published value keep the plasma shape.
2. PEAK = 1. Each curve is scaled to peak at 1, so a dose of X MME peaks at X
   active MME: an MME factor is an equianalgesic dose, read at peak effect. A
   short-acting drug keeps a short curve and less area per MME.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq, minimize_scalar


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
    ke0_per_h: float | None = None

    def __call__(self, t_h):
        """g(t) on `t_h` (hours since the dose, >= 0)."""
        t_h = np.asarray(t_h, dtype=np.float64)
        return sum(c * np.exp(-rate * t_h) for c, rate in zip(self.coefs, self.rates))


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


def peak_time(curve):
    """Hours from the dose to the peak of a single-peaked curve."""
    span = 10.0 * np.log(2) / min(curve.rates)
    return minimize_scalar(lambda t: -curve(t), bounds=(0.0, span),
                           method='bounded', options={'xatol': 1e-8}).x


def effect_site(plasma, ke0):
    """The biophase curve of `plasma` for a transfer rate `ke0` (1/h)."""
    coefs = tuple(c * ke0 / (ke0 - r) for c, r in zip(plasma.coefs, plasma.rates))
    return replace(plasma, coefs=coefs + (-sum(coefs),),
                   rates=plasma.rates + (ke0,), label_tmax_h=None, ke0_per_h=ke0)


def ke0_from_tpeak(plasma, tpeak_h):
    """The ke0 at which the effect site of `plasma` peaks at `tpeak_h`."""
    return brentq(lambda k: peak_time(effect_site(plasma, k)) - tpeak_h, 1e-2, 1e4)


def peak_normalized(curve):
    peak = float(curve(peak_time(curve)))
    return replace(curve, coefs=tuple(c / peak for c in curve.coefs))


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

PLASMA = {
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

#: Minutes from an IV bolus to the peak effect-site concentration, with source.
#: stanpumpR (Shafer) solves its ke0 from these against its own plasma models;
#: `KE0_PER_H` solves against `PLASMA`, so the peak time carries over and ke0
#: does not.
EFFECT_TPEAK_MIN = {
    'FENTANYL': (3.6, 'stanpumpR drugs_fentanyl.R tPeak (Shafer & Varvel 1991, '
                      'EEG)'),
    'HYDROMORPHONE': (19.6, 'stanpumpR drugs_hydromorphone.R tPeak'),
    'MORPHINE': (93.8, 'stanpumpR drugs_morphine.R tPeak (t1/2 ke0 ~138 min; '
                       'Lotsch et al. 2001, pupil: 2.8 h)'),
}

KE0_PER_H = {ingredient: ke0_from_tpeak(PLASMA[(ingredient, 'IV')], minutes / 60.0)
             for ingredient, (minutes, _) in EFFECT_TPEAK_MIN.items()}


def _unit_dose(key, plasma):
    ingredient = key[0]
    if ingredient not in KE0_PER_H:
        return peak_normalized(plasma)
    curve = effect_site(plasma, KE0_PER_H[ingredient])
    return peak_normalized(replace(curve, source=(
        f'{plasma.source}; effect site, IV tPeak '
        f'{EFFECT_TPEAK_MIN[ingredient][0]:g} min ({EFFECT_TPEAK_MIN[ingredient][1]})')))


UNIT_DOSE = {key: _unit_dose(key, plasma) for key, plasma in PLASMA.items()}


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
#
# mme2 (2026-10-19): mme1, but C at an assessment = mean curve over the neural
# epoch the pair's d_z uses for it, so a second epoch moved to
# [t2 - shift - 5, t2 - shift) (`pain_change --second-epoch-shift-min`) moves C_2
# with it. The window itself is the run's data token (`e2pre10to5`). Equal to mme1
# when no epoch is shifted, so an unshifted run keeps the mme1 name.
#
# mme3 (2026-10-21): mme1 on the effect-site, peak-normalized `UNIT_DOSE` curves
# (module docstring). mme1 and mme2 used the plasma curves, IV bolus at g(0) = 1.
EXPOSURE_VERSION = 'mme3'
#
# mme4 (2026-10-21): mme3 with mme2's rule, C at an assessment = mean curve over the
# epoch its d_z uses: a shifted second epoch moves C_2, a post-report first epoch
# (`pain_change --first-epoch-post`, [t1, t1 + 5)) moves C_1.
SHIFTED_EXPOSURE_VERSION = 'mme4'


def exposure_version(epochs_moved):
    """The version a pain_change run's exposure follows: mme4 when an epoch moved."""
    return SHIFTED_EXPOSURE_VERSION if epochs_moved else EXPOSURE_VERSION

#: One typical dose, the divisor that puts exposure in typical-dose units: the
#: median MME over every opioid dose of the 63 discovery subjects (624 doses,
#: 54 subjects), measured 2026-10-15. The median of subject medians is 7.5.
TYPICAL_DOSE_MME = 10.0
