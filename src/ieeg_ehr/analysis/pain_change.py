"""pain_change: does the change in z-scored power track the change in pain?

    python -m ieeg_ehr.analysis.pain_change --model opioid --roi-scheme roi_v2_ofc_ins \\
        --exclude-first-h 24 --insula-from-run <earlier pain_change run> \\
        --mask-label std10_rv-gross-std3_satmargin15_sw_logz4

Input is the `fullresz` z-score table (`views/build_pain_epoch_fullres_zscore.py`):
per subject-session, channel and native 0.5 Hz frequency, the epoch mean of
per-window z-scores against all of that session's pain-epoch windows.

Pairs are CONSECUTIVE assessments within a session, built by
`change_score.build_pairs` with its gap window (30-240 min by default). Per pair
x channel x frequency

    d_z = zmean(e2) - zmean(e1)

then the mean d_z over the native frequencies inside each band of
`--band-set` (default `paper_bands_6_hg200`, the six bands of
`run_bandpower_mixed.BAND_SETS`), arithmetic because d_z is a difference
(`axes.aggregate_bands`), line-noise bins removed first. `--band-set fullres`
keeps every native frequency as its own cell.

Every pair also carries the opioid exposure of `config.opioid_mme.exposure_version`
in typical doses (`add_exposure`): C at each assessment is the mean active-MME
curve over the epoch its d_z uses, `dC = C_2 - C_1`, and `Cbar` is the mean
curve over [t1, t2), centred on the cohort mean over pairs. Sessions with no MAR
export are dropped for every model, so all models of one level-4 folder fit the
same rows. `--exclude-first-h` drops pairs whose first epoch starts within that
many hours of `session_start`.

`--second-epoch-shift-min S` takes each pair's LATER epoch from the shifted tables
(`features/build_pain_change_shifted_z.py`), [t2 - S - 5, t2 - S) min, while the
earlier epoch stays [t1 - 5, t1). Consecutive pairs then share no neural window.
The data token gains `e2pre{S+5}to{S}` and the exposure version is
`opioid_mme.SHIFTED_EXPOSURE_VERSION`.

`--first-epoch-post` takes each pair's EARLIER epoch from the 5 min after its
report, [t1, t1 + 5), and the later from [t2 - 5, t2), both z-scored against the
AXIS 2 `post_report_epochs` baseline (`features/build_pain_change_post_z.py`).
Consecutive pairs share no neural window. The data token is
`zscorepostep-...-e1post0to5`, and C_1 and the autonomic epoch 1 move to
[t1, t1 + 5) with the neural epoch (`opioid_mme.SHIFTED_EXPOSURE_VERSION`).

One mixed model per ROI x band cell, rows = pair x channel, fixed and random
effects from `MODELS[--model]`. Every model has `RE_SUBJECT`,
`(1 | subject) + (0 + d_pain | subject)`, uncorrelated, and no channel component (a
channel's level cancels in the difference, see mixed_model.py). `opioid-pairint`
adds `(1 | pair_id)`, a shift each pair's channels share, which only lme4 fits. `pain_1_within`
is `pain_1` centred within subject over the cell's rows. The LRT on the random
`d_pain` slope uses `VC_CHANGE_REDUCED`. Every cell gets one `fit_status`
(`FIT_STATUSES`), and the run logs and writes the count per status.

BH FDR across all fitted cells and within each region, per term of `FDR_TERMS`
(`add_fdr`). Discovery cohort only. Output is NOMINATIONS, not findings.

Each run lands in
`analysis/pain/pain_change/bandpower/<data token>/<run token>_<timestamp>/`
(`fullres/` for `--band-set fullres`), names from docs/run_tokens.md:
`cells.csv`, `pair_index.csv`, `METHODS.md`, `provenance.json`, and every figure
in `figures/`.
"""

import argparse
import dataclasses
import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import (change_score, fullres_cells, med_state,
                               mixed_model as mm, view_tables)
from ieeg_ehr.analysis.run_bandpower_mixed import BAND_SETS
from ieeg_ehr.analysis.run_domain_lmer import check_r, r_environment
from ieeg_ehr.analysis.run_mixed_model_grid import FDR_Q, add_fdr
from ieeg_ehr.analysis.run_mixed_model_pilot import roi_maps
from ieeg_ehr.config import opioid_mme, roi_schemes
from ieeg_ehr.med_analysis import load, mme_curve
from ieeg_ehr.views import axes, fullres_reader, view_config
from ieeg_ehr.views import build_pain_epoch_fullres_zscore as zview

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/pain_change.py'
#: Level-3 folder per frequency axis. Figures go in each run's FIGURES_SUBDIR.
OUTPUT_TYPE = {'bands': 'bandpower', 'fullres': 'fullres'}
FIGURES_SUBDIR = 'figures'

#: The subject random effects of every model, `mixed_model.VC_CHANGE` in lme4 syntax.
RE_SUBJECT = '(1 | subject) + (0 + d_pain | subject)'


@dataclasses.dataclass(frozen=True)
class Model:
    """Fixed effects and random effects of one model ID."""
    fixed: str
    random: str = RE_SUBJECT

    @property
    def formula(self):
        return f'{self.fixed} + {self.random}'


_PAINCHANGE = 'd_z ~ d_pain + pain_1_within + gap_h'
_OPIOID = 'd_z ~ d_pain + dC + Cbar + d_pain:Cbar + pain_1_within + gap_h'
#: ΔC in place of Cbar, in the main effects and the interaction. dC is not centred:
#: 0 already means no exposure change, so `d_pain` is the slope at steady exposure.
_OPIOID_DC = 'd_z ~ d_pain + dC + d_pain:dC + pain_1_within + gap_h'
#: `_OPIOID_DC` plus one autonomic change between the pair's two epochs, the
#: autonomic-confound checks (`add_autonomic`). Neither is centred: 0 means no change.
_OPIOID_DC_DHR = 'd_z ~ d_pain + dC + d_pain:dC + dHR + pain_1_within + gap_h'
_OPIOID_DC_DHRV = 'd_z ~ d_pain + dC + d_pain:dC + dHRV + pain_1_within + gap_h'
_PAIR =f'{RE_SUBJECT} + (1 | pair_id)'
#: `_PAIR` without the subject intercept. d_z is a within-subject difference of two
#: epochs z-scored against the same session baseline, so a subject offset cancels
#: and `(1 | pair_id)` takes up what is left: the intercept variance fitted to zero
#: in every cell of the 2026-10-15 `opioid-dc-pairint` run.
_PAIR_NOSUBJINT = '(0 + d_pain | subject) + (1 | pair_id)'
#: Model ID -> model (docs/run_tokens.md). One ID always means one formula; a new
#: formula takes a new ID. `(1 | pair_id)` gives each pair one shift shared by its
#: channels, so pair-level predictors are tested against the number of pairs, not
#: of channel rows. pair_id is unique across subjects, so it is `(1 | subject:pair)`.
#: Only lme4 fits a random effect other than RE_SUBJECT.
MODELS = {
    'painchange': Model(_PAINCHANGE),
    'painchange-pairint': Model(_PAINCHANGE, _PAIR),
    'opioid': Model(_OPIOID),
    'opioid-pairint': Model(_OPIOID, _PAIR),
    'opioid-dc-pairint': Model(_OPIOID_DC, _PAIR),
    'opioid-dc-pairint-nosubjint': Model(_OPIOID_DC, _PAIR_NOSUBJINT),
    'opioid-dc-pairint-nosubjint-dhr': Model(_OPIOID_DC_DHR, _PAIR_NOSUBJINT),
    'opioid-dc-pairint-nosubjint-dhrv': Model(_OPIOID_DC_DHRV, _PAIR_NOSUBJINT),
}
#: Fixed-effect term -> column prefix in cells.csv.
TERM_PREFIX = {'d_pain': 'dpain', 'pain_1_within': 'baseline', 'gap_h': 'gap',
               'dC': 'dc', 'Cbar': 'cbar', 'd_pain:Cbar': 'dpain_x_cbar',
               'd_pain:dC': 'dpain_x_dc', 'dHR': 'dhr', 'dHRV': 'dhrv'}
#: Terms that each get their own BH family.
FDR_TERMS = ('d_pain', 'd_pain:Cbar', 'd_pain:dC', 'dC', 'Cbar', 'dHR', 'dHRV')
EXPOSURE_TERMS = ('dC', 'Cbar', 'd_pain:Cbar', 'd_pain:dC')
AUTONOMIC_TERMS = ('dHR', 'dHRV')
#: Runs fitted on the `--autonomic` rows sit in this folder under their data token.
AUTONOMIC_SUBDIR = 'autonomics'
#: Heart rate and HRV of a pair epoch (`add_autonomic`), from preprocessing/ekg_hr.py.
#: Only seconds and beats with template-match `quality` >= HR_MIN_QUALITY count, the
#: quality the extractor picks its lead by (a lead with no ECG scored a median 0.19,
#: clean leads 0.92-0.99). Either measure exists only when what it uses covers at
#: least HR_MIN_COVERAGE of the epoch.
HR_MIN_QUALITY = 0.8
HR_MIN_COVERAGE = 0.5
#: NN-interval filters for RMSSD: an interval outside [NN_MIN_S, NN_MAX_S] (30-200
#: bpm), or a successive difference above NN_MAX_JUMP of the interval before it
#: (the 20% rule), is an artifact and is skipped.
NN_MIN_S = 0.3
NN_MAX_S = 2.0
NN_MAX_JUMP = 0.2
#: Fitting-engine tokens (docs/run_tokens.md): statsmodels `MixedLM`, `lme4::lmer`.
ENGINES = ('sm', 'lme4')
LME4_SCRIPT = Path(__file__).with_name('pain_change_lmer.R')
LME4_OPTIMIZER = 'bobyqa'
#: Columns the lme4 frame carries besides cell_id, subject and d_z. dHR and dHRV
#: only under --autonomic.
LME4_COLUMNS = ['pair_id', 'd_pain', 'dC', 'Cbar', 'dHR', 'dHRV', 'pain_1_within', 'gap_h']
#: Pair columns a cell frame carries, of those the pairs have.
PAIR_COLUMNS = ['pair_id', 'd_pain', 'pain_1', 'gap_h', 'dC', 'Cbar', 'dHR', 'dHRV']

#: One per cell, in report order. `boundary`: a random-effect variance
#: contributes under mm.BOUNDARY_TOL of the residual variance (`boundary_component`
#: names it); the fixed effects still hold. `warn`: converged with a SERIOUS_WARNING,
#: so the standard errors are suspect. `failed`: raised, non-finite, or not
#: converged. `floor`: below the coverage floor, never fitted.
FIT_STATUSES = ('ok', 'boundary', 'warn', 'failed', 'floor')
#: statsmodels messages that make a converged fit's standard errors suspect. Its
#: "MLE may be on the boundary" is NOT one: it fires on any variance component
#: under an absolute 0.01, which the d_pain slope variance (z per NRS point,
#: squared) always is. The relative check of `fit_status` replaces it.
SERIOUS_WARNINGS = ('Hessian matrix', 'covariance is singular',           # statsmodels
                    'Hessian is numerically singular', 'nearly unidentifiable')  # lme4


def model_terms(model):
    """The fixed-effect terms of `model`, in formula order."""
    return [t.strip() for t in MODELS[model].fixed.split('~')[1].split('+')]


def uses_exposure(model):
    return any(t in EXPOSURE_TERMS for t in model_terms(model))


def uses_autonomic(model):
    return any(t in AUTONOMIC_TERMS for t in model_terms(model))


def data_token(vc, band_set, exclude_first_h, shift_min=0, first_post=False):
    """The level-4 folder: which rows go in and what their d_z is (docs/run_tokens.md)."""
    norm = (view_config.NORMALIZATION_CODES[vc.normalization]
            + view_config.BASELINE_CODES[vc.baseline])
    tokens = [norm, band_set.replace('_', ''), view_config.roi_code(vc.roi_scheme)]
    if exclude_first_h:
        tokens.append(f'nofirst{exclude_first_h:g}h')
    if shift_min:
        from ieeg_ehr.features.build_pain_change_shifted_z import window_label
        tokens.append(f'e2{window_label(shift_min)}')
    if first_post:
        from ieeg_ehr.features.build_pain_change_post_z import WINDOW_LABELS
        tokens.append(f"e1{WINDOW_LABELS['post']}")
    return '-'.join(tokens)


def run_token(model, engine, shift_min=0, first_post=False):
    """The run name before its timestamp: model, exposure version if used, engine."""
    exposure = ([opioid_mme.exposure_version(bool(shift_min) or first_post)]
                if uses_exposure(model) else [])
    return '_'.join([model, *exposure, engine])


def zscore_dirs(vc, shift_min=0, first_post=False):
    """(tables the rows come from, earlier-epoch tables, later-epoch tables).

    The second and third are None when the first holds that epoch.
    """
    if first_post:
        from ieeg_ehr.features.build_pain_change_post_z import post_zscore_dir
        return post_zscore_dir(vc, 'pre'), post_zscore_dir(vc, 'post'), None
    if shift_min:
        from ieeg_ehr.features.build_pain_change_shifted_z import shifted_zscore_dir
        return zview.zscore_dir(vc), None, shifted_zscore_dir(vc, shift_min)
    return zview.zscore_dir(vc), None, None


def discovery_paths(zdir):
    """z-score tables for discovery subjects only; the rest are logged and dropped."""
    discovery = set(config.discovery_subjects())
    paths, outside = [], []
    for p in sorted(zdir.glob('zmean_sub-*_ses-*.parquet')):
        subject, _ = fullres_cells.subject_session_of(p)
        (paths if subject in discovery else outside).append(p)
    if outside:
        logger.warning('SPLIT GATE: %d subject-session(s) outside discovery '
                       'excluded: %s', len(outside), [p.name for p in outside])
    if not paths:
        raise SystemExit(f'no discovery zmean_sub-*.parquet in {zdir}')
    return paths


def load_exposure():
    """(session_start per (subject, session), opioid doses in MME) from the MAR export."""
    admin = load.load_administrations()
    return (admin.groupby(['subject', 'session'])['session_start'].first(),
            mme_curve.opioid_doses(admin))


def add_exposure(pairs, exclude_first_h=None, shift_min=0, first_post=False):
    """(pairs with opioid exposure, report). Drops no-MAR sessions and early pairs.

    `C_1`, `C_2`: mean active MME over each assessment's epoch: [t - 5, t) min, the
    second moved to [t2 - shift_min - 5, t2 - shift_min) when `shift_min` is set,
    the first moved to [t1, t1 + 5) when `first_post` is set.
    `dC = C_2 - C_1`. `Cbar`: mean over [t1, t2), NOT centred here (`load_rows`
    centres it on the cohort). All in typical doses (`TYPICAL_DOSE_MME`).
    `e1_start_h`: hours from `session_start` to the start of the first epoch.

    A session with no MAR export has unknown exposure, not zero, so its pairs go.
    A session with an export and no opioid dose has C = 0 throughout.
    """
    starts, doses = load_exposure()
    has_mar = pd.MultiIndex.from_frame(pairs[['subject', 'session']]).isin(starts.index)
    no_mar = sorted(pairs.loc[~has_mar, 'subject_id'].unique())
    if no_mar:
        logger.warning('NO MAR EXPORT: dropping %d pairs of %s', int((~has_mar).sum()), no_mar)
    pairs = pairs[has_mar].reset_index(drop=True)

    start = starts.reindex(pd.MultiIndex.from_frame(pairs[['subject', 'session']])).to_numpy()
    t1_h = (pairs['t1'].to_numpy() - start) / np.timedelta64(1, 'h')
    t2_h = (pairs['t2'].to_numpy() - start) / np.timedelta64(1, 'h')
    epoch_h = config.EPOCH_MINUTES_BEFORE / 60.0
    shift_h = shift_min / 60.0
    e1_lo_h = t1_h if first_post else t1_h - epoch_h
    c1, c2, cbar = (np.zeros(len(pairs)) for _ in range(3))
    for (subject, session), idx in pairs.groupby(['subject', 'session']).indices.items():
        d = doses[(doses['subject'] == subject) & (doses['session'] == session)]
        c1[idx] = mme_curve.window_mean(d, e1_lo_h[idx], e1_lo_h[idx] + epoch_h)
        c2[idx] = mme_curve.window_mean(d, t2_h[idx] - shift_h - epoch_h,
                                        t2_h[idx] - shift_h)
        cbar[idx] = mme_curve.window_mean(d, t1_h[idx], t2_h[idx])
    pairs['e1_start_h'] = e1_lo_h
    pairs['C_1'] = c1 / opioid_mme.TYPICAL_DOSE_MME
    pairs['C_2'] = c2 / opioid_mme.TYPICAL_DOSE_MME
    pairs['dC'] = pairs['C_2'] - pairs['C_1']
    pairs['Cbar'] = cbar / opioid_mme.TYPICAL_DOSE_MME

    report = {'no_mar_subjects': no_mar, 'n_pairs_no_mar': int((~has_mar).sum()),
              'exclude_first_h': exclude_first_h, 'n_pairs_early': 0}
    if exclude_first_h:
        early = pairs['e1_start_h'] < exclude_first_h
        logger.info('FIRST %g H: dropping %d of %d pairs (%d subjects lose all their pairs)',
                    exclude_first_h, int(early.sum()), len(pairs),
                    len(set(pairs['subject']) - set(pairs.loc[~early, 'subject'])))
        report['n_pairs_early'] = int(early.sum())
        pairs = pairs[~early].reset_index(drop=True)
    return pairs, report


def window_sums(t, values, starts, stops):
    """(count, column sums of `values`) of the rows with `t` in each [start, stop). `t` sorted."""
    lo, hi = np.searchsorted(t, starts), np.searchsorted(t, stops)
    csum = np.vstack([np.zeros(values.shape[1]), np.cumsum(values, axis=0)])
    return hi - lo, csum[hi] - csum[lo]


def successive_differences(beats):
    """(time, squared successive NN difference in ms^2, NN in s) of every usable pair of
    adjacent NN intervals in one session's R-peak table.

    An NN interval needs both its beats at quality >= HR_MIN_QUALITY and a length in
    [NN_MIN_S, NN_MAX_S]. A difference needs both its intervals and a jump of at most
    NN_MAX_JUMP. Beats pair only within a run. Stamped at the later beat.
    """
    parts = []
    for _, b in beats.sort_values('time').groupby('run', sort=False):
        t = b['time'].to_numpy()
        good = b['quality'].to_numpy() >= HR_MIN_QUALITY
        nn = np.diff(t) / np.timedelta64(1, 's')
        ok = good[1:] & good[:-1] & (nn >= NN_MIN_S) & (nn <= NN_MAX_S)
        d = np.diff(nn)
        keep = ok[1:] & ok[:-1] & (np.abs(d) <= NN_MAX_JUMP * nn[:-1])
        parts.append(pd.DataFrame({'time': t[2:][keep], 'd2_ms2': (1000 * d[keep]) ** 2,
                                   'nn_s': nn[1:][keep]}))
    return pd.concat(parts, ignore_index=True).sort_values('time')


def epoch_autonomic(hr, diffs, starts, stops):
    """(mean HR in bpm, ln RMSSD in ms) over each [start, stop), NaN where either
    covers under HR_MIN_COVERAGE of the epoch.

    `hr`: the session's per-second table cut to quality >= HR_MIN_QUALITY, sorted by
    time; a second belongs to the epoch its start falls in. `diffs`:
    `successive_differences` of the session's R-peaks; coverage is the summed length
    of the NN intervals used.
    """
    seconds = (stops - starts) / np.timedelta64(1, 's')
    n_hr, s_hr = window_sums(hr['time'].to_numpy(), hr[['hr_bpm']].to_numpy(np.float64),
                             starts, stops)
    n_d, s_d = window_sums(diffs['time'].to_numpy(),
                           diffs[['d2_ms2', 'nn_s']].to_numpy(np.float64), starts, stops)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_hr = np.where(n_hr / seconds >= HR_MIN_COVERAGE, s_hr[:, 0] / n_hr, np.nan)
        ln_rmssd = np.where(s_d[:, 1] / seconds >= HR_MIN_COVERAGE,
                            0.5 * np.log(s_d[:, 0] / n_d), np.nan)
    return mean_hr, ln_rmssd


def add_autonomic(pairs, shift_min=0, first_post=False):
    """(pairs with HR and HRV, report). Drops pairs that lack either in either epoch.

    Over each epoch of d_z, [t1 - 5, t1) ([t1, t1 + 5) with `first_post`) and
    [t2 - shift_min - 5, t2 - shift_min) min (`epoch_autonomic`): `HR_1`, `HR_2` the mean bpm, `lnRMSSD_1`, `lnRMSSD_2` the
    natural log of RMSSD in ms. `dHR = HR_2 - HR_1`, `dHRV = lnRMSSD_2 - lnRMSSD_1`.
    A session with no HR table has neither.
    """
    epoch = pd.Timedelta(minutes=config.EPOCH_MINUTES_BEFORE).to_timedelta64()
    shift = pd.Timedelta(minutes=shift_min).to_timedelta64()
    t1_all = (pairs['t1'].to_numpy(dtype='datetime64[ns]')
              + (epoch if first_post else np.timedelta64(0, 'ns')))
    t2_all = pairs['t2'].to_numpy(dtype='datetime64[ns]') - shift
    cols = {c: np.full(len(pairs), np.nan)
            for c in ('HR_1', 'HR_2', 'lnRMSSD_1', 'lnRMSSD_2')}
    no_table = []
    for (subject, session), idx in pairs.groupby(['subject', 'session']).indices.items():
        hr_path = config.ekg_hr_path(subject, session)
        if not hr_path.exists():
            no_table.append(f'sub-{subject}_ses-{session}')
            continue
        hr = io.read_table(hr_path, columns=['time', 'hr_bpm', 'quality'], on_stale='warn')
        hr = hr[hr['quality'] >= HR_MIN_QUALITY].sort_values('time')
        diffs = successive_differences(io.read_table(
            config.ekg_rpeaks_path(subject, session), columns=['run', 'time', 'quality'],
            on_stale='warn'))
        for k, t in (('1', t1_all[idx]), ('2', t2_all[idx])):
            cols[f'HR_{k}'][idx], cols[f'lnRMSSD_{k}'][idx] = epoch_autonomic(
                hr, diffs, t - epoch, t)
    pairs = pairs.assign(**cols)
    pairs['dHR'] = pairs['HR_2'] - pairs['HR_1']
    pairs['dHRV'] = pairs['lnRMSSD_2'] - pairs['lnRMSSD_1']

    no_hr, no_hrv = pairs['dHR'].isna(), pairs['dHRV'].isna()
    missing = no_hr | no_hrv
    lost = sorted(set(pairs['subject_id']) - set(pairs.loc[~missing, 'subject_id']))
    if no_table:
        logger.warning('NO HR TABLE: %s', no_table)
    logger.info('AUTONOMIC: dropping %d of %d pairs (%d without HR, %d without HRV in both '
                'epochs); %d subjects lose all their pairs: %s', int(missing.sum()),
                len(pairs), int(no_hr.sum()), int(no_hrv.sum()), len(lost), lost)
    report = {'min_quality': HR_MIN_QUALITY, 'min_coverage': HR_MIN_COVERAGE,
              'nn_range_s': [NN_MIN_S, NN_MAX_S], 'nn_max_jump': NN_MAX_JUMP,
              'hr_root': str(config.EKG_HR_ROOT), 'sessions_without_table': no_table,
              'n_pairs_dropped': int(missing.sum()), 'n_pairs_without_hr': int(no_hr.sum()),
              'n_pairs_without_hrv': int(no_hrv.sum()), 'subjects_without_autonomic': lost}
    return pairs[~missing].reset_index(drop=True), report


def change_matrix(table, pairs, freq_cols, second=None, first=None):
    """{channel: (n_pairs, n_freq) d_z}, NaN where the channel is absent from either epoch.

    `second`, a shifted table in the same schema, gives the later epoch's z, and
    `first`, a post-report table, the earlier epoch's.
    """
    table = table.set_index(['channel', 'epoch_id'])[freq_cols]
    later = None if second is None else second.set_index(['channel', 'epoch_id'])[freq_cols]
    earlier = None if first is None else first.set_index(['channel', 'epoch_id'])[freq_cols]
    e1, e2 = pairs['e1'].to_numpy(), pairs['e2'].to_numpy()
    out = {}
    for channel, sub in table.groupby(level='channel', sort=True):
        sub = sub.droplevel('channel')
        z2 = (sub.reindex(e2) if later is None
              else later.reindex(pd.MultiIndex.from_product([[channel], e2])))
        z1 = (sub.reindex(e1) if earlier is None
              else earlier.reindex(pd.MultiIndex.from_product([[channel], e1])))
        out[channel] = (z2.to_numpy(dtype=config.CACHE_ACCUMULATE_DTYPE)
                        - z1.to_numpy(dtype=config.CACHE_ACCUMULATE_DTYPE))
    return out


def session_rows(path, pairs, region_of, freq_table, bands, second_path=None,
                 first_path=None):
    """(index frame, (n_rows, n_cols) d_z) for one subject-session, ROI-mapped channels only.

    `second_path`: the shifted table the later epoch comes from, `first_path` the
    post-report table the earlier epoch comes from (`change_matrix`).
    """
    subject, session = fullres_cells.subject_session_of(path)
    all_cols = fullres_reader.freq_columns()
    freq_cols = [all_cols[i] for i in freq_table.index]
    table = io.read_table(path, columns=['epoch_id', 'channel'] + freq_cols,
                          on_stale='warn')
    second, first = (None if p is None else
                     io.read_table(p, columns=['epoch_id', 'channel'] + freq_cols,
                                   on_stale='warn')
                     for p in (second_path, first_path))
    bin_table = freq_table.reset_index()
    index, values = [], []
    for channel, dz in change_matrix(table, pairs, freq_cols, second, first).items():
        region = region_of.get(channel)
        if region is None:
            continue
        if bands is not None:
            dz, _ = axes.aggregate_bands(dz, bin_table, bands=bands, is_difference=True)
        index.append(pd.DataFrame({
            'subject': f'sub-{subject}',
            'channel_uid': f'sub-{subject}|ses-{session}|{channel}',
            'pair_id': pairs['pair_id'].to_numpy(), 'region': region}))
        values.append(dz)
    if not index:
        return None, None
    return pd.concat(index, ignore_index=True), np.vstack(values)


def cell_frame(index, values, j, pairs):
    """One cell's model frame: rows with a finite d_z, covariates joined by pair."""
    df = index.assign(d_z=values[:, j])
    df = df[np.isfinite(df['d_z'])]
    df = df.merge(pairs[[c for c in PAIR_COLUMNS if c in pairs]], on='pair_id')
    df['pain_1_within'] = df['pain_1'] - df.groupby('subject')['pain_1'].transform('mean')
    return df.reset_index(drop=True)


def fit_status(converged, ratios, warnings):
    """(status, boundary component) for a cell that reached the fitter.

    `ratios` maps each variance component to its contribution over the residual
    variance.
    """
    if not converged:
        return 'failed', ''
    low = [name for name, r in ratios.items() if r < mm.BOUNDARY_TOL]
    if low:
        return 'boundary', ','.join(low)
    serious = any(s in w for w in warnings for s in SERIOUS_WARNINGS)
    return ('warn' if serious else 'ok'), ''


def cell_base(df, meta):
    """A cell's identity and counts, the start of its results row."""
    return {**meta, 'n_subjects': df['subject'].nunique(),
            'n_channels': df['channel_uid'].nunique(),
            'n_pairs': df['pair_id'].nunique(), 'n_rows': len(df)}


def floor_record(df, base, min_subjects):
    """The results row of a cell below the coverage floor, else None."""
    if (len(df) < mm.MIN_ROWS or base['n_subjects'] < min_subjects
            or df['d_pain'].std(ddof=0) == 0):
        return {**base, 'converged': False, 'fit_status': 'floor',
                'error': f'below floor: {base["n_rows"]} rows, '
                         f'{base["n_subjects"]} subjects (min {min_subjects})'}
    return None


def variance_ratios(var_int, var_dpain, scale, df, var_pair=None):
    """Each random-effect variance's contribution to the linear predictor over the residual."""
    ratios = {'subj_int': var_int / scale,
              'subj_dpain': var_dpain * float(df['d_pain'].var(ddof=0)) / scale}
    if var_pair is not None:
        ratios['pair'] = var_pair / scale
    return ratios


def fit_cell_record(df, meta, min_subjects, model, with_lrt=True):
    """One row of results for one ROI x frequency cell. `with_lrt=False` skips the reduced fit."""
    t0 = time.time()
    if MODELS[model].random != RE_SUBJECT:
        raise ValueError(f'model {model} has random effects only the lme4 engine fits')
    formula = MODELS[model].fixed
    base = cell_base(df, meta)
    floor = floor_record(df, base, min_subjects)
    if floor:
        return floor
    try:
        res, warn = mm.fit_cell(df, mm.VC_CHANGE, formula=formula)
    except mm.CellFitError as exc:
        return {**base, 'converged': False, 'fit_status': 'failed', 'error': str(exc)[:200]}
    stat, p_lrt = np.nan, np.nan
    if with_lrt:
        try:
            res_red, _ = mm.fit_cell(df, mm.VC_CHANGE_REDUCED, formula=formula)
            stat, p_lrt = mm.lrt(res, res_red)
        except mm.CellFitError:
            pass

    rec = dict(base)
    for term in model_terms(model):
        rec.update(mm.term_stats(res, term, TERM_PREFIX[term]))
    vc = mm.vcomp_by_name(res)
    var_int, var_dpain = float(vc['subj_int']), float(vc['subj_dpain'])
    scale = float(res.scale)
    ratios = variance_ratios(var_int, var_dpain, scale, df)
    status, boundary = fit_status(bool(res.converged), ratios, warn)
    rec.update({'var_subj_int': var_int, 'var_subj_dpain': var_dpain, 'var_resid': scale,
                'int_var_ratio': ratios['subj_int'], 'slope_var_ratio': ratios['subj_dpain'],
                'lrt_stat': float(stat), 'p_lrt_mixture': float(p_lrt),
                'converged': bool(res.converged), 'n_warnings': len(warn),
                'warnings': ' | '.join(warn), 'fit_status': status,
                'boundary_component': boundary,
                'error': '', 'fit_seconds': round(time.time() - t0, 1)})
    return rec


def fit_summary(cells):
    """Cells per fit status, with each fitted status's share of the cells above the floor."""
    counts = cells['fit_status'].value_counts().reindex(FIT_STATUSES, fill_value=0)
    fitted = int(counts.drop('floor').sum())
    pct = (100 * counts / fitted).round(1) if fitted else counts * np.nan
    return pd.DataFrame({'n_cells': counts, 'pct_of_fitted': pct.where(counts.index != 'floor')})


@dataclasses.dataclass
class Rows:
    """Every pair x channel d_z row of a fit, with what the cell fits need."""
    index: pd.DataFrame
    values: np.ndarray
    pairs: pd.DataFrame
    regions: list
    cols: list
    col_hz: dict
    bands: dict
    notch: list
    no_roi: list
    split_report: dict
    pair_report: dict
    cbar_center: float
    autonomic_report: dict = None


def load_rows(paths, roi_scheme, band_set, min_gap_min, max_gap_min,
              notch_half_width_hz=None, insula_threshold=None, exclude_first_h=None,
              shift_min=0, second_dir=None, autonomic=False, first_dir=None):
    """The d_z rows of the z-score tables at `paths`, ROI-mapped channels only.

    `insula_threshold` pins the coordinate insula split to an earlier run's cut.
    `shift_min` with `second_dir` (the shifted tables) takes the later epoch from
    [t2 - shift_min - 5, t2 - shift_min), and C_2 with it. `first_dir` (the
    post-report tables, with `paths` the pre-report tables on the same baseline)
    takes the earlier epoch from [t1, t1 + 5), and C_1 with it. `autonomic` adds dHR
    and dHRV and keeps only the pairs that have both (`add_autonomic`).
    """
    if bool(shift_min) != (second_dir is not None):
        raise ValueError('shift_min and second_dir go together')
    if shift_min and first_dir is not None:
        raise ValueError('a shifted second epoch and a post-report first epoch do not combine')
    first_post = first_dir is not None
    subjects = {f'sub-{fullres_cells.subject_session_of(p)[0]}' for p in paths}
    split_report = {}
    roi_by_subject, no_roi = roi_maps(paths, subjects, roi_scheme,
                                      insula_threshold=insula_threshold,
                                      report=split_report)
    regions = view_tables.roi_regions_for({'roi_scheme': roi_scheme})

    admin = med_state.load_admin_table(subclasses=med_state.DRUG_SETS['analgesics'])
    defs = med_state.load_epoch_defs(subjects=sorted(subjects))
    pairs, pair_report = add_exposure(
        change_score.build_pairs(defs, admin, min_gap_min, max_gap_min), exclude_first_h,
        shift_min, first_post)
    autonomic_report = None
    if autonomic:
        pairs, autonomic_report = add_autonomic(pairs, shift_min, first_post)

    freq_table, notch = fullres_cells.analysis_freq_table(notch_half_width_hz)
    bands = BAND_SETS.get(band_set)
    if bands is None:
        all_cols = fullres_reader.freq_columns()
        cols = [all_cols[i] for i in freq_table.index]
        col_hz = {c: (float(freq_table.at[i, 'bin_low_hz']),
                      float(freq_table.at[i, 'bin_high_hz']))
                  for c, i in zip(cols, freq_table.index)}
    else:
        cols, col_hz = list(bands), dict(bands)

    index_parts, value_parts = [], []
    for path in paths:
        subject, session = fullres_cells.subject_session_of(path)
        sid = f'sub-{subject}'
        sp = pairs[(pairs['subject'] == subject) & (pairs['session'] == session)]
        if sid not in roi_by_subject or sp.empty:
            logger.warning('%s ses-%s: %s, skipped', sid, session,
                           'no pairs' if sp.empty else 'no ROI-labelled channel')
            continue
        second_path, first_path = (None if d is None else Path(d) / path.name
                                   for d in (second_dir, first_dir))
        for p in (second_path, first_path):
            if p is not None and not p.exists():
                raise FileNotFoundError(f'no table {p}')
        idx, vals = session_rows(path, sp.reset_index(drop=True),
                                 roi_by_subject[sid], freq_table, bands, second_path,
                                 first_path)
        if idx is not None:
            index_parts.append(idx)
            value_parts.append(vals)
    index = pd.concat(index_parts, ignore_index=True)

    in_cohort = pairs['subject_id'].isin(index['subject'].unique())
    cbar_center = float(pairs.loc[in_cohort, 'Cbar'].mean())
    pairs['Cbar'] = pairs['Cbar'] - cbar_center
    logger.info('%d pair x channel rows, %d subjects, %d regions x %d %s cells; '
                'Cbar centred on %.4f typical doses', len(index), index['subject'].nunique(),
                len(regions), len(cols), band_set, cbar_center)
    return Rows(index, np.vstack(value_parts), pairs, regions, cols, col_hz, bands,
                notch, no_roi, split_report, pair_report, cbar_center, autonomic_report)


def cell_frames(rows, holdout=None):
    """(meta, model frame) per ROI x frequency cell, without `holdout`'s rows when given.

    A generator, so a caller builds each cell frame only when it needs it.
    """
    keep = (rows.index['subject'] != holdout).to_numpy()
    for region in rows.regions:
        in_region = (rows.index['region'] == region).to_numpy() & keep
        if not in_region.any():
            continue
        sub_index, sub_values = rows.index[in_region], rows.values[in_region]
        for j, col in enumerate(rows.cols):
            meta = {'region': region, 'freq': col, 'freq_low_hz': rows.col_hz[col][0],
                    'freq_high_hz': rows.col_hz[col][1]}
            if holdout is not None:
                meta['heldout_subject'] = holdout
            yield meta, cell_frame(sub_index, sub_values, j, rows.pairs)


def cell_jobs(rows, min_subjects, model, holdout=None, with_lrt=True):
    """One delayed statsmodels fit per ROI x frequency cell (`cell_frames`)."""
    from joblib import delayed
    for meta, df in cell_frames(rows, holdout):
        yield delayed(fit_cell_record)(df, meta, min_subjects, model, with_lrt)


def lme4_records(rows, min_subjects, model, n_jobs, holdout=None):
    """Every cell fitted by `lme4::lmer` (`pain_change_lmer.R`), in fit_cell_record's schema.

    The cell frames, floors, formula and random effects are the statsmodels path's,
    so the two engines differ only in the fitter. p-values are Wald z as in
    statsmodels. No LRT. `lme4_singular` is lme4's own `isSingular(tol = 1e-4)`.
    """
    from scipy import stats
    check_r()
    bases, records, frames = {}, [], []
    for k, (meta, df) in enumerate(cell_frames(rows, holdout)):
        base = cell_base(df, meta)
        floor = floor_record(df, base, min_subjects)
        if floor:
            records.append(floor)
            continue
        bases[k] = (base, df)
        frames.append(df.assign(cell_id=k))
    formula = MODELS[model].formula
    with tempfile.TemporaryDirectory(dir=os.environ.get('L_SCRATCH')) as tmp:
        csv = Path(tmp) / 'frames.csv'
        frame = pd.concat(frames)
        frame[['cell_id', 'subject', 'd_z'] + [c for c in LME4_COLUMNS if c in frame]].to_csv(
            csv, index=False)
        cmd = ['Rscript', str(LME4_SCRIPT), '--in', str(csv), '--out', tmp,
               '--formula', formula, '--cores', str(n_jobs), '--optimizer', LME4_OPTIMIZER]
        logger.info('lme4: %d cells, %s', len(frames), ' '.join(cmd))
        proc = subprocess.run(cmd, env=r_environment(), text=True, capture_output=True)
        for line in proc.stdout.splitlines():
            logger.info('[R] %s', line)
        if proc.returncode != 0:
            raise SystemExit(f'pain_change_lmer.R failed:\n{proc.stderr[-4000:]}')
        fits = pd.read_csv(Path(tmp) / 'lmer_cells.csv', keep_default_na=False,
                           na_values=['NA'])
        terms = pd.read_csv(Path(tmp) / 'lmer_terms.csv')

    terms = terms.groupby('cell_id')
    for fit in fits.itertuples():
        base, df = bases[fit.cell_id]
        rec = dict(base)
        warn = [w for w in str(fit.warnings).split(' | ') if w]
        if fit.error:
            records.append({**rec, 'converged': False, 'fit_status': 'failed',
                            'warnings': ' | '.join(warn), 'error': fit.error[:200]})
            continue
        t = terms.get_group(fit.cell_id).set_index('term')
        for term in model_terms(model):
            beta, se = float(t.at[term, 'beta']), float(t.at[term, 'se'])
            p = TERM_PREFIX[term]
            rec.update({f'{p}_beta': beta, f'{p}_se': se, f'{p}_z': beta / se,
                        f'{p}_p': float(2 * stats.norm.sf(abs(beta / se)))})
        has_pair = 'pair_id' in MODELS[model].random
        ratios = variance_ratios(fit.var_subj_int, fit.var_subj_dpain, fit.var_resid, df,
                                 fit.var_pair if has_pair else None)
        status, boundary = fit_status(bool(fit.converged), ratios, warn)
        rec.update({'var_subj_int': fit.var_subj_int, 'var_subj_dpain': fit.var_subj_dpain,
                    'var_resid': fit.var_resid, 'int_var_ratio': ratios['subj_int'],
                    'slope_var_ratio': ratios['subj_dpain'],
                    'var_pair': fit.var_pair if has_pair else np.nan,
                    'pair_var_ratio': ratios.get('pair', np.nan),
                    'converged': bool(fit.converged), 'lme4_singular': bool(fit.singular),
                    'n_warnings': len(warn), 'warnings': ' | '.join(warn),
                    'fit_status': status, 'boundary_component': boundary,
                    'error': '', 'fit_seconds': fit.fit_seconds})
        records.append(rec)
    return records


def random_effects_text(vc):
    """'(1 | subject) + (0 + d_pain | subject)' from a vc_formula dict, uncorrelated by construction."""
    return ' + '.join(f'({v} | subject)' for v in vc.values())


def fit_summary_text(summary):
    """The fit-status table as markdown."""
    lines = ['| Status | Cells | % of fitted |', '|---|---|---|']
    for status, r in summary.iterrows():
        pct = '' if pd.isna(r['pct_of_fitted']) else f"{r['pct_of_fitted']:g}"
        lines.append(f"| `{status}` | {int(r['n_cells'])} | {pct} |")
    return '\n'.join(lines)


def estimation_text(engine):
    """How `engine` fits a cell, for provenance and METHODS.md."""
    if engine == 'sm':
        fit = inspect.signature(mm.fit_cell).parameters
        return (f"statsmodels `MixedLM`, REML, maxiter={fit['maxiter'].default}, default "
                "optimizer, Wald z p-values, likelihood-ratio test of the random `d_pain` slope")
    return (f"`lme4::lmer`, REML, optimizer {LME4_OPTIMIZER} (maxfun 2e5), Wald z p-values "
            "as in statsmodels, no likelihood-ratio test. `lme4_singular` is lme4's "
            "`isSingular(tol = 1e-4)`")


def write_methods(run_dir, cells, params, n_pairs, cohort, summary):
    """METHODS.md built from the constants the fit used, so the text cannot drift from the code."""
    bands = params['bands']
    freq_text = (', '.join(f'{b} {lo}-{hi} Hz' for b, (lo, hi) in bands.items())
                 if bands else 'every native 0.5 Hz frequency as its own cell')
    model = params['model']
    reported = ', '.join(f'`{t}`' for t in model_terms(model))
    fdr = ', '.join(f'`{t}`' for t in FDR_TERMS if t in model_terms(model))
    pr = params['pair_report']
    pair_text = ("- `(1 | pair_id)`: one shift per pair, shared by its channels, so the\n"
                 "  pair-level predictors are tested against the number of pairs rather\n"
                 "  than of pair x channel rows (`var_pair`, `pair_var_ratio`).\n"
                 if 'pair_id' in MODELS[model].random else '')
    shift = params.get('second_epoch_shift_min') or 0
    exposure_vars = f"""- `dC`: C_2 - C_1, where C is the mean of the {params['exposure_version']}
  active-MME curve (`med_analysis/mme_curve.py`) over the assessment's epoch (the
  windows of `d_z` below), in typical doses of {params['typical_dose_mme']:g} MME (the median
  opioid dose of the discovery cohort, `config/opioid_mme.py`). Not centred.
""" if uses_exposure(model) else ''
    if 'Cbar' in model_terms(model):
        exposure_vars += f"""- `Cbar`: the mean of the same curve over [t1, t2), in typical doses, centred on
  the cohort mean over pairs ({params['cbar_center']:.4f} typical doses). The
  `d_pain` term is therefore the slope at average exposure.
"""
    elif 'd_pain:dC' in model_terms(model):
        exposure_vars += ("- `d_pain:dC`: the change in the pain slope per typical dose of exposure\n"
                          "  change. dC = 0 means steady exposure, so `d_pain` is the slope there.\n")
    au = params.get('autonomic')
    au_var = ''
    if 'dHR' in model_terms(model):
        au_var = (f"- `dHR`: HR_2 - HR_1 in bpm, HR the mean over the windows of `d_z` below of\n"
                  f"  the seconds of `preprocessing/ekg_hr.py` with template-match quality >=\n"
                  f"  {au['min_quality']:g}. Not centred: 0 means no change.\n")
    if 'dHRV' in model_terms(model):
        au_var = (f"- `dHRV`: ln RMSSD_2 - ln RMSSD_1, RMSSD in ms over the windows of `d_z`\n"
                  f"  below, from adjacent R peaks both at quality >= {au['min_quality']:g}, NN intervals in\n"
                  f"  {au['nn_range_s'][0]:g}-{au['nn_range_s'][1]:g} s, successive jumps under "
                  f"{au['nn_max_jump']:.0%}. Not centred.\n")
    au_rows = (f"Pairs without heart rate and HRV in both epochs (under "
               f"{au['min_coverage']:.0%} of an epoch covered, or no HR table) are dropped for "
               f"every model of `{AUTONOMIC_SUBDIR}/` ({au['n_pairs_dropped']} pairs). "
               if au else '')
    baseline_text = ('every masked window of every pain epoch in that session (AXIS 2 '
                     '`all_pain_epochs`)')
    if params.get('first_epoch_post'):
        dz_text = ('epoch-mean z over [t2 - 5, t2) min minus the epoch-mean z over '
                   '[t1, t1 + 5), t1 and t2 the two assessments.')
        baseline_text = ('every masked window of the 5 min after every pain report in that '
                         'session, [t, t + 5) (AXIS 2 `post_report_epochs`)')
        second_text = f", earlier epoch from `{params['first_zscore_dir']}`"
        shared_text = (
            '- Consecutive pairs share an assessment\'s pain score but no neural window:\n'
            '  the later epoch of one pair ([t - 5, t)) and the earlier epoch of the next\n'
            '  ([t, t + 5)) sit on either side of the same report.\n'
            '- The earlier epoch opens at the report: it holds the end of the bedside\n'
            '  visit and the first minutes of any dose charted at t1. C_1 covers the\n'
            '  same 5 min, so dC carries the drug part.')
    elif shift:
        dz_text = (f'epoch-mean z over [t2 - {shift + 5}, t2 - {shift}) min minus the '
                   'epoch-mean z over [t1 - 5, t1), t1 and t2 the two assessments.')
        second_text = f", later epoch from `{params['second_zscore_dir']}`"
        shared_text = (
            f'- Consecutive pairs share an assessment\'s pain score but no neural window:\n'
            f'  the later epoch of one pair ([t - {shift + 5}, t - {shift}) min) and the earlier\n'
            f'  epoch of the next ([t - 5, t)) are separate blocks of the same report.\n'
            f'- The two epochs of a pair sit at different distances from their report, so\n'
            f'  d_z also holds any difference between the 5 min just before an assessment\n'
            f'  and the 5 min {shift} min earlier (nurse arrival, waking, talking). It\n'
            f'  enters every pair alike unless it scales with pain.')
    else:
        dz_text = 'epoch-mean z of the later assessment minus the earlier one.'
        second_text = ''
        shared_text = ('- Consecutive pairs share an assessment, so neighbouring rows are '
                       'correlated\n  beyond what the subject random effects absorb.')
    subject_text = (
        'subject intercept and subject `d_pain` slope, uncorrelated (lme4 '
        f'`(d_pain || subject)`, statsmodels `vc_formula` `{mm.VC_CHANGE}`)'
        if RE_SUBJECT in MODELS[model].random else
        'subject `d_pain` slope only, no subject intercept: a subject offset cancels '
        'in the pair difference')
    medication = ("- Doses before `session_start` (the OR) are not in the MAR export, so\n"
                  "  exposure early in a session is a lower bound. Non-opioid analgesics\n"
                  "  are not modelled." if uses_exposure(model)
                  else '- No medication term: a dose between the two assessments is not modelled.')
    early = (f"Pairs whose first epoch starts within {pr['exclude_first_h']:g} h of "
             f"`session_start` are dropped ({pr['n_pairs_early']} pairs). "
             if pr['exclude_first_h'] else '')
    text = f"""# pain_change mixed-effects grid: `{params['run_name']}`

EXPLORATORY, discovery cohort. NOMINATIONS, NOT FINDINGS.

## Model

One linear mixed model per ROI x frequency cell. Rows are pair x channel.
Model `{model}`:

```
{MODELS[model].formula}
```

- Random effects: {subject_text}.
  No channel component: a channel's level cancels in the pair difference.
{pair_text}- Estimation (engine `{params['engine']}`): {params['estimation']}.
- Reported terms: {reported} and the variance components. The statsmodels LRT
  tests the random slope against `{random_effects_text(mm.VC_CHANGE_REDUCED)}` on the
  50:50 chi2(0)/chi2(1) mixture (`p_lrt_mixture`).
- Multiple comparisons: Benjamini-Hochberg at q={FDR_Q}, one family per term
  ({fdr}), across all fitted cells (`<term>_bh`) and within each region
  (`<term>_bh_within_region`).
- Inclusion floors per cell: at least {mm.MIN_ROWS} rows, at least
  {params['min_subjects']} subjects, and non-zero `d_pain` variance.

## Fit status

{fit_summary_text(summary)}

`boundary`: a subject random-effect variance (intercept or `d_pain` slope) contributes under
{mm.BOUNDARY_TOL:g} of the residual variance (`boundary_component`), so the fit is
in effect without that component and its fixed effects still hold. `warn`:
converged with a warning that makes the standard errors suspect
({', '.join(f'"{s}"' for s in SERIOUS_WARNINGS)}, column `warnings`). statsmodels'
"MLE may be on the boundary" does not count: it fires on any variance under an
absolute 0.01. `failed`: raised, non-finite, or not converged, drawn grey on the
heatmaps. `floor`: below the coverage floor, never fitted. Percentages are of the
cells above the floor.

## Variables

- `d_z`: {dz_text} Each 2 s
  window is z-scored per channel x frequency against the mean and SD over
  {baseline_text}, then averaged within the 5-min epoch. Tables: `{params['zscore_dir']}`{second_text}.
- Frequency cells (`{params['band_set']}`): {freq_text}. Bands average d_z arithmetically over the native
  frequencies inside them after removing {len(params['notch_bins_removed'])} line-noise bins.
- `d_pain`: NRS of the later assessment minus the earlier one.
{exposure_vars}{au_var}- `pain_1_within`: NRS of the earlier assessment, centred on the subject's mean
  over the cell's rows.
- `gap_h`: hours between the two assessments.

## Data

Consecutive assessment pairs within a session, {params['min_gap_min']:g}-{params['max_gap_min']:g} min apart
(`change_score.build_pairs`). Sessions with no MAR export are dropped
({pr['n_pairs_no_mar']} pairs of {', '.join(pr['no_mar_subjects']) or 'no subject'}). {early}{au_rows}ROI scheme
`{params['roi_scheme']}`, insula cut {params['insula_threshold']}. {len(cohort)} subjects,
{n_pairs} pairs, {len(cells)} cells.

## Limitations

{shared_text}
- Pairs with `d_pain == 0` inform the intercept and covariates only.
{medication}
- p-values are parametric (Wald z), not permutation-based.
"""
    (run_dir / 'METHODS.md').write_text(text)


def main(argv=None):
    from joblib import Parallel

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--engine', choices=ENGINES, default='sm')
    ap.add_argument('--exclude-first-h', type=float, default=None,
                    help='drop pairs whose first epoch starts within this many hours '
                         'of session_start')
    ap.add_argument('--insula-from-run', default=None,
                    help='pin the insula cut to this pain_change run\'s threshold')
    ap.add_argument('--min-gap-min', type=float, default=change_score.DEFAULT_MIN_GAP_MIN)
    ap.add_argument('--max-gap-min', type=float, default=change_score.DEFAULT_MAX_GAP_MIN)
    ap.add_argument('--min-subjects', type=int, default=mm.MIN_SUBJECTS)
    ap.add_argument('--second-epoch-shift-min', type=int, default=0,
                    help='take each pair\'s later epoch from [t2 - S - 5, t2 - S) min '
                         '(features/build_pain_change_shifted_z.py)')
    ap.add_argument('--first-epoch-post', action='store_true',
                    help='take each pair\'s earlier epoch from [t1, t1 + 5) min, both epochs '
                         'z-scored against the post-report baseline '
                         '(features/build_pain_change_post_z.py)')
    ap.add_argument('--autonomic', action='store_true',
                    help='add dHR and dHRV to the pairs, keep only pairs that have both '
                         f'(add_autonomic), and write under <data token>/{AUTONOMIC_SUBDIR}/')
    ap.add_argument('--notch-half-width-hz', type=float, default=None)
    ap.add_argument('--n-jobs', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    view_config.add_view_arguments(ap)
    ap.add_argument('--band-set', choices=list(BAND_SETS) + ['fullres'],
                    default='paper_bands_6_hg200')
    ap.set_defaults(baseline='all_pain_epochs', freq='fullres', region='none')
    args = ap.parse_args(argv)
    if args.engine == 'sm' and MODELS[args.model].random != RE_SUBJECT:
        ap.error(f'--model {args.model} needs --engine lme4')
    if uses_autonomic(args.model) and not args.autonomic:
        ap.error(f'--model {args.model} needs --autonomic')
    if args.first_epoch_post and args.second_epoch_shift_min:
        ap.error('--first-epoch-post and --second-epoch-shift-min do not combine')
    if args.first_epoch_post:
        args.baseline = 'post_report_epochs'

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    insula_threshold = None
    if args.insula_from_run and not roi_schemes.coordinate_regions(args.roi_scheme):
        ap.error(f'--insula-from-run: {args.roi_scheme} has no coordinate insula split')
    if args.insula_from_run:
        prov = json.loads((Path(args.insula_from_run) / 'provenance.json').read_text())
        insula_threshold = prov['insula_split']['thresholds']
    vc = view_config.from_args(args)
    shift, first_post = args.second_epoch_shift_min, args.first_epoch_post
    zdir, first_dir, second_dir = zscore_dirs(vc, shift, first_post)
    rows = load_rows(discovery_paths(zdir), args.roi_scheme, args.band_set,
                     args.min_gap_min, args.max_gap_min, args.notch_half_width_hz,
                     insula_threshold=insula_threshold,
                     exclude_first_h=args.exclude_first_h,
                     shift_min=shift, second_dir=second_dir,
                     autonomic=args.autonomic, first_dir=first_dir)
    index, pairs, bands = rows.index, rows.pairs, rows.bands
    logger.info('fitting model %s with %s, up to %d cells on %d worker(s)', args.model,
                args.engine, len(rows.regions) * len(rows.cols), args.n_jobs)
    if args.engine == 'sm':
        records = Parallel(n_jobs=args.n_jobs, verbose=5)(
            cell_jobs(rows, args.min_subjects, args.model))
    else:
        records = lme4_records(rows, args.min_subjects, args.model, args.n_jobs)
    cells = pd.DataFrame(records)
    # A failed fit is not a test: its p leaves the BH family (statsmodels still
    # returns estimates for a fit that did not converge).
    failed = cells['fit_status'].eq('failed')
    for term in FDR_TERMS:
        if term in model_terms(args.model):
            p_col = f'{TERM_PREFIX[term]}_p'
            cells = add_fdr(cells.assign(**{p_col: cells[p_col].mask(failed)}),
                            p_col, TERM_PREFIX[term])
    summary = fit_summary(cells)

    run_name = run_token(args.model, args.engine, shift, first_post)
    run_dir = config.analysis_run_dir(
        question=config.PAIN_CHANGE_QUESTION,
        output_type=OUTPUT_TYPE['fullres' if bands is None else 'bands'],
        view_scheme=data_token(vc, args.band_set, args.exclude_first_h, shift, first_post)
        + (f'/{AUTONOMIC_SUBDIR}' if args.autonomic else ''),
        run_name=run_name)
    cohort = sorted(index['subject'].unique())
    params = {'model': args.model, 'run_name': run_name, 'formula': MODELS[args.model].formula,
              'vc': mm.VC_CHANGE if RE_SUBJECT in MODELS[args.model].random else None,
              'engine': args.engine, 'reml': True,
              'estimation': estimation_text(args.engine), 'band_set': args.band_set,
              'bands': bands, 'roi_scheme': args.roi_scheme,
              'insula_threshold': rows.split_report.get('insula_split', {}).get('thresholds'),
              'insula_from_run': args.insula_from_run,
              'min_gap_min': args.min_gap_min, 'max_gap_min': args.max_gap_min,
              'min_subjects': args.min_subjects, 'notch_bins_removed': rows.notch,
              'exclude_first_h': args.exclude_first_h, 'pair_report': rows.pair_report,
              'exposure_version': opioid_mme.exposure_version(bool(shift) or first_post),
              'second_epoch_shift_min': shift,
              'second_zscore_dir': str(second_dir) if second_dir else None,
              'first_epoch_post': first_post,
              'first_zscore_dir': str(first_dir) if first_dir else None,
              'typical_dose_mme': opioid_mme.TYPICAL_DOSE_MME,
              'cbar_center': rows.cbar_center, 'autonomic': rows.autonomic_report,
              'zscore_dir': str(zdir), 'view_config': vc.provenance()}
    cohort_pairs = pairs[pairs['subject_id'].isin(cohort)]
    hr_parents = [str(config.EKG_HR_ROOT)] if args.autonomic else []
    io.write_table(cohort_pairs, run_dir / 'pair_index.csv',
                   params=params, parents=[str(med_state.ADMIN_TABLE)] + hr_parents,
                   subjects=cohort, script=SCRIPT)
    io.write_table(cells, run_dir / 'cells.csv', params=params,
                   parents=[str(d) for d in (zdir, first_dir, second_dir) if d],
                   subjects=cohort, script=SCRIPT)
    write_methods(run_dir, cells, params, len(cohort_pairs), cohort, summary)
    io.write_run_provenance(
        run_dir, script=SCRIPT, params=params,
        parents=[str(zdir), str(med_state.ADMIN_TABLE)] + hr_parents, subjects=cohort,
        extra={'status': 'EXPLORATORY mixed-model grid, NOT a finding',
               'n_cells': len(cells), 'fit_status': summary['n_cells'].to_dict(),
               'subjects_without_roi': sorted(rows.no_roi), **rows.split_report})
    io.log_analysis(f'pain_change {run_name}, {args.band_set}, {args.roi_scheme}, '
                    f'{len(cohort)} subjects, {len(cells)} cells', run_dir)
    n = summary['n_cells']
    logger.info('FIT STATUS (%d cells, %d above the floor):\n%s', len(cells),
                int(n.drop('floor').sum()), summary.to_string())
    logger.info('%d of %d fitted cells failed to converge, %d more converged with a '
                'warning', int(n['failed']), int(n.drop('floor').sum()), int(n['warn']))
    print(run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
