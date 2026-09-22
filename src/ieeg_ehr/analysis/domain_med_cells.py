"""Medication effects on power: ONE FIT PER DOMAIN x BAND (panel F).

    log10_power ~ NRS_within * med_within + NRS_submean + med_submean
                  + (NRS_within + med_within || subject)
                  + (1 | subject:channel)

HOW THIS DIFFERS FROM `run_domain_model`, and why it is a separate model rather
than a plotting option
----------------------------------------------------------------------------
The domain model fits ONE model per band with domain as a FIXED EFFECT, so every
domain is estimated against a reference domain and the headline test is an
omnibus asking "do the domains differ from each other". This module fits each
domain SEPARATELY. Three consequences, all of them deliberate:

  - THERE IS NO REFERENCE DOMAIN and no between-domain contrast. Control is a
    column like any other, estimated on its own contacts. You cannot read "is
    Sensory different from Control" off this figure; that question needs the
    other model, which puts both in one fit.
  - THERE IS NO OMNIBUS, and cannot be. An omnibus is a joint test on a block of
    domain interaction terms, and there is no such block when domain is not in
    the model. Requested explicitly.
  - EACH FIT SEES ONLY ITS OWN DOMAIN'S CONTACTS, so the residual variance, the
    subject random effects and the channel intercepts are all estimated
    within-domain. A domain with 82 contacts and one with 1035 no longer share
    anything, which is more honest about a small domain and also gives it wider
    intervals.

Multiplicity therefore moves entirely to BH across the domain x band grid, run
WITHIN each term -- 'med' is one family, 'pain_x_med' another. They are
different questions of the same fits and pooling them would correct one using
the other's p-values.

NO PARCEL OR ROI TERM, BY REQUEST -- and what that costs
--------------------------------------------------------
The model has `(1 | subject:channel)` but nothing at the parcel/ROI level, so
within a domain every contact is exchangeable beyond its own intercept. The
domains are not evenly composed: thalamus is 52% of Sensory's contacts and
superiorfrontal 24% of Cognitive's. A domain estimate is therefore a
CONTACT-WEIGHTED average that its largest constituent can dominate, and the
intervals do not price in parcel-to-parcel disagreement, so they are optimistic
by however much that disagreement is real. `run_domain_model --unit roi`
documents the same trade at the ROI level; the guard that moves out of the model
is the region-level consistency map, which shows each region's own slope and how
many subjects share its sign.

WHAT `med_within` IS READ AT -- the thing that keeps getting misread
--------------------------------------------------------------------
Both predictors are SUBJECT-MEAN-CENTRED, so each coefficient is read at the
other's patient-specific mean. `med_within` is the shift in power for a 0->1
dose-state switch AT THAT PATIENT'S OWN MEAN PAIN. Not zero pain, not a
pain-free baseline, not the cohort mean: a patient averaging NRS 6 has their
medication effect evaluated at 6. And medication is not randomised -- it is
given BECAUSE of pain -- so this is an association with the dosed state, never a
drug effect.

PANEL F2: THE UNDOSED AND DOSED PAIN SLOPES
-------------------------------------------
Because `med_within = med_state - p_subject`, the pain slope implied by the fit
depends on dose state:

    slope(med_state) = b_NRS + b_int * (med_state - p)
    undosed (med_state=0) = b_NRS - p * b_int
    dosed   (med_state=1) = b_NRS + (1 - p) * b_int

`p` is taken as `pbar`, the mean over SUBJECTS of each subject's dosed
proportion -- a subject-level mean, not the pooled epoch fraction, so a patient
with 200 epochs does not outweigh one with 20. THE TWO DIFFER whenever epoch
counts and dosing rates covary, and which one is meant changes the numbers, so
it is recorded per cell as `p_dosed` rather than left implicit.

pbar is treated as a KNOWN CONSTANT. Its own sampling error is not propagated
into the SEs, which are the exact linear-combination SEs from the fixed-effect
covariance (c' V c, including the NRS/interaction covariance -- not a sum of
variances). The two derived slopes are also not independent of each other or of
the interaction: they are three views of two parameters, which is why F2 marks
significance from the INTERACTION's BH decision rather than testing each end.

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import logging
import time

import numpy as np
import pandas as pd

from ieeg_ehr.analysis import mixed_model as mm

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/domain_med_cells.py'

#: No domain term: this formula is fitted once per domain.
CELL_FORMULA = ('log10_power ~ NRS_within * med_within '
                '+ NRS_submean + med_submean')

#: `(NRS_within + med_within || subject) + (1 | subject:channel)`. statsmodels
#: variance components are independent by construction, which is exactly what
#: the `||` asks for -- uncorrelated random intercept and two random slopes.
VC_CELL = {
    'subj_int': '1',
    'subj_nrs_slope': '0 + NRS_within',
    'subj_med_slope': '0 + med_within',
    'channel': '0 + C(channel_uid)',
}

#: A cell this thin cannot support a within-subject interaction, so it is
#: skipped and counted rather than fitted into a meaningless wide interval.
MIN_SUBJECTS = 5
MIN_ROWS = 200


def subject_dosed_fraction(df):
    """pbar: the mean over SUBJECTS of each subject's dosed epoch proportion.

    Subject-level, not the pooled epoch fraction -- see the module docstring.
    Computed on EPOCHS, de-duplicated first: `df` is one row per
    (epoch, channel), so a raw column mean would weight each subject's dosing
    rate by how many contacts they happen to have in this domain, which is a
    property of their implant and not of their medication.
    """
    per_epoch = df.drop_duplicates(['subject', 'epoch_id'])
    return float(per_epoch.groupby('subject')['med_state'].mean().mean())


def cell_contrasts(res, pbar, band, domain):
    """The five numbers panel F needs, as linear combinations of the fit.

    Every one carries an SE from the fixed-effect covariance block, so the two
    derived slopes get the NRS/interaction covariance term rather than a sum of
    variances (which would be wrong in both directions depending on its sign).
    """
    from scipy import stats

    names = list(res.fe_params.index)
    cov = np.asarray(res.cov_params())[:len(names), :len(names)]
    beta = res.fe_params.to_numpy()

    def ix(term):
        parts = set(term.split(':'))
        for i, n in enumerate(names):
            if set(n.split(':')) == parts:
                return i
        return None

    i_nrs, i_med = ix('NRS_within'), ix('med_within')
    i_int = ix('NRS_within:med_within')
    if i_nrs is None or i_med is None or i_int is None:
        logger.warning('%s/%s: missing a required term in %s', domain, band,
                       names)
        return pd.DataFrame()

    def row(term, weights):
        c = np.zeros(len(names))
        for i, w in weights:
            c[i] = w
        est = float(c @ beta)
        var = float(c @ cov @ c)
        se = float(np.sqrt(var)) if var > 0 else np.nan
        z = est / se if se and np.isfinite(se) and se > 0 else np.nan
        return {'band': band, 'domain': domain, 'term': term,
                'beta': est, 'se': se, 'z': z,
                'p': (float(2 * stats.norm.sf(abs(z)))
                      if np.isfinite(z) else np.nan),
                'ci_lo': est - 1.96 * se, 'ci_hi': est + 1.96 * se,
                'p_dosed': pbar}

    return pd.DataFrame([
        # The medication shift itself -- panel F1.
        row('med', [(i_med, 1.0)]),
        # The pain slope at the subject's OWN mean dosing, i.e. med_within = 0.
        row('pain', [(i_nrs, 1.0)]),
        # How much the pain slope changes per dose-state switch.
        row('pain_x_med', [(i_int, 1.0)]),
        # Panel F2's two ends, derived from the same two parameters.
        row('pain_undosed', [(i_nrs, 1.0), (i_int, -pbar)]),
        row('pain_dosed', [(i_nrs, 1.0), (i_int, 1.0 - pbar)]),
    ])


def fit_domain_cell(df, band, domain):
    """(record, contrast frame) for ONE domain x band fit, or (record, empty).

    Returns a record even when the cell is skipped or the fit fails, so the
    grid has a row for every cell and a blank in a figure is traceable to a
    reason rather than to a silently dropped combination.
    """
    n_sub = int(df['subject'].nunique())
    base = {'band': band, 'domain': domain, 'n_rows': int(len(df)),
            'n_subjects': n_sub,
            'n_channels': int(df['channel_uid'].nunique()),
            'n_parcels': int(df['parcel'].nunique())
            if 'parcel' in df.columns else np.nan}

    if len(df) < MIN_ROWS or n_sub < MIN_SUBJECTS:
        logger.warning('%-11s %-11s SKIPPED: %d rows, %d subjects '
                       '(need %d/%d)', band, domain, len(df), n_sub,
                       MIN_ROWS, MIN_SUBJECTS)
        return {**base, 'converged': False, 'skipped': True,
                'skip_reason': f'{len(df)} rows, {n_sub} subjects'}, pd.DataFrame()

    dosed = df.drop_duplicates(['subject', 'epoch_id'])['med_state']
    n_dosed_sub = int(df[df['med_state'] == 1]['subject'].nunique())
    if dosed.nunique() < 2 or n_dosed_sub < MIN_SUBJECTS:
        logger.warning('%-11s %-11s SKIPPED: medication does not vary enough '
                       '(%d subjects ever dosed)', band, domain, n_dosed_sub)
        return {**base, 'converged': False, 'skipped': True,
                'skip_reason': f'{n_dosed_sub} dosed subjects'}, pd.DataFrame()

    pbar = subject_dosed_fraction(df)
    t0 = time.time()
    res, warn = mm.fit_cell(df, VC_CELL, formula=CELL_FORMULA)
    elapsed = time.time() - t0

    slopes = cell_contrasts(res, pbar, band, domain)
    vc = mm.vcomp_by_name(res)
    rec = {
        **base, 'skipped': False, 'skip_reason': '',
        'p_dosed': pbar,
        'n_subjects_dosed': n_dosed_sub,
        'var_subj_int': float(vc.get('subj_int', np.nan)),
        'var_subj_nrs_slope': float(vc.get('subj_nrs_slope', np.nan)),
        'var_subj_med_slope': float(vc.get('subj_med_slope', np.nan)),
        'var_channel': float(vc.get('channel', np.nan)),
        'var_resid': float(res.scale),
        'converged': bool(res.converged),
        'fit_seconds': elapsed, 'n_warnings': len(warn),
        'warnings': ' | '.join(sorted(set(warn)))[:400],
    }
    if len(slopes):
        g = slopes.set_index('term')
        logger.info('%-11s %-11s | %6d rows %3d subj %4d chan | pbar %.3f | '
                    'med %+.5f (p %.3g)  int %+.5f (p %.3g) | %.0fs%s',
                    band, domain, rec['n_rows'], n_sub, rec['n_channels'],
                    pbar, g.loc['med', 'beta'], g.loc['med', 'p'],
                    g.loc['pain_x_med', 'beta'], g.loc['pain_x_med', 'p'],
                    elapsed, '' if res.converged else '  NOT CONVERGED')
    return rec, slopes


# ---------------------------------------------------------------------------
# Figures F1 and F2
# ---------------------------------------------------------------------------

F_DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
                'Not confirmed out of sample.')

NO_DOMAIN_CONTRAST_NOTE = (
    'ONE FIT PER DOMAIN x BAND, with NO DOMAIN TERM: each column is estimated on '
    'its own contacts, so there is NO reference domain, NO between-domain '
    'contrast and NO omnibus. Control is a column like any other, not a '
    'baseline -- this figure cannot tell you whether Sensory differs FROM '
    'Control, only what each looks like on its own. '
    'log10_power ~ NRS_within * med_within + NRS_submean + med_submean + '
    '(NRS_within + med_within || subject) + (1 | subject:channel). There is no '
    'parcel or ROI term by design, so a domain estimate is a CONTACT-WEIGHTED '
    'average its largest constituent can dominate (thalamus is 52% of Sensory, '
    'superiorfrontal 24% of Cognitive) and the intervals do not price in '
    'parcel-to-parcel disagreement. BOTH PREDICTORS ARE SUBJECT-MEAN-CENTRED, so '
    'each coefficient is read at the OTHER one\'s patient-specific mean. ')


def _grid(domains, bands, width=2.65, height=5.2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(domains),
                             figsize=(width * len(domains) + 1.8, height),
                             sharey=True, sharex=True, squeeze=False)
    return fig, axes[0]


def _band_labels(ax, bands, y, extra=None):
    ax.set_yticks(y)
    ax.set_yticklabels([f'{b}' if not extra else f'{b}\n{extra.get(b, "")}'
                        for b in bands], fontsize=8)
    ax.set_ylim(len(bands) - 0.5, -0.5)


def figure_f1(run_dir, slopes, cells, domains, bands, colours, subtitle=''):
    """F1: the medication shift, rows = bands, columns = domains."""
    import matplotlib.pyplot as plt

    d_all = slopes[slopes['term'] == 'med']
    if d_all.empty:
        logger.warning('no med rows -- skipping F1')
        return
    fig, axes = _grid(domains, bands)
    xmax = float(np.nanmax(np.abs(np.concatenate(
        [d_all['ci_lo'].to_numpy(), d_all['ci_hi'].to_numpy()])))) * 1.08
    y = np.arange(len(bands))

    for ax, dom in zip(axes, domains):
        colour = colours.get(dom, '0.4')
        d = d_all[d_all['domain'] == dom].set_index('band').reindex(bands)
        for i, band in enumerate(bands):
            r = d.loc[band]
            if not np.isfinite(r.get('beta', np.nan)):
                continue
            sig = bool(r.get('p_bh_reject', False))
            ax.errorbar(r['beta'], i, xerr=1.96 * r['se'], fmt='o',
                        ms=8 if sig else 5.5, lw=1.8 if sig else 1.1,
                        capsize=3, color=colour,
                        markerfacecolor=colour if sig else 'white', zorder=3)
        ax.axvline(0, color='0.5', lw=1.0, ls='--', zorder=1)
        c = cells[cells['domain'] == dom]
        n_c = int(c['n_channels'].max()) if len(c) else 0
        n_s = int(c['n_subjects'].max()) if len(c) else 0
        pbar = float(c['p_dosed'].mean()) if 'p_dosed' in c else np.nan
        ax.set_title(f'{dom}\n{n_c} chan, {n_s} subj, {pbar:.0%} dosed',
                     fontsize=9, color=colour)
        ax.set_xlim(-xmax, xmax)
        ax.set_xlabel('d log10 power when dosed', fontsize=8)
        ax.tick_params(labelsize=7)
    _band_labels(axes[0], bands, y)

    fig.suptitle('F1 -- MEDICATION effect on power, one fit per domain x band\n'
                 'shift in power when dosed, at the subject\'s own average pain; '
                 'filled = BH-significant across all domain x band fits'
                 + (f'\n{subtitle}' if subtitle else ''), fontsize=11.5)
    fig.tight_layout(rect=(0, 0.20, 1, 0.88))
    fig.text(0.01, 0.005,
             NO_DOMAIN_CONTRAST_NOTE
             + 'THE MEDICATION EFFECT IS READ AT THAT PATIENT\'S OWN MEAN PAIN '
             '-- not zero pain, not a pain-free baseline, not the cohort mean: '
             'a patient averaging NRS 6 has it evaluated at 6. Medication is '
             'not randomised, it is given BECAUSE of pain, so this is an '
             'association with the DOSED STATE and not a drug effect. BH runs '
             'across the whole domain x band grid WITHIN this term; the '
             'interaction is a separate family and the two are never pooled. '
             'X SCALE IS SHARED ACROSS DOMAINS. A blank cell was skipped for '
             'thin coverage -- see the run\'s cells table for the reason. '
             f'{F_DISCLAIMER}',
             fontsize=6.3, va='bottom', ha='left', color='0.35', wrap=True)
    out = run_dir / 'fig_F1_med_effect.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out)


def figure_f2(run_dir, slopes, cells, domains, bands, colours, subtitle=''):
    """F2: the pain slope undosed vs dosed, connected, per domain."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    und = slopes[slopes['term'] == 'pain_undosed'].set_index(['domain', 'band'])
    dos = slopes[slopes['term'] == 'pain_dosed'].set_index(['domain', 'band'])
    inter = slopes[slopes['term'] == 'pain_x_med'].set_index(['domain', 'band'])
    if und.empty or dos.empty:
        logger.warning('no derived slope rows -- skipping F2')
        return

    fig, axes = _grid(domains, bands, width=2.9)
    xmax = float(np.nanmax(np.abs(np.concatenate(
        [und['ci_lo'], und['ci_hi'], dos['ci_lo'], dos['ci_hi']])))) * 1.08
    y = np.arange(len(bands))
    OFF = 0.17

    for ax, dom in zip(axes, domains):
        colour = colours.get(dom, '0.4')
        for i, band in enumerate(bands):
            key = (dom, band)
            if key not in und.index or key not in dos.index:
                continue
            u, d = und.loc[key], dos.loc[key]
            sig = bool(inter.loc[key, 'p_bh_reject']) if key in inter.index \
                else False
            # The connector IS the interaction: its length is b_int, so a
            # BH-significant interaction is drawn as a solid emphasised link
            # and a null one stays faint.
            ax.plot([u['beta'], d['beta']], [i - OFF, i + OFF], '-',
                    color=colour, lw=2.0 if sig else 0.9,
                    alpha=0.9 if sig else 0.35, zorder=2)
            # Undosed: open marker. Dosed: filled. Shape carries the state too,
            # so the pair is readable without relying on fill alone.
            ax.errorbar(u['beta'], i - OFF, xerr=1.96 * u['se'], fmt='o',
                        ms=5.5, lw=1.0, capsize=2.5, color=colour,
                        markerfacecolor='white', zorder=3)
            ax.errorbar(d['beta'], i + OFF, xerr=1.96 * d['se'], fmt='D',
                        ms=5.0, lw=1.0, capsize=2.5, color=colour,
                        markerfacecolor=colour, zorder=3)
            if sig:
                ax.text(xmax * 0.94, i, '*', fontsize=13, color=colour,
                        ha='right', va='center', zorder=4)
        ax.axvline(0, color='0.5', lw=1.0, ls='--', zorder=1)
        for i in range(len(bands) - 1):
            ax.axhline(i + 0.5, color='0.9', lw=0.6, zorder=0)
        c = cells[cells['domain'] == dom]
        pbar = float(c['p_dosed'].mean()) if 'p_dosed' in c else np.nan
        ax.set_title(f'{dom}\npbar = {pbar:.3f}', fontsize=9, color=colour)
        ax.set_xlim(-xmax, xmax)
        ax.set_xlabel('d log10 power / pain point', fontsize=8)
        ax.tick_params(labelsize=7)
    _band_labels(axes[0], bands, y)

    axes[-1].legend(handles=[
        Line2D([], [], marker='o', color='0.35', markerfacecolor='white',
               ls='none', ms=6, label='undosed'),
        Line2D([], [], marker='D', color='0.35', markerfacecolor='0.35',
               ls='none', ms=5.5, label='dosed'),
        Line2D([], [], color='0.35', lw=2.0,
               label='* interaction BH-significant')],
        loc='lower right', fontsize=6.5, frameon=True, framealpha=0.9)

    fig.suptitle('F2 -- the PAIN SLOPE undosed vs dosed, derived from the same '
                 'fit\n'
                 'undosed = b_NRS - pbar*b_int   dosed = b_NRS + (1-pbar)*b_int; '
                 '* = interaction BH-significant'
                 + (f'\n{subtitle}' if subtitle else ''), fontsize=11.5)
    fig.tight_layout(rect=(0, 0.20, 1, 0.88))
    fig.text(0.01, 0.005,
             NO_DOMAIN_CONTRAST_NOTE
             + 'THE TWO ENDS ARE NOT TWO FITS. Both come from the same model: '
             'because med_within = med_state - p, the implied pain slope is '
             'b_NRS + b_int*(med_state - p), so the undosed and dosed ends are '
             'linear combinations of the SAME two coefficients and the '
             'connector length IS the interaction. Their SEs are exact '
             'linear-combination SEs (c\' V c from the fixed-effect '
             'covariance), which includes the NRS/interaction COVARIANCE -- not '
             'a sum of variances. pbar is the mean over SUBJECTS of each '
             'subject\'s dosed epoch proportion (de-duplicated to epochs first, '
             'so a subject is not weighted by contact count), NOT the pooled '
             'epoch fraction; it is printed per column and stored per cell as '
             '`p_dosed`. pbar is treated as a KNOWN CONSTANT and its own '
             'sampling error is NOT propagated. SIGNIFICANCE IS MARKED FROM THE '
             'INTERACTION\'S BH DECISION, not by testing each end: the two ends '
             'are not independent of each other or of the interaction, so '
             'overlapping intervals here are not a test of anything. '
             f'{F_DISCLAIMER}',
             fontsize=6.3, va='bottom', ha='left', color='0.35', wrap=True)
    out = run_dir / 'fig_F2_pain_slope_by_dose.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out)
