"""
Inherit a prior analysis run's cohort, exclusions and thresholds from its OWN
provenance, rather than re-deriving them.

A follow-up analysis that means to be comparable to an earlier one has to use the
same subjects, the same masks and the same coverage floors. Re-deriving them from
the same CLI defaults looks equivalent and is not: defaults move, and the only
record of what a run actually used is the `provenance.json` it wrote. So this
module reads that file and hands back the criteria as data.

    ref = load(REFERENCE_RUN)
    ref.describe()                 # log exactly what is being inherited
    ref.subjects                   # the 51
    ref.criteria['min_subjects']   # 8
    ref.view_params                # the ViewConfig dict the view was built with

`assert_cohort_matches` is the part that earns its keep. A re-derivation that
silently returns 50 subjects instead of 51 produces a figure that looks right and
is not comparable to the thing it is being compared against, so the default is to
REFUSE and print the diff.
"""

import json
import logging
from pathlib import Path

from ieeg_ehr import config, io

logger = logging.getLogger(__name__)

# The continuous-pain region x frequency cluster analysis this work extends.
# 24 clusters, 15 significant, n=51; commit 7e59751; see its METHODS.md.
CONTPAIN_HEATMAP = (config.PAIN_ANALYSIS_ROOT / 'psd_physiology'
                    / 'region_freq_heatmap' / 'contpain-roiv2'
                    / 'discovery_contpain_20260808-152707')

# Keys under provenance["params"] that are eligibility / coverage thresholds, as
# opposed to the permutation machinery this analysis replaces.
CRITERIA_KEYS = ('min_epochs', 'min_range', 'min_non_modal', 'min_subjects',
                 'keep_line_noise_bins')


class ReferenceRun:
    """One prior run's provenance, read once."""

    def __init__(self, run_dir, provenance):
        self.run_dir = Path(run_dir)
        self.provenance = provenance
        self.params = provenance.get('params', {})
        self.view_params = dict(self.params.get('view_params', {}))
        self.subjects = list(provenance.get('subjects') or [])
        self.criteria = {k: self.params[k] for k in CRITERIA_KEYS
                         if k in self.params}
        self.line_noise_bins_removed = list(
            provenance.get('line_noise_bins_removed') or [])
        self.git_commit = (provenance.get('git') or {}).get('commit')
        self.exclusions = list(provenance.get('exclusions') or [])

    # -- the artifacts, for comparison figures ------------------------------

    def group_pain_coef(self):
        """The two-stage group mean map: region, freq_bin_index, pain_coef.

        `on_stale='warn'`: this run is a fixed historical artifact and the point
        is to compare against it AS IT IS. Refusing on staleness would make the
        comparison impossible the moment the view is rebuilt, which is backwards.
        """
        return io.read_table(self.run_dir / 'pain_coef.parquet', on_stale='warn')

    def subject_diagnostics(self):
        return io.read_table(self.run_dir / 'subject_diagnostics.parquet',
                             on_stale='warn')

    def clusters(self):
        """NOTE: this table's `floor_ratio` column is all-NaN in the 2026-08-08
        run -- the script computed a floor and then passed floor=None. Do not use
        it."""
        return io.read_table(self.run_dir / 'clusters.parquet', on_stale='warn')

    # -- checks --------------------------------------------------------------

    def assert_cohort_matches(self, subjects, *, allow_drift=False):
        """Refuse a cohort that differs from the reference's, printing the diff."""
        want, got = set(self.subjects), set(subjects)
        if want == got:
            logger.info('cohort matches the reference exactly: %d subjects', len(got))
            return
        missing, extra = sorted(want - got), sorted(got - want)
        msg = (f'cohort DRIFT vs {self.run_dir.name}: '
               f'{len(got)} subjects here vs {len(want)} in the reference. '
               f'missing={missing} extra={extra}')
        if allow_drift:
            logger.warning('%s -- continuing because --allow-cohort-drift was passed. '
                           'Any comparison against the reference map is now '
                           'between different cohorts.', msg)
            return
        raise SystemExit(
            msg + '\nThe whole point of inheriting criteria is that the two runs '
                  'describe the same subjects. Fix the criteria, or pass '
                  '--allow-cohort-drift if the difference is intended and you are '
                  'prepared to caveat every comparison.')

    def describe(self):
        """Log exactly what is being inherited. Called at startup, always."""
        logger.info('=' * 70)
        logger.info('INHERITING criteria from %s', self.run_dir)
        logger.info('  recorded at    : %s  commit %s%s',
                    self.provenance.get('created'), (self.git_commit or '?')[:12],
                    ' (DIRTY)' if (self.provenance.get('git') or {}).get('dirty') else '')
        logger.info('  cohort         : %d subjects', len(self.subjects))
        logger.info('  excluded       : %d -- %s', len(self.exclusions),
                    '; '.join(f"{e.get('subject_id')} ({e.get('excluded_because')})"
                              for e in self.exclusions) or 'none')
        logger.info('  eligibility    : %s',
                    {k: v for k, v in self.criteria.items()
                     if k in ('min_epochs', 'min_range', 'min_non_modal')})
        logger.info('  coverage floor : min_subjects=%s',
                    self.criteria.get('min_subjects'))
        logger.info('  line-noise bins: %s (removed)', self.line_noise_bins_removed)
        logger.info('  view           : normalization=%s domain=%s region=%s '
                    'freq=%s roi_scheme=%s',
                    self.view_params.get('normalization'),
                    self.view_params.get('domain'), self.view_params.get('region'),
                    self.view_params.get('freq'), self.view_params.get('roi_scheme'))
        logger.info('  mask           : %s=%s max_excluded_frac=%s',
                    self.view_params.get('mask_level'),
                    self.view_params.get('mask_label'),
                    self.view_params.get('max_excluded_frac'))
        logger.info('  epoch          : %s min pre-report',
                    self.view_params.get('epoch_minutes'))
        logger.info('MASK CONTENT: signal quality ONLY (gross artifact, saturation, '
                    'square wave, flatline, bipolar variance). It does NOT exclude '
                    'opioid-administration windows or post-ictal periods -- no such '
                    'table exists in this project yet. Both are first-order '
                    'confounds for low-frequency power.')
        logger.info('=' * 70)

    def provenance_summary(self):
        """The inherited criteria, for this run's own provenance.json."""
        return {
            'reference_run': str(self.run_dir),
            'reference_commit': self.git_commit,
            'reference_created': self.provenance.get('created'),
            'inherited_subjects': list(self.subjects),
            'inherited_criteria': dict(self.criteria),
            'inherited_view_params': dict(self.view_params),
            'inherited_line_noise_bins_removed': list(self.line_noise_bins_removed),
            'inherited_exclusions': list(self.exclusions),
        }


def load(run_dir=CONTPAIN_HEATMAP):
    """Read a run directory's provenance.json into a ReferenceRun."""
    run_dir = Path(run_dir)
    path = run_dir / 'provenance.json'
    if not path.exists():
        raise SystemExit(f'no provenance.json in {run_dir} -- cannot inherit '
                         'criteria from a run that did not record any')
    return ReferenceRun(run_dir, json.loads(path.read_text()))
