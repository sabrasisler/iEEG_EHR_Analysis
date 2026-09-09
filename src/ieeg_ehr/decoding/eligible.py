"""Which subject-sessions the decoder can run on, as a list the array reads.

    python -m ieeg_ehr.decoding.eligible --view-dir <dir>          # one per line
    python -m ieeg_ehr.decoding.eligible --view-dir <dir> --count
    python -m ieeg_ehr.decoding.eligible --view-dir <dir> --table  # with reasons

ELIGIBILITY LIVES HERE, NOT IN THE VIEW. The view is built for the whole runnable
discovery split because a feature artifact must not bake in an analysis choice --
change the epoch threshold and you would otherwise be rebuilding features. This
module is that analysis choice, in one place, so the array bound and the run's
provenance cannot disagree about who was in it.

THE CRITERIA
  >=30 epochs   so k-fold keeps >=5 pain reports per fold (the paper's rule)
  discovery     the hold-out is unreachable unless deliberately requested
  a built view  a unit with no view table is not "ineligible", it is UNBUILT, and
                the distinction matters when the array count comes up short

The nonzero-median and pain-range criteria are NOT applied here, because they are
ARM-SPECIFIC: they gate the classification arm only, and a subject that fails
them is still perfectly usable for regression and ordinal. `classification_ok`
below is that gate, and it is applied per-arm inside run_decoder and again in
aggregate.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from ieeg_ehr import config
from ieeg_ehr.config import cohorts

logger = logging.getLogger(__name__)

MIN_EPOCHS = 30

#: The classification arm additionally requires the target paper's inclusion
#: criteria 2 and 3. ONE definition, used by run_decoder (to skip the arm) and by
#: aggregate (to filter results produced before the gate existed), so the two
#: cannot drift.
MIN_MEDIAN_PAIN = 0.0      # strictly greater than this
MIN_PAIN_RANGE = 5.0       # >=50% of the 0-10 scale


def classification_ok(median_pain, pain_range):
    """Is a median split MEANINGFUL for this subject?  (reason, or None if ok)

    WHY THE MEDIAN MUST BE NONZERO, and it is not a technicality. When a
    subject's median pain is 0, the "median split" is literally `0 vs >0` -- a
    PAIN vs NO-PAIN discrimination, not the LOW vs HIGH PAIN one the arm claims
    to measure. That is a different and easier question, so including such
    subjects makes the group AUC an average over two incomparable analyses.

    Measured on the 2026-09-08 run: 10 of 44 classification subjects had a
    median of 0, and the top TWO performers (sub-189 at AUC 0.814, sub-051 at
    0.809) were both among them -- i.e. the headline was partly a pain-detector
    result. Their class balance gives it away (median 0.28 vs 0.41 for the rest).
    Group AUC barely moves without them (0.598 -> 0.597, Mann-Whitney p = 0.90);
    the reason to exclude is interpretability, not effect size.

    The range criterion is the paper's "at least 50% of the total possible pain
    range", which on this cohort is nearly non-binding -- it removes exactly one
    subject (sub-189, range 4), already excluded by the median rule.
    """
    if median_pain is None or pain_range is None:
        return 'median/range unknown'
    if not median_pain > MIN_MEDIAN_PAIN:
        return f'median pain {median_pain:g} is not > {MIN_MEDIAN_PAIN:g}'
    if pain_range < MIN_PAIN_RANGE:
        return f'pain range {pain_range:g} < {MIN_PAIN_RANGE:g}'
    return None


def eligible_units(view_dir, split='discovery', min_epochs=MIN_EPOCHS):
    """[(subject, session, n_epochs, reason)] for every unit the view holds."""
    view_dir = Path(view_dir)
    allowed = set(cohorts.subjects_for_split(split)) if split else None

    rows = []
    for path in sorted(view_dir.glob('view_epochs_sub-*.parquet')):
        stem = path.stem.replace('view_epochs_', '')
        subject = stem.split('_')[0].replace('sub-', '')
        session = stem.split('_')[1].replace('ses-', '')

        defs_path = config.pain_epoch_defs_path(subject, session)
        n_epochs = len(pd.read_parquet(defs_path)) if defs_path.exists() else 0

        reason = None
        if allowed is not None and subject not in allowed:
            reason = f'not in {split}'
        elif n_epochs < min_epochs:
            reason = f'{n_epochs} epochs < {min_epochs}'
        rows.append({'subject': subject, 'session': session,
                     'n_epochs': n_epochs, 'eligible': reason is None,
                     'reason': reason or 'ok'})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--view-dir', required=True)
    ap.add_argument('--split', default='discovery')
    ap.add_argument('--min-epochs', type=int, default=MIN_EPOCHS)
    ap.add_argument('--count', action='store_true', help='print the count only')
    ap.add_argument('--table', action='store_true', help='print every unit + reason')
    args = ap.parse_args(argv)

    logging.basicConfig(level='WARNING')
    table = eligible_units(args.view_dir, args.split, args.min_epochs)
    if table.empty:
        print(f'no view tables in {args.view_dir}', file=sys.stderr)
        return 1

    if args.table:
        print(table.to_string(index=False))
        return 0

    keep = table[table['eligible']]
    if args.count:
        print(len(keep))
        return 0
    for row in keep.itertuples():
        print(f'{row.subject}_{row.session}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
