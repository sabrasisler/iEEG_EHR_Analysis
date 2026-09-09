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

The nonzero-median criterion is deliberately NOT applied here. It only affects
the classification arm (a median split at 0 is degenerate), and run_decoder skips
that arm per-unit rather than dropping the unit from the regression and ordinal
arms it is perfectly usable for.
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
