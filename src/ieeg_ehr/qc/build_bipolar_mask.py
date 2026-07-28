#!/usr/bin/env python3
"""
Roll the raw-voltage mask and the bipolar_variance exclusions into ONE pair-keyed
bipolar MASK -- the artifact the view layer consumes.

WHY THIS EXISTS
---------------
`qc/bipolar/masks/` was empty. The bipolar level has exactly one artifact type
(`bipolar_variance`), so `build_mask.py`'s job there -- OR across artifact types --
had nothing to do and was never run. But a *usable* bipolar mask is not an OR
across bipolar detectors; it is the JOIN of two levels:

    a pair is excluded in a 60s bin  IF
        EITHER contributing monopolar contact is excluded in the raw-voltage mask
        OR the pair's own variance z exceeded the bipolar_variance threshold

The first clause is THE PAIR RULE (`mask_projection`); the second is this level's
own detector. Doing the join once, here, means every view reads a single
pair-keyed mask instead of re-deriving the projection -- which is the difference
between one implementation and N subtly different ones.

WHAT IT IS *NOT*
----------------
Not a threshold sweep: the bipolar threshold is selected by LABEL
(`--bipolar-variance-label std10`), never re-computed, so the metric/threshold
split upstream is preserved. Sweeping means pointing this at a different label.

A NOTE ON THE TWO INPUTS' INDEPENDENCE
--------------------------------------
They are not fully independent, and the docstring says so because the number is
easy to misread: `build_bipolar_exclusions.py` computes its baseline
mask-AWARE, i.e. it already excludes raw-voltage-flagged windows from its own
mean/std. So the raw-voltage mask influenced *where* the bipolar threshold fell.
It did NOT put those windows into the exclusions table -- that is exactly why this
rollup is needed, and why `excluded_bipolar_variance` alone leaves saturation /
flatline / square-wave / gross-artifact windows in the data.

Output per subject/session: masks/<label>/sub-XXX_ses-YY.parquet, dense over the
input's (run, pair, 60s bin) grid, with one transparency column per contributing
input and `excluded` = their OR -- the same shape build_mask.py writes at the
raw-voltage level, so one reader understands both.

Usage:
  python -m ieeg_ehr.qc.build_bipolar_mask                       # pinned defaults
  python -m ieeg_ehr.qc.build_bipolar_mask --bipolar-variance-label std10 \
      --raw-voltage-mask-label gross-std3_satmargin15_sw_logz4 --subjects 019 039
"""

import argparse
import logging

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.qc import mask_projection

logger = logging.getLogger(__name__)

BIN_SEC = mask_projection.BIN_SEC

# Read only what the rollup needs. anode/cathode come along because they are the
# join keys against the monopolar raw-voltage mask, and they stay in the output so
# a consumer can audit the pair rule without re-reading the exclusions table.
EXCLUSION_USECOLS = ['session_id', 'run_id', 'channel', 'anode_channel',
                     'cathode_channel', 'bin_start', 'excluded']

KEY = ['run_id', 'channel', 'bin_start']
OUT_COLUMNS = (KEY + ['anode_channel', 'cathode_channel', 'bin_end',
                      'excluded_raw_voltage_anode', 'excluded_raw_voltage_cathode',
                      'excluded_raw_voltage', 'excluded_bipolar_variance', 'excluded'])


def _read_exclusions(path):
    """The bipolar_variance exclusions for one subject, CSV or Parquet.

    Reads either extension on purpose: the exclusions level is CSV today (82
    subjects on disk, plus an array job actively writing more), and converting it
    is a separate follow-up. Accepting both here means that conversion will not
    require touching this module.
    """
    if path.suffix == '.parquet':
        return pd.read_parquet(path, columns=EXCLUSION_USECOLS)
    return pd.read_csv(path, usecols=EXCLUSION_USECOLS)


def build_one(subject, session, exclusions, mask_lookup):
    """One subject/session's rolled-up mask, or None if this session has no rows.

    `exclusions` is the whole subject's table (session is a column at this level);
    `mask_lookup` is a mask_projection.load_mask_lookup() frame for the same
    subject.
    """
    df = exclusions[exclusions['session_id'] == f'ses-{session}'].copy()
    if df.empty:
        return None

    # The exclusions grid is already 60s-binned, so `_bin` is bin_start itself --
    # no re-flooring. Named `_bin` because that is what or_pair_flags_60s joins on.
    df['_bin'] = df['bin_start'].astype(np.float64)

    anode = mask_projection.or_pair_flags_60s(df, mask_lookup, 'anode_channel')
    cathode = mask_projection.or_pair_flags_60s(df, mask_lookup, 'cathode_channel')

    out = pd.DataFrame({
        'run_id': df['run_id'].to_numpy(),
        'channel': df['channel'].to_numpy(),
        'bin_start': df['bin_start'].to_numpy(),
        'anode_channel': df['anode_channel'].to_numpy(),
        'cathode_channel': df['cathode_channel'].to_numpy(),
        'excluded_raw_voltage_anode': anode,
        'excluded_raw_voltage_cathode': cathode,
        'excluded_bipolar_variance': df['excluded'].eq(True).to_numpy(),
    })
    out['excluded_raw_voltage'] = out['excluded_raw_voltage_anode'] | out['excluded_raw_voltage_cathode']
    out['excluded'] = out['excluded_raw_voltage'] | out['excluded_bipolar_variance']
    out['bin_end'] = out['bin_start'] + BIN_SEC
    out = out.sort_values(KEY).reset_index(drop=True)
    return out[OUT_COLUMNS]


def _report(tag, out):
    """Per-subject breakdown. The point is that a rollup where one input
    contributes NOTHING is the silent-failure mode -- a join that matched no rows
    looks exactly like a clean recording -- so every contributor is counted
    separately and printed, not just the union."""
    rv, bv, ex = out['excluded_raw_voltage'], out['excluded_bipolar_variance'], out['excluded']
    a, c = out['excluded_raw_voltage_anode'], out['excluded_raw_voltage_cathode']
    logger.info(
        '  %s: %d bins, %d excluded (%.2f%%) | rv-only %d, bv-only %d, both %d '
        '| anode-only %d, cathode-only %d, both-legs %d',
        tag, len(out), int(ex.sum()), 100.0 * ex.mean(),
        int((rv & ~bv).sum()), int((bv & ~rv).sum()), int((rv & bv).sum()),
        int((a & ~c).sum()), int((c & ~a).sum()), int((a & c).sum()),
    )
    if not rv.any():
        logger.warning('  %s: raw-voltage contributed ZERO excluded bins -- suspect a '
                       'failed join (channel naming / run_id / bin grid), not a clean '
                       'recording.', tag)
    if not bv.any():
        logger.warning('  %s: bipolar_variance contributed ZERO excluded bins.', tag)


def run(bipolar_variance_label, raw_voltage_mask_label, out_label, subjects=None):
    excl_dir = config.bipolar_exclusion_dir(bipolar_variance_label)
    rv_dir = config.raw_voltage_mask_dir(raw_voltage_mask_label)
    if not excl_dir.exists():
        raise SystemExit(f'No bipolar exclusions at {excl_dir} (run build_bipolar_exclusions first)')
    if not rv_dir.exists():
        raise SystemExit(f'No raw-voltage mask at {rv_dir} (run build_mask first)')

    # Subjects present in BOTH inputs. Anything present in only one is skipped and
    # named -- never written with the missing half zero-filled, because a label
    # must not claim a rollup it did not have (docs/labnotebook/2026-07-28.md,
    # sub-236's flatline sitting at `masked-...` with an unmasked baseline).
    excl_subjects = {p.stem for p in excl_dir.glob('sub-*.csv')} | \
                    {p.stem for p in excl_dir.glob('sub-*.parquet')}
    rv_pairs = {}
    for p in rv_dir.glob('sub-*_ses-*.csv'):
        subj, ses = p.stem.split('_ses-')
        rv_pairs.setdefault(subj, set()).add(ses)

    if subjects:
        wanted = {f'sub-{s.replace("sub-", "")}' for s in subjects}
        excl_subjects &= wanted

    have_both = sorted(excl_subjects & set(rv_pairs))
    only_excl = sorted(excl_subjects - set(rv_pairs))
    only_rv = sorted(set(rv_pairs) - excl_subjects) if not subjects else []
    if only_excl:
        logger.warning('NOTE: no raw-voltage mask at %s for %d subject(s) (skipped): %s',
                       raw_voltage_mask_label, len(only_excl), only_excl)
    if only_rv:
        logger.warning('NOTE: no bipolar_variance exclusions at %s for %d subject(s) '
                       '(skipped): %s', bipolar_variance_label, len(only_rv), only_rv)
    if not have_both:
        raise SystemExit('No subject has both inputs; nothing to roll up.')

    out_dir = config.bipolar_mask_dir(out_label)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = {
        'bipolar_variance_label': bipolar_variance_label,
        'raw_voltage_mask_label': raw_voltage_mask_label,
        'mask_label': out_label,
        'bin_sec': BIN_SEC,
        'rule': ('excluded = (raw_voltage[anode] | raw_voltage[cathode]) '
                 '| bipolar_variance'),
    }

    written = []
    for subject_id in have_both:
        subject = subject_id.replace('sub-', '')
        excl_path = config.bipolar_exclusion_path(subject, bipolar_variance_label)
        if not excl_path.exists():
            excl_path = excl_path.with_suffix('.parquet')
        exclusions = _read_exclusions(excl_path)
        mask_lookup = mask_projection.load_mask_lookup(rv_dir, subject_id)

        for session in sorted(rv_pairs[subject_id]):
            out = build_one(subject, session, exclusions, mask_lookup)
            tag = f'{subject_id}_ses-{session}'
            if out is None:
                logger.warning('  %s: no exclusion rows for this session, skipped', tag)
                continue
            _report(tag, out)
            path = config.bipolar_mask_path(subject, session, out_label)
            io.write_table(out, path, kind='mask',
                           script='ieeg_ehr/qc/build_bipolar_mask.py',
                           params=params,
                           parents=[str(excl_path), str(rv_dir)],
                           subjects=[subject_id])
            written.append(path)

    io.write_manifest(out_dir, script='ieeg_ehr/qc/build_bipolar_mask.py',
                      params=params,
                      subjects=have_both,
                      extra={'n_subject_sessions': len(written),
                             'skipped_no_raw_voltage': only_excl,
                             'skipped_no_bipolar_variance': only_rv,
                             'columns': OUT_COLUMNS})
    logger.info('Wrote %d subject-session masks -> %s', len(written), out_dir)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bipolar-variance-label', default='std10',
                    help='Which bipolar_variance exclusion label to roll up. The '
                         'THRESHOLD is not re-computed here -- it is whatever this '
                         'label already encodes. (default: %(default)s)')
    ap.add_argument('--raw-voltage-mask-label', default=None,
                    help=f'Raw-voltage mask label (default: pinned {config.CANONICAL_MASK_LABEL})')
    ap.add_argument('--label', default=None,
                    help='Output mask label (default: auto, encoding BOTH inputs)')
    ap.add_argument('--subjects', nargs='+', default=None,
                    help='Restrict to these subjects (default: every subject with both inputs)')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    rv_label = args.raw_voltage_mask_label or config.CANONICAL_MASK_LABEL
    out_label = args.label or config.bipolar_mask_label(args.bipolar_variance_label, rv_label)
    io.warn_if_dirty()
    logger.info('=== build_bipolar_mask: %s + rv-%s -> %s ===',
                args.bipolar_variance_label, rv_label, out_label)
    run(args.bipolar_variance_label, rv_label, out_label, subjects=args.subjects)


if __name__ == '__main__':
    main()
