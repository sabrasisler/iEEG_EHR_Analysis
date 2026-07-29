#!/usr/bin/env python3
"""
Feature-level QC, mask stage (Step B): the COMPLETE per-window exclusion mask --
every feature-level artifact type OR'd with the upstream (bipolar) mask. This is
the one artifact the view layer joins onto the epoch cache.

THE UNION IS THE POINT
----------------------
`excluded` is the FULL union: feature-level types OR the upstream mask named in
params.mask_label. A consumer needs that one column and nothing else, and must NOT
have to remember to also join qc/bipolar/masks/ itself.

An earlier version of this script emitted only feature-flagged rows and carried the
upstream verdict as a passenger column. That was a trap: for sub-039 the file held
7,124 rows while the upstream mask excluded 127,740 channel-windows, so 95% of the
upstream exclusions were simply ABSENT -- and because a window absent from a sparse
mask reads as "not excluded", anything joining it would have kept them while
looking authoritative. Cohort-wide the upstream mask excludes 2.505% of
channel-windows against the feature detector's 0.666%, so the omitted part was the
larger part.

Expanding the upstream mask from its 60s bins out to windows uses each run's exact
PSD rate/starting_time, never an assumption that they are 1.0/0.0 -- see
upstream_excluded_rows.

Cheap: it reads already-thresholded exclusion tables plus HDF5 metadata (never PSD
data), so a threshold sweep on one type = re-run that type's
build_feature_exclusions plus this, never the metric pass and never the other types.

Directly parallel to qc/build_mask.py at the raw-voltage level, with two
differences, both consequences of what this level's exclusions ARE:

  1. PER-WINDOW, NOT PER-60s-BIN. build_mask.py rolls up to 60s bins because its
     inputs are 2s detectors and its consumer (bipolar re-referencing) wants 60s.
     Here the consumer is the per-window epoch cache, so rolling up to 60s would
     throw away resolution the cache has. Join key is
     (run_id, channel, window_idx).

  2. SPARSE. The per-type inputs contain only excluded windows, so the union does
     too: a window absent from this file is NOT excluded by any type. That is
     safe because build_feature_exclusions enforces K > FEATURE_METRIC_STORE_FLOOR
     (see its docstring). Denominators live in metrics/summary/, linked from
     params.json.

With a single artifact type registered (power_outlier) this step is close to a
copy, and that is deliberate: it is the seam where a second detector -- a
`nonfinite` type for windows with any non-finite bin, or a spectral-shape
detector -- ORs in without any consumer changing. The view layer should depend on
masks/<label>/, never on exclusions/<type>/<label>/, for exactly that reason.

Output per subject/session: masks/<label>/sub-XXX_ses-YY.parquet with the join key
(run_id, channel, window_idx), window_start_time, one boolean per feature type
(excluded_<type>), `excluded_upstream` for the bipolar mask's contribution, and
`excluded` = the full OR of all of them. The per-source columns are there so a
window's exclusion can be attributed; `excluded` is what to filter on. A
params.json records which <type>/<label> and which upstream mask fed it, + git.

Usage:
  python -m ieeg_ehr.qc.feature_level.build_feature_mask --label featqc-z5binfrac20
  python -m ieeg_ehr.qc.feature_level.build_feature_mask --power_outlier z4_binfrac10
"""

import argparse
import json
import logging
import os
import re

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.qc import mask_projection

logger = logging.getLogger(__name__)

KEY = ['run_id', 'channel', 'window_idx']

_TAG_RE = re.compile(r'^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)$')


def _parse_tag(path):
    m = _TAG_RE.match(path.stem)
    if not m:
        raise ValueError(f'Unexpected exclusion filename: {path.name}')
    return m.group('subject'), m.group('session')


def _default_label(chosen):
    """A mask label that names its inputs, e.g.
    'featqc-z5_binfrac20_bp-std10_rv-gross-std3_satmargin15_sw_logz4'.

    Same instinct as build_exclusions.label_for: the path should say what produced
    it rather than 'default'. The per-type exclusion labels already carry the
    upstream mask scope, so this inherits it -- which is what keeps a
    bipolar-scoped mask from overwriting a raw-voltage-scoped one.

    Underscores are NOT stripped (an earlier version did): the scope contains them,
    and mangling it would defeat the point of naming it. Multi-type masks join with
    '+', so the name grows with the number of types -- pass --label explicitly once
    that gets unwieldy.
    """
    return 'featqc-' + '+'.join(lbl for _t, lbl in sorted(chosen.items()))


def upstream_excluded_rows(subject, session, mask_level, mask_label):
    """Every (run, channel, window) the UPSTREAM mask excludes, at window granularity.

    The upstream mask is stored per 60s bin, but this artifact is per window, so the
    60s verdicts are expanded out. The window grid comes from each run's PSD
    metadata (exact rate/starting_time), NOT from assuming `starting_time == 0` and
    `rate == 1` -- that assumption is what let the 60s-hop runs go unnoticed, and it
    is cheap to avoid since only HDF5 metadata is read, never the data.

    Returns an empty frame (right columns, no rows) when there is no upstream mask,
    so the caller's union degrades to feature-only rather than crashing -- but the
    caller MUST log that, because a missing upstream mask silently keeps artifact
    windows.
    """
    from ieeg_ehr.qc.feature_level.detect_power_outlier import psd_run_meta

    empty = pd.DataFrame({'run_id': pd.Series(dtype=object),
                          'channel': pd.Series(dtype=object),
                          'window_idx': pd.Series(dtype='int32'),
                          'window_start_time': pd.Series(dtype='float64'),
                          'excluded_upstream': pd.Series(dtype=bool)})
    mask_path = (config.bipolar_mask_path(subject, session, mask_label)
                 if mask_level == 'bipolar' else config.mask_csv(subject, session, mask_label))
    mask_df = mask_projection.load_mask(mask_path)
    if mask_df is None:
        return empty, mask_path

    project = (mask_projection.project_pair_mask_to_windows if mask_level == 'bipolar'
               else mask_projection.project_to_pairs)

    registry = pd.read_csv(config.FILE_REGISTRY_CSV)
    rows = registry[(registry.sub_id == f'sub-{subject}') & (registry.ses_id == f'ses-{session}')]
    frames = []
    for r in rows.run_id.unique():
        rid = str(r).replace('run-', '')
        path = config.bipolar_psd_nwb_path(subject, session, rid)
        if not path.exists():
            continue
        meta = psd_run_meta(path)
        excl = project(mask_df, f'run-{rid}', meta['channels'], meta['run_seconds'])
        if not excl.any():
            continue
        wi, pi = np.nonzero(excl)
        frames.append(pd.DataFrame({
            'run_id': f'run-{rid}',
            'channel': np.asarray(meta['channels'], dtype=object)[pi],
            'window_idx': wi.astype(np.int32),
            'window_start_time': meta['run_seconds'][wi],
            'excluded_upstream': True,
        }))
    return (pd.concat(frames, ignore_index=True) if frames else empty), mask_path


def build_one(subject, session, chosen, out_path, mask_label_rv, mask_level):
    merged = None
    per_type_counts = {}
    for artifact_type, label in sorted(chosen.items()):
        path = config.feature_exclusion_path(subject, session, artifact_type, label)
        cols = ['run_id', 'channel', 'window_idx', 'window_start_time']
        df = io.read_table(path, columns=cols + ['excluded'])
        df = df.rename(columns={'excluded': f'excluded_{artifact_type}'})
        per_type_counts[artifact_type] = int(df[f'excluded_{artifact_type}'].sum())
        if merged is None:
            merged = df
        else:
            # Outer join: each type's sparse table covers a different set of
            # windows, and a window missing from one type is simply not excluded
            # by that type (-> False), not absent from the mask.
            merged = merged.merge(df.drop(columns=['window_start_time']),
                                  on=KEY, how='outer')

    # THE UNION. Without this the mask would contain only feature-flagged windows,
    # and anything joining it would silently KEEP every window the upstream mask
    # excluded -- measured at 95% of sub-039's upstream exclusions missing, i.e. the
    # mask would have been worse than useless: it would look authoritative.
    upstream, upstream_path = upstream_excluded_rows(subject, session, mask_level, mask_label_rv)
    n_upstream = len(upstream)
    if n_upstream == 0:
        logger.warning('  sub-%s ses-%s: NO upstream mask rows from %s -- this mask is '
                       'feature-only and does NOT exclude upstream artifacts',
                       subject, session, upstream_path)
    merged = merged.merge(upstream, on=KEY, how='outer', suffixes=('', '_up'))
    # window_start_time comes from whichever side had the row.
    if 'window_start_time_up' in merged.columns:
        merged['window_start_time'] = merged['window_start_time'].fillna(
            merged['window_start_time_up'])
        merged = merged.drop(columns=['window_start_time_up'])

    excl_cols = [f'excluded_{t}' for t in sorted(chosen)] + ['excluded_upstream']
    for c in excl_cols:
        merged[c] = merged[c].fillna(False).astype(bool) if c in merged.columns else False
    # `excluded` is the FULL union: feature-level types OR the upstream mask. A
    # consumer should need this column and nothing else.
    merged['excluded'] = merged[excl_cols].any(axis=1)
    merged = merged.sort_values(KEY).reset_index(drop=True)
    out_cols = KEY + ['window_start_time'] + excl_cols + ['excluded']

    summary_path = config.feature_metrics_path('summary', subject, session,
                                               mask_label_rv, mask_level)
    n_windows_total = None
    if summary_path.exists():
        n_windows_total = int(io.read_table(summary_path, columns=['n_windows'])['n_windows'].sum())

    n_excl = int(merged['excluded'].sum())
    n_inc = int((merged['excluded'] & ~merged['excluded_upstream']).sum())
    io.write_table(merged[out_cols], out_path,
                   params={'types': sorted(chosen), 'per_type_labels': chosen,
                           'level': 'window', 'mask_level': mask_level,
                           'mask_label': mask_label_rv},
                   parents=[str(config.feature_exclusion_path(subject, session, t, l))
                            for t, l in sorted(chosen.items())],
                   subjects=[f'sub-{subject}'],
                   extra={'counts': {'n_excluded': n_excl,
                                     'n_excluded_not_upstream': n_inc,
                                     'n_upstream_excluded': n_upstream,
                                     'n_windows_total': n_windows_total,
                                     'per_type': per_type_counts},
                          'note': 'SPARSE UNION: contains every window excluded by a '
                                  'feature-level type OR by the upstream mask named in '
                                  "params.mask_label. `excluded` is the full union -- a "
                                  'consumer needs that column and nothing else. A window '
                                  'absent from this file is not excluded by anything. Join '
                                  'key (run_id, channel, window_idx); window_idx indexes '
                                  "that RUN's PSD rows, not epoch-relative."})

    rate = (100.0 * n_excl / n_windows_total) if n_windows_total else float('nan')
    logger.info('  sub-%s ses-%s: %d rows, %d excluded (%.3f%% of %s) '
                '[upstream=%d, %s, feature-only=%d]',
                subject, session, len(merged), n_excl, rate, n_windows_total,
                n_upstream, ', '.join(f'{t}={n}' for t, n in per_type_counts.items()), n_inc)
    return {'subject': subject, 'session': session, 'n_excluded': n_excl,
            'n_excluded_not_upstream': n_inc, 'n_upstream': n_upstream,
            'n_windows_total': n_windows_total}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', default=None,
                    help='Mask label / output folder (default: auto from the inputs, '
                         'e.g. featqc-z5binfrac20)')
    ap.add_argument('--types', default=None,
                    help=f'Comma-separated subset of feature-level artifact types to combine '
                         f'(default: all {config.FEATURE_ARTIFACT_TYPES})')
    for t in config.FEATURE_ARTIFACT_TYPES:
        ap.add_argument(f'--{t}', default=None,
                        help=f'Which {t} exclusion label to combine '
                             f'(default: config-default {config.feature_exclusion_label()})')
    ap.add_argument('--mask-level', default=config.FEATURE_BASELINE_MASK_LEVEL,
                    choices=sorted(config.FEATURE_MASK_LEVEL_PREFIX),
                    help='Which mask level the metrics were scoped by '
                         f'(default: {config.FEATURE_BASELINE_MASK_LEVEL}).')
    ap.add_argument('--mask-label', default=None,
                    help='Mask label within that level, used to locate the summary tables '
                         'for denominators; default depends on the level.')
    ap.add_argument('--subjects', default=None,
                    help='Comma-separated subject IDs to restrict to (default: all present)')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    types = ([t.strip() for t in args.types.split(',')] if args.types
             else list(config.FEATURE_ARTIFACT_TYPES))
    bad = [t for t in types if t not in config.FEATURE_ARTIFACT_TYPES]
    if bad:
        raise SystemExit(f'--types: unknown feature-level artifact type(s) {bad}, '
                         f'must be from {config.FEATURE_ARTIFACT_TYPES}')

    mask_level = args.mask_level
    if args.mask_label is None:
        mask_label_rv = (config.bipolar_mask_label(config.FEATURE_BASELINE_BIPOLAR_VARIANCE_LABEL)
                         if mask_level == 'bipolar' else config.CANONICAL_MASK_LABEL)
    else:
        mask_label_rv = args.mask_label
    if mask_label_rv == 'none':
        mask_label_rv = None

    scope = config.feature_mask_scope(mask_label_rv, mask_level)
    chosen = {t: (getattr(args, t) or config.feature_exclusion_label(scope=scope))
              for t in types}
    type_dirs = {t: config.exclusion_dir(config.FEATURE_LEVEL_ROOT, t, lbl)
                 for t, lbl in chosen.items()}
    for t, d in type_dirs.items():
        if not d.exists():
            raise SystemExit(f'Missing exclusion dir for {t}: {d} '
                             '(run build_feature_exclusions first)')

    # Only subject/sessions present for EVERY chosen type, so a mask can never be
    # a partial union that silently omits one detector -- same rule build_mask.py
    # applies at the raw-voltage level, and the reason the sub-236 gap surfaces as
    # a skip rather than a wrong mask.
    per_type = {t: {_parse_tag(p) for p in d.glob('sub-*_ses-*.parquet')}
                for t, d in type_dirs.items()}
    common = set.intersection(*per_type.values()) if per_type else set()
    for t, s in per_type.items():
        extra = s - common
        if extra:
            logger.warning('  %s has subject/sessions not shared by all types (skipped): %s',
                           t, sorted(extra))
    if args.subjects:
        wanted = {x.strip().replace('sub-', '') for x in args.subjects.split(',')}
        common = {(sub, ses) for sub, ses in common if sub in wanted}
    if not common:
        raise SystemExit('No subject/sessions common to all chosen types.')

    label = args.label or _default_label(chosen)
    out_dir = config.mask_dir(config.FEATURE_LEVEL_ROOT, label)
    out_dir.mkdir(parents=True, exist_ok=True)

    io.warn_if_dirty()
    logger.info('=== build_feature_mask: label=%s types=%s ===', label, chosen)

    rows = []
    for subject, session in sorted(common):
        out_path = config.feature_mask_path(subject, session, label)
        try:
            rows.append(build_one(subject, session, chosen, out_path,
                                  mask_label_rv, mask_level))
        except Exception:
            logger.exception('  sub-%s ses-%s: failed, skipping', subject, session)

    total_excl = sum(r['n_excluded'] for r in rows)
    total_inc = sum(r['n_excluded_not_upstream'] for r in rows)
    total_up = sum(r['n_upstream'] for r in rows)
    total_win = sum(r['n_windows_total'] or 0 for r in rows)
    params_out = {
        'mask_label': label,
        'level': 'window',
        'join_key': KEY,
        'types': sorted(chosen),
        'per_type_labels': chosen,
        'per_type_dirs': {t: str(d) for t, d in type_dirs.items()},
        'mask_level': mask_level,
        'mask_label': mask_label_rv,
        'mask_scope': scope,
        'metrics_summary_dir': str(config.feature_metrics_dir('summary', mask_label_rv,
                                                              mask_level)),
        'n_subject_sessions': len(rows),
        'totals': {'n_excluded': total_excl,
                   'n_excluded_not_upstream': total_inc,
                   'n_upstream_excluded': total_up,
                   'n_windows_total': total_win,
                   'pct_excluded': (100.0 * total_excl / total_win) if total_win else None},
        'sparse': True,
        'run_timestamp': config.run_timestamp(),
        'git': config.git_provenance(),
        'note': 'Feature-level window mask. This is the artifact the view layer should '
                'join onto the epoch cache; do not depend on exclusions/<type>/ directly. '
                'The epoch cascade (X, Y, Z) is applied in the view layer, not here.',
    }
    tmp = out_dir / f'params.json.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(params_out, indent=2, default=str))
    os.replace(tmp, out_dir / 'params.json')

    logger.info('%d subject/sessions -> %s | %d excluded channel-windows of %d (%.3f%%), '
                'incremental over the %s mask %d',
                len(rows), out_dir, total_excl, total_win,
                (100.0 * total_excl / total_win) if total_win else float('nan'),
                mask_level, total_inc)


if __name__ == '__main__':
    main()
