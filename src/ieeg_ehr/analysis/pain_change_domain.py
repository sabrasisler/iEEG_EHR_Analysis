"""pain_change frames for the lme4 PROCESSING DOMAIN model with medication.

    python -m ieeg_ehr.analysis.pain_change_domain \\
        --mask-level bipolar --mask-label std10_rv-gross-std3_satmargin15_sw_logz4

Builds one frame per band of `--band-set`, rows = assessment pair x channel,
for `run_domain_lmer` to fit as

    d_z ~ 0 + domain:med + domain:med:d_pain + domain:pain_1_within
          + domain:gap_h + med_submean
          + (1 + d_pain || ROI) + (1 + d_pain + med_within || subject)
          + (1 + d_pain || subj_roi)

`d_z`, the pairs and the band averaging are `pain_change`'s. `med` is
`med_between`: any analgesic administered in [t1, t2) (`change_score.build_pairs`,
DECISIONS 2026-10-06). Channels map to ROIs of the domain scheme's base scheme
through `roi_maps`, with the insula cut pinned by `--insula-threshold`, then to
domains through `run_domain_model.roi_domain_maps`, the same lookup as the
level-model domain runs.

Frames land in `analysis/pain/pain_change/domain_model/<scheme>/<run>_<timestamp>/frames/`.
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import change_score, fullres_cells, med_state
from ieeg_ehr.analysis.pain_change import discovery_paths, session_rows
from ieeg_ehr.analysis.run_bandpower_mixed import BAND_SETS
from ieeg_ehr.analysis.run_domain_model import roi_domain_maps
from ieeg_ehr.analysis.run_mixed_model_pilot import roi_maps
from ieeg_ehr.config import roi_schemes
from ieeg_ehr.views import view_config
from ieeg_ehr.views import build_pain_epoch_fullres_zscore as zview

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/pain_change_domain.py'
OUTPUT_TYPE = 'domain_model'
RUN_NAME = 'lmer_frames'
#: The level-model domain runs' pinned cut (submit_domain_lmer_med.sh).
INSULA_THRESHOLD = -2.245993821300736

PAIR_COLUMNS = ['pair_id', 'd_pain', 'pain_1', 'gap_h', 'med_between']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-scheme', default='pain_domains_v3')
    ap.add_argument('--insula-threshold', type=float, default=INSULA_THRESHOLD)
    ap.add_argument('--drug-set', default='analgesics',
                    choices=sorted(med_state.DRUG_SETS))
    ap.add_argument('--min-gap-min', type=float, default=change_score.DEFAULT_MIN_GAP_MIN)
    ap.add_argument('--max-gap-min', type=float, default=change_score.DEFAULT_MAX_GAP_MIN)
    ap.add_argument('--notch-half-width-hz', type=float, default=None)
    ap.add_argument('--band-set', choices=list(BAND_SETS), default='paper_bands_6_hg200')
    ap.add_argument('--run-name', default=RUN_NAME)
    view_config.add_view_arguments(ap)
    ap.set_defaults(baseline='all_pain_epochs', freq='fullres', region='none')
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    vc = view_config.from_args(args)
    zdir = zview.zscore_dir(vc)
    paths = discovery_paths(zdir)
    subjects = {f'sub-{fullres_cells.subject_session_of(p)[0]}' for p in paths}

    split_report = {}
    base = roi_schemes.domain_scheme(args.domain_scheme)['base']
    roi_by_subject, no_roi = roi_maps(paths, subjects, base,
                                      insula_threshold=args.insula_threshold,
                                      report=split_report)
    roi_of, roi_to_domain, _, unassigned = roi_domain_maps(
        roi_by_subject, subjects, args.domain_scheme)

    admin = med_state.load_admin_table(subclasses=med_state.DRUG_SETS[args.drug_set])
    defs = med_state.load_epoch_defs(subjects=sorted(subjects))
    pairs = change_score.build_pairs(defs, admin, args.min_gap_min, args.max_gap_min)

    freq_table, notch = fullres_cells.analysis_freq_table(args.notch_half_width_hz)
    bands = BAND_SETS[args.band_set]

    index_parts, value_parts = [], []
    for path in paths:
        subject, session = fullres_cells.subject_session_of(path)
        sid = f'sub-{subject}'
        sp = pairs[(pairs['subject'] == subject) & (pairs['session'] == session)]
        if sid not in roi_of or sp.empty:
            logger.warning('%s ses-%s: %s, skipped', sid, session,
                           'no pairs' if sp.empty else 'no channel in a domain')
            continue
        idx, vals = session_rows(path, sp.reset_index(drop=True), roi_of[sid],
                                 freq_table, bands)
        if idx is not None:
            index_parts.append(idx)
            value_parts.append(vals)
    index = pd.concat(index_parts, ignore_index=True).rename(columns={'region': 'parcel'})
    index['domain'] = index['parcel'].map(roi_to_domain)
    values = np.vstack(value_parts)
    cohort = sorted(index['subject'].unique())
    logger.info('%d pair x channel rows, %d subjects, %d ROIs, %d domains',
                len(index), len(cohort), index['parcel'].nunique(),
                index['domain'].nunique())

    scheme = f'{args.band_set.replace("_", "")}-{args.domain_scheme.replace("_", "")}-{args.drug_set}'
    run_dir = config.analysis_run_dir(
        question=config.PAIN_CHANGE_QUESTION, output_type=OUTPUT_TYPE,
        run_name=args.run_name, view_scheme=scheme)
    params = {'domain_scheme': args.domain_scheme, 'base_roi_scheme': base,
              'insula_threshold': args.insula_threshold, 'drug_set': args.drug_set,
              'med': 'med_between: any dose of the drug set in [t1, t2)',
              'band_set': args.band_set, 'bands': bands,
              'min_gap_min': args.min_gap_min, 'max_gap_min': args.max_gap_min,
              'notch_bins_removed': notch, 'zscore_dir': str(zdir),
              'view_config': vc.provenance()}
    parents = [str(zdir), str(med_state.ADMIN_TABLE)]

    for j, band in enumerate(bands):
        df = index.assign(d_z=values[:, j])
        df = df[np.isfinite(df['d_z'])]
        df = df.merge(pairs[PAIR_COLUMNS], on='pair_id')
        df['pain_1_within'] = df['pain_1'] - df.groupby('subject')['pain_1'].transform('mean')
        io.write_table(df.reset_index(drop=True), run_dir / 'frames' / f'{band}.parquet',
                       params={**params, 'band': band}, parents=parents,
                       subjects=cohort, script=SCRIPT)
        logger.info('[%s] %d rows, %d dosed pairs of %d', band, len(df),
                    df.loc[df['med_between'] == 1, 'pair_id'].nunique(),
                    df['pair_id'].nunique())

    io.write_table(pairs[pairs['subject_id'].isin(cohort)], run_dir / 'pair_index.csv',
                   params=params, parents=[str(med_state.ADMIN_TABLE)],
                   subjects=cohort, script=SCRIPT)
    io.write_run_provenance(
        run_dir, script=SCRIPT, params=params, parents=parents, subjects=cohort,
        extra={'status': 'model frames for run_domain_lmer, not a result',
               'subjects_without_roi': sorted(no_roi),
               'rois_unassigned_dropped': unassigned, **split_report})
    print(run_dir / 'frames')
    return 0


if __name__ == '__main__':
    sys.exit(main())
