"""The bit-exact gate for the continuous native-grid PSD. Run on the pilot output
BEFORE the array.

    python -m ieeg_ehr.preprocessing.bipolar_psd_fullres_audit \
        --runs 019:01:CA0652B0 208:01:EA1897F1

Per run, on the first, middle and last block of `run_pipeline_bipolar`:

CHECK A -- REPRODUCE THE RETIRED 50-BIN NWB. Re-read the raw block exactly as the
extractor does, reduce the linear spectrum with the PRODUCTION 50-bin functions
(`bipolar_reref.log_bin_edges` + `_band_average_linear`, via
`fullres_psd_audit._rebin_from_sxx`), and compare against the same rows of the
50-bin NWB on $GROUP_SCRATCH. Must be 0 ulp. That one comparison proves the block
offsets, the last-block length, the pair order and the window count against the
old whole-run pipeline.

CHECK B -- THE WRITE PATH. The stored block equals float32 log10 of the
recomputed linear block, exactly. Proves the transpose, the chunk offsets and the
log, which check A never reads.

Also asserts the window count, the pair count and the pair names agree with the
50-bin file.
"""

import argparse
import logging
import sys

import h5py
import numpy as np

from ieeg_ehr import config
from ieeg_ehr.features import fullres_psd_audit
from ieeg_ehr.preprocessing import bipolar_reref
from ieeg_ehr.preprocessing import run_pipeline_bipolar as rpb

logger = logging.getLogger(__name__)

OLD_DATA_PATH = 'processing/ecephys/psd_log_bins/data'


def audit_run(subject, session, run, raw_path):
    """Returns {'block_start', 'check_a_max_ulp', 'check_b_exact'} per audited block."""
    new_path = config.bipolar_psd_fullres_nwb_path(subject, session, run)
    old_path = config.bipolar_psd_nwb_path(subject, session, run)
    edges = bipolar_reref.log_bin_edges(config.PSD_N_LOG_BINS, config.PSD_FREQ_MIN_HZ,
                                        config.PSD_FREQ_MAX_HZ)
    out = []
    with (rpb.RawRun(raw_path) as raw, h5py.File(new_path, 'r') as fn,
          h5py.File(old_path, 'r') as fo):
        new, old = fn[rpb.DATA_PATH], fo[OLD_DATA_PATH]
        if new.shape[:2] != old.shape[:2] or new.shape[0] != raw.n_windows:
            raise AssertionError(f'shape: new {new.shape}, 50-bin {old.shape}, '
                                 f'raw implies {raw.n_windows} windows')
        names_new = fn[rpb.ELECTRODE_LOCATION_PATH].asstr()[:]
        names_old = fo[rpb.ELECTRODE_LOCATION_PATH].asstr()[:]
        if list(names_new) != list(names_old):
            raise AssertionError(f'pair names differ, first new {names_new[:3]}, '
                                 f'first 50-bin {names_old[:3]}')

        starts = list(rpb.block_starts(raw.n_windows))
        for w0 in sorted({starts[0], starts[len(starts) // 2], starts[-1]}):
            w1 = min(w0 + config.PSD_FULLRES_HDF5_CHUNK_WINDOWS, raw.n_windows)
            bipolar_v = raw.bipolar_block(*raw.block_span(w0, w1))
            # The full grid, because production binned the full grid.
            freqs_full, linear = raw.linear_block(bipolar_v, w1 - w0, None)
            rebinned = fullres_psd_audit._rebin_from_sxx(freqs_full, linear, edges)
            ulp = fullres_psd_audit._max_ulps(rebinned, old[w0:w1])

            with np.errstate(divide='ignore'):
                expected = np.log10(linear[raw.freq_sl]).transpose(1, 2, 0)
            exact = bool(np.array_equal(new[w0:w1], expected, equal_nan=True))
            out.append({'block_start': w0, 'check_a_max_ulp': ulp, 'check_b_exact': exact})
            logger.info('sub-%s run-%s windows %d-%d: check A %d ulp, check B %s',
                        subject, run, w0, w1, ulp, 'exact' if exact else 'MISMATCH')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, help='subject:session:run tokens')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    reg = rpb._registry(list(dict.fromkeys(t.split(':')[0] for t in args.runs)))
    raw_paths = {rpb._token(row): row.raw_file_path for row in reg.itertuples()}
    ok = True
    for token in args.runs:
        subject, session, run = token.split(':')
        for r in audit_run(subject, session, run, raw_paths[token]):
            ok &= r['check_a_max_ulp'] == 0 and r['check_b_exact']
    print('GATE PASSED' if ok else 'GATE FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
