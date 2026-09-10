"""The (epoch x channel-band) feature matrix for ONE subject-session.

Reads a materialized per-channel, six-band view and turns its long format into
the dense matrix a model consumes. Nothing here normalizes, standardizes or
imputes -- all three are fitted inside the training fold by the model pipeline
(`cv.py`), because all three estimate something from data.

WHAT A "UNIT" IS
----------------
One subject-session = one feature matrix = one model. Sessions are modelled
separately rather than pooled because a different session is a different
implant: the channel set is not the same, so the FEATURES are not the same, and
concatenating them would silently align column 7 of one montage with column 7 of
another.

WHY THE ROI IS NOT USED HERE
----------------------------
Features are per-CHANNEL. The ROI travels as metadata for interpreting which
features got selected, and is joined at analysis time -- never averaged into the
features. Averaging within an ROI would cancel opposite-signed contacts (the
target paper found insula reversing direction relative to everywhere else), and
would push the known-approximate anode-based DK assignment
(docs/view_registry.md) out of the labels and into the data itself.

MISSINGNESS
-----------
The view drops all-NaN cells to keep its table small, so a channel-epoch the QC
mask killed is simply ABSENT from the long format rather than present-and-NaN.
Pivoting therefore reconstructs the missingness pattern for free. Distinguishing
"masked out" from "never recorded" needs the run montages, which is why this
module reads channel_meta -- see cascade.py for why that distinction matters.
"""

import logging

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.decoding import cascade
from ieeg_ehr.views import cache_reader, channel_meta

logger = logging.getLogger(__name__)

#: What the decoder requires of each accepted FEATURE SOURCE. Checked by NAME
#: with a specific message rather than only by config hash, so a mismatch says
#: WHICH axis is wrong instead of "hash differs".
#:
#: Two sources are legitimate and they are NOT interchangeable in their fields:
#:
#:  psd_view   the materialized view of the bipolar Welch-PSD cache. Bands come
#:             from aggregating stored bins, so the line-noise flag matters.
#:  laplacian_bandpass_rms
#:             a fresh extraction off the raw time-domain signal (notch ->
#:             Laplacian -> Butterworth -> RMS). It has no view axes at all --
#:             `freq`/`normalization`/`region` are meaningless for it, and line
#:             noise is handled by the notch rather than by dropping bins.
#:
#: Channel sets differ between them and can never be joined: Laplacian names a
#: channel for its centre contact ('LA2'), bipolar names a pair ('LA1-LA2').
REQUIRED_AXES_BY_SOURCE = {
    'psd_view': {
        'freq': 'paper_bands_6',
        'normalization': 'none',
        'region': 'none',
        'drop_line_noise_bins': True,
    },
    'laplacian_bandpass_rms': {
        'source': 'laplacian_bandpass_rms',
        'reref': 'laplacian',
        'statistic': 'log10_rms',
    },
}

#: Kept for callers that still reference the single-source constant.
REQUIRED_VIEW_AXES = REQUIRED_AXES_BY_SOURCE['psd_view']


class ViewMismatchError(RuntimeError):
    """The view is not the one this analysis is defined on.

    Loud and its own type because every failure it guards against is SILENT: the
    50-bin view has the same file names and would pivot into a perfectly
    plausible 7,650-column matrix, and a baseline-normalized view would leak the
    labels into every cross-validated score.
    """


class FeatureMatrix:
    """X, y, and the provenance needed to explain either.

    `X` carries NaN where the cascade left a residual cell; imputation happens
    in-fold, so a NaN here is not an error -- it is information the pipeline is
    told about.
    """

    def __init__(self, subject, session, X, y, feature_names, channels, bands,
                 epoch_ids, pain_scores, report):
        self.subject, self.session = subject, session
        self.X, self.y = X, y
        self.feature_names = feature_names
        self.channels, self.bands = channels, bands
        self.epoch_ids = epoch_ids
        self.pain_scores = pain_scores
        self.report = report

    @property
    def unit(self):
        return f'sub-{self.subject}_ses-{self.session}'

    @property
    def shape(self):
        return self.X.shape

    def __repr__(self):
        n, p = self.X.shape
        return (f'<FeatureMatrix {self.unit}: {n} epochs x {p} features '
                f'({len(self.channels)} channels x {len(self.bands)} bands), '
                f'p/n={p / max(n, 1):.1f}>')


def verify_view(view_dir, epoch_path, on_stale='refuse'):
    """Read the view's own sidecar and refuse if it is not the required view.

    From the ARTIFACT, never from a CLI flag -- the same rule the plot scripts
    follow. `on_stale='refuse'` because reported numbers come out of this;
    check_commit stays off, so a later unrelated commit does not invalidate a
    perfectly good view.
    """
    sidecar = io.read_sidecar(epoch_path) or {}
    params = sidecar.get('params', {})
    if not params:
        raise ViewMismatchError(
            f'{epoch_path} has no readable provenance sidecar, so there is no way '
            'to confirm which view it is. Refusing rather than assuming.')

    # The sidecar's own `source` decides which contract applies. Absent means the
    # PSD view, which predates the field.
    source = params.get('source', 'psd_view')
    required = REQUIRED_AXES_BY_SOURCE.get(source)
    if required is None:
        raise ViewMismatchError(
            f'{epoch_path} declares source={source!r}, which this analysis does '
            f'not know how to consume. Known: {sorted(REQUIRED_AXES_BY_SOURCE)}')

    wrong = {k: params.get(k) for k, want in required.items()
             if params.get(k) != want}
    if wrong:
        hint = ('Build it with sbatch/build_paper6_view_array.sbatch. In particular '
                'drop_line_noise_bins MUST be True: the paper bands straddle '
                '60/120 Hz and views.axes.aggregate_bands does not exclude flagged '
                'bins itself.') if source == 'psd_view' else (
                'Build it with sbatch/build_laplacian_bandpass_array.sbatch.')
        raise ViewMismatchError(
            f'{view_dir.name} is not a feature table this analysis is defined on.\n'
            f'  source:   {source}\n'
            f'  required: {required}\n'
            f'  found:    {wrong}\n' + hint)

    io.assert_fresh(epoch_path, on_stale=on_stale)
    return params


def _contacts_from_pairs(pair_names):
    """{contact} from bipolar pair names: 'LA1-LA2' -> {'LA1', 'LA2'}.

    channel_meta only ever reports the BIPOLAR montage, because that is what the
    PSD derivative stored. The underlying contact set is recoverable from it, and
    is what a Laplacian channel has to be checked against.
    """
    contacts = set()
    for name in pair_names:
        contacts.update(name.split('-'))
    return contacts


def _recorded_matrix(subject, session, defs, channels, epoch_minutes=None,
                     source='psd_view'):
    """(n_epochs, n_channels) bool: was this channel in this epoch's run montage.

    Only 2 of 45 discovery units have run-varying montages, so this is all-True
    for most of the cohort -- but for those two it is the difference between
    dropping 60 channels as "bad" and correctly calling them uncovered.

    MUST BE SOURCE-AWARE, and this is exactly where it was got wrong once. The
    two feature sources name channels differently -- bipolar 'LAMY1-LAMY2' vs
    Laplacian 'LAMY2' -- and `channel_meta` only ever reports the BIPOLAR names.
    Looking a Laplacian channel up in that set matches NOTHING, so `recorded`
    came out all-False, the coverage rule found no common channel, and all 45
    subjects raised NoUsableDataError. Every task still exited 0.
    """
    runs = list(defs['run_id'].unique())
    meta = channel_meta.build(subject, session, runs, epoch_minutes)
    per_run_pairs = {run: set(channel_meta.channels_for_run(meta, run)[0])
                     for run in runs}

    if source == 'psd_view':
        per_run = per_run_pairs
        def is_present(ch, present):
            return ch in present
    else:
        # A Laplacian channel needs its CENTRE contact and BOTH shaft
        # neighbours; that is how the extractor built it, so that is what
        # "recorded" means here.
        from ieeg_ehr.preprocessing.bipolar_reref import parse_electrode_shaft
        per_run = {run: _contacts_from_pairs(pairs)
                   for run, pairs in per_run_pairs.items()}

        def is_present(ch, present):
            shaft, num = parse_electrode_shaft(ch)
            if shaft is None:
                return False
            return all(f'{shaft}{n}' in present for n in (num - 1, num, num + 1))

    index = {c: i for i, c in enumerate(channels)}
    recorded = np.zeros((len(defs), len(channels)), dtype=bool)
    for i, run in enumerate(defs['run_id'].to_numpy()):
        present = per_run.get(run)
        if present is None:
            continue                                   # no metadata -> nothing recorded
        cols = [index[c] for c in channels if is_present(c, present)]
        recorded[i, cols] = True
    return recorded


def build_matrix(subject, session, view_dir, epoch_minutes=None,
                 channel_max_bad=None, epoch_max_bad=None, min_epochs=30,
                 on_stale='refuse'):
    """Assemble one subject-session's feature matrix. See FeatureMatrix."""
    from pathlib import Path
    view_dir = Path(view_dir)
    epoch_path = view_dir / f'view_epochs_sub-{subject}_ses-{session}.parquet'
    if not epoch_path.exists():
        raise FileNotFoundError(
            f'no view table at {epoch_path}. Build this unit first '
            '(sbatch/build_paper6_view_array.sbatch).')

    view_params = verify_view(view_dir, epoch_path, on_stale=on_stale)
    long = io.read_table(epoch_path, on_stale='ignore')   # already checked above

    # Band ORDER comes from the band definition, not from the table, so the
    # column order is frequency-ascending and identical across units. A dict
    # preserves insertion order, and PAPER_BANDS_6_HZ is written low -> high.
    bands = [b for b in config.PAPER_BANDS_6_HZ if b in set(long['freq_bin_index'])]
    missing_bands = [b for b in config.PAPER_BANDS_6_HZ if b not in bands]
    if missing_bands:
        # Not fatal -- a band with no surviving bins is legitimately absent -- but
        # it changes the feature count, so it must be visible, not inferred later
        # from a column count that does not match the other units.
        logger.warning('sub-%s ses-%s: bands absent from the view: %s',
                       subject, session, missing_bands)

    # `region` holds the CHANNEL name when the view was built --region none, and
    # `freq_bin_index` holds the BAND NAME (not an index) for any banded view.
    # The column name is a pre-existing misnomer; do not coerce it to int.
    wide = long.pivot_table(index='epoch_id', columns=['region', 'freq_bin_index'],
                            values='value', aggfunc='first', dropna=False)
    channels = sorted({c for c, _ in wide.columns})
    wide = wide.reindex(columns=pd.MultiIndex.from_product([channels, bands]))

    defs = cache_reader.load_defs(subject, session, epoch_minutes)
    defs = defs.sort_values('epoch_id').reset_index(drop=True)
    # Restrict to the epochs the view actually emitted, in the same order.
    defs = defs[defs['epoch_id'].isin(wide.index)].reset_index(drop=True)
    wide = wide.reindex(index=defs['epoch_id'].to_numpy())

    values = wide.to_numpy(dtype=config.CACHE_ACCUMULATE_DTYPE)
    grid = values.reshape(len(defs), len(channels), len(bands))

    # A channel-epoch is BAD when it was recorded but has no finite value; the
    # view drops the whole channel-epoch at once, so any-NaN and all-NaN coincide
    # in practice. `any` is the conservative reading.
    recorded = _recorded_matrix(subject, session, defs, channels, epoch_minutes,
                                source=view_params.get('source', 'psd_view'))
    bad = recorded & ~np.isfinite(grid).all(axis=2)

    result = cascade.apply_cascade(
        bad, recorded=recorded, channel_max_bad=channel_max_bad,
        epoch_max_bad=epoch_max_bad, min_epochs=min_epochs)
    keep_e, keep_c = result['keep_epochs'], result['keep_channels']

    kept_channels = [c for c, k in zip(channels, keep_c) if k]
    X = grid[np.ix_(keep_e, keep_c)].reshape(int(keep_e.sum()), -1)
    y = defs.loc[keep_e, 'pain_score'].to_numpy(dtype=float)
    feature_names = [f'{c}|{b}' for c in kept_channels for b in bands]

    report = dict(result['report'])
    report.update({
        'subject_id': f'sub-{subject}', 'session_id': f'ses-{session}',
        'n_features': X.shape[1], 'n_bands': len(bands),
        'bands': bands, 'bands_absent': missing_bands,
        'p_over_n': X.shape[1] / max(X.shape[0], 1),
        'view_dir': str(view_dir), 'view_config_hash': view_params.get('config_hash'),
        'source': view_params.get('source', 'psd_view'),
        'mask_label': view_params.get('mask_label'),
        'pain_score_min': float(y.min()), 'pain_score_max': float(y.max()),
        'pain_score_median': float(np.median(y)),
        'n_zero_pain_epochs': int((y == 0).sum()),
    })

    out = FeatureMatrix(subject, session, X, y, feature_names, kept_channels,
                        bands, defs.loc[keep_e, 'epoch_id'].to_numpy(), y, report)
    logger.info('%r', out)
    return out
