"""Which channels and which epochs are allowed into a decoder's feature matrix.

Pure functions on boolean matrices -- no IO, so every rule here is testable
without touching Oak.

THE THREE KINDS OF MISSING, WHICH MUST NOT BE CONFLATED
-------------------------------------------------------
1. NOT RECORDED -- the channel is absent from that epoch's run montage. This is
   COVERAGE, not badness. Only 2 of 45 discovery units have run-varying montages
   (189: 120 of 230 channels; 167: 26 of 169), and on the first pass at this I
   counted these as artifact and inflated 189's channel loss by 60 channels. A
   channel that was never recorded must NEVER be imputed -- that would fabricate
   half a session.
2. ARTIFACT -- the channel was recorded but the QC mask took out more than
   `max_excluded_frac` of the epoch's windows, so the view dropped the
   channel-epoch. This is what Y and Z act on.
3. RESIDUAL -- a channel that survives Y, in an epoch that survives Z, but that
   one cell is still bad. Measured at 0.82% of cells yet SCATTERED ACROSS 23.6%
   OF EPOCHS, so dropping rows to purge them would cost a quarter of the training
   data to remove under 1% of cells. These are imputed, in-fold, by the model
   pipeline -- not here.

WHY CHANNELS ARE JUDGED BEFORE EPOCHS
-------------------------------------
The two rules are confounded: drop epochs first and a chronically bad channel
looks clean (its worst epochs are gone); drop channels first and a bad epoch
looks clean. The tie is broken on physical grounds -- a bad channel is a
PERSISTENT property of the electrode, tissue contact or amplifier, while a bad
epoch is a TRANSIENT event. Remove the persistent faults before judging the
transients, or a handful of permanently dead channels makes every epoch look 40%
bad and good data starts getting deleted.

WHY Y IS STRICT AND Z IS PERMISSIVE (the decoder-specific asymmetry)
--------------------------------------------------------------------
This is the opposite of what a group-level heatmap wants, and it is deliberate.
A dropped channel costs 6 features of ~918 -- the elastic-net penalty is zeroing
most of them anyway. A dropped epoch costs ~2% of a ~48-observation training set,
and the observation count is what decides whether the model is fittable at all.
So buy cleanliness with channels, and protect epochs.

Thresholds are set on STRUCTURAL grounds from the measured missingness
distribution, before looking at any pain relationship (CLAUDE.md; the rule
architecture.md PART 7 left as TODO). Y=0.2, Z=0.5, and the reasoning -- including
why Z is insensitive to its exact value -- is in DECISIONS 2026-09-08.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Drop a channel when more than this fraction of its epochs are artifact-bad.
#: 0.2 removes roughly the worst 1.5% of channels; the distribution's p95 is
#: 0.086 and p99 is 0.283, so this cuts into the tail without touching the bulk.
CHANNEL_MAX_BAD_EPOCH_FRAC = 0.2

#: Drop an epoch when more than this fraction of the SURVIVING channels are bad.
#: 0.5 sits in the middle of an empty gap: per-epoch bad fractions have p95 =
#: 0.067 and p99 = 0.689, with essentially nothing in between, so any value in
#: [0.2, 0.7] drops the same ~1% of epochs. The threshold does no work, which is
#: the best possible situation for one that had to be chosen.
EPOCH_MAX_BAD_CHANNEL_FRAC = 0.5


class NoUsableDataError(RuntimeError):
    """The cascade left too little to fit anything.

    Its own type so a Slurm array task can report "this unit is not decodable"
    and exit cleanly, rather than failing somewhere inside sklearn with a shape
    error that reads like a bug.
    """


def stable_coverage_channels(recorded):
    """Channels recorded in EVERY epoch under consideration.

    `recorded` is (n_epochs, n_channels) bool: was this channel present in this
    epoch's run montage at all. Returns a bool mask over channels.

    Restricting to the intersection rather than imputing is the only honest
    option for a channel that was never recorded. It is computed over the epochs
    that actually CONTRIBUTE, not over every run in the session -- a session can
    have 68 runs while only a handful carry pain epochs, so intersecting over all
    of them would discard coverage for no reason.
    """
    recorded = np.asarray(recorded, dtype=bool)
    if recorded.ndim != 2:
        raise ValueError(f'recorded must be 2-D (epochs x channels), got {recorded.shape}')
    return recorded.all(axis=0)


def apply_cascade(bad, recorded=None, channel_max_bad=None, epoch_max_bad=None,
                  min_epochs=30, min_channels=1):
    """Run the eligibility cascade over one unit's (epoch x channel) grid.

    Args:
        bad: (n_epochs, n_channels) bool -- the cell is ARTIFACT-bad. Cells that
            are merely not-recorded must be False here; `recorded` carries those.
        recorded: (n_epochs, n_channels) bool, or None to mean "all recorded".
        min_epochs / min_channels: refuse rather than return a matrix too small
            to cross-validate.

    Returns a dict with `keep_epochs`, `keep_channels` (bool masks over the
    ORIGINAL axes), `residual` (the bad cells surviving both rules, on the kept
    submatrix), and counts for the run's report.

    Order is coverage -> channels -> epochs, and each step is judged against what
    the previous one left.
    """
    bad = np.asarray(bad, dtype=bool)
    if bad.ndim != 2:
        raise ValueError(f'bad must be 2-D (epochs x channels), got {bad.shape}')
    n_epochs, n_channels = bad.shape

    y = CHANNEL_MAX_BAD_EPOCH_FRAC if channel_max_bad is None else channel_max_bad
    z = EPOCH_MAX_BAD_CHANNEL_FRAC if epoch_max_bad is None else epoch_max_bad

    if recorded is None:
        recorded = np.ones_like(bad, dtype=bool)
    recorded = np.asarray(recorded, dtype=bool)
    if recorded.shape != bad.shape:
        raise ValueError(f'recorded {recorded.shape} != bad {bad.shape}')

    # ---- step 0: coverage. Never-recorded channels leave before anything is
    # measured, so they cannot inflate a channel's or an epoch's bad fraction.
    keep_channels = stable_coverage_channels(recorded)
    n_coverage_dropped = int((~keep_channels).sum())
    if not keep_channels.any():
        raise NoUsableDataError(
            'no channel is recorded in every contributing epoch; the run montages '
            'share nothing, so there is no common feature set to fit on'
        )

    # ---- step 1: channels, over the coverage-stable set.
    chan_bad_frac = bad[:, keep_channels].mean(axis=0)
    surviving = chan_bad_frac <= y
    # Map back onto the original channel axis.
    keep_channels = keep_channels.copy()
    keep_channels[np.flatnonzero(keep_channels)[~surviving]] = False
    n_artifact_dropped = int((~surviving).sum())
    if not keep_channels.any():
        raise NoUsableDataError(
            f'every channel exceeds Y={y} bad-epoch fraction; nothing to fit on'
        )

    # ---- step 2: epochs, against the channels that SURVIVED step 1.
    epoch_bad_frac = bad[:, keep_channels].mean(axis=1)
    keep_epochs = epoch_bad_frac <= z

    n_kept_epochs, n_kept_channels = int(keep_epochs.sum()), int(keep_channels.sum())
    if n_kept_epochs < min_epochs:
        raise NoUsableDataError(
            f'{n_kept_epochs} epochs survive the cascade, below the {min_epochs} '
            'needed for cross-validation with >=5 observations per fold'
        )
    if n_kept_channels < min_channels:
        raise NoUsableDataError(
            f'{n_kept_channels} channels survive, below the required {min_channels}'
        )

    residual = bad[np.ix_(keep_epochs, keep_channels)]
    report = {
        'y_channel_max_bad': float(y),
        'z_epoch_max_bad': float(z),
        'n_epochs_in': n_epochs,
        'n_channels_in': n_channels,
        'n_epochs_kept': n_kept_epochs,
        'n_channels_kept': n_kept_channels,
        'n_channels_dropped_coverage': n_coverage_dropped,
        'n_channels_dropped_artifact': n_artifact_dropped,
        'n_epochs_dropped': int((~keep_epochs).sum()),
        'residual_cell_frac': float(residual.mean()) if residual.size else 0.0,
        'residual_cells': int(residual.sum()),
        'epochs_with_residual_frac': (float(residual.any(axis=1).mean())
                                      if residual.size else 0.0),
    }
    logger.info(
        'cascade: %d/%d epochs, %d/%d channels kept (coverage -%d, artifact -%d); '
        'residual %d cells (%.4f) across %.1f%% of kept epochs',
        n_kept_epochs, n_epochs, n_kept_channels, n_channels,
        n_coverage_dropped, n_artifact_dropped, report['residual_cells'],
        report['residual_cell_frac'], 100 * report['epochs_with_residual_frac'])

    return {'keep_epochs': keep_epochs, 'keep_channels': keep_channels,
            'residual': residual, 'report': report}
