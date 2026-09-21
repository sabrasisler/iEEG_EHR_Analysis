"""Anterior vs posterior insula, from a MEDIAN SPLIT on the MNI y coordinate.

WHY THIS MODULE EXISTS AT ALL
-----------------------------
`roi_schemes` maps an atlas LABEL to a region by substring match, and that is the
whole of its contract. Desikan-Killiany has exactly one `insula` parcel, so an
anterior/posterior distinction is not derivable from the label -- which is the
documented reason insula has been held out of every `pain_domains*` scheme
(roi_schemes.py module docstring; run_domain_model.DOMAIN_CAVEAT). The split has
to come from somewhere the label cannot reach: the contact's COORDINATE. That is
a different kind of fact, so it lives in a different module rather than being
smuggled into a pattern dict.

WHAT THE SPLIT IS, EXACTLY
--------------------------
Take every bipolar pair whose anode DK label contains 'insula', pool them across
subjects and hemispheres, take the MEDIAN of their MNI y, and call everything
above it anterior and everything at-or-below it posterior. One number, one axis,
no atlas.

IT IS DELIBERATELY COARSE AND YOU SHOULD READ IT AS COARSE. Three things it is
not:

  - It is not the Destrieux/a2009s assignment (G_insular_short vs
    S_circular_insula_ant/sup/inf), which is the anatomically correct answer and
    is still the right eventual fix. This is a stopgap that lets insula enter the
    model at all.
  - The threshold is DATA-DEPENDENT, not anatomical: it is this cohort's median,
    so it is a 50/50 split of THESE electrodes and not a landmark. A cohort with
    different coverage would put the line somewhere else, which is why the
    threshold is written into provenance as a number and not just as "median".
  - The insula is a slanted, folded structure; a plane normal to y is not its
    anterior-posterior axis. Contacts near the line are near-arbitrary. The
    claim this split can support is "on average, more anterior" -- not "this
    contact is in the anterior insula".

The midpoint of the bipolar pair is the coordinate used, matching
plot_electrode_locations: it is what the NWB's MNI_coord_1/2/3 stores and it is
the position a difference-of-two-contacts signal belongs at. The DK label comes
from the ANODE, matching every other region assignment in the project, so the
same midpoint/anode asymmetry noted there applies here.
"""

import logging

import numpy as np
import pandas as pd

from ieeg_ehr import config, io

logger = logging.getLogger(__name__)

#: Substring that identifies an insular DK label, case-insensitive. Same test
#: `roi_schemes` uses for its `Insula` category, kept identical on purpose: a
#: contact this module splits must be exactly a contact that scheme would have
#: called Insula, or the two disagree about the denominator.
INSULA_PATTERN = 'insula'

#: The axis the median is taken on. y is anterior-posterior in MNI152.
SPLIT_AXIS = 'mni_y'

#: Parcel-level names. These are the values that land in the model's `parcel`
#: column and in a figure legend, so they are stable and ASCII.
ANTERIOR = 'aIns'
POSTERIOR = 'pIns'

#: What a figure calls them.
DISPLAY = {ANTERIOR: 'Anterior insula (aIns)', POSTERIOR: 'Posterior insula (pIns)'}

#: Colours, taken from `run_domain_model.DOMAIN_COLOURS` for the domain each part
#: is destined to join -- aIns to Affective, pIns to Sensory. Matching them means
#: the split figure and the domain figures cannot say different things with the
#: same hue.
COLOURS = {ANTERIOR: '#8e44ad', POSTERIOR: '#b03a2e'}


def load_insula_contacts(subjects, session=None, epoch_minutes=None):
    """One row per (subject, insular bipolar pair) with its MNI midpoint.

    EVERY SESSION BY DEFAULT, not just ses-01. Three subjects have a second
    session, and the split is keyed by (subject, channel) with no session in it
    -- so reading only ses-01 would drop a two-session subject's ses-02-only
    contacts, and they would leave the model as "insula with no coordinate"
    rather than as anything anyone decided. Pass `session` to pin one.

    DE-DUPLICATED PER SUBJECT on `channel`: channel_meta is keyed
    (run_id, pair_index) because pair order is a per-run property, so the same
    physical electrode appears once per run and subjects here have up to 100+
    runs. Counting without de-duplicating would weight a subject by how many
    recordings they happen to have. The same de-duplication absorbs the overlap
    between two sessions, which is nearly total -- it is the same implant.

    Contacts with no finite MNI coordinate, or one outside MNI152, are dropped
    and COUNTED -- a median taken over a localisation failure is not a median of
    anything.
    """
    from ieeg_ehr.analysis.plot_electrode_locations import MNI_BOUNDS

    meta_dir = config.pain_epoch_channel_meta_path(
        '000', '01', epoch_minutes).parent

    frames = []
    for subject in subjects:
        bare = str(subject).replace('sub-', '')
        if session is not None:
            paths = [config.pain_epoch_channel_meta_path(bare, session,
                                                         epoch_minutes)]
            paths = [p for p in paths if p.exists()]
        else:
            paths = sorted(meta_dir.glob(f'sub-{bare}_ses-*_channels.parquet'))
        if not paths:
            logger.warning('sub-%s: no channel_meta under %s, skipped', bare,
                           meta_dir)
            continue

        per_subject = []
        for path in paths:
            meta = io.read_table(path, on_stale='ignore')
            missing = [c for c in ('mni_x', 'mni_y', 'mni_z', 'dk_anode')
                       if c not in meta.columns]
            if missing:
                logger.warning('%s: channel_meta lacks %s, skipped', path.name,
                               missing)
                continue
            per_subject.append(meta)
        if not per_subject:
            continue

        meta = pd.concat(per_subject, ignore_index=True)
        hit = meta['dk_anode'].astype(str).str.lower().str.contains(INSULA_PATTERN)
        meta = meta[hit].drop_duplicates('channel').copy()
        if meta.empty:
            continue
        meta['subject_id'] = f'sub-{bare}'
        frames.append(meta[['subject_id', 'channel', 'dk_anode',
                            'mni_x', 'mni_y', 'mni_z']])

    if not frames:
        raise SystemExit('no insular contact found in any subject')

    contacts = pd.concat(frames, ignore_index=True)
    n0 = len(contacts)

    coords = contacts[['mni_x', 'mni_y', 'mni_z']].to_numpy(dtype=float)
    ok = np.isfinite(coords).all(axis=1)
    for ax, (lo, hi) in zip(('x', 'y', 'z'), (MNI_BOUNDS['x'], MNI_BOUNDS['y'],
                                              MNI_BOUNDS['z'])):
        col = coords[:, 'xyz'.index(ax)]
        ok &= (col >= lo) & (col <= hi)
    n_drop = int((~ok).sum())
    if n_drop:
        logger.warning('dropped %d/%d insular contact(s) with no usable MNI '
                       'coordinate -- they cannot be split and are NOT assigned '
                       'to either part', n_drop, n0)
    contacts = contacts[ok].reset_index(drop=True)
    contacts['hemisphere'] = np.where(contacts['mni_x'].to_numpy(dtype=float) < 0,
                                      'L', 'R')
    logger.info('%d insular contact(s) in %d subject(s)', len(contacts),
                contacts['subject_id'].nunique())
    return contacts, {'n_insular_contacts_found': n0,
                      'n_dropped_no_usable_mni': n_drop}


def median_split(contacts, axis=SPLIT_AXIS, per_hemisphere=False, threshold=None):
    """Add an `ins_part` column; return (contacts, split description).

    `threshold` pins the cut instead of computing it, which is how a later run
    reproduces an earlier one's split exactly rather than re-deriving a median
    from whatever cohort it happens to have. Pass the number from the earlier
    run's provenance.

    ABOVE the median is anterior, AT-OR-BELOW is posterior. The tie rule matters
    at odd n, where the median IS one of the contacts: that contact goes
    posterior. It is one contact and either choice is arbitrary, but it has to be
    written down or two implementations will differ by it.
    """
    out = contacts.copy()

    if threshold is not None:
        thresholds = (dict(threshold) if isinstance(threshold, dict)
                      else {'all': float(threshold)})
        source = 'pinned'
    elif per_hemisphere:
        thresholds = {h: float(g[axis].median())
                      for h, g in out.groupby('hemisphere')}
        source = 'median of this cohort, within hemisphere'
    else:
        thresholds = {'all': float(out[axis].median())}
        source = 'median of this cohort, hemispheres pooled'

    # One cut value per ROW, so the pooled and per-hemisphere cases take the
    # identical comparison. A hemisphere with no pinned threshold is a hard
    # error, not a NaN comparison that silently calls every contact posterior.
    if 'all' in thresholds:
        cut = np.full(len(out), thresholds['all'], dtype=float)
    else:
        cut = out['hemisphere'].map(thresholds).to_numpy(dtype=float)
        if not np.isfinite(cut).all():
            missing = sorted(set(out.loc[~np.isfinite(cut), 'hemisphere']))
            raise ValueError(f'no insula split threshold for hemisphere {missing}')
    out['ins_part'] = np.where(out[axis].to_numpy(dtype=float) > cut,
                               ANTERIOR, POSTERIOR)

    desc = {'axis': axis, 'rule': 'value > threshold -> anterior; <= -> posterior',
            'per_hemisphere': bool(per_hemisphere), 'threshold_source': source,
            'thresholds': thresholds,
            'n_anterior': int((out['ins_part'] == ANTERIOR).sum()),
            'n_posterior': int((out['ins_part'] == POSTERIOR).sum()),
            'n_subjects_anterior':
                int(out.loc[out['ins_part'] == ANTERIOR, 'subject_id'].nunique()),
            'n_subjects_posterior':
                int(out.loc[out['ins_part'] == POSTERIOR, 'subject_id'].nunique())}
    logger.info('insula split on %s (%s): %s', axis, source, thresholds)
    logger.info('  anterior  %4d contacts, %2d subjects',
                desc['n_anterior'], desc['n_subjects_anterior'])
    logger.info('  posterior %4d contacts, %2d subjects',
                desc['n_posterior'], desc['n_subjects_posterior'])
    return out, desc


def channel_lookup(contacts):
    """{subject_id: {channel: ins_part}}, the shape the model's parcel map wants."""
    out = {}
    for sid, g in contacts.groupby('subject_id'):
        out[sid] = dict(zip(g['channel'], g['ins_part']))
    return out


def resolve_split(subjects, scheme, session=None, epoch_minutes=None,
                  axis=SPLIT_AXIS, per_hemisphere=False, threshold=None):
    """(contacts, split description) if `scheme` asks for a coordinate insula split.

    Returns (None, None) for a scheme that does not declare one, so a caller can
    do this unconditionally and let the SCHEME decide -- which is the point.
    Anything else and every analysis has to remember which schemes are split.
    """
    from ieeg_ehr.config import roi_schemes

    coord = roi_schemes.coordinate_regions(scheme)
    if not coord:
        return None, None
    parts = {p for v in coord.values() for p in v}
    if parts != {ANTERIOR, POSTERIOR}:
        raise ValueError(
            f'scheme declares coordinate regions {sorted(parts)}, but this module '
            f'only knows how to fill {sorted((ANTERIOR, POSTERIOR))}')
    contacts, counts = load_insula_contacts(subjects, session=session,
                                            epoch_minutes=epoch_minutes)
    contacts, desc = median_split(contacts, axis=axis,
                                  per_hemisphere=per_hemisphere,
                                  threshold=threshold)
    desc = {**desc, **counts}
    return contacts, desc


def apply_split(region_by_subject, contacts, scheme):
    """Replace the parent region with its part, IN PLACE, and report what moved.

    `region_by_subject` is `{subject_id: {channel: region}}` -- the shape both
    `run_fullres_grid.roi_maps` and `run_domain_model.parcel_domain_maps`
    already produce, which is why this is a post-step rather than a rewrite of
    either.

    AN INSULAR CHANNEL WITH NO USABLE COORDINATE IS DROPPED, not defaulted. It
    cannot be placed on either side of the cut, and guessing would put a contact
    in a domain on the strength of nothing. Dropped channels are counted and the
    count is returned, because coverage is a confound here and a shrinking
    denominator has to be visible.
    """
    from ieeg_ehr.config import roi_schemes

    coord = roi_schemes.coordinate_regions(scheme)
    if not coord:
        return {'applied': False}
    lookup = channel_lookup(contacts) if contacts is not None else {}

    moved = {p: 0 for parts in coord.values() for p in parts}
    dropped, subjects_touched = 0, set()
    for sid, mapping in region_by_subject.items():
        ins = lookup.get(sid, {})
        for ch, region in list(mapping.items()):
            if region not in coord:
                continue
            subjects_touched.add(sid)
            part = ins.get(ch)
            if part is None:
                del mapping[ch]
                dropped += 1
            else:
                mapping[ch] = part
                moved[part] += 1

    logger.info('insula split applied over %d subject(s): %s, %d channel(s) '
                'dropped for want of a coordinate', len(subjects_touched),
                ', '.join(f'{k} {v}' for k, v in sorted(moved.items())), dropped)
    return {'applied': True, 'parents': sorted(coord),
            'n_by_part': moved, 'n_dropped_no_coordinate': dropped,
            'n_subjects_touched': len(subjects_touched)}
