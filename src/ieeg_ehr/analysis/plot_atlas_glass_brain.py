#!/usr/bin/env python3
"""
Glass brain of the electrodes, one colour per Yeo 2011 network.

    analysis/electrode_localization/<yeo7|yeo17>_coverage/glass_brain/<run>_<timestamp>/

Reads a `plot_atlas_coverage --atlas yeo7` (or `yeo17`) run's `pair_labels.csv`,
so every dot is the pair's VIRTUAL ELECTRODE (anode-cathode MNI midpoint) and
carries exactly the network the coverage counts used. The cohort is that run's
`subjects[]`.

Only pairs IN a network are drawn. Pairs with no network (white matter,
subcortex, outside the mask) are counted in the log and the provenance, not
plotted.

Colours are Yeo's own (`Yeo2011_<n>Networks_ColorLUT.txt`), the convention a
reader of network figures recognises. Limbic's cream is too pale for the white
background on its own, so every marker gets a thin dark rim.

Run on Slurm, never the login node:
    sbatch sbatch/atlas_glass_brain.sbatch
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis.plot_atlas_coverage import EVENT, NO_LABEL, YEO_DIR
from ieeg_ehr.analysis.plot_electrode_locations import plot_glass_brain

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_atlas_glass_brain.py'
OUTPUT_TYPE = 'glass_brain'
EDGE_COLOR = '0.25'


def yeo_networks(n_networks):
    """[(network name, hex colour)] in Yeo's network order."""
    names = pd.read_csv(YEO_DIR / f'{n_networks}NetworksOrderedNames.csv',
                        skipinitialspace=True)
    colour = {}
    for line in (YEO_DIR / f'Yeo2011_{n_networks}Networks_ColorLUT.txt').read_text().splitlines():
        value, _, r, g, b, _ = line.split()
        colour[int(value)] = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
    return [(name, colour[v]) for v, name in
            zip(names['Network Order'], names['Network Name'])]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--coverage-run', required=True,
                    help='A plot_atlas_coverage yeo7 or yeo17 run directory.')
    ap.add_argument('--node-size', type=float, default=14)
    ap.add_argument('--run-name', default=None,
                    help='Default: the coverage run\'s own folder name, minus its stamp.')
    args = ap.parse_args()

    io.warn_if_dirty()

    coverage_run = Path(args.coverage_run)
    prov = json.loads((coverage_run / 'provenance.json').read_text())
    atlas = prov['params']['atlas']
    if atlas not in ('yeo7', 'yeo17'):
        raise SystemExit(f'{coverage_run} is a {atlas!r} run; this figure needs yeo7 or yeo17')
    n_networks = int(atlas.removeprefix('yeo'))
    subjects = prov['subjects']

    pairs_path = coverage_run / 'pair_labels.csv'
    pairs = io.read_table(pairs_path, on_stale='warn')
    in_network = ~pairs['label'].isin(NO_LABEL)
    logger.info('%d of %d pair(s) in a network; not drawn: %s', in_network.sum(),
                len(pairs), pairs.loc[~in_network, 'label'].value_counts().to_dict())
    contacts = pairs[in_network].rename(columns={'label': 'region'})

    networks = yeo_networks(n_networks)
    regions = [name for name, _ in networks]
    colours = dict(networks)
    unknown = sorted(set(contacts['region']) - set(regions))
    if unknown:
        raise SystemExit(f'labels in {pairs_path} that are not Yeo networks: {unknown}')

    run_name = args.run_name or coverage_run.name.rsplit('_', 1)[0]
    run_dir = config.analysis_run_dir(question=f'{atlas}_coverage', output_type=OUTPUT_TYPE,
                                      run_name=run_name, event=EVENT)
    parents = [io.parent_ref(coverage_run / 'provenance.json', digest=False),
               io.parent_ref(pairs_path, digest=False)]
    io.write_run_provenance(
        run_dir, script=SCRIPT, params={**vars(args), 'atlas': atlas},
        parents=parents, subjects=subjects,
        extra={'n_pairs_drawn': int(in_network.sum()),
               'n_pairs_not_drawn': int((~in_network).sum()),
               'not_drawn_by_label': pairs.loc[~in_network, 'label']
               .value_counts().to_dict(),
               'colours': colours})

    plot_glass_brain(
        contacts, regions, colours,
        f'Yeo 2011 {n_networks} networks at the pair midpoint: '
        f'{len(contacts)} of {len(pairs)} bipolar pairs, '
        f'{contacts["subject_id"].nunique()} subjects',
        run_dir / 'glass_brain.png', node_size=args.node_size,
        legend_title=f'Yeo {n_networks} network (bipolar pairs)', edge_color=EDGE_COLOR)

    io.log_analysis(f'{atlas} network glass brain, {len(contacts)} of {len(pairs)} pairs, '
                    f'n={len(subjects)}', run_dir)
    logger.info('figure + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()
