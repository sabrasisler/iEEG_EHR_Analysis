"""
HCPex v1.1 parcel -> paper ROI, as data.

Every one of HCPex's 213 parcel names (hemispheres merged: `L V1` and `R V1`
are both `V1`) appears under exactly one key; `plot_atlas_coverage` checks that
against the atlas's own label file and refuses to run otherwise. Key order is
the paper's ROI order.

THE CORTICAL BACKBONE is Glasser et al. 2016's 22 cortical sections (Nature,
Supplementary Neuroanatomical Results), split or merged into the paper's ROIs.
The calls that are judgment, not lookup:

  - MCC takes the "prime" cingulate areas (a24pr, p24pr, 33pr, a32pr, p32pr),
    which are Vogt's 24'/32' midcingulate areas, plus 24dd/24dv.
  - Premotor takes SMA/pre-SMA (6mp, SCEF, 6ma) from Glasser's paracentral
    section; area 5 (5L, 5m, 5mv) goes to SPL.
  - dlPFC is Glasser's whole dorsolateral section, including the dorsal SFG
    areas 8BL, 9a, 9p and SFL; dmPFC is the medial 8BM and 9m only.
  - aIns = AAIC, AVI, MI; pIns = Ig, PoI1, PoI2, PI. MI (mid-insula) and PI
    (para-insular, at the temporal edge) are the borderline ones.
  - Frontal operculum takes area 43 (subcentral) with FOP1-5; SII is the
    parietal operculum OP1, OP2-3, OP4 plus PFcm.
  - The STS splits by bank: STSd* -> STG, STSv* -> MTG. TPOJ1-3 -> IPL;
    STV, PSL and area 52 -> STG.
  - Occipital takes Glasser's visual sections whole, including the dorsal
    stream (V3A ... IPS1) and the MT+ complex (MT, MST, FST, PH, LO1-3,
    V3CD, V4t); Fusiform keeps FFC, VVC and PIT.
  - Thalamus follows nuclear groups: anterior = AV, LD; medial = MD,
    intralaminar (CM, Pf, CL, CeM) and midline (Re); lateral = the ventral
    nuclei, LP, the geniculates and L-Sg; pulvinar = PuA, PuI, PuL, PuM.

HIPPOCAMPUS is one HCPex parcel (Glasser's H, named `Hipp`), plus the
presubiculum. It is split into anterior and posterior by the virtual
electrode's MNI y at HIPPOCAMPUS_SPLIT_Y, the uncal apex (Poppenk et al. 2013,
doi 10.1016/j.tics.2013.03.005): y > -21 is anterior.

STN has no HCPex parcel, so it is listed with none and always counts zero.
Parcels with no paper ROI (piriform, VTA, mammillary bodies, septum, nucleus
basalis) sit under NO_ROI.

ITERATING WITHOUT A COMMIT: `load(path)` reads the same {ROI: [parcels]} shape
from a JSON file on Oak, so a regrouping is a file edit, and a run records the
mapping's full contents in its provenance. The key named HIPPOCAMPUS is the one
split by y; name a key differently to keep the hippocampus whole. NO_ROI must
keep its exact name.
"""

import json

#: The key whose pairs are split into ANTERIOR_/POSTERIOR_HIPPOCAMPUS by y.
HIPPOCAMPUS = 'Hippocampus'
ANTERIOR_HIPPOCAMPUS = 'Anterior hippocampus'
POSTERIOR_HIPPOCAMPUS = 'Posterior hippocampus'
HIPPOCAMPUS_SPLIT_Y = -21.0
NO_ROI = 'No ROI (—)'

PARCELS = {
    'dlPFC': ['46', '9-46d', 'a9-46v', 'p9-46v', '8C', '8Av', '8Ad', '8BL', '9a',
              '9p', 'i6-8', 's6-8', 'SFL'],
    'vlPFC': ['44', '45', '47l', 'p47r', 'IFJa', 'IFJp', 'IFSa', 'IFSp'],
    'Frontal pole': ['a10p', 'p10p', '10pp', '10d'],
    'OFC': ['11l', '13l', 'OFC', 'pOFC', '47m', '47s', 'a47r'],
    'vmPFC': ['10r', '10v', '25', 's32'],
    'dmPFC': ['8BM', '9m'],
    'Premotor': ['6a', '6d', '6v', '6r', '55b', 'FEF', 'PEF', '6ma', '6mp', 'SCEF'],
    'M1': ['4'],
    'Frontal operculum': ['FOP1', 'FOP2', 'FOP3', 'FOP4', 'FOP5', '43'],
    'aIns': ['AAIC', 'AVI', 'MI'],
    'pIns': ['Ig', 'PoI1', 'PoI2', 'PI'],
    'ACC': ['a24', 'p24', 'p32', 'd32'],
    'MCC': ['a24pr', 'p24pr', '33pr', 'a32pr', 'p32pr', '24dd', '24dv'],
    'PCC': ['23c', '23d', 'd23ab', 'v23ab', '31a', '31pd', '31pv', 'RSC'],
    'S1': ['3a', '3b', '1', '2'],
    'SII': ['OP1', 'OP2-3', 'OP4', 'PFcm'],
    'IPL': ['PF', 'PFm', 'PFop', 'PFt', 'PGi', 'PGp', 'PGs', 'IP0', 'IP1', 'IP2',
            'TPOJ1', 'TPOJ2', 'TPOJ3'],
    'SPL': ['7AL', '7PC', '7PL', 'AIP', 'LIPd', 'LIPv', 'MIP', 'VIP', '5L', '5m',
            '5mv'],
    'Precuneus': ['7m', '7Am', '7Pm', 'PCV', 'POS1', 'POS2'],
    'Temporal pole': ['TGd', 'TGv'],
    'STG': ['A1', 'LBelt', 'MBelt', 'PBelt', 'RI', '52', 'A4', 'A5', 'STGa', 'TA2',
            'STSda', 'STSdp', 'STV', 'PSL'],
    'MTG': ['TE1a', 'TE1m', 'TE1p', 'STSva', 'STSvp'],
    'ITG': ['TE2a', 'TE2p', 'PHT'],
    'Fusiform': ['FFC', 'VVC', 'PIT'],
    HIPPOCAMPUS: ['Hipp', 'PreS'],
    'Amygdala': ['Amyg'],
    'Parahippocampal': ['PHA1', 'PHA2', 'PHA3', 'TF'],
    'Entorhinal/perirhinal': ['EC', 'PeEc'],
    'Lateral thalamus': ['VA', 'VLa', 'VLp', 'VPL', 'LP', 'LGN', 'MGN', 'L-Sg'],
    'Medial thalamus': ['MDm', 'MDl', 'CM', 'Pf', 'CeM', 'CL', 'MV(Re)'],
    'Anterior thalamus': ['AV', 'LD'],
    'Pulvinar': ['PuA', 'PuI', 'PuL', 'PuM'],
    'Dorsal striatum': ['Putam', 'Caud'],
    'Ventral striatum': ['NAc'],
    'Globus pallidus': ['Gpe', 'Gpi'],
    'Substantia nigra': ['SNpc', 'SNpr'],
    'STN': [],
    'Occipital': ['V1', 'V2', 'V3', 'V4', 'V3A', 'V3B', 'V6', 'V6A', 'V7', 'IPS1',
                  'V8', 'VMV1', 'VMV2', 'VMV3', 'LO1', 'LO2', 'LO3', 'V3CD', 'V4t',
                  'MT', 'MST', 'FST', 'PH', 'DVT', 'ProS'],
    NO_ROI: ['Pir', 'VTA', 'MB', 'Septum', 'Nb'],
}


def load(path=None):
    """{ROI: [parcels]}: the built-in PARCELS, or a JSON file of the same shape."""
    if path is None:
        return PARCELS
    return json.loads(open(path).read())


def rois_of(key):
    """The display ROI(s) one mapping key becomes."""
    return (ANTERIOR_HIPPOCAMPUS, POSTERIOR_HIPPOCAMPUS) if key == HIPPOCAMPUS else (key,)


def roi_order(mapping):
    """Display order: the mapping's key order, with the hippocampus split in place."""
    return [r for key in mapping if key != NO_ROI for r in rois_of(key)]


def roi_by_parcel(mapping):
    """{HCPex short name: mapping key}, refusing a parcel listed twice."""
    out = {}
    for roi, parcels in mapping.items():
        for parcel in parcels:
            if parcel in out:
                raise ValueError(f'{parcel} is under both {out[parcel]} and {roi}')
            out[parcel] = roi
    return out
