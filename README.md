# iEEG_EHR_Analysis

Sensory-discriminative vs affective-evaluative pain encoding in intracranial EEG
from ~250 epilepsy-monitoring-unit patients (Sisler / Keller Lab, Stanford).
Runs on the Sherlock HPC cluster.

## The one rule: this repo holds CODE ONLY

All data, derivatives, caches, features, epoch definitions, QC outputs, cohort
files, analysis outputs, plots, models and results live on Oak:

```
/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/
```

Nothing writes into the repo. Every output path resolves through
`ieeg_ehr.config.paths`, never a repo-relative path. `.gitignore` is a backstop,
not the mechanism. See `CLAUDE.md` for the full rule set.

## Layout

```
CLAUDE.md            operating rules — read first
docs/                architecture, task plan, view registry, background context
src/ieeg_ehr/
  config/            single source of paths, thresholds, band defs, pinned mask
  io/                provenance + table/NWB helpers
  preprocessing/     bipolar re-reference + Welch PSD (the feature family)
  qc/                raw-voltage + bipolar QC: detectors, exclusions, masks
  features/          epoch-power cache builder
  views/             the seven view axes (P1.3)
  analysis/          plotting, sweeps, models
sbatch/              all Slurm jobs
notebooks/           scratch only; never imported by pipeline code
tests/
outdated/            superseded code kept for reference
logs/                gitignored
```

## Setup

The environment is a shared venv on `$GROUP_HOME`, not in this repo:

```bash
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
```

The package is installed editable, so `import ieeg_ehr` works from any
directory. To (re)install after moving the repo — on a compute node, never the
login node:

```bash
srun -p dev --pty bash -c '
  module load python/3.12
  source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
  pip install -e /home/groups/ckeller1/sisler/iEEG_EHR_Analysis --no-deps'
```

`--no-deps` is deliberate: it avoids upgrading the working numpy/pandas/pynwb.
Install new dependencies explicitly and one at a time.

## Running things

Never run Python on the login node. Use `sbatch`, or `srun -p dev` for quick
checks.

```bash
sbatch sbatch/run_pipeline_qc_raw_voltage.sbatch     # submit FROM THE REPO ROOT
squeue --me
seff <jobid>
```

Jobs invoke modules as `python -m ieeg_ehr.<subpackage>.<module>`; no `cd` and
no `PYTHONPATH` needed. Slurm log paths in the sbatch files are **relative**
(`logs/%x_%A_%a.out`), which Slurm resolves against the submission directory —
so submit from the repo root or the logs land somewhere unexpected.

Per-subject work is parallelised as Slurm **array** jobs, one task per subject.
Keep `ckeller1 --qos=high_p` (4-job cap) free for interactive work; put long
arrays on `normal`.

## Provenance

Commit **and push** before any definitive or array run, so the commit hash
recorded in each artifact's sidecar matches the code that actually ran.
`ieeg_ehr.io.provenance.warn_if_dirty()` will warn on a dirty tree.
