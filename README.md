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
not the mechanism. See `CLAUDE.md` for the full rule set, and the Oak tree's own
`README.md` for a map of what lives where over there.

## What `src/ieeg_ehr/` is

`src/ieeg_ehr/` is all of this project's code, as a single installable Python
**package** named `ieeg_ehr`. That name is what you import:

```python
from ieeg_ehr import config
from ieeg_ehr.qc import build_mask
```

```bash
python -m ieeg_ehr.qc.build_exclusions --level-root ... --artifact-type all
```

It replaced three loose top-level folders (`qc_scripts/`, `preprocessing/`,
`pain_analysis/`) that could only be imported if your shell happened to be
sitting in the repo root — which is why every Slurm job used to begin with a
hardcoded `cd /home/groups/ckeller1/sisler/iEEG_EHR_Analysis`. As an installed
package it works from any directory, so that line is gone.

The `src/` wrapper is the conventional Python "src layout": it keeps importable
code in one place, separate from things that are not code (docs, sbatch, tests).
`src/ieeg_ehr.egg-info/` sitting next to it is pip's auto-generated install
metadata — not source, gitignored, ignore it.

## Layout

```
CLAUDE.md            operating rules — read first
docs/                architecture, kickoff plan, view registry (normative)
                     + qc_context, pain_analysis_context (background)
src/ieeg_ehr/        THE CODE — one importable package
  config/            single source of paths, thresholds, band defs, pinned mask
  io/                provenance, table + NWB helpers, file registry builder
  preprocessing/     bipolar re-reference + Welch PSD (the stored feature family)
  qc/                raw-voltage + bipolar QC: detectors, exclusions, masks, plots
  features/          pain epoch-power cache builder
  views/             the seven view axes — EMPTY, P1.3 fills it
  analysis/          the five pain plot scripts
sbatch/              all Slurm jobs
tests/
outdated/            superseded, kept for reference — never imported
  notebooks/           12 exploratory notebooks, no longer used
  scripts/             superseded root scripts (raw_voltage_qc.py etc.)
  sbatch/              single-use / one-off jobs
  preprocessing/       the pre-bipolar pipeline
logs/                gitignored
```

## Setup

The environment is a shared venv on `$GROUP_HOME`, not in this repo:

```bash
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
```

The package is already installed editable, so `import ieeg_ehr` works from
anywhere and your edits take effect immediately — no reinstall needed. You only
need to reinstall if the repo itself moves. On a compute node, never the login
node:

```bash
srun -p dev --pty bash -c '
  module load python/3.12
  source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
  pip install -e /home/groups/ckeller1/sisler/iEEG_EHR_Analysis --no-deps'
```

`--no-deps` is deliberate: a bare `pip install -e .` would try to resolve every
dependency and could upgrade the working numpy/pandas/pynwb underneath a 1.8 TB
derivatives tree. Install new dependencies explicitly, one at a time.

**Not yet installed:** `pyarrow` and `joblib` (P0.3). Until then
`ieeg_ehr.io.tables.save_table` still writes CSV, not Parquet.

## Running things

Never run Python on the login node. Use `sbatch`, or `srun -p dev` for quick
checks.

```bash
sbatch sbatch/run_pipeline_qc_raw_voltage_normal.sbatch   # submit FROM THE REPO ROOT
squeue --me
seff <jobid>
```

Jobs invoke modules as `python -m ieeg_ehr.<subpackage>.<module>`; no `cd`, no
`PYTHONPATH`. Slurm log paths are **relative** (`logs/%x_%A_%a.out`), which
Slurm resolves against the submission directory — so submit from the repo root
or the logs land somewhere unexpected.

Array jobs take a `SUBJECTS_FILE` override and size the array to its line count:

```bash
sbatch --array=0-18%8 \
  --export=ALL,SUBJECTS_FILE=/oak/.../derivatives/sisler/cohorts/subjects_qc_raw_voltage_normal.txt \
  sbatch/build_exclusions_array.sbatch
```

Per-subject work is parallelised as Slurm **array** jobs, one task per subject —
not by running multiple agents. Keep `ckeller1 --qos=high_p` (4-job cap) free
for interactive work; put long arrays on `normal`.

## Provenance

Commit **and push** before any definitive or array run, so the commit hash
recorded in each artifact's sidecar matches the code that actually ran.
`ieeg_ehr.io.provenance.warn_if_dirty()` warns on a dirty tree.

## Known open items

- **`CANONICAL_MASK_LABEL` is not formally pinned** (P0.1). It is currently
  `gross-std3_satmargin15_sw_logz4`; the other full-cohort candidate is
  `gross-std3_satmargin15_sw`. It is baked into every downstream cache, so
  changing it means an expensive full re-run.
- **Band definitions disagree.** `docs/architecture.md` states beta 15-25 /
  gamma 25-70 / high_gamma 70-170; the code
  (`ieeg_ehr.config.psd_params.CANONICAL_BANDS_HZ`) uses beta 13-30 /
  low_gamma 30-58 / high_gamma1-3, split to avoid 60 Hz harmonics. Band choice
  is a P2.2 sweep axis — resolve before that sweep.
- **The pain plot scripts have no cache to read** until P1.1 rebuilds it. Both
  legacy CSV caches are archived under Oak `outdated/`.
