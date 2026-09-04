# data_sop.md — using the `iEEG_EHR` dataset on Sherlock

**Audience:** anyone who has been given access to the Keller Lab's `ckeller1`
iEEG + EHR dataset on Sherlock and needs to find their way around it. It assumes
you know Python and neuroscience but **not** NWB, Sherlock, or this project. If
you have been handed a SUNet ID and a vague pointer at `/oak/.../iEEG_EHR`, start
here.

**What this covers:** where the data is, what is in it, how to load it, what is
known to be wrong with it, and the conventions for writing your own outputs. The
first four sections are the ones you actually need on day one.

**Status:** v2, 2026-09-04. Every path, count, series name, and field below was
verified against disk on that date by reading the files — not from memory or from
older docs. Counts drift; §9 has the commands to re-check them.

**On conflict, these win:** `docs/architecture.md` (the layer model),
`docs/io_conventions.md` (the artifact contract), `docs/view_registry.md` (the
view axes), `CLAUDE.md` (standing rules). This file is a map, not an authority
over them. Fix it when you notice drift.

---

## 1. The rules that matter

1. **Code lives in a git repo; data lives on Oak.** Never write an output, plot,
   cache, or model into a repo — not even scratch.
2. **Never run Python on the Sherlock login node.** Use `sbatch`, or
   `srun -p dev` / `sh_dev` for a quick interactive check.
3. **Oak is not backed up.** There is no undelete. Never `rm -rf` a path you did
   not create.
4. **Everything here is de-identified and must stay that way** — anonymized
   subject IDs, offset-anchored timestamps. No real dates, no ages, no names, in
   data *or* in prose you commit or share.
5. **Reuse before you recompute.** The expensive derivatives (bipolar PSD, QC
   masks, the epoch cache) already exist for most subjects. §9 says what.
6. **Which subjects a result covers is read from its `provenance.json`**, never
   guessed from a folder name.

---

## 2. Access and environment

### 2.1 Access

1. A Sherlock account (SUNet ID sponsored by a PI).
2. Membership in `ckeller1` (compute) and `oak_ckeller1` (storage). The data
   directories are `drwxrws---+` owned by `oak_ckeller1`. If `ls` on the data
   root gives permission denied, you are not in the group — ask the PI.
3. **IRB coverage under the protocol this dataset sits under.** Unix permissions
   are not authorization. Confirm with the PI before your first read.

```bash
id                                                  # your groups
ls /oak/stanford/groups/ckeller1/data/iEEG_EHR      # should show 3 entries
sh_quota
```

### 2.2 Environment

A shared virtualenv on `$GROUP_HOME` — use it rather than building a Conda env in
`$HOME`:

```bash
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
```

It has numpy 2.4.2, scipy 1.16.3, pandas 2.3.3, **pynwb**, h5py, pyarrow 20,
joblib, nilearn, nibabel, scikit-learn. Those core versions have deliberately
never been bumped — a ~38 TB derivatives tree was written against them.

Installing something new: on a compute node, one package at a time:

```bash
srun -p dev --time=00:20:00 --mem=8G bash -c '
  module load python/3.12
  source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
  export PIP_CACHE_DIR=$SCRATCH/.cache/pip
  pip install --no-deps --only-binary=:all: <package>'
```

`--only-binary=:all:` matters: Sherlock is CentOS 7 (glibc 2.17) and modern
wheels are `manylinux_2_28`, so without it pip falls through to a source build
and fails. `--no-deps` matters because an unpinned resolve can upgrade numpy
underneath the whole tree.

---

## 3. The map — where everything is on Oak

Root: `/oak/stanford/groups/ckeller1/data/iEEG_EHR/`

```
iEEG_EHR/
  iEEG_NWB/       RAW iEEG + EHR — treat as read-only
  badchan/        a separate bad-channel-detection effort (not this project's QC)
  derivatives/    ALL processed output, one namespace per person
    sisler/         the pain project — the reference implementation
```

### 3.1 `iEEG_NWB/` — the raw data

107 subject folders. BIDS-flavored: subject → session → modality.

```
iEEG_NWB/
  sherlock_file_registry.csv                    ← the run index (§3.2)
  sub-019/
    ses-01/
      ieeg/          sub-019_ses-01_run-CA0652B0.nwb     RAW VOLTAGE, ~1.5 GB per run
      ehr/           sub-019_ses-01_pain-scores.csv      (§6)
                     sub-019_ses-01_med-admin.csv
                     sub-019_ses-01_diagnoses.csv
      preprocessed/  sub-019_ses-01_run-..._bipolar_psd.nwb       ⚠ LEGACY, see §14.1
                     sub-019_ses-01_run-..._bipolar_fullTFR.nwb   ⚠ LEGACY
```

- A **session** (`ses-NN`) is one EMU admission — typically 5–9 days. Most
  subjects have exactly one.
- A **run** is one continuous recording block within it, typically 1.5–5 hours,
  with an opaque alphanumeric ID. **Sessions have many runs** — 35 for sub-019, 58
  for sub-093, 85 for sub-085. Never assume one run per session.
- The `preprocessed/` folder *inside* `iEEG_NWB/` is **not** the current
  derivative. See §14.1; it is a genuine trap.

### 3.2 `sherlock_file_registry.csv` — the run index

One row per run. This is the right way to enumerate the dataset; do not glob.

```
sub_id, ses_id, run_id, raw_file_path, raw_file_name, raw_file_size_mb,
preprocessed_file_path, has_preprocessed,
start_datetime, end_datetime, duration_minutes,
n_channels, n_timepoints, sampling_rate
```

```python
import pandas as pd
reg = pd.read_csv('/oak/stanford/groups/ckeller1/data/iEEG_EHR/'
                  'iEEG_NWB/sherlock_file_registry.csv')
reg.groupby('sub_id').size()          # runs per subject
```

**Use `sub_id`, `ses_id`, `run_id`, `raw_file_path`, and `start_datetime`. Be
careful with the rest.** As of 2026-09-04 (7,902 rows, 104 subjects):

| Column | State |
|---|---|
| `raw_file_path` | reliable |
| `start_datetime` / `end_datetime` / `duration_minutes` | **null in 2,136 of 7,902 rows (27 %)** |
| `n_channels`, `n_timepoints`, `sampling_rate` | **describe the LEGACY `band_power` series, not the raw iEEG.** `sampling_rate` is ≈0.0167 Hz (one sample per minute). Not the iEEG sampling rate |
| `preprocessed_file_path` | points at the in-tree legacy folder, **not** anyone's `derivatives/` tree (§14.1) |
| subject coverage | **104 of 107.** `sub-204`, `sub-243`, `sub-246` are on disk but absent from the registry |

Read the sampling rate, channel count, and duration **from the NWB file** (§4),
not from the registry.

### 3.3 `badchan/`

A separate bad-channel pipeline with its own layout
(`sub-XXX/ses-YY/{feature_extraction,model_scores,badchan}/` plus a `production/`
folder of `backup_*` variants). It is **not** the pain project's QC and the two
are not interchangeable. Ask its owner before consuming it.

### 3.4 `derivatives/` — where processed output goes

One namespace per person or project. **Claim your own; never write into someone
else's.**

```bash
mkdir -p /oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/<yourname>
```

Put a `README.md` at its root before you write anything into it, mirroring
`derivatives/sisler/README.md`. A derivatives tree with no README is unreadable
to the next person within about three months.

`derivatives/sisler/` is the worked example, and its four tiers are a real
distinction worth copying:

```
derivatives/sisler/
  README.md
  preprocessed/        CONTINUOUS per-window features over whole runs — expensive
    bipolar_fft/         the current bipolar Welch PSD (§5). 97 subjects
    bipolar_pac/         (future)
    bipolar_connectivity/(future)
  qc/                  DATA-QUALITY FACTS — properties of the data, inherited by all users
    raw_voltage/         metrics/ exclusions/<type>/<label>/ masks/<label>/ validation/
    bipolar/             pair-level variance QC
    feature_level/       channel quality computed on cached power
    psd_timing/          which PSD runs used the current windowing design
  features/            feature families SLICED to event windows
    pain/psd_epochs/epoch-5min-pre/
      manifest.json        the unit's self-description — READ THIS FIRST
      cache/               sub-XXX_ses-YY_epochs.parquet
      epoch_defs/          the epoch index
      channel_meta/        pair order + anatomical labels
      views/               optional materialized views (disposable)
  cohorts/             subject_id -> cohort assignments
  analysis/            TERMINAL outputs — figures, tables, models (§11)
  outdated/            superseded, kept for reference, never read by new code
```

| Tier | Means | Rebuild cost | Audience |
|---|---|---|---|
| `preprocessed/` | continuous per-window quantity over a whole run | expensive (reads raw) | everyone |
| `qc/` | a fact about data quality | moderate; thresholds re-sweep cheaply | everyone |
| `features/` | a feature family sliced to event windows | moderate | one event definition |
| `analysis/` | something a human reads or a model consumes | cheap | one question |

### 3.5 The code

Repo: `github.com/sabrasisler/iEEG_EHR_Analysis`, on Sherlock at
`/home/groups/ckeller1/sisler/iEEG_EHR_Analysis`. One installable package,
`src/ieeg_ehr/`, invoked as `python -m ieeg_ehr.<module>`. Relevant here:

| Module | What |
|---|---|
| `ieeg_ehr/io/nwb.py` | **the NWB loaders** (§4.5) |
| `ieeg_ehr/config/paths.py` | every path this project reads or writes |
| `ieeg_ehr/io/build_file_registry.py` | how the registry is built |
| `ieeg_ehr/preprocessing/run_pipeline_bipolar.py` | raw → bipolar → Welch PSD |
| `ieeg_ehr/views/channel_meta.py` | pulling anatomy out of the PSD NWB |

---

## 4. The NWB files — what is in them and how to load them

If you have not used NWB before: an `.nwb` file is an HDF5 file with a schema. You
open it with `pynwb`, get an `NWBFile` object, and navigate named groups. You can
also open it with plain `h5py` if you just want to poke at the structure. **Data
is lazy** — indexing a dataset is what actually reads from disk, which matters a
lot here because one run is ~1.5 GB.

There are **two different kinds** of NWB in this dataset, and mixing them up is
the most common early mistake:

| | Raw | Derived PSD |
|---|---|---|
| Path | `iEEG_NWB/sub-*/ses-*/ieeg/*.nwb` | `derivatives/sisler/preprocessed/bipolar_fft/sub-*/ses-*/*_bipolar_psd.nwb` |
| Contains | continuous voltage, `/acquisition` | spectral power, `/processing/ecephys` |
| Rows are | individual contacts | **bipolar pairs** |
| Size | ~1.5 GB per run | ~99 MB per run |
| Use when | you need the time-domain signal | you need power (almost always) |

### 4.1 Raw files: the acquisition series

A raw run puts its signals in `/acquisition` as **`ElectricalSeries` objects, one
per electrode type**. Five names exist across the dataset. A typical file has
three or four of them:

| Series | What it is | Present in |
|---|---|---|
| `ElectricalSeries_sEEG` | **depth (stereo-EEG) contacts — the intracranial data** | 88 / 107 subjects |
| `ElectricalSeries_ECoG` | **subdural grids/strips — also intracranial** | 7 / 107 subjects |
| `ElectricalSeries_scalp_EEG` | simultaneous scalp EEG | 45 / 107 subjects |
| `ElectricalSeries_EKG` | cardiac leads | 104 / 107 subjects |
| `ElectricalSeries_misc` | DC/trigger/reference and unlabeled channels | 104 / 107 subjects |

Which combination you get varies by subject (first run of each of 107 subjects):

```
 45 subjects   EKG + misc + sEEG
 42            EKG + misc + sEEG + scalp_EEG
  9            EKG + misc                          ← NO intracranial data at all
  4            EKG + misc + ECoG
  2            EKG + misc + ECoG + scalp_EEG
  1            EKG + misc + ECoG + sEEG
  1            EKG + misc + scalp_EEG              ← NO intracranial data
```

**Consequences you must handle:**

- **16 subjects have no `ElectricalSeries_sEEG`**, and code that hardcodes that
  key raises `KeyError` on all of them. They split into two very different
  groups:

  | Group | Subjects | What to do |
  |---|---|---|
  | **ECoG instead of sEEG** (6) | `sub-034, sub-061, sub-091, sub-094, sub-106, sub-116` | usable intracranial data — read `ElectricalSeries_ECoG` |
  | **No intracranial series at all** (10) | `sub-117, sub-137, sub-156, sub-160, sub-161, sub-162, sub-164, sub-165, sub-171, sub-174` | **cannot be used for iEEG analysis.** Only `EKG` + `misc` (one also has `scalp_EEG`) |

  One subject, `sub-067`, has **both** sEEG and ECoG — so "pick the first series
  that exists" is a choice there, not a fallback. Decide deliberately if it is in
  your cohort.

  The ten no-intracranial subjects fall in a contiguous ID band (117–174), which
  looks more like a conversion backlog than ten genuinely scalp-only patients —
  worth asking about rather than assuming they are permanently unusable.
- The lab's own `ieeg_ehr/io/nwb.py` currently hardcodes
  `nwb.acquisition['ElectricalSeries_sEEG']` and therefore does not handle these
  subjects (§14.2). Prefer the pattern in §4.5.
- **`misc` and `EKG` are not brain data.** They are in the same file and the same
  electrodes table. Never include them in a neural analysis by accident — §4.3
  shows how to tell them apart.

Fields on each series (verified on sub-019):

| Field | Value / meaning |
|---|---|
| `.data` | `(n_samples, n_channels)`, **float64**, gzip level 1 |
| `.unit` | `'volts'` |
| `.conversion` | `1.0` — multiply `.data` by it to get volts (do it anyway; don't assume) |
| `.rate` | sampling rate in Hz. **Varies by subject** — 500 Hz for sub-019, 1000 Hz for sub-085/093. Never hardcode it |
| `.starting_time` | **always `0.0`** — run-relative, so useless for locating a run in the session (§7) |
| `.electrodes` | index array into the shared electrodes table (§4.3) |
| `.description` | e.g. `'Continuous recording from sEEG electrodes (gzip compression level 1)'` |

### 4.2 What is *not* in the raw files

`/processing`, `/analysis`, `/intervals`, and `/scratch` are empty. There are no
stored events, no seizure annotations, no task epochs. **All event information
comes from the EHR CSVs** (§6). `nwb.subject` carries only `subject_id` and a
generic description — no age, no sex, no dates. That is deliberate: PHI is not on
Sherlock.

### 4.3 The electrodes table — the key concept

There is **one** electrodes table per file, at
`/general/extracellular_ephys/electrodes`, holding **every** contact across
**all** series. For sub-019 it is 72 rows = 40 sEEG + 30 misc + 2 EKG. Each
series' `.electrodes` field gives the row indices *it* uses.

**The indices are not contiguous and not sorted into blocks** — sub-019's `misc`
series uses rows `[19, 37, 38, 39, 40, 43, ...]`. So you cannot slice; you must
index:

```python
elec_all = nwb.electrodes.to_dataframe()      # all 72 rows
idx      = series.electrodes.data[:]          # this series' row indices
elec     = elec_all.iloc[idx]                 # rows for THIS series, in data-column order
```

That last line's row order matches the column order of `series.data`. Getting
this wrong silently mislabels every channel, which is the single worst failure
mode in this dataset.

The table has 38 columns. The ones that matter:

| Column | What it holds |
|---|---|
| **`location`** | **the channel LABEL, not anatomy** — `'LA1'`, `'LA2'`, `'EKG1'`, `'DC01'`. This is what the code uses as the channel name |
| `group_name` | which series it belongs to: `'sEEG'`, `'ECoG'`, `'scalp_EEG'`, `'EKG'`, `'misc'` — **the clean way to separate neural from non-neural** |
| `sEEG_ECoG` | `'sEEG'` / `'ECoG'` / `''` for non-neural |
| `shaft` | which depth electrode / strip the contact is on |
| `contact_num`, `chan_num` | position on the shaft; acquisition channel number |
| `Desikan_Killiany`, `DK_lobe`, `DK_ind` | DK atlas parcel, lobe, index |
| `Destrieux`, `Destr_long`, `Destr_ind` | Destrieux atlas |
| `FS_label`, `FS_vol`, `FS_ind` | FreeSurfer label / volume |
| `Yeo7`, `Yeo17` (+ `_ind`) | Yeo network assignment |
| `WMvsGM`, `PTD_ind`, `surr_GM_vox`, `surr_WM_vox` | grey/white matter and tissue-proportion measures |
| `LvsR` | hemisphere |
| `MNI_coord_1..3` | **MNI coordinates — use these for group-level anatomy and plotting** |
| `LEPTO_coord_1..3` | leptomeningeal (pial-projected) coordinates |
| `MGRID_coord_1..3` | native-space MGRID coordinates |
| `fsaverageINF_coord_1..3`, `subINF_coord_1..3` | inflated-surface coordinates (fsaverage / native) |

**Anatomical columns are empty strings for `EKG` and `misc` rows.** An empty
`Desikan_Killiany` is therefore a reliable non-neural flag as well — but filter on
`group_name`, which says what you mean.

Which coordinate space to use: **MNI** for anything cross-subject (group maps,
region assignment, glass brains); **LEPTO** if you are projecting to a pial
surface; **MGRID/subINF** only for within-subject native-space work.

### 4.4 Rows are contacts here, pairs in the derivative

The raw table has one row per **contact**. The derived PSD file's table has one
row per **bipolar pair** — 35 pairs for sub-019's 40 sEEG contacts — with
`location` like `'LA1-LA2'` and 62 columns, because every anatomical column is
duplicated as `_anode` and `_cathode` (`Desikan_Killiany_anode`,
`Desikan_Killiany_cathode`, …). See §14.4 on which of the two the project
currently uses.

### 4.5 Worked example: load one raw run

This handles the sEEG/ECoG variation and the electrode indexing correctly.

```python
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

# In-repo equivalents: ieeg_ehr.io.nwb.load_all_channels / load_channels_subset
# (but note those hardcode ElectricalSeries_sEEG — see §14.2)

INTRACRANIAL = ('ElectricalSeries_sEEG', 'ElectricalSeries_ECoG')

def load_intracranial(nwb_path, series_name=None):
    """Load one run's intracranial voltage.

    Returns (data_v, channel_names, sfreq, elec_df).
      data_v  : (n_samples, n_channels) float32, in volts
      elec_df : electrodes rows for these channels, in data-column order
    """
    with NWBHDF5IO(nwb_path, 'r') as io:
        nwb = io.read()

        # Pick the series rather than assuming sEEG exists.
        if series_name is None:
            available = [s for s in INTRACRANIAL if s in nwb.acquisition]
            if not available:
                raise KeyError(f'no intracranial series in {nwb_path}; '
                               f'have {list(nwb.acquisition)}')
            series_name = available[0]
        series = nwb.acquisition[series_name]

        if series.unit != 'volts':
            raise ValueError(f"unexpected unit {series.unit!r} in {nwb_path}")

        sfreq = float(series.rate)                      # 500 or 1000 Hz — read it
        idx = series.electrodes.data[:]                 # row indices, NOT a slice
        elec_df = nwb.electrodes.to_dataframe().iloc[idx]
        channel_names = list(elec_df['location'].values)

        # series.data is lazy; this line is the disk read.
        data_v = series.data[:].astype(np.float32) * np.float32(series.conversion)

    return data_v, channel_names, sfreq, elec_df
```

Read **one channel** or **one time window** without pulling 1.5 GB:

```python
with NWBHDF5IO(path, 'r') as io:
    series = io.read().acquisition['ElectricalSeries_sEEG']
    sfreq  = float(series.rate)

    chunk = series.data[0:int(60 * sfreq), :]      # first 60 s, all channels
    one   = series.data[:, 3]                     # one channel, whole run
    cols  = series.data[:, [0, 1, 5]]             # column fancy-indexing works
```

HDF5 fancy-indexing requires **sorted, unique** column indices — sort them and
map back afterwards, which is what `io/nwb.py::load_channels_subset` does.

Just looking at structure? Skip pynwb:

```python
import h5py
with h5py.File(path, 'r') as f:
    print(list(f['acquisition']))
    print(f['acquisition/ElectricalSeries_sEEG/data'].shape)
```

### 4.6 The derived PSD file — what you probably want instead

`derivatives/sisler/preprocessed/bipolar_fft/sub-XXX/ses-YY/*_bipolar_psd.nwb`,
plus a JSON sidecar with the parameters and git provenance. Produced by
`run_pipeline_bipolar.py`: read raw once → bipolar re-reference → Welch PSD. The
bipolar time-domain trace is **never persisted** (that would have been ~40 TB).

Two objects under `/processing/ecephys`:

| Object | Type | Shape (sub-019 run) | Contents |
|---|---|---|---|
| `psd_log_bins` | `DecompositionSeries` | `(17999, 35, 50)` float32 | **the main product**: (window, bipolar pair, frequency bin), `unit='log10(V^2/Hz)'` |
| `broadband_log_power` | `TimeSeries` | `(17999, 35)` float32 | mean log-power across non-line-noise bins, per window/pair |

Parameters: **2 s Hann windows, 50 % overlap (1 s hop), 50 log-spaced bins from
1–250 Hz, stored as log10 power.** `rate = 1.0` Hz because the hop is 1 s. 17,999
windows ≈ 5 hours.

```python
from pynwb import NWBHDF5IO
with NWBHDF5IO(psd_path, 'r') as io:
    nwb = io.read()
    psd = nwb.processing['ecephys']['psd_log_bins']

    bands = psd.bands.to_dataframe()   # bin_00 … bin_49
    # columns: band_name, band_limits (lo, hi Hz), contains_line_noise,
    #          band_mean, band_stdev
    good = ~bands['contains_line_noise'].values      # drop 60 Hz harmonics

    pairs = nwb.electrodes.to_dataframe()            # 35 rows, location 'LA1-LA2'
    x = psd.data[0:100, :, :]                        # first 100 windows — lazy until here
```

**`contains_line_noise` is why the log bins are stored rather than bands.** Bins
straddling 60 Hz and its harmonics are flagged so downstream code can exclude
them; aggregating to bands too early throws that away. For sub-019, 6 of the 50
bins are flagged — spanning roughly 53–66, 115–129, 161–200 and 224–250 Hz — so
`good.sum()` is 44. Aggregate on demand with
`preprocessing/bipolar_bands.py`, which does it **linear-then-log** to avoid
Jensen bias — see §12 and §14.3.

**Values are log10 power. Do not average them in the linear domain by
accident**, and if you exponentiate, do it in float64 (§12).

---

## 5. Putting data in time — and what is wrong with the timing

**This is the part of the dataset most likely to mislead you.** Read it before you
align anything to an event.

### 5.1 `session_start_time` is really the *run* start time

Every raw run's NWB header has `session_start_time`. Despite the name, **it holds
the start of that run, not of the session.** Verified across subjects — successive
runs of one session each carry their own value:

```
sub-019, ses-01 (35 runs).  EHR session_start = 2000-01-01 15:59:39
  run-CA0652B0   session_start_time = 2000-01-01T15:59:39
  run-CA0652B1   session_start_time = 2000-01-01T20:59:50
  run-CA0652B2   session_start_time = 2000-01-02T02:00:00
  run-CA0652B3   session_start_time = 2000-01-02T07:00:11
```

Only the **first** run's value equals the true session start, which is why the
mistake is easy to miss on a spot check. `timestamps_reference_time` is identical
to it, and `series.starting_time` is **always 0.0**, so neither helps.

**Therefore:**

- **Absolute time of sample `i` in a run** = that run's `session_start_time` +
  `i / series.rate`. This is correct and is what the project does.
- **True session (admission) bounds** come from the **EHR CSVs'** `session_start`
  / `session_end` columns (§6), *not* from any NWB field.
- **Never treat `session_start_time` as a session-level quantity**, and never
  compare it across runs expecting a constant.
- **Runs are not gapless.** sub-019's runs start 5 h 0 m 11 s apart but each holds
  9,000,000 samples at 500 Hz = exactly 5 h, so there is an ~11 s gap. Some runs
  are back-to-back, others have real gaps. If you concatenate runs, you must
  reconstruct absolute time per run and check for gaps — do not assume
  contiguity. (This is exactly what `qc/build_run_start_times.py` exists for.)

### 5.2 Timestamps are offset-anchored, not real

Every subject's timeline is shifted by a per-subject offset held on the PHI side.
Sessions appear to start on `2000-01-01`. **Intervals and durations are true;
absolute dates are fiction.** Never treat one as a clinical date, publish one, or
try to invert the offset.

Two corollaries: diagnosis dates in `diagnoses.csv` can appear decades before the
session (historical problem-list entries, shifted by the same offset), and
`file_create_date` is a *real* processing timestamp, not part of the
de-identified timeline — don't confuse the two.

---

## 6. The EHR tables

Three CSVs per subject/session in `ehr/`. All carry `sub_id`, `ses_id`,
`session_start`, `session_end` — **and those two columns are the authoritative
session bounds** (§5.1).

**`pain-scores.csv`** — the outcome variable.
```
sub_id, ses_id, date, max_pain, session_start, session_end
```
Spot ratings, roughly every 1–2 hours, 0–10. `date` is the rating time; `max_pain`
is the score. These are nurse-recorded clinical ratings, not task events — the
sampling is irregular and sparse (tens of ratings across a multi-day admission).

**`med-admin.csv`** — medication administration records.
```
sub_id, ses_id, taken_date, session_start, session_end, medication, line,
mar_action, sig, route, site, infusion_rate, infusion_rate_unit,
dose_unit, mar_duration, mar_duration_unit
```
`medication` is free text including formulation and strength (e.g. `'CEFAZOLIN IN
DEXTROSE (ISO-OS) 1 GRAM/50 ML IV PGBK'`). Expect to normalize drug names
yourself. `mar_action` distinguishes `Given` from other MAR outcomes — filter on
it.

**`diagnoses.csv`** — coded diagnoses.
```
sub_id, ses_id, session_start, session_end, date, type, source,
icd9_code, icd10_code, description, performing_provider, billing_provider
```
Both ICD-9 and ICD-10 appear; `source` distinguishes e.g. `Historical - HL7` from
encounter diagnoses. As noted, `date` may long precede the admission.

---

## 7. Identity conventions

- **Subject** — `sub-XXX`, zero-padded, anonymized. Paths and filenames use the
  prefixed form (`sub-019`); cohort files and provenance use the bare form
  (`"019"`). Both appear — convert, don't guess.
- **Session** — `ses-NN`, one EMU admission.
- **Run** — `run-<alnum>`, one recording block.
- **The canonical key is `(subject, session, run)`.** Per-window data adds a
  window index; epoched data adds an `epoch_id`.
- **Filenames carry identity; rows do not.** `sub-085_ses-01_epochs.parquet` has
  no subject column inside. This is deliberate — it makes a Slurm array task's
  output path a pure function of its task ID. Don't "fix" it.

---

## 8. What already exists — reuse before you recompute

Verified 2026-09-04.

| Artifact | Where | Coverage |
|---|---|---|
| Raw iEEG | `iEEG_NWB/sub-*/ses-*/ieeg/` | 107 subjects |
| EHR tables | `iEEG_NWB/sub-*/ses-*/ehr/` | per session |
| Run index | `iEEG_NWB/sherlock_file_registry.csv` | 104 subjects, 7,902 runs |
| **Bipolar PSD** | `derivatives/sisler/preprocessed/bipolar_fft/` | **97 subjects** |
| Raw-voltage QC masks | `derivatives/sisler/qc/raw_voltage/masks/` | 90 subject-sessions per label |
| Bipolar variance QC | `derivatives/sisler/qc/bipolar/` | ~82 subjects |
| **Pain epoch cache** | `derivatives/sisler/features/pain/psd_epochs/epoch-5min-pre/` | **83 files / 81 subjects, 35 GB** |
| Discovery cohort | `derivatives/sisler/cohorts/discovery-core-2026-07-28.json` | 65 subjects |

```bash
D=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
ls -1d $D/preprocessed/bipolar_fft/sub-* | wc -l
ls -1 $D/qc/raw_voltage/masks/gross-std3_satmargin15_sw_logz4/*.csv | wc -l
ls -1 $D/features/pain/psd_epochs/epoch-5min-pre/cache/*.parquet | wc -l
du -sh $D/features/pain/psd_epochs/epoch-5min-pre
python -c "import json,sys;print(json.load(open(sys.argv[1]))['params'])" \
  $D/features/pain/psd_epochs/epoch-5min-pre/manifest.json
```

**Read a cache's `manifest.json` before using it.** It states what has and has not
been done to the data. For `epoch-5min-pre`:

```json
{"epoch_minutes_before": 5.0, "anchor": "pain_score_time",
 "window_sec": 2.0, "overlap_frac": 0.5, "n_log_bins": 50, "dtype": "float32",
 "schema": ["epoch_id","window_idx","channel","bin","log_power"],
 "masked": false, "averaged": false, "normalized": false}
```

Those last three flags are the contract: the cache is **5-minute pre-rating
windows of per-2 s-window log power, with masking, normalization, and averaging
all left to the caller**, in that order (§9, §12).

---

## 9. Quality control

### 9.1 The metric/threshold split

Every QC level is built the same way:

```
metrics/     expensive, continuous, computed ONCE per subject/session
exclusions/<artifact_type>/<label>/   cheap boolean tables, one per threshold choice
masks/<label>/                        the OR'd union of that level's exclusions
validation/                           diagnostics, incl. threshold_sweeps/
```

Because metrics are stored continuously, **changing a threshold is cheap** — new
`exclusions/` and a new `masks/` label, without re-reading a single NWB. If you
are about to recompute a metric to try a different cutoff, you have misread the
layout.

Three levels: **`raw_voltage/`** (four detectors on the raw trace — saturation,
flatline, square-wave, gross-artifact; channel × 60 s bins; CSV),
**`bipolar/`** (per-2 s pair variance), **`feature_level/`** (channel quality
computed on cached power pre-normalization, so it is choice-independent).

### 9.2 Which mask to use

`ieeg_ehr.config.CANONICAL_MASK_LABEL` = `gross-std3_satmargin15_sw_logz4` — the
stricter of two full-cohort candidates and the only one with summary tables and
example plots. The other is `gross-std3_satmargin15_sw`.

**Apply it as a join at load time** on `(run, channel, 60 s bin)`. It is
deliberately *not* baked into the epoch cache, which is what makes switching masks
free. Don't build a second cache to try a second mask.

`qc/raw_voltage/validation/threshold_sweeps/` records how the thresholds were
chosen. Descriptive background on the detectors: `docs/qc_context.md`.

---

## 10. Deciding what your output is

Apply in order, stop at the first match:

1. **Terminal** — a human looks at it, or a model consumes it as input? →
   **ANALYSIS**, under `analysis/`.
2. **Non-terminal and a cheap transform of stored data?** → **VIEW**. Write a
   function; recompute at load. **Do not save it.**
3. **Non-terminal and expensive** (a new extraction, or an intermediate many
   things depend on)? → **STORED FEATURE**, under `features/` or `preprocessed/`.

*A view is a step; an analysis is a stop.* The default of "don't save" is the part
people get wrong: a recomputed view cannot go stale, a saved one can, silently.
Materialize only when recompute is **measured** slow *and* something depends on
it.

Name a cache directory after exactly the inputs that force a rebuild, and nothing
else — the pain cache is `epoch-5min-pre/` because the epoch definition is its
only baked-in input. Conversely, when a name genuinely does encode two inputs,
encode both: the bipolar mask label is `std10_rv-<raw_voltage_label>` because the
same threshold rolled against a different upstream mask is a *different mask*. On
2026-07-27 an 82-subject run silently clobbered a 17-subject run of the same name
for all 17 overlapping subjects. A collision the directory name makes impossible
beats one a doc merely warns about.

---

## 11. Writing outputs (condensed)

Full contract: **`docs/io_conventions.md`** — read it before writing a script that
produces output. In brief:

- **Every write emits a provenance sidecar in the same call.** Go through
  `ieeg_ehr.io` (`write_table`, `read_table`, `write_manifest`,
  `write_run_provenance`, `save_model`, `log_analysis`, `warn_if_dirty`); a bare
  `to_parquet` / `to_csv` / `joblib.dump` in new code is a bug. Sidecars record
  script, git commit + dirty flag, `params`, `config_hash`, `parents[]`,
  `subjects[]`.
- **`params` is the config that changes the output and nothing else** — it is
  hashed into `config_hash`, which is what staleness compares. A timestamp in
  there makes every artifact unique and the check worthless.
- **Formats:** Parquet for caches / `features/` / `preprocessed/`; CSV for tables
  under `analysis/` and the existing QC tree; joblib for models; JSON for
  manifests and sidecars. Never pickle tabular data. New artifacts only — don't
  bulk-convert.
- **Analysis output layout:** `analysis/<event>/<question>/<output_type>/[<view_scheme>/]<run_name>_<timestamp>/`.
  Levels 1–2 are opened deliberately; 3–5 freely per run. Always end in a
  timestamp so two runs can't overwrite each other's `provenance.json`. Sweep
  combinatorics are **rows in a `results.parquet`, never folders**.
- **Commit and push before a definitive or array run**, so the recorded commit
  hash describes the code that ran.
- Build paths with `config/paths.py` builders (`analysis_run_dir`,
  `pain_epoch_cache_path`, `bipolar_psd_nwb_path`, …), never by hand.

The pain project also keeps written records — a lab notebook, task list,
scratchpad, and an append-only decisions log — routed per `docs/WORKFLOW.md`.
That is internal bookkeeping for this project; it is not something a
collaborator needs to adopt.

---

## 12. Numerical rules that are easy to get wrong

Each of these was found by an audit, not invented.

- **Store narrow, compute wide.** The cache and PSD are float32. **Upcast to
  float64 before any reduction.** numpy does *not* do this for you — a bare
  `arr.mean(axis=0)` on float32 accumulates in float32 and keeps only ~6
  significant figures over ~300 windows. Largest precision loss in the chain.
- **Exponentiate log-power to linear in float64.** The worst stored log-power is
  ≈ −36.8, barely a decade above float32's smallest normal, so `10**x` in float32
  is one baseline division from underflowing to exactly zero.
- **Normalize per window, THEN average.** Baseline → per-window normalize →
  epoch-average → channel-average. Steps 2 and 3 do not commute for nonlinear
  operations (Jensen). This is the whole reason the cache is stored per-window.
- **Aggregate frequency bins and regions linear-then-log**, same reason.
- **Drop line-noise bins** using `contains_line_noise` before broadband
  aggregation.
- **Report contributing n.** Electrode coverage varies enormously across
  subjects; a region average over 2 subjects and one over 40 are not comparable.

---

## 13. Compute and storage

### 13.1 Slurm

Nothing heavy on the login node — no Python, no compiling, no `pip`. Set every
resource explicitly; the defaults (1 CPU, tiny memory, short walltime) will burn
you.

```bash
#SBATCH -p normal            # normal | bigmem | gpu | dev | owners
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/%x_%A_%a.out
```

Verify partitions with `sh_part`, not memory. Sherlock has no `--account`.
**Per-subject work is a Slurm ARRAY job, one task per subject** — not a loop.

```bash
sbatch --array=0-64%8 --export=ALL,SUBJECTS_FILE=/path/to/subjects.txt job.sbatch
squeue --me        # check every few minutes, never in a tight loop
seff <jobid>       # post-mortem: did you ask for 10x the memory you used?
```

Log paths in this repo's sbatch files are relative, so **submit from the repo
root**. Job I/O goes to `$SCRATCH`, not `$HOME`. Never `find` from `/` or the root
of `$SCRATCH`/`$OAK` — the Lustre metadata server is shared. Never guess a module
version; `ml spider <name>` first.

A one-run raw NWB is ~1.5 GB on disk but float64 in memory — request memory for
the array you actually materialize, and prefer windowed reads (§4.5) over
`series.data[:]` when you can.

### 13.2 Storage

```
HOME           15 GB    backed up      never put job output here
GROUP_HOME      1 TB    backed up      venvs live here
SCRATCH       100 TB    NOT backed up  purged 90 days after last content write
OAK            80 TB    NOT backed up  the data; 37.8 TB used (47%) on 2026-09-04
```

- **Oak has no backup and no undelete.** Treat `rm` as permanent. If something
  should go, move it to `outdated/` and say so.
- **The `$SCRATCH` purge timer resets only on a real content write.** `touch`,
  rename, and `chmod` do **not** extend a file's life.
- **Inodes are a quota too** (12 M here) — which is why caches are one file per
  subject/session, not per epoch.
- Check before a big build: `sh_quota`, `du -sh ./*`, or `ml load system ncdu`.

---

## 14. Known inaccuracies and landmines

Each of these has already cost someone time.

**14.1 Two different files are named `*_bipolar_psd.nwb`.**

```
iEEG_NWB/sub-019/ses-01/preprocessed/sub-019_ses-01_run-CA0652B0_bipolar_psd.nwb
   → 575 KB, Feb 2026. LEGACY. Contains a per-minute `band_power` series.
     This is what the registry's preprocessed_file_path points at, and what its
     n_channels / n_timepoints / sampling_rate columns describe.

derivatives/sisler/preprocessed/bipolar_fft/sub-019/ses-01/sub-019_ses-01_run-CA0652B0_bipolar_psd.nwb
   → 99 MB, Jul 2026. CURRENT: 2 s / 50 % overlap / 50 log bins (§4.6).
```

Same filename, ~170× size difference. Use `config.bipolar_psd_nwb_path()`. The
in-tree `_bipolar_fullTFR.nwb` files are likewise legacy.

**14.2 `session_start_time` is the run start, not the session start.** See §5.1 —
the most consequential inaccuracy in the dataset. True session bounds are in the
EHR CSVs. `series.starting_time` is always 0.0 and cannot help.

**14.3 Code that hardcodes `ElectricalSeries_sEEG` breaks on 16 subjects** —
including the lab's own `ieeg_ehr/io/nwb.py`, whose three loaders all do
`nwb.acquisition['ElectricalSeries_sEEG']` and so raise `KeyError` on those
subjects. Six have ECoG instead, ten have no intracranial data, and `sub-067` has
both. §4.1 has the lists; §4.5 has the pattern that handles it.

**14.4 The registry is incomplete.** 104 of 107 subjects (`sub-204`, `sub-243`,
`sub-246` missing); 2,136 of 7,902 rows have null `start_datetime` /
`duration_minutes` / `n_channels`; and the `sampling_rate` column is the legacy
band_power rate (~1/60 Hz), not the iEEG rate. Read timing and geometry from the
NWB.

**14.5 Band definitions disagree between docs and code.**
`docs/architecture.md` says beta 15–25 / gamma 25–70 / high_gamma 70–170.
`ieeg_ehr.config.psd_params.CANONICAL_BANDS_HZ` says beta 13–30 / low_gamma 30–58
/ high_gamma1 65–115 / high_gamma2 125–175 / high_gamma3 185–235 — split
specifically to fall *between* 60 Hz harmonics, which the doc's edges straddle.
A third, coarser grouping (`VIOLIN_BANDS_HZ`) is used only by violin plots. **The
code is what ran.** Always state which band set a number came from.

**14.6 Bipolar-pair anatomy is anode-based, and that is a temporary stand-in.**
A pair inherits the DK parcel of its anode (`Desikan_Killiany_anode`), which is
wrong whenever the two contacts straddle a boundary. The intended replacement is a
lookup on the pair's midpoint coordinate. Treat assignments near parcel
boundaries as approximate.

**14.7 Reading the region list off `config.ROI_REGIONS` is a bug.** It is the
*default* scheme's 15 regions, resolved at import. Filtering a view's regions
against it silently keeps only the 8 of 21 whose names appear in the default set,
with no error. Use `analysis.view_tables.roi_regions_for(view_params)`.

**14.8 Documented layouts that no longer match disk.** The unit directory is
`epoch-5min-pre/`, not `epoch-5min-pre_mask-<label>/` as drawn in
`architecture.md` PART 4 and `io_conventions.md` §5 — the mask moved to the view
layer. And `analysis/scratch/` is flat, not `analysis/pain/scratch/`.
`config/paths.py` is correct; the drawings are stale.

**14.9 The de-identification anchor is documented as 2001 and appears on disk as
2000-01-01.** Don't resolve it by picking one: only offsets and intervals are
meaningful, and no absolute date from this dataset is a real date.

**14.10 A sweep result is a nomination, not a finding.** For this project:
per-subject effect sizes, sign-consistency across subjects, contributing n — never
a pooled p-value that ignores per-subject structure. The hold-out cohort is
unreachable by default and gated by an explicit flag; discovery subjects are
locked as discovery permanently. Looking at the hold-out during exploration is not
undoable.

---

## 15. Checklists

### 15.1 First day

- [ ] `id` shows `ckeller1` and `oak_ckeller1`
- [ ] IRB / human-subjects coverage confirmed with the PI
- [ ] `ls /oak/stanford/groups/ckeller1/data/iEEG_EHR` works
- [ ] Venv activates and `import pynwb` succeeds (on a dev node, not the login node)
- [ ] Loaded the registry into pandas; understand `(sub, ses, run)`
- [ ] Opened **one** raw NWB and listed `/acquisition` — saw which series that subject has
- [ ] Opened **one** derived PSD NWB and pulled `bands.to_dataframe()`
- [ ] Read §5 (timing) and §14 (landmines) end to end
- [ ] Read `derivatives/sisler/README.md`
- [ ] Claimed `derivatives/<yourname>/` with a README

### 15.2 Before an analysis run

- [ ] Does what I need already exist? (§8)
- [ ] Am I using the current PSD, not the legacy in-tree one? (§14.1)
- [ ] Does my code handle subjects with ECoG-only or no intracranial data? (§4.1)
- [ ] Am I indexing the electrodes table via `series.electrodes`, not slicing? (§4.3)
- [ ] Am I excluding `EKG` / `misc` channels? (§4.3)
- [ ] Am I reading `series.rate` per run rather than assuming 500 or 1000 Hz? (§4.1)
- [ ] Is my absolute-time math per-run, and am I checking for gaps? (§5.1)
- [ ] Which QC mask, applied at load time? (§9.2)
- [ ] Am I upcasting to float64 before any reduction? (§12)
- [ ] Output path from a `config` builder; committed and pushed? (§11)
- [ ] Slurm resources set explicitly; array not a loop; quota headroom? (§13)

---

## 16. Glossary

| Term | Means |
|---|---|
| **session** | one EMU admission, `ses-NN`, typically 5–9 days |
| **run** | one continuous recording block within a session; many per session |
| **`ElectricalSeries`** | an NWB object holding continuous voltage for one electrode type |
| **electrodes table** | the single per-file table of all contacts; each series indexes into it |
| **`DecompositionSeries`** | the NWB object holding the spectral decomposition (the PSD) |
| **bipolar re-reference** | subtracting adjacent contacts on a shaft; rows become pairs |
| **feature family** | a continuous per-window neural quantity over a whole run. `preprocessed/` |
| **cache** | a feature family sliced to event windows — pre-normalization, pre-averaging. `features/` |
| **view** | a cheap deterministic transform of a cache. A **function**, recomputed at load |
| **analysis** | a terminal output: a human reads it or a model consumes it. `analysis/` |
| **metric / exclusion / mask** | QC's three stages: expensive measurement → cheap threshold → union |
| **sidecar** | the `*.provenance.json` / `manifest.json` written alongside an artifact |
| **nomination** | a sweep result; not yet a finding |
