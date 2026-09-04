# iEEG_EHR on Sherlock — collaborator guide

**What this is:** everything you need to find, load, and interpret the Keller Lab
iEEG + EHR dataset on Sherlock. Written to be read start-to-finish in ten minutes,
or pasted into Claude as context. Self-contained — you should not need any other
doc in this repo.

**What the data is:** intracranial EEG (sEEG depth electrodes, some ECoG grids)
recorded continuously over multi-day epilepsy-monitoring-unit admissions, plus the
clinical EHR record for the same admission (nurse-recorded pain scores, medication
administrations, diagnoses). ~107 subjects. All de-identified.

**Contact:** Sabra Sisler (sisler@stanford.edu) for access, anything that looks
wrong, or before you write into shared storage.

There is a longer internal version of this (`docs/data_sop.md`) with the pain
project's own conventions, QC philosophy, and analysis layout. You do not need it.

---

> ### ⚠ Heads-up: the raw files are being reprocessed right now and during the reprocessing, an additional ~125 subjects will be added to Sherlock. 
>
> **Timeline:** starting 2026-09-04, over roughly the next two weeks.
>
> **Every raw `.nwb` in `iEEG_NWB/` is being rewritten but it will be saved in a different folder temporarily** with a different internal
> **chunk layout**, to make loading raw data faster going forward (§5.3).
>
> **The structure does not change.** Same paths, same filenames, same
> `ElectricalSeries` names, same electrodes table, same units and sampling rates.
> Code written against this guide will keep working — you do not need to wait, and
> you do not need to rewrite anything, with the exception of possibly needing to rewrite code for > timestamp extraction in the NWBs.
>
> **What it does mean:**
> - **File mtimes and sizes will change**, so anything you cache or fingerprint
>   against a raw file's `(bytes, mtime)` will look stale afterwards. Prefer
>   deriving from `(sub, ses, run)` identity over file metadata.
> - **A run may be mid-rewrite when you read it.** If a read fails oddly or a file
>   looks truncated, retry later before debugging your own code.
> - **Read performance will improve**, and the best access pattern may change —
>   re-time your loader afterwards rather than carrying over a workaround.
> - **The file registry (§4) predates this** and will need rebuilding.
> - **The timestamps in the NWB session start might change** I will be super clear with what that change looks like, but it should be relatively minor. I will also include an up to date csv to ensure that every subjust is documented with what version NWB file it is and what the timestamp conventions are while I am in the middle of reprocessing. more info will be added as that changes. 



---

## 1. Access

You need (a) a Sherlock account, (b) membership in the `ckeller1` and
`oak_ckeller1` groups, (c) human-subjects/IRB coverage under the protocol this
dataset sits under. Unix permissions are not authorization — confirm (c) with the
PI before your first read.

```bash
id                                                  # should list ckeller1, oak_ckeller1
ls /oak/stanford/groups/ckeller1/data/iEEG_EHR      # -> badchan  derivatives  iEEG_NWB
```

Permission denied means you are not in the group yet.

---

## 2. Environment

There is a **shared Python venv on Sherlock — you are welcome to use it** rather
than building your own:

```bash
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
```

It has everything needed to read this data: **pynwb** and **h5py** (the NWB
files), **numpy 2.4.2 / scipy 1.16.3 / pandas 2.3.3**, **pyarrow 20** (Parquet),
matplotlib, joblib, scikit-learn, nilearn + nibabel (for plotting electrodes on a
template brain).

Those core versions have deliberately never been bumped — a large derivatives tree
was written against them, so please don't upgrade numpy/scipy/pandas/pynwb in the
shared venv. To add a package, do it on a compute node, one at a time:

```bash
srun -p dev --time=00:20:00 --mem=8G bash -c '
  module load python/3.12
  source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
  export PIP_CACHE_DIR=$SCRATCH/.cache/pip
  pip install --no-deps --only-binary=:all: <package>'
```

`--only-binary=:all:` matters — Sherlock is CentOS 7 (glibc 2.17), so without it
pip falls through to a source build and fails. If you'd rather have your own env,
`python -m venv` on `$GROUP_HOME` or `$SCRATCH` (**not** `$HOME`, 15 GB quota).

**Never run Python on the login node.** Use `srun -p dev --pty bash` or `sh_dev`
for interactive work, `sbatch` for anything real. Set `--time`, `--mem`,
`--cpus-per-task` and `-p` explicitly; the defaults are tiny.

---

## 3. Code

Repo: **`github.com/sabrasisler/iEEG_EHR_Analysis`**, on Sherlock at
`/home/groups/ckeller1/sisler/iEEG_EHR_Analysis`. It is one installable package
(`src/ieeg_ehr/`), already installed editable in the shared venv, so
`import ieeg_ehr` works from any directory.

The files worth looking at for loading data:

| File | What it gives you |
|---|---|
| `src/ieeg_ehr/io/nwb.py` | NWB loaders (see the caveat in §9) |
| `src/ieeg_ehr/config/paths.py` | every path in the dataset, as functions — `bipolar_psd_nwb_path(sub, ses, run)`, `pain_scores_csv(sub, ses)`, … |
| `src/ieeg_ehr/preprocessing/run_pipeline_bipolar.py` | how the PSD derivative in §6 was produced |
| `src/ieeg_ehr/preprocessing/bipolar_bands.py` | aggregating the stored frequency bins into canonical bands |
| `src/ieeg_ehr/views/channel_meta.py` | pulling pair anatomy out of a PSD file |

The repo holds **code only** — no data. Everything data lives on Oak (§4).

---

## 4. Where the data is

Root: `/oak/stanford/groups/ckeller1/data/iEEG_EHR/`

```
iEEG_NWB/                                    RAW — treat as read-only
  sherlock_file_registry.csv                 one row per run (see below)
  sub-019/ses-01/
    ieeg/         sub-019_ses-01_run-CA0652B0.nwb    raw voltage, ~1.5 GB/run
    ehr/          sub-019_ses-01_pain-scores.csv     §7
                  sub-019_ses-01_med-admin.csv
                  sub-019_ses-01_diagnoses.csv
    preprocessed/ ..._bipolar_psd.nwb                ⚠ LEGACY — do not use, see §9
derivatives/                                 ALL analysis output on this dataset — §10
  sisler/                                    Sabra's outputs; one folder per person
    preprocessed/bipolar_fft/                the current PSD derivative — §6
    qc/                                      channel/window quality masks
    features/ analysis/ cohorts/             pain-project specific
  <yourname>/                                yours goes here — §10
badchan/                                     a separate bad-channel effort; ask before using
```

**Identity is `(subject, session, run)`.** A *session* (`ses-NN`) is one EMU
admission, typically 5–9 days; most subjects have one. A *run* is one continuous
recording block inside it, typically ~1.5–5 h, with an opaque ID like
`run-CA0652B0`. **Sessions have many runs** — 35, 58, 85 for different subjects.
Never assume one run per session.


---

## 5. The raw NWB files

An `.nwb` file is HDF5 with a schema. Open with `pynwb`, navigate named groups.
**Datasets are lazy** — indexing is what actually reads from disk, which matters
when one run is 1.5 GB.

Continuous voltage lives in `/acquisition` as **`ElectricalSeries` objects, one
per electrode type**. Five names exist across the dataset; a given file has three
or four of them:

| Series | What |
|---|---|
| `ElectricalSeries_sEEG` | depth contacts — the intracranial data, most subjects |
| `ElectricalSeries_ECoG` | subdural grids/strips — also intracranial, a handful of subjects |
| `ElectricalSeries_scalp_EEG` | simultaneous scalp EEG |
| `ElectricalSeries_EKG` | cardiac leads — **not brain data** |
| `ElectricalSeries_misc` | DC/trigger/reference/unlabeled — **not brain data** |

**Do not hardcode `ElectricalSeries_sEEG`.** Some subjects have ECoG instead, a
few have both, and some have no intracranial series at all. Check what's there.

Per-series fields:

| Field | Value |
|---|---|
| `.data` | `(n_samples, n_channels)`, **float64**, gzip level 1. ~1.5 GB compressed on disk but ~2.9 GB in memory for a 5 h / 40-channel / 500 Hz run — size your `--mem` for what you materialize, and prefer windowed reads |
| `.unit` | `'volts'` |
| `.conversion` | `1.0` — multiply anyway rather than assuming |
| `.rate` | **sampling rate in Hz — varies by subject (500, 1000, …). Read it, never hardcode** |
| `.starting_time` | always `0.0` (run-relative) — useless for absolute time, see §8 |
| `.electrodes` | index array into the file's single electrodes table (§5.2) |

`/processing`, `/intervals`, `/analysis` and `/scratch` are **empty** in the raw
files: no stored events, no seizure annotations, no task epochs. **All event
information comes from the EHR CSVs** (§7). `nwb.subject` carries only a subject
ID — no age, sex, or dates, by design.

### 5.1 Loading a run

```python
import numpy as np
from pynwb import NWBHDF5IO

INTRACRANIAL = ('ElectricalSeries_sEEG', 'ElectricalSeries_ECoG')

def load_intracranial(nwb_path):
    """-> (data_volts (n_samples, n_chan) float32, channel_names, sfreq, elec_df)"""
    with NWBHDF5IO(nwb_path, 'r') as io:
        nwb = io.read()

        available = [s for s in INTRACRANIAL if s in nwb.acquisition]
        if not available:
            raise KeyError(f'no intracranial series; have {list(nwb.acquisition)}')
        series = nwb.acquisition[available[0]]

        sfreq = float(series.rate)                       # read it per run
        idx = series.electrodes.data[:]                  # row indices, NOT a slice
        elec_df = nwb.electrodes.to_dataframe().iloc[idx]
        names = list(elec_df['location'].values)

        data = series.data[:].astype(np.float32) * np.float32(series.conversion)
    return data, names, sfreq, elec_df
```

Read a window or a channel instead of the whole 1.5 GB:

```python
with NWBHDF5IO(path, 'r') as io:
    series = io.read().acquisition['ElectricalSeries_sEEG']
    sfreq  = float(series.rate)
    chunk  = series.data[0:int(60 * sfreq), :]   # first 60 s, all channels
    one    = series.data[:, 3]                   # one channel, whole run
```

HDF5 fancy-indexing needs **sorted, unique** column indices — sort, then map back.

Just want the structure? Skip pynwb: `h5py.File(path)['acquisition']`.

### 5.2 The electrodes table — where the anatomy is

There is **one** electrodes table per file, at
`/general/extracellular_ephys/electrodes`, holding **every** contact across **all**
series. Each series' `.electrodes` field gives the row indices *it* uses, and
those indices are **not contiguous and not sorted into blocks** — you must index,
not slice:

```python
elec_all = nwb.electrodes.to_dataframe()   # ALL contacts, incl. EKG/misc
idx      = series.electrodes.data[:]
elec     = elec_all.iloc[idx]              # this series, in data-column order
```

That row order matches the column order of `series.data`. Getting it wrong
silently mislabels every channel — the worst failure mode in this dataset.

The 38 columns, verbatim:

```
location, group, group_name, chan_num,
FS_label, FS_vol, FS_ind, WMvsGM, LvsR, sEEG_ECoG,
Desikan_Killiany, DK_ind, DK_lobe,
Destrieux, Destr_ind, Destr_long,
Yeo7, Yeo7_ind, Yeo17, Yeo17_ind,
PTD_ind, surr_GM_vox, surr_WM_vox,
LEPTO_coord_1..3, MNI_coord_1..3, MGRID_coord_1..3,
subINF_coord_1..3, fsaverageINF_coord_1..3
```

The ones that matter:

| Column | Holds |
|---|---|
| **`location`** | the channel **label**, not anatomy — `'LA1'`, `'EKG1'`, `'DC01'`. This is the channel name everywhere in the pipeline |
| **`group_name`** | `'sEEG'` / `'ECoG'` / `'scalp_EEG'` / `'EKG'` / `'misc'` — **the clean way to keep non-brain channels out of your analysis** |
| `shaft`, `contact_num` | which depth electrode / strip, and position along it |
| **`Desikan_Killiany`**, `DK_lobe`, `DK_ind` | DK atlas parcel (e.g. `'precentral'`, `'superiortemporal'`, `'Left-Amygdala'`) |
| `Destrieux`, `Destr_long` | Destrieux atlas (e.g. `'G_temporal_middle'`) |
| `FS_label`, `FS_vol` | FreeSurfer label / volume |
| **`Yeo7`, `Yeo17`** (+ `_ind`) | Yeo network assignment (e.g. `'Default'`, `'Limbic'`, `'17Networks_9'`) |
| `WMvsGM` | `'GM'` / `'WM'` |
| `LvsR` | hemisphere |
| **`MNI_coord_1..3`** | **MNI coordinates — use these for anything cross-subject** |
| `LEPTO_coord_1..3` | pial-projected coordinates |
| `MGRID_coord_1..3`, `subINF_coord_1..3` | native space; within-subject work only |
| `fsaverageINF_coord_1..3` | fsaverage inflated surface |

Which coordinate space: **MNI** for group-level anatomy, region assignment, and
glass-brain plots; **LEPTO** for pial-surface projection; MGRID/subINF for
within-subject native space only.

**Three things to know about the anatomy columns:**

1. **Deep contacts read `'Depth'`, not a network.** A contact in white matter or a
   subcortical structure has `Yeo7 == 'Depth'`, `Yeo17 == 'Depth'`, and
   `Destrieux == 'Depth'` — a literal string, not a missing value. Yeo networks
   are a *surface* parcellation, so only contacts near cortex get a real network
   label. If you are selecting contacts by network, filter `Yeo7` against the
   networks you want rather than filtering out empties, and cross-check with
   `WMvsGM` and `Desikan_Killiany`.
2. **`EKG` and `misc` rows have empty anatomy fields.** Filter on `group_name`,
   which says what you mean.
3. **A few subjects' electrodes tables have no `Desikan_Killiany` column at all**
   (`sub-093`, `sub-154`, `sub-159`, `sub-240` — verified 2026-09-04), so a direct
   column access raises `KeyError`. Guard with `if 'Desikan_Killiany' in
   elec.columns` and fall back to `MNI_coord_*` or skip the subject.

### 5.3 Chunking — and why it is changing

The voltage datasets are gzip-compressed and **HDF5-chunked**, which sets what a
read actually costs. As of 2026-09-04 the chunk shape is **`(10000, n_channels)`**
— 20 s at 500 Hz, spanning *all* channels. Because a chunk is the unit of
decompression, that layout means:

- **Time-window reads are cheap.** `series.data[a:b, :]` touches only the chunks
  covering `[a, b)`.
- **Single-channel reads are expensive.** `series.data[:, 3]` still decompresses
  all 40 channels of every chunk in the run to return one column.

So on the current files, **prefer windowed reads over channel-wise reads**, and if
you need a few channels across a long stretch, read time-blocks and subset columns
in memory.

**This is exactly what the reprocessing at the top of this guide changes.** The
chunk layout is being rewritten for more efficient raw loading; the array shapes,
dtypes, and everything in §5.1–5.2 stay the same. Re-time your access pattern
against the new files rather than keeping a workaround built for the old ones.

---

## 6. The preprocessed PSD (in `derivatives/sisler/`)

Almost certainly what you want instead of raw voltage. Path:

```
derivatives/sisler/preprocessed/bipolar_fft/sub-XXX/ses-YY/
  sub-XXX_ses-YY_run-ZZZ_bipolar_psd.nwb     the data
  sub-XXX_ses-YY_run-ZZZ_bipolar_psd.json    parameters + git provenance
```

Present for **97 subject folders** (as of 2026-09-04); ~99 MB per run. Produced by
`run_pipeline_bipolar.py`: read each raw run once → bipolar re-reference → Welch
PSD. **The bipolar time-domain trace is never stored** (~40 TB avoided) — if you
need a bipolar *signal* rather than power, you re-derive it from raw.

**Exact parameters:**

| | |
|---|---|
| Reference | bipolar, adjacent contacts on the same shaft — **rows are PAIRS, not contacts** |
| Window | **2 s Hann, 50 % overlap → 1 s hop** |
| Frequency | **50 log-spaced bins, 1–250 Hz** (edges in the JSON sidecar) |
| Stored value | **log10 power**, `unit='log10(V^2/Hz)'`, **float32** |
| Series `rate` | `1.0` Hz (one value per 1 s hop) |

Two objects under `/processing/ecephys`:

| Object | Type | Shape | Contents |
|---|---|---|---|
| **`psd_log_bins`** | `DecompositionSeries` | `(n_windows, n_pairs, 50)` | the main product: (window, bipolar pair, frequency bin) |
| `broadband_log_power` | `TimeSeries` | `(n_windows, n_pairs)` | mean log-power across non-line-noise bins |

```python
from pynwb import NWBHDF5IO

with NWBHDF5IO(psd_path, 'r') as io:
    nwb = io.read()
    psd = nwb.processing['ecephys']['psd_log_bins']

    bands = psd.bands.to_dataframe()        # rows bin_00 … bin_49
    # columns: band_name, band_limits (lo, hi Hz), contains_line_noise,
    #          band_mean, band_stdev
    good = ~bands['contains_line_noise'].values

    pairs = nwb.electrodes.to_dataframe()   # one row per PAIR, location 'LA1-LA2'
    x = psd.data[0:100, :, :]               # lazy until this line
```

**`contains_line_noise` is why log bins are stored rather than bands.** Bins
straddling 60 Hz and its harmonics are flagged; drop them (`good`) before any
broadband aggregation. Aggregate to canonical bands with
`preprocessing/bipolar_bands.py`, which does it **linear-then-log** to avoid
Jensen bias.

**The pair electrodes table** has ~62 columns: every anatomical column from §5.2
duplicated as `_anode` and `_cathode` (`Desikan_Killiany_anode`,
`Desikan_Killiany_cathode`, …), with `location` like `'LA1-LA2'`. The pain project
currently assigns each pair the parcel of its **anode**, which is wrong when the
two contacts straddle a boundary; the better move is a lookup on the pair's
midpoint coordinate. Treat near-boundary assignments as approximate.

**Values are log10 power.** Don't average them as if they were linear, and if you
exponentiate, do it in **float64** — the smallest stored values are ~−36.8, close
enough to float32's floor that `10**x` in float32 can underflow to exactly zero.
Same rule for any reduction: upcast before you average, because
`arr.mean(axis=0)` on float32 accumulates in float32.

There are also **channel/window QC masks** under `derivatives/sisler/qc/` —
per-channel × 60 s-bin exclusions from four detectors on the raw trace
(saturation, flatline, square-wave, gross artifact). The current label is
`gross-std3_satmargin15_sw_logz4`, applied as a join on `(run, channel, 60 s
bin)`. Ask Sabra whether you want it; for a first pass you may not.

---

## 7. The EHR tables

Three CSVs per subject/session under `ieeg/../ehr/`. All carry `sub_id`, `ses_id`,
`session_start`, `session_end` — **and those last two are the authoritative
session bounds** (§8).

**`pain-scores.csv`** — `sub_id, ses_id, date, max_pain, session_start, session_end`

Nurse-recorded spot ratings, 0–10, roughly every 1–2 h, so tens of ratings across
a multi-day admission. `date` is the rating time, `max_pain` the score. These are
clinical ratings, **not task events** — irregular and sparse.

**`med-admin.csv`** — `sub_id, ses_id, taken_date, session_start, session_end,
medication, line, mar_action, sig, route, site, infusion_rate,
infusion_rate_unit, dose_unit, mar_duration, mar_duration_unit`

`medication` is free text including formulation and strength (e.g. `'CEFAZOLIN IN
DEXTROSE (ISO-OS) 1 GRAM/50 ML IV PGBK'`) — expect to normalize drug names
yourself. Filter on `mar_action == 'Given'`.

**`diagnoses.csv`** — `sub_id, ses_id, session_start, session_end, date, type,
source, icd9_code, icd10_code, description, performing_provider, billing_provider`

Both ICD-9 and ICD-10 appear; `source` distinguishes historical problem-list
entries (`'Historical - HL7'`) from encounter diagnoses, so `date` can precede the
admission by decades.

---

## 8. Timestamps — read this before aligning anything. This logic is going to change when the files are reprocessed. I will update the documentation for that once I start uploading the reprocessed data. 

**`session_start_time` in an NWB header is the start of that RUN, not the
session.** Despite the name. Successive runs of one session each carry their own
value:

```
sub-019, ses-01 (35 runs).   EHR session_start = 2000-01-01 15:59:39
  run-CA0652B0   session_start_time = 2000-01-01T15:59:39   <- only the first matches
  run-CA0652B1   session_start_time = 2000-01-01T20:59:50
  run-CA0652B2   session_start_time = 2000-01-02T02:00:00
```

Only the *first* run's value equals the true session start, which is why a spot
check misses it. `timestamps_reference_time` is identical to it, and
`series.starting_time` is always `0.0`, so neither helps.

Therefore:

- **Absolute time of sample `i` in a run** = that run's `session_start_time` +
  `i / series.rate`. Do this per run.
- **True session (admission) bounds** come from the EHR CSVs' `session_start` /
  `session_end`, never from an NWB field.
- **Runs are not gapless.** sub-019's runs start 5 h 0 m 11 s apart but hold
  exactly 5 h of samples each — an ~11 s gap. Others are back-to-back. If you
  concatenate runs, reconstruct absolute time per run and check for gaps.
- To align an EHR event (a pain rating) to signal: find the run whose
  `[session_start_time, session_start_time + n_samples/rate)` contains the event
  time, then convert to a sample index within that run.

**All timestamps are offset-anchored, not real.** Each date in the NWB files and the csvs in the EHR folder is
shifted by a per-subject offset held on the PHI side, so sessions appear to start
on `2000-01-01`. The recorded time of day is preserved  **Intervals and durations are true; absolute dates are fiction.**
Never treat one as a clinical date, publish one, or try to invert the offset.
(`file_create_date` is a real processing timestamp, not part of the de-identified
timeline — don't confuse them.).

---

## 9. Gotchas

1. **Two different files are named `*_bipolar_psd.nwb`.**
   `iEEG_NWB/sub-*/ses-*/preprocessed/..._bipolar_psd.nwb` is **legacy** — ~575 KB,
   a per-minute `band_power` series. The current one is
   `derivatives/sisler/preprocessed/bipolar_fft/...`, ~99 MB, described in §6.
   Same filename, ~170× size difference. Use
   `ieeg_ehr.config.paths.bipolar_psd_nwb_path()`.  The in-tree
   `_bipolar_fullTFR.nwb` files are not legacy, but were created by Sandon and I don't have information on those parameters. These files will be migrated shortly.
2. **`session_start_time` is the run start.** §8. The most consequential
   inaccuracy in the dataset.
3. **Don't hardcode `ElectricalSeries_sEEG`** — including via the repo's own
   `io/nwb.py`, whose loaders currently do exactly that and so raise `KeyError` on
   subjects with ECoG-only or no intracranial data. Use the §5.1 pattern.
4. **Four subjects so far have no `Desikan_Killiany` column.** §5.2.
5. **Ignore the file registry, it is not up to date.** 
7. **`EKG` and `misc` channels sit in the same file and the same electrodes
   table** as brain data. Filter on `group_name`.
8. **Band definitions differ between this repo's docs and its code**
   (`ieeg_ehr.config.psd_params.CANONICAL_BANDS_HZ` splits gamma to fall *between*
   60 Hz harmonics; the architecture doc's edges straddle them). The code is what
   ran — always state which band set a number came from.
9. **float32 storage, float64 arithmetic.** §6.

---

## 10. Where to save what you produce

**If you save anything derived from this dataset to Oak, it goes in
`derivatives/` — not next to the raw data, and not in someone else's folder.**
This is a convention we are actively trying to establish, so please follow it even
for throwaway work; the whole point is that a year from now it is still obvious
who made what.

**Claim your own folder** and put a `README.md` in it before you write anything
else:

```bash
cd /oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives
mkdir -p <yourname>
$EDITOR <yourname>/README.md      # what's in here, who made it, what produced it
```

Inside your folder, **ideally split reusable data from your own analysis**:

```
derivatives/<yourname>/
  README.md
  preprocessed/     data you PREPROCESSED that someone else could reuse
                    — re-referenced signals, spectral decompositions, filtered
                    or resampled data, epoched arrays. Document the parameters.
  analysis/         your own outputs — figures, stats tables, models.
                    Nobody else is expected to reuse these.
```

The `preprocessed/` split is the part that matters. Preprocessing is the expensive
step, and it is the step most likely to be duplicated: if you re-reference and
decompose the whole dataset and leave the result inside a folder called
`my_scan_analysis/`, the next person redoes it from raw. Put it in
`preprocessed/`, say what parameters produced it, and it becomes shared
infrastructure. `derivatives/sisler/preprocessed/bipolar_fft/` (§6) is the worked
example — data plus a JSON sidecar recording the parameters and git commit.

**Storage constraints are real — be mindful of what you write to Oak.** Oak is
purchased capacity, not free space, and it is **not backed up**. Before a big
build:

- **Estimate first.** Build one subject, `du -sh` it, multiply by ~100. If the
  answer is tens of TB, come talk to Sabra before launching.
- **Check headroom** with `sh_quota` — the group share is a shared pool, and
  filling it blocks everyone.
- **Don't store what you can cheaply recompute.** Anything that is a fast
  transform of data already on disk is better as a function than as a file.
- **Never persist a full-resolution derivative of the raw time series without
  asking.** A bipolar copy of this dataset would be ~40 TB, which is why the PSD
  pipeline (§6) never writes the bipolar trace at all.
- **Intermediates and scratch go to `$SCRATCH`** (100 TB, purged after 90 days of
  no content writes), not Oak. Only durable, documented artifacts earn an Oak
  path.
- **Prefer float32 and a columnar format** (Parquet) for large tables, and one
  file per subject/session rather than per epoch — **inodes are a quota too.**

---

## 11. Rules of the road

- **Never run Python (or `pip`, or a compile) on the login node.** `srun -p dev`
  or `sbatch`. Don't `find` from `/` or the root of `$OAK`/`$SCRATCH` — shared
  Lustre metadata server.
- **Write your outputs to `derivatives/<yourname>/`** — see §10, which is the
  convention we're trying to hold to. Job scratch goes to `$SCRATCH` (purged 90
  days after last *content* write — `touch` does not reset the timer), never
  `$HOME` (15 GB).
- **Everything here is de-identified and must stay that way** — anonymized subject
  IDs, offset-anchored timestamps. No real dates, no ages, no names, in data *or*
  in prose, code comments, figure titles, or anything you commit or share.
- **Record what produced each output** — at minimum the script, the git commit,
  and which subjects went in. This repo does it with JSON sidecars next to every
  artifact (`ieeg_ehr.io.write_table` and friends); any equivalent is fine, but
  a figure whose subject list can't be recovered is a figure that has to be
  regenerated.