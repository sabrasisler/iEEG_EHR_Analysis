# dataset_v2.md — `iEEG_EHR_V2`, and why loading code has to change

**Status:** written 2026-09-09 from a file-by-file audit of what was on disk that
day. Every claim below was verified by reading the NWBs, not inferred from the
conversion code. Raw output and the audit scripts:
`$OAK/data/iEEG_EHR/derivatives/sisler/analysis/scratch/v2_nwb_audit_2026-09-09/`
(`REPORT.md` + JSON).

**What this is:** a second, re-converted copy of the raw iEEG NWBs at
`/oak/stanford/groups/ckeller1/data/iEEG_EHR_V2/`, adjacent to the original
`iEEG_EHR/`. Same recordings, same voltages, different file-level conventions.

**Why you are reading this:** two of the differences will silently produce wrong
answers if existing code is pointed at V2 unchanged (§2, §3). One of them (§4)
is an opportunity worth measuring before the next expensive re-extraction.

**Relationship to the other docs:** `docs/data_sop.md` describes **V1** and is
still correct for V1. Where the two disagree about V2, this file wins. Nothing
in V2 changes the layer model, the artifact contract, or the view axes — those
docs are untouched by it.

---

## 1. Status and scope — V2 is not usable as a drop-in yet

As of 2026-09-09 V2 holds **3 subjects, 455 GB**, and is still being uploaded
(`sub-195` went 25 → 52 runs during a single afternoon's audit):

| Subject | V2 runs | V1 runs | Note |
|---|---|---|---|
| `sub-190` | 96 | 96 | identical run IDs |
| `sub-195` | 52 | 52 | identical run IDs |
| `sub-270` | 41 | — | **new**, not in V1 |

Also missing from V2, and needed before anything real runs off it:

- **No `ehr/` folder.** The pain scores, med-admin, and diagnoses CSVs exist
  only under V1. `ehr_aligned` is `False` for all 261 rows of the tracking file.
- **No derivatives.** `bipolar_fft` (97 subjects), the QC masks, and the pain
  epoch cache were all built from V1 files.
- The status tables are **stale relative to disk** — `n_files_uploaded` is 0 for
  `sub-190` and `sub-195` though 96 and 52 files are present; every participant
  row reads `participant_status: pending`. Trust the filesystem, not the CSV,
  for what has landed. Conversion overall: 4,345 completed, **455 failed**,
  13,441 pending.

So: V2 is a preview of the format, not yet a dataset to analyze.

## 2. The data is identical — this was checked, not assumed

24 shared runs across `sub-190` and `sub-195`, five 30,000-sample slices per run
(start / ¼ / ½ / ¾ / end), all channels:

- `ElectricalSeries_sEEG` — **bit-exact**, every run, every slice, max abs diff `0.0`
- `ElectricalSeries_EKG` — **bit-exact**, every run
- `ElectricalSeries_misc` — bit-exact after the `sub-190` remap in §5

`dtype` (float64), `unit` (volts), `conversion` (1.0), `resolution`, `offset`,
`rate`, `n_samples` and channel count are unchanged in every shared run. The
electrodes table is the same 126 rows and the same 44 columns in the same order,
with the same `location` ordering — the only value differences are the two rows
in §5.

**Consequence:** any derivative rebuilt from V2 should reproduce the V1
derivative exactly, which makes V2 a usable correctness check on the pipeline
rather than a confound.

## 3. Timestamps — the change that will silently corrupt results

This is the one that matters most.

| | V1 | V2 |
|---|---|---|
| `session_start_time` | start of the **run** (varies run to run) | start of the **session** (**constant** across all runs) |
| `timestamps_reference_time` | = `session_start_time` | = `session_start_time` |
| `series.starting_time` | **always 0.0** | **offset in seconds from session start** |
| `timestamps` array | absent | absent — still `starting_time` + `rate` |

V1's naming was wrong (`data_sop.md` §5.1, §14.2 exist only to warn about it);
V2 fixes it. But the fix means the V1 formula is now wrong in a way that does
not raise:

```python
# V1-only. On V2 this stacks EVERY run at the session start.
t_abs = nwb.session_start_time + i / series.rate          # WRONG on V2
```

**Use this instead — it is correct on BOTH versions**, because V1's
`starting_time` is always exactly 0.0:

```python
t_abs = nwb.session_start_time + timedelta(seconds=series.starting_time) \
        + timedelta(seconds=i / series.rate)              # correct on V1 and V2
```

Prefer the version-agnostic form over branching on dataset root. There is no
case where adding `starting_time` is wrong.

**Verified losslessly invertible.** For every shared run:

```
V2.session_start_time + V2.starting_time  ==  V1.session_start_time
```

Checked on **every shared filename — 148/148** (96 `sub-190` + 52 `sub-195`,
after `sub-195` finished uploading), across **all** `ElectricalSeries` in each
file, **max error 0.000000000 s**. Run *end* (`start + n_samples/rate`) also
matches 148/148, since `rate` and `n_samples` are identical throughout.

So for two files of the same name: the stored `session_start_time` **field**
differs in 146 of 148 — it agrees only on each subject's first run, where
`starting_time == 0` — but the **actual run start they encode is identical in
every case**. Nothing was lost in the re-encoding; V1 and V2 disagree about the
field's meaning, never about the instant.

Consistency checks that also passed 148/148: `timestamps_reference_time ==
session_start_time` in both versions; every series within a file decodes to the
same start; and V1's `starting_time` is `0.0` for every series of every run.
(Script: `starttime_confirm.py` in the audit directory.)

**V2 `starting_time` is wall-clock offset, not cumulative duration.** `sub-190`
happens to be gapless (7,200,000 samples @ 1000 Hz = exactly the 2 h run
spacing), so the two coincide there and a spot check on that subject would not
reveal the difference. `sub-195` is not gapless: cumulative duration falls
**86,916 s (~24 h) short** of the wall-clock offset by the end of the session.

So on V2 a single run's own header places it in the session — no cross-run
arithmetic, no gap reconstruction, no "runs are not gapless" trap.
`qc/build_run_start_times.py` and `data_sop.md` §5.1 / §14.2 exist to work
around a problem V2 does not have.

Session bounds are also now available without touching the NWBs:
`status/participant_tracking.csv` carries `session_first_timepoint` (matches V2
`session_start_time` exactly) and `session_last_timepoint` (matches
`last_run_start + last_run_duration` exactly).

## 4. Chunking — and the loading strategy worth testing

| | V1 | V2 |
|---|---|---|
| Chunk shape | `(10000, n_channels)` | `(min(120 × rate, n_samples), 1)` |
| In seconds | 10 s × **all** channels | **120 s × 1** channel |
| Rate-adaptive | no (fixed 10,000 samples) | yes (240,000 for `sub-270` @ 2000 Hz) |
| Compression | gzip 1, no shuffle | gzip 1, no shuffle (unchanged) |
| File size | — | ~11 % smaller (2.38 GB → 2.11 GB, `sub-190` run E0) |

Verified on all 180 V2 runs: `chunk[1]` is always 1, and
`chunk[0] == min(120 × rate, n_samples)` in 180/180 (short final runs get a
short chunk).

### 4.1 Measured cost

`sub-190` run `EA1896E0`, `ElectricalSeries_sEEG`, 116 ch @ 1000 Hz, best of 3:

| Access pattern | V1 | V2 | |
|---|---|---|---|
| 60 s × all channels | 0.19 s | 0.41 s | 2.09× slower |
| 5 min × all channels | 1.04 s | 1.30 s | 1.25× slower |
| 10 min mid × all channels | 1.70 s | 1.81 s | 1.07× slower |
| **whole run × 1 channel** | 18.91 s | **0.16 s** | **118× faster** |
| **60 s × 1 channel** | 0.18 s | ~0.00 s | ~50× faster |

The time-slice penalty shrinks as the slice lengthens, because a whole 120 s
chunk gets decompressed either way. **Reading in ≥120 s blocks aligned to chunk
boundaries makes it nearly free.** The current pipeline's habit of pulling short
full-width windows is the worst case for V2 and the easiest thing to fix.

### 4.2 The open question: is per-channel the right loop order?

**Not a decision, a thing to measure when the pipeline next gets touched.**

Pulling a whole channel for a whole run is now essentially free (0.16 s vs 19 s),
which makes a channel-major pipeline plausible where it previously was not:

- **Memory.** One channel-run at 1000 Hz × 2 h is 7.2 M float64 = 58 MB, against
  ~1.5–2.4 GB for the full run. That is the difference between a `--mem=32G`
  array task and a `--mem=4G` one, and `data_sop.md` §13.1's warning about
  float64 memory largely evaporates.
- **Parallelism.** The natural array-task unit could become
  `(subject, run, channel)` instead of `(subject)`. Far more tasks, each tiny —
  which suits `normal` backfill and keeps `ckeller1 --qos=high_p` (4-job cap)
  free. It also removes the current straggler problem where one 85-run subject
  sets the wall-clock for the whole array.
- **Fit with what already exists.** Welch PSD, the four raw-voltage QC
  detectors, and the bipolar variance metric are all *already* per-channel or
  per-pair computations that currently happen to sit inside a load-everything
  loop. Bipolar re-referencing needs exactly two channels at a time, which is
  still a cheap read under V2 chunking.

**What would have to be checked before committing to it:**

1. **Does the per-channel win survive the per-task overhead?** 116 channels ×
   96 runs is ~11 k Slurm tasks per subject, and Python + venv + `NWBHDF5IO`
   startup is seconds each. A per-channel *loop inside* a per-run task probably
   captures most of the benefit with none of the scheduler cost — measure that
   shape first, it is the boring answer and likely the right one.
2. **Inode and provenance cost.** One output file per channel would blow the
   quota and violate the "one file per subject/session" cache rule. Channel-major
   *compute* must still write subject-major *output*.
3. **Anything genuinely cross-channel.** A common-average or Laplacian reference,
   and any connectivity feature, needs many channels at one timepoint — the
   access pattern V2 made slower. Those stay time-major.
4. **Whether it matters at all.** The expensive extractions are already done for
   V1. This only pays off on a re-extraction, so measure it *when* one is
   scheduled, not before.

Do not restructure the pipeline on the strength of a benchmark on one run of one
subject. Do treat "read whole channels, not short full-width windows" as the
default shape for **new** V2 code.

## 5. Channel classification — V2 reclassified `C3`/`C4`, and V2 is probably right

For **`sub-190`**, V2 has **no `ElectricalSeries_scalp_EEG`**. Its two channels,
labelled **`C3`** and **`C4`** (electrodes rows 123, 124), were reclassified
`group_name: scalp_EEG → misc` and folded into the misc series, which grew
6 → 8 channels. The data is preserved exactly, column for column:

```
V2 misc col 0..4 (E, DC01, DC02, CCEP, DC04)  <- V1 misc  col 0..4   exact
V2 misc col 5    (C3)                         <- V1 scalp col 0      exact
V2 misc col 6    (C4)                         <- V1 scalp col 1      exact
V2 misc col 7    (Events)                     <- V1 misc  col 5      exact
```

This is not a blanket policy: `sub-270` keeps a proper
`ElectricalSeries_scalp_EEG` with 8 channels (`Fp1 F7 T3 T5 Fp2 F8 T4 T6`).
`sub-195` has no scalp channels in either version.

**First read of this was that V2 had dropped two real scalp electrodes. That was
wrong — the V1 label is the error.** The measurements below are why.
(Scripts + JSON: `validate_scalp.py`, `c3c4_deep.py` in the audit directory.)

### 5.1 The structural tell

Surveying `group_name == 'scalp_EEG'` across all 45 V1 subjects that have the
series: montage sizes are 8 (20 subjects), 19, 16, 14, 10, 3 — the recognisable
10-20 montages — **and 2 (14 subjects). In all 14, the two channels are exactly
`C3` and `C4`.** No clinician places C3 and C4 and nothing else. That is an
amplifier's default channel table, not a montage.

### 5.2 The signal, versus a real scalp montage

`sub-270`'s 8-channel montage is the positive control. 300 s, first run each:

| Subject | ch | std | uniq | flat | slope | max abs corr vs intracranial | corr(C3,C4) |
|---|---|---|---|---|---|---|---|
| **sub-270 (true scalp)** | Fp1 | **10.4 µV** | 2447 | 5.4 % | −2.59 | 0.92 | — |
| **sub-270 (true scalp)** | F8 | **4.4 µV** | 1448 | 7.8 % | −2.38 | 0.92 | — |
| sub-206 | C3 | 25.0 µV | **2** | **99.9 %** | −2.00 | **1.0000** (RORB1) | **+1.0000** |
| sub-099 | C3 | 784.7 µV | 23648 | 1.8 % | −0.89 | **1.0000** (ROF2) | **+1.0000** |
| sub-122 | C3 | 3022.8 µV | 31895 | 64.2 % | −2.50 | **0.9969** (ROF7 / REF) | **+0.9999** |
| sub-088 | C3 | 257.1 µV | 13767 | 0.6 % | −2.28 | 0.4663 | +0.7811 |
| sub-190 | C3 | 283.5 µV | 17626 | 3.1 % | −2.95 | 0.2732 | +0.9530 |

Three of the five are provably not EEG:

- **`sub-206`** — 2 unique values, 99.9 % flat. A constant. Dead channel.
- **`sub-099`** — `corr = 1.0000` with `ROF2` (and C4 with `ROF7`),
  `corr(C3,C4) = 1.0000`, 60 Hz relative power of **24,110**. Duplicated,
  saturated garbage.
- **`sub-122`** — `corr = 0.997` with `ROF7` and with the `REF` channel, 64 %
  flat. A copy of the reference.

**The discriminator that actually works is amplitude.** Real scalp EEG in the
control sits at **4–16 µV**. Every `C3`/`C4` channel measured is **205–3023 µV** —
20× to 200× too large, i.e. on the intracranial amplifier's range, not a scalp
one. `sub-190` and `sub-088` are the least obviously broken of the set (no exact
duplicate, plausible 1/f slope) but both fail the amplitude test by ~20×, and
`sub-190`'s `corr(C3,C4) = 0.953` is more consistent with two channels seeing the
same reference than with two independent central derivations.

Caveat on the obvious-looking test: high correlation with an intracranial channel
is **not** by itself disqualifying — the true `sub-270` scalp channels correlate
0.82–0.93 with sEEG through the shared reference. It is `|corr|` of exactly
1.0000 that means duplication.

### 5.3 What this means

- **V2's reclassification is a fix, not a regression.** Treating `sub-190`'s
  `C3`/`C4` as `misc` is the correct call, and the same is very likely true for
  the other 13 two-channel subjects.
- **Do not classify EEG channels by name.** The V1 labels look like they came
  from pattern-matching 10-20 names, and the naming is not trustworthy: 30
  distinct labels, inconsistent case (`FP1` vs `Fp1`, `FZ` vs `Fz`), two
  conventions mixed across subjects (old 10-20 `T3/T4/T5/T6` vs 10-10
  `T7/T8/P7/P8`), and noise channels that carry ordinary-looking names. Validate
  against the signal.
- **A workable validator**, in rough order of decisiveness: reject constants and
  near-constants (unique-value count, flat fraction); reject exact duplicates
  (`|corr| ≥ 0.999` against any other channel in the file, including `REF`);
  reject on amplitude (scalp should be single-digit-to-low-tens of µV, not
  hundreds); then sanity-check the 1/f slope and that a montage is a montage
  (a 2-channel "scalp montage" is a red flag on its own).
- **No effect on the pain project**, which selects `group_name == 'sEEG'` and
  never touched these channels. This matters for whoever wants scalp EEG, and
  for the ~250 subjects still to be converted.

Open items tracked in `TASKS.md` (validate the remaining 9 of 14, and settle the
rule before the full re-conversion).

## 6. Everything else that changed

| Field | V1 | V2 |
|---|---|---|
| `identifier` | `sub-190_ses-01` — **same for all 96 runs** | `sub-190_ses-01_EA1896E0` — **unique per run** |
| `session_description` | `iEEG recording from patient sub-190` | `iEEG sub-190_ses-01_run-EA1896E0` |
| `source_script` / `_file_name` | absent | `…/conversion_scripts/convert_session_to_nwb_multithread` |
| series `description` | `Continuous recording from sEEG electrodes (gzip compression level 1)` | `sEEG, gzip level 1` |
| device description | `Nihon Kohden EEG-1200A` | `EEG-1200A V01.00 Version 2` (real amp firmware string) |
| electrodes table `description` | `Electrode metadata with custom columns from CSV` | `Canonical session electrode table` |
| `file_create_date` | 1 entry | 2 entries (written, then reopened) |
| all `object_id` attrs | — | regenerated (expected for a new file) |

The `identifier` change is a real fix: in V1 every run of a session shared one
NWB `identifier`, which the spec says must be unique.

**Unchanged:** `institution`, `lab`, `experimenter`, `subject` (id + generic
description only — still no PHI), the `general/*` group layout, the 44-column
electrodes schema, the `ElectricalSeries_*` naming convention, and the fact that
`/processing`, `/intervals`, `/analysis` and `/scratch` are all still empty.

## 7. Directory layout and the new index files

| | V1 | V2 |
|---|---|---|
| Root | `iEEG_NWB/` | `raw/` |
| Per-session folders | `ieeg/`, `ehr/`, `preprocessed/` | **`ieeg/` only** |
| EHR CSVs | present | **absent** (§1) |
| Legacy in-tree `preprocessed/` | present — the `data_sop.md` §14.1 trap | **gone** |
| Run index | `sherlock_file_registry.csv` — 7,902 rows, 27 % null timing | `status/ieeg_file_catalog.csv` — 18,241 rows, 210 subjects |
| Subject index | — | `status/participant_tracking.csv` — 261 rows |

The new indexes are strictly better and worth reading instead of the NWB headers
where they suffice:

- `ieeg_file_catalog.csv` — one row per source `.EEG` file:
  `anonymized_datetime`, `anonymized_start_time`, `recording_duration_sec`,
  `is_first_file`, plus per-file `nwb_conversion_status` and
  `sherlock_upload_status`.
- `participant_tracking.csv` — one row per subject/session:
  `session_first_timepoint`, `session_last_timepoint`, `n_seeg_channels`,
  `n_ecog_channels`, `n_scalp_channels`, `elec_loc_filename`.

Both replace the V1 registry's misleading `n_channels` / `sampling_rate` columns,
which described the legacy per-minute `band_power` series rather than the iEEG
(`data_sop.md` §14.4). Note §1: the status columns lag the filesystem.

## 8. Checklist before pointing anything at V2

- [ ] Absolute time uses `session_start_time + starting_time + i/rate` (§3) —
      the version-agnostic form, not a branch on dataset root
- [ ] Nothing reads `session_start_time` expecting a per-run value, or compares
      it across runs expecting it to vary
- [ ] Reads are ≥120 s and chunk-aligned, or per-channel — not short full-width
      windows (§4.1)
- [ ] `ElectricalSeries_sEEG` is not hardcoded (`io/nwb.py` still does this —
      `data_sop.md` §14.3; `sub-270` is fine but the ECoG-only subjects will
      break again on re-conversion)
- [ ] EHR still resolves to V1 (§1)
- [ ] Output paths still come from `config/paths.py` and land on Oak — V2 changes
      nothing about the artifact contract (`docs/io_conventions.md`)
