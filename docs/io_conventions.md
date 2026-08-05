# io_conventions.md — how artifacts are written, read, and traced

**Read this before writing any script that produces output.** It is the
operational half of the rules in `CLAUDE.md` ("IO / naming", "Git / provenance")
and of `architecture.md` PART 2's `save_path` rule. Status: v1, built at P0.3
(2026-07-27), when the helpers below became real code rather than a plan.

Companions: `architecture.md` (what the layers ARE), `view_registry.md` (the
seven view axes), `WORKFLOW.md` (where *records* go, as opposed to artifacts).

---

## 1. The contract

1. **Parquet** for tables, **joblib** for fitted objects, **JSON** for manifests
   and sidecars. Never pickle tabular data.
2. **Every write emits a provenance sidecar in the same call.** A bare
   `df.to_parquet` / `joblib.dump` / `to_csv` is forbidden — not by convention
   alone, but because the sanctioned writers make it the easier path.
3. **Every read of a saved artifact checks staleness.** Warn while exploring,
   `on_stale='refuse'` for anything a reported number comes out of.
4. **Commit + push before a definitive/array run**, so the commit hash the
   sidecar records describes the code that actually ran. `io.warn_if_dirty()` at
   script start makes a dirty tree loud.
5. **One line per run into the repo's text index**: `log_analysis(desc, run_dir)`,
   called beside the sidecar write. This is the only sanctioned write *into* the
   repo (`docs/WORKFLOW.md`).

Everything lands on Oak under the config module's derivatives base. No output
path is ever repo-relative — including scratch plots (`CLAUDE.md`, CODE/DATA
BOUNDARY).

---

## 2. The API

`from ieeg_ehr import io` gives one flat namespace; you never have to remember
which submodule a helper lives in.

| Call | Use it for |
|---|---|
| `io.write_table(df, path, params=..., parents=..., subjects=...)` | any table. Parquet by extension, sidecar in the same call |
| `io.read_table(path, columns=[...], parents=..., config=..., on_stale=...)` | reading one back, staleness-checked. `columns` is why the cache is Parquet |
| `io.write_manifest(unit_dir, params=...)` | a cache / base-unit directory's `manifest.json` |
| `io.read_manifest(unit_dir)` / `io.manifest_ref(unit_dir)` | reading it / referencing it as a parent |
| `io.write_view_sidecar(save_path, view_config=..., cache_manifest=...)` | a **materialized** view's staleness sidecar |
| `io.check_view_fresh(save_path, view_config=..., cache_manifest=...)` | the load-side half of the above |
| `io.write_run_provenance(run_dir, script=..., params=vars(args), parents=..., subjects=...)` | an analysis run directory |
| `io.save_model(obj, path.joblib, params=..., parents=..., subjects=...)` / `io.load_model` | fitted GLMM / sklearn / FOOOF objects |
| `io.assert_fresh(target, ...)` / `io.check_stale(target, ...)` | staleness for anything the readers above don't cover |
| `io.parent_ref(path)` | a reference to an input artifact |
| `io.config_hash(cfg)` | the ONE sanctioned fingerprint (materialized-view dirs only) |
| `io.downcast_floats(df, config.CACHE_FLOAT_DTYPE)` | the cache's dtype standard, applied explicitly |
| `io.log_analysis(desc, run_dir)` | the terse run index |
| `io.warn_if_dirty()` | at script start |

Legacy, kept for the existing QC writers: `io.save_table` (extension-dispatched,
**no sidecar**), `io.append_table`, `io.reset_table`.

---

## 3. The sidecar envelope

One shape, three homes, so a single reader understands all of them:

| Sidecar | Written by | Describes |
|---|---|---|
| `<artifact>.provenance.json` | `write_table`, `save_model`, `write_sidecar` | one file |
| `<dir>/manifest.json` | `write_manifest` | a cache / base-unit directory |
| `<run_dir>/provenance.json` | `write_run_provenance` | one analysis run |

```json
{
  "schema_version": 1,
  "kind": "table",
  "artifact": "sub-085_ses-01_epochs.parquet",
  "created": "2026-07-27T16:04:11-07:00",
  "script": "src/ieeg_ehr/features/build_pain_epoch_power.py",
  "git": {"available": true, "commit": "…", "dirty": false, "modified_files": []},
  "params": {"epoch_minutes": 5.0, "mask_label": "gross-std3_satmargin15_sw_logz4"},
  "config_hash": "3f9c1ab2d004",
  "parents": [
    {"path": "…/manifest.json", "kind": "manifest", "exists": true,
     "bytes": 1204, "mtime": "2026-07-27T15:58:02-07:00", "digest": "9a1c…",
     "provenance": {"kind": "manifest", "created": "…", "script": "…",
                    "config_hash": "…", "commit": "…"}}
  ],
  "subjects": ["085"]
}
```

- **`params` is the config that changes the output — and nothing else.** It is
  hashed into `config_hash`, which is what staleness compares. Do not put
  timestamps or per-array-task paths in it.
- **`subjects[]` is the only sanctioned answer to "which subjects were in this
  run?"** Never the folder name.
- **`extra=` merges at the top level**, for run-specific blocks (`counts`,
  `inputs`, `environment`).
- **`script` auto-detects the calling module** when you don't pass it, so the
  lazy call still records who wrote the file.
- **File sidecars APPEND the suffix** (`x.parquet.provenance.json`). Replacing it
  would collapse `x.parquet` and `x.csv` onto one name — the exact collision you
  hit while converting a table from one format to the other. Readers still find
  the pre-P0.3 replaced form (`x.provenance.json`), which is what the legacy pain
  caches under `outdated/` have on disk.

---

## 4. Staleness — what is compared, and how hard

`check_stale` compares only the sidecar against what the caller wants **now**; it
never reads the artifact, so it is cheap enough to call on every load. Reasons it
can return:

| Comparison | Triggered by |
|---|---|
| no sidecar | provenance unverifiable |
| `config_hash` differs | you asked for a different config than was saved |
| parent changed | digest (small files) or `(bytes, mtime)` differs |
| parent missing from the sidecar | you passed an input the artifact never saw |
| commit differs | only when `check_commit=True` |
| written from a dirty tree | only when `allow_dirty=False` |

`on_stale`: `'warn'` (default — keeps exploration moving), `'refuse'` (raises
`StaleArtifactError`; use for anything a reported number comes out of),
`'ignore'`.

**Parents are fingerprinted, not content-hashed.** A per-window cache file is
hundreds of MB to GB; sha256-ing it on every write and again on every check would
cost more than the recompute the check exists to avoid. So a parent reference is
`(path, bytes, mtime)`, plus a real digest only for small files. The one thing
always digested is a `manifest.json` — which is exactly why view staleness is
defined against **the cache manifest's hash** rather than against the cache data.
`io.file_digest` refuses files above 64 MB to keep that guarantee honest.

`check_commit` defaults differ on purpose: `True` for views and models (the code
*is* the numbers), `False` for `read_table` (a data table is not wrong just
because the repo moved on).

---

## 5. Where things go

Path builders live in `config/paths.py`; never hand-assemble one of these.

**The Phase-1 cache base unit** = one (epoch definition, QC mask) pair, because
those two are the only things baked into the cache and so the only two that force
a rebuild:

```
features/pain/psd_epochs/epoch-5min-pre_mask-<label>/     pain_epoch_unit_dir()
  manifest.json                                           pain_epoch_manifest_path()
  cache/       sub-XXX_ses-YY_epochs.parquet               pain_epoch_cache_path()
  epoch_defs/  sub-XXX_ses-YY_defs.parquet                 pain_epoch_defs_path()
  views/       <label>_<config_hash>/                      pain_epoch_views_dir()
```

Subject/session go in the FILENAME, not in rows; epochs stack inside one file
under `epoch_id`.

**Analysis runs** follow the 5-level scheme via
`analysis_run_dir(question, output_type, run_name, view_scheme=None)` and
`sweep_run_dir(run_name)`. Levels 1-2 (`<event>/<question>/`) are opened
deliberately — a named question that exists in the exploration log; levels 3-5
are created freely per run. Sweep combinatorics are ROWS in a `sweeps/`
`results.parquet`, never folders. A run directory always ends in a timestamp, so
two runs cannot overwrite each other's `provenance.json` (this has bitten the
project once).

**Fingerprints:** `config_hash` appears in exactly one place — a materialized
view's directory name. Runs and plots get a human label plus a timestamp.

---

## 6. dtype at the IO boundary

Settled by P0.6 (`config/cache_params.py`): tables in the cache are written at
`config.CACHE_FLOAT_DTYPE` (float32) — pass it as `write_table(...,
float_dtype=config.CACHE_FLOAT_DTYPE)`. `io.write_table` never downcasts on its
own, because silently halving the precision of an effect-size table should be
visible in the diff of the script that asked for it.

**Store narrow, compute wide.** float32 storage does not license float32
arithmetic: anything reducing over windows upcasts to
`config.CACHE_ACCUMULATE_DTYPE` first, and exponentiating log-power to linear
uses `config.CACHE_LINEAR_DOMAIN_DTYPE`. That is a rule on *views*, not on IO,
but this is where the narrow dtype enters the pipeline, so it is worth knowing
here. Full reasoning: `cache_params.py` and `DECISIONS.md`.

---

## 7. What stays CSV, and why

The raw-voltage QC tree (~85 subject-sessions of per-window metrics, exclusions,
masks) **stays CSV.** It has a working metric/threshold split and a large body of
on-disk artifacts; converting it would invalidate them for no analytical benefit.
`io.save_table` therefore still writes CSV for a `.csv` path, and
`io.append_table` is CSV-only by nature (Parquet has no meaningful
append-a-few-rows mode — a streaming metrics target needs one).

Everything new is Parquet through `io.write_table`. **New artifacts only — do not
bulk-convert.** When you next touch a script that writes an old CSV, that is the
moment to convert it *and* give it a sidecar.

---

## 8. Dependencies

Everything lives in the shared venv `$GROUP_HOME/venvs/ieeg_ehr_analysis`:

| Added | Packages |
|---|---|
| 2026-07-27 | **pyarrow 20.0.0, joblib 1.5.3** — the IO layer |
| 2026-08-05 | **nilearn 0.14.0, nibabel 5.4.2, scikit-learn 1.7.2, threadpoolctl 3.6.0, colorcet 3.2.1** — the electrode glass brain (`analysis/plot_electrode_locations.py`) |

**numpy 2.4.2 / scipy 1.16.3 / pandas 2.3.3 have never been touched**, and each
install verifies that explicitly before and after rather than assuming it.

Two things learned on the 2026-08-05 install, both worth knowing before the next
one. `nilearn` needs `scikit-learn`, which `--no-deps` correctly does not pull, so
it imports fine and then fails at `nilearn.plotting` — install it in the same pass.
And the glass-brain outlines ship inside nilearn as vector data, so `plot_markers`
renders with **no network**; that was verified deliberately, because a compute node
cannot fetch a template and a figure job that needs one fails only at the end.

Reinstalling elsewhere has one trap worth writing down. Sherlock is CentOS 7
(**glibc 2.17**) and modern pyarrow wheels are `manylinux_2_28`, so a plain
`pip install pyarrow` falls through to a source build and dies on a missing Rust
toolchain. `--only-binary=:all:` makes pip back off to the newest version that
still ships a `manylinux2014` wheel:

```bash
srun -p dev --time=00:20:00 --mem=8G bash -c '
  module load python/3.12
  source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
  export PIP_CACHE_DIR=$SCRATCH/.cache/pip
  pip install --no-deps --only-binary=:all: pyarrow joblib'
```

`--no-deps` because the venv sits under a 1.9 TB derivatives tree and an
unpinned resolve could upgrade numpy/pandas/pynwb underneath it. Never pip on the
login node. `io.tables` and `io.models` raise this exact recipe if the import
fails.

---

## 9. Worked example

A cache builder (the shape P1.1 takes):

```python
from ieeg_ehr import config, io

io.warn_if_dirty()                              # loud if the hash won't describe the run

unit = config.pain_epoch_unit_dir(mask_label=config.CANONICAL_MASK_LABEL)
params = {'epoch_minutes': config.EPOCH_MINUTES_BEFORE,
          'mask_label': config.CANONICAL_MASK_LABEL,
          'bin_edges': bin_edges, 'dtype': str(config.CACHE_FLOAT_DTYPE),
          'domain': 'log'}
io.write_manifest(unit, params=params, subjects=subjects)      # once per unit

for subject, session in subject_sessions:                      # one Slurm array task each
    io.write_table(defs_df, config.pain_epoch_defs_path(subject, session),
                   params=params, parents=[mask_path, scores_path])
    io.write_table(cache_df, config.pain_epoch_cache_path(subject, session),
                   params=params, float_dtype=config.CACHE_FLOAT_DTYPE,
                   parents=[io.manifest_ref(unit), psd_nwb_path])

io.log_analysis('pain epoch cache, discovery cohort', unit)     # one index line
```

A view stays a function and does **not** save (`architecture.md` PART 2). Only
when recompute is *measured* slow and something depends on it:

```python
if save_path:
    io.write_table(view_df, save_path, kind='view', params=view_config,
                   parents=[io.manifest_ref(unit)])
    io.write_view_sidecar(save_path, view_config=view_config, cache_manifest=unit)

# on load
io.check_view_fresh(save_path, view_config=view_config, cache_manifest=unit,
                    on_stale='refuse')
```
