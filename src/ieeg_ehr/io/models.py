"""
Fitted-object IO: joblib, never raw pickle, always with a sidecar.

Scope is deliberately narrow — this is for objects that are *fits*, not data:
GLMM/sklearn models, FOOOF/specparam fit objects, fitted scalers. Tabular data
goes through `io.tables` (Parquet); a table saved as joblib is a bug, because it
becomes unreadable the moment pandas changes its internals.

joblib rather than raw pickle for the array handling (it stores large numpy
arrays efficiently instead of inlining them into the pickle stream), but the
version-fragility caveat still applies: a joblib file is only guaranteed
loadable by the same library versions that wrote it. That is why the sidecar
records the environment — an unloadable model with a sidecar tells you what to
recreate; one without leaves you guessing.
"""

from pathlib import Path

from ieeg_ehr.io.sidecar import assert_fresh, write_sidecar

MODEL_SUFFIX = '.joblib'


def _require_joblib():
    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            'joblib is required to save/load fitted models. Install it into the shared '
            'venv from a compute node (never the login node):\n'
            '  srun -p dev --time=00:20:00 --mem=8G bash -c \'module load python/3.12 && '
            'source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate && '
            'pip install --no-deps --only-binary=:all: pyarrow joblib\''
        ) from exc
    return joblib


def _environment():
    """Library versions a joblib file's loadability depends on. Recorded in the
    sidecar so a load failure years later is diagnosable rather than mysterious."""
    env = {}
    import sys
    env['python'] = sys.version.split()[0]
    for name in ('numpy', 'pandas', 'scipy', 'sklearn', 'statsmodels', 'joblib'):
        try:
            module = __import__(name)
            env[name] = getattr(module, '__version__', None)
        except ImportError:
            continue
    return env


def save_model(obj, path, *, params=None, parents=None, subjects=None,
               script=None, extra=None, compress=3):
    """Persist a fitted object with joblib and write its provenance sidecar.

    Args:
        path: must end in `.joblib`.
        params: the model spec / fit config — hashed into the sidecar so a later
            load can tell whether it matches the spec being asked for.
        parents: the tables (or view/cache manifest) the fit consumed.
        subjects: the resolved cohort the model was fit on.
        compress: joblib compression level; 3 is a good size/speed tradeoff and
            matters on Oak.

    Returns the artifact path.
    """
    joblib = _require_joblib()
    path = Path(path)
    if path.suffix != MODEL_SUFFIX:
        raise ValueError(f'fitted objects are saved as {MODEL_SUFFIX}, got {path.name!r}. '
                         f'Tabular data belongs in Parquet via io.tables.write_table.')
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=compress)

    sidecar_extra = {'environment': _environment()}
    if extra:
        sidecar_extra.update(extra)
    write_sidecar(path, kind='model', script=script, params=params,
                  parents=parents, subjects=subjects, extra=sidecar_extra)
    return path


def load_model(path, *, parents=None, config=None, check_commit=True,
               on_stale='warn'):
    """Load a fitted object, checking its sidecar first.

    `check_commit=True` by default here (unlike `read_table`): a model IS the
    code that fit it, so a commit change is genuinely a reason to look before
    trusting the object.
    """
    joblib = _require_joblib()
    path = Path(path)
    assert_fresh(path, parents=parents, config=config,
                 check_commit=check_commit, on_stale=on_stale)
    return joblib.load(path)
