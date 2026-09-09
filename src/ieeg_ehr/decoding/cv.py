"""Nested cross-validation, bootstrapping, and the shuffled-label null.

THE NESTING, AND WHY IT IS THE WHOLE POINT
------------------------------------------
Outer loop holds out a fold. Inside the remaining k-1 folds, an inner CV searches
the penalty grid, picks a winner, refits on the whole outer-training set, and
only then predicts the held-out fold. The test fold never influences the penalty,
which is what makes the reported score an estimate of generalization rather than
of how well the grid was searched. Tuning on all the data and reporting the
best inner score is the single most common way this analysis is done wrong.

TWO CV SCHEMES, AND WHY THE SECOND ONE EXISTS
---------------------------------------------
`random` is the paper's: KFold(shuffle=True). `blocked` is contiguous in time.

Random k-fold assumes epochs are exchangeable. They are not. Pain reports are
~2 h apart and the target paper's OWN timescale analysis shows classifier output
staying stable across 3 hours -- direct evidence of temporal autocorrelation.
Under random folds a test epoch's temporal neighbours sit in the training set, so
the model can partly recognize WHEN rather than HOW MUCH PAIN, and performance is
inflated. The paper used random folds and did not address this.

Running both costs one extra column and makes the gap between them legible. If a
result survives blocking it is materially stronger than the paper's; if it
collapses, that is worth knowing before anyone writes it down.

FOLD COUNT
----------
k = 10 when n >= 50, else 5. The paper's rule is "five or ten, ensuring that each
fold contained at least five observations"; since every eligible unit has >= 30
epochs, this satisfies it deterministically instead of by inspection.

CHANCE IS NOT ZERO -- READ THE PERMUTATION p, NOT THE POINT ESTIMATE
---------------------------------------------------------------------
Cross-validated Pearson r on pure noise is NOT centred on zero. It is biased
NEGATIVE: a model that overfits each training fold produces out-of-fold
predictions that actively anti-correlate with the truth. Measured here on
n=60, p=25 noise, it lands near r = -0.35 (tests/test_decoding.py).

Two consequences, both of which matter when reading real results:

  * A per-subject r of, say, +0.05 is ABOVE the noise expectation, not "at
    chance". Comparing to 0 understates the evidence.
  * How far below zero the null sits depends on n, p and the selected penalty,
    so it differs BETWEEN subjects and cannot be quoted as one number.

This is exactly why every reported effect travels with its own shuffled-label
null and a permutation p, and why the group summary counts units that beat their
OWN null rather than units whose r exceeds some fixed threshold.
"""

import logging
import warnings

import numpy as np
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             mean_absolute_error, r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold

from ieeg_ehr.decoding import arms as arms_mod

logger = logging.getLogger(__name__)

CV_SCHEMES = ('random', 'blocked')
N_BOOTSTRAPS = 100
MIN_PER_FOLD = 5


def fold_count(n, min_per_fold=MIN_PER_FOLD):
    """k = 10 if that still leaves >=5 per fold, else 5."""
    return 10 if n // 10 >= min_per_fold else 5


def make_labels(y, arm, median=None):
    """The target for one arm.

    regression / ordinal use the raw 0-10 score. classification uses a MEDIAN
    SPLIT with ties going low, which is the paper's primary binarization.

    `median` may be passed so every bootstrap of a unit splits at the same place
    -- recomputing it per fold would make the LABEL depend on the split, which is
    a different and much worse kind of leakage than a rescaled feature.
    """
    y = np.asarray(y, dtype=float)
    if arm in ('regression', 'ordinal'):
        return y
    if arm == 'classification':
        cut = np.median(y) if median is None else median
        return (y > cut).astype(int)
    raise ValueError(f'arm={arm!r} not one of {arms_mod.ARMS}')


def outer_splitter(scheme, n_splits, y=None, seed=0, stratify=False):
    """The outer CV splitter for one bootstrap.

    `blocked` is KFold(shuffle=False): sklearn's unshuffled folds ARE contiguous
    index ranges, and epochs are ordered by pain_time, so a fold is a contiguous
    stretch of the hospitalization. It takes no seed -- which is why a blocked
    bootstrap varies only through the model's own randomness, and its spread is
    correspondingly narrower. That is a property of the scheme, not a bug.
    """
    if scheme == 'blocked':
        return KFold(n_splits=n_splits, shuffle=False)
    if scheme != 'random':
        raise ValueError(f'scheme={scheme!r} not one of {CV_SCHEMES}')
    if stratify:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _score(arm, y_true, y_pred, y_prob=None):
    """Metrics appropriate to the arm. Different arms, different schemas."""
    out = {}
    if arm == 'classification':
        out['auc'] = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else np.nan
        out['accuracy'] = float(accuracy_score(y_true, y_pred))
        out['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
        return out
    # Pearson for comparability with the paper; Spearman because the pain scale
    # is ordinal and monotone-but-nonlinear prediction should not be penalized.
    if np.std(y_pred) == 0:                     # a fully-penalized model is constant
        out['pearson_r'] = out['spearman_rho'] = 0.0
    else:
        out['pearson_r'] = float(stats.pearsonr(y_true, y_pred)[0])
        out['spearman_rho'] = float(stats.spearmanr(y_true, y_pred)[0])
    out['r2'] = float(r2_score(y_true, y_pred))
    out['mae'] = float(mean_absolute_error(y_true, y_pred))
    return out


def _fit_one_split(arm, X, y, train, test, inner_k, seed):
    """Fit on `train`, predict `test`. Returns predictions and the chosen penalty.

    The inner penalty search happens INSIDE the estimator (ElasticNetCV /
    LogisticRegressionCV / the per-threshold CV in FrankHallOrdinal), so this is
    still a nested CV -- the estimator is fitted on X[train] only, and everything
    it estimates, including the penalty, is estimated from that. It is simply not
    wrapped in a GridSearchCV, which would refit from scratch at every alpha
    instead of walking the path with warm starts (see arms.py: the difference was
    measured at ~50x).
    """
    pipe = arms_mod.make_pipeline(
        arms_mod.base_estimator(arm, random_state=seed, inner_cv=inner_k))
    with warnings.catch_warnings():
        # saga on p >> n hits max_iter routinely; the penalty is what determines
        # the solution, so a loose optimum is expected rather than alarming.
        warnings.simplefilter('ignore', category=ConvergenceWarning)
        pipe.fit(X[train], y[train])
        pred = pipe.predict(X[test])
        prob = (pipe.predict_proba(X[test])[:, 1]
                if arm == 'classification' else None)
    alpha, l1_ratio = arms_mod.selected_hyperparams(pipe, n_samples=len(train))
    return pred, prob, pipe, alpha, l1_ratio


def run_one_bootstrap(arm, X, y, scheme, seed, inner_k=None):
    """One full outer CV pass. Returns out-of-fold predictions and metrics."""
    n = len(y)
    k = fold_count(n)
    inner_k = inner_k or arms_mod.INNER_CV

    stratify = arm == 'classification'
    splitter = outer_splitter(scheme, k, seed=seed, stratify=stratify)
    split_args = (X, y) if stratify else (X,)

    pred = np.full(n, np.nan)
    prob = np.full(n, np.nan)
    alphas, ratios = [], []
    for train, test in splitter.split(*split_args):
        p, pr, _, alpha, ratio = _fit_one_split(arm, X, y, train, test,
                                                inner_k, seed)
        pred[test] = p
        if pr is not None:
            prob[test] = pr
        alphas.append(alpha)
        ratios.append(ratio)

    metrics = _score(arm, y, pred, prob if arm == 'classification' else None)
    metrics.update({
        'k_outer': k, 'k_inner': inner_k,
        'alpha_median': float(np.nanmedian([a for a in alphas if a is not None]))
                        if any(a is not None for a in alphas) else np.nan,
        'l1_ratio_median': float(np.nanmedian([r for r in ratios if r is not None]))
                           if any(r is not None for r in ratios) else np.nan,
    })
    return {'predictions': pred, 'probabilities': prob, 'metrics': metrics}


def run_bootstraps(arm, X, y, scheme, n_bootstraps=N_BOOTSTRAPS, shuffle_labels=False,
                   base_seed=0, progress_every=25):
    """`n_bootstraps` outer CV passes, re-randomizing the fold assignment each time.

    This IS the paper's "random selection of cross-validation indices each time".

    With `shuffle_labels`, y is permuted ONCE PER BOOTSTRAP before the whole
    nested procedure runs -- not per fold. Permuting inside the CV would leave
    some label structure intact and give an optimistic null.
    """
    rng = np.random.default_rng(base_seed)
    rows = []
    for b in range(n_bootstraps):
        yb = rng.permutation(y) if shuffle_labels else y
        try:
            result = run_one_bootstrap(arm, X, yb, scheme, seed=base_seed + b)
        except (arms_mod.NotFittableError, ValueError) as exc:
            logger.warning('bootstrap %d (%s, shuffled=%s) failed: %s',
                           b, scheme, shuffle_labels, exc)
            continue
        row = dict(result['metrics'])
        row.update({'bootstrap': b, 'cv_scheme': scheme, 'arm': arm,
                    'shuffled': bool(shuffle_labels)})
        rows.append(row)
        if progress_every and (b + 1) % progress_every == 0:
            logger.info('  %s/%s%s: %d/%d bootstraps', arm, scheme,
                        ' [null]' if shuffle_labels else '', b + 1, n_bootstraps)
    if not rows:
        raise arms_mod.NotFittableError(
            f'every bootstrap failed for arm={arm} scheme={scheme}')
    return rows


def summarize(observed_rows, null_rows, metric):
    """Point estimate, bootstrap CI, null mean, and a permutation p-value.

    p = (1 + #{null >= observed}) / (1 + n_null) -- the +1 is Phipson & Smyth:
    a permutation p can never honestly be 0, and reporting one invites a claim
    the resampling cannot support.
    """
    obs = np.array([r[metric] for r in observed_rows if np.isfinite(r.get(metric, np.nan))])
    null = np.array([r[metric] for r in null_rows if np.isfinite(r.get(metric, np.nan))])
    if obs.size == 0:
        return {f'{metric}': np.nan}
    point = float(np.mean(obs))
    out = {
        metric: point,
        f'{metric}_ci_lo': float(np.percentile(obs, 2.5)),
        f'{metric}_ci_hi': float(np.percentile(obs, 97.5)),
        f'{metric}_n_bootstraps': int(obs.size),
    }
    if null.size:
        out[f'{metric}_null_mean'] = float(np.mean(null))
        out[f'{metric}_null_ci_hi'] = float(np.percentile(null, 97.5))
        out[f'{metric}_p_perm'] = float((1 + np.sum(null >= point)) / (1 + null.size))
        out[f'{metric}_n_null'] = int(null.size)
    return out


def primary_metric(arm):
    return {'regression': 'pearson_r', 'ordinal': 'spearman_rho',
            'classification': 'auc'}[arm]
