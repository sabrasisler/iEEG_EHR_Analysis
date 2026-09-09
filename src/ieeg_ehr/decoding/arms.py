"""The three model families, and the pipeline every one of them is wrapped in.

ONE PIPELINE SHAPE FOR ALL THREE:

    SimpleImputer(median) -> StandardScaler -> <penalized linear model>

The order matters and so does the fact that it is a `Pipeline`: sklearn refits
every step on the training split of each CV fold, so the imputer's medians, the
scaler's mean/SD and the penalty are all fitted WITHOUT the test fold. Leakage
would take deliberate effort rather than mere forgetfulness, which is the point
-- doing this by hand is where these analyses usually go wrong.

WHY EVERY ARM IS PENALIZED
--------------------------
~918 features against ~48 epochs. With p > n, ordinary least squares has no
unique solution at all (X'X is singular; infinitely many coefficient vectors fit
the training data exactly and every one generalizes terribly). The penalty is
what makes the problem well posed, not merely what improves it. Elastic net
rather than lasso because intracranial channels are heavily correlated and pure
L1 picks one of a correlated group arbitrarily -- the target paper says this is
exactly why it chose elastic net.

HOW THE REGULARIZATION IS CHOSEN -- AND WHY VIA A PATH, NOT A GRID SEARCH
-------------------------------------------------------------------------
`alpha` (strength) and `l1_ratio` (the L1/L2 mix) are selected by an inner CV
that lives INSIDE the final estimator (`ElasticNetCV` / `LogisticRegressionCV`).
sklearn refits the whole pipeline on each outer training fold, so the inner
search still never sees the outer test fold -- the nesting is intact -- but the
cost collapses.

This replaced a `GridSearchCV` over an explicit alpha grid, which was MEASURED at
3.6 MILLION fits per unit: 601 fits per outer fold x 5 folds x 100 bootstraps x
2 CV schemes x 2 (observed + null) x 3 arms, i.e. 5-20 hours PER UNIT and
225-900 hours across the cohort. A grid search refits from scratch at every
alpha; coordinate descent walks the entire regularization path with warm starts
for barely more than the cost of one fit. Same estimator, same nesting, same
answer, ~50x less compute.

The alpha path is DATA-SCALED either way: sklearn derives alpha_max (the smallest
penalty that zeros every coefficient) from the training fold and descends to
alpha_max * eps, so it adapts to the feature scale instead of being hand-picked.

With ~48 epochs the selected alpha is genuinely noisy across bootstraps; that is
why feature STABILITY across 100 runs is reported rather than one coefficient
vector.

CLASS IMBALANCE: `class_weight='balanced'`, NOT SMOTE
-----------------------------------------------------
A departure from the paper, taken deliberately. SMOTE interpolates between
nearest neighbours, and in ~918 dimensions with ~20 minority samples the nearest
neighbours are close to arbitrary, so the synthetic points are not obviously
meaningful. A median split is also near-balanced by construction -- imbalance
here comes from TIES AT THE MEDIAN (a subject whose median is 2 with 40% of
scores exactly 2), which reweighting handles directly. Achieved balance is
recorded per unit so the assumption stays checkable.
"""

import logging

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (ElasticNet, ElasticNetCV, LogisticRegression,
                                  LogisticRegressionCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ARMS = ('regression', 'ordinal', 'classification')

#: Folds for the penalty search INSIDE the estimator. 3, not 5: the inner split
#: is taken from an outer training fold of ~38 epochs, so 5 would leave ~7 per
#: inner fold and the selected alpha would be noise on top of noise.
INNER_CV = 3

#: Inner folds for the LOGISTIC arms. 2, not 3: saga pays per fold and the
#: penalty estimate is noise-dominated at this n regardless.
INNER_CV_LOGISTIC = 2

#: How far down the regularization path to walk, as a fraction of alpha_max.
ALPHA_PATH_EPS = 1e-3

#: saga on p >> n does not converge tightly and does not need to -- the penalty
#: is what determines the solution. Capped so a pathological fold cannot stall a
#: whole array task.
MAX_ITER = 400

#: The L1/L2 mix searched in the inner loop. Small on purpose: it is cheap at
#: this data size and it controls precisely the behaviour that makes per-subject
#: coefficients hard to compare -- near 1.0 the model picks one of a correlated
#: group and zeros its neighbours, lower values spread weight across the group.
L1_RATIO_GRID = (0.1, 0.5, 0.9)

#: The ratio grid the LOGISTIC arms search. Two points, not three: saga pays
#: full price per point and 0.1 vs 0.5 rarely separate at this n.
L1_RATIO_GRID_LOGISTIC = (0.5, 0.9)

#: Path resolution for the REGRESSION arm. Coordinate descent walks these with
#: warm starts, so a fine path is nearly free.
N_ALPHAS = 50

#: Path resolution for the LOGISTIC arms (classification, and the ordinal arm's
#: threshold models). Much coarser on purpose: saga does NOT warm-start across
#: the C grid the way coordinate descent does, so it pays full price per point --
#: and with ~48 observations the selected penalty is noisy enough that 50 points
#: buys precision the data cannot support. 10 log-spaced points still span the
#: whole path.
N_ALPHAS_LOGISTIC = 5

#: A Frank-Hall threshold needs at least this many observations on BOTH sides
#: before it is worth fitting. Below it the "model" is predicting a constant.
MIN_PER_THRESHOLD_CLASS = 3


class NotFittableError(RuntimeError):
    """This arm cannot be fitted on this unit's labels (e.g. one class only)."""


class FrankHallOrdinal(BaseEstimator, RegressorMixin):
    """Ordinal regression by binary decomposition (Frank & Hall).

    For ordered levels l_0 < l_1 < ... < l_K, fit one penalized logistic model per
    threshold for P(y > l_k), then predict the EXPECTED RANK as

        E[rank] = sum_k P(y > l_k)

    which is exact for integer ranks starting at zero. Using the sum rather than
    reconstructing per-level probabilities sidesteps the usual Frank-Hall wart:
    the threshold models are fitted independently, so their cumulative
    probabilities can be non-monotone and the implied P(y = l_k) can come out
    negative. The sum is well behaved regardless.

    Chosen over `mord` deliberately: no new dependency, and the penalty is
    IDENTICAL to the other two arms, so a difference between arms is a difference
    in link function rather than in how hard they were regularized.

    Degenerate thresholds (fewer than MIN_PER_THRESHOLD_CLASS on either side) are
    SKIPPED rather than fitted, and `thresholds_` records which survived -- with
    ~48 epochs a 0-10 score can easily have levels represented once.
    """

    def __init__(self, alpha=None, l1_ratio=None, cv=3, n_alphas=N_ALPHAS_LOGISTIC,
                 max_iter=MAX_ITER, random_state=None):
        self.alpha = alpha              # None -> tuned once and shared across thresholds
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.n_alphas = n_alphas
        self.max_iter = max_iter
        self.random_state = random_state

    def _common(self):
        return dict(penalty='elasticnet', solver='saga', max_iter=self.max_iter,
                    random_state=self.random_state, class_weight='balanced')

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y, dtype=float)
        levels = np.unique(y)
        if levels.size < 2:
            raise NotFittableError('ordinal arm needs at least two distinct levels')
        self.levels_ = levels

        usable = []
        for t in levels[:-1]:
            target = (y > t).astype(int)
            n_minor = min(int(target.sum()), int((1 - target).sum()))
            if n_minor >= MIN_PER_THRESHOLD_CLASS:
                usable.append((float(t), target, n_minor))
        if not usable:
            raise NotFittableError(
                f'no threshold of {levels.tolist()} has >={MIN_PER_THRESHOLD_CLASS} '
                'observations on both sides')

        # ONE PENALTY, SHARED ACROSS THRESHOLDS -- tuned on the most balanced
        # threshold and reused. This is a modelling choice before it is a speed
        # one: every threshold model sees the SAME feature matrix and a similar
        # n, so tuning each independently gives each its own noisy penalty
        # estimated from ~48 observations, and the resulting cumulative
        # probabilities are then regularized inconsistently. Sharing is more
        # stable, not less.
        #
        # It is also ~8x cheaper. Tuning per threshold was MEASURED at over six
        # minutes for a single bootstrap on a 95 x 972 matrix, i.e. days per unit.
        if self.alpha is not None:
            alpha, l1_ratio = float(self.alpha), float(self.l1_ratio or 0.5)
        else:
            # Most balanced = closest to a 50/50 split = the best-conditioned
            # threshold to estimate a penalty from.
            _, target, n_minor = max(usable, key=lambda u: u[2])
            # refit=True, though only C_ and l1_ratio_ are used. With
            # refit=False AND l1_ratios set, sklearn 1.7.2 raises
            # "only integer scalar arrays can be converted to a scalar index"
            # from _logistic.py:2188 -- it indexes the l1_ratios_ LIST with an
            # array. The extra refit is one fit and is not worth working around.
            tuner = LogisticRegressionCV(
                Cs=self.n_alphas, l1_ratios=list(L1_RATIO_GRID_LOGISTIC),
                cv=max(2, min(self.cv, n_minor)), scoring='roc_auc',
                refit=True, n_jobs=1, **self._common())
            tuner.fit(X, target)
            alpha = C_to_alpha(_scalar(tuner.C_), len(y))
            l1_ratio = _scalar(tuner.l1_ratio_) or 0.5

        self.alpha_, self.l1_ratio_ = alpha, l1_ratio
        self.thresholds_, self.models_ = [], []
        for t, target, _ in usable:
            model = LogisticRegression(C=alpha_to_C(alpha, len(y)),
                                       l1_ratio=l1_ratio, **self._common())
            model.fit(X, target)
            self.thresholds_.append(t)
            self.models_.append(model)

        if not self.models_:
            raise NotFittableError(
                f'no threshold of {levels.tolist()} has >={MIN_PER_THRESHOLD_CLASS} '
                'observations on both sides')

        # Map expected RANK back onto the SCORE scale. Without this, predict()
        # returns "number of thresholds exceeded" (0..m) while y is a 0-10 pain
        # score, so MAE and R^2 would silently compare two different scales --
        # and MAE is what the inner grid search optimizes, so the penalty would
        # be chosen against a meaningless quantity. Rank r means "above r
        # thresholds", so r=0 maps to the lowest level and r=k to threshold k.
        self.scale_ = np.asarray([levels[0]] + self.thresholds_, dtype=float)
        return self

    def selected_alpha(self, n_samples=None):
        """The one penalty shared by every threshold model."""
        return getattr(self, 'alpha_', None)

    def selected_l1_ratio(self):
        return getattr(self, 'l1_ratio_', None)

    def predict(self, X):
        X = np.asarray(X)
        probs = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])
        expected_rank = probs.sum(axis=1)
        return np.interp(expected_rank, np.arange(len(self.scale_)), self.scale_)


def make_pipeline(estimator):
    """impute -> scale -> model, as one refittable unit.

    `add_indicator=False`: a missingness indicator would add ~918 more columns to
    a matrix that is already p >> n, and residual missingness is 0.8% of cells,
    so there is little signal there to recover.
    """
    return Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=False)),
        ('scale', StandardScaler()),
        ('model', estimator),
    ])


def base_estimator(arm, random_state=None, inner_cv=INNER_CV, n_alphas=N_ALPHAS):
    """The unfitted model for one arm, with its OWN inner CV over the penalty.

    The inner search lives inside the estimator rather than in a surrounding
    GridSearchCV -- see the module docstring for why (it is the difference
    between 3.6M fits per unit and ~70k). The nesting is unchanged: sklearn
    refits the pipeline on each outer training fold, so this inner CV never sees
    the outer test fold.
    """
    if arm == 'regression':
        # `alphas=<int>` not `n_alphas=<int>`: the latter is deprecated in
        # sklearn 1.7 and removed in 1.9, and this venv is already on 1.7.2.
        return ElasticNetCV(l1_ratio=list(L1_RATIO_GRID), alphas=n_alphas,
                            eps=ALPHA_PATH_EPS, cv=inner_cv, max_iter=MAX_ITER,
                            n_jobs=1, random_state=random_state, selection='random')
    if arm == 'ordinal':
        return FrankHallOrdinal(cv=INNER_CV_LOGISTIC, n_alphas=N_ALPHAS_LOGISTIC,
                                random_state=random_state)
    if arm == 'classification':
        return LogisticRegressionCV(
            Cs=N_ALPHAS_LOGISTIC, l1_ratios=list(L1_RATIO_GRID_LOGISTIC),
            cv=INNER_CV_LOGISTIC,
            penalty='elasticnet', solver='saga', scoring='roc_auc',
            max_iter=MAX_ITER, random_state=random_state,
            class_weight='balanced', refit=True, n_jobs=1)
    raise ValueError(f'arm={arm!r} not one of {ARMS}')


def pinned_estimator(arm, alpha, l1_ratio, n_samples, random_state=None):
    """The same model family with the penalty FIXED -- for the index-model refit.

    Separate from base_estimator because the index model must not re-run an inner
    CV: it is fitted on every epoch, so there is no held-out data to tune against
    and doing so would just re-select on the training set.
    """
    if arm == 'regression':
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=MAX_ITER,
                          random_state=random_state, selection='random')
    if arm == 'ordinal':
        return FrankHallOrdinal(alpha=alpha, l1_ratio=l1_ratio,
                                random_state=random_state)
    if arm == 'classification':
        return LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=l1_ratio,
            C=alpha_to_C(alpha, n_samples), max_iter=MAX_ITER,
            random_state=random_state, class_weight='balanced')
    raise ValueError(f'arm={arm!r} not one of {ARMS}')


def alpha_to_C(alpha, n_samples):
    """sklearn's LogisticRegression takes C = 1 / (alpha * n).

    ONE definition of the conversion, used by both the grid and the index-model
    refit. When these were written separately they disagreed: the grid divided by
    n and the read-back did not, so the refit re-multiplied and the index model
    was fitted at n times the intended penalty -- a silent, plausible-looking
    wrong answer.
    """
    return 1.0 / max(float(alpha) * int(n_samples), 1e-12)


def C_to_alpha(C, n_samples):
    return 1.0 / max(float(C) * int(n_samples), 1e-12)


def selected_hyperparams(fitted_pipeline, n_samples):
    """(alpha, l1_ratio) out of a fitted pipeline, on ONE scale for every arm.

    `n_samples` is required rather than optional precisely because the
    classification arm's C depends on it; making it a parameter means the
    conversion cannot be forgotten at a call site.

    Recorded per bootstrap so the SPREAD of the selected penalty is visible
    rather than hidden behind a single number -- with ~5 observations per inner
    fold it is expected to be wide.
    """
    model = fitted_pipeline.named_steps['model']

    if isinstance(model, FrankHallOrdinal):
        return model.selected_alpha(n_samples), model.selected_l1_ratio()

    # A *CV estimator carries its CHOICE in a trailing-underscore attribute; the
    # plain estimator carries what it was told. Reading `alpha` off a fitted
    # ElasticNetCV would return None (the constructor default), not the choice.
    if hasattr(model, 'alpha_'):
        return float(model.alpha_), _scalar(getattr(model, 'l1_ratio_', None))
    if hasattr(model, 'C_'):
        return (C_to_alpha(_scalar(model.C_), n_samples),
                _scalar(getattr(model, 'l1_ratio_', None)))
    if getattr(model, 'alpha', None) is not None:
        return float(model.alpha), _scalar(getattr(model, 'l1_ratio', None))
    if getattr(model, 'C', None):
        return C_to_alpha(model.C, n_samples), _scalar(getattr(model, 'l1_ratio', None))
    return None, None


def _scalar(value):
    """LogisticRegressionCV reports per-class arrays even for binary problems."""
    if value is None:
        return None
    arr = np.atleast_1d(value)
    return float(arr.ravel()[0]) if arr.size else None


def coefficients(fitted_pipeline):
    """A flat coefficient vector, whichever arm it is.

    Frank-Hall has one vector PER THRESHOLD; they are averaged so a feature's
    importance is one number per unit per arm. That is a summary and is treated
    as one -- selection FREQUENCY across bootstraps is the more trustworthy
    statistic, for the collinearity reason in the module docstring.
    """
    model = fitted_pipeline.named_steps['model']
    if isinstance(model, FrankHallOrdinal):
        return np.mean([m.coef_.ravel() for m in model.models_], axis=0)
    coef = np.asarray(model.coef_)
    return coef.ravel() if coef.ndim == 1 else coef.mean(axis=0)


def clone_with(estimator, **params):
    est = clone(estimator)
    est.set_params(**params)
    return est
