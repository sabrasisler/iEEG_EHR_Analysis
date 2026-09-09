"""Unit tests for the per-subject pain decoder.

The theme, as elsewhere in this repo: every failure guarded here would be SILENT.
A leaked scaler, a coverage gap counted as artifact, or a null distribution that
is not actually null all produce a perfectly plausible AUC.

THE TWO THAT MATTER MOST are the controls at the bottom: a pipeline that cannot
recover a planted signal is broken, and one that "finds" signal in pure noise is
worse than broken. Everything else is detail by comparison.
"""

import numpy as np
import pytest

from ieeg_ehr.decoding import arms, cascade, cv


# ---------------------------------------------------------------------------
# Cascade: the three kinds of missing must not be conflated
# ---------------------------------------------------------------------------

def test_transient_badness_drops_the_epoch_not_the_channels():
    """The 40%-of-channels-in-one-epoch case.

    Those channels are bad only HERE, so their per-channel fraction stays low and
    the channel rule must not fire; the epoch rule then sees a majority gone and
    drops the epoch. This is the case the cascade exists to get right.
    """
    bad = np.zeros((40, 10), dtype=bool)
    bad[7, :6] = True                       # 60% of channels, one epoch only
    out = cascade.apply_cascade(bad, min_epochs=30)
    assert out['keep_channels'].all()       # no channel is chronically bad
    assert not out['keep_epochs'][7]        # the epoch goes
    assert out['keep_epochs'].sum() == 39


def test_persistent_badness_drops_the_channel_not_the_epochs():
    bad = np.zeros((40, 10), dtype=bool)
    bad[:30, 3] = True                      # channel 3 bad in 75% of epochs
    out = cascade.apply_cascade(bad, min_epochs=30)
    assert not out['keep_channels'][3]
    assert out['keep_channels'].sum() == 9
    assert out['keep_epochs'].all()         # one dead channel condemns no epoch


def test_channels_are_judged_before_epochs():
    """Order matters: judged the other way round, the dead channel would push
    every epoch over the epoch threshold and delete the entire unit."""
    bad = np.zeros((40, 2), dtype=bool)
    bad[:, 0] = True                        # one of two channels always bad
    out = cascade.apply_cascade(bad, min_epochs=30)
    assert not out['keep_channels'][0]
    assert out['keep_epochs'].all()         # nothing lost to a persistent fault


def test_coverage_is_not_artifact():
    """A never-recorded channel must leave as COVERAGE and never count as bad.

    Counting these as artifact is the mistake that inflated sub-189's apparent
    channel loss by 60 channels on the first pass at this.
    """
    bad = np.zeros((40, 5), dtype=bool)
    recorded = np.ones((40, 5), dtype=bool)
    recorded[:20, 4] = False                # channel 4 absent from half the runs
    out = cascade.apply_cascade(bad, recorded=recorded, min_epochs=30)
    assert not out['keep_channels'][4]
    assert out['report']['n_channels_dropped_coverage'] == 1
    assert out['report']['n_channels_dropped_artifact'] == 0
    assert out['keep_epochs'].all()


def test_residual_cells_survive_both_rules_and_are_reported():
    """Scattered cells are left for in-fold imputation, NOT dropped -- purging
    0.8% of cells by deleting rows would cost ~24% of the observations."""
    rng = np.random.default_rng(0)
    bad = rng.random((60, 20)) < 0.01
    out = cascade.apply_cascade(bad, min_epochs=30)
    assert out['report']['residual_cells'] == int(out['residual'].sum())
    assert out['keep_epochs'].sum() >= 55


def test_too_few_epochs_refuses_rather_than_returning_a_tiny_matrix():
    bad = np.zeros((12, 5), dtype=bool)
    with pytest.raises(cascade.NoUsableDataError, match='survive the cascade'):
        cascade.apply_cascade(bad, min_epochs=30)


# ---------------------------------------------------------------------------
# Labels and folds
# ---------------------------------------------------------------------------

def test_fold_count_keeps_at_least_five_per_fold():
    assert cv.fold_count(30) == 5
    assert cv.fold_count(49) == 5
    assert cv.fold_count(50) == 10          # 50/10 = 5 exactly
    assert cv.fold_count(95) == 10
    for n in (30, 33, 48, 50, 95, 122):
        assert n // cv.fold_count(n) >= 5


def test_median_split_sends_ties_low():
    """Ties at the median are the ONLY source of imbalance in a median split, so
    where they go is a real choice and must be pinned."""
    y = np.array([0, 2, 2, 2, 5, 9], dtype=float)   # median 2
    labels = cv.make_labels(y, 'classification')
    assert labels.tolist() == [0, 0, 0, 0, 1, 1]


def test_regression_and_ordinal_keep_the_raw_score():
    y = np.array([0, 3, 7, 10], dtype=float)
    assert cv.make_labels(y, 'regression').tolist() == y.tolist()
    assert cv.make_labels(y, 'ordinal').tolist() == y.tolist()


def test_blocked_folds_are_contiguous_in_time():
    """The whole point of the blocked scheme: a fold is a stretch of the
    hospitalization, not a scatter of epochs whose neighbours are in training."""
    splitter = cv.outer_splitter('blocked', 5)
    for _, test in splitter.split(np.zeros((50, 3))):
        assert np.array_equal(test, np.arange(test[0], test[-1] + 1))


def test_permutation_p_is_never_zero():
    """Phipson & Smyth: a permutation p of 0 claims more than resampling can
    support, and invites exactly that overclaim downstream."""
    obs = [{'auc': 0.99} for _ in range(10)]
    null = [{'auc': 0.5} for _ in range(20)]
    out = cv.summarize(obs, null, 'auc')
    assert out['auc_p_perm'] > 0
    assert out['auc_p_perm'] == pytest.approx(1 / 21)


# ---------------------------------------------------------------------------
# Frank-Hall ordinal
# ---------------------------------------------------------------------------

def test_frank_hall_expected_rank_tracks_the_ordinal_target():
    rng = np.random.default_rng(0)
    n = 80
    y = rng.integers(0, 5, size=n).astype(float)
    X = np.column_stack([y + rng.normal(0, 0.3, n), rng.normal(0, 1, (n, 4)).T[0]])
    model = arms.FrankHallOrdinal(alpha=0.01, l1_ratio=0.5).fit(X, y)
    pred = model.predict(X)
    assert len(model.thresholds_) == 4                    # levels 0..4 -> 4 splits
    from scipy import stats as st
    assert st.spearmanr(y, pred)[0] > 0.8


def test_frank_hall_skips_degenerate_thresholds():
    """A level with one observation cannot support a binary model; fitting it
    anyway would produce a 'classifier' that predicts a constant."""
    y = np.array([0.0] * 20 + [1.0] * 20 + [9.0])         # level 9 has n=1
    X = np.random.default_rng(1).normal(size=(41, 5))
    model = arms.FrankHallOrdinal(alpha=0.1, l1_ratio=0.5).fit(X, y)
    assert 9.0 not in model.thresholds_
    assert model.thresholds_ == [0.0]


def test_frank_hall_refuses_a_single_level():
    X = np.zeros((10, 3))
    with pytest.raises(arms.NotFittableError):
        arms.FrankHallOrdinal(alpha=0.1, l1_ratio=0.5).fit(X, np.ones(10))


def test_frank_hall_predicts_on_the_score_scale_not_the_rank_scale():
    """Without the rank -> score mapping, predict() returns 'number of thresholds
    exceeded' (0..m) while y is a 0-10 pain score. MAE and R^2 would then compare
    two different scales -- and MAE is what the inner grid search optimizes, so
    the penalty would be selected against a meaningless quantity.
    """
    rng = np.random.default_rng(2)
    n = 80
    y = rng.choice([0.0, 4.0, 8.0], size=n)          # scores far from ranks 0,1,2
    X = np.column_stack([y + rng.normal(0, 0.5, n), rng.normal(0, 1, n)])
    model = arms.FrankHallOrdinal(alpha=0.01, l1_ratio=0.5).fit(X, y)
    pred = model.predict(X)
    assert pred.min() >= 0.0 and pred.max() <= 8.0   # inside the SCORE range
    assert pred.max() > 2.5                          # not stuck on the rank scale


def test_alpha_C_round_trip_is_one_definition():
    """The grid and the index-model refit must agree on the conversion. When
    these were written separately they did not, and the index model was fitted at
    n times the intended penalty -- silent, and plausible-looking."""
    for alpha in (1e-4, 0.01, 1.0, 10.0):
        for n in (30, 48, 122):
            assert arms.C_to_alpha(arms.alpha_to_C(alpha, n), n) == pytest.approx(alpha)


def test_selected_hyperparams_reports_alpha_on_one_scale_for_every_arm():
    """A classification alpha read back must be comparable to a regression one,
    otherwise the per-bootstrap penalty column mixes two scales."""
    rng = np.random.default_rng(4)
    X, y = rng.normal(size=(40, 6)), (rng.normal(size=40) > 0).astype(float)

    pipe = arms.make_pipeline(
        arms.pinned_estimator('classification', 0.05, 0.5, n_samples=40))
    pipe.fit(X, y)
    alpha, ratio = arms.selected_hyperparams(pipe, n_samples=40)
    assert alpha == pytest.approx(0.05)
    assert ratio == 0.5

    reg = arms.make_pipeline(arms.pinned_estimator('regression', 0.05, 0.5, 40))
    reg.fit(X, rng.normal(size=40))
    assert arms.selected_hyperparams(reg, n_samples=40)[0] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# The pipeline fits everything in-fold
# ---------------------------------------------------------------------------

def test_pipeline_imputes_and_scales_inside_itself():
    """Both steps are IN the pipeline, so sklearn refits them per fold. If either
    ever moved outside, the test fold would help set its own scaling."""
    pipe = arms.make_pipeline(arms.base_estimator('regression'))
    assert list(pipe.named_steps) == ['impute', 'scale', 'model']

    X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    pipe.fit(X, np.array([0.0, 1.0, 2.0]))
    # The imputer learned column 1's median from the TRAINING rows only.
    assert pipe.named_steps['impute'].statistics_[1] == pytest.approx(5.0)


def test_inner_search_lives_inside_the_estimator():
    """The nesting is what makes the score honest, and it now depends on the
    final step being a *CV estimator rather than on a surrounding GridSearchCV.
    If someone swaps in a plain ElasticNet, the penalty stops being tuned at all
    and nothing else would complain."""
    from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
    assert isinstance(arms.base_estimator('regression'), ElasticNetCV)
    assert isinstance(arms.base_estimator('classification'), LogisticRegressionCV)
    assert isinstance(arms.base_estimator('ordinal'), arms.FrankHallOrdinal)

    # ...and the index-model refit must NOT re-tune: it sees every epoch, so an
    # inner CV there would just re-select on the training set.
    assert not isinstance(arms.pinned_estimator('regression', 0.1, 0.5, 40),
                          ElasticNetCV)


def test_selected_alpha_comes_from_the_choice_not_the_default():
    """A fitted ElasticNetCV keeps its CHOICE in `alpha_`; reading `alpha` would
    return the constructor default (None) and the recorded penalty would be
    silently empty for every bootstrap."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 12))
    y = X[:, 0] * 2 + rng.normal(0, 0.2, 60)
    pipe = arms.make_pipeline(arms.base_estimator('regression', inner_cv=3))
    pipe.fit(X, y)
    alpha, ratio = arms.selected_hyperparams(pipe, n_samples=60)
    assert alpha is not None and np.isfinite(alpha) and alpha > 0
    assert ratio in arms.L1_RATIO_GRID
    assert alpha in pipe.named_steps['model'].alphas_.ravel()


# ---------------------------------------------------------------------------
# THE CONTROLS -- the gate. Nothing real runs until these pass.
# ---------------------------------------------------------------------------

def _planted(n=60, p=25, strength=2.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = strength * X[:, 3] - strength * X[:, 7] + rng.normal(0, 0.5, n)
    return X, y


@pytest.mark.parametrize('scheme', ['random', 'blocked'])
def test_positive_control_recovers_a_planted_signal(scheme):
    """A decoder that cannot find a signal that IS there is broken."""
    X, y = _planted()
    rows = cv.run_bootstraps('regression', X, y, scheme, n_bootstraps=3,
                             progress_every=0)
    r = np.mean([row['pearson_r'] for row in rows])
    assert r > 0.7, f'planted signal not recovered under {scheme}: r={r:.3f}'


def test_negative_control_finds_nothing_in_noise():
    """A decoder that 'finds' signal in pure noise is worse than broken.

    NOTE WHAT THIS DOES *NOT* ASSERT. An earlier version required
    |r| < 0.35 and failed at r = -0.3501 -- not because the pipeline was wrong,
    but because the assertion was. Cross-validated Pearson r on pure noise is NOT
    centred on zero: it is biased NEGATIVE, because a model that overfits each
    training fold produces out-of-fold predictions that actively anti-correlate
    with the truth. Chance is only 0 for an in-sample fit.

    So the control checks the two things the method actually promises: no
    spurious POSITIVE signal, and a permutation p that is not significant. The
    permutation null is the reference precisely because the point estimate's null
    location is not 0 and depends on n, p and the penalty.
    """
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 25))
    y = rng.normal(size=60)

    obs = cv.run_bootstraps('regression', X, y, 'random', n_bootstraps=5,
                            base_seed=0, progress_every=0)
    null = cv.run_bootstraps('regression', X, y, 'random', n_bootstraps=5,
                             shuffle_labels=True, base_seed=999, progress_every=0)
    summary = cv.summarize(obs, null, 'pearson_r')

    assert summary['pearson_r'] < 0.2, 'spurious positive signal found in noise'
    assert summary['pearson_r_p_perm'] > 0.05, 'noise called significant'


def test_the_null_is_centred_where_the_observed_noise_estimate_is():
    """The corollary of the test above, and the reason the null is trustworthy:
    the shuffled-label null lands in the SAME place as an honest fit to noise.

    If the null sat at 0 while noise fits sat at -0.35, every real p-value would
    be systematically wrong in the optimistic direction.
    """
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 25))
    y = rng.normal(size=60)

    obs = cv.run_bootstraps('regression', X, y, 'random', n_bootstraps=4,
                            base_seed=0, progress_every=0)
    null = cv.run_bootstraps('regression', X, y, 'random', n_bootstraps=4,
                             shuffle_labels=True, base_seed=500, progress_every=0)
    obs_r = np.mean([r['pearson_r'] for r in obs])
    null_r = np.mean([r['pearson_r'] for r in null])
    assert abs(obs_r - null_r) < 0.3, (
        f'null ({null_r:.3f}) is not where honest noise lands ({obs_r:.3f})')


def test_classification_null_sits_at_chance():
    """The shuffled-label null for a balanced binary problem must centre on
    AUC 0.5. If it does not, the null is not null and every p is wrong."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 20))
    y = np.array([0] * 30 + [1] * 30)
    rng.shuffle(y)

    null = cv.run_bootstraps('classification', X, y.astype(float), 'random',
                             n_bootstraps=4, shuffle_labels=True, base_seed=7,
                             progress_every=0)
    aucs = [r['auc'] for r in null if np.isfinite(r['auc'])]
    assert aucs, 'no null bootstrap produced a finite AUC'
    assert abs(np.mean(aucs) - 0.5) < 0.15


def test_positive_control_classification():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(60, 20))
    y = (X[:, 2] + rng.normal(0, 0.4, 60) > 0).astype(float)
    rows = cv.run_bootstraps('classification', X, y, 'random', n_bootstraps=3,
                             progress_every=0)
    assert np.mean([r['auc'] for r in rows]) > 0.75


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
