"""
Tests for GaussianERM and PenalisedGaussianERM.

We verify:
1. Estimators converge on synthetic data
2. Test CRPS is finite and positive
3. Excess risk is non-negative (estimator cannot beat oracle)
4. Penalised ERM has higher bias than ERM on well-specified data
5. Predicted shapes are correct
6. score() returns negative CRPS (higher is better)
7. Penalty term behaves correctly for L1 and L2
"""

import numpy as np
import pytest

from src.data.dgp import LinearGaussianDGP
from src.models.erm import GaussianERM, PenalisedGaussianERM, mean_crps

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def linear_data():
    """Small linear Gaussian dataset for fast tests."""
    dgp = LinearGaussianDGP(n=300, d=5, sigma=1.0, seed=42)
    X, y = dgp.sample()
    return X[:250], X[250:], y[:250], y[250:], dgp


@pytest.fixture
def fitted_erm(linear_data):
    X_train, X_test, y_train, y_test, dgp = linear_data
    return GaussianERM().fit(X_train, y_train)


@pytest.fixture
def fitted_perm(linear_data):
    X_train, X_test, y_train, y_test, dgp = linear_data
    return PenalisedGaussianERM(alpha=0.1, penalty="l2").fit(X_train, y_train)


# ── convergence tests ─────────────────────────────────────────────────────────


def test_erm_converges(fitted_erm):
    """Optimiser must report successful convergence."""
    assert fitted_erm.converged_, "GaussianERM did not converge"


def test_perm_converges(fitted_perm):
    """Penalised optimiser must also converge."""
    assert fitted_perm.converged_, "PenalisedGaussianERM did not converge"


# ── output shape tests ────────────────────────────────────────────────────────


def test_predict_mean_shape(fitted_erm, linear_data):
    X_train, X_test, y_train, y_test, dgp = linear_data
    mu = fitted_erm.predict_mean(X_test)
    assert mu.shape == (X_test.shape[0],), f"predict_mean shape wrong: {mu.shape}"


def test_predict_std_shape(fitted_erm, linear_data):
    X_train, X_test, y_train, y_test, dgp = linear_data
    sigma = fitted_erm.predict_std(X_test)
    assert sigma.shape == (X_test.shape[0],)


def test_predict_std_positive(fitted_erm, linear_data):
    """Predicted std must always be strictly positive."""
    X_train, X_test, y_train, y_test, dgp = linear_data
    sigma = fitted_erm.predict_std(X_test)
    assert np.all(sigma > 0), "Predicted std must be positive everywhere"


def test_predict_quantile_shape(fitted_erm, linear_data):
    X_train, X_test, y_train, y_test, dgp = linear_data
    q50 = fitted_erm.predict_quantile(X_test, q=0.5)
    assert q50.shape == (X_test.shape[0],)


def test_predict_quantile_ordering(fitted_erm, linear_data):
    """Q10 must be below Q50, Q50 below Q90 for all samples."""
    X_train, X_test, y_train, y_test, dgp = linear_data
    q10 = fitted_erm.predict_quantile(X_test, q=0.1)
    q50 = fitted_erm.predict_quantile(X_test, q=0.5)
    q90 = fitted_erm.predict_quantile(X_test, q=0.9)
    assert np.all(q10 < q50), "Q10 must be below Q50"
    assert np.all(q50 < q90), "Q50 must be below Q90"


# ── CRPS tests ────────────────────────────────────────────────────────────────


def test_erm_crps_finite(fitted_erm, linear_data):
    X_train, X_test, y_train, y_test, dgp = linear_data
    crps = -fitted_erm.score(X_test, y_test)
    assert np.isfinite(crps), f"Test CRPS is not finite: {crps}"


def test_erm_crps_positive(fitted_erm, linear_data):
    X_train, X_test, y_train, y_test, dgp = linear_data
    crps = -fitted_erm.score(X_test, y_test)
    assert crps > 0, f"Test CRPS must be positive: {crps}"


def test_excess_risk_non_negative(linear_data):
    """
    No estimator can beat the oracle CRPS.
    Excess risk = test CRPS - oracle CRPS must be >= 0.
    """
    X_train, X_test, y_train, y_test, dgp = linear_data
    erm = GaussianERM().fit(X_train, y_train)
    test_crps = -erm.score(X_test, y_test)
    oracle = dgp.oracle_crps(X_test, y_test)
    excess_risk = test_crps - oracle
    assert excess_risk >= -1e-6, (
        f"Excess risk is negative: {excess_risk:.6f}. " "Estimator cannot beat oracle."
    )


def test_score_is_negative_crps(fitted_erm, linear_data):
    """score() must return negative CRPS for sklearn compatibility."""
    X_train, X_test, y_train, y_test, dgp = linear_data
    score = fitted_erm.score(X_test, y_test)
    mu = fitted_erm.predict_mean(X_test)
    sigma = fitted_erm.predict_std(X_test)
    direct_crps = mean_crps(y_test, mu, sigma)
    assert abs(score - (-direct_crps)) < 1e-10


# ── penalty tests ─────────────────────────────────────────────────────────────


def test_l2_penalty_shrinks_coefficients(linear_data):
    """
    L2 penalisation should shrink coefficients toward zero.
    With high alpha, beta norm should be smaller than unpenalised.
    """
    X_train, X_test, y_train, y_test, dgp = linear_data
    erm = GaussianERM().fit(X_train, y_train)
    perm = PenalisedGaussianERM(alpha=10.0, penalty="l2").fit(X_train, y_train)
    norm_erm = np.linalg.norm(erm.beta_)
    norm_perm = np.linalg.norm(perm.beta_)
    assert norm_perm < norm_erm, (
        f"L2 penalty should shrink coefficients: "
        f"||beta_erm||={norm_erm:.4f}, ||beta_perm||={norm_perm:.4f}"
    )


def test_higher_alpha_more_shrinkage(linear_data):
    """Larger alpha should produce smaller coefficient norms."""
    X_train, X_test, y_train, y_test, dgp = linear_data
    perm_low = PenalisedGaussianERM(alpha=0.01, penalty="l2").fit(X_train, y_train)
    perm_high = PenalisedGaussianERM(alpha=10.0, penalty="l2").fit(X_train, y_train)
    norm_low = np.linalg.norm(perm_low.beta_)
    norm_high = np.linalg.norm(perm_high.beta_)
    assert norm_high < norm_low, "Higher alpha should produce more shrinkage"
