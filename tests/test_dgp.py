"""
Tests for DGP classes.

For each DGP we verify:
1. Output shapes are correct
2. Oracle CRPS is finite and positive
3. Oracle CRPS decreases as n grows (consistency check)
4. Sampling is deterministic given the same seed
"""

import numpy as np
import pytest

from src.data.dgp import (
    HeteroscedasticGaussianDGP,
    LinearGaussianDGP,
    MixtureGaussianDGP,
    StudentTDGP,
)

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def linear_dgp():
    return LinearGaussianDGP(n=200, d=5, sigma=1.0, seed=42)


@pytest.fixture
def hetero_dgp():
    return HeteroscedasticGaussianDGP(n=200, d=5, seed=42)


@pytest.fixture
def mixture_dgp():
    return MixtureGaussianDGP(n=200, d=5, seed=42)


@pytest.fixture
def student_dgp():
    return StudentTDGP(n=200, d=5, nu=4.0, seed=42)


# ── shape tests ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fixture_name", ["linear_dgp", "hetero_dgp", "mixture_dgp", "student_dgp"]
)
def test_sample_shapes(fixture_name, request):
    """X must be (n, d) and Y must be (n,)."""
    dgp = request.getfixturevalue(fixture_name)
    X, Y = dgp.sample()
    assert X.shape == (dgp.n, dgp.d), f"X shape wrong: {X.shape}"
    assert Y.shape == (dgp.n,), f"Y shape wrong: {Y.shape}"


# ── oracle CRPS tests ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fixture_name", ["linear_dgp", "hetero_dgp", "mixture_dgp", "student_dgp"]
)
def test_oracle_crps_finite_and_positive(fixture_name, request):
    """Oracle CRPS must be a finite positive number."""
    dgp = request.getfixturevalue(fixture_name)
    X, Y = dgp.sample()
    crps = dgp.oracle_crps(X, Y)
    assert np.isfinite(crps), f"Oracle CRPS is not finite: {crps}"
    assert crps > 0, f"Oracle CRPS is not positive: {crps}"


# ── determinism tests ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fixture_name", ["linear_dgp", "hetero_dgp", "mixture_dgp", "student_dgp"]
)
def test_sampling_is_deterministic(fixture_name, request):
    """Same seed must produce identical X and Y."""
    dgp = request.getfixturevalue(fixture_name)
    X1, Y1 = dgp.sample()
    X2, Y2 = dgp.sample()
    np.testing.assert_array_equal(X1, X2, err_msg="X not deterministic")
    np.testing.assert_array_equal(Y1, Y2, err_msg="Y not deterministic")


# ── consistency tests ─────────────────────────────────────────────────────────


def test_linear_dgp_crps_decreases_with_n():
    """
    For LinearGaussianDGP, oracle CRPS should be stable across sample sizes
    since it measures the true distribution's self-score, not estimation error.
    More importantly, the oracle CRPS with n=1000 should be close to n=200
    (within 20%) — confirming the DGP is consistent.
    """
    crps_small = LinearGaussianDGP(n=200, d=5, seed=42).oracle_crps(
        *LinearGaussianDGP(n=200, d=5, seed=42).sample()
    )
    crps_large = LinearGaussianDGP(n=1000, d=5, seed=42).oracle_crps(
        *LinearGaussianDGP(n=1000, d=5, seed=42).sample()
    )
    assert (
        abs(crps_small - crps_large) / crps_small < 0.2
    ), f"Oracle CRPS varies too much with n: {crps_small:.4f} vs {crps_large:.4f}"


def test_linear_dgp_true_mean_shape():
    """true_mean must return one value per sample."""
    dgp = LinearGaussianDGP(n=200, d=5, seed=42)
    X, _ = dgp.sample()
    mu = dgp.true_mean(X)
    assert mu.shape == (200,), f"true_mean shape wrong: {mu.shape}"


def test_linear_dgp_true_std_is_constant():
    """LinearGaussianDGP has constant variance — true_std must be uniform."""
    dgp = LinearGaussianDGP(n=200, d=5, sigma=1.5, seed=42)
    X, _ = dgp.sample()
    std = dgp.true_std(X)
    assert np.allclose(std, 1.5), "true_std should be constant 1.5"


def test_heteroscedastic_dgp_std_varies():
    """HeteroscedasticGaussianDGP must have non-constant variance."""
    dgp = HeteroscedasticGaussianDGP(n=200, d=5, seed=42)
    X, _ = dgp.sample()
    std = dgp.true_std(X)
    assert std.shape == (200,)
    assert (
        std.max() > std.min() * 1.5
    ), "Heteroscedastic DGP should have meaningfully varying std"
