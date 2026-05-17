"""
Tests for real dataset pipelines.

We verify:
1. Output shapes are consistent with train/test split
2. No data leakage — test set scaler is fitted on train only
3. Feature names are returned correctly
4. Targets are finite and positive (both datasets have positive targets)
"""

import numpy as np

from src.data.datasets import load_abalone, load_california_housing

# ── California Housing ────────────────────────────────────────────────────────


def test_california_shapes():
    """Train/test shapes must be consistent with test_size=0.2."""
    d = load_california_housing(test_size=0.2, seed=42)
    n_total = d["X_train"].shape[0] + d["X_test"].shape[0]
    assert d["X_train"].shape[1] == 8, "California Housing must have 8 features"
    assert d["X_test"].shape[1] == 8
    assert d["y_train"].shape[0] == d["X_train"].shape[0]
    assert d["y_test"].shape[0] == d["X_test"].shape[0]
    assert abs(d["X_test"].shape[0] / n_total - 0.2) < 0.01


def test_california_feature_names():
    """Feature names must be returned and have correct length."""
    d = load_california_housing()
    assert len(d["feature_names"]) == 8
    assert "MedInc" in d["feature_names"]


def test_california_no_leakage():
    """
    Scaler is fitted on train only.
    Train mean should be near 0, test mean may differ.
    """
    d = load_california_housing(seed=42)
    train_means = np.abs(d["X_train"].mean(axis=0))
    assert np.all(
        train_means < 0.01
    ), "Train features should be zero-centred after StandardScaler"


def test_california_targets_finite():
    """All targets must be finite."""
    d = load_california_housing()
    assert np.all(np.isfinite(d["y_train"]))
    assert np.all(np.isfinite(d["y_test"]))


def test_california_deterministic():
    """Same seed must produce identical splits."""
    d1 = load_california_housing(seed=42)
    d2 = load_california_housing(seed=42)
    np.testing.assert_array_equal(d1["X_train"], d2["X_train"])
    np.testing.assert_array_equal(d1["y_train"], d2["y_train"])


# ── Abalone ───────────────────────────────────────────────────────────────────


def test_abalone_shapes():
    """Train/test shapes must be consistent with test_size=0.2."""
    d = load_abalone(test_size=0.2, seed=42)
    n_total = d["X_train"].shape[0] + d["X_test"].shape[0]
    assert d["X_train"].shape[1] == 10, "Abalone must have 10 features"
    assert d["X_test"].shape[1] == 10
    assert d["y_train"].shape[0] == d["X_train"].shape[0]
    assert d["y_test"].shape[0] == d["X_test"].shape[0]
    assert abs(d["X_test"].shape[0] / n_total - 0.2) < 0.01


def test_abalone_feature_names():
    """Feature names must include physical measurements and sex dummies."""
    d = load_abalone()
    assert len(d["feature_names"]) == 10
    assert "rings" not in d["feature_names"], "Target must not appear in features"
    assert "sex_M" in d["feature_names"]
    assert "sex_F" in d["feature_names"]


def test_abalone_targets_positive():
    """Ring counts must be positive integers."""
    d = load_abalone()
    assert np.all(d["y_train"] > 0), "Ring counts must be positive"
    assert np.all(d["y_test"] > 0)


def test_abalone_no_leakage():
    """Train features should be zero-centred after StandardScaler."""
    d = load_abalone(seed=42)
    train_means = np.abs(d["X_train"].mean(axis=0))
    assert np.all(
        train_means < 0.01
    ), "Train features should be zero-centred after StandardScaler"


def test_abalone_deterministic():
    """Same seed must produce identical splits."""
    d1 = load_abalone(seed=42)
    d2 = load_abalone(seed=42)
    np.testing.assert_array_equal(d1["X_train"], d2["X_train"])
    np.testing.assert_array_equal(d1["y_train"], d2["y_train"])
