"""
Competitor models for benchmarking against GaussianERM.

All competitors implement the same interface:
    fit(X_train, y_train)
    predict_mean(X_test)
    predict_std(X_test)
    score(X_test, y_test)  → negative mean CRPS

This allows direct CRPS comparison across all models.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

from src.models.erm import mean_crps


class GaussianWrapper(BaseEstimator):
    """
    Wraps a point-prediction model into a distributional predictor
    by assuming constant residual variance estimated from training data.

    This is the simplest possible distributional extension —
    it captures the mean correctly but assumes homoscedasticity.

    Parameters
    ----------
    base_model : sklearn estimator
        Any regressor with fit() and predict().
    """

    def __init__(self, base_model):
        self.base_model = base_model

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.base_model.fit(X, y)
        # Estimate residual std from training data
        y_hat = self.base_model.predict(X)
        self.sigma_ = float(np.std(y - y_hat) + 1e-6)
        return self

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        return self.base_model.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.sigma_)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Negative mean CRPS — higher is better."""
        mu = self.predict_mean(X)
        sigma = self.predict_std(X)
        return -mean_crps(y, mu, sigma)


class NGBoostWrapper(BaseEstimator):
    """
    NGBoost distributional regressor wrapped with CRPS scoring.

    NGBoost natively predicts a full distribution — no wrapper needed
    for the distributional output, only for the scoring interface.
    """

    def __init__(self, n_estimators: int = 500, learning_rate: float = 0.01):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray):
        from ngboost import NGBRegressor
        from ngboost.distns import Normal

        self.model_ = NGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            Dist=Normal,
            verbose=False,
        )
        self.model_.fit(X, y)
        return self

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        dist = self.model_.pred_dist(X)
        return dist.params["scale"]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        mu = self.predict_mean(X)
        sigma = self.predict_std(X)
        return -mean_crps(y, mu, sigma)


def get_competitors() -> dict:
    """
    Return all competitor models as a dictionary.

    Returns
    -------
    dict mapping model name to unfitted estimator instance
    """
    return {
        "Ridge (wrapped)": GaussianWrapper(Ridge(alpha=1.0)),
        "KNN (wrapped)": GaussianWrapper(KNeighborsRegressor(n_neighbors=10)),
        "RandomForest (wrapped)": GaussianWrapper(
            RandomForestRegressor(n_estimators=100, random_state=42)
        ),
        "NGBoost": NGBoostWrapper(n_estimators=200),
    }
