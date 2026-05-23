"""
ERM and Penalised ERM estimators for distributional regression.

Both estimators minimise the empirical CRPS over a parametric
Gaussian family — i.e. they learn both the conditional mean
and conditional standard deviation of Y | X.

The scikit-learn BaseEstimator interface is used throughout
so these estimators work with cross_val_score, GridSearchCV,
and Pipeline out of the box.
"""

import numpy as np
from scipy import optimize, stats
from sklearn.base import BaseEstimator


def gaussian_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Closed-form CRPS for a Gaussian predictive distribution.

    CRPS(N(mu, sigma^2), y) =
        sigma * (z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi))
    where z = (y - mu) / sigma.

    Parameters
    ----------
    y     : ndarray of shape (n,) — observed values
    mu    : ndarray of shape (n,) — predicted means
    sigma : ndarray of shape (n,) — predicted standard deviations (> 0)

    Returns
    -------
    ndarray of shape (n,) — CRPS for each observation
    """
    z = (y - mu) / sigma
    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))


def mean_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Mean CRPS across all observations — the empirical risk."""
    return float(np.mean(gaussian_crps(y, mu, sigma)))


class GaussianERM(BaseEstimator):
    """
    ERM estimator for Gaussian distributional regression.

    Minimises the empirical CRPS over the family:
        P(Y | X) = Normal(X @ beta, exp(X @ gamma)^2)

    The mean is linear in X. The log-std is also linear in X,
    which ensures sigma > 0 everywhere (log link for scale).

    Parameters
    ----------
    fit_intercept : bool
        Whether to add a bias term to both mean and log-std.
    max_iter : int
        Maximum number of optimiser iterations.
    tol : float
        Convergence tolerance for the optimiser.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Prepend a column of ones if fit_intercept is True."""
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def _unpack(self, params: np.ndarray, p: int):
        """Split parameter vector into beta (mean) and gamma (log-std)."""
        beta = params[:p]
        gamma = params[p:]
        return beta, gamma

    def _objective(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Empirical CRPS — the objective function to minimise."""
        p = X.shape[1]
        beta, gamma = self._unpack(params, p)
        mu = X @ beta
        sigma = np.exp(X @ gamma)  # log link ensures sigma > 0
        return mean_crps(y, mu, sigma)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the ERM estimator by minimising empirical CRPS.

        Parameters
        ----------
        X : ndarray of shape (n, d)
        y : ndarray of shape (n,)

        Returns
        -------
        self
        """
        X = self._add_intercept(X)
        p = X.shape[1]

        # Initialise: mean coefficients near zero, log-std near log(std(y))
        init_gamma = np.zeros(p)
        init_gamma[0] = np.log(np.std(y) + 1e-6)
        params0 = np.concatenate([np.zeros(p), init_gamma])

        result = optimize.minimize(
            self._objective,
            params0,
            args=(X, y),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        self.beta_, self.gamma_ = self._unpack(result.x, p)
        self.converged_ = result.success
        self.n_iter_ = result.nit
        return self

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """Predicted conditional mean E[Y | X]."""
        X = self._add_intercept(X)
        return X @ self.beta_

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """Predicted conditional std sqrt(Var[Y | X])."""
        X = self._add_intercept(X)
        return np.exp(X @ self.gamma_)

    def predict_quantile(self, X: np.ndarray, q: float) -> np.ndarray:
        """
        Predicted conditional quantile at level q.

        Parameters
        ----------
        X : ndarray of shape (m, d)
        q : float in (0, 1)
        """
        mu = self.predict_mean(X)
        sigma = self.predict_std(X)
        return mu + sigma * stats.norm.ppf(q)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return negative mean CRPS (higher is better, consistent with sklearn).

        Parameters
        ----------
        X : ndarray of shape (n, d)
        y : ndarray of shape (n,)
        """
        mu = self.predict_mean(X)
        sigma = self.predict_std(X)
        return -mean_crps(y, mu, sigma)


class PenalisedGaussianERM(GaussianERM):
    """
    Penalised ERM estimator for Gaussian distributional regression.

    Adds an L2 (Ridge) or L1 (Lasso) penalty to the empirical CRPS:

        R_pen(beta, gamma) = CRPS_n(beta, gamma) + alpha * penalty(beta, gamma)

    The intercept terms are never penalised (standard practice).

    Parameters
    ----------
    alpha : float
        Regularisation strength. Larger = more regularisation.
    penalty : str
        'l2' for Ridge, 'l1' for Lasso.
    fit_intercept : bool
        Whether to add a bias term.
    max_iter : int
        Maximum optimiser iterations.
    tol : float
        Convergence tolerance.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        penalty: str = "l2",
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        super().__init__(fit_intercept=fit_intercept, max_iter=max_iter, tol=tol)
        self.alpha = alpha
        self.penalty = penalty

    def _penalty_term(self, params: np.ndarray, p: int) -> float:
        """
        Compute the penalty on beta and gamma, excluding intercepts.

        The intercept is always params[0] (beta) and params[p] (gamma)
        when fit_intercept=True — these are not penalised.
        """
        beta, gamma = self._unpack(params, p)

        # Exclude intercept from penalisation
        start = 1 if self.fit_intercept else 0
        beta_pen = beta[start:]
        gamma_pen = gamma[start:]
        coeffs = np.concatenate([beta_pen, gamma_pen])

        if self.penalty == "l2":
            return float(np.sum(coeffs**2))
        elif self.penalty == "l1":
            return float(np.sum(np.abs(coeffs)))
        else:
            raise ValueError(f"Unknown penalty: {self.penalty}. Use 'l1' or 'l2'.")

    def _objective(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Penalised empirical CRPS."""
        p = X.shape[1]
        crps = super()._objective(params, X, y)
        penalty = self._penalty_term(params, p)
        return crps + self.alpha * penalty
