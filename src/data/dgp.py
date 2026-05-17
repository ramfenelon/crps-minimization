"""
Data Generating Processes (DGPs) for distributional regression experiments.

Each DGP knows its own true conditional distribution P(Y | X),
which allows computation of the oracle CRPS — the theoretical
lower bound that no estimator can beat.
"""

import numpy as np
from scipy import stats


class LinearGaussianDGP:
    """
    Y | X ~ Normal(X @ beta, sigma^2)

    The simplest possible DGP: linear mean, constant variance.
    Used to verify that ERM recovers the true parameters as n grows.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    d : int
        Number of input features.
    sigma : float
        Standard deviation of the noise (constant across X).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n: int, d: int, sigma: float = 1.0, seed: int = 42):
        self.n = n
        self.d = d
        self.sigma = sigma
        self.seed = seed
        rng = np.random.default_rng(seed)
        # True coefficients — fixed for this DGP
        self.beta = rng.standard_normal(d)

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate n samples (X, Y) from the DGP.

        Returns
        -------
        X : ndarray of shape (n, d)
        Y : ndarray of shape (n,)
        """
        rng = np.random.default_rng(self.seed)
        X = rng.standard_normal((self.n, self.d))
        mu = X @ self.beta
        Y = mu + self.sigma * rng.standard_normal(self.n)
        return X, Y

    def true_mean(self, X: np.ndarray) -> np.ndarray:
        """Oracle conditional mean at each row of X."""
        return X @ self.beta

    def true_std(self, X: np.ndarray) -> np.ndarray:
        """Oracle conditional std at each row of X."""
        return np.full(X.shape[0], self.sigma)

    def oracle_crps(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Mean CRPS of the true distribution evaluated at observations Y.

        This is the lowest CRPS any estimator can achieve on this DGP.
        Used to measure excess risk: CRPS(estimator) - oracle_crps.

        Parameters
        ----------
        X : ndarray of shape (m, d)
        Y : ndarray of shape (m,)

        Returns
        -------
        float : mean oracle CRPS across all samples
        """
        mu = self.true_mean(X)
        sigma = self.true_std(X)
        # Closed-form CRPS for Gaussian:
        # CRPS(N(mu, sigma), y) = sigma * (z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi))
        # where z = (y - mu) / sigma
        z = (Y - mu) / sigma
        phi = stats.norm.pdf(z)
        Phi = stats.norm.cdf(z)
        crps_values = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
        return float(np.mean(crps_values))


class HeteroscedasticGaussianDGP:
    """
    Y | X ~ Normal(X @ beta, sigma(X)^2)

    The variance grows with the first feature: sigma(X) = exp(0.5 * X[:,0]).
    This tests whether your estimator can learn a non-constant variance.

    Parameters
    ----------
    n : int
        Number of samples.
    d : int
        Number of features.
    seed : int
        Random seed.
    """

    def __init__(self, n: int, d: int, seed: int = 42):
        self.n = n
        self.d = d
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.beta = rng.standard_normal(d)

    def _sigma(self, X: np.ndarray) -> np.ndarray:
        """Conditional std as a function of X[:,0]."""
        return np.exp(0.5 * X[:, 0])

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        X = rng.standard_normal((self.n, self.d))
        mu = X @ self.beta
        sigma = self._sigma(X)
        Y = mu + sigma * rng.standard_normal(self.n)
        return X, Y

    def true_mean(self, X: np.ndarray) -> np.ndarray:
        return X @ self.beta

    def true_std(self, X: np.ndarray) -> np.ndarray:
        return self._sigma(X)

    def oracle_crps(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Closed-form Gaussian CRPS with sample-varying sigma."""
        mu = self.true_mean(X)
        sigma = self.true_std(X)
        z = (Y - mu) / sigma
        phi = stats.norm.pdf(z)
        Phi = stats.norm.cdf(z)
        crps_values = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
        return float(np.mean(crps_values))


class MixtureGaussianDGP:
    """
    Y | X ~ 0.5 * Normal(X @ beta1, sigma1^2) + 0.5 * Normal(X @ beta2, sigma2^2)

    A bimodal conditional distribution. Tests whether your estimator
    can capture multimodality — something a single Gaussian ERM cannot.

    Parameters
    ----------
    n : int
        Number of samples.
    d : int
        Number of features.
    sigma1, sigma2 : float
        Component standard deviations.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n: int,
        d: int,
        sigma1: float = 0.5,
        sigma2: float = 0.5,
        seed: int = 42,
    ):
        self.n = n
        self.d = d
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.beta1 = rng.standard_normal(d)
        self.beta2 = rng.standard_normal(d)

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        X = rng.standard_normal((self.n, self.d))
        # For each sample, pick component 1 or 2 with equal probability
        component = rng.integers(0, 2, size=self.n)
        mu1 = X @ self.beta1
        mu2 = X @ self.beta2
        Y = np.where(
            component == 0,
            mu1 + self.sigma1 * rng.standard_normal(self.n),
            mu2 + self.sigma2 * rng.standard_normal(self.n),
        )
        return X, Y

    def oracle_crps(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        CRPS for a Gaussian mixture via Monte Carlo integration.
        No closed form exists so we approximate with 2000 samples.
        """
        mu1 = X @ self.beta1
        mu2 = X @ self.beta2
        rng = np.random.default_rng(self.seed + 1)
        n_mc = 2000
        crps_values = np.zeros(len(Y))
        for i in range(len(Y)):
            # Draw samples from the true mixture at X[i]
            comp = rng.integers(0, 2, size=n_mc)
            draws = np.where(
                comp == 0,
                mu1[i] + self.sigma1 * rng.standard_normal(n_mc),
                mu2[i] + self.sigma2 * rng.standard_normal(n_mc),
            )
            # CRPS = E|Y - Z| - 0.5 * E|Z - Z'|
            draws2 = np.where(
                rng.integers(0, 2, size=n_mc) == 0,
                mu1[i] + self.sigma1 * rng.standard_normal(n_mc),
                mu2[i] + self.sigma2 * rng.standard_normal(n_mc),
            )
            crps_values[i] = np.mean(np.abs(Y[i] - draws)) - 0.5 * np.mean(
                np.abs(draws - draws2)
            )
        return float(np.mean(crps_values))


class StudentTDGP:
    """
    Y | X ~ mu(X) + sigma * t(nu)

    Heavy-tailed conditional distribution. Tests robustness of scoring
    rules and estimators to tail behaviour beyond Gaussian.

    Parameters
    ----------
    n : int
        Number of samples.
    d : int
        Number of features.
    nu : float
        Degrees of freedom. Lower = heavier tails. nu=30 is near-Gaussian.
    sigma : float
        Scale parameter.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n: int,
        d: int,
        nu: float = 4.0,
        sigma: float = 1.0,
        seed: int = 42,
    ):
        self.n = n
        self.d = d
        self.nu = nu
        self.sigma = sigma
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.beta = rng.standard_normal(d)

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        X = rng.standard_normal((self.n, self.d))
        mu = X @ self.beta
        # Draw from scaled t distribution
        t_draws = stats.t.rvs(
            df=self.nu, scale=self.sigma, size=self.n, random_state=self.seed
        )
        Y = mu + t_draws
        return X, Y

    def true_mean(self, X: np.ndarray) -> np.ndarray:
        return X @ self.beta

    def oracle_crps(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        CRPS for scaled t distribution via Monte Carlo integration.
        """
        mu = self.true_mean(X)
        rng = np.random.default_rng(self.seed + 1)
        n_mc = 2000
        crps_values = np.zeros(len(Y))
        for i in range(len(Y)):
            draws = mu[i] + stats.t.rvs(
                df=self.nu,
                scale=self.sigma,
                size=n_mc,
                random_state=int(rng.integers(0, 999999)),
            )
            draws2 = mu[i] + stats.t.rvs(
                df=self.nu,
                scale=self.sigma,
                size=n_mc,
                random_state=int(rng.integers(0, 999999)),
            )
            crps_values[i] = np.mean(np.abs(Y[i] - draws)) - 0.5 * np.mean(
                np.abs(draws - draws2)
            )
        return float(np.mean(crps_values))
