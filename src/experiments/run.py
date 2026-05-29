"""
Experiment runner for CRPS minimisation experiments.

Loads config via Hydra, runs the experiment, logs everything to MLflow.

Usage (from repo root in Git Bash):
    python -m src.experiments.run \
        --config-path ../../configs \
        --config-name default \
        +data=linear_gaussian \
        +model=gaussian_erm
"""

import subprocess
import time

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig

from src.data.dgp import (
    HeteroscedasticGaussianDGP,
    LinearGaussianDGP,
    MixtureGaussianDGP,
    StudentTDGP,
)
from src.models.erm import GaussianERM, PenalisedGaussianERM


def get_dgp(cfg: DictConfig):
    """Instantiate the DGP from config."""
    name = cfg.data.name
    if name == "linear_gaussian":
        return LinearGaussianDGP(
            n=cfg.data.n, d=cfg.data.d, sigma=cfg.data.sigma, seed=cfg.data.seed
        )
    elif name == "heteroscedastic_gaussian":
        return HeteroscedasticGaussianDGP(
            n=cfg.data.n, d=cfg.data.d, seed=cfg.data.seed
        )
    elif name == "mixture_gaussian":
        return MixtureGaussianDGP(n=cfg.data.n, d=cfg.data.d, seed=cfg.data.seed)
    elif name == "student_t":
        return StudentTDGP(n=cfg.data.n, d=cfg.data.d, seed=cfg.data.seed)
    else:
        raise ValueError(f"Unknown DGP: {name}")


def get_model(cfg: DictConfig):
    """Instantiate the model from config."""
    name = cfg.model.name
    if name == "gaussian_erm":
        return GaussianERM(
            fit_intercept=cfg.model.fit_intercept,
            max_iter=cfg.model.max_iter,
            tol=cfg.model.tol,
        )
    elif name == "penalised_gaussian_erm":
        return PenalisedGaussianERM(
            alpha=cfg.model.alpha,
            penalty=cfg.model.penalty,
            fit_intercept=cfg.model.fit_intercept,
            max_iter=cfg.model.max_iter,
            tol=cfg.model.tol,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def get_git_sha() -> str:
    """Return current Git commit hash for run traceability."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


@hydra.main(config_path="../../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    # Clear any run ID injected by Azure ML to avoid conflicts
    import os

    os.environ.pop("MLFLOW_RUN_ID", None)

    # Set up MLflow
    # Use Azure ML tracking URI if injected, otherwise use config
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        # Log git SHA for traceability
        mlflow.set_tag("git_sha", get_git_sha())
        mlflow.set_tag("dgp", cfg.data.name)
        mlflow.set_tag("model", cfg.model.name)

        # Log all config parameters
        mlflow.log_param("n", cfg.data.n)
        mlflow.log_param("d", cfg.data.d)
        mlflow.log_param("seed", cfg.data.seed)
        mlflow.log_param("model_name", cfg.model.name)
        mlflow.log_param("test_fraction", cfg.experiment.test_fraction)

        if hasattr(cfg.model, "alpha"):
            mlflow.log_param("alpha", cfg.model.alpha)
            mlflow.log_param("penalty", cfg.model.penalty)

        # Run multiple trials with different seeds
        test_crps_list = []
        excess_risk_list = []

        for trial in range(cfg.experiment.n_trials):
            trial_seed = cfg.experiment.seed + trial

            # Generate data
            dgp = get_dgp(cfg)
            dgp.seed = trial_seed
            X, y = dgp.sample()

            # Train/test split
            n_test = int(len(y) * cfg.experiment.test_fraction)
            X_train, X_test = X[:-n_test], X[-n_test:]
            y_train, y_test = y[:-n_test], y[-n_test:]

            # Fit model
            model = get_model(cfg)
            start = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - start

            # Evaluate
            test_crps = -model.score(X_test, y_test)
            oracle_crps = dgp.oracle_crps(X_test, y_test)
            excess_risk = test_crps - oracle_crps

            test_crps_list.append(test_crps)
            excess_risk_list.append(excess_risk)

            # Log per-trial metrics
            mlflow.log_metric("test_crps", test_crps, step=trial)
            mlflow.log_metric("oracle_crps", oracle_crps, step=trial)
            mlflow.log_metric("excess_risk", excess_risk, step=trial)
            mlflow.log_metric("fit_time", fit_time, step=trial)

        # Log summary metrics
        mlflow.log_metric("mean_test_crps", float(np.mean(test_crps_list)))
        mlflow.log_metric("std_test_crps", float(np.std(test_crps_list)))
        mlflow.log_metric("mean_excess_risk", float(np.mean(excess_risk_list)))
        mlflow.log_metric("std_excess_risk", float(np.std(excess_risk_list)))

        print(f"DGP:              {cfg.data.name}")
        print(f"Model:            {cfg.model.name}")
        print(f"Mean test CRPS:   {np.mean(test_crps_list):.4f}")
        print(f"Mean excess risk: {np.mean(excess_risk_list):.4f}")
        print(f"Converged:        {model.converged_}")


if __name__ == "__main__":
    main()
