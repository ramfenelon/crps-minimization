"""
Convergence figure: excess risk vs sample size n (log-log scale).

Queries MLflow for GaussianERM runs across different n values,
plots the empirical convergence rate alongside the theoretical n^{-1/2} bound.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd


def load_convergence_runs(
    experiment_name: str = "crps-minimization",
    dgp_name: str = "linear_gaussian",
    model_name: str = "gaussian_erm",
    tracking_uri: str = "mlruns",
) -> pd.DataFrame:
    """
    Load all runs for a given DGP and model across different n values.

    Returns
    -------
    pd.DataFrame with columns: n, mean_excess_risk, std_excess_risk
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(f"tags.dgp = '{dgp_name}' AND " f"tags.model = '{model_name}'"),
    )

    records = []
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        if "n" not in params:
            continue
        records.append(
            {
                "n": int(params["n"]),
                "mean_excess_risk": metrics.get("mean_excess_risk", np.nan),
                "std_excess_risk": metrics.get("std_excess_risk", np.nan),
            }
        )

    if not records:
        raise ValueError(f"No runs found for DGP '{dgp_name}', model '{model_name}'.")

    df = pd.DataFrame(records).groupby("n").mean().reset_index().sort_values("n")
    return df


def plot_convergence(
    df: pd.DataFrame,
    output_path: str = "reports/figures/convergence.pdf",
    dgp_name: str = "linear_gaussian",
    theoretical_rate: float = -0.5,
) -> None:
    """
    Plot excess risk vs n on log-log axes with theoretical rate overlay.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_convergence_runs().
    output_path : str
        Where to save the figure.
    dgp_name : str
        DGP name for the title.
    theoretical_rate : float
        Slope of the theoretical bound on log-log scale.
        -0.5 corresponds to O(n^{-1/2}) rate.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    n_vals = df["n"].values
    excess = df["mean_excess_risk"].values
    std = df["std_excess_risk"].values

    # Empirical curve
    ax.plot(
        n_vals,
        excess,
        marker="o",
        color="#1565C0",
        linewidth=2,
        markersize=6,
        label="GaussianERM (empirical)",
        zorder=3,
    )

    # Error band
    if not np.all(np.isnan(std)):
        ax.fill_between(
            n_vals,
            np.maximum(excess - std, excess * 0.1),
            excess + std,
            alpha=0.15,
            color="#1565C0",
            label="±1 std",
        )

    # Theoretical rate overlay: C * n^{rate}
    # Anchor the constant C to the first data point
    C = excess[0] / (n_vals[0] ** theoretical_rate)
    n_theory = np.logspace(np.log10(n_vals[0]), np.log10(n_vals[-1]), 100)
    theory_curve = C * n_theory**theoretical_rate

    ax.plot(
        n_theory,
        theory_curve,
        color="#E53935",
        linewidth=1.5,
        linestyle="--",
        label=f"Theoretical rate O(n^{{{theoretical_rate}}})",
        zorder=2,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample size n (log scale)", fontsize=10)
    ax.set_ylabel("Mean excess risk (log scale)", fontsize=10)
    ax.set_title(
        f"Convergence of excess risk — {dgp_name}\n"
        f"Empirical rate vs theoretical bound",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # Annotate empirical slope
    log_n = np.log10(n_vals)
    log_e = np.log10(np.maximum(excess, 1e-10))
    empirical_slope = np.polyfit(log_n, log_e, 1)[0]
    ax.text(
        0.97,
        0.95,
        f"Empirical slope: {empirical_slope:.2f}\n"
        f"Theoretical slope: {theoretical_rate:.2f}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot convergence figure.")
    parser.add_argument("--dgp", default="linear_gaussian")
    parser.add_argument("--model", default="gaussian_erm")
    parser.add_argument("--output", default="reports/figures/convergence.pdf")
    parser.add_argument("--theoretical-rate", type=float, default=-0.5)
    args = parser.parse_args()

    df = load_convergence_runs(dgp_name=args.dgp, model_name=args.model)
    print(df.to_string(index=False))
    plot_convergence(
        df,
        output_path=args.output,
        dgp_name=args.dgp,
        theoretical_rate=args.theoretical_rate,
    )


if __name__ == "__main__":
    main()
