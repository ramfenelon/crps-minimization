"""
Penalty path figure: excess risk vs regularisation parameter alpha.

Queries MLflow for all penalised ERM runs on a given DGP,
plots the U-shaped bias-variance tradeoff curve.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd


def load_penalty_path_runs(
    experiment_name: str = "crps-minimization",
    dgp_name: str = "linear_gaussian",
    tracking_uri: str = "mlruns",
    n: int = None,
) -> pd.DataFrame:
    """
    Load all penalised ERM runs for a given DGP from MLflow.

    Returns
    -------
    pd.DataFrame with columns: alpha, mean_excess_risk, std_excess_risk
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")

    n_filter = f" AND params.n = '{n}'" if n is not None else ""
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.dgp = '{dgp_name}' AND "
            f"tags.model = 'penalised_gaussian_erm'"
            f"{n_filter}"
        ),
    )

    records = []
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        if "alpha" not in params:
            continue
        records.append(
            {
                "alpha": float(params["alpha"]),
                "mean_excess_risk": metrics.get("mean_excess_risk", np.nan),
                "std_excess_risk": metrics.get("std_excess_risk", np.nan),
            }
        )

    if not records:
        raise ValueError(
            f"No penalised ERM runs found for DGP '{dgp_name}'. "
            "Run experiments first."
        )

    df = pd.DataFrame(records).sort_values("alpha").reset_index(drop=True)
    return df


def plot_penalty_path(
    df: pd.DataFrame,
    erm_excess_risk: float,
    output_path: str = "reports/figures/penalty_path.pdf",
    dgp_name: str = "linear_gaussian",
) -> None:
    """
    Plot excess risk vs alpha (log scale) with error bands.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_penalty_path_runs().
    erm_excess_risk : float
        Excess risk of the unpenalised ERM (horizontal reference line).
    output_path : str
        Where to save the figure.
    dgp_name : str
        DGP name for the title.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot penalised ERM curve with error band
    ax.plot(
        df["alpha"],
        df["mean_excess_risk"],
        marker="o",
        color="#1565C0",
        linewidth=2,
        markersize=6,
        label="Penalised ERM (L2)",
        zorder=3,
    )

    if df["std_excess_risk"].notna().all():
        ax.fill_between(
            df["alpha"],
            df["mean_excess_risk"] - df["std_excess_risk"],
            df["mean_excess_risk"] + df["std_excess_risk"],
            alpha=0.15,
            color="#1565C0",
            label="±1 std",
        )

    # Reference line: unpenalised ERM
    ax.axhline(
        erm_excess_risk,
        color="#E53935",
        linewidth=1.5,
        linestyle="--",
        label=f"Unpenalised ERM ({erm_excess_risk:.4f})",
        zorder=2,
    )

    # Mark optimal alpha
    idx_opt = df["mean_excess_risk"].idxmin()
    alpha_opt = df.loc[idx_opt, "alpha"]
    risk_opt = df.loc[idx_opt, "mean_excess_risk"]
    ax.axvline(
        alpha_opt,
        color="#43A047",
        linewidth=1.2,
        linestyle=":",
        label=f"Optimal α* = {alpha_opt}",
        zorder=2,
    )
    ax.scatter([alpha_opt], [risk_opt], color="#43A047", s=80, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Regularisation parameter α (log scale)", fontsize=10)
    ax.set_ylabel("Mean excess risk", fontsize=10)
    ax.set_title(
        f"Penalty path — {dgp_name}\n" f"Bias-variance tradeoff as α varies",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot penalty path figure.")
    parser.add_argument("--dgp", default="linear_gaussian")
    parser.add_argument("--output", default="reports/figures/penalty_path.pdf")
    parser.add_argument("--erm-excess-risk", type=float, default=0.0163)
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    df = load_penalty_path_runs(dgp_name=args.dgp, n=args.n)
    print(df.to_string(index=False))
    plot_penalty_path(
        df,
        erm_excess_risk=args.erm_excess_risk,
        output_path=args.output,
        dgp_name=args.dgp,
    )


if __name__ == "__main__":
    main()
