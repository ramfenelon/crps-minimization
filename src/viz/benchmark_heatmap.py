"""
Benchmark heatmap: test CRPS across models and DGPs.

Runs all models on all DGPs and plots a colour-coded heatmap
where lower CRPS = better = darker blue.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dgp import (
    HeteroscedasticGaussianDGP,
    LinearGaussianDGP,
    MixtureGaussianDGP,
    StudentTDGP,
)
from src.models.competitors import get_competitors
from src.models.erm import GaussianERM, PenalisedGaussianERM


def get_dgps(n: int = 500, d: int = 10, seed: int = 42) -> dict:
    """Return all DGPs as a dictionary."""
    return {
        "LinearGaussian": LinearGaussianDGP(n=n, d=d, sigma=1.0, seed=seed),
        "Heteroscedastic": HeteroscedasticGaussianDGP(n=n, d=d, seed=seed),
        "MixtureGaussian": MixtureGaussianDGP(n=n, d=d, seed=seed),
        "StudentT": StudentTDGP(n=n, d=d, nu=4.0, seed=seed),
    }


def run_benchmark(
    n: int = 500,
    d: int = 10,
    seed: int = 42,
    n_trials: int = 5,
) -> pd.DataFrame:
    """
    Run all models on all DGPs and return mean test CRPS.

    Parameters
    ----------
    n : int
        Sample size per trial.
    d : int
        Number of features.
    seed : int
        Base random seed.
    n_trials : int
        Number of trials per model-DGP combination.

    Returns
    -------
    pd.DataFrame with models as rows, DGPs as columns, CRPS as values.
    """
    models = {
        "GaussianERM": GaussianERM(),
        "PenalisedERM (L2)": PenalisedGaussianERM(alpha=0.01, penalty="l2"),
        **get_competitors(),
    }

    dgp_factories = {
        "LinearGaussian": lambda s: LinearGaussianDGP(n=n, d=d, sigma=1.0, seed=s),
        "Heteroscedastic": lambda s: HeteroscedasticGaussianDGP(n=n, d=d, seed=s),
        "MixtureGaussian": lambda s: MixtureGaussianDGP(n=n, d=d, seed=s),
        "StudentT": lambda s: StudentTDGP(n=n, d=d, nu=4.0, seed=s),
    }

    results = {}

    for model_name, model in models.items():
        print(f"  Running {model_name}...")
        results[model_name] = {}

        for dgp_name, dgp_factory in dgp_factories.items():
            trial_crps = []

            for trial in range(n_trials):
                trial_seed = seed + trial
                dgp = dgp_factory(trial_seed)
                X, y = dgp.sample()

                n_test = n // 5
                X_train, X_test = X[:-n_test], X[-n_test:]
                y_train, y_test = y[:-n_test], y[-n_test:]

                try:
                    import sklearn

                    m = (
                        sklearn.clone(model)
                        if hasattr(model, "get_params")
                        else model.__class__(**model.get_params())
                    )
                except Exception:
                    m = model.__class__()

                m.fit(X_train, y_train)
                crps = -m.score(X_test, y_test)
                trial_crps.append(crps)

            results[model_name][dgp_name] = float(np.mean(trial_crps))
            print(f"    {dgp_name}: CRPS = {results[model_name][dgp_name]:.4f}")

    return pd.DataFrame(results).T


def plot_heatmap(
    df: pd.DataFrame,
    output_path: str = "reports/figures/benchmark_heatmap.pdf",
) -> None:
    """
    Plot benchmark results as a colour-coded heatmap.

    Lower CRPS = better = darker blue.

    Parameters
    ----------
    df : pd.DataFrame
        Output of run_benchmark() — models x DGPs.
    output_path : str
        Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    im = ax.imshow(df.values, cmap="Blues_r", aspect="auto")

    # Axis labels
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, fontsize=10)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=10)

    # Annotate each cell with the CRPS value
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.values[i, j]
            # Use white text on dark cells, black on light cells
            text_color = "white" if val < df.values.mean() else "black"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax, label="Mean test CRPS (lower = better)")
    ax.set_title(
        "Benchmark: mean test CRPS across models and DGPs\n"
        "Lower is better — darker = lower CRPS",
        fontsize=11,
    )
    ax.set_xlabel("DGP", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run benchmark and plot heatmap.")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--output", default="reports/figures/benchmark_heatmap.pdf")
    args = parser.parse_args()

    print("Running benchmark...")
    df = run_benchmark(n=args.n, d=args.d, n_trials=args.n_trials)
    print("\nResults:")
    print(df.round(4).to_string())
    plot_heatmap(df, output_path=args.output)


if __name__ == "__main__":
    main()
