"""
Calibration diagram for Gaussian distributional regression.

For each nominal coverage level p in [0.1, 0.2, ..., 0.9],
compute the fraction of test observations that fall within
the symmetric p-level prediction interval.

A perfectly calibrated model follows the diagonal y=x.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.dgp import (
    HeteroscedasticGaussianDGP,
    LinearGaussianDGP,
    MixtureGaussianDGP,
    StudentTDGP,
)
from src.models.erm import GaussianERM


def compute_coverage(
    model: GaussianERM,
    X_test: np.ndarray,
    y_test: np.ndarray,
    levels: np.ndarray,
) -> np.ndarray:
    """
    Compute empirical coverage at each nominal level.

    For each level p, the symmetric prediction interval is:
        [Q_{(1-p)/2}(X), Q_{(1+p)/2}(X)]

    Parameters
    ----------
    model : fitted GaussianERM
    X_test : ndarray of shape (n, d)
    y_test : ndarray of shape (n,)
    levels : ndarray of nominal coverage levels in (0, 1)

    Returns
    -------
    ndarray of empirical coverage at each level
    """
    empirical = np.zeros(len(levels))
    for i, p in enumerate(levels):
        lower = model.predict_quantile(X_test, q=(1 - p) / 2)
        upper = model.predict_quantile(X_test, q=(1 + p) / 2)
        empirical[i] = float(np.mean((y_test >= lower) & (y_test <= upper)))
    return empirical


def get_dgp(name: str, n: int, d: int, seed: int):
    """Instantiate DGP by name."""
    if name == "linear_gaussian":
        return LinearGaussianDGP(n=n, d=d, sigma=1.0, seed=seed)
    elif name == "heteroscedastic_gaussian":
        return HeteroscedasticGaussianDGP(n=n, d=d, seed=seed)
    elif name == "mixture_gaussian":
        return MixtureGaussianDGP(n=n, d=d, seed=seed)
    elif name == "student_t":
        return StudentTDGP(n=n, d=d, seed=seed)
    else:
        raise ValueError(f"Unknown DGP: {name}")


def plot_calibration(
    dgp_names: list,
    n: int = 1000,
    d: int = 10,
    seed: int = 42,
    output_path: str = "reports/figures/calibration.pdf",
) -> None:
    """
    Plot calibration diagram for GaussianERM across multiple DGPs.

    Parameters
    ----------
    dgp_names : list of str
        DGP names to plot — one curve per DGP.
    n : int
        Sample size for training.
    d : int
        Number of features.
    seed : int
        Random seed.
    output_path : str
        Where to save the figure.
    """
    levels = np.linspace(0.1, 0.9, 9)
    colours = ["#1565C0", "#00695C", "#E53935", "#F9A825"]
    labels = {
        "linear_gaussian": "LinearGaussian",
        "heteroscedastic_gaussian": "HeteroscedasticGaussian",
        "mixture_gaussian": "MixtureGaussian",
        "student_t": "StudentT",
    }

    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration diagonal
    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        linewidth=1.2,
        linestyle="--",
        label="Perfect calibration",
        zorder=1,
    )

    for dgp_name, colour in zip(dgp_names, colours):
        dgp = get_dgp(dgp_name, n=n, d=d, seed=seed)
        X, y = dgp.sample()

        # Train/test split
        n_test = n // 5
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        # Fit model
        model = GaussianERM()
        model.fit(X_train, y_train)

        # Compute coverage
        empirical = compute_coverage(model, X_test, y_test, levels)

        ax.plot(
            levels,
            empirical,
            marker="o",
            color=colour,
            linewidth=2,
            markersize=5,
            label=labels.get(dgp_name, dgp_name),
            zorder=3,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal coverage level", fontsize=10)
    ax.set_ylabel("Empirical coverage", fontsize=10)
    ax.set_title(
        "Calibration diagram — GaussianERM\n"
        "Empirical vs nominal prediction interval coverage",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Shade the overconfident / underconfident regions
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color="red", label="_nolegend_")
    ax.fill_between(
        [0, 1], [0, 1], [1, 1], alpha=0.04, color="blue", label="_nolegend_"
    )
    ax.text(0.72, 0.15, "Overconfident", fontsize=8, color="red", alpha=0.7)
    ax.text(0.05, 0.82, "Underconfident", fontsize=8, color="blue", alpha=0.7)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot calibration diagram.")
    parser.add_argument(
        "--dgps",
        nargs="+",
        default=[
            "linear_gaussian",
            "heteroscedastic_gaussian",
            "mixture_gaussian",
            "student_t",
        ],
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--output", default="reports/figures/calibration.pdf")
    args = parser.parse_args()

    plot_calibration(
        dgp_names=args.dgps,
        n=args.n,
        d=args.d,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
