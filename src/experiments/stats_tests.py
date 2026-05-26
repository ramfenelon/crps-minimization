"""
Statistical tests for benchmark comparisons.

Implements the methodology from Demsar (JMLR, 2006):
- Wilcoxon signed-rank test for pairwise model comparison
- Friedman test for comparing K models across M datasets
- Nemenyi post-hoc test with critical difference diagram

Reference:
    Demsar, J. (2006). Statistical comparisons of classifiers
    over multiple data sets. JMLR, 7, 1-30.
"""

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    model_a: str = "Model A",
    model_b: str = "Model B",
    alpha: float = 0.05,
) -> dict:
    """
    Wilcoxon signed-rank test comparing two models.

    Tests H0: the two models have the same median CRPS.
    Non-parametric — does not assume normality of differences.

    Parameters
    ----------
    scores_a : np.ndarray
        CRPS scores for model A across datasets/trials.
    scores_b : np.ndarray
        CRPS scores for model B across datasets/trials.
    model_a : str
        Name of model A.
    model_b : str
        Name of model B.
    alpha : float
        Significance level.

    Returns
    -------
    dict with keys: statistic, p_value, significant, winner
    """
    statistic, p_value = stats.wilcoxon(scores_a, scores_b)
    significant = p_value < alpha
    winner = None
    if significant:
        winner = model_a if np.median(scores_a) < np.median(scores_b) else model_b

    return {
        "model_a": model_a,
        "model_b": model_b,
        "statistic": statistic,
        "p_value": p_value,
        "significant": significant,
        "winner": winner,
        "median_a": float(np.median(scores_a)),
        "median_b": float(np.median(scores_b)),
    }


def friedman_test(scores_df: pd.DataFrame) -> dict:
    """
    Friedman test comparing K models across M datasets.

    Tests H0: all models have the same median rank.
    scores_df rows = datasets/trials, columns = models.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Shape (M, K) — M datasets, K models. Values are CRPS scores.

    Returns
    -------
    dict with keys: statistic, p_value, significant, ranks
    """
    statistic, p_value = stats.friedmanchisquare(
        *[scores_df[col].values for col in scores_df.columns]
    )
    ranks = scores_df.rank(axis=1).mean()

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "ranks": ranks.to_dict(),
    }


def nemenyi_test(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Nemenyi post-hoc test after a significant Friedman test.

    Returns a matrix of p-values for all pairwise comparisons.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Shape (M, K) — M datasets, K models. Values are CRPS scores.

    Returns
    -------
    pd.DataFrame of shape (K, K) — pairwise p-values.
    """
    return sp.posthoc_nemenyi_friedman(scores_df)


def run_all_tests(benchmark_df: pd.DataFrame) -> None:
    """
    Run the full Demsar (2006) statistical testing pipeline.

    Parameters
    ----------
    benchmark_df : pd.DataFrame
        Output of run_benchmark() — models as rows, DGPs as columns.
    """
    # Transpose so rows=DGPs, columns=models (required by Friedman)
    scores = benchmark_df.T

    print("=" * 60)
    print("STATISTICAL TESTS — Demsar (JMLR, 2006)")
    print("=" * 60)

    # ── Friedman test ──────────────────────────────────────────
    print("\n1. Friedman Test (all models simultaneously)")
    print("-" * 40)
    result = friedman_test(scores)
    print(f"   Statistic: {result['statistic']:.4f}")
    print(f"   p-value:   {result['p_value']:.4f}")
    print(f"   Significant (p<0.05): {result['significant']}")
    print("\n   Average ranks (lower = better):")
    ranks = sorted(result["ranks"].items(), key=lambda x: x[1])
    for model, rank in ranks:
        print(f"     {model:<30} {rank:.3f}")

    # ── Nemenyi post-hoc ───────────────────────────────────────
    if result["significant"]:
        print("\n2. Nemenyi Post-hoc Test (pairwise p-values)")
        print("-" * 40)
        p_matrix = nemenyi_test(scores)
        print(p_matrix.round(4).to_string())

        print("\n   Significant pairs (p < 0.05):")
        models = list(scores.columns)
        found = False
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                p = p_matrix.iloc[i, j]
                if p < 0.05:
                    mean_i = scores[models[i]].mean()
                    mean_j = scores[models[j]].mean()
                    winner = models[i] if mean_i < mean_j else models[j]
                    print(
                        f"     {models[i]} vs {models[j]}: "
                        f"p={p:.4f} → {winner} wins"
                    )
                    found = True
        if not found:
            print("     None found.")
    else:
        print("\n   Friedman test not significant — " "no post-hoc tests performed.")

    # ── Wilcoxon: ERM vs each competitor ──────────────────────
    print("\n3. Wilcoxon Tests: GaussianERM vs each competitor")
    print("-" * 40)
    erm_scores = scores["GaussianERM"].values
    competitors = [c for c in scores.columns if c != "GaussianERM"]
    for comp in competitors:
        comp_scores = scores[comp].values
        result = wilcoxon_test(
            erm_scores, comp_scores, model_a="GaussianERM", model_b=comp
        )
        sig = "✓ significant" if result["significant"] else "✗ not significant"
        winner = result["winner"] or "no difference"
        print(f"   vs {comp:<28} p={result['p_value']:.4f}  " f"{sig}  →  {winner}")
