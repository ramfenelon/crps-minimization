![CI](https://github.com/ramfenelon/crps-minimization/actions/workflows/ci.yml/badge.svg)
[![DOI](https://zenodo.org/badge/1240585720.svg)](https://doi.org/10.5281/zenodo.20402443)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# CRPS Minimization

Empirical validation of generalisation theory for empirical risk minimisation
(ERM) and penalised ERM in **distributional regression** under the Continuous
Ranked Probability Score (CRPS).

This repository contains all code, data pipelines, and experiment results
supporting the paper:

> [Your Name] ([Year]). *[Paper Title]*. [Journal/Conference].

---

## What This Repository Contains

| Directory | Contents |
|-----------|----------|
| `src/data/` | Data generating processes (DGPs) and real dataset pipelines |
| `src/models/` | GaussianERM, PenalisedGaussianERM, and competitor models |
| `src/experiments/` | Experiment runner (Hydra + MLflow) and statistical tests |
| `src/viz/` | Figure generation scripts |
| `configs/` | Hydra YAML configs for all experiments |
| `notebooks/` | EDA notebooks for California Housing and Abalone |
| `reports/figures/` | All paper figures (PDF) |
| `tests/` | Unit and integration tests |

---

## Reproducing the Paper's Results

### 1. Clone the repository

```bash
git clone git@github.com:ramfenelon/crps-minimization.git
cd crps-minimization
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate crps-minimization
```

### 3. Pull the data

```bash
dvc pull
```

### 4. Run all experiments

```bash
inv experiments
```

This runs the convergence sweep (n = 50 to 5000) and the
regularisation parameter sweep (α = 10⁻⁵ to 1) and logs
all results to MLflow.

### 5. Regenerate all figures

```bash
inv figures
```

All figures are saved to `reports/figures/` as PDF.

### 6. Run the test suite

```bash
inv test
```

---

## Key Results

### Convergence of excess risk

The empirical convergence rate is O(n⁻¹·⁰⁶), exceeding the theoretical
O(n⁻⁰·⁵) bound — consistent with the parametric rate under correct
specification.

### Penalty path

The optimal regularisation parameter α* decreases as n grows. At n = 100
on the heteroscedastic DGP, penalised ERM reduces excess risk by 12.7%
relative to unpenalised ERM (α* = 0.01).

### Benchmark

GaussianERM and PenalisedERM outperform all competitor models by 40–70%
CRPS across all DGPs. The Friedman test confirms overall differences are
statistically significant (χ² = 16.86, p = 0.0048). GaussianERM
significantly outperforms NGBoost by Nemenyi post-hoc test (p = 0.030).

---

## Data

| Dataset | Source | Size | Used for |
|---------|--------|------|----------|
| Synthetic DGPs | Generated | Varies | Theory validation |
| California Housing | sklearn.datasets | 20,640 rows | Real-data benchmark |
| Abalone | UCI ML Repository | 4,177 rows | Real-data benchmark |

---

## Installation Notes (Windows)

All commands are run in **Git Bash**. The conda environment name is
`crps-minimization`. See `environment.yml` for the full dependency list
including required version pins (`setuptools=69.5.1`, `pathspec==0.11.2`).

---

## Project Structure

crps-minimization/
├── src/
│   ├── data/          # DGP classes and dataset loaders
│   ├── models/        # ERM estimators and competitors
│   ├── experiments/   # Experiment runner and statistical tests
│   └── viz/           # Figure generation
├── configs/           # Hydra experiment configs
├── notebooks/         # EDA notebooks
├── tests/             # pytest test suite
├── tasks.py           # Invoke task runner (inv figures, inv test)
└── environment.yml    # Conda environment

---

## Citation

If you use this code, please cite:

```bibtex
@software{fenelon2026crps,
  author  = {[Your Name]},
  title   = {CRPS Minimization},
  year    = {2026},
  doi     = {10.5281/zenodo.20402444},
  url     = {https://github.com/ramfenelon/crps-minimization},
}
```

---

## Licence

MIT — see [LICENSE](LICENSE) for details.