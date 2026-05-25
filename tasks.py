"""
Invoke task runner for the crps-minimization project.

Replaces GNU Make on Windows. Run tasks from Git Bash:

    inv figures        # regenerate all paper figures
    inv test           # run test suite
    inv experiments    # run all experiments
    inv clean          # remove generated figures
    inv all            # experiments + figures
"""

from pathlib import Path

from invoke import task

# ── configuration ─────────────────────────────────────────────────────────────

FIGURES_DIR = Path("reports/figures")
SRC_DIR = Path("src")


# ── helper ────────────────────────────────────────────────────────────────────


def run_module(c, module: str, args: str = "") -> None:
    """Run a Python module from the repo root."""
    import os

    env = {**os.environ, "PYTHONPATH": str(Path(".").resolve())}
    c.run(f"python -m {module} {args}", pty=False, env=env)


# ── tasks ─────────────────────────────────────────────────────────────────────


@task
def test(c):
    """Run the full test suite with coverage."""
    c.run("python -m pytest tests/ -v")


@task
def clean(c):
    """Remove all generated figures."""
    for f in FIGURES_DIR.glob("*.pdf"):
        f.unlink()
        print(f"Removed: {f}")
    print("Clean done.")


@task
def experiments(c):
    """
    Run all experiments needed to produce paper figures.

    Runs convergence sweep (7 sample sizes) and alpha sweep.
    Results are logged to MLflow and used by figure scripts.
    """
    print("=== Running convergence experiments ===")
    for n in [50, 100, 200, 500, 1000, 2000, 5000]:
        print(f"  n={n}...")
        run_module(c, "src.experiments.run", f"data.n={n}")

    print("\n=== Running alpha sweep ===")
    for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]:
        print(f"  alpha={alpha}...")
        run_module(c, "src.experiments.run", f"model=penalised_erm model.alpha={alpha}")

    print("\nAll experiments done.")


@task
def fig_convergence(c):
    """Regenerate convergence figure (excess risk vs n)."""
    print("Generating convergence figure...")
    run_module(c, "src.viz.convergence", f"--output {FIGURES_DIR}/convergence.pdf")


@task
def fig_penalty_path(c):
    """Regenerate penalty path figure (excess risk vs alpha)."""
    print("Generating penalty path figure...")
    run_module(
        c,
        "src.viz.penalty_path",
        f"--dgp linear_gaussian "
        f"--erm-excess-risk 0.0163 "
        f"--output {FIGURES_DIR}/penalty_path.pdf",
    )
    run_module(
        c,
        "src.viz.penalty_path",
        f"--dgp heteroscedastic_gaussian "
        f"--n 100 "
        f"--erm-excess-risk 0.0830 "
        f"--output {FIGURES_DIR}/penalty_path_hetero_n100.pdf",
    )


@task
def fig_calibration(c):
    """Regenerate calibration diagram."""
    print("Generating calibration diagram...")
    run_module(c, "src.viz.calibration", f"--output {FIGURES_DIR}/calibration.pdf")


@task
def fig_benchmark(c):
    """Regenerate benchmark heatmap."""
    print("Generating benchmark heatmap...")
    run_module(
        c, "src.viz.benchmark_heatmap", f"--output {FIGURES_DIR}/benchmark_heatmap.pdf"
    )


@task(pre=[fig_convergence, fig_penalty_path, fig_calibration, fig_benchmark])
def figures(c):
    """Regenerate ALL paper figures."""
    print("\n=== All figures complete ===")
    for f in sorted(FIGURES_DIR.glob("*.pdf")):
        print(f"  {f}")


@task(pre=[experiments, figures])
def all(c):
    """Run all experiments then regenerate all figures."""
    print("\n=== Pipeline complete ===")
