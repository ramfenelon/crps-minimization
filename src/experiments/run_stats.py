"""Run statistical tests on benchmark results."""

from src.experiments.stats_tests import run_all_tests
from src.viz.benchmark_heatmap import run_benchmark

if __name__ == "__main__":
    print("Running benchmark (5 trials)...")
    df = run_benchmark(n=500, d=10, n_trials=5)
    print("\nBenchmark results:")
    print(df.round(4).to_string())
    run_all_tests(df)
