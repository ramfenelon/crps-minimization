"""
Industry-standard Azure ML experiment pipeline.

Submits a job, waits for completion, downloads results.

Usage from Git Bash:
    python azure/run_experiment_pipeline.py
    python azure/run_experiment_pipeline.py --n 1000
    python azure/run_experiment_pipeline.py --n 500 --alpha 0.01 --model penalised_erm
"""

import argparse
import time
from pathlib import Path

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# ── config ────────────────────────────────────────────────────────────────────
SUBSCRIPTION_ID = "e94d7bd0-e767-418f-86d1-c64056a0b776"
RESOURCE_GROUP = "crps-minimization-rg"
WORKSPACE_NAME = "crps-workspace"
ACR_NAME = "crpsminimizationacr"
IMAGE_VERSION = "v1.5"
DOWNLOAD_DIR = Path("results/azure")


# ── helpers ───────────────────────────────────────────────────────────────────


def get_client() -> MLClient:
    """Create authenticated Azure ML client."""
    return MLClient(
        DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


def submit_job(ml_client: MLClient, experiment_command: str) -> str:
    """
    Submit experiment job to Azure ML.

    Parameters
    ----------
    ml_client : MLClient
    experiment_command : str
        Command to pass to the experiment runner e.g. 'data.n=500'

    Returns
    -------
    str : job name
    """
    print(f"\n[1/3] Submitting job: {experiment_command}")

    env = Environment(
        name="crps-env",
        image=f"{ACR_NAME}.azurecr.io/crps-experiment:{IMAGE_VERSION}",
    )
    env = ml_client.environments.create_or_update(env)

    job = command(
        display_name=f"crps-{experiment_command.replace(' ', '-').replace('=', '')}",
        experiment_name="crps-minimization",
        command=f"bash run_experiment.sh {experiment_command}",
        environment=f"crps-env:{env.version}",
        compute="crps-cluster",
        code="./",
    )

    returned_job = ml_client.jobs.create_or_update(job)
    print(f"      Job name:   {returned_job.name}")
    print(f"      Status:     {returned_job.status}")
    print(f"      Studio URL: {returned_job.studio_url}")
    return returned_job.name


def wait_for_completion(ml_client: MLClient, job_name: str) -> str:
    """
    Poll job status until terminal state.

    Parameters
    ----------
    ml_client : MLClient
    job_name : str

    Returns
    -------
    str : final status ('Completed' or 'Failed')
    """
    print(f"\n[2/3] Waiting for job '{job_name}' to complete...")

    terminal_states = {"Completed", "Failed", "Canceled", "NotResponding"}
    poll_interval = 15  # seconds

    while True:
        job = ml_client.jobs.get(job_name)
        status = job.status
        print(f"      Status: {status}")

        if status in terminal_states:
            return status

        time.sleep(poll_interval)


def download_results(ml_client: MLClient, job_name: str) -> Path:
    """
    Download job outputs and logs to local results directory.

    Parameters
    ----------
    ml_client : MLClient
    job_name : str

    Returns
    -------
    Path : directory where results were saved
    """
    print(f"\n[3/3] Downloading results for job '{job_name}'...")

    job_dir = DOWNLOAD_DIR / job_name
    job_dir.mkdir(parents=True, exist_ok=True)

    ml_client.jobs.download(
        name=job_name,
        download_path=str(job_dir),
        all=True,
    )

    print(f"      Saved to: {job_dir}")

    # Show what was downloaded
    files = list(job_dir.rglob("*"))
    print(f"      Files downloaded: {len(files)}")
    for f in sorted(files)[:20]:
        if f.is_file():
            print(f"        {f.relative_to(job_dir)}")

    return job_dir


def print_metrics(ml_client: MLClient, job_name: str) -> None:
    """Print key metrics from the completed job."""
    print("\n      Key metrics:")
    job = ml_client.jobs.get(job_name)
    try:
        metrics = job.properties
        for k, v in metrics.items():
            if any(m in k.lower() for m in ["crps", "risk", "loss"]):
                print(f"        {k}: {v}")
    except Exception:
        print("        (metrics available in Studio UI)")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Submit, wait, download Azure ML experiment."
    )
    parser.add_argument("--n", type=int, default=500, help="Sample size")
    parser.add_argument(
        "--model", type=str, default="gaussian_erm", help="Model config name"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Regularisation parameter (penalised ERM only)",
    )
    parser.add_argument(
        "--dgp", type=str, default="linear_gaussian", help="DGP config name"
    )
    args = parser.parse_args()

    # Build experiment command
    cmd = f"data.n={args.n} data={args.dgp} model={args.model}"
    if args.alpha is not None:
        cmd += f" model.alpha={args.alpha}"

    print("=" * 60)
    print("CRPS MINIMIZATION — Azure ML Pipeline")
    print("=" * 60)
    print(f"Experiment: {cmd}")

    ml_client = get_client()

    # ── Step 1: Submit ────────────────────────────────────────
    job_name = submit_job(ml_client, cmd)

    # ── Step 2: Wait ──────────────────────────────────────────
    final_status = wait_for_completion(ml_client, job_name)

    if final_status != "Completed":
        print(f"\nJob failed with status: {final_status}")
        print("Check logs at: ml.azure.com")
        return

    print("\n      Job completed successfully.")

    # ── Step 3: Download ──────────────────────────────────────
    result_dir = download_results(ml_client, job_name)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Job name:    {job_name}")
    print(f"Status:      {final_status}")
    print(f"Results dir: {result_dir}")
    print("Studio URL:  https://ml.azure.com")
    print("\nNext steps:")
    print("  inv figures   ← regenerate all paper figures locally")
    print("=" * 60)


if __name__ == "__main__":
    main()
