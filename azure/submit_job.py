"""
Submit an experiment job to Azure ML.

Run from Git Bash:
    python azure/submit_job.py
"""

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

SUBSCRIPTION_ID = "e94d7bd0-e767-418f-86d1-c64056a0b776"
RESOURCE_GROUP = "crps-minimization-rg"
WORKSPACE_NAME = "crps-workspace"
ACR_NAME = "crpsminimizationacr"

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

# ── register environment ──────────────────────────────────────────────────────
print("Registering environment...")
env = Environment(
    name="crps-env",
    description="CRPS minimisation experiment environment",
    image=f"{ACR_NAME}.azurecr.io/crps-experiment:v1.3",
)
env = ml_client.environments.create_or_update(env)
print(f"Environment registered: {env.name}:{env.version}")

# ── submit job ────────────────────────────────────────────────────────────────
print("Submitting job...")
job = command(
    display_name="crps-erm-linear-gaussian",
    experiment_name="crps-minimization",
    command="bash run_experiment.sh data.n=500",
    environment=f"crps-env:{env.version}",
    compute="crps-cluster",
    code="./",
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
print(f"Status:        {returned_job.status}")
print(f"Studio URL:    {returned_job.studio_url}")
