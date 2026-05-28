"""Attach ACR to Azure ML workspace."""

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

SUBSCRIPTION_ID = "e94d7bd0-e767-418f-86d1-c64056a0b776"
RESOURCE_GROUP = "crps-minimization-rg"
WORKSPACE_NAME = "crps-workspace"
ACR_ID = (
    f"/subscriptions/{SUBSCRIPTION_ID}"
    f"/resourceGroups/{RESOURCE_GROUP}"
    f"/providers/Microsoft.ContainerRegistry"
    f"/registries/crpsminimizationacr"
)

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

print("Attaching ACR to workspace...")
ws = ml_client.workspaces.get(WORKSPACE_NAME)
ws.container_registry = ACR_ID
ml_client.workspaces.begin_update(ws, update_dependent_resources=True).result()
print("Done.")
