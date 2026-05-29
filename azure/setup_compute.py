"""
Create Azure ML compute cluster for experiment jobs.

Run once from Git Bash:
    python azure/setup_compute.py
"""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, IdentityConfiguration
from azure.identity import DefaultAzureCredential

SUBSCRIPTION_ID = "e94d7bd0-e767-418f-86d1-c64056a0b776"
RESOURCE_GROUP = "crps-minimization-rg"
WORKSPACE_NAME = "crps-workspace"

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

print("Creating compute cluster...")


cluster = AmlCompute(
    name="crps-cluster",
    type="amlcompute",
    size="Standard_DS2_v2",
    min_instances=0,
    max_instances=4,
    idle_time_before_scale_down=120,
    identity=IdentityConfiguration(type="system_assigned"),
)

poller = ml_client.begin_create_or_update(cluster)
result = poller.result()

print(f"Cluster created: {result.name}")
print(f"VM size:         {result.size}")
print(f"Min instances:   {result.min_instances}")
print(f"Max instances:   {result.max_instances}")
print("Cost when idle:  £0.00 (scales to zero)")
