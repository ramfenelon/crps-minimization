"""Assign roles to compute cluster managed identity."""

import uuid

from azure.identity import DefaultAzureCredential
from azure.mgmt.authorization import AuthorizationManagementClient
from azure.mgmt.authorization.models import RoleAssignmentCreateParameters

SUBSCRIPTION_ID = "e94d7bd0-e767-418f-86d1-c64056a0b776"
RESOURCE_GROUP = "crps-minimization-rg"
WORKSPACE_NAME = "crps-workspace"
PRINCIPAL_ID = "e94d4780-c6a2-4b62-8f58-d1c2057e8b66"

credential = DefaultAzureCredential()
auth_client = AuthorizationManagementClient(credential, SUBSCRIPTION_ID)

# Scope for workspace
workspace_scope = (
    f"/subscriptions/{SUBSCRIPTION_ID}"
    f"/resourceGroups/{RESOURCE_GROUP}"
    f"/providers/Microsoft.MachineLearningServices"
    f"/workspaces/{WORKSPACE_NAME}"
)

# Scope for resource group
rg_scope = f"/subscriptions/{SUBSCRIPTION_ID}" f"/resourceGroups/{RESOURCE_GROUP}"

# Role definition IDs (fixed Azure built-in role IDs)
roles = {
    "AzureML Data Scientist": "f6c7c914-8db3-469d-8ca1-694a8f32e121",
    "Storage Blob Data Contributor": "ba92f5b4-2d11-453d-a403-e96b0029c9fe",
}

assignments = [
    (roles["AzureML Data Scientist"], workspace_scope),
    (roles["Storage Blob Data Contributor"], rg_scope),
]

for role_id, scope in assignments:
    role_name = [k for k, v in roles.items() if v == role_id][0]
    print(f"Assigning {role_name}...")
    try:
        auth_client.role_assignments.create(
            scope=scope,
            role_assignment_name=str(uuid.uuid4()),
            parameters=RoleAssignmentCreateParameters(
                role_definition_id=(
                    f"/subscriptions/{SUBSCRIPTION_ID}"
                    f"/providers/Microsoft.Authorization"
                    f"/roleDefinitions/{role_id}"
                ),
                principal_id=PRINCIPAL_ID,
                principal_type="ServicePrincipal",
            ),
        )
        print("  Done.")
    except Exception as e:
        print(f"  Error: {e}")
