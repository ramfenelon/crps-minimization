#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate crps-minimization

# Use Azure ML's MLflow tracking server if available
if [ -n "$MLFLOW_TRACKING_URI" ]; then
    echo "Using Azure ML MLflow tracking: $MLFLOW_TRACKING_URI"
else
    echo "Using local MLflow tracking"
fi

python -m src.experiments.run "$@"