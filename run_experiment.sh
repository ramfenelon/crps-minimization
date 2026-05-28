#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate crps-minimization
python -m src.experiments.run "$@"