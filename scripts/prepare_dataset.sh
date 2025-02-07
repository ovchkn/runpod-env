#!/bin/bash

# Configuration
NETWORK_PATH="/network/mlops"
DATASET_PATH="${NETWORK_PATH}/datasets"
TERRADS_URL="https://zenodo.org/records/14217386/files"

# Create directories
mkdir -p "${DATASET_PATH}/terrads"
cd "${DATASET_PATH}/terrads"

echo "Downloading TerraDS dataset..."

# Download dataset files
echo "Downloading SQLite database..."
wget "${TERRADS_URL}/TerraDS.sqlite?download=1" -O TerraDS.sqlite

echo "Downloading source code archive..."
wget "${TERRADS_URL}/TerraDS.tar.gz?download=1" -O TerraDS.tar.gz

echo "Downloading tools..."
wget "${TERRADS_URL}/HCL%20Dataset%20Tools.zip?download=1" -O HCL_Dataset_Tools.zip

# Extract archives
echo "Extracting source code..."
tar xzf TerraDS.tar.gz -C "${DATASET_PATH}/terrads/source"

echo "Extracting tools..."
unzip HCL_Dataset_Tools.zip -d "${DATASET_PATH}/terrads/tools"

# Create experiment directories
mkdir -p "${NETWORK_PATH}/experiments/cost_optimization"
mkdir -p "${NETWORK_PATH}/experiments/cost_optimization/results"
mkdir -p "${NETWORK_PATH}/experiments/cost_optimization/models"

# Set up symlinks for JupyterHub
ln -sf "${DATASET_PATH}/terrads" "/workspace/notebooks/datasets/terrads"
ln -sf "${NETWORK_PATH}/experiments/cost_optimization" "/workspace/notebooks/experiments/cost_optimization"

echo "Dataset preparation complete. Files available at:"
echo "- SQLite DB: ${DATASET_PATH}/terrads/TerraDS.sqlite"
echo "- Source Code: ${DATASET_PATH}/terrads/source"
echo "- Tools: ${DATASET_PATH}/terrads/tools"
echo "- Experiment Directory: ${NETWORK_PATH}/experiments/cost_optimization"