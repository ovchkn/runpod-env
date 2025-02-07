#!/bin/bash

# Initialize environment
echo "Initializing RunPod environment..."

# Create network storage directories
NETWORK_ROOT="/network/mlops"
mkdir -p "${NETWORK_ROOT}/"{datasets,models,experiments,checkpoints,logs}
mkdir -p "${NETWORK_ROOT}/models/"{fine_tuned,base,checkpoints}
mkdir -p "${NETWORK_ROOT}/experiments/"{cost,security,performance}

# Sync code repository to network storage
REPO_DIR="${NETWORK_ROOT}/code/runpod-env"
echo "Syncing code repository to network storage..."
mkdir -p "${REPO_DIR}"
rsync -av --exclude '.git' /workspace/ "${REPO_DIR}/"

# Create symlinks for workspace
ln -sf "${NETWORK_ROOT}/datasets" /workspace/datasets
ln -sf "${NETWORK_ROOT}/models" /workspace/models
ln -sf "${NETWORK_ROOT}/experiments" /workspace/experiments
ln -sf "${NETWORK_ROOT}/logs" /workspace/logs

# Copy configuration files if not exists
if [ ! -f /workspace/configs/api_keys.env ]; then
    cp /workspace/configs/api_keys.env.template /workspace/configs/api_keys.env
    echo "Please update API keys in /workspace/configs/api_keys.env"
fi

# Start monitoring and ML platform services
echo "Starting monitoring and ML platform services..."

# Start MLflow
systemctl start mlflow
echo "MLflow started on http://localhost:5000"

# Start KubeFlow
systemctl start kubeflow
echo "KubeFlow started on http://localhost:8000"

# Start LangFuse
systemctl start langfuse
echo "LangFuse started on http://localhost:3000"

# Start JupyterHub
systemctl start jupyterhub
echo "JupyterHub started on http://localhost:8888"

# Start Ollama
systemctl start ollama
echo "Ollama service started"

# Start model training service
systemctl start model_training
echo "Model training service started"

# Start model server
systemctl start model_server
echo "Model server started"

# Download and prepare base model if not exists
if [ ! -f "${NETWORK_ROOT}/models/base/Cline_FuseO1-DeepSeekR1-Qwen2.5-Coder-32B-Preview" ]; then
    echo "Downloading base model..."
    ollama pull nuibang/Cline_FuseO1-DeepSeekR1-Qwen2.5-Coder-32B-Preview
fi

# Prepare dataset if not exists
if [ ! -f "${NETWORK_ROOT}/datasets/terrads/TerraDS.sqlite" ]; then
    echo "Preparing TerraDS dataset..."
    /workspace/scripts/prepare_dataset.sh
fi

# Initialize experiment tracking
echo "Initializing experiment tracking..."
python3 -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.create_experiment('terraform_unified_optimization')
"

# Health check
echo "Performing health check..."
services=("mlflow" "kubeflow" "langfuse" "jupyterhub" "ollama" "model_training" "model_server")
for service in "${services[@]}"; do
    if systemctl is-active --quiet $service; then
        echo "$service: Running"
    else
        echo "$service: Failed to start"
        systemctl status $service
    fi
done

echo "Environment setup complete. Access points:"
echo "- MLflow UI: http://localhost:5000"
echo "- KubeFlow UI: http://localhost:8000"
echo "- LangFuse UI: http://localhost:3000"
echo "- JupyterHub: http://localhost:8888"
echo "- Unified notebook: /workspace/notebooks/terraform_optimization_unified.ipynb"