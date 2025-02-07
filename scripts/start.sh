#!/bin/bash

# Function to check if a service is running
check_service() {
    systemctl is-active --quiet $1
    return $?
}

# Function to wait for a service to be ready
wait_for_service() {
    local service=$1
    local max_attempts=30
    local attempt=1

    echo "Waiting for $service to be ready..."
    while ! check_service $service; do
        if [ $attempt -ge $max_attempts ]; then
            echo "Service $service failed to start after $max_attempts attempts"
            return 1
        fi
        echo "Attempt $attempt: $service is not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo "$service is ready!"
    return 0
}

# Function to handle container shutdown
cleanup() {
    echo "Container shutdown initiated..."
    systemctl stop docker
    systemctl stop ollama
    systemctl stop mlflow
    systemctl stop jupyterhub
    exit 0
}

# Set up signal handling
trap cleanup SIGTERM SIGINT SIGQUIT

# Ensure sysbox-runc is properly initialized
if [ ! -f "/var/lib/sysbox/kubelet/kubelet.conf" ]; then
    echo "Initializing sysbox-runc..."
    mkdir -p /var/lib/sysbox/kubelet
    touch /var/lib/sysbox/kubelet/kubelet.conf
fi

# Create necessary directories
mkdir -p /workspace/{notebooks,models,datasets,pipelines,experiments,scripts,configs,logs}
mkdir -p /workspace/services/{mlflow,kubeflow,langfuse,ollama}
mkdir -p /workspace/datasets/training
mkdir -p /workspace/models/fine_tuned
mkdir -p /workspace/cache/ollama

# Set up data pipeline directories
mkdir -p /workspace/data_pipeline/raw_data
mkdir -p /workspace/data_pipeline/processed_data
mkdir -p /workspace/data_pipeline/embeddings

# Set up model training directories
mkdir -p /workspace/model_training/checkpoints
mkdir -p /workspace/model_training/logs
mkdir -p /workspace/model_training/artifacts

# Set up network storage if available
if [ ! -z "${NETWORK_STORAGE_ROOT}" ]; then
    echo "Setting up network storage at ${NETWORK_STORAGE_ROOT}..."
    mkdir -p "${NETWORK_STORAGE_ROOT}/models/ollama"
    mkdir -p "${NETWORK_STORAGE_ROOT}/models/fine_tuned"
    
    # Create symlinks for easier access
    ln -sf "${NETWORK_STORAGE_ROOT}/models" /workspace/network_models
fi

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate mlops

# Start systemd services
echo "Starting core services..."
systemctl start docker
wait_for_service docker

# Configure Docker runtime for Ollama
if [ ! -f "/etc/docker/daemon.json" ]; then
    echo "Configuring Docker runtime..."
    mkdir -p /etc/docker
    cat > /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "sysbox-runc": {
            "path": "/usr/bin/sysbox-runc"
        }
    }
}
EOF
    systemctl restart docker
    wait_for_service docker
fi

# Configure external access for Ollama
if [ ! -z "${EXTERNAL_HOST}" ] && [ ! -z "${EXTERNAL_PORT}" ]; then
    echo "Configuring external access for Ollama..."
    /workspace/scripts/manage_model.sh configure-external "${EXTERNAL_HOST}" "${EXTERNAL_PORT}"
else
    # Start Ollama service with default configuration
    echo "Starting Ollama service..."
    systemctl start ollama
fi

wait_for_service ollama

# Initialize MLflow
echo "Starting MLflow service..."
systemctl start mlflow
wait_for_service mlflow

# Create default MLflow experiments
mlflow experiments create -n "infrastructure_dataset_creation" 2>/dev/null || true
mlflow experiments create -n "infrastructure_reasoning_fine_tuning" 2>/dev/null || true
mlflow experiments create -n "terraform-optimization" 2>/dev/null || true
mlflow experiments create -n "model-fine-tuning" 2>/dev/null || true

# Start JupyterHub
echo "Starting JupyterHub..."
systemctl start jupyterhub
wait_for_service jupyterhub

# Start TensorBoard
tensorboard --logdir=/workspace/experiments --port=6006 --bind_all &

# Initialize model management
echo "Initializing model management..."
/workspace/scripts/manage_model.sh sync

# Configure Ollama model if not already present
if ! ollama list | grep -q "fuse-ai"; then
    echo "Setting up FuseAI model in Ollama..."
    ollama pull FuseAI/FuseO1-DeepSeekR1-Qwen2.5-Coder-32B-Preview
    if [ -f "/workspace/configs/ollama_model.yaml" ]; then
        ollama create fuse-ai -f /workspace/configs/ollama_model.yaml
    else
        echo "Warning: ollama_model.yaml not found, using default model configuration"
        ollama create fuse-ai -f - <<EOF
FROM FuseAI/FuseO1-DeepSeekR1-Qwen2.5-Coder-32B-Preview

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 8192
PARAMETER num_thread 8
EOF
    fi
fi

# Set up environment variables for data pipeline
if [ -f "/workspace/configs/api_keys.env" ]; then
    echo "Loading API keys from config file..."
    source /workspace/configs/api_keys.env
else
    echo "Warning: API keys config file not found. Data pipeline features may be limited."
fi

# Print environment information
echo "Environment initialized successfully!"
echo "Available services:"
echo "- MLflow: http://localhost:5000"
echo "- KubeFlow: http://localhost:8000"
echo "- Jupyter Lab: http://localhost:8888"
echo "- Ollama: http://localhost:11434"
echo "- LangFuse: http://localhost:3000"
echo "- TensorBoard: http://localhost:6006"
echo ""
echo "Data Pipeline: /workspace/data_pipeline"
echo "Model Training: /workspace/model_training"
echo "Datasets: /workspace/datasets"
echo "Models: /workspace/models"
if [ ! -z "${NETWORK_STORAGE_ROOT}" ]; then
    echo "Network Storage: ${NETWORK_STORAGE_ROOT}"
fi
if [ ! -z "${EXTERNAL_HOST}" ]; then
    echo "External Ollama Access: http://${EXTERNAL_HOST}:${EXTERNAL_PORT}"
fi
echo ""
echo "Model Management Commands:"
echo "- Sync models: /workspace/scripts/manage_model.sh sync"
echo "- Update model: /workspace/scripts/manage_model.sh update"
echo "- Show status: /workspace/scripts/manage_model.sh status"

# Keep the container running with proper signal handling
exec /lib/systemd/systemd --system