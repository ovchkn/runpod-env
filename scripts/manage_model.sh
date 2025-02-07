#!/bin/bash

# Enhanced model management script for Ollama models with network storage and external access

# Default paths
NETWORK_ROOT=${NETWORK_STORAGE_ROOT:-"/network/mlops"}
NETWORK_MODELS="${NETWORK_ROOT}/models"
LOCAL_CACHE="/workspace/cache/ollama"
FINE_TUNED_DIR="/workspace/models/fine_tuned"

# Model names
BASE_MODEL="nuibang/Cline_FuseO1-DeepSeekR1-Qwen2.5-Coder-32B-Preview:q8_0"
FINETUNED_MODEL="fuse-ai-gitops:latest"

# External access configuration
EXTERNAL_PORT=${EXTERNAL_PORT:-11434}
EXTERNAL_HOST=${EXTERNAL_HOST:-"0.0.0.0"}
EXTERNAL_ACCESS_TOKEN=${EXTERNAL_ACCESS_TOKEN:-""}

# Function to check if Ollama is running
check_ollama() {
    if ! pgrep -x "ollama" > /dev/null; then
        echo "Starting Ollama service..."
        systemctl start ollama
        sleep 5
    fi
}

# Function to configure external access
configure_external_access() {
    echo "Configuring external access..."
    
    # Update Ollama service configuration
    cat > /etc/systemd/system/ollama.service <<EOF
[Unit]
Description=Ollama Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
Environment="OLLAMA_HOST=${EXTERNAL_HOST}"
Environment="OLLAMA_PORT=${EXTERNAL_PORT}"
Environment="OLLAMA_MODELS=${NETWORK_MODELS}/ollama"
Environment="OLLAMA_ORIGINS=*"
ExecStart=/usr/bin/ollama serve
WorkingDirectory=/workspace/services/ollama
Restart=always
RuntimeDirectory=ollama
RuntimeDirectoryMode=0755

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and restart Ollama
    systemctl daemon-reload
    systemctl restart ollama
    
    echo "External access configured at ${EXTERNAL_HOST}:${EXTERNAL_PORT}"
}

# Function to save model state
save_model() {
    local model_name=$1
    local output_path=$2

    echo "Saving model ${model_name} to ${output_path}..."
    if ollama save "${model_name}" "${output_path}"; then
        echo "Model saved successfully"
        return 0
    else
        echo "Failed to save model"
        return 1
    fi
}

# Function to load model
load_model() {
    local model_path=$1

    echo "Loading model from ${model_path}..."
    if ollama load "${model_path}"; then
        echo "Model loaded successfully"
        return 0
    else
        echo "Failed to load model"
        return 1
    fi
}

# Function to switch active model
switch_model() {
    local model_name=$1

    echo "Switching to model ${model_name}..."
    
    # Stop any running models
    ollama list | grep -v "REPOSITORY" | awk '{print $1}' | xargs -r ollama stop
    
    # Start new model
    if ollama run "${model_name}" &> /dev/null & then
        echo "Model ${model_name} is now active"
        # Save current model name
        echo "${model_name}" > "${LOCAL_CACHE}/current_model"
        return 0
    else
        echo "Failed to switch to model ${model_name}"
        return 1
    fi
}

# Function to sync with network storage
sync_models() {
    echo "Syncing models with network storage..."
    
    # Create network storage directories if they don't exist
    mkdir -p "${NETWORK_MODELS}/ollama"
    mkdir -p "${NETWORK_MODELS}/fine_tuned"
    
    # Sync fine-tuned models to network storage
    if [ -d "${FINE_TUNED_DIR}" ]; then
        echo "Syncing fine-tuned models..."
        rsync -av "${FINE_TUNED_DIR}/" "${NETWORK_MODELS}/fine_tuned/"
    fi
    
    # Load models from network storage if available
    if [ -d "${NETWORK_MODELS}/ollama" ]; then
        echo "Loading models from network storage..."
        for model in "${NETWORK_MODELS}/ollama"/*; do
            if [ -f "${model}" ]; then
                load_model "${model}"
            fi
        done
    fi
}

# Function to update model after fine-tuning
update_finetuned_model() {
    echo "Updating fine-tuned model..."
    
    # Check if new fine-tuned model exists
    if [ -d "${FINE_TUNED_DIR}/latest" ]; then
        # Save to network storage
        save_model "${FINETUNED_MODEL}" "${NETWORK_MODELS}/fine_tuned/latest"
        
        # Update Ollama model
        ollama create "${FINETUNED_MODEL}" -f "${FINE_TUNED_DIR}/latest/Modelfile"
        
        echo "Fine-tuned model updated successfully"
        return 0
    else
        echo "No new fine-tuned model found"
        return 1
    fi
}

# Function to show model status
show_status() {
    echo "Current Model Status:"
    echo "-------------------"
    
    # Show running models
    echo "Running Models:"
    ollama list
    
    # Show current model
    if [ -f "${LOCAL_CACHE}/current_model" ]; then
        echo -e "\nActive Model:"
        cat "${LOCAL_CACHE}/current_model"
    fi
    
    # Show available models in network storage
    if [ -d "${NETWORK_MODELS}/ollama" ]; then
        echo -e "\nNetwork Storage Models:"
        ls -l "${NETWORK_MODELS}/ollama"
    fi
    
    # Show fine-tuned models
    if [ -d "${NETWORK_MODELS}/fine_tuned" ]; then
        echo -e "\nFine-tuned Models:"
        ls -l "${NETWORK_MODELS}/fine_tuned"
    fi
    
    # Show external access configuration
    echo -e "\nExternal Access:"
    echo "Host: ${EXTERNAL_HOST}"
    echo "Port: ${EXTERNAL_PORT}"
}

# Main script logic
case "$1" in
    "save")
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Usage: $0 save <model_name> <output_path>"
            exit 1
        fi
        check_ollama
        save_model "$2" "$3"
        ;;
    
    "load")
        if [ -z "$2" ]; then
            echo "Usage: $0 load <model_path>"
            exit 1
        fi
        check_ollama
        load_model "$2"
        ;;
    
    "switch")
        if [ -z "$2" ]; then
            echo "Usage: $0 switch <model_name>"
            exit 1
        fi
        check_ollama
        switch_model "$2"
        ;;
    
    "sync")
        check_ollama
        sync_models
        ;;
    
    "update")
        check_ollama
        update_finetuned_model
        ;;
    
    "configure-external")
        if [ ! -z "$2" ]; then
            EXTERNAL_HOST="$2"
        fi
        if [ ! -z "$3" ]; then
            EXTERNAL_PORT="$3"
        fi
        configure_external_access
        ;;
    
    "status")
        check_ollama
        show_status
        ;;
    
    "backup")
        check_ollama
        if [ -d "${NETWORK_MODELS}/ollama" ]; then
            echo "Backing up current models to network storage..."
            current_model=$(cat "${LOCAL_CACHE}/current_model" 2>/dev/null)
            if [ ! -z "${current_model}" ]; then
                save_model "${current_model}" "${NETWORK_MODELS}/ollama/${current_model}.backup"
            fi
        else
            echo "Network storage not available"
            exit 1
        fi
        ;;
    
    *)
        echo "Usage: $0 {save|load|switch|sync|update|configure-external|status|backup}"
        echo "Examples:"
        echo "  $0 save ${FINETUNED_MODEL} /path/to/save"
        echo "  $0 load /path/to/model"
        echo "  $0 switch ${BASE_MODEL}"
        echo "  $0 sync                               # Sync with network storage"
        echo "  $0 update                            # Update after fine-tuning"
        echo "  $0 configure-external [host] [port]  # Configure external access"
        echo "  $0 status"
        echo "  $0 backup"
        exit 1
        ;;
esac

exit 0