# Start with PyTorch CUDA base image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL org.opencontainers.image.source="https://github.com/ovchkn/runpod-env"
LABEL org.opencontainers.image.description="ML/AI Development Environment with Sysbox Runtime for RunPod"
LABEL org.opencontainers.image.licenses="MIT"
LABEL maintainer="ovchkn"
LABEL version="1.0"
LABEL io.runpod.runtime="sysbox-runc"

# Initialize environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    MLFLOW_TRACKING_URI=http://localhost:5000 \
    PYTHONPATH=/workspace \
    NETWORK_STORAGE_ROOT=/network/mlops \
    EXTERNAL_HOST=0.0.0.0 \
    EXTERNAL_PORT=11434

# Add repositories and initial packages
RUN apt-get update && apt-get install -y \
    curl \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add Docker repository
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list

# Add Kubernetes repository
RUN curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.29/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg && \
    echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /' | tee /etc/apt/sources.list.d/kubernetes.list

# Install system packages in groups for better error handling
RUN apt-get update && apt-get install -y \
    systemd \
    systemd-sysv \
    dbus \
    udev \
    init \
    sudo \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# Install development tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    wget \
    pciutils \
    lshw \
    && rm -rf /var/lib/apt/lists/*

# Install media and graphics libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Docker and Kubernetes
RUN apt-get update && apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    kubectl \
    && rm -rf /var/lib/apt/lists/*

# Install sysbox-runc from GitHub release
RUN mkdir -p /tmp/sysbox && \
    cd /tmp/sysbox && \
    wget https://downloads.nestybox.com/sysbox/releases/v0.6.2/sysbox-ce_0.6.2-0.linux_amd64.deb && \
    apt-get update && \
    apt-get install -y ./sysbox-ce_0.6.2-0.linux_amd64.deb && \
    rm -rf /tmp/sysbox && \
    rm -rf /var/lib/apt/lists/*

# Configure Docker to use sysbox-runc by default
RUN mkdir -p /etc/docker && \
    echo '{\n  "default-runtime": "sysbox-runc",\n  "runtimes": {\n    "sysbox-runc": {\n      "path": "/usr/bin/sysbox-runc"\n    }\n  }\n}' > /etc/docker/daemon.json

# Configure systemd
RUN cd /lib/systemd/system/sysinit.target.wants/ && \
    ls | grep -v systemd-tmpfiles-setup | xargs rm -f $1 && \
    rm -f /lib/systemd/system/multi-user.target.wants/* && \
    rm -f /etc/systemd/system/*.wants/* && \
    rm -f /lib/systemd/system/local-fs.target.wants/* && \
    rm -f /lib/systemd/system/sockets.target.wants/*udev* && \
    rm -f /lib/systemd/system/sockets.target.wants/*initctl* && \
    rm -f /lib/systemd/system/basic.target.wants/* && \
    rm -f /lib/systemd/system/anaconda.target.wants/

# Set up conda environment
ENV PATH=/opt/conda/bin:$PATH
RUN conda update -n base -c defaults conda && \
    conda create -n mlops python=3.10 anaconda -y && \
    conda init bash && \
    echo "conda activate mlops" >> ~/.bashrc

# Install additional ML/AI packages
RUN /opt/conda/envs/mlops/bin/pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    pytorch-lightning \
    transformers \
    datasets \
    langchain \
    langfuse \
    huggingface_hub \
    mlflow \
    kfp \
    jupyterhub \
    jupyterlab \
    wandb \
    optuna \
    ray[tune] \
    tensorboard \
    beautifulsoup4 \
    scrapy \
    selenium \
    tavily-python \
    sentence-transformers \
    PyGithub \
    requests \
    parquet \
    fastparquet \
    pyarrow

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create workspace structure
RUN mkdir -p /workspace/{notebooks,models,datasets,pipelines,experiments,scripts,configs,logs} && \
    mkdir -p /workspace/services/{mlflow,kubeflow,langfuse,ollama} && \
    mkdir -p /workspace/data_pipeline && \
    mkdir -p /workspace/model_training && \
    mkdir -p /workspace/datasets/training && \
    mkdir -p /workspace/cache/ollama

# Create network storage mount point
RUN mkdir -p /network/mlops/models/{ollama,fine_tuned}

# Copy pipeline files
COPY data_pipeline/ /workspace/data_pipeline/
COPY model_training/ /workspace/model_training/

# Copy configuration files
COPY configs/jupyterhub_config.py /workspace/configs/
COPY configs/services/mlflow.service /etc/systemd/system/
COPY configs/services/ollama.service /etc/systemd/system/
COPY configs/services/jupyterhub.service /etc/systemd/system/
COPY configs/api_keys.env.template /workspace/configs/

# Copy scripts
COPY scripts/start.sh /workspace/scripts/
COPY scripts/manage_model.sh /workspace/scripts/
COPY scripts/deploy.sh /workspace/scripts/

# Set proper permissions
RUN chmod 644 /etc/systemd/system/*.service && \
    chmod 644 /workspace/configs/jupyterhub_config.py && \
    chmod 644 /workspace/data_pipeline/*.py && \
    chmod 644 /workspace/model_training/*.py && \
    chmod +x /workspace/scripts/*.sh && \
    systemctl enable mlflow.service && \
    systemctl enable ollama.service && \
    systemctl enable jupyterhub.service

# Create welcome message
RUN echo 'echo "Welcome to RunPod AI/ML Environment\n\
- Conda environment: mlops (auto-activated)\n\
- Full Anaconda Distribution\n\
- PyTorch + CUDA + Lightning support\n\
- MLflow at http://localhost:5000\n\
- KubeFlow at http://localhost:8000\n\
- Jupyter Lab at http://localhost:8888\n\
- Ollama at http://localhost:11434\n\
- LangFuse at http://localhost:3000\n\
- TensorBoard at http://localhost:6006\n\
\n\
Container Runtime: sysbox-runc (Default)\n\
Network Storage: ${NETWORK_STORAGE_ROOT}\n\
External Ollama: http://${EXTERNAL_HOST}:${EXTERNAL_PORT}\n\
Run /workspace/scripts/start.sh to initialize services\n"' >> ~/.bashrc

# Set working directory
WORKDIR /workspace

# Expose ports
EXPOSE 11434 5000 8000 8888 3000 6006

# Set entrypoint to systemd
ENTRYPOINT ["/lib/systemd/systemd"]