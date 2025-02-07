# Start with Ubuntu 24.04 (Noble)
FROM ubuntu:noble

LABEL org.opencontainers.image.source="https://github.com/ovchkn/runpod-env"
LABEL org.opencontainers.image.description="Comprehensive ML/AI Development Environment for RunPod"
LABEL org.opencontainers.image.licenses="MIT"
LABEL maintainer="ovchkn"
LABEL version="1.0"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Add repositories
RUN apt-get update && apt-get install -y \
    curl \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add Kubernetes repository
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /etc/apt/keyrings/kubernetes-archive-keyring.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | tee /etc/apt/sources.list.d/kubernetes.list

# Add Docker repository
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list

# Install systemd and essential packages
RUN apt-get update && apt-get install -y \
    systemd \
    systemd-sysv \
    dbus \
    udev \
    init \
    sudo \
    curl \
    wget \
    git \
    build-essential \
    software-properties-common \
    pciutils \
    lshw \
    nvidia-utils-535 \
    cmake \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    kubectl \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    && rm -rf /var/lib/apt/lists/*

# Initialize environment variables
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create welcome message
RUN echo 'echo "Welcome to RunPod AI/ML Environment\n\
- Conda environment: mlops (auto-activated)\n\
- Full AI/ML/DL/RL Toolset\n\
- PyTorch + CUDA support\n\
- TensorFlow + JAX\n\
- MLflow at http://localhost:5000\n\
- KubeFlow at http://localhost:8000\n\
- Jupyter Lab at http://localhost:8888\n\
- Ollama at http://localhost:11434\n\
\n\
Run /workspace/scripts/start.sh to initialize services\n"' >> ~/.bashrc

# Set up systemd as entrypoint
ENTRYPOINT ["/lib/systemd/systemd"]

# Expose ports
EXPOSE 11434 5000 8000 8888 3000 6006