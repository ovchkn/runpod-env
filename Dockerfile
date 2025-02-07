# Start with Ubuntu 24.04 (Noble)
FROM ubuntu:noble

LABEL org.opencontainers.image.source="https://github.com/ovchkn/runpod-env"
LABEL org.opencontainers.image.description="Comprehensive ML/AI Development Environment for RunPod"
LABEL org.opencontainers.image.licenses="MIT"
LABEL maintainer="ovchkn"
LABEL version="1.0"

# Initialize environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

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

# Add Kubernetes repository (using latest repository)
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