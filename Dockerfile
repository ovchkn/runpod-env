# Start with Ubuntu 24.04 (Noble)
FROM ubuntu:noble

LABEL org.opencontainers.image.source="https://github.com/ovchkn/runpod-env"
LABEL org.opencontainers.image.description="Comprehensive ML/AI Development Environment for RunPod"
LABEL org.opencontainers.image.licenses="MIT"
LABEL maintainer="ovchkn"
LABEL version="1.0"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

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
    libgl1-mesa-glx \
    kubectl \
    docker.io \
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
    rm -f /lib/systemd/system/anaconda.target.wants/*

# Create workspace structure first
RUN mkdir -p /workspace/{models,data,experiments,logs,notebooks,scripts,configs} && \
    mkdir -p /workspace/mlflow/{artifacts,runs,metrics} && \
    mkdir -p /workspace/kubeflow/{pipelines,components,artifacts} && \
    chmod -R 777 /workspace

# Install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    bash /tmp/anaconda.sh -b -p /opt/conda && \
    rm /tmp/anaconda.sh

# Add conda to path and create environment
ENV PATH="/opt/conda/bin:$PATH"
RUN conda create -n mlops python=3.11 anaconda -y && \
    conda init bash && \
    echo "conda activate mlops" >> ~/.bashrc

# Install comprehensive ML/AI/DL/RL toolset
RUN /bin/bash -c "source activate mlops && \
    # PyTorch Ecosystem
    conda install -y -c pytorch -c nvidia \
    pytorch \
    torchvision \
    torchaudio \
    pytorch-cuda=12.1 \
    pytorch-lightning \
    torchtext \
    torchmetrics \
    && \
    # Deep Learning
    conda install -y -c conda-forge \
    tensorflow \
    keras \
    jax \
    jaxlib \
    flax \
    transformers \
    datasets \
    diffusers \
    accelerate \
    optimum \
    huggingface_hub \
    && \
    # ML Tools and Libraries
    conda install -y -c conda-forge \
    scikit-learn \
    xgboost \
    lightgbm \
    catboost \
    rapids \
    cudf \
    cuml \
    cugraph \
    cusignal \
    optuna \
    && \
    # Reinforcement Learning
    conda install -y -c conda-forge \
    gym \
    stable-baselines3 \
    mujoco \
    box2d-py \
    && \
    # Visualization and Notebooks
    conda install -y -c conda-forge \
    jupyterlab \
    jupyter-dash \
    ipywidgets \
    plotly \
    bokeh \
    holoviews \
    hvplot \
    altair \
    seaborn \
    && \
    # Data Processing
    conda install -y -c conda-forge \
    pandas \
    polars \
    modin \
    dask \
    vaex \
    fastparquet \
    pyarrow \
    && \
    # ML Ops
    conda install -y -c conda-forge \
    mlflow \
    ray-default \
    wandb \
    tensorboard \
    && \
    # NLP
    conda install -y -c conda-forge \
    spacy \
    nltk \
    gensim \
    textblob \
    fastai \
    && \
    # Computer Vision
    conda install -y -c conda-forge \
    opencv \
    albumentations \
    imgaug \
    && \
    # Additional Tools
    pip install \
    langfuse \
    kubeflow-pipelines \
    gradio \
    streamlit \
    fastapi \
    uvicorn \
    ray[tune] \
    ray[rllib] \
    ray[serve] \
    deepspeed \
    accelerate \
    bitsandbytes \
    vllm \
    einops \
    kfp \
    kfserving \
    kubeflow-training-operator \
    kubeflow-katib \
    kubeflow-kale \
    kubeflow-metadata \
    kubeflow-pytorch-operator \
    kubeflow-tensorflow-operator \
    mlflow[extras] \
    mlflow-skinny \
    mlflow-dbstore \
    azure-mlflow-skinny \
    mlflow-flexattr \
    && \
    # Download models and data
    python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_md && \
    python -c 'import nltk; nltk.download(\"popular\")'"

# Install Node.js and Ollama
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g configurable-http-proxy && \
    curl -fsSL https://ollama.com/install.sh | sh

# Create default config files
RUN echo '[server]\nhost = 0.0.0.0\nport = 5000\nworkers = 4' > /workspace/configs/mlflow.conf && \
    echo 'apiVersion: v1\nkind: Config\ncurrent-context: kubeflow' > /workspace/configs/kubeflow.conf

# Set up environment variables
ENV PYTHONPATH=/workspace \
    OLLAMA_HOST=0.0.0.0:11434 \
    MLFLOW_TRACKING_URI=http://localhost:5000 \
    MLFLOW_ARTIFACT_ROOT=/workspace/mlflow/artifacts \
    KUBEFLOW_HOST=http://localhost:8000 \
    KF_PIPELINES_ENDPOINT=http://localhost:8000/pipeline \
    KFSERVING_ENDPOINT=http://localhost:8000/kfserving \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
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