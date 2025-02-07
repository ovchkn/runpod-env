# RunPod AI/ML Development Environment

A comprehensive ML/AI development environment with sysbox-runc support, designed for running Ollama models, conducting ML experiments, and fine-tuning models. This environment integrates MLflow, KubeFlow, JupyterHub, and various ML tools in a containerized setup.

## Features

### Core Infrastructure
- **Sysbox Runtime Integration**: Secure nested container support for advanced containerization
- **Systemd Integration**: Proper service management and process control
- **Docker Support**: Full Docker support with sysbox-runc runtime

### ML/AI Framework Support
- PyTorch with CUDA support
- PyTorch Lightning
- Transformers & HuggingFace
- LangChain & LangFuse
- MLflow & KubeFlow
- Sentence Transformers
- Scikit-learn & NumPy

### Development Tools
- JupyterHub with Lab interface
- TensorBoard
- Git integration
- VSCode compatibility

### Model Management
- Ollama integration
- HuggingFace Hub support
- Model versioning
- Experiment tracking

### Data Pipeline
- GitHub repository scraping
- Tavily search integration
- Custom dataset generation
- Embedding creation
- MLflow tracking

### Model Training
- Fine-tuning pipeline
- Experiment tracking
- Model checkpointing
- Performance monitoring

## Prerequisites

- Docker with sysbox-runc support
- NVIDIA GPU with appropriate drivers
- At least 32GB RAM recommended
- 100GB+ storage space
- API keys for data collection (GitHub, Tavily)

## Quick Start

1. Build the container:
```bash
docker build -t ghcr.io/ovchkn/runpod:1.0 .
```

2. Set up API keys:
```bash
cd /workspace/configs
cp api_keys.env.template api_keys.env
# Edit api_keys.env with your API keys
```

3. Run the container:
```bash
docker run -d \
  --runtime=sysbox-runc \
  --name runpod-env \
  -p 11434:11434 \
  -p 5000:5000 \
  -p 8000:8000 \
  -p 8888:8888 \
  -p 3000:3000 \
  -p 6006:6006 \
  --gpus all \
  ghcr.io/ovchkn/runpod:1.0
```

4. Access services:
- JupyterHub: http://localhost:8888
- MLflow: http://localhost:5000
- KubeFlow: http://localhost:8000
- Ollama: http://localhost:11434
- LangFuse: http://localhost:3000
- TensorBoard: http://localhost:6006

## Environment Structure

```
/workspace/
├── data_pipeline/       # Data collection and processing
│   ├── scraper.py      # GitHub and Tavily data collection
│   ├── raw_data/       # Raw collected data
│   └── processed_data/ # Processed datasets
├── model_training/     # Model training and fine-tuning
│   ├── fine_tuning.py # Fine-tuning pipeline
│   ├── checkpoints/   # Model checkpoints
│   └── artifacts/     # Training artifacts
├── notebooks/         # Jupyter notebooks
├── models/           # Model files and checkpoints
├── datasets/         # Training and test datasets
├── pipelines/        # ML pipelines and workflows
├── experiments/      # Experiment tracking
├── configs/         # Configuration files
└── logs/           # Service and application logs
```

## Data Pipeline Usage

1. Configure API keys:
```bash
cd /workspace/configs
cp api_keys.env.template api_keys.env
# Edit api_keys.env with your keys
```

2. Run data collection:
```bash
python /workspace/data_pipeline/scraper.py
```

The pipeline will:
- Search GitHub for relevant repositories
- Extract pull request data
- Perform Tavily searches
- Generate embeddings
- Save processed datasets

## Model Training

1. Prepare your dataset:
```bash
# Datasets should be in /workspace/datasets/training
```

2. Run fine-tuning:
```bash
python /workspace/model_training/fine_tuning.py
```

Training progress can be monitored via:
- MLflow UI: http://localhost:5000
- TensorBoard: http://localhost:6006

## Service Management

The environment uses systemd for service management:

```bash
# Start all services
/workspace/scripts/start.sh

# Individual service control
systemctl status mlflow
systemctl status ollama
systemctl status jupyterhub
```

## Ollama Model Management

```bash
# List available models
ollama list

# Pull additional models
ollama pull <model-name>

# Run model
ollama run fuse-ai
```

## MLflow Experiment Tracking

```python
import mlflow

# Start experiment
mlflow.set_experiment("my-experiment")
with mlflow.start_run():
    # Your training code here
    mlflow.log_param("param1", value1)
    mlflow.log_metric("metric1", value1)
```

## Development Workflow

1. Start JupyterLab and create a new notebook
2. Use the pre-configured `mlops` conda environment
3. Import required libraries (all major ML libraries are pre-installed)
4. Connect to MLflow for experiment tracking
5. Use Ollama for model inference
6. Monitor training with TensorBoard

## Customization

- Edit `/workspace/configs/jupyterhub_config.py` for JupyterHub settings
- Modify service configurations in `/workspace/configs/services/`
- Add custom conda packages to the `mlops` environment
- Configure model parameters in `configs/ollama_model.yaml`

## Troubleshooting

1. Service Issues:
```bash
# Check service status
systemctl status <service-name>

# View service logs
journalctl -u <service-name>
```

2. Container Runtime:
```bash
# Check sysbox-runc status
systemctl status sysbox
```

3. GPU Access:
```bash
# Verify GPU visibility
nvidia-smi
```

4. Data Pipeline Issues:
```bash
# Check API keys
cat /workspace/configs/api_keys.env

# Check logs
tail -f /workspace/logs/data_pipeline.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

- GitHub Issues: [Report bugs or suggest features](https://github.com/ovchkn/runpod-env/issues)
- Documentation: See `/docs` directory for detailed guides