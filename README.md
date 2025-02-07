# RunPod ML Environment for Terraform Optimization

A comprehensive ML environment for optimizing Terraform configurations using advanced language models and MLOps tools.

## Architecture

```
[MLflow] ←→ [KubeFlow] ←→ [LangFuse]
   ↑            ↑             ↑
   └────────────┴─────────────┘
          ↓             ↓
[Model Training] ←→ [Inference]
```

## Features

- **Multi-Target Optimization**
  - Cost optimization with performance constraints
  - Security improvements with compliance metrics
  - Performance optimization with availability guarantees

- **Advanced ML Pipeline**
  - Distributed training on A40 GPU
  - Multi-task learning
  - Continuous fine-tuning
  - Automated PR generation

- **Comprehensive Monitoring**
  - MLflow experiment tracking
  - KubeFlow pipeline orchestration
  - LangFuse inference monitoring

- **Network Storage Integration**
  - Persistent model storage
  - Dataset management
  - Experiment tracking
  - Checkpoint management

## Prerequisites

- RunPod.io account with A40 GPU access
- Network storage volume mounted at `/network`
- Git repository access

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/runpod-env.git
cd runpod-env
```

2. Configure environment:
```bash
cp configs/api_keys.env.template configs/api_keys.env
# Edit api_keys.env with your credentials
```

3. Start the environment:
```bash
./scripts/start.sh
```

4. Access interfaces:
- MLflow UI: http://localhost:5000
- KubeFlow UI: http://localhost:8000
- LangFuse UI: http://localhost:3000
- JupyterHub: http://localhost:8888

## Directory Structure

```
/network/mlops/
├── datasets/          # Training datasets
├── models/           # Model files
│   ├── base/         # Base models
│   ├── fine_tuned/   # Fine-tuned models
│   └── checkpoints/  # Training checkpoints
├── experiments/      # Experiment results
│   ├── cost/        # Cost optimization
│   ├── security/    # Security optimization
│   └── performance/ # Performance optimization
└── logs/            # Service logs

/workspace/
├── configs/         # Service configurations
├── notebooks/       # Jupyter notebooks
├── scripts/        # Utility scripts
└── model_training/ # Training code
```

## Components

1. **MLflow**
   - Experiment tracking
   - Model versioning
   - Parameter optimization
   - Metric visualization

2. **KubeFlow**
   - Pipeline orchestration
   - Resource management
   - Distributed training
   - Model serving

3. **LangFuse**
   - Inference monitoring
   - Cost tracking
   - Quality metrics
   - Performance analysis

4. **Base Model**
   - nuibang/Cline_FuseO1-DeepSeekR1-Qwen2.5-Coder-32B-Preview
   - Optimized for code generation
   - Fine-tuned on Terraform configurations

## Usage

1. **Start Environment**
```bash
./scripts/start.sh
```

2. **Access Unified Notebook**
- Open JupyterHub: http://localhost:8888
- Navigate to: `notebooks/terraform_optimization_unified.ipynb`

3. **Monitor Experiments**
- Track experiments in MLflow
- Monitor pipelines in KubeFlow
- Analyze inference in LangFuse

4. **View Results**
- Generated PRs in GitHub
- Optimization metrics in MLflow
- Cost analysis in LangFuse

## Development

1. **Code Organization**
- Use network storage for persistence
- Keep configurations in version control
- Follow MLOps best practices

2. **Adding Features**
- Create new experiment types in `notebooks/`
- Add monitoring metrics in `model_training/`
- Update pipeline components in KubeFlow

3. **Maintenance**
- Regular model updates via `manage_model.sh`
- Log rotation and cleanup
- Metric aggregation and analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see LICENSE file for details