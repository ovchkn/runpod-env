import os
from typing import Dict, Any, Optional, List
import mlflow
from mlflow.tracking import MlflowClient
from langfuse import Langfuse
from langfuse.model import CreateTrace, CreateGeneration, CreateSpan
import pytorch_lightning as pl
from datetime import datetime

class MonitoringCallback(pl.Callback):
    """
    Combined MLflow and LangFuse monitoring callback for PyTorch Lightning
    """
    def __init__(
        self,
        langfuse_public_key: str,
        langfuse_secret_key: str,
        experiment_name: str = "terraform_optimization"
    ):
        super().__init__()
        
        # Initialize MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(experiment_name)
        self.mlflow_client = MlflowClient()
        
        # Initialize LangFuse
        self.langfuse = Langfuse(
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
            host="http://localhost:3000"
        )
        
        self.current_trace = None
        self.current_generation = None
        self.current_span = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Start MLflow run and LangFuse trace"""
        # Start MLflow run
        mlflow.start_run()
        
        # Log model parameters
        mlflow.log_params({
            "model_name": pl_module.model_name,
            "learning_rate": pl_module.learning_rate,
            "batch_size": trainer.train_dataloader.batch_size,
            "max_epochs": trainer.max_epochs
        })
        
        # Create LangFuse trace
        self.current_trace = self.langfuse.trace(
            name="model_training",
            metadata={
                "model_name": pl_module.model_name,
                "experiment": "terraform_optimization"
            }
        )
        
        # Create training span
        self.current_span = self.current_trace.span(
            name="training_run",
            metadata={
                "max_epochs": trainer.max_epochs,
                "batch_size": trainer.train_dataloader.batch_size
            }
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int
    ):
        """Log batch metrics to both platforms"""
        # Extract metrics
        metrics = {
            k: v.item() if hasattr(v, 'item') else v
            for k, v in outputs.items()
            if isinstance(v, (int, float)) or hasattr(v, 'item')
        }
        
        # Log to MLflow
        mlflow.log_metrics(metrics, step=trainer.global_step)
        
        # Log to LangFuse
        self.current_span.update(
            metadata={
                "step": trainer.global_step,
                "metrics": metrics
            }
        )

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log validation metrics"""
        metrics = trainer.callback_metrics
        
        # Log to MLflow
        mlflow.log_metrics({
            k: v.item() if hasattr(v, 'item') else v
            for k, v in metrics.items()
        })
        
        # Create validation span in LangFuse
        validation_span = self.current_trace.span(
            name="validation",
            metadata={
                "metrics": {
                    k: v.item() if hasattr(v, 'item') else v
                    for k, v in metrics.items()
                }
            }
        )
        validation_span.end()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """End training monitoring"""
        # End MLflow run
        mlflow.end_run()
        
        # End LangFuse spans and trace
        if self.current_span:
            self.current_span.end()
        if self.current_trace:
            self.current_trace.end()

class InferenceMonitor:
    """
    Monitor model inference and optimization suggestions
    """
    def __init__(
        self,
        langfuse_public_key: str,
        langfuse_secret_key: str,
        model_version: str
    ):
        # Initialize LangFuse
        self.langfuse = Langfuse(
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
            host="http://localhost:3000"
        )
        
        self.model_version = model_version
        
        # Get MLflow run info
        self.mlflow_client = MlflowClient()
        self.run = self.mlflow_client.get_run(model_version)

    def start_optimization_trace(
        self,
        resource_type: str,
        optimization_target: str
    ) -> Any:
        """Start tracing an optimization request"""
        return self.langfuse.trace(
            name="terraform_optimization",
            metadata={
                "resource_type": resource_type,
                "optimization_target": optimization_target,
                "model_version": self.model_version
            }
        )

    def log_optimization_generation(
        self,
        trace: Any,
        input_config: str,
        output_config: str,
        metrics: Dict[str, float]
    ):
        """Log an optimization suggestion"""
        # Create generation in LangFuse
        generation = trace.generation(
            name="optimization_suggestion",
            model=self.model_version,
            prompt=input_config,
            completion=output_config,
            metadata={
                "metrics": metrics
            }
        )
        
        # Log metrics to MLflow
        with mlflow.start_run(run_id=self.run.info.run_id):
            mlflow.log_metrics({
                f"inference_{k}": v for k, v in metrics.items()
            })
        
        return generation

    def log_validation_result(
        self,
        trace: Any,
        validation_result: Dict[str, Any]
    ):
        """Log validation results"""
        span = trace.span(
            name="validation",
            metadata=validation_result
        )
        span.end()
        
        # Log to MLflow
        with mlflow.start_run(run_id=self.run.info.run_id):
            mlflow.log_metrics({
                "validation_success": 1 if validation_result["success"] else 0,
                "cost_reduction": validation_result.get("cost_reduction", 0)
            })

def setup_monitoring(config_path: str = "/workspace/configs/api_keys.env") -> Dict[str, Any]:
    """Setup monitoring with keys from config"""
    # Load keys from config
    with open(config_path) as f:
        config = {}
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                config[key] = value
    
    return {
        "langfuse_public_key": config["LANGFUSE_PUBLIC_KEY"],
        "langfuse_secret_key": config["LANGFUSE_SECRET_KEY"]
    }

# Example usage:
if __name__ == "__main__":
    # Setup monitoring
    config = setup_monitoring()
    
    # Create training callback
    callback = MonitoringCallback(
        langfuse_public_key=config["langfuse_public_key"],
        langfuse_secret_key=config["langfuse_secret_key"]
    )
    
    # Create inference monitor
    monitor = InferenceMonitor(
        langfuse_public_key=config["langfuse_public_key"],
        langfuse_secret_key=config["langfuse_secret_key"],
        model_version="latest"
    )
    
    # Example optimization trace
    trace = monitor.start_optimization_trace(
        resource_type="aws_instance",
        optimization_target="cost"
    )
    
    # Log optimization
    monitor.log_optimization_generation(
        trace=trace,
        input_config="original terraform",
        output_config="optimized terraform",
        metrics={"cost_reduction": 0.15}
    )
    
    # Log validation
    monitor.log_validation_result(
        trace=trace,
        validation_result={
            "success": True,
            "cost_reduction": 0.15,
            "performance_impact": 0.02
        }
    )
    
    trace.end()