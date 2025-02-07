import os
from typing import Dict, Optional
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from transformers import AutoModelForCausalLM, AutoTokenizer
import mlflow
from pathlib import Path
import shutil
import threading
import time

class ContinuousFineTuningModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 2e-5,
        network_path: str = "/network/mlops/models/fine_tuned",
        checkpoint_interval: int = 100,
    ):
        super().__init__()
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.network_path = network_path
        self.checkpoint_interval = checkpoint_interval
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Setup model serving directory
        self.serving_dir = Path(network_path) / "serving"
        self.serving_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model version
        self.current_version = 0
        self.last_checkpoint_step = 0
        
        # Start model update watcher
        self.start_model_watcher()

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        self.log("train_loss", loss)
        
        # Check if we should save a new version
        global_step = self.global_step
        if global_step - self.last_checkpoint_step >= self.checkpoint_interval:
            self.save_model_version()
            self.last_checkpoint_step = global_step
        
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def save_model_version(self):
        """Save current model state as a new version"""
        self.current_version += 1
        version_dir = self.serving_dir / f"version_{self.current_version}"
        
        # Save to temporary directory first
        temp_dir = version_dir.with_suffix(".tmp")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(temp_dir)
        self.tokenizer.save_pretrained(temp_dir)
        
        # Atomic rename
        if version_dir.exists():
            shutil.rmtree(version_dir)
        temp_dir.rename(version_dir)
        
        # Update latest symlink
        latest_link = self.serving_dir / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(version_dir)
        
        # Log to MLflow
        mlflow.log_param("model_version", self.current_version)
        mlflow.log_artifact(str(version_dir))

    def start_model_watcher(self):
        """Start background thread to watch for model updates"""
        def watch_model_updates():
            while True:
                # Check for new version
                latest_link = self.serving_dir / "latest"
                if latest_link.exists():
                    latest_version = latest_link.resolve().name
                    version_num = int(latest_version.split("_")[1])
                    
                    # Log version info
                    mlflow.log_metric("serving_version", version_num)
                
                time.sleep(10)  # Check every 10 seconds
        
        watcher = threading.Thread(target=watch_model_updates, daemon=True)
        watcher.start()

def setup_training(
    model_name: str,
    network_path: str,
    experiment_name: str,
    training_args: Dict,
):
    """Setup continuous fine-tuning with MLflow tracking"""
    
    # Initialize MLflow
    mlflow.set_experiment(experiment_name)
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri="http://localhost:5000"
    )
    
    # Create model module
    model = ContinuousFineTuningModule(
        model_name=model_name,
        network_path=network_path,
        **training_args
    )
    
    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=network_path,
        filename="{epoch}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss"
    )
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=training_args.get("max_epochs", 10),
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        precision=16,  # Use mixed precision
        accumulate_grad_batches=training_args.get("grad_accum", 4),
    )
    
    return trainer, model

def load_latest_model(network_path: str) -> Optional[AutoModelForCausalLM]:
    """Load the latest model version for serving"""
    serving_dir = Path(network_path) / "serving" / "latest"
    if serving_dir.exists():
        return AutoModelForCausalLM.from_pretrained(str(serving_dir))
    return None

# Example usage:
if __name__ == "__main__":
    # Setup paths
    NETWORK_PATH = "/network/mlops/models/fine_tuned"
    MODEL_NAME = "nuibang/Cline_FuseO1-DeepSeekR1-Qwen2.5-Coder-32B-Preview"
    
    # Training arguments
    training_args = {
        "learning_rate": 2e-5,
        "max_epochs": 100,
        "checkpoint_interval": 100,
        "grad_accum": 4
    }
    
    # Initialize training
    trainer, model = setup_training(
        model_name=MODEL_NAME,
        network_path=NETWORK_PATH,
        experiment_name="continuous_finetuning",
        training_args=training_args
    )
    
    # Start training
    trainer.fit(model)