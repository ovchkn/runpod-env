import os
import time
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import mlflow
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ModelUpdateHandler(FileSystemEventHandler):
    def __init__(
        self,
        serving_dir: str,
        ollama_model_name: str,
        modelfile_template: str = "/workspace/configs/ollama_model.yaml"
    ):
        self.serving_dir = Path(serving_dir)
        self.ollama_model_name = ollama_model_name
        self.modelfile_template = Path(modelfile_template)
        self.current_version = None
        
        # Ensure Ollama is running
        self._ensure_ollama_running()

    def on_modified(self, event):
        if not event.is_directory:
            path = Path(event.src_path)
            if path.name == "latest":
                self._handle_model_update()

    def _handle_model_update(self):
        """Handle new model version update"""
        latest_link = self.serving_dir / "latest"
        if not latest_link.exists():
            return

        # Get version number
        version_dir = latest_link.resolve()
        version_num = int(version_dir.name.split("_")[1])
        
        # Skip if we've already processed this version
        if version_num == self.current_version:
            return
            
        print(f"Detected new model version: {version_num}")
        
        try:
            # Convert model to Ollama format
            self._convert_to_ollama(version_dir)
            
            # Update current version
            self.current_version = version_num
            
            # Log to MLflow
            mlflow.log_metric("serving_version", version_num)
            
            print(f"Successfully updated to version {version_num}")
            
        except Exception as e:
            print(f"Error updating model: {str(e)}")

    def _convert_to_ollama(self, model_dir: Path):
        """Convert HuggingFace model to Ollama format"""
        # Create temporary Modelfile
        modelfile = self._create_modelfile(model_dir)
        
        try:
            # Create new Ollama model
            subprocess.run(
                ["ollama", "create", self.ollama_model_name, "-f", modelfile],
                check=True
            )
            
            print(f"Created new Ollama model: {self.ollama_model_name}")
            
        finally:
            # Cleanup temporary file
            if modelfile.exists():
                modelfile.unlink()

    def _create_modelfile(self, model_dir: Path) -> Path:
        """Create Ollama Modelfile from template"""
        # Read template
        template = self.modelfile_template.read_text()
        
        # Update model path
        modelfile_content = template.replace(
            "FROM FuseAI/FuseO1-DeepSeekR1-Qwen2.5-Coder-32B-Preview",
            f"FROM {str(model_dir)}"
        )
        
        # Write temporary Modelfile
        modelfile = model_dir / "Modelfile"
        modelfile.write_text(modelfile_content)
        
        return modelfile

    def _ensure_ollama_running(self):
        """Ensure Ollama service is running"""
        try:
            subprocess.run(["systemctl", "is-active", "--quiet", "ollama"])
        except subprocess.CalledProcessError:
            print("Starting Ollama service...")
            subprocess.run(["systemctl", "start", "ollama"], check=True)

class ModelServer:
    def __init__(
        self,
        network_path: str = "/network/mlops/models/fine_tuned",
        ollama_model_name: str = "cline-fused",
        modelfile_template: str = "/workspace/configs/ollama_model.yaml"
    ):
        self.network_path = Path(network_path)
        self.serving_dir = self.network_path / "serving"
        self.ollama_model_name = ollama_model_name
        self.modelfile_template = modelfile_template
        
        # Create serving directory
        self.serving_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("model_serving")

    def start(self):
        """Start model serving"""
        print(f"Starting model server watching {self.serving_dir}")
        
        # Create event handler
        handler = ModelUpdateHandler(
            serving_dir=self.serving_dir,
            ollama_model_name=self.ollama_model_name,
            modelfile_template=self.modelfile_template
        )
        
        # Create observer
        observer = Observer()
        observer.schedule(handler, str(self.serving_dir), recursive=False)
        observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

if __name__ == "__main__":
    # Start model server
    server = ModelServer()
    server.start()