import os
import logging
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    AdamW, 
    get_linear_schedule_with_warmup
)
import mlflow
import mlflow.pytorch

class CustomDataset(Dataset):
    def __init__(self, 
                 embeddings: torch.Tensor, 
                 labels: torch.Tensor):
        """
        Custom PyTorch Dataset for embeddings
        
        Args:
            embeddings (torch.Tensor): Input embeddings
            labels (torch.Tensor): Corresponding labels
        """
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.embeddings[idx],
            'labels': self.labels[idx]
        }

class ModelFineTuner:
    def __init__(
        self, 
        model_name: str = 'sentence-transformers/all-mpnet-base-v2',
        output_dir: str = '/workspace/models/fine_tuned',
        log_dir: str = '/workspace/logs',
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        num_epochs: int = 5
    ):
        """
        Initialize ModelFineTuner for infrastructure reasoning model
        
        Args:
            model_name (str): Pretrained model to fine-tune
            output_dir (str): Directory to save fine-tuned models
            log_dir (str): Directory for logging
            learning_rate (float): Learning rate for training
            batch_size (int): Training batch size
            num_epochs (int): Number of training epochs
        """
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/fine_tuning.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Model and training configuration
        self.model_name = model_name
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # MLflow tracking
        mlflow.set_experiment('infrastructure_reasoning_fine_tuning')

    def load_dataset(self, dataset_path: str) -> Dict[str, torch.Tensor]:
        """
        Load preprocessed dataset
        
        Args:
            dataset_path (str): Path to PyTorch dataset file
        
        Returns:
            Dictionary with train/test embeddings and labels
        """
        try:
            dataset = torch.load(dataset_path)
            return dataset
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise

    def prepare_dataloaders(self, dataset: Dict[str, torch.Tensor]):
        """
        Prepare PyTorch DataLoaders for training and validation
        
        Args:
            dataset (Dict[str, torch.Tensor]): Dataset dictionary
        
        Returns:
            Tuple of train and validation DataLoaders
        """
        train_dataset = CustomDataset(
            dataset['train_embeddings'], 
            dataset['train_labels']
        )
        
        val_dataset = CustomDataset(
            dataset['test_embeddings'], 
            dataset['test_labels']
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        return train_loader, val_loader

    def create_model(self, num_labels: int = 2):
        """
        Create and prepare model for fine-tuning
        
        Args:
            num_labels (int): Number of classification labels
        
        Returns:
            Prepared model for training
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels
        )
        
        return model

    def create_optimizer_and_scheduler(
        self, 
        model: nn.Module, 
        train_loader: DataLoader
    ):
        """
        Create optimizer and learning rate scheduler
        
        Args:
            model (nn.Module): Model to optimize
            train_loader (DataLoader): Training data loader
        
        Returns:
            Tuple of optimizer and scheduler
        """
        optimizer = AdamW(
            model.parameters(), 
            lr=self.learning_rate
        )
        
        total_steps = len(train_loader) * self.num_epochs
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler

    def train(self, dataset_path: str):
        """
        Execute complete fine-tuning process
        
        Args:
            dataset_path (str): Path to preprocessed dataset
        """
        with mlflow.start_run(run_name='infrastructure_reasoning_fine_tuning'):
            # Load dataset
            dataset = self.load_dataset(dataset_path)
            
            # Prepare data loaders
            train_loader, val_loader = self.prepare_dataloaders(dataset)
            
            # Create model
            model = self.create_model()
            
            # Create optimizer and scheduler
            optimizer, scheduler = self.create_optimizer_and_scheduler(
                model, train_loader
            )
            
            # Training loop
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            for epoch in range(self.num_epochs):
                model.train()
                total_train_loss = 0
                
                for batch in train_loader:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Zero gradients
                    model.zero_grad()
                    
                    # Forward pass
                    outputs = model(
                        input_ids=input_ids, 
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    total_train_loss += loss.item()
                    
                    # Backward pass
                    loss.backward()
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # Update parameters
                    optimizer.step()
                    scheduler.step()
                
                # Validation
                model.eval()
                total_val_loss = 0
                total_val_correct = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(device)
                        labels = batch['labels'].to(device)
                        
                        outputs = model(
                            input_ids=input_ids, 
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        total_val_loss += loss.item()
                        
                        # Compute accuracy
                        preds = torch.argmax(outputs.logits, dim=1)
                        total_val_correct += (preds == labels).sum().item()
                
                # Log metrics
                avg_train_loss = total_train_loss / len(train_loader)
                avg_val_loss = total_val_loss / len(val_loader)
                val_accuracy = total_val_correct / len(val_loader.dataset)
                
                mlflow.log_metrics({
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_accuracy
                }, step=epoch)
            
            # Save fine-tuned model
            model_save_path = os.path.join(
                self.output_dir, 
                f'infrastructure_reasoning_model_v{self.num_epochs}'
            )
            model.save_pretrained(model_save_path)
            
            # Log model artifact
            mlflow.pytorch.log_model(model, 'fine_tuned_model')
            
            self.logger.info(f"Fine-tuned model saved to {model_save_path}")

def main():
    # Initialize fine-tuner
    fine_tuner = ModelFineTuner()
    
    # Path to preprocessed dataset
    dataset_path = '/workspace/datasets/training/infrastructure_dataset_task_classification.pt'
    
    # Run fine-tuning
    fine_tuner.train(dataset_path)

if __name__ == '__main__':
    main()