"""
Training Script for Keystroke Dynamics Model
Trains the embedding model on keystroke timing data
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config_loader import load_config
from src.keystroke_preprocessing import KeystrokePreprocessor
from src.keystroke_embedding import KeystrokeEmbeddingModel, TripletLoss, ContrastiveLoss
from src.anomaly_detection import AnomalyDetector


class KeystrokeTrainer:
    """Trainer for keystroke dynamics model"""
    
    def __init__(self, config):
        """Initialize trainer"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.preprocessor = KeystrokePreprocessor(config)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def load_data(self):
        """Load and preprocess dataset"""
        logger.info("Loading dataset...")
        
        # Load DSL dataset
        dataset_path = os.path.join('Dataset', 'DSL-StrongPasswordData-Original_Dataset.xls')
        if not os.path.exists(dataset_path):
            # Try alternative path
            dataset_path = os.path.join('Keystroke Dynamics', 'Dataset', 
                                       'DSL-StrongPasswordData-Original_Dataset.xls')
        
        df = self.preprocessor.load_dsl_dataset(dataset_path)
        
        # Split by subject
        train_df, val_df, test_df = self.preprocessor.split_by_subject(
            df,
            train_ratio=self.config.training.train_ratio,
            val_ratio=self.config.training.val_ratio
        )
        
        # Preprocess
        logger.info("Preprocessing training data...")
        X_train, y_train = self.preprocessor.preprocess_pipeline(
            train_df, fit=True, augment=True
        )
        
        logger.info("Preprocessing validation data...")
        X_val, y_val = self.preprocessor.preprocess_pipeline(
            val_df, fit=False, augment=False
        )
        
        logger.info("Preprocessing test data...")
        X_test, y_test = self.preprocessor.preprocess_pipeline(
            test_df, fit=False, augment=False
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Data loaded: Train={len(train_dataset)}, "
                   f"Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return X_train.shape[1]  # Return input dimension
    
    def build_model(self, input_dim):
        """Build model and training components"""
        logger.info("Building model...")
        
        # Create model
        self.model = KeystrokeEmbeddingModel(input_dim, self.config)
        self.model = self.model.to(self.device)
        
        # Create loss function
        if self.config.training.loss_type == 'triplet':
            self.criterion = TripletLoss(margin=self.config.training.triplet_margin)
        elif self.config.training.loss_type == 'contrastive':
            self.criterion = ContrastiveLoss(margin=1.0)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Create optimizer
        if self.config.training.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        
        # Create scheduler
        if self.config.training.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs
            )
        elif self.config.training.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        
        logger.info(f"Model built with {sum(p.numel() for p in self.model.parameters())} parameters")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.training.epochs}")

        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            embeddings = self.model(data)

            # Compute loss based on type
            if self.config.training.loss_type == 'triplet':
                # Create triplets
                loss = self.compute_triplet_loss(embeddings, labels)
            elif self.config.training.loss_type == 'contrastive':
                loss = self.compute_contrastive_loss(embeddings, labels)
            else:
                # Classification loss (would need classifier head)
                loss = self.criterion(embeddings, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def compute_triplet_loss(self, embeddings, labels):
        """Compute triplet loss"""
        batch_size = embeddings.size(0)

        # Simple triplet mining: for each anchor, find positive and negative
        loss = 0
        count = 0

        for i in range(batch_size):
            anchor = embeddings[i]
            anchor_label = labels[i]

            # Find positive (same label)
            pos_mask = (labels == anchor_label) & (torch.arange(batch_size, device=self.device) != i)
            if pos_mask.sum() > 0:
                pos_idx = torch.where(pos_mask)[0][0]
                positive = embeddings[pos_idx]

                # Find negative (different label)
                neg_mask = labels != anchor_label
                if neg_mask.sum() > 0:
                    neg_idx = torch.where(neg_mask)[0][0]
                    negative = embeddings[neg_idx]

                    # Compute triplet loss
                    loss += self.criterion(anchor.unsqueeze(0),
                                         positive.unsqueeze(0),
                                         negative.unsqueeze(0))
                    count += 1

        return loss / max(count, 1)

    def compute_contrastive_loss(self, embeddings, labels):
        """Compute contrastive loss"""
        batch_size = embeddings.size(0)

        loss = 0
        count = 0

        for i in range(batch_size):
            for j in range(i+1, batch_size):
                emb1 = embeddings[i].unsqueeze(0)
                emb2 = embeddings[j].unsqueeze(0)
                label = (labels[i] == labels[j]).float().unsqueeze(0)

                loss += self.criterion(emb1, emb2, label)
                count += 1

        return loss / max(count, 1)

    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, labels in self.val_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                embeddings = self.model(data)

                if self.config.training.loss_type == 'triplet':
                    loss = self.compute_triplet_loss(embeddings, labels)
                elif self.config.training.loss_type == 'contrastive':
                    loss = self.compute_contrastive_loss(embeddings, labels)
                else:
                    loss = self.criterion(embeddings, labels)

                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")

        patience_counter = 0

        for epoch in range(self.config.training.epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            logger.info(f"Epoch {epoch+1}/{self.config.training.epochs} - "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                patience_counter = 0
                logger.info(f"New best model saved! Val Loss: {val_loss:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')

        logger.info("Training completed!")
        self.plot_training_history()

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_dir = self.config.paths.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        filepath = os.path.join(checkpoint_dir, filename)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config.to_dict()
        }, filepath)

        logger.debug(f"Checkpoint saved: {filepath}")

    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(self.config.paths.logs_dir, 'training_history.png')
        plt.savefig(plot_path)
        logger.info(f"Training history plot saved: {plot_path}")
        plt.close()


def main():
    """Main training function"""
    # Setup logging
    logger.add("logs/training_{time}.log", rotation="100 MB")

    # Load configuration
    config = load_config('config.yaml')

    # Create trainer
    trainer = KeystrokeTrainer(config)

    # Load data
    input_dim = trainer.load_data()

    # Build model
    trainer.build_model(input_dim)

    # Train
    trainer.train()

    logger.info("Training script completed successfully!")


if __name__ == '__main__':
    main()

