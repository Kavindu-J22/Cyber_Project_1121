"""
Keystroke Embedding Model
Deep learning model for extracting behavioral embeddings from keystroke patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from loguru import logger


class KeystrokeEmbeddingModel(nn.Module):
    """
    Deep neural network for keystroke dynamics embedding
    Extracts behavioral biometric features from typing patterns
    """
    
    def __init__(self, input_dim: int, config):
        """
        Initialize embedding model
        
        Args:
            input_dim: Input feature dimension
            config: Configuration object
        """
        super(KeystrokeEmbeddingModel, self).__init__()
        
        self.config = config
        self.input_dim = input_dim
        self.embedding_dim = config.model.embedding_dim
        self.hidden_dims = config.model.hidden_dims
        self.dropout = config.model.dropout
        
        # Build encoder network
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if config.model.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if config.model.activation == 'relu':
                layers.append(nn.ReLU())
            elif config.model.activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            elif config.model.activation == 'elu':
                layers.append(nn.ELU())
            
            # Dropout
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            
            prev_dim = hidden_dim
        
        # Final embedding layer
        layers.append(nn.Linear(prev_dim, self.embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # L2 normalization for embeddings
        self.normalize = True
        
        logger.info(f"KeystrokeEmbeddingModel initialized: {input_dim} -> {self.embedding_dim}")
        logger.info(f"Architecture: {self.hidden_dims}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            Embeddings (batch_size, embedding_dim)
        """
        # Encode
        embeddings = self.encoder(x)
        
        # L2 normalize embeddings
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for input (inference mode)
        
        Args:
            x: Input features
            
        Returns:
            Normalized embeddings
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(x)
        return embeddings


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning
    Ensures anchor-positive distance < anchor-negative distance
    """
    
    def __init__(self, margin: float = 0.5):
        """
        Initialize triplet loss
        
        Args:
            margin: Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, 
                positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings (same user)
            negative: Negative embeddings (different user)
            
        Returns:
            Triplet loss
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for siamese networks
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss
        
        Args:
            margin: Margin for negative pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1: torch.Tensor,
                embedding2: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            embedding1: First embeddings
            embedding2: Second embeddings
            label: 1 if same user, 0 if different
            
        Returns:
            Contrastive loss
        """
        # Euclidean distance
        distance = F.pairwise_distance(embedding1, embedding2, p=2)
        
        # Contrastive loss
        loss_positive = label * torch.pow(distance, 2)
        loss_negative = (1 - label) * torch.pow(F.relu(self.margin - distance), 2)
        
        loss = loss_positive + loss_negative
        
        return loss.mean()

