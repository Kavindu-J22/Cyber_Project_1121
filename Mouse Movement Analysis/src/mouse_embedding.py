"""
Mouse Movement Embedding Model
Siamese neural network with triplet loss for extracting behavioral embeddings
from mouse movement patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from loguru import logger


class MouseEmbeddingModel(nn.Module):
    """
    Siamese neural network for mouse dynamics embedding
    Extracts behavioral biometric features from mouse movement patterns
    Uses shared weights to learn similarity metrics
    """
    
    def __init__(self, input_dim: int, config):
        """
        Initialize embedding model
        
        Args:
            input_dim: Input feature dimension
            config: Configuration object
        """
        super(MouseEmbeddingModel, self).__init__()
        
        self.config = config
        self.input_dim = input_dim
        self.embedding_dim = config.model.embedding_dim
        self.hidden_dims = config.model.hidden_dims
        self.dropout = config.model.dropout
        
        # Build encoder network (shared weights for Siamese architecture)
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
            elif config.model.activation == 'selu':
                layers.append(nn.SELU())
            
            # Dropout
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            
            prev_dim = hidden_dim
        
        # Final embedding layer
        layers.append(nn.Linear(prev_dim, self.embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # L2 normalization for embeddings
        self.normalize = True
        
        logger.info(f"MouseEmbeddingModel initialized: {input_dim} -> {self.embedding_dim}")
        logger.info(f"Architecture: {self.hidden_dims}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters())}")
    
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
        
        # L2 normalize embeddings for better similarity computation
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
            return self.forward(x)
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor, 
                          metric: str = 'cosine') -> torch.Tensor:
        """
        Compute similarity between embeddings
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            metric: Similarity metric ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            Similarity score
        """
        if metric == 'cosine':
            # Cosine similarity (already normalized, so just dot product)
            similarity = F.cosine_similarity(emb1, emb2, dim=-1)
        elif metric == 'euclidean':
            # Euclidean distance (convert to similarity)
            distance = torch.norm(emb1 - emb2, p=2, dim=-1)
            similarity = 1.0 / (1.0 + distance)
        elif metric == 'manhattan':
            # Manhattan distance (convert to similarity)
            distance = torch.norm(emb1 - emb2, p=1, dim=-1)
            similarity = 1.0 / (1.0 + distance)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
        
        return similarity


class TripletLoss(nn.Module):
    """
    Triplet loss for Siamese network training
    Learns to minimize distance between anchor and positive,
    while maximizing distance between anchor and negative
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize triplet loss
        
        Args:
            margin: Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings (same user)
            negative: Negative embeddings (different user)

        Returns:
            Triplet loss value
        """
        # Compute distances
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)

        # Triplet loss: max(d(a,p) - d(a,n) + margin, 0)
        losses = F.relu(pos_distance - neg_distance + self.margin)

        return losses.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese network training
    Alternative to triplet loss
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss

        Args:
            margin: Margin for contrastive loss
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss

        Args:
            emb1: First embeddings
            emb2: Second embeddings
            label: 1 if same user, 0 if different user

        Returns:
            Contrastive loss value
        """
        # Euclidean distance
        distance = F.pairwise_distance(emb1, emb2, p=2)

        # Contrastive loss
        loss_positive = label * torch.pow(distance, 2)
        loss_negative = (1 - label) * torch.pow(F.relu(self.margin - distance), 2)

        loss = loss_positive + loss_negative

        return loss.mean()


class HardTripletMiner:
    """
    Hard triplet mining for more effective training
    Selects hardest positive and negative examples
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize hard triplet miner

        Args:
            margin: Margin for triplet selection
        """
        self.margin = margin

    def mine_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor,
                     mining_strategy: str = 'hard') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine triplets from batch

        Args:
            embeddings: Batch of embeddings
            labels: Batch of labels
            mining_strategy: 'hard', 'semi-hard', or 'all'

        Returns:
            Tuple of (anchor_indices, positive_indices, negative_indices)
        """
        batch_size = embeddings.size(0)

        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        anchor_indices = []
        positive_indices = []
        negative_indices = []

        for i in range(batch_size):
            anchor_label = labels[i]

            # Find positive samples (same label, different index)
            positive_mask = (labels == anchor_label) & (torch.arange(batch_size, device=labels.device) != i)

            if positive_mask.sum() == 0:
                continue

            # Find negative samples (different label)
            negative_mask = labels != anchor_label

            if negative_mask.sum() == 0:
                continue

            if mining_strategy == 'hard':
                # Hardest positive (farthest same-class sample)
                positive_distances = distances[i].clone()
                positive_distances[~positive_mask] = -float('inf')
                positive_idx = torch.argmax(positive_distances)

                # Hardest negative (closest different-class sample)
                negative_distances = distances[i].clone()
                negative_distances[~negative_mask] = float('inf')
                negative_idx = torch.argmin(negative_distances)

            elif mining_strategy == 'semi-hard':
                # Semi-hard negative: d(a,p) < d(a,n) < d(a,p) + margin
                positive_distances = distances[i].clone()
                positive_distances[~positive_mask] = float('inf')
                positive_idx = torch.argmin(positive_distances)

                d_ap = distances[i, positive_idx]

                # Find negatives in semi-hard range
                negative_distances = distances[i].clone()
                semi_hard_mask = negative_mask & (negative_distances > d_ap) & (negative_distances < d_ap + self.margin)

                if semi_hard_mask.sum() > 0:
                    negative_distances[~semi_hard_mask] = float('inf')
                    negative_idx = torch.argmin(negative_distances)
                else:
                    # Fall back to hard negative
                    negative_distances[~negative_mask] = float('inf')
                    negative_idx = torch.argmin(negative_distances)

            else:  # 'all'
                # Random positive and negative
                positive_idx = torch.where(positive_mask)[0][0]
                negative_idx = torch.where(negative_mask)[0][0]

            anchor_indices.append(i)
            positive_indices.append(positive_idx.item())
            negative_indices.append(negative_idx.item())

        if len(anchor_indices) == 0:
            return None, None, None

        return (torch.tensor(anchor_indices, device=embeddings.device),
                torch.tensor(positive_indices, device=embeddings.device),
                torch.tensor(negative_indices, device=embeddings.device))


