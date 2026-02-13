"""
Face Verification Model Implementation
ResNet50 with Triplet Loss for Face Recognition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Any


class ResNet50TripletModel(nn.Module):
    """
    ResNet50 backbone with custom embedding layer for face verification
    Outputs L2-normalized embeddings for cosine similarity comparison
    """
    
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        """
        Initialize ResNet50 Triplet model
        
        Args:
            embedding_dim: Dimension of output embeddings (default: 128)
            pretrained: Use ImageNet pretrained weights for backbone
        """
        super(ResNet50TripletModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Load ResNet50 backbone
        # Use weights parameter instead of deprecated pretrained parameter
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Keep ResNet structure intact (don't wrap in Sequential)
        # This preserves the original layer names for checkpoint loading
        self.backbone = nn.Module()
        self.backbone.conv1 = resnet.conv1
        self.backbone.bn1 = resnet.bn1
        self.backbone.relu = resnet.relu
        self.backbone.maxpool = resnet.maxpool
        self.backbone.layer1 = resnet.layer1
        self.backbone.layer2 = resnet.layer2
        self.backbone.layer3 = resnet.layer3
        self.backbone.layer4 = resnet.layer4
        self.backbone.avgpool = resnet.avgpool
        
        # Get the feature dimension from ResNet50
        # ResNet50 outputs 2048 features before the final FC
        resnet_output_dim = 2048
        
        # Simple fc layer for embedding (matches checkpoint structure)
        # The checkpoint has a simple 2048->512 linear layer
        self.fc = nn.Linear(resnet_output_dim, embedding_dim)
        
        # Store actual embedding dim (will be overridden from checkpoint)
        self._actual_embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            L2-normalized embeddings of shape (batch_size, embedding_dim)
        """
        # Extract features using backbone (standard ResNet50 forward)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        
        # Generate embeddings using fc layer
        embeddings = self.fc(features)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension (actual output dimension of fc layer)"""
        return self.fc.out_features


def load_model_checkpoint(
    checkpoint_path: str,
    embedding_dim: int = 128,
    device: str = 'cpu'
) -> ResNet50TripletModel:
    """
    Load model from checkpoint with flexible loading strategy
    
    Args:
        checkpoint_path: Path to model checkpoint
        embedding_dim: Expected embedding dimension (ignored if checkpoint has fc layer)
        device: Device to load model on ('cpu' or 'cuda')
        
    Returns:
        Loaded model in eval mode
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if model was saved with DataParallel
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    # Detect actual embedding dimension from checkpoint fc layer
    if 'fc.weight' in new_state_dict:
        actual_embedding_dim = new_state_dict['fc.weight'].shape[0]
        print(f"Detected embedding dimension from checkpoint: {actual_embedding_dim}")
    else:
        actual_embedding_dim = embedding_dim
    
    # Initialize model with correct embedding dimension
    model = ResNet50TripletModel(embedding_dim=actual_embedding_dim, pretrained=False)
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    # Log any issues (mostly for debugging)
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    return model


def compute_cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding tensor
        embedding2: Second embedding tensor
        
    Returns:
        Cosine similarity score (-1 to 1)
    """
    # Ensure embeddings are normalized
    embedding1 = F.normalize(embedding1, p=2, dim=-1)
    embedding2 = F.normalize(embedding2, p=2, dim=-1)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(embedding1, embedding2, dim=-1)
    
    return similarity.item()


def compute_euclidean_distance(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
    """
    Compute Euclidean distance between two normalized embeddings
    
    Args:
        embedding1: First embedding tensor
        embedding2: Second embedding tensor
        
    Returns:
        Euclidean distance
    """
    distance = torch.dist(embedding1, embedding2, p=2)
    return distance.item()


if __name__ == "__main__":
    # Test model creation
    print("Testing ResNet50 Triplet Model...")
    
    model = ResNet50TripletModel(embedding_dim=128)
    model.eval()
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        embeddings = model(dummy_input)
    
    print(f"✓ Model created successfully")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {embeddings.shape}")
    print(f"  Embedding dimension: {model.get_embedding_dim()}")
    print(f"  Embedding norm: {torch.norm(embeddings[0]).item():.4f} (should be ~1.0)")
    
    # Test similarity computation
    similarity = compute_cosine_similarity(embeddings[0], embeddings[1])
    distance = compute_euclidean_distance(embeddings[0], embeddings[1])
    print(f"  Cosine similarity: {similarity:.4f}")
    print(f"  Euclidean distance: {distance:.4f}")
