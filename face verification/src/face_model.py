"""
Face Verification Model Implementation
ResNet50 with Triplet Loss for Face Recognition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections.abc import Mapping


class ResNet50TripletModel(nn.Module):
    """
    ResNet50 backbone with custom embedding layer for face verification
    Outputs L2-normalized embeddings for cosine similarity comparison
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
        head_type: str = "mlp",
        hidden_dim: int = 512,
        dropout: float = 0.3
    ):
        """
        Initialize ResNet50 Triplet model
        
        Args:
            embedding_dim: Dimension of output embeddings (default: 128)
            pretrained: Use ImageNet pretrained weights for backbone
            head_type: Embedding head format ("mlp" for best_model.pt, "fc" for legacy checkpoints)
            hidden_dim: Hidden dimension used by the MLP embedding head
            dropout: Dropout probability used by the MLP embedding head
        """
        super(ResNet50TripletModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.head_type = head_type
        
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
        
        if head_type == "mlp":
            # Matches best_model.pt: head.0/head.1/head.4 keys.
            self.head = nn.Sequential(
                nn.Linear(resnet_output_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, embedding_dim)
            )
        elif head_type == "fc":
            # Legacy checkpoint compatibility: a single fc layer.
            self.fc = nn.Linear(resnet_output_dim, embedding_dim)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")
        
        # Store actual embedding dim (will be overridden from checkpoint)
        self._actual_embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
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
        
        # Generate embeddings using the active checkpoint-compatible head.
        if self.head_type == "mlp":
            embeddings = self.head(features)
        else:
            embeddings = self.fc(features)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension (actual model output dimension)"""
        if self.head_type == "mlp":
            return self.head[-1].out_features
        return self.fc.out_features


def _extract_state_dict(checkpoint) -> Mapping:
    """Extract model weights from supported checkpoint formats."""
    if isinstance(checkpoint, Mapping):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint:
            return checkpoint["model"]

        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint

        raise ValueError(
            "Unsupported checkpoint format. Expected one of: "
            "'model_state_dict', 'state_dict', 'model', or a raw state dict."
        )

    raise ValueError(f"Unsupported checkpoint object type: {type(checkpoint).__name__}")


def _strip_wrapper_prefixes(state_dict: Mapping) -> dict:
    """Remove common wrapper prefixes such as DataParallel or torch.compile."""
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned_state_dict[new_key] = value
    return cleaned_state_dict


def _detect_head_config(state_dict: Mapping) -> dict:
    """Detect whether a checkpoint uses the new MLP head or legacy fc head."""
    if "head.0.weight" in state_dict and "head.4.weight" in state_dict:
        return {
            "head_type": "mlp",
            "hidden_dim": state_dict["head.0.weight"].shape[0],
            "embedding_dim": state_dict["head.4.weight"].shape[0],
        }

    if "fc.weight" in state_dict:
        return {
            "head_type": "fc",
            "hidden_dim": None,
            "embedding_dim": state_dict["fc.weight"].shape[0],
        }

    head_keys = [key for key in state_dict if key.startswith(("head.", "fc."))]
    raise ValueError(
        "Could not detect a supported face embedding head in checkpoint. "
        f"Found head-like keys: {head_keys[:10]}"
    )


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
    
    state_dict = _strip_wrapper_prefixes(_extract_state_dict(checkpoint))
    head_config = _detect_head_config(state_dict)
    actual_embedding_dim = head_config["embedding_dim"]
    print(
        "Detected face model head: "
        f"{head_config['head_type']} (embedding_dim={actual_embedding_dim})"
    )
    
    # Initialize model with correct embedding dimension
    model = ResNet50TripletModel(
        embedding_dim=actual_embedding_dim,
        pretrained=False,
        head_type=head_config["head_type"],
        hidden_dim=head_config["hidden_dim"] or 512
    )
    
    # Strict loading prevents silently serving a partially initialized model.
    model.load_state_dict(state_dict, strict=True)
    
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
    dummy_input = torch.randn(2, 3, 112, 112)
    
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
