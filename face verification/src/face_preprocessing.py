"""
Face Image Preprocessing
Handles image loading, validation, and preprocessing for face verification
"""
import torch
from PIL import Image
from torchvision import transforms
from typing import Union, Tuple
import numpy as np
from pathlib import Path


class FacePreprocessor:
    """
    Face image preprocessing pipeline
    MUST match training preprocessing exactly
    """
    
    def __init__(self, face_size: int = 224):
        """
        Initialize preprocessor with ImageNet normalization
        
        Args:
            face_size: Target image size (default: 224 for ResNet)
        """
        self.face_size = face_size
        
        # Define preprocessing pipeline - MUST MATCH TRAINING
        self.transform = transforms.Compose([
            transforms.Resize((face_size, face_size)),
            transforms.ToTensor(),  # Converts to [0, 1] range
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image in RGB format
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Open image and convert to RGB
        image = Image.open(image_path)
        
        # Ensure RGB format (handle RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def load_image_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """
        Load image from bytes
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            PIL Image in RGB format
        """
        from io import BytesIO
        
        image = Image.open(BytesIO(image_bytes))
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def preprocess(self, image: Union[Image.Image, str, Path]) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image, path to image, or image bytes
            
        Returns:
            Preprocessed tensor of shape (1, 3, face_size, face_size)
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        elif isinstance(image, bytes):
            image = self.load_image_from_bytes(image)
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Validate image
        self.validate_image(image)
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def validate_image(self, image: Image.Image) -> None:
        """
        Validate image meets requirements
        
        Args:
            image: PIL Image to validate
            
        Raises:
            ValueError: If image is invalid
        """
        # Check image mode
        if image.mode not in ['RGB', 'RGBA', 'L']:
            raise ValueError(f"Unsupported image mode: {image.mode}")
        
        # Check image size
        width, height = image.size
        if width < 50 or height < 50:
            raise ValueError(f"Image too small: {width}x{height}. Minimum 50x50 pixels.")
        
        if width > 4000 or height > 4000:
            raise ValueError(f"Image too large: {width}x{height}. Maximum 4000x4000 pixels.")
    
    def preprocess_batch(self, images: list) -> torch.Tensor:
        """
        Preprocess batch of images
        
        Args:
            images: List of PIL Images or paths
            
        Returns:
            Batch tensor of shape (batch_size, 3, face_size, face_size)
        """
        tensors = []
        for image in images:
            tensor = self.preprocess(image)
            tensors.append(tensor.squeeze(0))
        
        # Stack into batch
        batch = torch.stack(tensors, dim=0)
        return batch
    
    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize tensor back to viewable image
        Useful for debugging/visualization
        
        Args:
            tensor: Normalized tensor
            
        Returns:
            Numpy array in [0, 255] range
        """
        # Reverse normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy and scale to [0, 255]
        image = tensor.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        
        return image


def is_valid_image_format(filename: str) -> bool:
    """
    Check if file has valid image extension
    
    Args:
        filename: File name or path
        
    Returns:
        True if valid image format
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    extension = Path(filename).suffix.lower()
    return extension in valid_extensions


if __name__ == "__main__":
    # Test preprocessor
    print("Testing Face Preprocessor...")
    
    preprocessor = FacePreprocessor(face_size=224)
    
    # Create dummy image
    dummy_image = Image.new('RGB', (300, 300), color='red')
    
    # Test preprocessing
    tensor = preprocessor.preprocess(dummy_image)
    
    print(f"✓ Preprocessor working")
    print(f"  Input image size: {dummy_image.size}")
    print(f"  Output tensor shape: {tensor.shape}")
    print(f"  Tensor dtype: {tensor.dtype}")
    print(f"  Tensor range: [{tensor.min():.2f}, {tensor.max():.2f}]")
