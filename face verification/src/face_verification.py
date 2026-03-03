"""
Face Verification Engine
Handles enrollment and verification using ResNet50 Triplet model
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
import time
from loguru import logger

from .face_model import (
    ResNet50TripletModel,
    load_model_checkpoint,
    compute_cosine_similarity,
    compute_euclidean_distance
)
from .face_preprocessing import FacePreprocessor


class FaceVerificationEngine:
    """
    Face verification engine for enrollment and verification
    """
    
    def __init__(self, config):
        """
        Initialize face verification engine
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Setup device
        self.device = self._setup_device()
        logger.info(f"Using device: {self.device}")
        
        # Initialize preprocessor
        face_size = config.get('image.face_size', 224)
        self.preprocessor = FacePreprocessor(face_size=face_size)
        
        # Load model
        self.model = self._load_model()
        
        # Verification parameters
        self.threshold = config.get('verification.threshold', 0.8096)
        self.similarity_metric = config.get('verification.similarity_metric', 'cosine')
        
        # In-memory enrollment storage
        # In production, this should use a database
        self.enrollments: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Face verification engine initialized successfully")
    
    def _setup_device(self) -> str:
        """Setup computation device (CPU or CUDA)"""
        device_config = self.config.get('performance.device', 'auto')
        
        if device_config == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device_config
        
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        
        return device
    
    def _load_model(self) -> ResNet50TripletModel:
        """Load face verification model"""
        checkpoint_path = self.config.get('model.checkpoint_path')
        embedding_dim = self.config.get('model.embedding_dim', 128)
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}\n"
                f"Please ensure the model file is in the correct location."
            )
        
        logger.info(f"Loading model from: {checkpoint_path}")
        
        try:
            model = load_model_checkpoint(
                str(checkpoint_path),
                embedding_dim=embedding_dim,
                device=self.device
            )
            # Get actual embedding dimension from loaded model
            actual_embedding_dim = model.get_embedding_dim()
            logger.info(f"✓ Model loaded successfully (embedding_dim={actual_embedding_dim})")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @torch.inference_mode()
    def extract_embedding(
        self,
        image: Union[str, Path, bytes],
        return_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Extract face embedding from image
        
        Args:
            image: Path to image or image bytes
            return_numpy: Return as numpy array instead of tensor
            
        Returns:
            Face embedding (128-d vector)
        """
        # Preprocess image
        tensor = self.preprocessor.preprocess(image)
        tensor = tensor.to(self.device)
        
        # Extract embedding
        embedding = self.model(tensor)
        
        # Move to CPU if needed
        if self.device == 'cuda':
            embedding = embedding.cpu()
        
        if return_numpy:
            return embedding.squeeze(0).numpy()
        
        return embedding.squeeze(0)
    
    def enroll_user(
        self,
        user_id: str,
        face_samples: List[Union[str, Path, bytes]]
    ) -> Dict[str, Any]:
        """
        Enroll user with multiple face samples
        
        Args:
            user_id: Unique user identifier
            face_samples: List of face image paths or bytes
            
        Returns:
            Enrollment result with statistics
        """
        start_time = time.time()
        
        if len(face_samples) == 0:
            raise ValueError("At least one face sample is required")
        
        logger.info(f"Enrolling user {user_id} with {len(face_samples)} samples")
        
        # Extract embeddings for all samples
        embeddings = []
        for idx, sample in enumerate(face_samples):
            try:
                embedding = self.extract_embedding(sample, return_numpy=True)
                embeddings.append(embedding)
                logger.debug(f"  ✓ Processed sample {idx + 1}/{len(face_samples)}")
            except Exception as e:
                logger.warning(f"  ✗ Failed to process sample {idx + 1}: {e}")
                continue
        
        if len(embeddings) == 0:
            raise ValueError("Failed to extract embeddings from any sample")
        
        embeddings_array = np.array(embeddings)
        
        # Compute enrollment quality (average pairwise similarity)
        quality_score = self._compute_enrollment_quality(embeddings_array)
        
        # Store enrollment
        self.enrollments[user_id] = {
            'embeddings': embeddings_array,
            'num_samples': len(embeddings),
            'enrollment_time': time.time(),
            'quality_score': quality_score,
            'mean_embedding': np.mean(embeddings_array, axis=0)
        }
        
        elapsed = (time.time() - start_time) * 1000
        
        logger.info(f"✓ User {user_id} enrolled successfully")
        logger.info(f"  Samples: {len(embeddings)}, Quality: {quality_score:.4f}, Time: {elapsed:.0f}ms")
        
        return {
            'user_id': user_id,
            'num_samples': len(embeddings),
            'enrollment_quality': quality_score,
            'success': True,
            'latency_ms': elapsed
        }
    
    def verify_user(
        self,
        user_id: str,
        face_sample: Union[str, Path, bytes],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Verify user against enrolled face
        
        Args:
            user_id: User identifier
            face_sample: Face image to verify
            threshold: Custom threshold (uses default if None)
            
        Returns:
            Verification result with confidence score
        """
        start_time = time.time()
        
        # Check if user is enrolled
        if user_id not in self.enrollments:
            return {
                'verified': False,
                'confidence_score': 0.0,
                'reason': 'User not enrolled',
                'success': False
            }
        
        # Use default threshold if not provided
        if threshold is None:
            threshold = self.threshold
        
        # Extract embedding from probe image
        try:
            probe_embedding = self.extract_embedding(face_sample, return_numpy=True)
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return {
                'verified': False,
                'confidence_score': 0.0,
                'reason': f'Failed to process image: {str(e)}',
                'success': False
            }
        
        # Get enrolled embeddings
        enrollment_data = self.enrollments[user_id]
        enrolled_embeddings = enrollment_data['embeddings']
        
        # Compute similarities with all enrolled samples
        similarities = []
        for enrolled_emb in enrolled_embeddings:
            if self.similarity_metric == 'cosine':
                # Cosine similarity
                similarity = np.dot(probe_embedding, enrolled_emb) / (
                    np.linalg.norm(probe_embedding) * np.linalg.norm(enrolled_emb)
                )
            else:
                # Euclidean distance (convert to similarity)
                distance = np.linalg.norm(probe_embedding - enrolled_emb)
                similarity = 1.0 / (1.0 + distance)
            
            similarities.append(similarity)
        
        # Use maximum similarity for decision
        max_similarity = float(np.max(similarities))
        mean_similarity = float(np.mean(similarities))
        
        # Make decision
        verified = max_similarity >= threshold
        
        elapsed = (time.time() - start_time) * 1000
        
        logger.info(
            f"Verification for {user_id}: "
            f"{'✓ MATCH' if verified else '✗ MISMATCH'} "
            f"(similarity: {max_similarity:.4f}, threshold: {threshold:.4f})"
        )
        
        return {
            'verified': verified,
            'confidence_score': max_similarity,
            'mean_confidence': mean_similarity,
            'threshold': threshold,
            'decision': 'MATCH' if verified else 'MISMATCH',
            'latency_ms': elapsed,
            'success': True
        }
    
    def _compute_enrollment_quality(self, embeddings: np.ndarray) -> float:
        """
        Compute enrollment quality based on consistency of samples
        Higher score = more consistent samples
        
        Args:
            embeddings: Array of embeddings (num_samples, embedding_dim)
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        if len(embeddings) < 2:
            return 1.0
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
        
        # Return mean similarity as quality score
        return float(np.mean(similarities))
    
    def get_enrolled_users(self) -> List[str]:
        """Get list of enrolled user IDs"""
        return list(self.enrollments.keys())
    
    def remove_enrollment(self, user_id: str) -> bool:
        """
        Remove user enrollment
        
        Args:
            user_id: User to remove
            
        Returns:
            True if removed, False if not found
        """
        if user_id in self.enrollments:
            del self.enrollments[user_id]
            logger.info(f"Removed enrollment for user: {user_id}")
            return True
        return False


if __name__ == "__main__":
    # Test verification engine
    print("Face Verification Engine - Test Mode")
