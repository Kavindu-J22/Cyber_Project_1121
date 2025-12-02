"""
Speaker Embedding Model using ECAPA-TDNN
Extracts 192-dimensional speaker embeddings
"""
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Union, List
from pathlib import Path
from speechbrain.pretrained import EncoderClassifier

from src.config_loader import get_config


class SpeakerEmbeddingModel:
    """ECAPA-TDNN based speaker embedding extractor"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.embedding_dim = self.config.get('model.embedding_dim', 192)
        self.device = self._get_device()
        self.model = None
        self._load_model()
    
    def _get_device(self) -> torch.device:
        """Get computation device (GPU/CPU)"""
        use_gpu = self.config.get('performance.use_gpu', True)
        if use_gpu and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def _load_model(self):
        """Load pre-trained ECAPA-TDNN model"""
        try:
            # Load pre-trained ECAPA-TDNN from SpeechBrain
            # This model is trained on VoxCeleb and produces 192-dim embeddings
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/pretrained/ecapa_tdnn",
                run_opts={"device": str(self.device)}
            )
            
            print(f"✓ ECAPA-TDNN model loaded successfully on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ECAPA-TDNN model: {str(e)}")
    
    def extract_embedding(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract speaker embedding from audio
        
        Args:
            audio: Audio array or tensor (single channel)
            
        Returns:
            192-dimensional speaker embedding
        """
        try:
            # Convert to tensor if numpy array
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()
            
            # Ensure correct shape [batch, samples] or [samples]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Move to device
            audio = audio.to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(audio)
                
            # Convert to numpy and squeeze
            embedding = embedding.squeeze().cpu().numpy()
            
            # Ensure 192 dimensions
            if embedding.shape[-1] != self.embedding_dim:
                raise ValueError(f"Expected {self.embedding_dim}-dim embedding, got {embedding.shape[-1]}")
            
            return embedding
            
        except Exception as e:
            raise RuntimeError(f"Error extracting embedding: {str(e)}")
    
    def extract_embeddings_batch(self, audio_list: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for multiple audio samples
        
        Args:
            audio_list: List of audio arrays
            
        Returns:
            Array of embeddings [num_samples, embedding_dim]
        """
        embeddings = []
        
        for audio in audio_list:
            embedding = self.extract_embedding(audio)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                          metric: str = 'cosine') -> float:
        """
        Compute similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            Similarity score
        """
        if metric == 'cosine':
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            return float(similarity)
            
        elif metric == 'euclidean':
            # Negative euclidean distance (higher is more similar)
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(-distance)
            
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        L2 normalize embedding
        
        Args:
            embedding: Speaker embedding
            
        Returns:
            Normalized embedding
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def save_embedding(self, embedding: np.ndarray, save_path: str):
        """
        Save embedding to file
        
        Args:
            embedding: Speaker embedding
            save_path: Path to save embedding
        """
        np.save(save_path, embedding)
    
    def load_embedding(self, load_path: str) -> np.ndarray:
        """
        Load embedding from file
        
        Args:
            load_path: Path to embedding file
            
        Returns:
            Speaker embedding
        """
        return np.load(load_path)

