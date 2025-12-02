"""
Security and Privacy Module
Handles embedding encryption, TLS, and privacy compliance
"""
import os
import base64
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from typing import Tuple
import json

from src.config_loader import get_config


class EmbeddingEncryption:
    """
    Encrypt and decrypt speaker embeddings
    Ensures embeddings are never stored in plaintext
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.encryption_key = self._get_encryption_key()
        self.cipher = Fernet(self.encryption_key)
    
    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key"""
        key_env = self.config.get('security.embedding_encryption_key_env', 'EMBEDDING_ENCRYPTION_KEY')
        key_str = os.getenv(key_env)
        
        if not key_str:
            # Generate a new key (in production, this should be stored securely)
            key = Fernet.generate_key()
            print("⚠️  Warning: Generated new encryption key. Store this securely!")
            print(f"   Set {key_env}={key.decode()}")
            return key
        
        # Use existing key
        return key_str.encode()
    
    def encrypt_embedding(self, embedding: np.ndarray) -> str:
        """
        Encrypt speaker embedding
        
        Args:
            embedding: Speaker embedding array
            
        Returns:
            Encrypted embedding as base64 string
        """
        # Convert to bytes
        embedding_bytes = embedding.tobytes()
        
        # Encrypt
        encrypted = self.cipher.encrypt(embedding_bytes)
        
        # Encode to base64 for storage
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt_embedding(self, encrypted_str: str, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Decrypt speaker embedding
        
        Args:
            encrypted_str: Encrypted embedding as base64 string
            shape: Original shape of the embedding
            
        Returns:
            Decrypted embedding array
        """
        # Decode from base64
        encrypted = base64.b64decode(encrypted_str.encode('utf-8'))
        
        # Decrypt
        decrypted_bytes = self.cipher.decrypt(encrypted)
        
        # Convert back to numpy array
        embedding = np.frombuffer(decrypted_bytes, dtype=np.float32)
        
        # Reshape
        return embedding.reshape(shape)
    
    def encrypt_embedding_dict(self, embedding: np.ndarray) -> dict:
        """
        Encrypt embedding and return with metadata
        
        Args:
            embedding: Speaker embedding
            
        Returns:
            Dictionary with encrypted embedding and metadata
        """
        return {
            'encrypted_data': self.encrypt_embedding(embedding),
            'shape': embedding.shape,
            'dtype': str(embedding.dtype),
            'encrypted': True
        }
    
    def decrypt_embedding_dict(self, encrypted_dict: dict) -> np.ndarray:
        """
        Decrypt embedding from dictionary
        
        Args:
            encrypted_dict: Dictionary with encrypted embedding
            
        Returns:
            Decrypted embedding
        """
        return self.decrypt_embedding(
            encrypted_dict['encrypted_data'],
            tuple(encrypted_dict['shape'])
        )


class PrivacyCompliance:
    """
    Ensure privacy compliance
    - Never store raw audio
    - Only store feature vectors (embeddings)
    - Automatic data retention policies
    """
    
    @staticmethod
    def validate_no_raw_audio(data: dict) -> bool:
        """
        Validate that no raw audio is being stored
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            True if compliant, False otherwise
        """
        # Check for common audio data keys
        forbidden_keys = ['audio', 'raw_audio', 'waveform', 'audio_data', 'audio_bytes']
        
        for key in forbidden_keys:
            if key in data:
                raise ValueError(f"Privacy violation: Raw audio data found in key '{key}'")
        
        return True
    
    @staticmethod
    def sanitize_storage_data(speaker_data: dict) -> dict:
        """
        Sanitize data before storage to ensure privacy compliance
        
        Args:
            speaker_data: Speaker enrollment data
            
        Returns:
            Sanitized data safe for storage
        """
        # Only keep essential metadata and encrypted embeddings
        safe_keys = [
            'speaker_id', 'voiceprint_template', 'num_samples',
            'num_embeddings', 'enrollment_quality', 'enrolled_at',
            'embedding_dim', 'encrypted'
        ]
        
        sanitized = {k: v for k, v in speaker_data.items() if k in safe_keys}
        
        # Validate no raw audio
        PrivacyCompliance.validate_no_raw_audio(sanitized)
        
        return sanitized


class SecureStorage:
    """
    Secure storage for speaker embeddings
    Combines encryption and privacy compliance
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.encryption = EmbeddingEncryption(config)
        self.store_raw_audio = self.config.get('security.store_raw_audio', False)
        
        if self.store_raw_audio:
            raise ValueError("Configuration error: store_raw_audio must be False for privacy compliance")
    
    def prepare_for_storage(self, speaker_data: dict) -> dict:
        """
        Prepare speaker data for secure storage
        
        Args:
            speaker_data: Speaker enrollment data
            
        Returns:
            Encrypted and sanitized data
        """
        # Encrypt the voiceprint template
        if 'voiceprint_template' in speaker_data:
            embedding = speaker_data['voiceprint_template']
            speaker_data['voiceprint_template'] = self.encryption.encrypt_embedding_dict(embedding)
        
        # Sanitize data
        sanitized = PrivacyCompliance.sanitize_storage_data(speaker_data)
        
        return sanitized
    
    def retrieve_from_storage(self, stored_data: dict) -> dict:
        """
        Retrieve and decrypt speaker data from storage
        
        Args:
            stored_data: Encrypted stored data
            
        Returns:
            Decrypted speaker data
        """
        # Decrypt the voiceprint template
        if 'voiceprint_template' in stored_data and isinstance(stored_data['voiceprint_template'], dict):
            if stored_data['voiceprint_template'].get('encrypted', False):
                stored_data['voiceprint_template'] = self.encryption.decrypt_embedding_dict(
                    stored_data['voiceprint_template']
                )
        
        return stored_data

