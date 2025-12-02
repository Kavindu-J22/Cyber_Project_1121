"""
Speaker Verification Engine
Handles enrollment, verification, and continuous authentication
"""
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

from src.config_loader import get_config
from src.speaker_embedding import SpeakerEmbeddingModel
from src.audio_preprocessing import AudioPreprocessor


class SpeakerVerificationEngine:
    """Speaker verification with continuous authentication"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.embedding_model = SpeakerEmbeddingModel(config)
        self.audio_preprocessor = AudioPreprocessor(config)
        
        # Verification parameters
        self.threshold = self.config.get('verification.threshold', 0.65)
        self.similarity_metric = self.config.get('verification.similarity_metric', 'cosine')
        self.max_latency_ms = self.config.get('verification.max_latency_ms', 800)
        self.enrollment_samples = self.config.get('verification.enrollment_samples', 3)
        
        # Enrolled speakers database (in-memory, will be replaced with MongoDB)
        self.enrolled_speakers: Dict[str, Dict] = {}
    
    def enroll_speaker(self, speaker_id: str, audio_samples: List[str]) -> Dict:
        """
        Enroll a new speaker with multiple audio samples
        
        Args:
            speaker_id: Unique identifier for the speaker (e.g., doctor ID)
            audio_samples: List of paths to enrollment audio files
            
        Returns:
            Enrollment result with statistics
        """
        if len(audio_samples) < self.enrollment_samples:
            raise ValueError(
                f"Need at least {self.enrollment_samples} samples for enrollment, "
                f"got {len(audio_samples)}"
            )
        
        embeddings = []
        
        # Extract embeddings from all enrollment samples
        for audio_path in audio_samples:
            # Preprocess audio
            segments = self.audio_preprocessor.preprocess(audio_path)
            
            # Extract embeddings from all segments
            for segment in segments:
                embedding = self.embedding_model.extract_embedding(segment)
                embedding = self.embedding_model.normalize_embedding(embedding)
                embeddings.append(embedding)
        
        # Compute mean embedding (voiceprint template)
        embeddings_array = np.array(embeddings)
        mean_embedding = np.mean(embeddings_array, axis=0)
        mean_embedding = self.embedding_model.normalize_embedding(mean_embedding)
        
        # Compute enrollment quality metrics
        intra_speaker_similarities = []
        for emb in embeddings:
            sim = self.embedding_model.compute_similarity(
                emb, mean_embedding, self.similarity_metric
            )
            intra_speaker_similarities.append(sim)
        
        enrollment_quality = np.mean(intra_speaker_similarities)
        
        # Store enrollment
        self.enrolled_speakers[speaker_id] = {
            'speaker_id': speaker_id,
            'voiceprint_template': mean_embedding,
            'num_samples': len(audio_samples),
            'num_embeddings': len(embeddings),
            'enrollment_quality': float(enrollment_quality),
            'enrolled_at': datetime.utcnow().isoformat(),
            'embedding_dim': mean_embedding.shape[0]
        }
        
        return {
            'speaker_id': speaker_id,
            'status': 'enrolled',
            'enrollment_quality': float(enrollment_quality),
            'num_samples': len(audio_samples),
            'num_embeddings': len(embeddings)
        }
    
    def verify_speaker(self, speaker_id: str, audio_path: str) -> Dict:
        """
        Verify speaker identity from audio
        
        Args:
            speaker_id: Speaker ID to verify against
            audio_path: Path to verification audio
            
        Returns:
            Verification result with confidence score
        """
        start_time = time.time()
        
        # Check if speaker is enrolled
        if speaker_id not in self.enrolled_speakers:
            raise ValueError(f"Speaker {speaker_id} is not enrolled")
        
        # Get enrolled voiceprint
        enrolled_data = self.enrolled_speakers[speaker_id]
        enrolled_embedding = enrolled_data['voiceprint_template']
        
        # Preprocess audio
        segments = self.audio_preprocessor.preprocess(audio_path)
        
        # Extract embeddings and compute similarities
        similarities = []
        for segment in segments:
            embedding = self.embedding_model.extract_embedding(segment)
            embedding = self.embedding_model.normalize_embedding(embedding)
            
            similarity = self.embedding_model.compute_similarity(
                embedding, enrolled_embedding, self.similarity_metric
            )
            similarities.append(similarity)
        
        # Compute average similarity (confidence score)
        confidence_score = float(np.mean(similarities))
        
        # Verification decision
        is_verified = confidence_score >= self.threshold
        
        # Compute latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Check latency requirement
        if latency_ms > self.max_latency_ms:
            print(f"Warning: Verification latency {latency_ms:.2f}ms exceeds target {self.max_latency_ms}ms")
        
        return {
            'speaker_id': speaker_id,
            'verified': is_verified,
            'confidence_score': confidence_score,
            'threshold': self.threshold,
            'num_segments': len(segments),
            'latency_ms': latency_ms,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def verify_speaker_streaming(self, speaker_id: str, audio_bytes: bytes) -> Dict:
        """
        Verify speaker from streaming audio (real-time)
        
        Args:
            speaker_id: Speaker ID to verify against
            audio_bytes: Audio data as bytes
            
        Returns:
            Verification result
        """
        start_time = time.time()
        
        # Check if speaker is enrolled
        if speaker_id not in self.enrolled_speakers:
            raise ValueError(f"Speaker {speaker_id} is not enrolled")
        
        # Get enrolled voiceprint
        enrolled_data = self.enrolled_speakers[speaker_id]
        enrolled_embedding = enrolled_data['voiceprint_template']

        # Preprocess streaming audio
        segments = self.audio_preprocessor.preprocess_streaming(audio_bytes)

        if len(segments) == 0:
            return {
                'speaker_id': speaker_id,
                'verified': False,
                'confidence_score': 0.0,
                'error': 'No valid audio segments found',
                'latency_ms': (time.time() - start_time) * 1000
            }

        # Extract embeddings and compute similarities
        similarities = []
        for segment in segments:
            embedding = self.embedding_model.extract_embedding(segment)
            embedding = self.embedding_model.normalize_embedding(embedding)

            similarity = self.embedding_model.compute_similarity(
                embedding, enrolled_embedding, self.similarity_metric
            )
            similarities.append(similarity)

        # Compute average similarity
        confidence_score = float(np.mean(similarities))
        is_verified = confidence_score >= self.threshold
        latency_ms = (time.time() - start_time) * 1000

        return {
            'speaker_id': speaker_id,
            'verified': is_verified,
            'confidence_score': confidence_score,
            'threshold': self.threshold,
            'num_segments': len(segments),
            'latency_ms': latency_ms,
            'timestamp': datetime.utcnow().isoformat()
        }

    def continuous_verification(self, speaker_id: str, audio_stream_path: str,
                               window_duration: float = 2.5) -> List[Dict]:
        """
        Perform continuous verification on audio stream

        Args:
            speaker_id: Speaker ID to verify
            audio_stream_path: Path to audio stream file
            window_duration: Duration of each verification window

        Returns:
            List of verification results for each window
        """
        # Preprocess entire audio
        segments = self.audio_preprocessor.preprocess(audio_stream_path)

        results = []
        for i, segment in enumerate(segments):
            start_time = time.time()

            # Get enrolled voiceprint
            enrolled_embedding = self.enrolled_speakers[speaker_id]['voiceprint_template']

            # Extract embedding
            embedding = self.embedding_model.extract_embedding(segment)
            embedding = self.embedding_model.normalize_embedding(embedding)

            # Compute similarity
            confidence_score = self.embedding_model.compute_similarity(
                embedding, enrolled_embedding, self.similarity_metric
            )

            is_verified = confidence_score >= self.threshold
            latency_ms = (time.time() - start_time) * 1000

            result = {
                'window_index': i,
                'speaker_id': speaker_id,
                'verified': is_verified,
                'confidence_score': float(confidence_score),
                'threshold': self.threshold,
                'latency_ms': latency_ms,
                'timestamp': datetime.utcnow().isoformat(),
                'alert_triggered': confidence_score < self.threshold
            }

            results.append(result)

        return results

    def update_threshold(self, new_threshold: float):
        """
        Dynamically update verification threshold

        Args:
            new_threshold: New threshold value (0.0 to 1.0)
        """
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        self.threshold = new_threshold
        print(f"Verification threshold updated to {new_threshold}")

    def get_enrolled_speakers(self) -> List[str]:
        """Get list of enrolled speaker IDs"""
        return list(self.enrolled_speakers.keys())

    def remove_speaker(self, speaker_id: str) -> bool:
        """
        Remove enrolled speaker

        Args:
            speaker_id: Speaker ID to remove

        Returns:
            True if removed, False if not found
        """
        if speaker_id in self.enrolled_speakers:
            del self.enrolled_speakers[speaker_id]
            return True
        return False

    def get_speaker_info(self, speaker_id: str) -> Optional[Dict]:
        """
        Get enrollment information for a speaker

        Args:
            speaker_id: Speaker ID

        Returns:
            Speaker enrollment info or None
        """
        if speaker_id not in self.enrolled_speakers:
            return None

        info = self.enrolled_speakers[speaker_id].copy()
        # Don't return the actual embedding for security
        info.pop('voiceprint_template', None)
        return info
