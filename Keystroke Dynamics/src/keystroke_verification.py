"""
Keystroke Verification Module
Continuous user verification through keystroke dynamics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import mahalanobis
import time


class KeystrokeVerifier:
    """
    Keystroke dynamics verifier for continuous authentication
    """
    
    def __init__(self, model, config):
        """
        Initialize verifier
        
        Args:
            model: Trained embedding model
            config: Configuration object
        """
        self.model = model
        self.config = config
        self.enrolled_templates = {}  # user_id -> template embeddings
        self.verification_history = {}  # user_id -> verification scores
        
        self.threshold = config.verification.threshold
        self.similarity_metric = config.verification.similarity_metric
        
        logger.info("KeystrokeVerifier initialized")
    
    def enroll_user(self, user_id: str, keystroke_samples: torch.Tensor) -> Dict:
        """
        Enroll a user by creating their behavioral template
        
        Args:
            user_id: User identifier
            keystroke_samples: Keystroke feature samples (n_samples, n_features)
            
        Returns:
            Enrollment result dictionary
        """
        logger.info(f"Enrolling user: {user_id}")
        
        if len(keystroke_samples) < self.config.enrollment.min_samples:
            logger.warning(f"Insufficient samples for enrollment: {len(keystroke_samples)}")
            return {
                'success': False,
                'message': f'Need at least {self.config.enrollment.min_samples} samples'
            }
        
        # Generate embeddings
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(keystroke_samples)
        
        # Create template (mean embedding)
        template = embeddings.mean(dim=0)
        
        # Store template
        self.enrolled_templates[user_id] = {
            'template': template,
            'embeddings': embeddings,
            'n_samples': len(keystroke_samples),
            'enrollment_time': time.time()
        }
        
        logger.info(f"User {user_id} enrolled successfully with {len(keystroke_samples)} samples")
        
        return {
            'success': True,
            'user_id': user_id,
            'n_samples': len(keystroke_samples),
            'embedding_dim': template.shape[0]
        }
    
    def verify_user(self, user_id: str, keystroke_sample: torch.Tensor) -> Dict:
        """
        Verify a user's identity based on keystroke sample
        
        Args:
            user_id: User identifier
            keystroke_sample: Single keystroke feature sample
            
        Returns:
            Verification result dictionary
        """
        start_time = time.time()
        
        if user_id not in self.enrolled_templates:
            logger.warning(f"User {user_id} not enrolled")
            return {
                'verified': False,
                'confidence': 0.0,
                'message': 'User not enrolled'
            }
        
        # Generate embedding
        self.model.eval()
        with torch.no_grad():
            if keystroke_sample.dim() == 1:
                keystroke_sample = keystroke_sample.unsqueeze(0)
            embedding = self.model(keystroke_sample)
        
        # Get template
        template = self.enrolled_templates[user_id]['template'].unsqueeze(0)
        
        # Compute similarity
        similarity = self.compute_similarity(embedding, template)
        
        # Verification decision
        verified = similarity >= self.threshold
        confidence = float(similarity)
        
        # Determine confidence level
        if confidence >= self.config.verification.confidence_levels.high:
            confidence_level = 'high'
        elif confidence >= self.config.verification.confidence_levels.medium:
            confidence_level = 'medium'
        elif confidence >= self.config.verification.confidence_levels.low:
            confidence_level = 'low'
        else:
            confidence_level = 'very_low'
        
        # Check for alerts
        alert = confidence < self.config.verification.alert_threshold
        critical = confidence < self.config.verification.critical_threshold
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Store verification history
        if user_id not in self.verification_history:
            self.verification_history[user_id] = []
        self.verification_history[user_id].append({
            'timestamp': time.time(),
            'confidence': confidence,
            'verified': verified
        })
        
        result = {
            'verified': verified,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'alert': alert,
            'critical': critical,
            'latency_ms': latency_ms,
            'user_id': user_id
        }
        
        logger.info(f"Verification: {user_id} - Verified={verified}, "
                   f"Confidence={confidence:.3f}, Latency={latency_ms:.1f}ms")
        
        return result
    
    def compute_similarity(self, embedding1: torch.Tensor, 
                          embedding2: torch.Tensor) -> float:
        """
        Compute similarity between embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        emb1 = embedding1.cpu().numpy()
        emb2 = embedding2.cpu().numpy()
        
        if self.similarity_metric == 'cosine':
            similarity = cosine_similarity(emb1, emb2)[0, 0]
            # Convert to 0-1 range
            similarity = (similarity + 1) / 2
        elif self.similarity_metric == 'euclidean':
            distance = euclidean_distances(emb1, emb2)[0, 0]
            # Convert distance to similarity
            similarity = 1 / (1 + distance)
        elif self.similarity_metric == 'mahalanobis':
            # Simplified mahalanobis (would need covariance matrix)
            distance = np.linalg.norm(emb1 - emb2)
            similarity = 1 / (1 + distance)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return float(similarity)

    def continuous_verification(self, user_id: str,
                               keystroke_stream: List[torch.Tensor],
                               window_size: int = None) -> List[Dict]:
        """
        Perform continuous verification on a stream of keystrokes

        Args:
            user_id: User identifier
            keystroke_stream: List of keystroke samples
            window_size: Number of samples per verification window

        Returns:
            List of verification results
        """
        if window_size is None:
            window_size = self.config.verification.min_keystrokes

        logger.info(f"Continuous verification for {user_id}: {len(keystroke_stream)} samples")

        results = []
        for i in range(0, len(keystroke_stream), window_size):
            window = keystroke_stream[i:i+window_size]
            if len(window) >= self.config.verification.min_keystrokes:
                # Aggregate window samples
                window_tensor = torch.stack(window).mean(dim=0)
                result = self.verify_user(user_id, window_tensor)
                result['window_index'] = i // window_size
                results.append(result)

        return results

    def update_template(self, user_id: str, new_sample: torch.Tensor,
                       update_rate: float = None) -> bool:
        """
        Update user template with new verified sample (adaptive learning)

        Args:
            user_id: User identifier
            new_sample: New keystroke sample
            update_rate: Weight for new sample (0-1)

        Returns:
            Success status
        """
        if not self.config.enrollment.adaptive_templates:
            return False

        if user_id not in self.enrolled_templates:
            logger.warning(f"Cannot update template: {user_id} not enrolled")
            return False

        if update_rate is None:
            update_rate = self.config.enrollment.template_update_rate

        # Generate embedding for new sample
        self.model.eval()
        with torch.no_grad():
            if new_sample.dim() == 1:
                new_sample = new_sample.unsqueeze(0)
            new_embedding = self.model(new_sample).squeeze(0)

        # Update template with exponential moving average
        old_template = self.enrolled_templates[user_id]['template']
        new_template = (1 - update_rate) * old_template + update_rate * new_embedding

        # Normalize
        new_template = torch.nn.functional.normalize(new_template, p=2, dim=0)

        self.enrolled_templates[user_id]['template'] = new_template

        logger.debug(f"Updated template for {user_id}")

        return True

    def get_verification_statistics(self, user_id: str) -> Dict:
        """
        Get verification statistics for a user

        Args:
            user_id: User identifier

        Returns:
            Statistics dictionary
        """
        if user_id not in self.verification_history:
            return {'message': 'No verification history'}

        history = self.verification_history[user_id]
        confidences = [h['confidence'] for h in history]
        verified_count = sum(1 for h in history if h['verified'])

        stats = {
            'user_id': user_id,
            'total_verifications': len(history),
            'verified_count': verified_count,
            'rejection_count': len(history) - verified_count,
            'verification_rate': verified_count / len(history) if history else 0,
            'mean_confidence': np.mean(confidences) if confidences else 0,
            'std_confidence': np.std(confidences) if confidences else 0,
            'min_confidence': np.min(confidences) if confidences else 0,
            'max_confidence': np.max(confidences) if confidences else 0,
        }

        return stats

    def compute_eer(self, genuine_scores: List[float],
                    impostor_scores: List[float]) -> Tuple[float, float]:
        """
        Compute Equal Error Rate (EER) and optimal threshold

        Args:
            genuine_scores: Similarity scores for genuine users
            impostor_scores: Similarity scores for impostors

        Returns:
            Tuple of (EER, optimal_threshold)
        """
        # Combine scores and labels
        scores = np.array(genuine_scores + impostor_scores)
        labels = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))

        # Sort by scores
        sorted_indices = np.argsort(scores)
        scores = scores[sorted_indices]
        labels = labels[sorted_indices]

        # Compute FAR and FRR for different thresholds
        n_genuine = len(genuine_scores)
        n_impostor = len(impostor_scores)

        best_eer = 1.0
        best_threshold = 0.5

        for threshold in np.linspace(0, 1, 1000):
            # False Accept Rate (impostor accepted)
            far = np.sum((scores >= threshold) & (labels == 0)) / n_impostor if n_impostor > 0 else 0

            # False Reject Rate (genuine rejected)
            frr = np.sum((scores < threshold) & (labels == 1)) / n_genuine if n_genuine > 0 else 0

            # EER is where FAR = FRR
            eer = (far + frr) / 2

            if abs(far - frr) < abs(best_eer - best_threshold):
                best_eer = (far + frr) / 2
                best_threshold = threshold

        logger.info(f"EER: {best_eer:.4f}, Optimal Threshold: {best_threshold:.4f}")

        return best_eer, best_threshold

    def save_templates(self, filepath: str):
        """Save enrolled templates to file"""
        torch.save(self.enrolled_templates, filepath)
        logger.info(f"Templates saved to {filepath}")

    def load_templates(self, filepath: str):
        """Load enrolled templates from file"""
        self.enrolled_templates = torch.load(filepath)
        logger.info(f"Templates loaded from {filepath}")
        logger.info(f"Loaded {len(self.enrolled_templates)} user templates")

