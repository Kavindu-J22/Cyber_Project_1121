"""
Mouse Movement Verification Module
Handles user enrollment, verification, and continuous authentication
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from collections import defaultdict
import time


class MouseVerifier:
    """
    Mouse movement verification system
    Handles enrollment, verification, and continuous monitoring
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # User templates (enrolled embeddings)
        self.user_templates = {}
        
        # Verification history
        self.verification_history = defaultdict(list)
        
        # Statistics
        self.stats = defaultdict(lambda: {
            'total_verifications': 0,
            'successful_verifications': 0,
            'failed_verifications': 0,
            'confidences': []
        })
        
        logger.info("MouseVerifier initialized")
    
    def enroll_user(self, user_id: str, samples: torch.Tensor) -> Dict:
        """
        Enroll a user by creating their behavioral template
        
        Args:
            user_id: User identifier
            samples: Feature samples for enrollment (n_samples, n_features)
            
        Returns:
            Enrollment result dictionary
        """
        logger.info(f"Enrolling user: {user_id} with {len(samples)} samples")
        
        if len(samples) < self.config.enrollment.min_samples:
            raise ValueError(f"Insufficient samples for enrollment. Need at least {self.config.enrollment.min_samples}")
        
        # Move to device
        samples = samples.to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model(samples)
        
        # Create template (mean embedding)
        template = embeddings.mean(dim=0)
        
        # Store template
        self.user_templates[user_id] = {
            'template': template.cpu(),
            'embeddings': embeddings.cpu(),
            'enrollment_time': time.time(),
            'num_samples': len(samples)
        }
        
        logger.info(f"User {user_id} enrolled successfully")
        
        return {
            'user_id': user_id,
            'enrolled': True,
            'num_samples': len(samples),
            'template_shape': template.shape
        }
    
    def verify_user(self, user_id: str, sample: torch.Tensor, 
                   return_embedding: bool = False) -> Dict:
        """
        Verify a user against their enrolled template
        
        Args:
            user_id: User identifier
            sample: Feature sample for verification (n_features,)
            return_embedding: Whether to return the embedding
            
        Returns:
            Verification result dictionary
        """
        if user_id not in self.user_templates:
            raise ValueError(f"User {user_id} not enrolled")
        
        # Ensure sample is 2D
        if sample.dim() == 1:
            sample = sample.unsqueeze(0)
        
        # Move to device
        sample = sample.to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model(sample)
        
        # Get template
        template = self.user_templates[user_id]['template'].to(self.device)
        if template.dim() == 1:
            template = template.unsqueeze(0)
        
        # Compute similarity
        similarity = self.model.compute_similarity(
            embedding, 
            template,
            metric=self.config.verification.similarity_metric
        )
        
        confidence = similarity.item()
        verified = confidence >= self.config.verification.threshold
        
        # Determine confidence level
        if confidence >= self.config.verification.confidence_levels.high:
            confidence_level = 'high'
        elif confidence >= self.config.verification.confidence_levels.medium:
            confidence_level = 'medium'
        elif confidence >= self.config.verification.confidence_levels.low:
            confidence_level = 'low'
        else:
            confidence_level = 'very_low'
        
        # Update statistics
        self.stats[user_id]['total_verifications'] += 1
        if verified:
            self.stats[user_id]['successful_verifications'] += 1
        else:
            self.stats[user_id]['failed_verifications'] += 1
        self.stats[user_id]['confidences'].append(confidence)
        
        # Store in history
        self.verification_history[user_id].append({
            'timestamp': time.time(),
            'confidence': confidence,
            'verified': verified
        })
        
        result = {
            'user_id': user_id,
            'verified': verified,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'threshold': self.config.verification.threshold
        }
        
        if return_embedding:
            result['embedding'] = embedding.cpu()
        
        return result

    def continuous_verify(self, user_id: str, samples: torch.Tensor,
                         window_size: int = None) -> Dict:
        """
        Continuous verification over a sequence of samples

        Args:
            user_id: User identifier
            samples: Sequence of feature samples (n_samples, n_features)
            window_size: Size of sliding window (default from config)

        Returns:
            Continuous verification result
        """
        if window_size is None:
            window_size = self.config.verification.min_movements

        if len(samples) < window_size:
            raise ValueError(f"Insufficient samples. Need at least {window_size}")

        # Verify each sample
        results = []
        for i in range(len(samples)):
            result = self.verify_user(user_id, samples[i])
            results.append(result)

        # Aggregate results
        confidences = [r['confidence'] for r in results]
        verifications = [r['verified'] for r in results]

        mean_confidence = np.mean(confidences)
        verification_rate = np.mean(verifications)

        # Overall verification (majority vote or mean confidence)
        overall_verified = mean_confidence >= self.config.verification.threshold

        return {
            'user_id': user_id,
            'overall_verified': overall_verified,
            'mean_confidence': mean_confidence,
            'verification_rate': verification_rate,
            'num_samples': len(samples),
            'individual_results': results
        }

    def update_template(self, user_id: str, sample: torch.Tensor):
        """
        Update user template with new verified sample (adaptive learning)

        Args:
            user_id: User identifier
            sample: New verified sample
        """
        if not self.config.enrollment.adaptive_templates:
            return

        if user_id not in self.user_templates:
            raise ValueError(f"User {user_id} not enrolled")

        # Verify sample first
        result = self.verify_user(user_id, sample, return_embedding=True)

        if not result['verified']:
            logger.warning(f"Sample not verified, skipping template update")
            return

        # Get current template and new embedding
        template = self.user_templates[user_id]['template']
        new_embedding = result['embedding'].squeeze()

        # Update with exponential moving average
        alpha = self.config.enrollment.template_update_rate
        updated_template = (1 - alpha) * template + alpha * new_embedding

        # Normalize
        updated_template = updated_template / torch.norm(updated_template)

        # Update stored template
        self.user_templates[user_id]['template'] = updated_template

        logger.debug(f"Template updated for user {user_id}")

    def get_verification_statistics(self, user_id: str) -> Dict:
        """
        Get verification statistics for a user

        Args:
            user_id: User identifier

        Returns:
            Statistics dictionary
        """
        if user_id not in self.stats:
            return {
                'total_verifications': 0,
                'verification_rate': 0.0,
                'mean_confidence': 0.0
            }

        stats = self.stats[user_id]

        return {
            'total_verifications': stats['total_verifications'],
            'successful_verifications': stats['successful_verifications'],
            'failed_verifications': stats['failed_verifications'],
            'verification_rate': stats['successful_verifications'] / max(stats['total_verifications'], 1),
            'mean_confidence': np.mean(stats['confidences']) if stats['confidences'] else 0.0,
            'std_confidence': np.std(stats['confidences']) if stats['confidences'] else 0.0
        }

    def get_verification_history(self, user_id: str, limit: int = 100) -> List[Dict]:
        """
        Get recent verification history for a user

        Args:
            user_id: User identifier
            limit: Maximum number of recent verifications to return

        Returns:
            List of verification records
        """
        if user_id not in self.verification_history:
            return []

        history = self.verification_history[user_id]
        return history[-limit:]

    def detect_anomaly(self, user_id: str, confidence: float) -> bool:
        """
        Detect if verification result is anomalous

        Args:
            user_id: User identifier
            confidence: Verification confidence score

        Returns:
            True if anomaly detected
        """
        # Check against critical threshold
        if confidence < self.config.verification.critical_threshold:
            logger.warning(f"Critical threshold breach for user {user_id}: {confidence:.3f}")
            return True

        # Check against alert threshold
        if confidence < self.config.verification.alert_threshold:
            logger.info(f"Alert threshold breach for user {user_id}: {confidence:.3f}")
            return True

        return False

    def reset_user(self, user_id: str):
        """
        Reset user enrollment and statistics

        Args:
            user_id: User identifier
        """
        if user_id in self.user_templates:
            del self.user_templates[user_id]

        if user_id in self.stats:
            del self.stats[user_id]

        if user_id in self.verification_history:
            del self.verification_history[user_id]

        logger.info(f"User {user_id} reset")

    def save_templates(self, filepath: str):
        """
        Save user templates to file

        Args:
            filepath: Path to save templates
        """
        torch.save(self.user_templates, filepath)
        logger.info(f"Templates saved to {filepath}")

    def load_templates(self, filepath: str):
        """
        Load user templates from file

        Args:
            filepath: Path to load templates from
        """
        self.user_templates = torch.load(filepath)
        logger.info(f"Templates loaded from {filepath}")


