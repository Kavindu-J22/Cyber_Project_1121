"""
Anomaly Detection Module
Detects unusual typing patterns and potential attacks
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope


class AnomalyDetector:
    """
    Anomaly detector for keystroke dynamics
    Identifies unusual typing patterns that may indicate attacks or unauthorized access
    """
    
    def __init__(self, config):
        """
        Initialize anomaly detector
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.method = config.anomaly_detection.method
        self.contamination = config.anomaly_detection.contamination
        
        # Initialize detector based on method
        if self.method == 'isolation_forest':
            self.detector = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.method == 'one_class_svm':
            self.detector = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='auto'
            )
        elif self.method == 'elliptic_envelope':
            self.detector = EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown anomaly detection method: {self.method}")
        
        self.is_fitted = False
        self.baseline_stats = {}
        
        logger.info(f"AnomalyDetector initialized with method: {self.method}")
    
    def fit(self, X: np.ndarray, user_id: str = None):
        """
        Fit anomaly detector on normal behavior
        
        Args:
            X: Training data (normal behavior)
            user_id: Optional user identifier
        """
        logger.info(f"Fitting anomaly detector on {len(X)} samples")
        
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        # Fit detector
        self.detector.fit(X)
        self.is_fitted = True
        
        # Compute baseline statistics
        self.baseline_stats = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'median': np.median(X, axis=0),
            'q25': np.percentile(X, 25, axis=0),
            'q75': np.percentile(X, 75, axis=0),
        }
        
        logger.info("Anomaly detector fitted successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies
        
        Args:
            X: Input data
            
        Returns:
            Predictions (1 for normal, -1 for anomaly)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        predictions = self.detector.predict(X)
        return predictions
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores
        
        Args:
            X: Input data
            
        Returns:
            Anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        scores = self.detector.score_samples(X)
        
        # Normalize scores to 0-1 range (higher = more normal)
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores_normalized
    
    def detect_anomaly(self, X: np.ndarray, threshold: float = None) -> Dict:
        """
        Detect if input is anomalous
        
        Args:
            X: Input sample
            threshold: Anomaly score threshold
            
        Returns:
            Detection result dictionary
        """
        if threshold is None:
            threshold = self.config.anomaly_detection.anomaly_score_threshold
        
        # Get prediction and score
        prediction = self.predict(X)[0]
        score = self.score_samples(X)[0]
        
        is_anomaly = (prediction == -1) or (score < threshold)
        
        # Analyze specific anomaly types
        anomaly_types = self.analyze_anomaly_types(X)
        
        result = {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(score),
            'prediction': int(prediction),
            'threshold': threshold,
            'anomaly_types': anomaly_types
        }
        
        if is_anomaly:
            logger.warning(f"Anomaly detected! Score: {score:.3f}, Types: {anomaly_types}")
        
        return result

    def analyze_anomaly_types(self, X: np.ndarray) -> List[str]:
        """
        Analyze specific types of anomalies

        Args:
            X: Input sample

        Returns:
            List of detected anomaly types
        """
        if not self.baseline_stats:
            return []

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        if X.ndim == 1:
            X = X.reshape(1, -1)

        anomaly_types = []

        # Speed anomaly detection
        if self.config.anomaly_detection.detect_speed_anomalies:
            mean_timing = np.mean(X)
            baseline_mean = np.mean(self.baseline_stats['mean'])

            if mean_timing < baseline_mean * 0.5:
                anomaly_types.append('typing_too_fast')
            elif mean_timing > baseline_mean * 2.0:
                anomaly_types.append('typing_too_slow')

        # Rhythm anomaly detection
        if self.config.anomaly_detection.detect_rhythm_anomalies:
            std_timing = np.std(X)
            baseline_std = np.mean(self.baseline_stats['std'])

            if std_timing > baseline_std * 2.0:
                anomaly_types.append('irregular_rhythm')

        # Pattern anomaly detection
        if self.config.anomaly_detection.detect_pattern_anomalies:
            # Check for outliers in individual features
            z_scores = np.abs((X - self.baseline_stats['mean']) / (self.baseline_stats['std'] + 1e-8))
            if np.any(z_scores > 3.0):
                anomaly_types.append('unusual_pattern')

        return anomaly_types

    def detect_consecutive_anomalies(self, anomaly_history: List[bool]) -> bool:
        """
        Detect consecutive anomalies that trigger alert

        Args:
            anomaly_history: List of recent anomaly flags

        Returns:
            Whether to trigger alert
        """
        threshold = self.config.anomaly_detection.consecutive_anomalies_alert

        if len(anomaly_history) < threshold:
            return False

        # Check last N samples
        recent = anomaly_history[-threshold:]
        consecutive_count = sum(recent)

        return consecutive_count >= threshold

    def get_anomaly_report(self, X_samples: np.ndarray) -> Dict:
        """
        Generate comprehensive anomaly report for multiple samples

        Args:
            X_samples: Multiple input samples

        Returns:
            Anomaly report dictionary
        """
        if isinstance(X_samples, torch.Tensor):
            X_samples = X_samples.cpu().numpy()

        predictions = self.predict(X_samples)
        scores = self.score_samples(X_samples)

        anomaly_count = np.sum(predictions == -1)
        anomaly_rate = anomaly_count / len(predictions)

        report = {
            'total_samples': len(X_samples),
            'anomaly_count': int(anomaly_count),
            'normal_count': int(len(predictions) - anomaly_count),
            'anomaly_rate': float(anomaly_rate),
            'mean_score': float(np.mean(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'std_score': float(np.std(scores)),
        }

        logger.info(f"Anomaly Report: {anomaly_count}/{len(X_samples)} anomalies "
                   f"({anomaly_rate*100:.1f}%)")

        return report

    def save_detector(self, filepath: str):
        """Save detector to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'detector': self.detector,
                'baseline_stats': self.baseline_stats,
                'is_fitted': self.is_fitted
            }, f)
        logger.info(f"Anomaly detector saved to {filepath}")

    def load_detector(self, filepath: str):
        """Load detector from file"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.detector = data['detector']
            self.baseline_stats = data['baseline_stats']
            self.is_fitted = data['is_fitted']
        logger.info(f"Anomaly detector loaded from {filepath}")

