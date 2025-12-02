"""
Anomaly Detection Module
Detects automated behavior, remote desktop artifacts, and user substitution
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import torch


class AnomalyDetector:
    """
    Anomaly detector for mouse movement patterns
    Detects automated scripts, remote desktop usage, and unauthorized access
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
        else:
            self.detector = None
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Anomaly history
        self.anomaly_history = []
        
        logger.info(f"AnomalyDetector initialized with method: {self.method}")
    
    def fit(self, X: np.ndarray):
        """
        Fit anomaly detector on normal data
        
        Args:
            X: Normal behavior features
        """
        logger.info(f"Fitting anomaly detector on {len(X)} samples")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit detector
        if self.detector is not None:
            self.detector.fit(X_scaled)
        
        self.is_fitted = True
        logger.info("Anomaly detector fitted")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies
        
        Args:
            X: Feature samples
            
        Returns:
            Array of predictions (1 for normal, -1 for anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.detector.predict(X_scaled)
        
        return predictions
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores
        
        Args:
            X: Feature samples
            
        Returns:
            Array of anomaly scores (lower is more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get scores
        if hasattr(self.detector, 'score_samples'):
            scores = self.detector.score_samples(X_scaled)
        else:
            scores = self.detector.decision_function(X_scaled)
        
        return scores
    
    def detect_automated_behavior(self, features: Dict[str, float]) -> bool:
        """
        Detect automated/scripted mouse behavior
        
        Args:
            features: Extracted movement features
            
        Returns:
            True if automated behavior detected
        """
        if not self.config.anomaly_detection.detect_automated_behavior:
            return False
        
        # Check for unnatural patterns
        anomalies = []
        
        # 1. Constant velocity (robotic movement)
        if self.config.anomaly_detection.detect_constant_velocity:
            if 'velocity_std' in features and features['velocity_std'] < 1.0:
                anomalies.append('constant_velocity')
        
        # 2. Perfect linear movements
        if self.config.anomaly_detection.detect_linear_movements:
            if 'trajectory_efficiency' in features and features['trajectory_efficiency'] > 0.98:
                anomalies.append('linear_movement')
        
        # 3. Impossible speed
        if self.config.anomaly_detection.detect_impossible_speed:
            if 'velocity_max' in features and features['velocity_max'] > 10000:  # pixels/sec
                anomalies.append('impossible_speed')
        
        if anomalies:
            logger.warning(f"Automated behavior detected: {anomalies}")
            return True
        
        return False
    
    def detect_remote_desktop(self, features: Dict[str, float]) -> bool:
        """
        Detect remote desktop artifacts
        
        Args:
            features: Extracted movement features
            
        Returns:
            True if RDP artifacts detected
        """
        if not self.config.anomaly_detection.detect_remote_desktop:
            return False
        
        # RDP often introduces latency and jitter
        anomalies = []
        
        # Check for high jitter
        if 'jerk_std' in features and features['jerk_std'] > 1000:
            anomalies.append('high_jitter')
        
        # Check for unusual pause patterns
        if 'num_pauses' in features and features['num_pauses'] > 50:
            anomalies.append('excessive_pauses')
        
        if anomalies:
            logger.warning(f"Remote desktop artifacts detected: {anomalies}")
            return True

        return False

    def detect_user_substitution(self, current_confidence: float,
                                 recent_confidences: List[float]) -> bool:
        """
        Detect potential user substitution

        Args:
            current_confidence: Current verification confidence
            recent_confidences: Recent confidence scores

        Returns:
            True if substitution suspected
        """
        if not self.config.anomaly_detection.detect_substitution:
            return False

        # Check for sudden drop in confidence
        if len(recent_confidences) >= 5:
            mean_recent = np.mean(recent_confidences[-5:])

            # Significant drop
            if mean_recent > 0.8 and current_confidence < 0.5:
                logger.warning(f"Sudden confidence drop: {mean_recent:.3f} -> {current_confidence:.3f}")
                return True

        # Check for consecutive low confidences
        if len(recent_confidences) >= self.config.anomaly_detection.consecutive_anomalies_alert:
            recent = recent_confidences[-self.config.anomaly_detection.consecutive_anomalies_alert:]
            if all(c < self.config.verification.alert_threshold for c in recent):
                logger.warning(f"Consecutive low confidences detected")
                return True

        return False

    def analyze_session(self, features_list: List[Dict[str, float]],
                       confidences: List[float]) -> Dict:
        """
        Comprehensive session analysis

        Args:
            features_list: List of feature dictionaries
            confidences: List of confidence scores

        Returns:
            Analysis results
        """
        results = {
            'automated_behavior': False,
            'remote_desktop': False,
            'user_substitution': False,
            'anomaly_score': 0.0,
            'risk_level': 'low'
        }

        # Check each feature set
        automated_count = 0
        rdp_count = 0

        for features in features_list:
            if self.detect_automated_behavior(features):
                automated_count += 1
            if self.detect_remote_desktop(features):
                rdp_count += 1

        # Determine if anomalies are significant
        if automated_count > len(features_list) * 0.3:
            results['automated_behavior'] = True

        if rdp_count > len(features_list) * 0.3:
            results['remote_desktop'] = True

        # Check for user substitution
        if len(confidences) > 0:
            for i in range(len(confidences)):
                recent = confidences[:i] if i > 0 else []
                if self.detect_user_substitution(confidences[i], recent):
                    results['user_substitution'] = True
                    break

        # Calculate overall anomaly score
        anomaly_score = 0.0
        if results['automated_behavior']:
            anomaly_score += 0.4
        if results['remote_desktop']:
            anomaly_score += 0.3
        if results['user_substitution']:
            anomaly_score += 0.3

        results['anomaly_score'] = anomaly_score

        # Determine risk level
        if anomaly_score >= 0.7:
            results['risk_level'] = 'critical'
        elif anomaly_score >= 0.4:
            results['risk_level'] = 'high'
        elif anomaly_score >= 0.2:
            results['risk_level'] = 'medium'
        else:
            results['risk_level'] = 'low'

        return results

    def log_anomaly(self, anomaly_type: str, details: Dict):
        """
        Log anomaly for audit trail

        Args:
            anomaly_type: Type of anomaly
            details: Anomaly details
        """
        import time

        record = {
            'timestamp': time.time(),
            'type': anomaly_type,
            'details': details
        }

        self.anomaly_history.append(record)

        logger.warning(f"Anomaly logged: {anomaly_type} - {details}")

    def get_anomaly_history(self, limit: int = 100) -> List[Dict]:
        """
        Get recent anomaly history

        Args:
            limit: Maximum number of records to return

        Returns:
            List of anomaly records
        """
        return self.anomaly_history[-limit:]

    def clear_history(self):
        """Clear anomaly history"""
        self.anomaly_history = []
        logger.info("Anomaly history cleared")

