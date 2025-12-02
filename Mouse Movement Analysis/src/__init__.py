"""
Mouse Movement Analysis Package
Zero Trust Telehealth Platform - Continuous Behavioral Authentication
"""

__version__ = "1.0.0"
__author__ = "Zero Trust Telehealth Team"

from .config_loader import load_config
from .mouse_preprocessing import MousePreprocessor
from .mouse_embedding import MouseEmbeddingModel
from .mouse_verification import MouseVerifier
from .anomaly_detection import AnomalyDetector

__all__ = [
    'load_config',
    'MousePreprocessor',
    'MouseEmbeddingModel',
    'MouseVerifier',
    'AnomalyDetector'
]

