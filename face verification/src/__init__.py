"""
Face Verification Module
Zero Trust Telehealth Platform
"""

__version__ = "1.0.0"
__author__ = "Zero Trust Telehealth Team"

from .face_model import ResNet50TripletModel, load_model_checkpoint
from .face_preprocessing import FacePreprocessor
from .face_verification import FaceVerificationEngine
from .config_loader import load_config, get_config

__all__ = [
    'ResNet50TripletModel',
    'load_model_checkpoint',
    'FacePreprocessor',
    'FaceVerificationEngine',
    'load_config',
    'get_config'
]
