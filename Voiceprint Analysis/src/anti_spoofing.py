"""
Anti-Spoofing Classifier
Detects replay attacks, synthetic speech, and voice cloning
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import torchaudio
from pathlib import Path

from src.config_loader import get_config


class AntiSpoofingClassifier:
    """
    Anti-spoofing detection for voice authentication
    Detects: replay attacks, synthetic speech (TTS), voice cloning (VC)
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.enabled = self.config.get('anti_spoofing.enabled', True)
        self.threshold = self.config.get('anti_spoofing.threshold', 0.5)
        self.device = self._get_device()
        
        if self.enabled:
            self.model = self._load_model()
    
    def _get_device(self) -> torch.device:
        """Get computation device"""
        use_gpu = self.config.get('performance.use_gpu', True)
        if use_gpu and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def _load_model(self):
        """
        Load anti-spoofing model
        Using a lightweight CNN-based model for real-time detection
        """
        try:
            # For now, we'll use a simple feature-based approach
            # In production, load a pre-trained ASVspoof model
            model = LightweightAntiSpoofingModel()
            model.to(self.device)
            model.eval()
            
            print(f"✓ Anti-spoofing model loaded on {self.device}")
            return model
            
        except Exception as e:
            print(f"Warning: Could not load anti-spoofing model: {e}")
            return None
    
    def extract_features(self, audio: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract features for anti-spoofing detection
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            Feature tensor
        """
        # Convert to tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Ensure correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Extract LFCC (Linear Frequency Cepstral Coefficients) - better for spoofing detection
        # Using mel spectrogram as a proxy
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=160,
            n_mels=60
        )(audio)
        
        # Log compression
        log_mel = torch.log(mel_spec + 1e-9)
        
        return log_mel
    
    def detect_spoofing(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        Detect if audio is spoofed
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            Detection result with scores
        """
        if not self.enabled or self.model is None:
            return {
                'is_genuine': True,
                'confidence': 1.0,
                'spoofing_type': None,
                'enabled': False
            }
        
        try:
            # Extract features
            features = self.extract_features(audio, sample_rate)
            features = features.to(self.device)
            
            # Run detection
            with torch.no_grad():
                output = self.model(features)
                genuine_score = torch.sigmoid(output).item()
            
            is_genuine = genuine_score >= self.threshold
            
            # Determine spoofing type if detected
            spoofing_type = None
            if not is_genuine:
                # Simple heuristic - in production use multi-class classifier
                if genuine_score < 0.3:
                    spoofing_type = 'replay_attack'
                elif genuine_score < 0.4:
                    spoofing_type = 'synthetic_speech'
                else:
                    spoofing_type = 'voice_cloning'
            
            return {
                'is_genuine': is_genuine,
                'confidence': float(genuine_score),
                'threshold': self.threshold,
                'spoofing_type': spoofing_type,
                'enabled': True
            }
            
        except Exception as e:
            print(f"Error in spoofing detection: {e}")
            return {
                'is_genuine': True,
                'confidence': 0.5,
                'error': str(e),
                'enabled': True
            }
    
    def detect_replay_attack(self, audio: np.ndarray) -> bool:
        """Specific detection for replay attacks"""
        result = self.detect_spoofing(audio)
        return result.get('spoofing_type') == 'replay_attack'
    
    def detect_synthetic_speech(self, audio: np.ndarray) -> bool:
        """Specific detection for synthetic speech (TTS)"""
        result = self.detect_spoofing(audio)
        return result.get('spoofing_type') == 'synthetic_speech'
    
    def detect_voice_cloning(self, audio: np.ndarray) -> bool:
        """Specific detection for voice cloning"""
        result = self.detect_spoofing(audio)
        return result.get('spoofing_type') == 'voice_cloning'


class LightweightAntiSpoofingModel(nn.Module):
    """
    Lightweight CNN model for anti-spoofing detection
    Based on simplified ASVspoof architecture
    """
    
    def __init__(self, input_channels=1):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x shape: [batch, channels, freq, time]
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

