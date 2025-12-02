"""
Audio Preprocessing Module
Handles audio loading, segmentation, VAD, and noise reduction
"""
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
import webrtcvad
import noisereduce as nr
from typing import Tuple, List, Optional
from pathlib import Path
import io

from src.config_loader import get_config


class AudioPreprocessor:
    """Audio preprocessing for speaker verification"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.sample_rate = self.config.get('audio.sample_rate', 16000)
        self.window_duration = self.config.get('audio.window_duration', 2.5)
        self.window_overlap = self.config.get('audio.window_overlap', 0.5)
        self.vad_enabled = self.config.get('audio.vad_enabled', True)
        self.noise_reduction = self.config.get('audio.noise_reduction', True)
        
        # Initialize VAD
        if self.vad_enabled:
            self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (0-3)
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Load audio using torchaudio
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to numpy
            audio = waveform.squeeze().numpy()
            
            return audio, self.sample_rate
            
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {audio_path}: {str(e)}")
    
    def load_audio_from_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Load audio from bytes (for real-time streaming)
        
        Args:
            audio_bytes: Audio data as bytes
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Load from bytes
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            
            # Resample if necessary
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            return audio, self.sample_rate
            
        except Exception as e:
            raise RuntimeError(f"Error loading audio from bytes: {str(e)}")
    
    def apply_vad(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply Voice Activity Detection to remove silence
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            Audio with silence removed
        """
        if not self.vad_enabled:
            return audio
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32768).astype(np.int16)
        
        # Frame duration in ms (10, 20, or 30 ms for WebRTC VAD)
        frame_duration_ms = 30
        frame_length = int(sample_rate * frame_duration_ms / 1000)
        
        # Process frames
        voiced_frames = []
        for i in range(0, len(audio_int16) - frame_length, frame_length):
            frame = audio_int16[i:i + frame_length]
            
            # VAD requires exactly frame_length samples
            if len(frame) == frame_length:
                is_speech = self.vad.is_speech(frame.tobytes(), sample_rate)
                if is_speech:
                    voiced_frames.append(audio[i:i + frame_length])
        
        if len(voiced_frames) == 0:
            return audio  # Return original if no speech detected
        
        return np.concatenate(voiced_frames)
    
    def reduce_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply noise reduction
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            Denoised audio
        """
        if not self.noise_reduction:
            return audio
        
        try:
            # Apply noise reduction
            reduced_noise = nr.reduce_noise(y=audio, sr=sample_rate, stationary=True)
            return reduced_noise
        except:
            return audio  # Return original if noise reduction fails
    
    def segment_audio(self, audio: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """
        Segment audio into overlapping windows of 2-3 seconds

        Args:
            audio: Audio array
            sample_rate: Sample rate

        Returns:
            List of audio segments
        """
        window_samples = int(self.window_duration * sample_rate)
        hop_samples = int(window_samples * (1 - self.window_overlap))

        segments = []
        for start in range(0, len(audio) - window_samples + 1, hop_samples):
            segment = audio[start:start + window_samples]
            segments.append(segment)

        # Add last segment if there's remaining audio
        if len(audio) > window_samples and (len(audio) - window_samples) % hop_samples != 0:
            segments.append(audio[-window_samples:])

        return segments

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range

        Args:
            audio: Audio array

        Returns:
            Normalized audio
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio

    def preprocess(self, audio_path: str) -> List[np.ndarray]:
        """
        Complete preprocessing pipeline

        Args:
            audio_path: Path to audio file

        Returns:
            List of preprocessed audio segments
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)

        # Apply noise reduction
        audio = self.reduce_noise(audio, sr)

        # Apply VAD
        audio = self.apply_vad(audio, sr)

        # Normalize
        audio = self.normalize_audio(audio)

        # Segment into windows
        segments = self.segment_audio(audio, sr)

        return segments

    def preprocess_streaming(self, audio_bytes: bytes) -> List[np.ndarray]:
        """
        Preprocessing pipeline for streaming audio

        Args:
            audio_bytes: Audio data as bytes

        Returns:
            List of preprocessed audio segments
        """
        # Load from bytes
        audio, sr = self.load_audio_from_bytes(audio_bytes)

        # Apply noise reduction
        audio = self.reduce_noise(audio, sr)

        # Apply VAD
        audio = self.apply_vad(audio, sr)

        # Normalize
        audio = self.normalize_audio(audio)

        # Segment into windows
        segments = self.segment_audio(audio, sr)

        return segments

