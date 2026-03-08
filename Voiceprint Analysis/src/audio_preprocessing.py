"""
Audio Preprocessing Module
Handles audio loading, segmentation, VAD, and noise reduction
"""
import torch
import torchaudio
import numpy as np
import soundfile as sf
# import webrtcvad  # Removed - requires C++ compiler
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
        
        # VAD settings (using energy-based VAD instead of webrtcvad)
        self.vad_threshold = self.config.get('audio.vad_threshold', 0.01)  # Energy threshold
        self._noisereduce_module = None
        self._noisereduce_import_attempted = False

    def _get_noisereduce(self):
        """Lazily import noisereduce to avoid startup crashes when optional deps are missing."""
        if self._noisereduce_import_attempted:
            return self._noisereduce_module

        self._noisereduce_import_attempted = True
        try:
            import noisereduce as nr  # Local import by design
            self._noisereduce_module = nr
        except Exception as e:
            print(f"Warning: noisereduce unavailable ({e}). Continuing without noise reduction.")
            self._noisereduce_module = None

        return self._noisereduce_module
    
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

            # soundfile.read returns:
            #   mono   → shape (samples,)     1-D  ✓
            #   stereo → shape (samples, ch)  2-D  — must average channels to mono
            if audio.ndim == 2:
                audio = audio.mean(axis=1)  # (samples, channels) → (samples,)

            # Resample if necessary
            if sr != self.sample_rate:
                # Use torchaudio resampling to avoid hard dependency on librosa at import time.
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                audio_tensor = torchaudio.functional.resample(audio_tensor, sr, self.sample_rate)
                audio = audio_tensor.squeeze(0).numpy()

            return audio, self.sample_rate

        except Exception as e:
            raise RuntimeError(f"Error loading audio from bytes: {str(e)}")
    
    def apply_vad(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply Voice Activity Detection to remove silence using energy-based method

        Args:
            audio: Audio array
            sample_rate: Sample rate

        Returns:
            Audio with silence removed
        """
        if not self.vad_enabled:
            return audio

        # Energy-based VAD (no C++ compilation required)
        # Frame duration in ms
        frame_duration_ms = 30
        frame_length = int(sample_rate * frame_duration_ms / 1000)

        # Handle short audio
        if len(audio) < frame_length:
            return audio

        # Calculate energy for each frame
        voiced_frames = []
        for i in range(0, len(audio) - frame_length, frame_length):
            frame = audio[i:i + frame_length]

            # Calculate frame energy (RMS)
            energy = np.sqrt(np.mean(frame ** 2))

            # Keep frame if energy is above threshold
            if energy > self.vad_threshold:
                voiced_frames.append(frame)

        if len(voiced_frames) == 0:
            print(f"Warning: VAD filtered out all audio. Returning original. Threshold: {self.vad_threshold}")
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
            nr = self._get_noisereduce()
            if nr is None:
                return audio

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

        # If audio is shorter than window, return it as a single segment
        if len(audio) <= window_samples:
            return [audio]

        segments = []
        for start in range(0, len(audio) - window_samples + 1, hop_samples):
            segment = audio[start:start + window_samples]
            segments.append(segment)

        # Add last segment if there's remaining audio
        if len(audio) > window_samples and (len(audio) - window_samples) % hop_samples != 0:
            segments.append(audio[-window_samples:])

        # Ensure we have at least one segment
        if len(segments) == 0:
            segments = [audio]

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

