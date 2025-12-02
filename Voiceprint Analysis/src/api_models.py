"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class EnrollmentRequest(BaseModel):
    """Request model for speaker enrollment"""
    speaker_id: str = Field(..., description="Unique identifier for the speaker (e.g., doctor ID)")
    audio_files: List[str] = Field(..., description="List of paths to enrollment audio files")


class EnrollmentResponse(BaseModel):
    """Response model for speaker enrollment"""
    speaker_id: str
    status: str
    enrollment_quality: float
    num_samples: int
    num_embeddings: int
    message: Optional[str] = None


class VerificationRequest(BaseModel):
    """Request model for speaker verification"""
    speaker_id: str = Field(..., description="Speaker ID to verify against")
    audio_file: str = Field(..., description="Path to verification audio file")


class VerificationResponse(BaseModel):
    """Response model for speaker verification"""
    speaker_id: str
    verified: bool
    confidence_score: float
    threshold: float
    num_segments: int
    latency_ms: float
    timestamp: str
    anti_spoofing: Optional[dict] = None
    alert_triggered: bool = False


class StreamingVerificationRequest(BaseModel):
    """Request model for streaming verification"""
    speaker_id: str
    session_id: str


class ContinuousVerificationResponse(BaseModel):
    """Response model for continuous verification"""
    speaker_id: str
    session_id: str
    total_windows: int
    verified_windows: int
    failed_windows: int
    average_confidence: float
    alerts: List[dict]
    results: List[dict]


class ThresholdUpdateRequest(BaseModel):
    """Request model for updating verification threshold"""
    threshold: float = Field(..., ge=0.0, le=1.0, description="New threshold value (0.0 to 1.0)")


class SpeakerInfoResponse(BaseModel):
    """Response model for speaker information"""
    speaker_id: str
    num_samples: int
    num_embeddings: int
    enrollment_quality: float
    enrolled_at: str
    embedding_dim: int


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str
    models_loaded: dict
    timestamp: str


class AlertResponse(BaseModel):
    """Response model for security alerts"""
    alert_id: str
    speaker_id: str
    alert_type: str
    confidence_score: float
    threshold: float
    timestamp: str
    session_id: Optional[str] = None
    details: Optional[dict] = None

