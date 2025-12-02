"""
REST API for Keystroke Dynamics System
Provides endpoints for enrollment, verification, and monitoring
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import numpy as np
from loguru import logger
import time
from datetime import datetime

from .config_loader import load_config
from .keystroke_preprocessing import KeystrokePreprocessor
from .keystroke_embedding import KeystrokeEmbeddingModel
from .keystroke_verification import KeystrokeVerifier
from .anomaly_detection import AnomalyDetector


# Pydantic models for API
class KeystrokeFeatures(BaseModel):
    """Keystroke timing features"""
    hold_times: List[float] = Field(..., description="Hold times for each key")
    dd_times: List[float] = Field(..., description="Keydown-keydown times")
    ud_times: List[float] = Field(..., description="Keyup-keydown times")
    timestamp: Optional[str] = Field(default=None, description="Timestamp of keystroke")


class EnrollmentRequest(BaseModel):
    """User enrollment request"""
    user_id: str = Field(..., description="Unique user identifier")
    keystroke_samples: List[List[float]] = Field(..., description="List of keystroke feature vectors")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class VerificationRequest(BaseModel):
    """User verification request"""
    user_id: str = Field(..., description="User identifier to verify")
    keystroke_sample: List[float] = Field(..., description="Single keystroke feature vector")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class ContinuousVerificationRequest(BaseModel):
    """Continuous verification request"""
    user_id: str = Field(..., description="User identifier")
    keystroke_stream: List[List[float]] = Field(..., description="Stream of keystroke samples")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class EnrollmentResponse(BaseModel):
    """Enrollment response"""
    success: bool
    user_id: str
    n_samples: int
    embedding_dim: int
    message: Optional[str] = None


class VerificationResponse(BaseModel):
    """Verification response"""
    verified: bool
    confidence: float
    confidence_level: str
    alert: bool
    critical: bool
    latency_ms: float
    user_id: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    enrolled_users: int
    timestamp: str


# Initialize FastAPI app
app = FastAPI(
    title="Keystroke Dynamics API",
    description="Zero Trust Telehealth - Continuous Authentication via Keystroke Dynamics",
    version="1.0.0"
)

# Global variables
config = None
model = None
verifier = None
preprocessor = None
anomaly_detector = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global config, model, verifier, preprocessor, anomaly_detector
    
    logger.info("Starting Keystroke Dynamics API...")
    
    # Load configuration
    config = load_config('config.yaml')
    
    # Add CORS middleware
    if config.api.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Initialize preprocessor
    preprocessor = KeystrokePreprocessor(config)
    
    # Initialize anomaly detector
    anomaly_detector = AnomalyDetector(config)
    
    # Load model
    try:
        # Determine input dimension (will be set after first data load)
        # For now, use a placeholder
        input_dim = 31  # DSL dataset has 31 timing features
        
        model = KeystrokeEmbeddingModel(input_dim, config)
        
        # Try to load checkpoint
        checkpoint_path = f"{config.paths.checkpoint_dir}/best_model.pth"
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model loaded from checkpoint")
        except:
            logger.warning("No checkpoint found, using untrained model")
        
        model.eval()
        
        # Initialize verifier
        verifier = KeystrokeVerifier(model, config)
        
        logger.info("Keystroke Dynamics API started successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "service": "Keystroke Dynamics API",
        "version": "1.0.0",
        "status": "running",
        "description": "Zero Trust Telehealth - Continuous Authentication"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        enrolled_users=len(verifier.enrolled_templates) if verifier else 0,
        timestamp=datetime.now().isoformat()
    )


@app.post("/enroll", response_model=EnrollmentResponse)
async def enroll_user(request: EnrollmentRequest):
    """
    Enroll a new user

    Creates a behavioral template from keystroke samples
    """
    try:
        logger.info(f"Enrollment request for user: {request.user_id}")

        # Convert samples to tensor
        samples_array = np.array(request.keystroke_samples, dtype=np.float32)
        samples_tensor = torch.FloatTensor(samples_array)

        # Enroll user
        result = verifier.enroll_user(request.user_id, samples_tensor)

        if result['success']:
            return EnrollmentResponse(
                success=True,
                user_id=result['user_id'],
                n_samples=result['n_samples'],
                embedding_dim=result['embedding_dim'],
                message="User enrolled successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get('message', 'Enrollment failed')
            )

    except Exception as e:
        logger.error(f"Enrollment error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/verify", response_model=VerificationResponse)
async def verify_user(request: VerificationRequest):
    """
    Verify user identity

    Compares keystroke sample against enrolled template
    """
    try:
        logger.info(f"Verification request for user: {request.user_id}")

        # Convert sample to tensor
        sample_array = np.array(request.keystroke_sample, dtype=np.float32)
        sample_tensor = torch.FloatTensor(sample_array)

        # Verify user
        result = verifier.verify_user(request.user_id, sample_tensor)

        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()

        # Log alert if needed
        if result['alert']:
            logger.warning(f"ALERT: Low confidence for user {request.user_id}: {result['confidence']:.3f}")

        if result['critical']:
            logger.error(f"CRITICAL: Very low confidence for user {request.user_id}: {result['confidence']:.3f}")

        return VerificationResponse(**result)

    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/continuous-verify")
async def continuous_verify(request: ContinuousVerificationRequest):
    """
    Continuous verification on keystroke stream

    Analyzes multiple keystroke samples for ongoing authentication
    """
    try:
        logger.info(f"Continuous verification for user: {request.user_id}")

        # Convert stream to tensors
        stream_tensors = [
            torch.FloatTensor(np.array(sample, dtype=np.float32))
            for sample in request.keystroke_stream
        ]

        # Perform continuous verification
        results = verifier.continuous_verification(request.user_id, stream_tensors)

        # Add timestamps
        for result in results:
            result['timestamp'] = datetime.now().isoformat()

        return {
            "user_id": request.user_id,
            "total_windows": len(results),
            "results": results,
            "overall_verified": all(r['verified'] for r in results),
            "mean_confidence": np.mean([r['confidence'] for r in results])
        }

    except Exception as e:
        logger.error(f"Continuous verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/statistics/{user_id}")
async def get_statistics(user_id: str):
    """Get verification statistics for a user"""
    try:
        stats = verifier.get_verification_statistics(user_id)
        return stats
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/detect-anomaly")
async def detect_anomaly(keystroke_sample: List[float]):
    """Detect anomalies in keystroke pattern"""
    try:
        sample_array = np.array(keystroke_sample, dtype=np.float32)

        if not anomaly_detector.is_fitted:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Anomaly detector not trained"
            )

        result = anomaly_detector.detect_anomaly(sample_array)
        return result

    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/enrolled-users")
async def get_enrolled_users():
    """Get list of enrolled users"""
    return {
        "enrolled_users": list(verifier.enrolled_templates.keys()),
        "count": len(verifier.enrolled_templates)
    }


if __name__ == "__main__":
    import uvicorn

    config = load_config('config.yaml')

    uvicorn.run(
        "api:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=config.api.reload,
        log_level=config.api.log_level
    )

