"""
FastAPI Server for Mouse Movement Analysis
Provides REST API endpoints for enrollment, verification, and continuous monitoring
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import numpy as np
from loguru import logger
import os
import time

from .config_loader import load_config
from .mouse_preprocessing import MousePreprocessor
from .mouse_embedding import MouseEmbeddingModel
from .mouse_verification import MouseVerifier
from .anomaly_detection import AnomalyDetector


# Load configuration
config = load_config('config.yaml')

# Initialize FastAPI app
app = FastAPI(
    title="Mouse Movement Analysis API",
    description="Zero Trust Telehealth Platform - Continuous Behavioral Authentication",
    version="1.0.0"
)

# CORS middleware
if config.api.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Global components
preprocessor = None
model = None
verifier = None
anomaly_detector = None


# Pydantic models
class MouseEvent(BaseModel):
    """Single mouse event"""
    timestamp: float
    x: int
    y: int
    button: str = "NoButton"
    state: str = "Move"


class EnrollmentRequest(BaseModel):
    """Enrollment request"""
    user_id: str
    events: List[MouseEvent]


class VerificationRequest(BaseModel):
    """Verification request"""
    user_id: str
    events: List[MouseEvent]


class ContinuousMonitoringRequest(BaseModel):
    """Continuous monitoring request"""
    user_id: str
    events: List[MouseEvent]
    session_id: Optional[str] = None


class EnrollmentResponse(BaseModel):
    """Enrollment response"""
    user_id: str
    enrolled: bool
    num_samples: int
    message: str


class VerificationResponse(BaseModel):
    """Verification response"""
    user_id: str
    verified: bool
    confidence: float
    confidence_level: str
    threshold: float
    timestamp: float


class ContinuousMonitoringResponse(BaseModel):
    """Continuous monitoring response"""
    user_id: str
    overall_verified: bool
    mean_confidence: float
    verification_rate: float
    anomaly_detected: bool
    risk_level: str
    timestamp: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: float


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global preprocessor, model, verifier, anomaly_detector
    
    logger.info("Initializing Mouse Movement Analysis API...")
    
    # Initialize preprocessor
    preprocessor = MousePreprocessor(config)
    
    # Load model
    checkpoint_path = os.path.join(config.paths.checkpoint_dir, 'best_model.pth')
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading model from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get input dimension (we'll set a default and update when we get first data)
        # For now, use a placeholder
        input_dim = 50  # Will be updated dynamically
        
        model = MouseEmbeddingModel(input_dim, config)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            model = None
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}")
        model = None
    
    # Initialize verifier
    if model is not None:
        verifier = MouseVerifier(model, config)
        logger.info("Verifier initialized")
    
    # Initialize anomaly detector
    anomaly_detector = AnomalyDetector(config)
    logger.info("Anomaly detector initialized")
    
    logger.info("API initialization complete")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }


def events_to_features(events: List[MouseEvent]) -> Optional[np.ndarray]:
    """Convert mouse events to feature vector"""
    if len(events) < config.features.min_events:
        return None

    # Create DataFrame from events
    import pandas as pd

    data = {
        'client_timestamp': [e.timestamp for e in events],
        'x': [e.x for e in events],
        'y': [e.y for e in events],
        'button': [e.button for e in events],
        'state': [e.state for e in events]
    }

    df = pd.DataFrame(data)

    # Extract features using preprocessor
    features = preprocessor._extract_window_features(df)

    return features


@app.post("/enroll", response_model=EnrollmentResponse)
async def enroll_user(request: EnrollmentRequest):
    """
    Enroll a user with mouse movement samples

    Args:
        request: Enrollment request with user_id and mouse events

    Returns:
        Enrollment response
    """
    if model is None or verifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info(f"Enrollment request for user: {request.user_id}")

    # Convert events to features
    features = events_to_features(request.events)

    if features is None:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient events. Need at least {config.features.min_events}"
        )

    # Normalize features
    features = features.reshape(1, -1)

    # Update model input dimension if needed
    if features.shape[1] != model.input_dim:
        logger.warning(f"Feature dimension mismatch: {features.shape[1]} vs {model.input_dim}")

    features_normalized = preprocessor.normalize_features(features, fit=True)
    features_tensor = torch.FloatTensor(features_normalized)

    # Enroll user
    try:
        result = verifier.enroll_user(request.user_id, features_tensor)

        return EnrollmentResponse(
            user_id=request.user_id,
            enrolled=True,
            num_samples=len(features_tensor),
            message="User enrolled successfully"
        )
    except Exception as e:
        logger.error(f"Enrollment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify", response_model=VerificationResponse)
async def verify_user(request: VerificationRequest):
    """
    Verify a user against their enrolled template

    Args:
        request: Verification request with user_id and mouse events

    Returns:
        Verification response
    """
    if model is None or verifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info(f"Verification request for user: {request.user_id}")

    # Convert events to features
    features = events_to_features(request.events)

    if features is None:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient events. Need at least {config.features.min_events}"
        )

    # Normalize features
    features = features.reshape(1, -1)
    features_normalized = preprocessor.normalize_features(features, fit=False)
    features_tensor = torch.FloatTensor(features_normalized)

    # Verify user
    try:
        result = verifier.verify_user(request.user_id, features_tensor[0])

        return VerificationResponse(
            user_id=request.user_id,
            verified=result['verified'],
            confidence=result['confidence'],
            confidence_level=result['confidence_level'],
            threshold=result['threshold'],
            timestamp=time.time()
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/monitor", response_model=ContinuousMonitoringResponse)
async def continuous_monitoring(request: ContinuousMonitoringRequest):
    """
    Continuous monitoring of user behavior

    Args:
        request: Monitoring request with user_id and mouse events

    Returns:
        Monitoring response with verification and anomaly detection results
    """
    if model is None or verifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info(f"Continuous monitoring for user: {request.user_id}")

    # Convert events to features
    features = events_to_features(request.events)

    if features is None:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient events. Need at least {config.features.min_events}"
        )

    # Normalize features
    features = features.reshape(1, -1)
    features_normalized = preprocessor.normalize_features(features, fit=False)
    features_tensor = torch.FloatTensor(features_normalized)

    # Continuous verification
    try:
        result = verifier.continuous_verify(request.user_id, features_tensor)

        # Anomaly detection
        anomaly_detected = False
        risk_level = "low"

        if anomaly_detector is not None:
            # Get recent confidences
            stats = verifier.get_verification_statistics(request.user_id)
            recent_confidences = stats.get('confidences', [])

            # Check for anomalies
            if len(recent_confidences) > 0:
                current_confidence = result['mean_confidence']
                anomaly_detected = anomaly_detector.detect_user_substitution(
                    current_confidence, recent_confidences
                )

            # Determine risk level
            if result['mean_confidence'] < config.verification.critical_threshold:
                risk_level = "critical"
            elif result['mean_confidence'] < config.verification.alert_threshold:
                risk_level = "high"
            elif result['mean_confidence'] < config.verification.threshold:
                risk_level = "medium"

        return ContinuousMonitoringResponse(
            user_id=request.user_id,
            overall_verified=result['overall_verified'],
            mean_confidence=result['mean_confidence'],
            verification_rate=result['verification_rate'],
            anomaly_detected=anomaly_detected,
            risk_level=risk_level,
            timestamp=time.time()
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/stats")
async def get_user_stats(user_id: str):
    """Get verification statistics for a user"""
    if verifier is None:
        raise HTTPException(status_code=503, detail="Verifier not initialized")

    try:
        stats = verifier.get_verification_statistics(user_id)
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/users/{user_id}")
async def delete_user(user_id: str):
    """Delete user enrollment"""
    if verifier is None:
        raise HTTPException(status_code=503, detail="Verifier not initialized")

    try:
        verifier.reset_user(user_id)
        return {"message": f"User {user_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.api.host, port=config.api.port)

