"""
FastAPI REST API for Voiceprint Analysis
Provides endpoints for enrollment, verification, and continuous authentication
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn
import logging
from datetime import datetime
import json
import asyncio
import uuid

from src.config_loader import get_config
from src.speaker_verification import SpeakerVerificationEngine
from src.anti_spoofing import AntiSpoofingClassifier
from src.api_models import (
    EnrollmentRequest, EnrollmentResponse,
    VerificationRequest, VerificationResponse,
    ContinuousVerificationResponse, ThresholdUpdateRequest,
    SpeakerInfoResponse, HealthCheckResponse, AlertResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Voiceprint Analysis API",
    description="Zero Trust Continuous Speaker Verification for Telehealth",
    version="1.0.0"
)

# Load configuration
config = get_config()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get('api.cors_origins', ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize verification engine and anti-spoofing
verification_engine: Optional[SpeakerVerificationEngine] = None
anti_spoofing: Optional[AntiSpoofingClassifier] = None

# Alert storage (in production, use database)
alerts_storage = []


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global verification_engine, anti_spoofing
    
    logger.info("Initializing Voiceprint Analysis System...")
    
    try:
        verification_engine = SpeakerVerificationEngine(config)
        logger.info("✓ Speaker verification engine initialized")
        
        anti_spoofing = AntiSpoofingClassifier(config)
        logger.info("✓ Anti-spoofing classifier initialized")
        
        logger.info("🚀 Voiceprint Analysis API is ready!")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise


@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint - health check"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": {
            "verification_engine": verification_engine is not None,
            "anti_spoofing": anti_spoofing is not None
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return await root()


@app.post("/api/v1/enroll", response_model=EnrollmentResponse)
async def enroll_speaker(request: EnrollmentRequest):
    """
    Enroll a new speaker with multiple audio samples
    
    - **speaker_id**: Unique identifier (e.g., doctor ID)
    - **audio_files**: List of paths to enrollment audio files (minimum 3)
    """
    try:
        if verification_engine is None:
            raise HTTPException(status_code=503, detail="Verification engine not initialized")
        
        result = verification_engine.enroll_speaker(
            speaker_id=request.speaker_id,
            audio_samples=request.audio_files
        )
        
        logger.info(f"Speaker {request.speaker_id} enrolled successfully")
        
        return EnrollmentResponse(
            **result,
            message="Speaker enrolled successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Enrollment error: {e}")
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")


@app.post("/api/v1/verify", response_model=VerificationResponse)
async def verify_speaker(request: VerificationRequest):
    """
    Verify speaker identity from audio file
    
    - **speaker_id**: Speaker ID to verify against
    - **audio_file**: Path to verification audio file
    """
    try:
        if verification_engine is None:
            raise HTTPException(status_code=503, detail="Verification engine not initialized")
        
        # Perform verification
        result = verification_engine.verify_speaker(
            speaker_id=request.speaker_id,
            audio_path=request.audio_file
        )
        
        # Run anti-spoofing check
        anti_spoof_result = None
        if anti_spoofing and anti_spoofing.enabled:
            from src.audio_preprocessing import AudioPreprocessor
            preprocessor = AudioPreprocessor(config)
            audio, _ = preprocessor.load_audio(request.audio_file)
            anti_spoof_result = anti_spoofing.detect_spoofing(audio)
        
        # Trigger alert if verification fails
        alert_triggered = not result['verified']
        if alert_triggered:
            await trigger_alert(request.speaker_id, result, anti_spoof_result)
        
        return VerificationResponse(
            **result,
            anti_spoofing=anti_spoof_result,
            alert_triggered=alert_triggered
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.post("/api/v1/verify/upload")
async def verify_speaker_upload(
    speaker_id: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """
    Verify speaker identity from uploaded audio file
    
    - **speaker_id**: Speaker ID to verify against
    - **audio_file**: Audio file upload
    """
    try:
        if verification_engine is None:
            raise HTTPException(status_code=503, detail="Verification engine not initialized")
        
        # Read audio bytes
        audio_bytes = await audio_file.read()
        
        # Perform verification
        result = verification_engine.verify_speaker_streaming(
            speaker_id=speaker_id,
            audio_bytes=audio_bytes
        )

        # Run anti-spoofing check
        anti_spoof_result = None
        if anti_spoofing and anti_spoofing.enabled:
            from src.audio_preprocessing import AudioPreprocessor
            preprocessor = AudioPreprocessor(config)
            audio, _ = preprocessor.load_audio_from_bytes(audio_bytes)
            anti_spoof_result = anti_spoofing.detect_spoofing(audio)

        # Trigger alert if verification fails
        alert_triggered = not result.get('verified', False)
        if alert_triggered:
            await trigger_alert(speaker_id, result, anti_spoof_result)

        return {
            **result,
            "anti_spoofing": anti_spoof_result,
            "alert_triggered": alert_triggered
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.post("/api/v1/continuous-verify")
async def continuous_verification(
    speaker_id: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """
    Perform continuous verification on audio stream

    - **speaker_id**: Speaker ID to verify
    - **audio_file**: Audio stream file
    """
    try:
        if verification_engine is None:
            raise HTTPException(status_code=503, detail="Verification engine not initialized")

        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(await audio_file.read())
            tmp_path = tmp_file.name

        # Perform continuous verification
        results = verification_engine.continuous_verification(
            speaker_id=speaker_id,
            audio_stream_path=tmp_path
        )

        # Analyze results
        total_windows = len(results)
        verified_windows = sum(1 for r in results if r['verified'])
        failed_windows = total_windows - verified_windows
        average_confidence = sum(r['confidence_score'] for r in results) / total_windows if total_windows > 0 else 0

        # Collect alerts
        alerts = [r for r in results if r.get('alert_triggered', False)]

        # Trigger alerts for failed verifications
        for alert in alerts:
            await trigger_alert(speaker_id, alert, None)

        # Clean up temp file
        import os
        os.unlink(tmp_path)

        return ContinuousVerificationResponse(
            speaker_id=speaker_id,
            session_id=str(uuid.uuid4()),
            total_windows=total_windows,
            verified_windows=verified_windows,
            failed_windows=failed_windows,
            average_confidence=average_confidence,
            alerts=alerts,
            results=results
        )

    except Exception as e:
        logger.error(f"Continuous verification error: {e}")
        raise HTTPException(status_code=500, detail=f"Continuous verification failed: {str(e)}")


@app.put("/api/v1/threshold")
async def update_threshold(request: ThresholdUpdateRequest):
    """
    Update verification threshold dynamically

    - **threshold**: New threshold value (0.0 to 1.0)
    """
    try:
        if verification_engine is None:
            raise HTTPException(status_code=503, detail="Verification engine not initialized")

        verification_engine.update_threshold(request.threshold)

        return {
            "status": "success",
            "message": f"Threshold updated to {request.threshold}",
            "new_threshold": request.threshold
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/speakers")
async def get_enrolled_speakers():
    """Get list of all enrolled speakers"""
    try:
        if verification_engine is None:
            raise HTTPException(status_code=503, detail="Verification engine not initialized")

        speakers = verification_engine.get_enrolled_speakers()

        return {
            "total_speakers": len(speakers),
            "speakers": speakers
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/speakers/{speaker_id}", response_model=SpeakerInfoResponse)
async def get_speaker_info(speaker_id: str):
    """Get enrollment information for a specific speaker"""
    try:
        if verification_engine is None:
            raise HTTPException(status_code=503, detail="Verification engine not initialized")

        info = verification_engine.get_speaker_info(speaker_id)

        if info is None:
            raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")

        return SpeakerInfoResponse(**info)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/speakers/{speaker_id}")
async def remove_speaker(speaker_id: str):
    """Remove an enrolled speaker"""
    try:
        if verification_engine is None:
            raise HTTPException(status_code=503, detail="Verification engine not initialized")

        removed = verification_engine.remove_speaker(speaker_id)

        if not removed:
            raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")

        return {
            "status": "success",
            "message": f"Speaker {speaker_id} removed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/alerts")
async def get_alerts(limit: int = 100):
    """Get recent security alerts"""
    try:
        return {
            "total_alerts": len(alerts_storage),
            "alerts": alerts_storage[-limit:]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def trigger_alert(speaker_id: str, verification_result: dict, anti_spoof_result: dict = None):
    """Trigger security alert when verification fails"""
    alert = {
        "alert_id": str(uuid.uuid4()),
        "speaker_id": speaker_id,
        "alert_type": "verification_failed",
        "confidence_score": verification_result.get('confidence_score', 0.0),
        "threshold": verification_result.get('threshold', 0.0),
        "timestamp": datetime.utcnow().isoformat(),
        "details": {
            "verification": verification_result,
            "anti_spoofing": anti_spoof_result
        }
    }

    alerts_storage.append(alert)
    logger.warning(f"🚨 ALERT: Verification failed for speaker {speaker_id}")

    # In production, send email/SMS alerts here
    return alert


# WebSocket endpoint for real-time streaming verification
@app.websocket("/ws/verify/{speaker_id}")
async def websocket_verify(websocket: WebSocket, speaker_id: str):
    """
    WebSocket endpoint for real-time continuous verification

    Client sends audio chunks, server responds with verification results
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established for speaker {speaker_id}")

    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()

            # Perform verification
            result = verification_engine.verify_speaker_streaming(
                speaker_id=speaker_id,
                audio_bytes=data
            )

            # Send result back
            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for speaker {speaker_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


if __name__ == "__main__":
    # Run the API server
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 8001)

    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=config.get('api.reload', False),
        log_level=config.get('api.log_level', 'info')
    )
