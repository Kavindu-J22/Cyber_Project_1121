"""
FastAPI Server for Face Verification
RESTful API for enrollment and verification
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
from pathlib import Path
from loguru import logger
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import get_config
from src.face_verification import FaceVerificationEngine

# Initialize configuration
config = get_config()

# Persistence path — enrollments survive restarts
ENROLLMENTS_PATH = Path(__file__).parent.parent / 'data' / 'enrollments.pkl'

# Initialize FastAPI app
app = FastAPI(
    title="Face Verification API",
    description="ResNet50 Triplet Face Verification for Zero Trust Telehealth",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get('api.cors_origins', ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize verification engine
verification_engine: Optional[FaceVerificationEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initialize verification engine on startup"""
    global verification_engine
    
    logger.info("="*70)
    logger.info("🔐 Face Verification API - Zero Trust Telehealth Platform")
    logger.info("="*70)
    
    try:
        verification_engine = FaceVerificationEngine(config)
        logger.info("✓ Verification engine initialized")
        # Load persisted enrollments if available
        if ENROLLMENTS_PATH.exists():
            try:
                verification_engine.load_enrollments(str(ENROLLMENTS_PATH))
            except Exception as load_err:
                logger.warning(f"Could not load persisted enrollments: {load_err}")
    except Exception as e:
        logger.error(f"✗ Failed to initialize verification engine: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Face Verification API",
        "version": "1.0.0",
        "status": "running",
        "model": config.get('model.type'),
        "embedding_dim": config.get('model.embedding_dim')
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": verification_engine is not None,
        "device": verification_engine.device if verification_engine else "unknown",
        "threshold": config.get('verification.threshold')
    }


@app.get("/api/config")
async def get_configuration():
    """Get API configuration"""
    return {
        "default_threshold": config.get('verification.threshold'),
        "face_size": config.get('image.face_size'),
        "embed_dim": config.get('model.embedding_dim'),
        "device": verification_engine.device if verification_engine else "unknown",
        "similarity_metric": config.get('verification.similarity_metric'),
        "enrollment_samples": config.get('verification.enrollment_samples')
    }


@app.post("/api/v1/enroll")
async def enroll_user(
    user_id: str = Form(...),
    face_samples: List[UploadFile] = File(...)
):
    """
    Enroll user with multiple face samples
    
    Args:
        user_id: Unique user identifier
        face_samples: List of face image files (3+ recommended)
        
    Returns:
        Enrollment result with quality metrics
    """
    if verification_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verification engine not initialized"
        )
    
    if not face_samples or len(face_samples) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one face sample is required"
        )
    
    logger.info(f"Enrollment request for user: {user_id} ({len(face_samples)} samples)")
    
    # Save uploaded files temporarily
    temp_files = []
    try:
        for upload_file in face_samples:
            # Validate file type
            if not upload_file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid file type: {upload_file.content_type}"
                )
            
            # Read file content
            content = await upload_file.read()
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(content)
                temp_files.append(temp_file.name)
        
        # Perform enrollment
        result = verification_engine.enroll_user(user_id, temp_files)

        logger.info(f"✓ Enrollment successful for user: {user_id}")

        # Persist enrollments to disk so they survive restarts
        try:
            verification_engine.save_enrollments(str(ENROLLMENTS_PATH))
        except Exception as save_err:
            logger.warning(f"Could not persist enrollments: {save_err}")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "message": "User enrolled successfully",
                "data": result
            }
        )
    
    except Exception as e:
        logger.error(f"✗ Enrollment failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enrollment failed: {str(e)}"
        )
    
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass


@app.post("/api/v1/verify")
async def verify_user(
    user_id: str = Form(...),
    face_sample: UploadFile = File(...),
    threshold: Optional[float] = Form(None)
):
    """
    Verify user against enrolled face
    
    Args:
        user_id: User identifier
        face_sample: Face image file
        threshold: Optional custom threshold
        
    Returns:
        Verification result with confidence score
    """
    if verification_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verification engine not initialized"
        )
    
    logger.info(f"Verification request for user: {user_id}")
    
    temp_file = None
    try:
        # Validate file type
        if not face_sample.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {face_sample.content_type}"
            )
        
        # Read file content
        content = await face_sample.read()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            temp.write(content)
            temp_file = temp.name
        
        # Perform verification
        result = verification_engine.verify_user(user_id, temp_file, threshold)

        # Handle case where user is not enrolled (no 'decision' key)
        if not result.get('success', True):
            logger.warning(f"User {user_id} not enrolled or verification failed: {result.get('reason', 'unknown')}")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "success": True,
                    "data": {
                        **result,
                        "decision": "NOT_ENROLLED",
                        "confidence_score": 0.0
                    }
                }
            )

        logger.info(
            f"✓ Verification complete for {user_id}: "
            f"{result.get('decision', 'UNKNOWN')} (confidence: {result.get('confidence_score', 0):.4f})"
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "data": result
            }
        )
    
    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {str(e)}"
        )
    
    finally:
        # Clean up temp file
        if temp_file:
            try:
                os.unlink(temp_file)
            except:
                pass


@app.post("/api/verify")
async def verify_two_images(
    reference: UploadFile = File(...),
    probe: UploadFile = File(...),
    threshold: Optional[float] = Form(None)
):
    """
    Direct comparison of two face images (no enrollment needed)
    
    Args:
        reference: Reference face image
        probe: Probe face image to verify
        threshold: Optional custom threshold
        
    Returns:
        Similarity score and decision
    """
    if verification_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verification engine not initialized"
        )
    
    logger.info("Direct face comparison request")
    
    temp_files = []
    try:
        # Process both images
        ref_content = await reference.read()
        probe_content = await probe.read()
        
        # Save to temp files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            temp.write(ref_content)
            temp_files.append(temp.name)
            ref_path = temp.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            temp.write(probe_content)
            temp_files.append(temp.name)
            probe_path = temp.name
        
        # Extract embeddings
        ref_embedding = verification_engine.extract_embedding(ref_path, return_numpy=True)
        probe_embedding = verification_engine.extract_embedding(probe_path, return_numpy=True)
        
        # Compute similarity
        import numpy as np
        similarity = float(np.dot(ref_embedding, probe_embedding) / (
            np.linalg.norm(ref_embedding) * np.linalg.norm(probe_embedding)
        ))
        
        # Use default threshold if not provided
        if threshold is None:
            threshold = config.get('verification.threshold')
        
        decision = "MATCH" if similarity >= threshold else "MISMATCH"
        
        logger.info(f"✓ Comparison complete: {decision} (similarity: {similarity:.4f})")
        
        return {
            "similarity": similarity,
            "threshold": threshold,
            "decision": decision,
            "device": verification_engine.device
        }
    
    except Exception as e:
        logger.error(f"✗ Comparison failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}"
        )
    
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass


@app.get("/api/v1/users")
async def get_enrolled_users():
    """Get list of enrolled users"""
    if verification_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verification engine not initialized"
        )
    
    users = verification_engine.get_enrolled_users()
    
    return {
        "success": True,
        "count": len(users),
        "users": users
    }


@app.delete("/api/v1/users/{user_id}")
async def remove_user(user_id: str):
    """Remove user enrollment"""
    if verification_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verification engine not initialized"
        )
    
    success = verification_engine.remove_enrollment(user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {user_id}"
        )
    
    return {
        "success": True,
        "message": f"User {user_id} removed successfully"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get config
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 8004)
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
