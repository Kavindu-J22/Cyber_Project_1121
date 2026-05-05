"""
Main Entry Point for Face Verification System
Zero Trust Telehealth Platform - Continuous Face Authentication
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import load_config


def setup_logging(config):
    """Setup logging configuration"""
    logs_dir = config.get('paths.logs_dir', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    logger.add(
        os.path.join(logs_dir, "face_verification_{time}.log"),
        rotation=config.get('logging.rotation', "500 MB"),
        retention=config.get('logging.retention', "10 days"),
        level=config.get('logging.level', "INFO"),
        format=config.get('logging.format', 
                         "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                         "<level>{level: <8}</level> | "
                         "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
                         "<level>{message}</level>")
    )


def run_api(config):
    """Run API server"""
    logger.info("Starting Face Verification API server...")
    import uvicorn
    
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 8004)
    workers = config.get('api.workers', 1)
    reload = config.get('api.reload', False)
    log_level = config.get('api.log_level', 'info').lower()
    
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Workers: {workers}")
    logger.info(f"  Docs: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
    
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level=log_level
    )


def run_demo(config):
    """Run interactive demo"""
    logger.info("Starting Face Verification interactive demo...")
    
    from src.face_verification import FaceVerificationEngine
    from PIL import Image
    import numpy as np
    
    # Initialize engine
    engine = FaceVerificationEngine(config)
    
    logger.info("\n" + "="*70)
    logger.info("🔐 FACE VERIFICATION - INTERACTIVE DEMO")
    logger.info("="*70)
    
    # Demo: Create dummy faces for testing
    logger.info("\nDemo: Creating test face embeddings...")
    
    # Create two different colored images to simulate faces
    face_size = config.get('image.face_size', 112)
    face1 = Image.new('RGB', (face_size, face_size), color='red')
    face2 = Image.new('RGB', (face_size, face_size), color='blue')
    face3 = Image.new('RGB', (face_size, face_size), color='red')  # Similar to face1
    
    # Save temp images
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f1:
        face1.save(f1.name)
        face1_path = f1.name
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f2:
        face2.save(f2.name)
        face2_path = f2.name
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f3:
        face3.save(f3.name)
        face3_path = f3.name
    
    try:
        # Enroll user
        logger.info("\n📝 Enrolling user 'demo_user' with 2 samples...")
        enroll_result = engine.enroll_user('demo_user', [face1_path, face3_path])
        logger.info(f"  ✓ Enrollment successful!")
        logger.info(f"    Samples: {enroll_result['num_samples']}")
        logger.info(f"    Quality: {enroll_result['enrollment_quality']:.4f}")
        logger.info(f"    Latency: {enroll_result['latency_ms']:.2f} ms")
        
        # Verify with genuine sample
        logger.info("\n🔍 Verifying with genuine sample (same user)...")
        verify_result = engine.verify_user('demo_user', face1_path)
        logger.info(f"  Decision: {verify_result['decision']}")
        logger.info(f"  Confidence: {verify_result['confidence_score']:.4f}")
        logger.info(f"  Threshold: {verify_result['threshold']:.4f}")
        logger.info(f"  Latency: {verify_result['latency_ms']:.2f} ms")
        
        # Verify with impostor sample
        logger.info("\n🔍 Verifying with impostor sample (different user)...")
        impostor_result = engine.verify_user('demo_user', face2_path)
        logger.info(f"  Decision: {impostor_result['decision']}")
        logger.info(f"  Confidence: {impostor_result['confidence_score']:.4f}")
        logger.info(f"  Threshold: {impostor_result['threshold']:.4f}")
        logger.info(f"  Latency: {impostor_result['latency_ms']:.2f} ms")
        
        # Show enrolled users
        logger.info("\n📊 Enrolled users:")
        users = engine.get_enrolled_users()
        for user in users:
            logger.info(f"  - {user}")
        
        logger.info("\n✓ Demo completed successfully!")
        
    finally:
        # Clean up temp files
        os.unlink(face1_path)
        os.unlink(face2_path)
        os.unlink(face3_path)


def run_test(config):
    """Run test suite"""
    logger.info("Running Face Verification tests...")
    
    from src.face_verification import FaceVerificationEngine
    from src.face_preprocessing import FacePreprocessor
    from src.face_model import ResNet50TripletModel
    
    logger.info("\n1. Testing preprocessor...")
    face_size = config.get('image.face_size', 112)
    preprocessor = FacePreprocessor(face_size=face_size)
    from PIL import Image
    test_image = Image.new('RGB', (300, 300), color='green')
    tensor = preprocessor.preprocess(test_image)
    logger.info(f"  ✓ Preprocessor working (output shape: {tensor.shape})")
    
    logger.info("\n2. Testing model loading...")
    try:
        engine = FaceVerificationEngine(config)
        logger.info(f"  ✓ Model loaded successfully")
        logger.info(f"    Device: {engine.device}")
        logger.info(f"    Embedding dim: {engine.model.get_embedding_dim()}")
    except Exception as e:
        logger.error(f"  ✗ Model loading failed: {e}")
        return
    
    logger.info("\n3. Testing embedding extraction...")
    try:
        embedding = engine.extract_embedding(test_image, return_numpy=True)
        logger.info(f"  ✓ Embedding extracted (shape: {embedding.shape})")
        logger.info(f"    L2 norm: {np.linalg.norm(embedding):.4f} (should be ~1.0)")
    except Exception as e:
        logger.error(f"  ✗ Embedding extraction failed: {e}")
        return
    
    logger.info("\n✓ All tests passed!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Face Verification System - Zero Trust Telehealth Platform"
    )
    
    parser.add_argument(
        'mode',
        choices=['api', 'demo', 'test'],
        help='Execution mode: api (start server), demo (interactive demo), test (run tests)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Setup logging
    setup_logging(config)
    
    # Run selected mode
    try:
        if args.mode == 'api':
            run_api(config)
        elif args.mode == 'demo':
            run_demo(config)
        elif args.mode == 'test':
            run_test(config)
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
