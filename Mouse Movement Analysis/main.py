"""
Main Entry Point for Mouse Movement Analysis System
Zero Trust Telehealth Platform - Continuous Behavioral Authentication
"""

import os
import sys
import argparse
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config_loader import load_config


def setup_logging(config):
    """Setup logging configuration"""
    os.makedirs(config.paths.logs_dir, exist_ok=True)
    logger.add(
        os.path.join(config.paths.logs_dir, "mouse_dynamics_{time}.log"),
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        level=config.logging.level,
        format=config.logging.format
    )


def run_training(config):
    """Run training pipeline"""
    logger.info("Starting training pipeline...")
    from train import main as train_main
    train_main()


def run_testing(config):
    """Run testing pipeline"""
    logger.info("Starting testing pipeline...")
    from test import main as test_main
    test_main()


def run_api(config):
    """Run API server"""
    logger.info("Starting API server...")
    import uvicorn
    
    uvicorn.run(
        "src.api:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=config.api.reload,
        log_level=config.api.log_level.lower()
    )


def run_demo(config):
    """Run interactive demo"""
    logger.info("Starting interactive demo...")
    
    import torch
    import numpy as np
    from src.mouse_preprocessing import MousePreprocessor
    from src.mouse_embedding import MouseEmbeddingModel
    from src.mouse_verification import MouseVerifier
    
    # Load dataset
    preprocessor = MousePreprocessor(config)
    
    try:
        # Load training data for demo
        train_data = preprocessor.load_balabit_dataset(
            config.dataset.training_files,
            config.dataset.labels_file,
            is_training=True
        )
        
        logger.info(f"Loaded {len(train_data)} training sessions")
        
        # Get first user for demo
        users = list(train_data.keys())
        demo_user = users[0]
        
        logger.info(f"Demo user: {demo_user}")
        
        # Extract features from user sessions
        user_sessions = train_data[demo_user]
        features_list = []
        
        for session_file in user_sessions[:5]:  # Use first 5 sessions
            features = preprocessor.extract_session_features(session_file)
            if features is not None:
                features_list.append(features)
        
        if len(features_list) == 0:
            logger.error("No features extracted")
            return
        
        # Concatenate features
        X = np.vstack(features_list)
        X = preprocessor.normalize_features(X, fit=True)
        X_tensor = torch.FloatTensor(X)
        
        logger.info(f"Extracted features shape: {X.shape}")
        
        # Initialize model
        input_dim = X.shape[1]
        model = MouseEmbeddingModel(input_dim, config)
        
        # Try to load checkpoint
        checkpoint_path = os.path.join(config.paths.checkpoint_dir, 'best_model.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model loaded from checkpoint")
        else:
            logger.warning("No checkpoint found, using untrained model")
        
        model.eval()
        
        # Initialize verifier
        verifier = MouseVerifier(model, config)
        
        # Enroll user with first 50 samples
        enroll_samples = X_tensor[:50]
        result = verifier.enroll_user(demo_user, enroll_samples)
        
        logger.info(f"Enrolled user: {result}")
        
        # Verify with next 20 samples (genuine)
        logger.info("\n=== Genuine Verification ===")
        for i in range(50, min(70, len(X_tensor))):
            verify_result = verifier.verify_user(demo_user, X_tensor[i])
            logger.info(f"Sample {i-49}: Verified={verify_result['verified']}, "
                       f"Confidence={verify_result['confidence']:.3f}")
        
        # Get statistics
        stats = verifier.get_verification_statistics(demo_user)
        logger.info(f"\n=== Statistics ===")
        logger.info(f"Total verifications: {stats['total_verifications']}")
        logger.info(f"Verification rate: {stats['verification_rate']*100:.1f}%")
        logger.info(f"Mean confidence: {stats['mean_confidence']:.3f}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Mouse Movement Analysis - Zero Trust Telehealth Platform"
    )
    
    parser.add_argument(
        'mode',
        choices=['train', 'test', 'api', 'demo'],
        help='Operation mode'
    )

    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(config)

    logger.info("=" * 80)
    logger.info("MOUSE MOVEMENT ANALYSIS - ZERO TRUST TELEHEALTH PLATFORM")
    logger.info("=" * 80)

    # Run selected mode
    if args.mode == 'train':
        run_training(config)
    elif args.mode == 'test':
        run_testing(config)
    elif args.mode == 'api':
        run_api(config)
    elif args.mode == 'demo':
        run_demo(config)


if __name__ == '__main__':
    main()


