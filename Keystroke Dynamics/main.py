"""
Main Entry Point for Keystroke Dynamics System
Zero Trust Telehealth Platform - Continuous Authentication
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
    logger.add(
        os.path.join(config.paths.logs_dir, "keystroke_dynamics_{time}.log"),
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
    from src.keystroke_preprocessing import KeystrokePreprocessor
    from src.keystroke_embedding import KeystrokeEmbeddingModel
    from src.keystroke_verification import KeystrokeVerifier
    
    # Load dataset
    preprocessor = KeystrokePreprocessor(config)
    dataset_path = os.path.join('Dataset', 'DSL-StrongPasswordData-Original_Dataset.xls')
    
    try:
        df = preprocessor.load_dsl_dataset(dataset_path)
    except:
        dataset_path = os.path.join('Keystroke Dynamics', 'Dataset', 
                                   'DSL-StrongPasswordData-Original_Dataset.xls')
        df = preprocessor.load_dsl_dataset(dataset_path)
    
    # Preprocess
    X, y, _ = preprocessor.extract_timing_features(df)
    X = preprocessor.normalize_features(X, fit=True)
    X_tensor = torch.FloatTensor(X)
    
    # Initialize model
    input_dim = X.shape[1]
    model = KeystrokeEmbeddingModel(input_dim, config)
    
    # Try to load checkpoint
    checkpoint_path = os.path.join(config.paths.checkpoint_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model loaded from checkpoint")
    else:
        logger.warning("No checkpoint found, using untrained model")
    
    model.eval()
    
    # Initialize verifier
    verifier = KeystrokeVerifier(model, config)
    
    # Demo: Enroll first subject
    subjects = df['subject'].unique()
    demo_subject = subjects[0]
    
    subject_mask = df['subject'] == demo_subject
    subject_indices = np.where(subject_mask.values)[0]
    
    # Enroll with first 50 samples
    enroll_samples = X_tensor[subject_indices[:50]]
    result = verifier.enroll_user(demo_subject, enroll_samples)
    
    logger.info(f"Enrolled user: {result}")
    
    # Verify with next 10 samples (genuine)
    logger.info("\n=== Genuine Verification ===")
    for i in range(50, 60):
        idx = subject_indices[i]
        verify_result = verifier.verify_user(demo_subject, X_tensor[idx])
        logger.info(f"Sample {i-50+1}: Verified={verify_result['verified']}, "
                   f"Confidence={verify_result['confidence']:.3f}")
    
    # Verify with different subject (impostor)
    impostor_subject = subjects[1]
    impostor_mask = df['subject'] == impostor_subject
    impostor_indices = np.where(impostor_mask.values)[0]
    
    logger.info("\n=== Impostor Verification ===")
    for i in range(10):
        idx = impostor_indices[i]
        verify_result = verifier.verify_user(demo_subject, X_tensor[idx])
        logger.info(f"Sample {i+1}: Verified={verify_result['verified']}, "
                   f"Confidence={verify_result['confidence']:.3f}")
    
    # Get statistics
    stats = verifier.get_verification_statistics(demo_subject)
    logger.info(f"\n=== Statistics ===")
    logger.info(f"Total verifications: {stats['total_verifications']}")
    logger.info(f"Verification rate: {stats['verification_rate']*100:.1f}%")
    logger.info(f"Mean confidence: {stats['mean_confidence']:.3f}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Keystroke Dynamics - Zero Trust Telehealth Platform"
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
    logger.info("KEYSTROKE DYNAMICS - ZERO TRUST TELEHEALTH PLATFORM")
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

