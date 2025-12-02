"""
Basic functionality test for Keystroke Dynamics System
Tests core components without requiring trained models
"""

import os
import sys
import torch
import numpy as np
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config_loader import load_config
from src.keystroke_preprocessing import KeystrokePreprocessor
from src.keystroke_embedding import KeystrokeEmbeddingModel, TripletLoss
from src.keystroke_verification import KeystrokeVerifier
from src.anomaly_detection import AnomalyDetector


def test_config_loading():
    """Test configuration loading"""
    logger.info("Testing configuration loading...")
    
    try:
        config = load_config('config.yaml')
        assert config.model.embedding_dim == 128
        assert config.training.batch_size == 32
        logger.info("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration loading failed: {e}")
        return False


def test_preprocessing():
    """Test preprocessing module"""
    logger.info("Testing preprocessing module...")
    
    try:
        config = load_config('config.yaml')
        preprocessor = KeystrokePreprocessor(config)
        
        # Create dummy data
        dummy_data = np.random.randn(100, 31)
        
        # Test normalization
        normalized = preprocessor.normalize_features(dummy_data, fit=True)
        assert normalized.shape == dummy_data.shape
        
        # Test outlier handling
        cleaned = preprocessor.handle_outliers(dummy_data)
        assert cleaned.shape == dummy_data.shape
        
        logger.info("✓ Preprocessing module works correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Preprocessing test failed: {e}")
        return False


def test_embedding_model():
    """Test embedding model"""
    logger.info("Testing embedding model...")
    
    try:
        config = load_config('config.yaml')
        
        # Create model
        input_dim = 31
        model = KeystrokeEmbeddingModel(input_dim, config)
        
        # Test forward pass
        dummy_input = torch.randn(16, input_dim)
        embeddings = model(dummy_input)
        
        assert embeddings.shape == (16, config.model.embedding_dim)
        assert torch.allclose(torch.norm(embeddings, dim=1), torch.ones(16), atol=1e-5)
        
        logger.info("✓ Embedding model works correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Embedding model test failed: {e}")
        return False


def test_triplet_loss():
    """Test triplet loss"""
    logger.info("Testing triplet loss...")
    
    try:
        criterion = TripletLoss(margin=0.5)
        
        # Create dummy embeddings
        anchor = torch.randn(8, 128)
        positive = anchor + torch.randn(8, 128) * 0.1  # Similar to anchor
        negative = torch.randn(8, 128)  # Different from anchor
        
        loss = criterion(anchor, positive, negative)
        
        assert loss.item() >= 0
        
        logger.info("✓ Triplet loss works correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Triplet loss test failed: {e}")
        return False


def test_verification():
    """Test verification module"""
    logger.info("Testing verification module...")
    
    try:
        config = load_config('config.yaml')
        
        # Create model
        input_dim = 31
        model = KeystrokeEmbeddingModel(input_dim, config)
        model.eval()
        
        # Create verifier
        verifier = KeystrokeVerifier(model, config)
        
        # Create dummy enrollment data
        enroll_samples = torch.randn(50, input_dim)
        
        # Enroll user
        result = verifier.enroll_user("test_user", enroll_samples)
        assert result['success'] == True
        
        # Verify user
        verify_sample = torch.randn(input_dim)
        verify_result = verifier.verify_user("test_user", verify_sample)
        
        assert 'verified' in verify_result
        assert 'confidence' in verify_result
        assert 0 <= verify_result['confidence'] <= 1
        
        logger.info("✓ Verification module works correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Verification test failed: {e}")
        return False


def test_anomaly_detection():
    """Test anomaly detection"""
    logger.info("Testing anomaly detection...")
    
    try:
        config = load_config('config.yaml')
        
        # Create detector
        detector = AnomalyDetector(config)
        
        # Create dummy normal data
        normal_data = np.random.randn(100, 31)
        
        # Fit detector
        detector.fit(normal_data)
        
        # Test prediction
        test_sample = np.random.randn(1, 31)
        prediction = detector.predict(test_sample)
        
        assert prediction.shape == (1,)
        assert prediction[0] in [-1, 1]
        
        # Test anomaly detection
        result = detector.detect_anomaly(test_sample)
        assert 'is_anomaly' in result
        assert 'anomaly_score' in result
        
        logger.info("✓ Anomaly detection works correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Anomaly detection test failed: {e}")
        return False


def main():
    """Run all basic tests"""
    logger.info("=" * 80)
    logger.info("KEYSTROKE DYNAMICS - BASIC FUNCTIONALITY TESTS")
    logger.info("=" * 80)
    logger.info("")
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Preprocessing Module", test_preprocessing),
        ("Embedding Model", test_embedding_model),
        ("Triplet Loss", test_triplet_loss),
        ("Verification Module", test_verification),
        ("Anomaly Detection", test_anomaly_detection),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*80}")
        
        result = test_func()
        results.append((test_name, result))
        
        logger.info("")
    
    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:.<50} {status}")
    
    logger.info("")
    logger.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())

