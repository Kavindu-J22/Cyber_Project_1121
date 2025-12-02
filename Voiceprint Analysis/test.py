"""
Test Script for Voiceprint Analysis System
Tests enrollment, verification, and continuous authentication
"""
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import get_config
from src.speaker_verification import SpeakerVerificationEngine
from src.anti_spoofing import AntiSpoofingClassifier


def test_enrollment_and_verification():
    """Test speaker enrollment and verification"""
    print("\n" + "="*60)
    print("🧪 TEST: Speaker Enrollment and Verification")
    print("="*60)
    
    # Initialize system
    config = get_config()
    verification_engine = SpeakerVerificationEngine(config)
    
    # Get dataset paths
    dataset_base = Path(config.get('dataset.base_path'))
    dev_audio_path = dataset_base / config.get('dataset.dev_audio')
    
    # Test with speaker id10001
    speaker_id = "id10001"
    speaker_path = dev_audio_path / speaker_id
    
    if not speaker_path.exists():
        print(f"❌ Speaker path not found: {speaker_path}")
        return
    
    # Get audio files for enrollment
    audio_files = list(speaker_path.rglob("*.wav"))[:5]  # Use first 5 files
    
    if len(audio_files) < 3:
        print(f"❌ Not enough audio files for enrollment (need 3, found {len(audio_files)})")
        return
    
    print(f"\n📝 Enrolling speaker {speaker_id} with {len(audio_files)} samples...")
    
    # Enroll speaker
    enrollment_result = verification_engine.enroll_speaker(
        speaker_id=speaker_id,
        audio_samples=[str(f) for f in audio_files[:3]]
    )
    
    print(f"✓ Enrollment successful!")
    print(f"  Quality: {enrollment_result['enrollment_quality']:.4f}")
    print(f"  Embeddings: {enrollment_result['num_embeddings']}")
    
    # Test verification with same speaker (should pass)
    print(f"\n🔍 Testing verification with genuine sample...")
    if len(audio_files) > 3:
        verify_result = verification_engine.verify_speaker(
            speaker_id=speaker_id,
            audio_path=str(audio_files[3])
        )
        
        print(f"  Verified: {verify_result['verified']}")
        print(f"  Confidence: {verify_result['confidence_score']:.4f}")
        print(f"  Latency: {verify_result['latency_ms']:.2f} ms")
        print(f"  Status: {'✓ PASS' if verify_result['verified'] else '✗ FAIL'}")
    
    # Test verification with different speaker (should fail)
    print(f"\n🔍 Testing verification with impostor sample...")
    impostor_id = "id10002"
    impostor_path = dev_audio_path / impostor_id
    
    if impostor_path.exists():
        impostor_files = list(impostor_path.rglob("*.wav"))
        if len(impostor_files) > 0:
            impostor_result = verification_engine.verify_speaker(
                speaker_id=speaker_id,
                audio_path=str(impostor_files[0])
            )
            
            print(f"  Verified: {impostor_result['verified']}")
            print(f"  Confidence: {impostor_result['confidence_score']:.4f}")
            print(f"  Latency: {impostor_result['latency_ms']:.2f} ms")
            print(f"  Status: {'✓ PASS' if not impostor_result['verified'] else '✗ FAIL (should reject impostor)'}")
    
    print("\n" + "="*60)


def test_continuous_verification():
    """Test continuous verification"""
    print("\n" + "="*60)
    print("🧪 TEST: Continuous Verification")
    print("="*60)
    
    config = get_config()
    verification_engine = SpeakerVerificationEngine(config)
    
    # Get dataset paths
    dataset_base = Path(config.get('dataset.base_path'))
    dev_audio_path = dataset_base / config.get('dataset.dev_audio')
    
    # Enroll speaker
    speaker_id = "id10001"
    speaker_path = dev_audio_path / speaker_id
    
    if not speaker_path.exists():
        print(f"❌ Speaker path not found: {speaker_path}")
        return
    
    audio_files = list(speaker_path.rglob("*.wav"))[:5]
    
    if len(audio_files) < 4:
        print(f"❌ Not enough audio files")
        return
    
    # Enroll
    verification_engine.enroll_speaker(
        speaker_id=speaker_id,
        audio_samples=[str(f) for f in audio_files[:3]]
    )
    
    print(f"✓ Speaker {speaker_id} enrolled")
    
    # Test continuous verification
    print(f"\n🔄 Running continuous verification...")
    results = verification_engine.continuous_verification(
        speaker_id=speaker_id,
        audio_stream_path=str(audio_files[3])
    )
    
    print(f"  Total windows: {len(results)}")
    print(f"  Verified windows: {sum(1 for r in results if r['verified'])}")
    print(f"  Failed windows: {sum(1 for r in results if not r['verified'])}")
    print(f"  Average confidence: {sum(r['confidence_score'] for r in results) / len(results):.4f}")
    print(f"  Average latency: {sum(r['latency_ms'] for r in results) / len(results):.2f} ms")
    
    print("\n" + "="*60)


def test_anti_spoofing():
    """Test anti-spoofing detection"""
    print("\n" + "="*60)
    print("🧪 TEST: Anti-Spoofing Detection")
    print("="*60)
    
    config = get_config()
    anti_spoofing = AntiSpoofingClassifier(config)
    
    if not anti_spoofing.enabled:
        print("⚠️  Anti-spoofing is disabled in config")
        return
    
    # Get test audio
    dataset_base = Path(config.get('dataset.base_path'))
    dev_audio_path = dataset_base / config.get('dataset.dev_audio')
    
    speaker_path = dev_audio_path / "id10001"
    if speaker_path.exists():
        audio_files = list(speaker_path.rglob("*.wav"))
        if len(audio_files) > 0:
            from src.audio_preprocessing import AudioPreprocessor
            preprocessor = AudioPreprocessor(config)
            
            audio, sr = preprocessor.load_audio(str(audio_files[0]))
            
            print(f"\n🔍 Testing anti-spoofing on genuine audio...")
            result = anti_spoofing.detect_spoofing(audio, sr)
            
            print(f"  Is genuine: {result['is_genuine']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Spoofing type: {result.get('spoofing_type', 'None')}")
            print(f"  Status: {'✓ PASS' if result['is_genuine'] else '✗ FAIL'}")
    
    print("\n" + "="*60)


def test_latency_benchmark():
    """Benchmark verification latency"""
    print("\n" + "="*60)
    print("🧪 TEST: Latency Benchmark")
    print("="*60)
    
    config = get_config()
    verification_engine = SpeakerVerificationEngine(config)
    
    # Get dataset paths
    dataset_base = Path(config.get('dataset.base_path'))
    dev_audio_path = dataset_base / config.get('dataset.dev_audio')
    
    # Enroll speaker
    speaker_id = "id10001"
    speaker_path = dev_audio_path / speaker_id
    
    if not speaker_path.exists():
        print(f"❌ Speaker path not found")
        return
    
    audio_files = list(speaker_path.rglob("*.wav"))[:10]
    
    if len(audio_files) < 5:
        print(f"❌ Not enough audio files")
        return
    
    # Enroll
    verification_engine.enroll_speaker(
        speaker_id=speaker_id,
        audio_samples=[str(f) for f in audio_files[:3]]
    )
    
    # Benchmark verification latency
    print(f"\n⚡ Running latency benchmark (5 iterations)...")
    latencies = []
    
    for i in range(5):
        result = verification_engine.verify_speaker(
            speaker_id=speaker_id,
            audio_path=str(audio_files[3 + i])
        )
        latencies.append(result['latency_ms'])
    
    import numpy as np
    print(f"  Average latency: {np.mean(latencies):.2f} ms")
    print(f"  Min latency: {np.min(latencies):.2f} ms")
    print(f"  Max latency: {np.max(latencies):.2f} ms")
    print(f"  Target: < 800 ms")
    print(f"  Status: {'✓ PASS' if np.mean(latencies) < 800 else '✗ FAIL'}")
    
    print("\n" + "="*60)


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 VOICEPRINT ANALYSIS SYSTEM - TEST SUITE")
    print("="*70)
    
    try:
        test_enrollment_and_verification()
        test_continuous_verification()
        test_anti_spoofing()
        test_latency_benchmark()
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

