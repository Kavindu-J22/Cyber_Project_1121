"""
Main Entry Point for Voiceprint Analysis System
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def run_api_server():
    """Run the FastAPI server"""
    from src.api import app
    from src.config_loader import get_config
    import uvicorn
    
    config = get_config()
    
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 8001)
    reload = config.get('api.reload', False)
    log_level = config.get('api.log_level', 'info')
    
    print(f"\n🚀 Starting Voiceprint Analysis API Server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Docs: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
    print()
    
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )


def run_training():
    """Run training/evaluation"""
    from train import main as train_main
    train_main()


def run_tests():
    """Run test suite"""
    from test import main as test_main
    test_main()


def run_demo():
    """Run interactive demo"""
    print("\n" + "="*70)
    print("🎤 VOICEPRINT ANALYSIS - INTERACTIVE DEMO")
    print("="*70)
    
    from src.config_loader import get_config
    from src.speaker_verification import SpeakerVerificationEngine
    from pathlib import Path
    
    config = get_config()
    verification_engine = SpeakerVerificationEngine(config)
    
    # Get dataset paths
    dataset_base = Path(config.get('dataset.base_path'))
    dev_audio_path = dataset_base / config.get('dataset.dev_audio')
    
    print("\n📋 Available speakers:")
    speakers = sorted([d.name for d in dev_audio_path.iterdir() if d.is_dir()])[:10]
    for i, speaker in enumerate(speakers, 1):
        print(f"   {i}. {speaker}")
    
    # Enroll a speaker
    print("\n📝 Enrolling speaker id10001...")
    speaker_id = "id10001"
    speaker_path = dev_audio_path / speaker_id
    
    if speaker_path.exists():
        audio_files = list(speaker_path.rglob("*.wav"))[:5]
        
        if len(audio_files) >= 3:
            result = verification_engine.enroll_speaker(
                speaker_id=speaker_id,
                audio_samples=[str(f) for f in audio_files[:3]]
            )
            
            print(f"✓ Enrollment successful!")
            print(f"  Quality: {result['enrollment_quality']:.4f}")
            print(f"  Embeddings: {result['num_embeddings']}")
            
            # Verify with genuine sample
            if len(audio_files) > 3:
                print(f"\n🔍 Verifying with genuine sample...")
                verify_result = verification_engine.verify_speaker(
                    speaker_id=speaker_id,
                    audio_path=str(audio_files[3])
                )
                
                print(f"  ✓ Verified: {verify_result['verified']}")
                print(f"  Confidence: {verify_result['confidence_score']:.4f}")
                print(f"  Latency: {verify_result['latency_ms']:.2f} ms")
            
            # Test with impostor
            impostor_id = "id10002"
            impostor_path = dev_audio_path / impostor_id
            
            if impostor_path.exists():
                impostor_files = list(impostor_path.rglob("*.wav"))
                if len(impostor_files) > 0:
                    print(f"\n🔍 Testing with impostor ({impostor_id})...")
                    impostor_result = verification_engine.verify_speaker(
                        speaker_id=speaker_id,
                        audio_path=str(impostor_files[0])
                    )
                    
                    print(f"  ✗ Verified: {impostor_result['verified']}")
                    print(f"  Confidence: {impostor_result['confidence_score']:.4f}")
                    print(f"  Latency: {impostor_result['latency_ms']:.2f} ms")
        else:
            print(f"❌ Not enough audio files for enrollment")
    else:
        print(f"❌ Speaker path not found: {speaker_path}")
    
    print("\n" + "="*70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Voiceprint Analysis System - Zero Trust Continuous Speaker Verification"
    )
    
    parser.add_argument(
        'command',
        choices=['api', 'train', 'test', 'demo'],
        help='Command to run'
    )
    
    args = parser.parse_args()
    
    if args.command == 'api':
        run_api_server()
    elif args.command == 'train':
        run_training()
    elif args.command == 'test':
        run_tests()
    elif args.command == 'demo':
        run_demo()


if __name__ == "__main__":
    main()

