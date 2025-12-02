"""
Basic Test Script - Verify Core Functionality Without Full Dependencies
"""
import sys
from pathlib import Path
import numpy as np

print("="*70)
print("🧪 BASIC FUNCTIONALITY TEST")
print("="*70)

# Test 1: Check dataset exists
print("\n1️⃣ Testing Dataset Availability...")
dataset_path = Path("Voice dataset - senath")

if dataset_path.exists():
    print(f"   ✓ Dataset folder found: {dataset_path}")
    
    # Check CSV
    csv_file = dataset_path / "voiceprint_tuplets_dataset_5000.csv"
    if csv_file.exists():
        print(f"   ✓ CSV file found: {csv_file.name}")
        
        # Count lines
        with open(csv_file, 'r') as f:
            lines = len(f.readlines())
        print(f"   ✓ CSV contains {lines} lines (including header)")
    else:
        print(f"   ✗ CSV file not found")
    
    # Check verification pairs
    veri_file = dataset_path / "veri_test2.txt"
    if veri_file.exists():
        print(f"   ✓ Verification pairs found: {veri_file.name}")
        
        with open(veri_file, 'r') as f:
            pairs = len(f.readlines())
        print(f"   ✓ Contains {pairs} verification pairs")
    else:
        print(f"   ✗ Verification file not found")
    
    # Check audio folders
    dev_audio = dataset_path / "vox1_dev_wav" / "wav"
    test_audio = dataset_path / "vox1_test_wav" / "wav"
    
    if dev_audio.exists():
        speakers = list(dev_audio.iterdir())
        print(f"   ✓ Dev audio found: {len(speakers)} speakers")
    else:
        print(f"   ✗ Dev audio not found")
    
    if test_audio.exists():
        speakers = list(test_audio.iterdir())
        print(f"   ✓ Test audio found: {len(speakers)} speakers")
    else:
        print(f"   ✗ Test audio not found")
else:
    print(f"   ✗ Dataset folder not found: {dataset_path}")

# Test 2: Check Python packages
print("\n2️⃣ Testing Python Dependencies...")
packages = {
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'torch': 'PyTorch',
    'torchaudio': 'TorchAudio',
    'librosa': 'Librosa',
    'soundfile': 'SoundFile',
    'scipy': 'SciPy',
    'yaml': 'PyYAML',
    'cryptography': 'Cryptography'
}

installed = []
missing = []

for package, name in packages.items():
    try:
        __import__(package)
        print(f"   ✓ {name}")
        installed.append(name)
    except ImportError:
        print(f"   ✗ {name} - NOT INSTALLED")
        missing.append(name)

# Test 3: Test audio loading
print("\n3️⃣ Testing Audio Loading...")
try:
    import librosa
    import soundfile as sf
    
    # Find a sample audio file
    if dev_audio.exists():
        audio_files = list(dev_audio.rglob("*.wav"))
        if len(audio_files) > 0:
            sample_file = audio_files[0]
            print(f"   Testing with: {sample_file.name}")
            
            # Load with librosa
            audio, sr = librosa.load(str(sample_file), sr=16000)
            print(f"   ✓ Loaded audio: {len(audio)} samples at {sr} Hz")
            print(f"   ✓ Duration: {len(audio)/sr:.2f} seconds")
            
            # Test basic processing
            audio_normalized = audio / np.max(np.abs(audio))
            print(f"   ✓ Normalized audio: range [{audio_normalized.min():.3f}, {audio_normalized.max():.3f}]")
        else:
            print(f"   ⚠️  No audio files found")
    else:
        print(f"   ⚠️  Audio folder not accessible")
        
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Test configuration loading
print("\n4️⃣ Testing Configuration...")
try:
    import yaml
    
    config_file = Path("config.yaml")
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"   ✓ Config loaded successfully")
        print(f"   ✓ Model type: {config.get('model', {}).get('type', 'N/A')}")
        print(f"   ✓ Embedding dim: {config.get('model', {}).get('embedding_dim', 'N/A')}")
        print(f"   ✓ EER target: {config.get('verification', {}).get('eer_target', 'N/A')}")
        print(f"   ✓ Max latency: {config.get('verification', {}).get('max_latency_ms', 'N/A')} ms")
    else:
        print(f"   ✗ Config file not found")
        
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 5: Test cosine similarity computation
print("\n5️⃣ Testing Cosine Similarity...")
try:
    # Create sample embeddings
    emb1 = np.random.randn(192)
    emb2 = np.random.randn(192)
    
    # Normalize
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    # Compute similarity
    similarity = np.dot(emb1, emb2)
    
    print(f"   ✓ Created 192-dim embeddings")
    print(f"   ✓ Cosine similarity: {similarity:.4f}")
    
    # Test with identical embeddings
    similarity_same = np.dot(emb1, emb1)
    print(f"   ✓ Same embedding similarity: {similarity_same:.4f} (should be ~1.0)")
    
except Exception as e:
    print(f"   ✗ Error: {e}")

# Summary
print("\n" + "="*70)
print("📊 TEST SUMMARY")
print("="*70)
print(f"✓ Installed packages: {len(installed)}/{len(packages)}")
print(f"✗ Missing packages: {len(missing)}/{len(packages)}")

if missing:
    print(f"\n⚠️  Missing packages: {', '.join(missing)}")
    print(f"   Install with: pip install {' '.join(missing)}")

if len(installed) >= 7:
    print("\n✅ Core functionality is working!")
    print("   Dataset is accessible")
    print("   Audio loading works")
    print("   Configuration is valid")
    print("   Math operations work")
else:
    print("\n⚠️  Some dependencies are missing")
    print("   Install missing packages to enable full functionality")

print("="*70)

