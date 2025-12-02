"""
Setup script for Voiceprint Analysis System
"""
import os
import sys
from pathlib import Path


def create_directories():
    """Create necessary directories"""
    directories = [
        'models/checkpoints',
        'models/pretrained',
        'models/anti_spoofing',
        'logs',
        'temp'
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")


def check_dataset():
    """Check if dataset exists"""
    print("\n📊 Checking dataset...")
    
    dataset_path = Path("Voice dataset - senath")
    
    if not dataset_path.exists():
        print(f"  ⚠️  Dataset not found at: {dataset_path}")
        print(f"  Please ensure the VoxCeleb dataset is in the correct location")
        return False
    
    # Check for required files
    required_files = [
        'voiceprint_tuplets_dataset_5000.csv',
        'veri_test2.txt',
        'list_test_all2.txt'
    ]
    
    for file in required_files:
        file_path = dataset_path / file
        if file_path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} not found")
    
    return True


def create_env_file():
    """Create .env file from example"""
    print("\n🔐 Setting up environment variables...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("  ✓ .env file already exists")
        return
    
    if env_example.exists():
        # Generate random keys
        from cryptography.fernet import Fernet
        import secrets
        
        encryption_key = Fernet.generate_key().decode()
        jwt_secret = secrets.token_urlsafe(32)
        
        # Read example and replace
        with open(env_example, 'r') as f:
            content = f.read()
        
        content = content.replace(
            'your-256-bit-encryption-key-here-change-in-production',
            encryption_key
        )
        content = content.replace(
            'your-jwt-secret-key-here-change-in-production',
            jwt_secret
        )
        
        # Write .env
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("  ✓ .env file created with generated keys")
        print("  ⚠️  IMPORTANT: Store these keys securely!")
    else:
        print("  ✗ .env.example not found")


def check_dependencies():
    """Check if dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'torch',
        'torchaudio',
        'speechbrain',
        'fastapi',
        'uvicorn',
        'librosa',
        'soundfile',
        'cryptography'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} not installed")
            missing.append(package)
    
    if missing:
        print(f"\n  ⚠️  Missing packages: {', '.join(missing)}")
        print(f"  Run: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main setup function"""
    print("="*70)
    print("🚀 VOICEPRINT ANALYSIS SYSTEM - SETUP")
    print("="*70)
    
    # Create directories
    create_directories()
    
    # Check dataset
    check_dataset()
    
    # Create .env file
    create_env_file()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    print("\n" + "="*70)
    if deps_ok:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("  1. Review and update .env file with your settings")
        print("  2. Run tests: python main.py test")
        print("  3. Run training: python main.py train")
        print("  4. Start API server: python main.py api")
    else:
        print("⚠️  Setup completed with warnings")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
    
    print("="*70)


if __name__ == "__main__":
    main()

