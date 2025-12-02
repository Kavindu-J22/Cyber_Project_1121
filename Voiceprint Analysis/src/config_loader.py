"""
Configuration Loader for Voiceprint Analysis System
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration management class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Override with environment variables
        config = self._override_with_env(config)
        return config
    
    def _override_with_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration with environment variables"""
        # Security keys
        if os.getenv('EMBEDDING_ENCRYPTION_KEY'):
            config['security']['embedding_encryption_key'] = os.getenv('EMBEDDING_ENCRYPTION_KEY')
        if os.getenv('JWT_SECRET'):
            config['security']['jwt_secret'] = os.getenv('JWT_SECRET')
            
        # Database
        if os.getenv('MONGODB_URI'):
            config['database']['uri'] = os.getenv('MONGODB_URI')
            
        # API
        if os.getenv('API_HOST'):
            config['api']['host'] = os.getenv('API_HOST')
        if os.getenv('API_PORT'):
            config['api']['port'] = int(os.getenv('API_PORT'))
            
        # Model paths
        if os.getenv('MODEL_CHECKPOINT_PATH'):
            config['model']['checkpoint_path'] = os.getenv('MODEL_CHECKPOINT_PATH')
        if os.getenv('ANTI_SPOOFING_MODEL_PATH'):
            config['anti_spoofing']['model_path'] = os.getenv('ANTI_SPOOFING_MODEL_PATH')
            
        # Performance
        if os.getenv('USE_GPU'):
            config['performance']['use_gpu'] = os.getenv('USE_GPU').lower() == 'true'
            
        return config
    
    def _validate_config(self):
        """Validate critical configuration parameters"""
        required_keys = ['model', 'audio', 'verification', 'security', 'database', 'api']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration section: {key}")
                
        # Validate EER target
        if self.config['verification']['eer_target'] >= 0.05:
            print(f"Warning: EER target is {self.config['verification']['eer_target']}, should be < 0.03")
            
        # Validate latency requirement
        if self.config['verification']['max_latency_ms'] > 1000:
            print(f"Warning: Max latency is {self.config['verification']['max_latency_ms']}ms, should be < 800ms")
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'model.embedding_dim')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return self.config[key]
    
    def __contains__(self, key):
        """Check if key exists in config"""
        return key in self.config


# Global configuration instance
_config_instance = None

def get_config(config_path: str = "config.yaml") -> Config:
    """Get global configuration instance (singleton pattern)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance

