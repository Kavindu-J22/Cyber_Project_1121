"""
Configuration Loader for Keystroke Dynamics System
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
from loguru import logger


class Config:
    """Configuration class for keystroke dynamics system"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        
    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default"""
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return self._config


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    # Try to find config file
    if not os.path.exists(config_path):
        # Try in parent directory
        parent_config = os.path.join("..", config_path)
        if os.path.exists(parent_config):
            config_path = parent_config
        else:
            # Try in Keystroke Dynamics directory
            kd_config = os.path.join("Keystroke Dynamics", config_path)
            if os.path.exists(kd_config):
                config_path = kd_config
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create necessary directories
    paths = config_dict.get('paths', {})
    for path_key, path_value in paths.items():
        if path_key.endswith('_dir'):
            os.makedirs(path_value, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path_value}")
    
    logger.info("Configuration loaded successfully")
    return Config(config_dict)


def get_default_config() -> Config:
    """Get default configuration"""
    default_config = {
        'model': {
            'embedding_dim': 128,
            'hidden_dims': [256, 512, 256, 128],
            'dropout': 0.3,
        },
        'training': {
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
        },
        'verification': {
            'threshold': 0.75,
            'similarity_metric': 'cosine',
        },
        'paths': {
            'model_dir': 'models',
            'logs_dir': 'logs',
        }
    }
    return Config(default_config)

