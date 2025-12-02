"""
Configuration Loader Module
Loads and validates YAML configuration files
"""

import yaml
from pathlib import Path
from typing import Any, Dict
from loguru import logger


class DotDict(dict):
    """
    Dictionary that supports dot notation access
    Example: config.model.embedding_dim instead of config['model']['embedding_dim']
    """
    
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")
    
    def to_dict(self):
        """Convert back to regular dict"""
        result = {}
        for key, value in self.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str = 'config.yaml') -> DotDict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DotDict configuration object
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Try alternative paths
        alt_paths = [
            Path('Mouse Movement Analysis') / config_path,
            Path(__file__).parent.parent / config_path
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                config_file = alt_path
                break
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_file}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = DotDict(config_dict)
    
    # Validate configuration
    validate_config(config)
    
    logger.info("Configuration loaded successfully")
    
    return config


def validate_config(config: DotDict):
    """
    Validate configuration parameters
    
    Args:
        config: Configuration object to validate
    """
    # Check required sections
    required_sections = ['model', 'features', 'training', 'verification', 'dataset', 'paths']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate model configuration
    if config.model.embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    
    if not config.model.hidden_dims:
        raise ValueError("hidden_dims cannot be empty")
    
    # Validate training configuration
    if config.training.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    if config.training.epochs <= 0:
        raise ValueError("epochs must be positive")
    
    if config.training.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    
    # Validate data split ratios
    total_ratio = config.training.train_ratio + config.training.val_ratio + config.training.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Data split ratios must sum to 1.0, got {total_ratio}")
    
    # Validate verification thresholds
    if not 0 <= config.verification.threshold <= 1:
        raise ValueError("verification threshold must be between 0 and 1")
    
    logger.debug("Configuration validation passed")


def save_config(config: DotDict, output_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration object
        output_path: Path to save configuration
    """
    config_dict = config.to_dict()
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Configuration saved to: {output_path}")


if __name__ == '__main__':
    # Test configuration loading
    config = load_config('config.yaml')
    print(f"Model: {config.model.name}")
    print(f"Embedding dim: {config.model.embedding_dim}")
    print(f"Training epochs: {config.training.epochs}")

