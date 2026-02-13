"""
Configuration Loader for Face Verification System
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration class with dot notation access"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation"""
        keys = key.split('.')
        value = self
        try:
            for k in keys:
                if isinstance(value, Config):
                    value = getattr(value, k)
                elif isinstance(value, dict):
                    value = value[k]
                else:
                    return default
            return value
        except (AttributeError, KeyError):
            return default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file. If None, uses default location
        
    Returns:
        Config object with nested attribute access
    """
    if config_path is None:
        # Try to find config.yaml in project root
        current_dir = Path(__file__).parent.parent
        config_path = current_dir / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def get_config() -> Config:
    """Get configuration singleton"""
    return load_config()


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Model type: {config.model.type}")
    print(f"API port: {config.api.port}")
    print(f"Embedding dimension: {config.model.embedding_dim}")
    print(f"Verification threshold: {config.verification.threshold}")
