"""
Utility functions for the your_package module.

This module contains helper functions and utilities that support
the main functionality of the package.
"""

from typing import Any, Dict, List, Optional, Union
import json
import logging
from pathlib import Path

# Set up logging for the module
logger = logging.getLogger(__name__)


def utility_function(
    data: Union[str, int, float], 
    operation: str = "stringify"
) -> str:
    """
    A utility function that demonstrates proper function structure.
    
    This function performs basic operations on input data and returns
    a string representation. It serves as an example of how to structure
    utility functions with proper type hints and documentation.
    
    Args:
        data: The input data to process (string, int, or float).
        operation: The operation to perform. Options: "stringify", "format".
                  Defaults to "stringify".
                  
    Returns:
        A string representation of the processed data.
        
    Raises:
        ValueError: If operation is not supported.
        TypeError: If data type is not supported.
        
    Example:
        >>> utility_function(42, "stringify")
        '42'
        >>> utility_function(3.14159, "format")
        '3.14'
    """
    # Validate input type
    if not isinstance(data, (str, int, float)):
        raise TypeError(
            f"Data must be str, int, or float, got {type(data).__name__}"
        )
    
    # Define available operations
    if operation == "stringify":
        result = str(data)
    elif operation == "format":
        if isinstance(data, float):
            result = f"{data:.2f}"
        else:
            result = str(data)
    else:
        raise ValueError(
            f"Unsupported operation '{operation}'. "
            f"Supported operations: ['stringify', 'format']"
        )
    
    logger.debug(f"Applied '{operation}' operation to {type(data).__name__} data")
    return result


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate a configuration dictionary.
    
    This function checks if a configuration dictionary contains
    the required keys and valid values.
    
    Args:
        config: Configuration dictionary to validate.
        
    Returns:
        True if configuration is valid, False otherwise.
        
    Example:
        >>> config = {"name": "test", "version": "1.0.0"}
        >>> validate_config(config)
        True
    """
    if not isinstance(config, dict):
        logger.error("Config must be a dictionary")
        return False
    
    # Required keys
    required_keys = ["name"]
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return False
        
        if not config[key]:
            logger.error(f"Config key '{key}' cannot be empty")
            return False
    
    logger.debug("Configuration validation passed")
    return True


def load_config_from_file(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load configuration from a JSON file.
    
    Args:
        file_path: Path to the JSON configuration file.
        
    Returns:
        Dictionary containing configuration data, or None if loading fails.
        
    Example:
        >>> config = load_config_from_file("config.json")
        >>> print(config)
        {'name': 'example', 'version': '1.0.0'}
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"Configuration file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if validate_config(config):
            logger.info(f"Successfully loaded configuration from {file_path}")
            return config
        else:
            logger.error(f"Invalid configuration in {file_path}")
            return None
            
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON configuration: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        return None


def save_config_to_file(
    config: Dict[str, Any], 
    file_path: Union[str, Path]
) -> bool:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary to save.
        file_path: Path where to save the configuration file.
        
    Returns:
        True if save was successful, False otherwise.
        
    Example:
        >>> config = {"name": "example", "version": "1.0.0"}
        >>> save_config_to_file(config, "config.json")
        True
    """
    if not validate_config(config):
        logger.error("Cannot save invalid configuration")
        return False
    
    file_path = Path(file_path)
    
    try:
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved configuration to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        return False


def format_data_summary(data: List[Any]) -> str:
    """
    Create a formatted summary of data.
    
    Args:
        data: List of data items to summarize.
        
    Returns:
        Formatted string summary of the data.
        
    Example:
        >>> data = [1, 2, 3, "hello", 3.14]
        >>> print(format_data_summary(data))
        Data Summary:
        - Total items: 5
        - Types: int (3), str (1), float (1)
    """
    if not isinstance(data, list):
        return "Invalid data: must be a list"
    
    if not data:
        return "Data Summary: Empty list"
    
    # Count types
    type_counts = {}
    for item in data:
        type_name = type(item).__name__
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    # Format type summary
    type_summary = ", ".join(
        f"{type_name} ({count})" for type_name, count in sorted(type_counts.items())
    )
    
    summary = f"""Data Summary:
- Total items: {len(data)}
- Types: {type_summary}"""
    
    return summary 