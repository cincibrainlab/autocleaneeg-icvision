"""
Core functionality for the your_package module.

This module contains the main classes and functions that provide
the primary functionality of the package.
"""

from typing import Any, Dict, List, Optional, Union
import logging

# Set up logging for the module
logger = logging.getLogger(__name__)


class YourMainClass:
    """
    Main class providing the core functionality of the package.
    
    This class serves as an example of how to structure a main class
    with proper documentation, type hints, and error handling.
    
    Attributes:
        name (str): The name identifier for this instance.
        config (Dict[str, Any]): Configuration parameters.
        
    Example:
        >>> instance = YourMainClass(name="example")
        >>> result = instance.process_data(["item1", "item2"])
        >>> print(result)
        ['ITEM1', 'ITEM2']
    """
    
    def __init__(
        self, 
        name: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the YourMainClass instance.
        
        Args:
            name: A string identifier for this instance.
            config: Optional configuration dictionary. Defaults to empty dict.
            
        Raises:
            ValueError: If name is empty or None.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Name must be a non-empty string")
            
        self.name = name.strip()
        self.config = config or {}
        
        # Log initialization
        logger.info(f"Initialized {self.__class__.__name__} with name: {self.name}")
    
    def process_data(
        self, 
        data: List[str], 
        transform: str = "upper"
    ) -> List[str]:
        """
        Process a list of strings with the specified transformation.
        
        Args:
            data: List of strings to process.
            transform: Type of transformation ("upper", "lower", "title").
                      Defaults to "upper".
                      
        Returns:
            List of transformed strings.
            
        Raises:
            ValueError: If transform type is not supported.
            TypeError: If data is not a list of strings.
            
        Example:
            >>> instance = YourMainClass("processor")
            >>> instance.process_data(["hello", "world"], "title")
            ['Hello', 'World']
        """
        # Validate inputs
        if not isinstance(data, list):
            raise TypeError("Data must be a list")
            
        if not all(isinstance(item, str) for item in data):
            raise TypeError("All items in data must be strings")
            
        # Define available transformations
        transformations = {
            "upper": str.upper,
            "lower": str.lower,
            "title": str.title,
        }
        
        if transform not in transformations:
            raise ValueError(
                f"Unsupported transform '{transform}'. "
                f"Supported transforms: {list(transformations.keys())}"
            )
        
        # Apply transformation
        transform_func = transformations[transform]
        result = [transform_func(item) for item in data]
        
        logger.debug(f"Processed {len(data)} items with '{transform}' transformation")
        return result
    
    def get_info(self) -> Dict[str, Union[str, Dict[str, Any]]]:
        """
        Get information about this instance.
        
        Returns:
            Dictionary containing instance information.
        """
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "config": self.config.copy(),
        }
    
    def __str__(self) -> str:
        """String representation of the instance."""
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the instance."""
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config})"


class YourSecondaryClass:
    """
    A secondary class demonstrating additional functionality.
    
    This class shows how to structure additional classes within
    your package with proper inheritance and composition patterns.
    """
    
    def __init__(self, main_instance: YourMainClass) -> None:
        """
        Initialize with a reference to the main class.
        
        Args:
            main_instance: An instance of YourMainClass.
        """
        if not isinstance(main_instance, YourMainClass):
            raise TypeError("main_instance must be a YourMainClass instance")
            
        self.main_instance = main_instance
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def extended_process(self, data: List[str]) -> Dict[str, List[str]]:
        """
        Perform extended processing using the main instance.
        
        Args:
            data: List of strings to process.
            
        Returns:
            Dictionary with multiple transformation results.
        """
        results = {}
        
        for transform in ["upper", "lower", "title"]:
            try:
                results[transform] = self.main_instance.process_data(data, transform)
            except Exception as e:
                logger.error(f"Failed to apply {transform} transform: {e}")
                results[transform] = []
        
        return results 