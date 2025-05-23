"""
Your Package - A professional Python package template.

This module provides a template for creating high-quality Python packages
that are ready for distribution on PyPI.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Public API - Import and expose main classes/functions here
from .core import YourMainClass, YourSecondaryClass
from .utils import utility_function, validate_config, load_config_from_file, save_config_to_file, format_data_summary

# CLI is available but not imported by default to avoid import overhead
# Users can import it explicitly with: from your_package import cli

# Define what gets imported with "from your_package import *"
__all__ = [
    "YourMainClass",
    "YourSecondaryClass", 
    "utility_function",
    "validate_config",
    "load_config_from_file",
    "save_config_to_file",
    "format_data_summary",
] 