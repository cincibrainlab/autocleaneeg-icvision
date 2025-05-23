"""
ICVision - Automated ICA component classification using OpenAI Vision API.

This package provides automated classification of ICA components in EEG data
using OpenAI's Vision API, enabling efficient artifact detection and removal.
"""

__version__ = "0.1.0"
__author__ = "Gavin Gammoh"
__email__ = "gavin.gammoh@gmail.com"

# Public API - Import and expose main functions
from .core import label_components
from .config import COMPONENT_LABELS, DEFAULT_MODEL

# CLI is available but not imported by default to avoid import overhead
# Users can import it explicitly with: from icvision import cli

# Define what gets imported with "from icvision import *"
__all__ = [
    "label_components",
    "COMPONENT_LABELS", 
    "DEFAULT_MODEL",
] 