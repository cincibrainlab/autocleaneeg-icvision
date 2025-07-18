"""
ICVision Web Interface Module.

This module provides a FastAPI web interface for interactive ICA component
classification with visual browsing and manual override capabilities.
"""

try:
    from .app import app
    __all__ = ["app"]
except ImportError:
    # Handle case where web dependencies are not installed
    app = None
    __all__ = []