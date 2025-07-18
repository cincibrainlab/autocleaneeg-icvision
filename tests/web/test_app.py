"""
Unit tests for ICVision web application.

This module tests the FastAPI endpoints and web interface functionality.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path

# Import with error handling for optional web dependencies
try:
    from icvision.web.app import app
    from icvision.web.models import ProcessFolderRequest, ComponentData
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    app = None


@pytest.mark.skipif(not WEB_AVAILABLE, reason="Web dependencies not available")
class TestWebApp:
    """Test cases for the FastAPI web application."""
    
    def setup_method(self):
        """Set up test client and mock data."""
        self.client = TestClient(app)
        self.test_session_id = "test-session-123"
        
    def test_root_endpoint(self):
        """Test that the root endpoint serves the HTML interface."""
        response = self.client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "ICVision Component Browser" in response.text
        
    def test_process_folder_invalid_path(self):
        """Test folder processing with invalid path."""
        response = self.client.post(
            "/api/process-folder",
            json={
                "folder_path": "/nonexistent/path",
                "confidence_threshold": 0.8
            }
        )
        assert response.status_code == 400
        assert "does not exist" in response.json()["detail"]
        
    @patch('icvision.web.app.scan_folder_for_eeg_files')
    def test_process_folder_no_files(self, mock_scan):
        """Test folder processing with no EEG files found."""
        mock_scan.return_value = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            response = self.client.post(
                "/api/process-folder",
                json={
                    "folder_path": temp_dir,
                    "confidence_threshold": 0.8
                }
            )
            assert response.status_code == 400
            assert "No valid EEG files found" in response.json()["detail"]
            
    @patch('icvision.web.app.scan_folder_for_eeg_files')
    def test_process_folder_success(self, mock_scan):
        """Test successful folder processing."""
        # Mock successful file scanning
        mock_subject = Mock()
        mock_subject.subject_id = "test_subject"
        mock_scan.return_value = [mock_subject]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            response = self.client.post(
                "/api/process-folder",
                json={
                    "folder_path": temp_dir,
                    "confidence_threshold": 0.8
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data
            assert data["subjects_found"] == 1
            assert data["processing_status"] == "ready"
            
    def test_get_components_invalid_session(self):
        """Test getting components with invalid session ID."""
        response = self.client.get("/api/components/invalid-session")
        assert response.status_code == 404
        assert "Session not found" in response.json()["detail"]
        
    @patch('icvision.web.app.sessions')
    def test_get_components_success(self, mock_sessions):
        """Test successful component retrieval."""
        # Mock session data
        mock_session = Mock()
        mock_session.subjects = []
        mock_sessions.__getitem__.return_value = mock_session
        mock_sessions.__contains__.return_value = True
        
        response = self.client.get(f"/api/components/{self.test_session_id}")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "subjects" in data
        assert "total_components" in data
        
    def test_override_classification_invalid_session(self):
        """Test classification override with invalid session."""
        response = self.client.post(
            "/api/override-classification",
            json={
                "session_id": "invalid-session",
                "subject_id": "test_subject",
                "component_index": 0,
                "new_label": "brain",
                "confidence": 0.9,
                "reason": "Manual override"
            }
        )
        assert response.status_code == 404
        assert "Session not found" in response.json()["detail"]
        
    def test_export_overrides_invalid_session(self):
        """Test export overrides with invalid session."""
        response = self.client.get("/api/export-overrides/invalid-session")
        assert response.status_code == 404
        assert "Session not found" in response.json()["detail"]
        
    def test_session_status_invalid_session(self):
        """Test session status with invalid session."""
        response = self.client.get("/api/sessions/invalid-session/status")
        assert response.status_code == 404
        assert "Session not found" in response.json()["detail"]


@pytest.mark.skipif(not WEB_AVAILABLE, reason="Web dependencies not available")
class TestWebModels:
    """Test cases for Pydantic models."""
    
    def test_process_folder_request_validation(self):
        """Test ProcessFolderRequest validation."""
        # Valid request
        request = ProcessFolderRequest(
            folder_path="/valid/path",
            confidence_threshold=0.8
        )
        assert request.folder_path == "/valid/path"
        assert request.confidence_threshold == 0.8
        
        # Invalid confidence threshold
        with pytest.raises(ValueError):
            ProcessFolderRequest(
                folder_path="/valid/path",
                confidence_threshold=1.5  # > 1.0
            )
            
    def test_component_data_model(self):
        """Test ComponentData model validation."""
        component = ComponentData(
            component_index=0,
            component_name="IC0",
            current_label="brain",
            confidence=0.95,
            reason="Clear brain activity",
            exclude_vision=False,
            image_base64="base64encodedimage"
        )
        assert component.component_index == 0
        assert component.current_label == "brain"
        assert component.override_label is None


@pytest.mark.skipif(WEB_AVAILABLE, reason="Testing import error handling")
def test_web_import_error():
    """Test that web module handles missing dependencies gracefully."""
    # This test runs when web dependencies are not available
    try:
        from icvision.web import app
        # Should be None when dependencies are missing
        assert app is None
    except ImportError:
        # This is also acceptable behavior
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])