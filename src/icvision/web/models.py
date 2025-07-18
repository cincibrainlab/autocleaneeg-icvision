"""
Pydantic models for ICVision web API.

This module defines the data models used for API requests and responses
in the FastAPI web interface.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class ComponentData(BaseModel):
    """Data model for individual ICA component information."""
    
    component_index: int = Field(..., description="Zero-based component index")
    component_name: str = Field(..., description="Component name (e.g., 'IC0', 'IC1')")
    current_label: str = Field(..., description="Current classification label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score")
    reason: str = Field(..., description="Detailed reasoning for classification")
    exclude_vision: bool = Field(..., description="Whether component should be excluded")
    image_base64: str = Field(..., description="Base64-encoded component plot image")
    override_label: Optional[str] = Field(None, description="User override label")
    override_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="User override confidence")
    override_reason: Optional[str] = Field(None, description="User override reasoning")


class SubjectData(BaseModel):
    """Data model for individual subject/file information."""
    
    subject_id: str = Field(..., description="Subject identifier (basename)")
    raw_file_path: str = Field(..., description="Path to raw EEG data file")
    ica_file_path: str = Field(..., description="Path to ICA decomposition file")
    results_file_path: Optional[str] = Field(None, description="Path to existing results CSV")
    components: List[ComponentData] = Field(default_factory=list, description="List of component data")
    processing_status: str = Field(default="pending", description="Processing status")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")


class SessionData(BaseModel):
    """Data model for web session information."""
    
    session_id: str = Field(..., description="Unique session identifier")
    folder_path: str = Field(..., description="Path to input folder")
    subjects: List[SubjectData] = Field(default_factory=list, description="List of subjects in session")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation timestamp")
    last_modified: datetime = Field(default_factory=datetime.now, description="Last modification timestamp")
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence threshold used")


class ProcessFolderRequest(BaseModel):
    """Request model for folder processing endpoint."""
    
    folder_path: str = Field(..., description="Path to folder containing EEG files")
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence threshold for classification")


class ProcessFolderResponse(BaseModel):
    """Response model for folder processing endpoint."""
    
    session_id: str = Field(..., description="Unique session identifier for subsequent requests")
    subjects_found: int = Field(..., description="Number of subjects found in folder")
    processing_status: str = Field(..., description="Overall processing status")
    message: str = Field(..., description="Status message")


class ComponentDataResponse(BaseModel):
    """Response model for component data retrieval."""
    
    session_id: str = Field(..., description="Session identifier")
    subjects: List[SubjectData] = Field(..., description="List of subjects with component data")
    total_components: int = Field(..., description="Total number of components across all subjects")


class OverrideClassificationRequest(BaseModel):
    """Request model for classification override."""
    
    session_id: str = Field(..., description="Session identifier")
    subject_id: str = Field(..., description="Subject identifier")
    component_index: int = Field(..., description="Component index to override")
    new_label: str = Field(..., description="New classification label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in new classification")
    reason: str = Field(..., description="Reasoning for override")


class OverrideResponse(BaseModel):
    """Response model for classification override."""
    
    success: bool = Field(..., description="Whether override was successful")
    message: str = Field(..., description="Status message")
    updated_component: Optional[ComponentData] = Field(None, description="Updated component data")


class ExportOverridesResponse(BaseModel):
    """Response model for export overrides endpoint."""
    
    session_id: str = Field(..., description="Session identifier")
    files_exported: int = Field(..., description="Number of override CSV files exported")
    export_paths: List[str] = Field(..., description="Paths to exported CSV files")
    message: str = Field(..., description="Status message")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Detailed error message")
    details: Optional[dict] = Field(None, description="Additional error details")