"""
FastAPI application for ICVision web interface.

This module provides the main FastAPI application with endpoints for
interactive ICA component classification and management.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .models import (
    ProcessFolderRequest,
    ProcessFolderResponse,
    ComponentDataResponse,
    OverrideClassificationRequest,
    OverrideResponse,
    ExportOverridesResponse,
    ErrorResponse,
    SessionData,
)
from .utils import (
    scan_folder_for_eeg_files,
    process_session_data,
    validate_folder_path,
    export_override_csvs,
)

# Set up logging
logger = logging.getLogger("icvision.web")

# Create FastAPI app
app = FastAPI(
    title="ICVision Web Interface",
    description="Interactive ICA Component Classification Browser",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# In-memory session storage (for MVP - would use Redis/database in production)
sessions: Dict[str, SessionData] = {}

# Set up templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(templates_dir))

# Mount static files
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main component browser interface."""
    return templates.TemplateResponse("component_browser.html", {"request": request})


@app.post("/api/process-folder", response_model=ProcessFolderResponse)
async def process_folder(folder_request: ProcessFolderRequest):
    """
    Process a folder containing EEG files and create a new session.
    
    This endpoint scans the specified folder for EEG files, validates them,
    and creates a new session for interactive component classification.
    """
    try:
        # Validate folder path
        folder_path = validate_folder_path(folder_request.folder_path)
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Scan folder for EEG files
        subjects_data = await asyncio.to_thread(
            scan_folder_for_eeg_files, 
            folder_path
        )
        
        if not subjects_data:
            raise HTTPException(
                status_code=400,
                detail="No valid EEG files found in the specified folder"
            )
        
        # Create session data
        session_data = SessionData(
            session_id=session_id,
            folder_path=str(folder_path),
            subjects=subjects_data,
            confidence_threshold=folder_request.confidence_threshold,
        )
        
        # Store session
        sessions[session_id] = session_data
        
        logger.info(f"Created session {session_id} with {len(subjects_data)} subjects")
        
        return ProcessFolderResponse(
            session_id=session_id,
            subjects_found=len(subjects_data),
            processing_status="ready",
            message=f"Successfully found {len(subjects_data)} subjects in folder"
        )
        
    except Exception as e:
        logger.error(f"Error processing folder: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/components/{session_id}", response_model=ComponentDataResponse)
async def get_components(session_id: str):
    """
    Retrieve component data for a session.
    
    This endpoint returns all component data including images and classifications
    for the specified session.
    """
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = sessions[session_id]
        
        # Process component data if not already done
        if not any(subject.components for subject in session_data.subjects):
            await asyncio.to_thread(
                process_session_data,
                session_data
            )
        
        total_components = sum(len(subject.components) for subject in session_data.subjects)
        
        return ComponentDataResponse(
            session_id=session_id,
            subjects=session_data.subjects,
            total_components=total_components
        )
        
    except Exception as e:
        logger.error(f"Error retrieving components for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/override-classification", response_model=OverrideResponse)
async def override_classification(override_request: OverrideClassificationRequest):
    """
    Override the classification of a specific component.
    
    This endpoint allows users to manually override the automated classification
    of an ICA component with their own assessment.
    """
    try:
        session_id = override_request.session_id
        
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = sessions[session_id]
        
        # Find the subject and component
        subject = None
        component = None
        
        for subj in session_data.subjects:
            if subj.subject_id == override_request.subject_id:
                subject = subj
                for comp in subj.components:
                    if comp.component_index == override_request.component_index:
                        component = comp
                        break
                break
        
        if not subject or not component:
            raise HTTPException(
                status_code=404, 
                detail="Subject or component not found"
            )
        
        # Apply override
        component.override_label = override_request.new_label
        component.override_confidence = override_request.confidence
        component.override_reason = override_request.reason
        
        # Update session timestamp
        session_data.last_modified = datetime.now()
        
        logger.info(
            f"Override applied to session {session_id}, "
            f"subject {override_request.subject_id}, "
            f"component {override_request.component_index}: "
            f"{override_request.new_label}"
        )
        
        return OverrideResponse(
            success=True,
            message="Classification override applied successfully",
            updated_component=component
        )
        
    except Exception as e:
        logger.error(f"Error applying override: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export-overrides/{session_id}", response_model=ExportOverridesResponse)
async def export_overrides(session_id: str):
    """
    Export override results as CSV files.
    
    This endpoint generates CSV files containing the user's classification
    overrides for integration with the main ICVision pipeline.
    """
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = sessions[session_id]
        
        # Export override CSV files
        export_paths = await asyncio.to_thread(
            export_override_csvs,
            session_data
        )
        
        return ExportOverridesResponse(
            session_id=session_id,
            files_exported=len(export_paths),
            export_paths=export_paths,
            message=f"Successfully exported {len(export_paths)} override CSV files"
        )
        
    except Exception as e:
        logger.error(f"Error exporting overrides for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download-overrides/{session_id}")
async def download_overrides(session_id: str):
    """
    Download override CSV files as a zip archive.
    
    This endpoint creates a zip file containing all override CSV files
    for easy download and integration.
    """
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = sessions[session_id]
        
        # Create zip file with override CSVs
        from .utils import create_override_zip
        zip_path = await asyncio.to_thread(
            create_override_zip,
            session_data
        )
        
        return FileResponse(
            path=zip_path,
            filename=f"icvision_overrides_{session_id[:8]}.zip",
            media_type="application/zip"
        )
        
    except Exception as e:
        logger.error(f"Error creating download for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get the current status of a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    
    return {
        "session_id": session_id,
        "created_at": session_data.created_at,
        "last_modified": session_data.last_modified,
        "subjects_count": len(session_data.subjects),
        "total_components": sum(len(subject.components) for subject in session_data.subjects),
        "overrides_count": sum(
            sum(1 for comp in subject.components if comp.override_label)
            for subject in session_data.subjects
        )
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return ErrorResponse(
        error="HTTPException",
        message=exc.detail,
        details={"status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return ErrorResponse(
        error="InternalServerError",
        message="An unexpected error occurred",
        details={"type": type(exc).__name__}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)