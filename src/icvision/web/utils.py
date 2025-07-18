"""
Utility functions for ICVision web interface.

This module provides helper functions for file processing, validation,
and data management in the web interface.
"""

import base64
import io
import logging
import zipfile
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from fastapi import HTTPException

from ..config import COMPONENT_LABELS
from ..plotting import plot_component_for_web, validate_web_plotting_inputs
from ..utils import load_raw_data, load_ica_data, extract_input_basename

# Import models with try/except to handle circular imports
try:
    from .models import SubjectData, ComponentData, SessionData
except ImportError:
    # Define minimal types for development
    from typing import Any
    SubjectData = Any
    ComponentData = Any  
    SessionData = Any

logger = logging.getLogger("icvision.web.utils")

# Allowed file extensions for EEG data
ALLOWED_EXTENSIONS = {'.set', '.fif', '.edf', '.bdf', '.vhdr', '.csv'}


def validate_folder_path(folder_path: str) -> Path:
    """
    Validate and resolve folder path with security checks.
    
    Args:
        folder_path: Path to folder to validate
        
    Returns:
        Resolved Path object
        
    Raises:
        HTTPException: If path is invalid or doesn't exist
    """
    try:
        path = Path(folder_path).resolve()
        
        if not path.exists():
            raise HTTPException(status_code=400, detail="Folder does not exist")
        
        if not path.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")
        
        return path
        
    except Exception as e:
        logger.error(f"Error validating folder path {folder_path}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid folder path: {str(e)}")


def validate_file_extension(file_path: Path) -> bool:
    """
    Validate file extension against allowed types.
    
    Args:
        file_path: Path to file to validate
        
    Returns:
        True if extension is allowed, False otherwise
    """
    return file_path.suffix.lower() in ALLOWED_EXTENSIONS


def scan_folder_for_eeg_files(folder_path: Path) -> List[SubjectData]:
    """
    Scan folder for EEG files and organize by subject.
    
    Args:
        folder_path: Path to folder containing EEG files
        
    Returns:
        List of SubjectData objects for found subjects
    """
    subjects = {}
    
    # Scan for all relevant files
    for file_path in folder_path.iterdir():
        if not file_path.is_file() or not validate_file_extension(file_path):
            continue
        
        # Extract basename for subject identification
        basename = extract_input_basename(str(file_path))
        
        if basename not in subjects:
            subjects[basename] = {
                'subject_id': basename,
                'raw_file_path': None,
                'ica_file_path': None,
                'results_file_path': None,
            }
        
        # Categorize file by type
        if file_path.suffix.lower() == '.set':
            subjects[basename]['raw_file_path'] = str(file_path)
        elif file_path.suffix.lower() == '.fif':
            # Could be raw or ICA - need to check content
            try:
                # Try to load as ICA first
                load_ica_data(str(file_path))
                subjects[basename]['ica_file_path'] = str(file_path)
            except:
                # If ICA loading fails, assume it's raw data
                subjects[basename]['raw_file_path'] = str(file_path)
        elif file_path.name.endswith('_icvis_results.csv'):
            subjects[basename]['results_file_path'] = str(file_path)
    
    # Create SubjectData objects for subjects with both raw and ICA files
    subject_data_list = []
    for basename, files in subjects.items():
        if files['raw_file_path'] and files['ica_file_path']:
            subject_data = SubjectData(
                subject_id=basename,
                raw_file_path=files['raw_file_path'],
                ica_file_path=files['ica_file_path'],
                results_file_path=files['results_file_path'],
                processing_status="ready"
            )
            subject_data_list.append(subject_data)
        else:
            logger.warning(f"Incomplete file set for subject {basename}")
    
    return subject_data_list


def fig_to_base64(fig: plt.Figure) -> str:
    """
    Convert matplotlib figure to base64 string for web display.
    
    Args:
        fig: Matplotlib figure object
        
    Returns:
        Base64-encoded PNG image string
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)  # Important: cleanup matplotlib figures
    return img_base64


def process_session_data(session_data: SessionData) -> None:
    """
    Process all subjects in a session to generate component data.
    
    Args:
        session_data: Session data to process
    """
    for subject in session_data.subjects:
        try:
            process_subject_components(subject)
            subject.processing_status = "completed"
        except Exception as e:
            logger.error(f"Error processing subject {subject.subject_id}: {str(e)}")
            subject.processing_status = "error"
            subject.error_message = str(e)


def process_subject_components(subject: SubjectData) -> None:
    """
    Process individual subject to generate component data and images.
    
    Args:
        subject: Subject data to process
    """
    # Load raw and ICA data
    raw_obj = load_raw_data(subject.raw_file_path)
    ica_obj = load_ica_data(subject.ica_file_path)
    
    # Load existing results if available
    existing_results = {}
    if subject.results_file_path and Path(subject.results_file_path).exists():
        results_df = pd.read_csv(subject.results_file_path)
        for _, row in results_df.iterrows():
            existing_results[row['component_index']] = {
                'label': row['label'],
                'confidence': row['confidence'],
                'reason': row['reason'],
                'exclude_vision': row['exclude_vision']
            }
    
    # Generate component data
    components = []
    n_components = ica_obj.n_components_
    
    for comp_idx in range(n_components):
        try:
            # Generate component plot using web helper
            image_base64 = plot_component_for_web(
                ica_obj, raw_obj, comp_idx
            )
            
            # Get existing classification or use defaults
            if comp_idx in existing_results:
                result = existing_results[comp_idx]
                label = result['label']
                confidence = result['confidence']
                reason = result['reason']
                exclude_vision = result['exclude_vision']
            else:
                # Default classification for components without existing results
                label = "other_artifact"  # Conservative default
                confidence = 0.5
                reason = "No existing classification available"
                exclude_vision = True
            
            # Create component data
            component_data = ComponentData(
                component_index=comp_idx,
                component_name=f"IC{comp_idx}",
                current_label=label,
                confidence=confidence,
                reason=reason,
                exclude_vision=exclude_vision,
                image_base64=image_base64
            )
            
            components.append(component_data)
            
        except Exception as e:
            logger.error(f"Error processing component {comp_idx} for subject {subject.subject_id}: {str(e)}")
            continue
    
    subject.components = components


def export_override_csvs(session_data: SessionData) -> List[str]:
    """
    Export override CSV files for all subjects in session.
    
    Args:
        session_data: Session data containing override information
        
    Returns:
        List of paths to exported CSV files
    """
    export_paths = []
    
    for subject in session_data.subjects:
        # Check if subject has any overrides
        has_overrides = any(comp.override_label for comp in subject.components)
        
        if has_overrides:
            export_path = export_subject_override_csv(subject)
            if export_path:
                export_paths.append(export_path)
    
    return export_paths


def export_subject_override_csv(subject: SubjectData) -> Optional[str]:
    """
    Export override CSV for a single subject.
    
    Args:
        subject: Subject data to export
        
    Returns:
        Path to exported CSV file, or None if no overrides
    """
    # Collect override data
    override_data = []
    
    for component in subject.components:
        if component.override_label:
            override_data.append({
                'component_index': component.component_index,
                'component_name': component.component_name,
                'label': component.override_label,
                'confidence': component.override_confidence or 1.0,
                'reason': component.override_reason or "Manual override",
                'exclude_vision': component.override_label != "brain"
            })
    
    if not override_data:
        return None
    
    # Create DataFrame and export
    df = pd.DataFrame(override_data)
    
    # Generate output path
    raw_path = Path(subject.raw_file_path)
    output_path = raw_path.parent / f"{subject.subject_id}_icvis_results_override.csv"
    
    # Save CSV
    df.to_csv(output_path, index=False)
    
    logger.info(f"Exported override CSV for subject {subject.subject_id}: {output_path}")
    
    return str(output_path)


def create_override_zip(session_data: SessionData) -> str:
    """
    Create a zip file containing all override CSV files for a session.
    
    Args:
        session_data: Session data to create zip for
        
    Returns:
        Path to created zip file
    """
    # Export individual CSV files first
    csv_paths = export_override_csvs(session_data)
    
    if not csv_paths:
        raise HTTPException(status_code=400, detail="No override data to export")
    
    # Create zip file
    folder_path = Path(session_data.folder_path)
    zip_path = folder_path / f"icvision_overrides_{session_data.session_id[:8]}.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for csv_path in csv_paths:
            csv_file = Path(csv_path)
            zipf.write(csv_file, csv_file.name)
    
    logger.info(f"Created override zip file: {zip_path}")
    
    return str(zip_path)