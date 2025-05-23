"""
Utility functions for ICVision.

This module contains helper functions for file I/O, validation, and data loading
that support the main functionality of the ICVision package.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd

from .config import COMPONENT_LABELS, ICVISION_TO_MNE_LABEL_MAP

# Set up logging for the module
logger = logging.getLogger(__name__)


def load_raw_data(
    raw_input: Union[str, Path, mne.io.Raw]
) -> mne.io.Raw:
    """
    Load raw EEG data from file path or return existing Raw object.
    
    Supports EEGLAB .set/.fdt format and MNE-compatible formats.
    
    Args:
        raw_input: Either a file path (str/Path) or an existing mne.io.Raw object.
                  For EEGLAB format, provide path to .set file.
                  
    Returns:
        Loaded mne.io.Raw object.
        
    Raises:
        FileNotFoundError: If file path does not exist.
        ValueError: If file format is not supported.
        RuntimeError: If data loading fails.
        
    Example:
        >>> raw = load_raw_data("data/sub-01_task-rest_eeg.set")
        >>> raw = load_raw_data(existing_raw_object)
    """
    if isinstance(raw_input, mne.io.Raw):
        logger.info("Using provided mne.io.Raw object")
        return raw_input
    
    # Convert to Path object for easier handling
    file_path = Path(raw_input)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")
    
    file_extension = file_path.suffix.lower()
    
    try:
        if file_extension == ".set":
            logger.info(f"Loading EEGLAB data from: {file_path}")
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
        elif file_extension == ".fif":
            logger.info(f"Loading MNE FIF data from: {file_path}")
            raw = mne.io.read_raw_fif(file_path, preload=True)
        elif file_extension in [".bdf", ".edf"]:
            logger.info(f"Loading EDF/BDF data from: {file_path}")
            raw = mne.io.read_raw_edf(file_path, preload=True)
        elif file_extension == ".vhdr":
            logger.info(f"Loading BrainVision data from: {file_path}")
            raw = mne.io.read_raw_brainvision(file_path, preload=True)
        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: .set (EEGLAB), .fif (MNE), .edf, .bdf, .vhdr"
            )
            
        logger.info(
            f"Successfully loaded raw data: {raw.info['nchan']} channels, "
            f"{raw.n_times} samples, {raw.info['sfreq']:.1f} Hz"
        )
        return raw
        
    except Exception as e:
        raise RuntimeError(f"Failed to load raw data from {file_path}: {e}")


def load_ica_data(
    ica_input: Union[str, Path, mne.preprocessing.ICA]
) -> mne.preprocessing.ICA:
    """
    Load ICA data from file path or return existing ICA object.
    
    Supports MNE .fif format for ICA objects.
    
    Args:
        ica_input: Either a file path (str/Path) or an existing mne.preprocessing.ICA object.
                  
    Returns:
        Loaded mne.preprocessing.ICA object.
        
    Raises:
        FileNotFoundError: If file path does not exist.
        ValueError: If file format is not supported.
        RuntimeError: If data loading fails.
        
    Example:
        >>> ica = load_ica_data("data/sub-01_task-rest_ica.fif")
        >>> ica = load_ica_data(existing_ica_object)
    """
    if isinstance(ica_input, mne.preprocessing.ICA):
        logger.info("Using provided mne.preprocessing.ICA object")
        return ica_input
    
    # Convert to Path object for easier handling
    file_path = Path(ica_input)
    
    if not file_path.exists():
        raise FileNotFoundError(f"ICA file not found: {file_path}")
    
    file_extension = file_path.suffix.lower()
    
    try:
        if file_extension == ".fif":
            logger.info(f"Loading MNE ICA from: {file_path}")
            ica = mne.preprocessing.read_ica(file_path)
        else:
            raise ValueError(
                f"Unsupported ICA file format: {file_extension}. "
                f"Supported formats: .fif (MNE)"
            )
            
        logger.info(
            f"Successfully loaded ICA: {ica.n_components_} components"
        )
        return ica
        
    except Exception as e:
        raise RuntimeError(f"Failed to load ICA from {file_path}: {e}")


def validate_inputs(
    raw: mne.io.Raw,
    ica: mne.preprocessing.ICA
) -> None:
    """
    Validate that raw and ICA data are compatible.
    
    Args:
        raw: The mne.io.Raw object.
        ica: The mne.preprocessing.ICA object.
        
    Raises:
        ValueError: If inputs are not compatible.
    """
    # Check if ICA was fitted
    if not hasattr(ica, 'n_components_') or ica.n_components_ is None:
        raise ValueError("ICA object appears to not be fitted. Please fit ICA first.")
    
    # Check basic compatibility (number of channels)
    if len(ica.ch_names) != len(raw.ch_names):
        logger.warning(
            f"Channel count mismatch: ICA has {len(ica.ch_names)} channels, "
            f"Raw has {len(raw.ch_names)} channels. This may cause issues."
        )
    
    # Check for sufficient data length
    if raw.n_times < 1000:  # Arbitrary minimum
        logger.warning(
            f"Raw data is very short ({raw.n_times} samples). "
            f"Results may be unreliable."
        )
    
    logger.debug("Input validation passed")


def create_output_directory(output_dir: Optional[Union[str, Path]]) -> Path:
    """
    Create output directory for results.
    
    Args:
        output_dir: Directory path. If None, uses current directory.
        
    Returns:
        Path to the created directory.
    """
    if output_dir is None:
        output_dir = Path.cwd() / "icvision_results"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    return output_dir


def validate_api_key(api_key: Optional[str]) -> str:
    """
    Validate and retrieve OpenAI API key.
    
    Args:
        api_key: API key string or None to use environment variable.
        
    Returns:
        Valid API key string.
        
    Raises:
        ValueError: If no valid API key is found.
    """
    if api_key:
        return api_key
    
    # Try environment variable
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    
    raise ValueError(
        "No OpenAI API key provided. Either pass api_key parameter or set "
        "OPENAI_API_KEY environment variable."
    )


def save_results(
    results_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "icvision_results.csv"
) -> Path:
    """
    Save classification results to CSV file.
    
    Args:
        results_df: DataFrame with classification results.
        output_dir: Output directory.
        filename: Filename for the CSV file.
        
    Returns:
        Path to the saved file.
    """
    output_path = output_dir / filename
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to: {output_path}")
    return output_path


def format_summary_stats(results_df: pd.DataFrame) -> str:
    """
    Create a formatted summary of classification results.
    
    Args:
        results_df: DataFrame with classification results.
        
    Returns:
        Formatted string summary.
    """
    if results_df.empty:
        return "No classification results available."
    
    total_components = len(results_df)
    
    # Count by label
    label_counts = results_df['label'].value_counts()
    
    # Count excluded components
    excluded_count = results_df.get('exclude_vision', pd.Series(dtype=bool)).sum()
    
    summary_lines = [
        "ICVision Classification Summary:",
        "=" * 35,
        f"Total components analyzed: {total_components}",
        f"Components marked for exclusion: {excluded_count}",
        "",
        "Classification breakdown:",
    ]
    
    for label, count in label_counts.items():
        percentage = (count / total_components) * 100
        summary_lines.append(f"  {label:<15}: {count:3d} ({percentage:5.1f}%)")
    
    return "\n".join(summary_lines)


def validate_classification_results(results_df: pd.DataFrame) -> bool:
    """
    Validate that classification results are properly formatted.
    
    Args:
        results_df: DataFrame with classification results.
        
    Returns:
        True if results are valid, False otherwise.
    """
    required_columns = ['component_index', 'label', 'confidence', 'reason']
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in results_df.columns]
    if missing_columns:
        logger.error(f"Missing required columns in results: {missing_columns}")
        return False
    
    # Check label validity
    invalid_labels = set(results_df['label']) - set(COMPONENT_LABELS)
    if invalid_labels:
        logger.error(f"Invalid labels found in results: {invalid_labels}")
        return False
    
    # Check confidence range
    if not results_df['confidence'].between(0, 1).all():
        logger.error("Confidence values must be between 0 and 1")
        return False
    
    logger.debug("Classification results validation passed")
    return True 