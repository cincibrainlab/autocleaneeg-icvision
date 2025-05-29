"""
Utility functions for ICVision.

This module contains helper functions for file I/O, validation, and data loading
that support the main functionality of the ICVision package.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import mne
import pandas as pd

from .config import COMPONENT_LABELS

# Set up logging for the module
logger = logging.getLogger(__name__)


def load_raw_data(raw_input: Union[str, Path, mne.io.BaseRaw]) -> mne.io.BaseRaw:
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
    if isinstance(raw_input, mne.io.BaseRaw):
        logger.info("Using provided mne.io.BaseRaw object")
        return raw_input

    # Convert to Path object for easier handling
    file_path = Path(raw_input)

    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    file_extension = file_path.suffix.lower()

    if file_extension == ".set":
        logger.debug("Loading EEGLAB data from: %s", file_path)
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
    elif file_extension == ".fif":
        logger.debug("Loading MNE FIF data from: %s", file_path)
        raw = mne.io.read_raw_fif(file_path, preload=True)
    elif file_extension in [".bdf", ".edf"]:
        logger.debug("Loading EDF/BDF data from: %s", file_path)
        raw = mne.io.read_raw_edf(file_path, preload=True)
    elif file_extension == ".vhdr":
        logger.debug("Loading BrainVision data from: %s", file_path)
        raw = mne.io.read_raw_brainvision(file_path, preload=True)
    else:
        raise ValueError(
            "Unsupported file format: {}. Supported formats: .set (EEGLAB), .fif (MNE), .edf, .bdf, .vhdr".format(
                file_extension
            )
        )

    logger.debug(
        "Successfully loaded raw data: %d channels, %d samples, %.1f Hz",
        raw.info["nchan"],
        raw.n_times,
        raw.info["sfreq"],
    )
    return raw


def check_eeglab_ica_availability(set_file_path: Union[str, Path]) -> bool:
    """
    Check if an EEGLAB .set file contains ICA data.

    Args:
        set_file_path: Path to the EEGLAB .set file.

    Returns:
        True if ICA data is available, False otherwise.
    """
    try:
        # Attempt to read ICA data from the .set file
        mne.preprocessing.read_ica_eeglab(set_file_path)
        return True
    except Exception:
        # If any exception occurs, assume ICA data is not available
        return False


def load_ica_data(ica_input: Union[str, Path, mne.preprocessing.ICA]) -> mne.preprocessing.ICA:
    """
    Load ICA data from file path or return existing ICA object.

    Supports MNE .fif format and EEGLAB .set format for ICA objects.

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
        >>> ica = load_ica_data("data/sub-01_task-rest_eeg.set")  # EEGLAB with ICA
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

    if file_extension == ".fif":
        logger.debug("Loading MNE ICA from: %s", file_path)
        ica = mne.preprocessing.read_ica(file_path)
    elif file_extension == ".set":
        logger.debug("Loading EEGLAB ICA from: %s", file_path)
        ica = mne.preprocessing.read_ica_eeglab(file_path)
    else:
        raise ValueError(
            "Unsupported ICA file format: {}. Supported formats: .fif (MNE), .set (EEGLAB)".format(file_extension)
        )

    logger.debug("Successfully loaded ICA: %d components", ica.n_components_)
    return ica


def validate_inputs(raw: mne.io.Raw, ica: mne.preprocessing.ICA) -> None:
    """
    Validate that raw and ICA data are compatible.

    Args:
        raw: The mne.io.Raw object.
        ica: The mne.preprocessing.ICA object.

    Raises:
        ValueError: If inputs are not compatible.
    """
    # Check if ICA was fitted
    if not hasattr(ica, "n_components_") or ica.n_components_ is None:
        raise ValueError("ICA object appears to not be fitted. Please fit ICA first.")

    # Check basic compatibility (number of channels)
    if len(ica.ch_names) != len(raw.ch_names):
        logger.warning(
            "Channel count mismatch: ICA has %d channels, Raw has %d channels. This may cause issues.",
            len(ica.ch_names),
            len(raw.ch_names),
        )

    # Check for sufficient data length
    if raw.n_times < 1000:  # Arbitrary minimum
        logger.warning(
            "Raw data is very short (%d samples). Results may be unreliable.",
            raw.n_times,
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
    logger.debug("Output directory: %s", output_dir)
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
        "No OpenAI API key provided. Either pass api_key parameter or set " "OPENAI_API_KEY environment variable."
    )


def save_results(results_df: pd.DataFrame, output_dir: Path, filename: str = "icvision_results.csv") -> Path:
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
    if results_df.empty and len(results_df.columns) == 0:
        # Create empty CSV with expected headers for completely empty dataframe
        expected_columns = [
            "component_index",
            "component_name",
            "label",
            "confidence",
            "reason",
            "exclude_vision",
        ]
        empty_df_with_columns = pd.DataFrame(columns=expected_columns)
        empty_df_with_columns.to_csv(output_path, index=False)
    else:
        results_df.to_csv(output_path, index=False)
    logger.debug("Results saved to: %s", output_path)
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
        summary_lines = [
            "ICVision Classification Summary:",
            "=" * 35,
            "Total components classified: 0",
            "Components marked for exclusion: 0",
            "",
            "Classification breakdown:",
        ]
        for label in COMPONENT_LABELS:
            summary_lines.append(f"- {label.title()}: 0")
        return "\n".join(summary_lines)

    total_components = len(results_df)

    # Count by label
    label_counts = results_df["label"].value_counts()

    # Count excluded components
    excluded_count = results_df.get("exclude_vision", pd.Series(dtype=bool)).sum()

    summary_lines = [
        "ICVision Classification Summary:",
        "=" * 35,
        f"Total components classified: {total_components}",
        f"Components marked for exclusion: {excluded_count}",
        "",
        "Classification breakdown:",
    ]

    # Show all labels, even if count is 0
    for label in COMPONENT_LABELS:
        count = label_counts.get(label, 0)
        summary_lines.append(f"- {label.title()}: {count}")

    return "\n".join(summary_lines)


def validate_classification_results(results_df: pd.DataFrame) -> bool:
    """
    Validate that classification results are properly formatted.

    Args:
        results_df: DataFrame with classification results.

    Returns:
        True if results are valid, False otherwise.
    """
    required_cols = {
        "component_index",
        "label",
        "confidence",
        "reason",
        "exclude_vision",
    }

    # Check required columns
    if not required_cols.issubset(results_df.columns):
        missing = required_cols - set(results_df.columns)
        missing_col = next(iter(missing))  # Get first missing column for specific error message
        raise ValueError(f"Missing required column: {missing_col}")

    # Validate 'label' values
    invalid_labels = set(results_df["label"]) - set(COMPONENT_LABELS)
    if invalid_labels:
        logger.error("Invalid labels found in results: %s", invalid_labels)
        invalid_label = next(iter(invalid_labels))  # Get first invalid label
        raise ValueError(f"Invalid label '{invalid_label}' found")

    # Check confidence range and type
    try:
        invalid_confidences = results_df[~results_df["confidence"].between(0, 1)]
        if not invalid_confidences.empty:
            logger.error("Confidence values must be between 0 and 1")
            invalid_conf = invalid_confidences["confidence"].iloc[0]
            if isinstance(invalid_conf, str):
                raise ValueError(f"Confidence score '{invalid_conf}' is not a float")
            else:
                raise ValueError(f"Confidence score {invalid_conf:.2f} is outside the valid range")
    except TypeError:
        # Handle non-numeric confidence values
        for idx, conf in results_df["confidence"].items():
            if not isinstance(conf, (int, float)):
                raise ValueError(f"Confidence score '{conf}' is not a float")

    logger.debug("Classification results validation passed")
    return True
