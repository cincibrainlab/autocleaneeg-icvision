"""
Core functionality for ICVision.

This module contains the main label_components function that orchestrates
the entire ICA component classification workflow using OpenAI Vision API.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mne
import pandas as pd

from .api import classify_components_batch
from .config import DEFAULT_EXCLUDE_LABELS
from .plotting import save_ica_data
from .reports import generate_classification_report
from .utils import (
    check_eeglab_ica_availability,
    create_output_directory,
    extract_input_basename,
    format_summary_stats,
    load_ica_data,
    load_raw_data,
    save_cleaned_raw_data,
    save_results,
    validate_api_key,
    validate_classification_results,
    validate_inputs,
)

# Set up logging for the module
logger = logging.getLogger("icvision.core")


def label_components(
    raw_data: Union[str, Path, mne.io.Raw],
    ica_data: Optional[Union[str, Path, mne.preprocessing.ICA]] = None,
    api_key: Optional[str] = None,
    confidence_threshold: float = 0.8,
    auto_exclude: bool = True,
    labels_to_exclude: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    generate_report: bool = True,
    batch_size: int = 10,
    max_concurrency: int = 4,
    model_name: str = "gpt-4.1",
    custom_prompt: Optional[str] = None,
    psd_fmax: Optional[float] = None,
) -> Tuple[mne.io.Raw, mne.preprocessing.ICA, pd.DataFrame]:
    """
    Classify ICA components using OpenAI Vision API and apply artifact rejection.

    This is the main function of ICVision that orchestrates the entire workflow:
    1. Load raw EEG and ICA data from files or objects
    2. Generate component visualizations
    3. Classify components using OpenAI Vision API
    4. Update ICA object with classifications and exclusions
    5. Apply artifact rejection to raw data
    6. Generate comprehensive report (optional)

    Args:
        raw_data: Raw EEG data. Can be:
                 - Path to EEGLAB .set file
                 - Path to MNE .fif file
                 - Existing mne.io.Raw object
        ica_data: ICA decomposition (optional if raw_data is .set file with ICA). Can be:
                 - Path to MNE .fif file containing ICA
                 - Path to EEGLAB .set file containing ICA
                 - Existing mne.preprocessing.ICA object
                 - None (auto-detects ICA from .set file if raw_data is .set format)
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
        confidence_threshold: Minimum confidence for auto-exclusion (0.0-1.0).
        auto_exclude: Whether to automatically exclude classified artifact components.
        labels_to_exclude: List of labels to exclude. If None, excludes all except 'brain'.
        output_dir: Directory for saving results. If None, creates 'icvision_results'.
        generate_report: Whether to generate PDF report with visualizations.
        batch_size: Number of components to classify per API request (1-20).
        max_concurrency: Maximum concurrent API requests (1-10).
        model_name: OpenAI model to use (e.g., 'gpt-4.1', 'gpt-4.1-mini').
        custom_prompt: Custom classification prompt. If None, uses default.
        psd_fmax: Maximum frequency for PSD plot (default: None, uses 80 Hz or Nyquist).

    Returns:
        Tuple containing:
        - raw_cleaned: Raw data with artifacts removed
        - ica_updated: ICA object with component labels and exclusions
        - results_df: DataFrame with classification results

    Raises:
        FileNotFoundError: If input files don't exist.
        ValueError: If inputs are invalid or incompatible.
        RuntimeError: If API calls fail or processing errors occur.

    Example:
        >>> # Using EEGLAB .set file with auto-detected ICA
        >>> raw, ica, results = label_components(
        ...     raw_data="data/sub-01_eeg.set",  # ICA auto-detected from .set file
        ...     api_key="sk-...",
        ...     output_dir="results/"
        ... )

        >>> # Using separate files
        >>> raw, ica, results = label_components(
        ...     raw_data="data/sub-01_eeg.set",
        ...     ica_data="data/sub-01_ica.fif",
        ...     api_key="sk-...",
        ...     output_dir="results/"
        ... )

        >>> # Using MNE objects
        >>> raw, ica, results = label_components(
        ...     raw_data=raw_obj,
        ...     ica_data=ica_obj,
        ...     confidence_threshold=0.9
        ... )

        >>> # Print summary
        >>> print(f"Processed {len(results)} components")
        >>> print(f"Excluded {results['exclude_vision'].sum()} artifacts")
    """
    logger.debug("Starting ICVision component classification workflow")

    # Suppress MNE montage warnings for cleaner output
    # These warnings about EOG channel positions don't affect ICA classification
    import warnings

    warnings.filterwarnings("ignore", message="Not setting positions.*eog channels.*montage", category=RuntimeWarning)

    # Step 1: Validate and prepare inputs
    logger.debug("Loading and validating input data...")

    # Validate API key early
    validated_api_key = validate_api_key(api_key)

    # Track original raw data path for format preservation and extract basename
    original_raw_path = raw_data if isinstance(raw_data, (str, Path)) else None
    input_basename = extract_input_basename(original_raw_path)

    # Load data
    try:
        raw = load_raw_data(raw_data)

        # Auto-detect ICA from .set file if ica_data is None
        if ica_data is None:
            if isinstance(raw_data, (str, Path)):
                raw_path = Path(raw_data)
                if raw_path.suffix.lower() == ".set":
                    logger.info("Attempting to auto-detect ICA data from EEGLAB .set file: %s", raw_path)
                    if check_eeglab_ica_availability(raw_path):
                        ica_data = raw_path
                        logger.debug("Successfully detected ICA data in .set file")
                    else:
                        raise ValueError(
                            f"No ICA components found in EEGLAB .set file: {raw_path}. "
                            f"Run ICA decomposition in EEGLAB first, or provide separate MNE .fif ICA file."
                        )
                else:
                    raise ValueError(
                        "ica_data parameter is required when raw_data is not an EEGLAB .set file with ICA data"
                    )
            else:
                raise ValueError("ica_data parameter is required when raw_data is an MNE object")

        ica = load_ica_data(ica_data)
    except Exception as e:
        logger.error("Failed to load input data: %s", e)
        raise

    # Validate compatibility
    validate_inputs(raw, ica)

    # Set up configuration
    if labels_to_exclude is None:
        labels_to_exclude = DEFAULT_EXCLUDE_LABELS.copy()

    # Create output directory
    output_path = create_output_directory(output_dir, input_basename)

    logger.debug(
        "Configuration: %d components, confidence_threshold=%s, model=%s, batch_size=%d",
        ica.n_components_,
        confidence_threshold,
        model_name,
        batch_size,
    )

    # Step 2: Classify components using OpenAI Vision API
    logger.debug("Classifying ICA components using OpenAI Vision API...")

    try:
        results_df, cost_tracking = classify_components_batch(
            ica_obj=ica,
            raw_obj=raw,
            api_key=validated_api_key,
            model_name=model_name,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            custom_prompt=custom_prompt,
            confidence_threshold=confidence_threshold,
            auto_exclude=auto_exclude,
            labels_to_exclude=labels_to_exclude,
            output_dir=output_path,
            psd_fmax=psd_fmax,
        )
    except Exception as e:
        logger.error("Component classification failed: %s", e)
        raise RuntimeError("Failed to classify components: {}".format(e))

    # Validate results
    if not validate_classification_results(results_df):
        raise RuntimeError("Invalid classification results received")

    # Step 3: Update ICA object with classifications
    logger.debug("Updating ICA object with classification results...")

    try:
        ica_updated = _update_ica_with_classifications(ica, results_df)
    except Exception as e:
        logger.error("Failed to update ICA object: %s", e)
        raise RuntimeError("Failed to update ICA object: {}".format(e))

    # Step 4: Apply artifact rejection
    logger.debug("Applying artifact rejection to raw data...")

    try:
        raw_cleaned = _apply_artifact_rejection(raw, ica_updated)
    except Exception as e:
        logger.error("Failed to apply artifact rejection: %s", e)
        raise RuntimeError("Failed to apply artifact rejection: {}".format(e))

    # Step 5: Save results
    logger.debug("Saving classification results...")

    try:
        # Save CSV results
        save_results(results_df, output_path, input_basename)

        # Save updated ICA object
        save_ica_data(ica_updated, output_path, input_basename)

        # Save cleaned raw data in original format
        cleaned_data_path = save_cleaned_raw_data(raw_cleaned, original_raw_path, output_path, input_basename)
        if cleaned_data_path:
            logger.info("Cleaned raw data saved to: %s", cleaned_data_path)

        # Generate summary statistics
        summary = format_summary_stats(results_df, cost_tracking, model_name)
        logger.info("\n%s", summary)

        # Save summary to file
        summary_filename = (
            f"{input_basename}_icvis_summary.txt" if input_basename != "icvision" else "classification_summary.txt"
        )
        summary_path = output_path / summary_filename
        with open(summary_path, "w") as f:
            f.write(summary)

    except Exception as e:
        logger.warning("Failed to save some results: %s", e)

    # Step 6: Generate comprehensive report
    if generate_report:
        logger.debug("Generating comprehensive PDF report...")
        try:
            # Extract filename for PDF footer
            source_filename = None
            if original_raw_path:
                source_filename = Path(original_raw_path).name

            report_path = generate_classification_report(
                ica_obj=ica_updated,
                raw_obj=raw_cleaned,
                results_df=results_df,
                output_dir=output_path,
                input_basename=input_basename,
                source_filename=source_filename,
                psd_fmax=psd_fmax,  # Pass through PSD frequency limit to PDF
            )
            logger.info("Report saved to: %s", report_path)
        except Exception as e:
            logger.warning("Failed to generate PDF report: %s", e)

    # Final summary
    excluded_count = results_df.get("exclude_vision", pd.Series(dtype=bool)).sum()
    logger.info(
        "ICVision workflow completed successfully! Processed %d components, excluded %d artifacts. "
        "Results saved to: %s",
        len(results_df),
        excluded_count,
        output_path,
    )

    return raw_cleaned, ica_updated, results_df


def _update_ica_with_classifications(ica: mne.preprocessing.ICA, results_df: pd.DataFrame) -> mne.preprocessing.ICA:
    """
    Update ICA object with classification results.

    Args:
        ica: Original ICA object.
        results_df: DataFrame with classification results.

    Returns:
        Updated ICA object with labels and exclusions.
    """
    import numpy as np

    from .config import COMPONENT_LABELS, ICVISION_TO_MNE_LABEL_MAP

    # Modify ICA object in-place
    ica_updated = ica

    # Initialize labels_scores_ array
    n_components = ica_updated.n_components_
    n_label_categories = len(COMPONENT_LABELS)
    labels_scores_array = np.zeros((n_components, n_label_categories))

    # Fill scores array
    for comp_idx, row in results_df.iterrows():
        comp_idx = int(comp_idx)  # Index is component_index
        label = row["label"]
        confidence = float(row["confidence"])

        if label in COMPONENT_LABELS and comp_idx < n_components:
            label_idx = COMPONENT_LABELS.index(label)
            labels_scores_array[comp_idx, label_idx] = confidence

    ica_updated.labels_scores_ = labels_scores_array

    # Update labels_ dictionary
    ica_updated.labels_ = {mne_label: [] for mne_label in ICVISION_TO_MNE_LABEL_MAP.values()}

    for comp_idx, row in results_df.iterrows():
        comp_idx = int(comp_idx)  # Index is component_index
        icvision_label = row["label"]
        mne_label = ICVISION_TO_MNE_LABEL_MAP.get(icvision_label, "other")

        if comp_idx < n_components:
            ica_updated.labels_[mne_label].append(comp_idx)

    # Sort component lists
    for label in ica_updated.labels_:
        ica_updated.labels_[label].sort()

    # Update exclude list
    if "exclude_vision" in results_df.columns:
        excluded_components = results_df[results_df["exclude_vision"]].index.tolist()
    else:
        excluded_components = []

    # Ensure exclude list exists and merge with any existing exclusions
    if ica_updated.exclude is None:
        ica_updated.exclude = []

    # Add new exclusions
    current_exclusions = set(ica_updated.exclude)
    for comp_idx in excluded_components:
        current_exclusions.add(int(comp_idx))

    ica_updated.exclude = sorted(list(current_exclusions))

    logger.info(
        "Updated ICA object: %d new exclusions, %d total exclusions",
        len(excluded_components),
        len(ica_updated.exclude),
    )

    return ica_updated


def _apply_artifact_rejection(raw: mne.io.Raw, ica: mne.preprocessing.ICA) -> mne.io.Raw:
    """
    Apply ICA artifact rejection to raw data.

    Args:
        raw: Original raw data.
        ica: ICA object with exclusions set.

    Returns:
        Cleaned raw data with artifacts removed.
    """
    # Apply ICA in-place if there are components to exclude
    if ica.exclude:
        logger.info("Applying ICA to remove %d components", len(ica.exclude))
        ica.apply(raw)
    else:
        logger.info("No components marked for exclusion, returning original data")

    return raw
