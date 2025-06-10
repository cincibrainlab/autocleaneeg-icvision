"""
EEGLAB compatibility functions for ICVision.
Simple module with just the essential functions needed for MATLAB integration.
"""

from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from .core import label_components

# EEGLAB ICLabel class definitions
EEGLAB_CLASSES = ["Brain", "Muscle", "Eye", "Heart", "Line Noise", "Channel Noise", "Other"]

# ICVision to EEGLAB class mapping
CLASS_MAPPING = {
    "brain": "Brain",
    "muscle": "Muscle",
    "eye": "Eye",
    "heart": "Heart",
    "line_noise": "Line Noise",
    "channel_noise": "Channel Noise",
    "other_artifact": "Other",
}


def create_probability_matrix(labels: List[str], confidences: List[float]) -> np.ndarray:
    """Convert ICVision labels to EEGLAB 7-class probability matrix."""
    n_components = len(labels)
    prob_matrix = np.zeros((n_components, 7))

    class_to_index = {cls: idx for idx, cls in enumerate(EEGLAB_CLASSES)}

    for i, (label, confidence) in enumerate(zip(labels, confidences)):
        eeglab_class = CLASS_MAPPING.get(label, "Other")
        class_idx = class_to_index[eeglab_class]

        prob_matrix[i, class_idx] = confidence
        remaining_prob = (1.0 - confidence) / 6.0

        for j in range(7):
            if j != class_idx:
                prob_matrix[i, j] = remaining_prob

    # Normalize rows to sum to 1.0
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    prob_matrix = prob_matrix / row_sums

    return prob_matrix


def classify_components_for_matlab(eeg_file_path: Union[str, Path], mode: str = "classify") -> Dict:
    """
    Run ICVision classification and optionally export cleaned data.

    Args:
        eeg_file_path: Path to EEGLAB .set file
        mode: 'classify' or 'clean'

    Returns:
        Dictionary with ICLabel structure for MATLAB
    """
    # Apply cleaning based on mode
    apply_cleaning = mode == "clean"

    # Run ICVision classification
    raw_cleaned, ica_updated, results_df = label_components(
        raw_data=eeg_file_path,
        generate_report=False,
        auto_exclude=apply_cleaning,  # Actually exclude artifacts when cleaning
        confidence_threshold=0.8,
        labels_to_exclude=["eye", "muscle", "heart", "line_noise", "channel_noise", "other_artifact"],
    )

    # Log cleaning results
    if mode == "clean":
        excluded_components = ica_updated.exclude
        print(f"Cleaning mode: excluded {len(excluded_components)} components: {excluded_components}")
    else:
        print("Classification mode: no components excluded")

    # Extract results
    labels = results_df["label"].tolist()
    confidences = results_df["confidence"].tolist()

    # Create probability matrix
    prob_matrix = create_probability_matrix(labels, confidences)

    # Use actual excluded components instead of guessing from labels
    artifacts_removed = list(ica_updated.exclude)  # These are the actually excluded components

    # Export cleaned data if requested
    if mode == "clean":
        cleaned_file = str(eeg_file_path).replace(".set", "_cleaned.set")
        print(f"Exporting cleaned data to: {cleaned_file}")
        raw_cleaned.export(cleaned_file, fmt="eeglab", overwrite=True)

    # Create ICLabel structure for MATLAB
    ic_classification = {
        "classes": np.array(EEGLAB_CLASSES, dtype=object),
        "classification": prob_matrix,
        "version": "ICVision1.0",
        "n_components": len(labels),
        "artifacts_removed": artifacts_removed,
        "mode": mode,
    }

    return ic_classification
