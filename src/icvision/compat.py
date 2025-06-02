"""
ICLabel compatibility module for ICVision.

This module provides drop-in replacement functions that match the MNE-ICALabel API,
allowing ICVision to be used as a direct substitute for mne_icalabel.label_components.
"""

import logging
from typing import Dict, List, Optional, Union

import mne
import numpy as np

from .core import label_components as icvision_label_components

# Set up logging for the module
logger = logging.getLogger("icvision.compat")

# ICLabel class definitions (in order of probability matrix columns)
ICALABEL_CLASSES = [
    "brain",
    "muscle artifact", 
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other"
]

# ICVision to ICLabel class mapping
ICVISION_TO_ICALABEL_DISPLAY = {
    "brain": "brain",
    "eye": "eye blink",
    "muscle": "muscle artifact",
    "heart": "heart beat",
    "line_noise": "line noise",
    "channel_noise": "channel noise",
    "other_artifact": "other"
}

# ICVision to MNE ICLabel key mapping (for ica.labels_ dict)
ICVISION_TO_MNE_ICALABEL = {
    "brain": "brain",
    "eye": "eog",
    "muscle": "muscle",
    "heart": "ecg",
    "line_noise": "line_noise",
    "channel_noise": "ch_noise",
    "other_artifact": "other"
}

# Create reverse mapping for probability matrix indexing
ICALABEL_CLASS_TO_INDEX = {cls: idx for idx, cls in enumerate(ICALABEL_CLASSES)}


def create_probability_matrix(
    labels: List[str], 
    confidences: List[float]
) -> np.ndarray:
    """
    Convert ICVision single confidence values to ICLabel-style probability matrix.
    
    Uses Option 1 approach: predicted class gets the confidence value, 
    remaining probability is distributed uniformly across other classes.
    
    Args:
        labels: List of ICVision class labels (e.g., ['brain', 'eye', 'muscle'])
        confidences: List of confidence values (0.0-1.0)
        
    Returns:
        Probability matrix of shape (n_components, 7) where rows sum to 1.0
        
    Example:
        >>> labels = ['brain', 'eye', 'other_artifact']
        >>> confidences = [0.9, 0.7, 0.6]
        >>> prob_matrix = create_probability_matrix(labels, confidences)
        >>> prob_matrix.shape
        (3, 7)
        >>> np.allclose(prob_matrix.sum(axis=1), 1.0)
        True
    """
    n_components = len(labels)
    prob_matrix = np.zeros((n_components, 7), dtype=np.float32)
    
    for i, (icvision_label, confidence) in enumerate(zip(labels, confidences)):
        # Map ICVision label to ICLabel display name
        icalabel_display = ICVISION_TO_ICALABEL_DISPLAY.get(icvision_label, "other")
        
        # Get the index for this class in the probability matrix
        class_idx = ICALABEL_CLASS_TO_INDEX[icalabel_display]
        
        # Set predicted class probability to confidence
        prob_matrix[i, class_idx] = confidence
        
        # Distribute remaining probability uniformly across other 6 classes
        remaining_prob = 1.0 - confidence
        other_prob = remaining_prob / 6.0
        
        for j in range(7):
            if j != class_idx:
                prob_matrix[i, j] = other_prob
    
    # Ensure rows sum to exactly 1.0 (handle floating point precision)
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    prob_matrix = prob_matrix / row_sums
    
    return prob_matrix


def update_ica_with_icalabel_format(
    ica: mne.preprocessing.ICA,
    labels: List[str],
    confidences: List[float],
    prob_matrix: np.ndarray
) -> None:
    """
    Update ICA object with ICLabel-style labels and scores.
    
    Modifies the ICA object in-place to match what mne_icalabel.label_components does:
    - Sets ica.labels_scores_ to full probability matrix
    - Sets ica.labels_ to component indices grouped by class
    
    Args:
        ica: MNE ICA object to modify
        labels: List of ICVision class labels
        confidences: List of confidence values
        prob_matrix: Full probability matrix (n_components, 7)
    """
    # Set the full probability matrix
    ica.labels_scores_ = prob_matrix
    
    # Initialize labels dictionary with all MNE ICLabel keys
    ica.labels_ = {
        "brain": [],
        "muscle": [],
        "eog": [],
        "ecg": [],
        "line_noise": [],
        "ch_noise": [],
        "other": []
    }
    
    # Group component indices by predicted class
    for comp_idx, icvision_label in enumerate(labels):
        mne_key = ICVISION_TO_MNE_ICALABEL.get(icvision_label, "other")
        ica.labels_[mne_key].append(comp_idx)
    
    # Sort component lists for consistency
    for key in ica.labels_:
        ica.labels_[key].sort()
    
    logger.debug(
        "Updated ICA object with ICLabel-compatible format: %d components classified",
        len(labels)
    )


def label_components(
    inst: Union[mne.io.Raw, mne.BaseEpochs],
    ica: mne.preprocessing.ICA,
    method: str = "icvision",
    generate_report: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Union[np.ndarray, List[str]]]:
    """
    Drop-in replacement for mne_icalabel.label_components.
    
    Runs ICVision classification and returns results in ICLabel-compatible format.
    Also modifies the ICA object in-place to set labels_ and labels_scores_ attributes.
    
    Args:
        inst: Raw or Epochs object used to fit the ICA
        ica: Fitted ICA object  
        method: Classification method ('icvision' only)
        generate_report: Whether to generate ICVision PDF report (default: True)
        output_dir: Output directory for ICVision files (default: auto-generated)
        
    Returns:
        Dictionary with ICLabel-compatible structure:
        {
            'y_pred_proba': np.ndarray of shape (n_components,) with max probabilities,
            'labels': list of str with predicted class names (ICLabel vocabulary)
        }
        
    Raises:
        ValueError: If method is not 'icvision'
        RuntimeError: If ICVision classification fails
        
    Example:
        >>> # Drop-in replacement for mne_icalabel
        >>> from icvision.compat import label_components
        >>> result = label_components(raw, ica, method='icvision')
        >>> print(result['labels'])
        ['brain', 'other', 'eye blink', 'muscle artifact', ...]
        >>> print(ica.labels_scores_.shape)
        (20, 7)  # Full probability matrix set on ICA object
    """
    if method != "icvision":
        raise ValueError(f"Unsupported method '{method}'. Only 'icvision' is supported.")
    
    logger.info("Running ICVision classification with ICLabel-compatible output format")
    
    try:
        # Run ICVision classification using existing pipeline
        # Note: We use the inst (Raw/Epochs) object directly, and the fitted ICA
        _, ica_updated, results_df = icvision_label_components(
            raw_data=inst,
            ica_data=ica,
            auto_exclude=False,  # Don't modify exclude list in compatibility mode
            generate_report=generate_report,  # Allow ICVision PDF report generation
            output_dir=output_dir  # Use specified output directory
        )
        
        # Extract classification results
        labels = results_df['label'].tolist()
        confidences = results_df['confidence'].tolist()
        
        # Create ICLabel-style probability matrix
        prob_matrix = create_probability_matrix(labels, confidences)
        
        # Update ICA object with ICLabel-compatible format
        update_ica_with_icalabel_format(ica, labels, confidences, prob_matrix)
        
        # Create ICLabel-compatible return format
        # Convert ICVision labels to ICLabel display names
        icalabel_display_names = [
            ICVISION_TO_ICALABEL_DISPLAY.get(label, "other") 
            for label in labels
        ]
        
        # Extract max probability for each component (ICLabel y_pred_proba format)
        max_probabilities = np.max(prob_matrix, axis=1)
        
        result = {
            'y_pred_proba': max_probabilities,
            'labels': icalabel_display_names
        }
        
        logger.info(
            "ICVision classification completed: %d components, %d classes",
            len(labels),
            len(ICALABEL_CLASSES)
        )
        
        return result
        
    except Exception as e:
        logger.error("ICVision classification failed: %s", e)
        raise RuntimeError(f"ICVision classification failed: {e}")


def get_icalabel_class_mapping() -> Dict[str, str]:
    """
    Get the mapping from ICVision class names to ICLabel display names.
    
    Returns:
        Dictionary mapping ICVision labels to ICLabel labels
        
    Example:
        >>> mapping = get_icalabel_class_mapping()
        >>> mapping['eye']
        'eye blink'
    """
    return ICVISION_TO_ICALABEL_DISPLAY.copy()


def get_mne_icalabel_key_mapping() -> Dict[str, str]:
    """
    Get the mapping from ICVision class names to MNE ICLabel keys.
    
    Returns:
        Dictionary mapping ICVision labels to MNE ICLabel keys
        
    Example:
        >>> mapping = get_mne_icalabel_key_mapping()
        >>> mapping['eye']
        'eog'
    """
    return ICVISION_TO_MNE_ICALABEL.copy()


def validate_icalabel_compatibility(ica: mne.preprocessing.ICA) -> bool:
    """
    Validate that an ICA object has been properly formatted for ICLabel compatibility.
    
    Args:
        ica: ICA object to validate
        
    Returns:
        True if ICA object has proper ICLabel-compatible attributes
        
    Example:
        >>> from icvision.compat import label_components, validate_icalabel_compatibility
        >>> result = label_components(raw, ica)
        >>> validate_icalabel_compatibility(ica)
        True
    """
    # Check for required attributes
    if not hasattr(ica, 'labels_scores_'):
        logger.warning("ICA object missing labels_scores_ attribute")
        return False
        
    if not hasattr(ica, 'labels_'):
        logger.warning("ICA object missing labels_ attribute")
        return False
    
    # Validate labels_scores_ format
    if ica.labels_scores_ is None:
        logger.warning("ICA labels_scores_ is None")
        return False
        
    if ica.labels_scores_.shape[1] != 7:
        logger.warning(
            "ICA labels_scores_ has wrong shape: expected (n_components, 7), got %s",
            ica.labels_scores_.shape
        )
        return False
    
    # Validate labels_ format
    expected_keys = {"brain", "muscle", "eog", "ecg", "line_noise", "ch_noise", "other"}
    if set(ica.labels_.keys()) != expected_keys:
        logger.warning(
            "ICA labels_ has wrong keys: expected %s, got %s",
            expected_keys,
            set(ica.labels_.keys())
        )
        return False
    
    # Check probability matrix properties
    row_sums = ica.labels_scores_.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-6):
        logger.warning("ICA labels_scores_ rows do not sum to 1.0")
        return False
    
    logger.debug("ICA object passes ICLabel compatibility validation")
    return True