"""
OpenAI API interaction for ICVision.

This module handles communication with the OpenAI Vision API, including
batching image classification requests and parsing responses.
"""

import base64
import concurrent.futures
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import mne
import openai
import pandas as pd

from .config import (
    COMPONENT_LABELS,
    DEFAULT_CONFIG,
    ICVISION_TO_MNE_LABEL_MAP,
    OPENAI_ICA_PROMPT,
)
from .plotting import plot_component_for_classification

# Set up logging for the module
logger = logging.getLogger(__name__)


def classify_component_image_openai(
    image_path: Path,
    api_key: str,
    model_name: str,
    custom_prompt: Optional[str] = None,
) -> Tuple[str, float, str]:
    """
    Sends a single component image to OpenAI Vision API for classification.

    Args:
        image_path: Path to the component image file (WebP format preferred).
        api_key: OpenAI API key.
        model_name: OpenAI model to use (e.g., "gpt-4.1").
        custom_prompt: Optional custom prompt to use instead of default.

    Returns:
        Tuple: (label, confidence, reason).
               Defaults to ("other_artifact", 1.0, "API error or parsing failure") on error.
               The label is one of the COMPONENT_LABELS.
    """
    if not image_path or not image_path.exists():
        logger.error(f"Invalid or non-existent image path: {image_path}")
        return "other_artifact", 1.0, "Invalid image path"

    prompt_to_use = custom_prompt or OPENAI_ICA_PROMPT

    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        client = openai.OpenAI(api_key=api_key)
        logger.debug(f"Sending {image_path.name} to OpenAI (model: {model_name})...")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_to_use},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/webp;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.1,  # Low temperature for more deterministic output
        )

        resp_text = None
        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            resp_text = response.choices[0].message.content.strip()

        if resp_text:
            logger.debug(
                f"Raw OpenAI response for {image_path.name}: '{resp_text[:100]}...'"
            )

            # Regex to parse ("label", confidence, "reason") format
            labels_pattern = "|".join(re.escape(label) for label in COMPONENT_LABELS)
            # Concatenate all parts of the regex pattern into a single string
            regex_pattern = (
                r"^\\s*\\(\\s*"
                r"['\"]?(" + labels_pattern + r")['\"]?"  # Group 1: Label
                r"\\s*,\\s*"
                r"([01](?:\\.\\d+)?)"  # Group 2: Confidence (0.0-1.0)
                r"\\s*,\\s*"
                r"['\"](.*?)['\"]"  # Group 3: Reasoning (non-greedy)
                r"\\s*\\)\\s*$"
            )
            match = re.search(regex_pattern, resp_text, re.IGNORECASE | re.DOTALL)

            if match:
                label = match.group(1).lower()
                if label not in COMPONENT_LABELS:
                    logger.warning(
                        f"OpenAI returned an unexpected label '{label}' not in "
                        f"COMPONENT_LABELS for {image_path.name}. Raw: '{resp_text}'"
                    )
                    return (
                        "other_artifact",
                        1.0,
                        f"Unexpected label: {label}. Parsed from: {resp_text}",
                    )

                confidence = float(match.group(2))
                reason = match.group(3).strip().replace('\\"', '"').replace("\\'", "'")
                logger.debug(
                    f"Parsed classification for {image_path.name}: Label={label}, Conf={confidence:.2f}"
                )
                return label, confidence, reason
            else:
                logger.warning(
                    f"Could not parse OpenAI response for {image_path.name}: '{resp_text}'. Defaulting."
                )
                return "other_artifact", 1.0, f"Failed to parse response: {resp_text}"
        else:
            logger.error(f"No text content in OpenAI response for {image_path.name}")
            return "other_artifact", 1.0, "Invalid response (no text content)"

    except openai.APIConnectionError as e:
        logger.error(f"OpenAI API connection error for {image_path.name}: {e}")
        return "other_artifact", 1.0, f"API Connection Error: {str(e)[:100]}"
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI API authentication error: {e}. Check API key.")
        return "other_artifact", 1.0, f"API Authentication Error: {str(e)[:100]}"
    except openai.RateLimitError as e:
        logger.error(f"OpenAI API rate limit exceeded for {image_path.name}: {e}")
        return "other_artifact", 1.0, f"API Rate Limit Error: {str(e)[:100]}"
    except openai.APIStatusError as e:
        logger.error(
            f"OpenAI API status error for {image_path.name}: Status={e.status_code}, "
            f"Response={e.response}"
        )
        return (
            "other_artifact",
            1.0,
            f"API Status Error {e.status_code}: {str(e.response)[:100]}",
        )
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return "other_artifact", 1.0, f"Configuration error: {str(e)[:100]}"
    except Exception as e:
        logger.error(
            f"Unexpected error during vision classification for {image_path.name}: "
            f"{type(e).__name__} - {e}"
        )
        return (
            "other_artifact",
            1.0,
            f"Unexpected Exception: {type(e).__name__} - {str(e)[:100]}",
        )


def _classify_single_component_wrapper(
    args_tuple: Tuple[int, Optional[Path], str, str, Optional[str]]
) -> Tuple[int, str, float, str]:
    """Helper for parallel execution of classify_component_image_openai."""
    comp_idx, image_path, api_key, model_name, custom_prompt = args_tuple
    if image_path is None:
        return comp_idx, "other_artifact", 1.0, "Plotting failed for this component"

    # Call the classification function and prepend component index to its result tuple
    classification_result = classify_component_image_openai(
        image_path, api_key, model_name, custom_prompt
    )
    return (comp_idx,) + classification_result


def classify_components_batch(
    ica_obj: mne.preprocessing.ICA,
    raw_obj: mne.io.Raw,
    api_key: str,
    model_name: str = cast(str, DEFAULT_CONFIG["model_name"]),
    batch_size: int = cast(int, DEFAULT_CONFIG["batch_size"]),
    max_concurrency: int = cast(int, DEFAULT_CONFIG["max_concurrency"]),
    custom_prompt: Optional[str] = None,
    confidence_threshold: float = cast(float, DEFAULT_CONFIG["confidence_threshold"]),
    auto_exclude: bool = cast(bool, DEFAULT_CONFIG["auto_exclude"]),
    labels_to_exclude: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Classifies ICA components in batches using OpenAI Vision API with parallel processing.

    Args:
        ica_obj: Fitted MNE ICA object.
        raw_obj: MNE Raw object used for ICA.
        api_key: OpenAI API key.
        model_name: OpenAI model (e.g., "gpt-4.1").
        batch_size: Number of images per concurrent processing batch (not API batch).
        max_concurrency: Max parallel API requests.
        custom_prompt: Optional custom prompt string.
        confidence_threshold: Min confidence for auto-exclusion.
        auto_exclude: If True, mark components for exclusion.
        labels_to_exclude: List of labels to exclude (e.g., ["eye", "muscle"]).
        output_dir: Directory to save temporary component images.

    Returns:
        pd.DataFrame with classification results for each component.
    """
    logger.info(f"Starting batch classification of {ica_obj.n_components_} components.")

    if labels_to_exclude is None:
        labels_to_exclude = [lbl for lbl in COMPONENT_LABELS if lbl != "brain"]

    classification_results_list: List[Dict[str, Any]] = []
    num_components = ica_obj.n_components_
    if num_components == 0:
        logger.warning("No ICA components found to classify.")
        return pd.DataFrame()

    # Use a temporary directory if output_dir is not specified for images
    temp_dir_context = tempfile.TemporaryDirectory(prefix="icvision_temp_plots_")
    image_output_path = output_dir if output_dir else Path(temp_dir_context.name)

    try:
        logger.info(f"Generating component images in: {image_output_path}")
        component_plot_args = []
        for i in range(num_components):
            try:
                image_path = plot_component_for_classification(
                    ica_obj, raw_obj, i, image_output_path, return_fig_object=False
                )
                component_plot_args.append(
                    (i, image_path, api_key, model_name, custom_prompt)
                )
            except Exception as plot_err:
                logger.warning(
                    f"Failed to plot component IC{i}: {plot_err}. Will mark as plotting_failed."
                )
                component_plot_args.append(
                    (i, None, api_key, model_name, custom_prompt)
                )

        processed_count = 0
        # Process in batches for concurrency management
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrency
        ) as executor:
            futures = []
            for i in range(0, len(component_plot_args), batch_size):
                current_batch_args = component_plot_args[i : i + batch_size]
                for single_arg_set in current_batch_args:
                    futures.append(
                        executor.submit(
                            _classify_single_component_wrapper, single_arg_set
                        )
                    )

            for future in concurrent.futures.as_completed(futures):
                try:
                    comp_idx, label, confidence, reason = future.result()
                except Exception as e:
                    # This catches errors from _classify_single_component_wrapper itself
                    # Fallback if future.result() fails unexpectedly for a component
                    # We need comp_idx, so we try to find it if possible, but it might not be in e.
                    # This is a safety net; errors should ideally be caught within the wrapper.
                    logger.error(
                        f"Error processing a component future: {e}. Defaulting result."
                    )
                    # We don't know which comp_idx this was if the future itself failed badly.
                    # This case should be rare if _classify_single_component_wrapper is robust.
                    # To handle, we might need to track futures to args more directly or accept some loss.
                    # For now, log and skip adding a result for this one failure.
                    continue

                mne_label = ICVISION_TO_MNE_LABEL_MAP.get(label, "other")
                exclude_this_component = (
                    auto_exclude
                    and label in labels_to_exclude
                    and confidence >= confidence_threshold
                )

                classification_results_list.append(
                    {
                        "component_index": comp_idx,
                        "component_name": f"IC{comp_idx}",
                        "label": label,
                        "mne_label": mne_label,
                        "confidence": confidence,
                        "reason": reason,
                        "exclude_vision": exclude_this_component,
                    }
                )

                log_level = (
                    "debug"
                    if label == "brain" and not exclude_this_component
                    else "warning" if exclude_this_component else "info"
                )
                logger.log(
                    logging.getLevelName(log_level.upper()),
                    f"IC{comp_idx:03d} | Label: {label.upper():<13} "
                    f"(MNE: {mne_label:<10}) | Conf: {confidence:.2f} | Excl: {exclude_this_component}",
                )
                processed_count += 1

        logger.info(
            f"OpenAI classification complete. Processed {processed_count}/{num_components} components."
        )

    finally:
        if isinstance(temp_dir_context, tempfile.TemporaryDirectory):
            temp_dir_context.cleanup()
            logger.debug("Cleaned up temporary image directory.")

    # Ensure results are sorted by component index for consistency
    classification_results_list.sort(key=lambda x: x["component_index"])
    results_df = pd.DataFrame(classification_results_list)

    if not results_df.empty:
        results_df = results_df.set_index("component_index", drop=False)

    return results_df
