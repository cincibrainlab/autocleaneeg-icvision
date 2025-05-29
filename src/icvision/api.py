"""
OpenAI API interaction for ICVision.

This module handles communication with the OpenAI Vision API, including
batching image classification requests and parsing responses.
"""

import base64
import concurrent.futures
import logging
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
from .plotting import plot_components_batch

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
        logger.error("Invalid or non-existent image path: %s", image_path)
        return "other_artifact", 1.0, "Invalid image path"

    prompt_to_use = custom_prompt or OPENAI_ICA_PROMPT

    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        client = openai.OpenAI(api_key=api_key)
        logger.debug("Sending %s to OpenAI (model: %s)...", image_path.name, model_name)

        # Define JSON schema for structured output
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "component_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": COMPONENT_LABELS},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reason": {"type": "string", "description": "Detailed reasoning for the classification"},
                    },
                    "required": ["label", "confidence", "reason"],
                    "additionalProperties": False,
                },
            },
        }

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
                            },
                        },
                    ],
                }
            ],
            response_format=response_format,
            temperature=0.2,  # Low temperature for more deterministic output
        )

        # Parse structured response
        if response and response.choices and response.choices[0].message:
            message_content = response.choices[0].message.content
            if message_content:
                try:
                    import json

                    structured_response = json.loads(message_content)

                    label = structured_response.get("label", "").lower()
                    confidence = float(structured_response.get("confidence", 0.0))
                    reason = structured_response.get("reason", "")

                    # Validate label
                    if label not in COMPONENT_LABELS:
                        logger.warning(
                            "OpenAI returned unexpected label '%s' for %s. Falling back to other_artifact.",
                            label,
                            image_path.name,
                        )
                        return "other_artifact", 1.0, f"Invalid label '{label}' returned. Reason: {reason}"

                    # Validate confidence range
                    confidence = max(0.0, min(1.0, confidence))

                    logger.debug(
                        "Structured classification for %s: Label=%s, Conf=%.2f",
                        image_path.name,
                        label,
                        confidence,
                    )

                    # Log response metadata
                    logger.debug(
                        "Response ID: %s, Usage: %s",
                        getattr(response, "id", "N/A"),
                        getattr(response, "usage", "N/A"),
                    )

                    return label, confidence, reason

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.error(
                        "Failed to parse structured response for %s: %s. Content: %s",
                        image_path.name,
                        e,
                        message_content[:200],
                    )
                    return "other_artifact", 1.0, f"JSON parsing error: {str(e)}"

        logger.error("No valid structured content in OpenAI response for %s", image_path.name)
        return "other_artifact", 1.0, "No valid structured response content"

    except openai.APIConnectionError as e:
        logger.error("OpenAI API connection error for %s: %s", image_path.name, e)
        return "other_artifact", 1.0, "API Connection Error: {}".format(str(e)[:100])
    except openai.AuthenticationError as e:
        logger.error("OpenAI API authentication error: %s. Check API key.", e)
        return (
            "other_artifact",
            1.0,
            "API Authentication Error: {}".format(str(e)[:100]),
        )
    except openai.RateLimitError as e:
        logger.error("OpenAI API rate limit exceeded for %s: %s", image_path.name, e)
        return "other_artifact", 1.0, "API Rate Limit Error: {}".format(str(e)[:100])
    except openai.APIStatusError as e:
        logger.error(
            "OpenAI API status error for %s: Status=%s, Response=%s",
            image_path.name,
            e.status_code,
            e.response,
        )
        return (
            "other_artifact",
            1.0,
            "API Status Error {}: {}".format(e.status_code, str(e.response)[:100]),
        )
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return "other_artifact", 1.0, "Configuration error: {}".format(str(e)[:100])
    except Exception as e:
        logger.error(
            "Unexpected error during vision classification for %s: %s - %s",
            image_path.name,
            type(e).__name__,
            e,
        )
        return (
            "other_artifact",
            1.0,
            "Unexpected Exception: {} - {}".format(type(e).__name__, str(e)[:100]),
        )


def _classify_single_component_wrapper(
    args_tuple: Tuple[int, Optional[Path], str, str, Optional[str]],
) -> Tuple[int, str, float, str]:
    """Helper for parallel execution of classify_component_image_openai."""
    comp_idx, image_path, api_key, model_name, custom_prompt = args_tuple
    if image_path is None:
        return comp_idx, "other_artifact", 1.0, "Plotting failed for this component"

    # Call the classification function and prepend component index to its result tuple
    classification_result = classify_component_image_openai(image_path, api_key, model_name, custom_prompt)
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
    logger.debug("Starting batch classification of %d components.", ica_obj.n_components_)

    if labels_to_exclude is None:
        labels_to_exclude = [lbl for lbl in COMPONENT_LABELS if lbl != "brain"]

    classification_results_list: List[Dict[str, Any]] = []
    num_components = ica_obj.n_components_
    if num_components == 0:
        logger.warning("No ICA components found to classify.")
        return pd.DataFrame()

    # Always use a temporary directory for component images (keeps output dir clean)
    temp_dir_context = tempfile.TemporaryDirectory(prefix="icvision_temp_plots_")
    image_output_path = Path(temp_dir_context.name)

    try:
        logger.info("Generating component images in temporary directory: %s", image_output_path)

        # Use improved batch plotting with better error handling
        component_indices = list(range(num_components))
        plotting_results = plot_components_batch(
            ica_obj,
            raw_obj,
            component_indices,
            image_output_path,
            batch_size=5,  # Process in small batches to manage memory
        )

        # Convert plotting results to the format expected by the classification pipeline
        component_plot_args = []
        for i in range(num_components):
            image_path = plotting_results.get(i, None)
            component_plot_args.append((i, image_path, api_key, model_name, custom_prompt))

        processed_count = 0
        # Process in batches for concurrency management
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = []
            for i in range(0, len(component_plot_args), batch_size):
                current_batch_args = component_plot_args[i : i + batch_size]
                for single_arg_set in current_batch_args:
                    futures.append(executor.submit(_classify_single_component_wrapper, single_arg_set))

            for future in concurrent.futures.as_completed(futures):
                try:
                    comp_idx, label, confidence, reason = future.result()
                except Exception as e:
                    # This catches errors from _classify_single_component_wrapper itself
                    # Fallback if future.result() fails unexpectedly for a component
                    # We need comp_idx, so we try to find it if possible, but it might not be in e.
                    # This is a safety net; errors should ideally be caught within the wrapper.
                    logger.error("Error processing a component future: %s. Defaulting result.", e)
                    # We don't know which comp_idx this was if the future itself failed badly.
                    # This case should be rare if _classify_single_component_wrapper is robust.
                    # To handle, we might need to track futures to args more directly or accept some loss.
                    # For now, log and skip adding a result for this one failure.
                    continue

                mne_label = ICVISION_TO_MNE_LABEL_MAP.get(label, "other")
                exclude_this_component = (
                    auto_exclude and label in labels_to_exclude and confidence >= confidence_threshold
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
                    getattr(logging, log_level.upper()),
                    "IC%03d | Label: %-13s (MNE: %-10s) | Conf: %.2f | Excl: %s",
                    comp_idx,
                    label.upper(),
                    mne_label,
                    confidence,
                    exclude_this_component,
                )
                processed_count += 1

        logger.info(
            "OpenAI classification complete. Processed %d/%d components.",
            processed_count,
            num_components,
        )

    finally:
        # Always cleanup temporary image directory
        temp_dir_context.cleanup()
        logger.debug("Cleaned up temporary image directory: %s", temp_dir_context.name)

    # Ensure results are sorted by component index for consistency
    classification_results_list.sort(key=lambda x: x["component_index"])
    results_df = pd.DataFrame(classification_results_list)

    if not results_df.empty:
        results_df = results_df.set_index("component_index", drop=False)

    return results_df
