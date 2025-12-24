"""
OpenAI API interaction for ICVision.

This module handles communication with the OpenAI Vision API using the new
responses endpoint for computer vision tasks. Includes batching image
classification requests and parsing structured JSON responses.
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
logger = logging.getLogger("icvision.api")


def classify_component_image_openai(
    image_path: Path,
    api_key: str,
    model_name: str,
    custom_prompt: Optional[str] = None,
) -> Tuple[str, float, str, Optional[Dict[str, Any]]]:
    """
    Sends a single component image to OpenAI Vision API for classification.

    Uses OpenAI's new responses API endpoint for computer vision tasks.
    Sends the image with base64 encoding and processes structured JSON responses.

    Args:
        image_path: Path to the component image file (WebP format preferred).
        api_key: OpenAI API key.
        model_name: OpenAI model to use (e.g., "gpt-4.1").
        custom_prompt: Optional custom prompt to use instead of default.

    Returns:
        Tuple: (label, confidence, reason, cost_info).
               Defaults to ("other_artifact", 1.0, "API error or parsing failure", None) on error.
               The label is one of the COMPONENT_LABELS.
    """
    if not image_path or not image_path.exists():
        logger.error("Invalid or non-existent image path: %s", image_path)
        return "other_artifact", 1.0, "Invalid image path", None

    import time
    function_start_time = time.time()
    logger.info("Starting classification for %s", image_path.name)
    
    prompt_to_use = custom_prompt or OPENAI_ICA_PROMPT

    try:
        import time
        encode_start_time = time.time()
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        encode_end_time = time.time()
        logger.debug("Base64 encoding for %s took %.3f seconds", image_path.name, encode_end_time - encode_start_time)

        client = openai.OpenAI(api_key=api_key)
        logger.debug("Sending %s to OpenAI (model: %s)...", image_path.name, model_name)

        # Use OpenAI's new responses API for computer vision
        # Text prompt and image are sent as separate input entries
        # Note: response_format not supported in responses API - structured output must be requested in prompt
        
        import time
        api_start_time = time.time()
        logger.info("Starting OpenAI API call for %s", image_path.name)
        
        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "user", "content": prompt_to_use},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/webp;base64,{base64_image}",
                        }
                    ],
                },
            ],
            temperature=0.2,  # Low temperature for more deterministic output
        )
        
        api_end_time = time.time()
        api_elapsed_time = api_end_time - api_start_time
        logger.info("OpenAI API call completed for %s in %.2f seconds", image_path.name, api_elapsed_time)

        # Parse structured response from new responses API
        # Response structure (based on OpenAI responses API format):
        # {
        #   "output": [
        #     {"type": "reasoning", "content": None, ...},  # reasoning item (may be first)
        #     {"type": "message", "content": [{"type": "output_text", "text": "..."}], ...}
        #   ]
        # }
        # Find the message item in output (may not be first due to reasoning)
        message_content = None
        if response and hasattr(response, "output") and response.output:
            for output_item in response.output:
                if (
                    hasattr(output_item, "type")
                    and output_item.type == "message"
                    and hasattr(output_item, "content")
                    and output_item.content
                    and len(output_item.content) > 0
                ):
                    content_item = output_item.content[0]
                    if hasattr(content_item, "text"):
                        message_content = content_item.text
                        break

        if message_content:
            try:
                import json

                parse_start_time = time.time()
                structured_response = json.loads(message_content)
                parse_end_time = time.time()
                logger.debug("JSON parsing for %s took %.3f seconds", image_path.name, parse_end_time - parse_start_time)

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
                    return "other_artifact", 1.0, f"Invalid label '{label}' returned. Reason: {reason}", None

                # Validate confidence range
                confidence = max(0.0, min(1.0, confidence))

                logger.debug(
                    "Structured classification for %s: Label=%s, Conf=%.2f",
                    image_path.name,
                    label,
                    confidence,
                )

                # Log response metadata and calculate costs (updated for new API structure)
                cost_info = None
                usage = getattr(response, "usage", None)
                if usage:
                    from .utils import calculate_openai_cost

                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)
                    cached_tokens = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0

                    cost_info = calculate_openai_cost(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        model_name=model_name,
                        cached_tokens=cached_tokens,
                    )
                    # Add token counts to cost_info for tracking
                    cost_info["input_tokens"] = input_tokens
                    cost_info["output_tokens"] = output_tokens
                    cost_info["cached_tokens"] = cached_tokens

                    logger.debug(
                        "Response ID: %s, Tokens: %d in/%d out/%d cached, Cost: $%.6f",
                        getattr(response, "id", "N/A"),
                        input_tokens,
                        output_tokens,
                        cached_tokens,
                        cost_info["total_cost"],
                    )
                else:
                    logger.debug("Response ID: %s, Usage: N/A", getattr(response, "id", "N/A"))

                function_end_time = time.time()
                function_elapsed_time = function_end_time - function_start_time
                logger.info("Classification completed for %s in %.2f seconds (Label: %s, Confidence: %.2f)",
                           image_path.name, function_elapsed_time, label, confidence)
                return label, confidence, reason, cost_info

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(
                    "Failed to parse structured response for %s: %s. Content: %s",
                    image_path.name,
                    e,
                    message_content[:200],
                )
                function_end_time = time.time()
                function_elapsed_time = function_end_time - function_start_time
                logger.warning("Classification failed for %s in %.2f seconds (JSON parsing error)",
                              image_path.name, function_elapsed_time)
                return "other_artifact", 1.0, f"JSON parsing error: {str(e)}", None

        function_end_time = time.time()
        function_elapsed_time = function_end_time - function_start_time
        logger.warning("Classification failed for %s in %.2f seconds (No valid response content)", 
                      image_path.name, function_elapsed_time)
        logger.error("No valid structured content in OpenAI response for %s", image_path.name)
        return "other_artifact", 1.0, "No valid structured response content", None

    except openai.APIConnectionError as e:
        function_end_time = time.time()
        function_elapsed_time = function_end_time - function_start_time
        logger.error("OpenAI API connection error for %s after %.2f seconds: %s", image_path.name, function_elapsed_time, e)
        return "other_artifact", 1.0, "API Connection Error: {}".format(str(e)[:100]), None
    except openai.AuthenticationError as e:
        logger.error("OpenAI API authentication error: %s. Check API key.", e)
        return (
            "other_artifact",
            1.0,
            "API Authentication Error: {}".format(str(e)[:100]),
            None,
        )
    except openai.RateLimitError as e:
        function_end_time = time.time()
        function_elapsed_time = function_end_time - function_start_time
        logger.error("OpenAI API rate limit exceeded for %s after %.2f seconds: %s", image_path.name, function_elapsed_time, e)
        return "other_artifact", 1.0, "API Rate Limit Error: {}".format(str(e)[:100]), None
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
            None,
        )
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return "other_artifact", 1.0, "Configuration error: {}".format(str(e)[:100]), None
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
            None,
        )


def _classify_single_component_wrapper(
    args_tuple: Tuple[int, Optional[Path], str, str, Optional[str]],
) -> Tuple[int, str, float, str, Optional[Dict[str, Any]]]:
    """Helper for parallel execution of classify_component_image_openai."""
    comp_idx, image_path, api_key, model_name, custom_prompt = args_tuple
    if image_path is None:
        return comp_idx, "other_artifact", 1.0, "Plotting failed for this component", None

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
    psd_fmax: Optional[float] = None,
    component_indices: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, dict]:
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
        psd_fmax: Maximum frequency for PSD plot (default: None, uses 80 Hz or Nyquist).
        component_indices: Optional list of component indices to classify. If None,
            all components are processed.

    Returns:
        Tuple of (pd.DataFrame with classification results, dict with cost tracking information).
    """
    logger.debug("Starting batch classification of ICA components.")

    if labels_to_exclude is None:
        labels_to_exclude = [lbl for lbl in COMPONENT_LABELS if lbl != "brain"]

    classification_results_list: List[Dict[str, Any]] = []
    import time
    batch_start_time = time.time()

    if component_indices is None:
        component_indices = list(range(ica_obj.n_components_))
    else:
        component_indices = sorted({i for i in component_indices if 0 <= i < ica_obj.n_components_})

    num_components = len(component_indices)

    # Initialize cost tracking
    total_cost_tracking = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cached_tokens": 0,
        "total_cost": 0.0,
        "requests_count": 0,
    }

    if num_components == 0:
        logger.warning("No ICA components found to classify.")
        return pd.DataFrame(), total_cost_tracking

    logger.info(
        "Starting classify_components_batch: processing %d components", num_components
    )

    # Always use a temporary directory for component images (keeps output dir clean)
    temp_dir_context = tempfile.TemporaryDirectory(prefix="icvision_temp_plots_")
    image_output_path = Path(temp_dir_context.name)

    try:
        logger.info("Generating component images in temporary directory: %s", image_output_path)

        # Use improved batch plotting with better error handling
        plotting_results = plot_components_batch(
            ica_obj,
            raw_obj,
            component_indices,
            image_output_path,
            batch_size=5,  # Process in small batches to manage memory
            psd_fmax=psd_fmax,
        )

        # Convert plotting results to the format expected by the classification pipeline
        component_plot_args = []
        for i in component_indices:
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
                    comp_idx, label, confidence, reason, cost_info = future.result()

                    # Update cost tracking with actual costs
                    if cost_info:
                        total_cost_tracking["total_input_tokens"] += cost_info.get("input_tokens", 0)
                        total_cost_tracking["total_output_tokens"] += cost_info.get("output_tokens", 0)
                        total_cost_tracking["total_cached_tokens"] += cost_info.get("cached_tokens", 0)
                        total_cost_tracking["total_cost"] += cost_info.get("total_cost", 0.0)
                        total_cost_tracking["requests_count"] += 1

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

                log_level = "debug" if label == "brain" and not exclude_this_component else "info"
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

        # Log actual costs from cost tracking
        if total_cost_tracking["requests_count"] > 0:
            logger.info(
                "Actual total cost for %d components: $%.6f "
                "(Model: %s, %d input + %d output + %d cached tokens, %d requests)",
                processed_count,
                total_cost_tracking["total_cost"],
                model_name,
                total_cost_tracking["total_input_tokens"],
                total_cost_tracking["total_output_tokens"],
                total_cost_tracking["total_cached_tokens"],
                total_cost_tracking["requests_count"],
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

    batch_end_time = time.time()
    batch_elapsed_time = batch_end_time - batch_start_time
    logger.info(
        "classify_components_batch completed: %d components processed in %.2f seconds (%.2f sec/component)",
        num_components,
        batch_elapsed_time,
        batch_elapsed_time / num_components if num_components > 0 else 0,
    )

    return results_df, total_cost_tracking
