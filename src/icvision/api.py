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
    get_strip_prompt,
)
from .plotting import create_strip_image, plot_components_batch

# Set up logging for the module
logger = logging.getLogger("icvision.api")


def classify_component_image_openai(
    image_path: Path,
    api_key: str,
    model_name: str,
    custom_prompt: Optional[str] = None,
    base_url: Optional[str] = None,
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
        base_url: Optional custom API base URL for OpenAI-compatible endpoints.

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

        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        logger.debug("Sending %s to API (model: %s, base_url: %s)...", image_path.name, model_name, base_url or "default")

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
    args_tuple: Tuple[int, Optional[Path], str, str, Optional[str], Optional[str]],
) -> Tuple[int, str, float, str, Optional[Dict[str, Any]]]:
    """Helper for parallel execution of classify_component_image_openai."""
    comp_idx, image_path, api_key, model_name, custom_prompt, base_url = args_tuple
    if image_path is None:
        return comp_idx, "other_artifact", 1.0, "Plotting failed for this component", None

    # Call the classification function and prepend component index to its result tuple
    classification_result = classify_component_image_openai(image_path, api_key, model_name, custom_prompt, base_url)
    return (comp_idx,) + classification_result


# ============================================================================
# Strip Classification Functions (Phase 1 Implementation)
# ============================================================================


def classify_strip_image(
    image_path: Path,
    component_indices: List[int],
    api_key: str,
    model_name: str = "gpt-5.2",
    base_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Classify multiple ICA components from a single strip image.

    Sends a strip image containing multiple components to the vision API
    and parses the batch response.

    Args:
        image_path: Path to the strip image (webp format)
        component_indices: List of component indices in the strip (in order)
        api_key: OpenAI API key
        model_name: Model to use (default: gpt-5.2)
        base_url: Optional custom API base URL

    Returns:
        List of classification results, each with keys:
        - component_idx: int (actual IC index)
        - label: str (classification label)
        - confidence: float (0.0-1.0)
        - reason: str (explanation)

    Example:
        >>> results = classify_strip_image(
        ...     Path("strip.webp"),
        ...     [0, 1, 2, 3, 4, 5, 6, 7, 8],
        ...     api_key="sk-..."
        ... )
        >>> for r in results:
        ...     print(f"IC{r['component_idx']}: {r['label']}")
    """
    import json
    import time

    if not image_path or not image_path.exists():
        logger.error("Invalid or non-existent strip image path: %s", image_path)
        return []

    n_components = len(component_indices)
    if n_components == 0:
        logger.error("No component indices provided")
        return []

    # Generate labels for mapping: A, B, C, ... Z, AA, AB, ...
    single_labels = [chr(ord("A") + i) for i in range(26)]
    double_labels = ["A" + chr(ord("A") + i) for i in range(26)]
    all_labels = single_labels + double_labels
    labels = all_labels[:n_components]
    label_to_idx = {labels[i]: component_indices[i] for i in range(n_components)}

    start_time = time.time()
    logger.info(
        "Starting strip classification for %d components: %s",
        n_components,
        component_indices,
    )

    try:
        # Encode image
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        # Get prompt for this number of components
        prompt = get_strip_prompt(n_components)

        # Create client and send request
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        logger.debug(
            "Sending strip image to API (model: %s, base_url: %s, components: %d)",
            model_name,
            base_url or "default",
            n_components,
        )

        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "user", "content": prompt},
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
            temperature=0.2,
        )

        # Parse response - find the message content
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

        if not message_content:
            logger.error("No valid response content from strip classification")
            return []

        logger.debug("Raw strip response: %s", message_content[:500])

        # Parse JSON - handle markdown code blocks
        json_str = message_content.strip()
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_str = "\n".join(
                lines[1:-1] if lines[-1].startswith("```") else lines[1:]
            )

        raw_results = json.loads(json_str)

        # Map letter labels back to component indices
        results = []
        for r in raw_results:
            letter = r.get("component", "?")
            actual_idx = label_to_idx.get(letter)
            if actual_idx is None:
                logger.warning("Unknown component label in response: %s", letter)
                continue

            label = r.get("label", "other_artifact").lower()
            if label not in COMPONENT_LABELS:
                logger.warning("Unknown label '%s' for IC%d, using 'other_artifact'", label, actual_idx)
                label = "other_artifact"

            results.append(
                {
                    "component_idx": actual_idx,
                    "label": label,
                    "confidence": float(r.get("confidence", 0.8)),
                    "reason": r.get("reason", ""),
                }
            )

        elapsed = time.time() - start_time
        logger.info(
            "Strip classification completed: %d/%d components in %.2fs (%.2fs per component)",
            len(results),
            n_components,
            elapsed,
            elapsed / n_components if n_components > 0 else 0,
        )

        return results

    except json.JSONDecodeError as e:
        logger.error("JSON parse error in strip response: %s", e)
        return []
    except openai.APIError as e:
        logger.error("OpenAI API error during strip classification: %s", e)
        return []
    except Exception as e:
        logger.error("Unexpected error during strip classification: %s", e)
        return []


def classify_components_strip_batch(
    ica_obj: mne.preprocessing.ICA,
    raw_obj: mne.io.Raw,
    api_key: str,
    component_indices: List[int],
    model_name: str = "gpt-5.2",
    strip_size: int = 9,
    output_dir: Optional[Path] = None,
    psd_fmax: Optional[float] = None,
    base_url: Optional[str] = None,
    confidence_threshold: float = 0.8,
    auto_exclude: bool = True,
    labels_to_exclude: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Classify ICA components using strip layout (batches of 9).

    This function creates strip images containing multiple components
    and classifies them in batches, reducing API calls significantly.

    Args:
        ica_obj: Fitted MNE ICA object
        raw_obj: MNE Raw object used for ICA
        api_key: OpenAI API key
        component_indices: List of component indices to classify
        model_name: Model to use (default: gpt-5.2)
        strip_size: Components per strip image (default: 9)
        output_dir: Directory for temporary images
        psd_fmax: Maximum frequency for PSD plots
        base_url: Optional custom API base URL
        confidence_threshold: Min confidence for auto-exclusion
        auto_exclude: If True, mark components for exclusion
        labels_to_exclude: Labels to exclude (default: all except brain)

    Returns:
        Tuple of (results_df, metadata_dict) with same schema as
        classify_components_batch for drop-in compatibility.
    """
    import time

    start_time = time.time()
    n_total = len(component_indices)
    logger.info(
        "Starting strip batch classification: %d components in batches of %d",
        n_total,
        strip_size,
    )

    if labels_to_exclude is None:
        labels_to_exclude = cast(List[str], DEFAULT_CONFIG["labels_to_exclude"])

    # Precompute ICA sources once
    precomputed_sources = ica_obj.get_sources(raw_obj)

    # Create output directory
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="icvision_strip_"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    n_batches = (n_total + strip_size - 1) // strip_size

    # Process in batches of strip_size
    for batch_idx in range(n_batches):
        batch_start = batch_idx * strip_size
        batch_end = min(batch_start + strip_size, n_total)
        batch_indices = component_indices[batch_start:batch_end]

        logger.info(
            "Processing batch %d/%d: components %s",
            batch_idx + 1,
            n_batches,
            batch_indices,
        )

        # Create strip image for this batch
        strip_path = output_dir / f"strip_batch_{batch_idx}.webp"
        try:
            create_strip_image(
                ica_obj,
                raw_obj,
                batch_indices,
                strip_path,
                psd_fmax=psd_fmax,
                precomputed_sources=precomputed_sources,
            )
        except Exception as e:
            logger.error("Failed to create strip image for batch %d: %s", batch_idx, e)
            # Add fallback results for failed batch
            for idx in batch_indices:
                all_results.append(
                    {
                        "component_idx": idx,
                        "label": "other_artifact",
                        "confidence": 1.0,
                        "reason": f"Strip image creation failed: {e}",
                    }
                )
            continue

        # Classify the strip
        batch_results = classify_strip_image(
            strip_path,
            batch_indices,
            api_key,
            model_name=model_name,
            base_url=base_url,
        )

        if batch_results:
            all_results.extend(batch_results)
        else:
            # Fallback for failed classification
            logger.warning("Strip classification failed for batch %d, using fallback", batch_idx)
            for idx in batch_indices:
                all_results.append(
                    {
                        "component_idx": idx,
                        "label": "other_artifact",
                        "confidence": 1.0,
                        "reason": "Strip classification API call failed",
                    }
                )

    # Build DataFrame with same schema as single-image classification
    df_data = []
    for r in all_results:
        comp_idx = r["component_idx"]
        label = r["label"]
        confidence = r["confidence"]
        reason = r["reason"]

        # Determine exclusion
        should_exclude = (
            auto_exclude
            and label in labels_to_exclude
            and confidence >= confidence_threshold
        )

        df_data.append(
            {
                "component_index": comp_idx,
                "component_name": f"IC{comp_idx}",
                "label": label,
                "confidence": confidence,
                "reason": reason,
                "mne_label": ICVISION_TO_MNE_LABEL_MAP.get(label, "other"),
                "exclude_vision": should_exclude,
            }
        )

    results_df = pd.DataFrame(df_data)
    results_df = results_df.sort_values("component_index").reset_index(drop=True)
    results_df = results_df.set_index("component_index", drop=False)

    elapsed = time.time() - start_time
    metadata = {
        "total_components": n_total,
        "n_batches": n_batches,
        "strip_size": strip_size,
        "elapsed_seconds": elapsed,
        "seconds_per_component": elapsed / n_total if n_total > 0 else 0,
        "model_name": model_name,
        "layout": "strip",
    }

    logger.info(
        "Strip batch classification completed: %d components in %.2fs (%.2fs/component, %d API calls)",
        n_total,
        elapsed,
        elapsed / n_total if n_total > 0 else 0,
        n_batches,
    )

    return results_df, metadata


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
    base_url: Optional[str] = None,
    layout: str = "single",
    strip_size: int = 9,
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
        base_url: Optional custom API base URL for OpenAI-compatible endpoints.
        layout: Classification layout - "single" (one image per component) or
            "strip" (multiple components per image). Default: "single".
        strip_size: Components per strip image when layout="strip". Default: 9.

    Returns:
        Tuple of (pd.DataFrame with classification results, dict with cost tracking information).
    """
    # Validate and prepare component indices
    if component_indices is None:
        component_indices = list(range(ica_obj.n_components_))
    else:
        component_indices = sorted({i for i in component_indices if 0 <= i < ica_obj.n_components_})

    # Dispatch to strip layout if requested
    if layout == "strip":
        logger.info("Using strip layout with %d components per image", strip_size)
        return classify_components_strip_batch(
            ica_obj=ica_obj,
            raw_obj=raw_obj,
            api_key=api_key,
            component_indices=component_indices,
            model_name=model_name,
            strip_size=strip_size,
            output_dir=output_dir,
            psd_fmax=psd_fmax,
            base_url=base_url,
            confidence_threshold=confidence_threshold,
            auto_exclude=auto_exclude,
            labels_to_exclude=labels_to_exclude,
        )

    # Original single-image classification logic
    logger.debug("Starting batch classification of ICA components (single layout).")

    if labels_to_exclude is None:
        labels_to_exclude = [lbl for lbl in COMPONENT_LABELS if lbl != "brain"]

    classification_results_list: List[Dict[str, Any]] = []
    import time
    batch_start_time = time.time()

    # component_indices already validated and prepared above
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
            component_plot_args.append((i, image_path, api_key, model_name, custom_prompt, base_url))

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
