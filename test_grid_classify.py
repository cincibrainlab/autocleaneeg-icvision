#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mne>=1.6.0",
#     "openai>=1.0.0",
#     "matplotlib>=3.7.0",
#     "numpy>=1.24.0",
#     "scipy>=1.10.0",
# ]
# ///
"""
Test script for 2x2 grid component classification with GPT-5.2.

This script creates a 2x2 grid of ICA component images and sends them
to the vision API in a single request for batch classification.

Usage:
    OPENAI_API_KEY=dev-local-key OPENAI_BASE_URL=https://vision.autocleaneeg.org/v1 \
        uv run test_grid_classify.py /path/to/raw.fif /path/to/ica.fif
"""

import argparse
import base64
import json
import logging
import math
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.gridspec import GridSpec
from mne.time_frequency import psd_array_welch
from scipy.ndimage import uniform_filter1d

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Grid prompt template - supports variable number of components
GRID_PROMPT_TEMPLATE = """Classify each of the {n} ICA components shown in this grid (labeled {labels}).

Each component shows:
- Topography map (scalp distribution)
- Time series (first 2.5 seconds)
- ERP-style image (continuous data segments)
- Power spectrum (1-55Hz)

Categories:
- "brain": Dipolar pattern (can be central, parietal, OR lateral/temporal), 1/f spectrum with alpha (8-12Hz) or beta (13-30Hz) peaks. NOTE: Lateral/edge topography with alpha peak = brain, not muscle
- "eye": Frontal/periocular focus with low-frequency dominated spectrum (<4Hz) AND large slow deflections in time series. Frontal focal + slow deflections = eye, even if topography looks focal
- "muscle": Edge-focused topography AND flat/rising high-frequency spectrum (no alpha peak). Must have BOTH features
- "heart": ~1Hz rhythmic deflections in time series, broad scalp distribution
- "line_noise": Sharp narrow peak at 50/60Hz
- "channel_noise": Single isolated focal spot (one sensor) with flat/noisy spectrum AND erratic/random time series. NOT eye if spectrum is low-frequency dominated with slow deflections
- "other_artifact": Doesn't fit above categories

Respond with JSON array (one object per component):
{json_example}"""

def get_grid_prompt(n_components):
    """Generate prompt for N components."""
    all_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    labels = all_labels[:n_components]
    labels_str = ", ".join(labels)
    json_lines = [f'  {{"component": "{l}", "label": "category", "confidence": 0.0-1.0, "reason": "brief explanation"}}' for l in labels]
    json_example = "[\n" + ",\n".join(json_lines) + "\n]"
    return GRID_PROMPT_TEMPLATE.format(n=n_components, labels=labels_str, json_example=json_example)


def plot_single_component_subplot(ica_obj, raw_obj, component_idx, axes_dict, label, minimal=False, label_only=False):
    """Plot a single component into the provided axes dictionary.

    Args:
        ica_obj: MNE ICA object
        raw_obj: MNE Raw object
        component_idx: Component index to plot
        axes_dict: Dict with keys 'topo', 'ts', 'erp', 'psd'
        label: Label string (A, B, C, D) for this component
        minimal: If True, remove all text labels/titles/ticks
        label_only: If True, minimal but add component label to topography
    """
    ax_topo = axes_dict['topo']
    ax_ts = axes_dict['ts']
    ax_erp = axes_dict['erp']
    ax_psd = axes_dict['psd']

    # Get component data
    sources = ica_obj.get_sources(raw_obj)
    sfreq = sources.info["sfreq"]
    component_data = sources.get_data(picks=[component_idx])[0]

    # 1. Topography
    try:
        ica_obj.plot_components(
            picks=component_idx,
            axes=ax_topo,
            ch_type="eeg",
            show=False,
            colorbar=False,
            cmap="jet",
            outlines="head",
            sensors=True,
            contours=6,
        )
        if not minimal and not label_only:
            ax_topo.set_title(f"{label}: IC{component_idx} Topo", fontsize=9)
        elif label_only:
            ax_topo.set_title("")
            # Add label with IC index in top-left corner (e.g., "A:IC44")
            label_text = f"{label}:IC{component_idx}"
            ax_topo.text(0.05, 0.95, label_text, transform=ax_topo.transAxes,
                        fontsize=10, fontweight='bold', va='top', ha='left',
                        color='white', bbox=dict(boxstyle='round,pad=0.2',
                        facecolor='black', alpha=0.7))
        else:
            ax_topo.set_title("")
    except Exception as e:
        if not minimal and not label_only:
            ax_topo.text(0.5, 0.5, "Topo failed", ha="center", va="center")
            ax_topo.set_title(f"{label}: IC{component_idx}", fontsize=9)
        elif label_only:
            ax_topo.set_title("")
            label_text = f"{label}:IC{component_idx}"
            ax_topo.text(0.05, 0.95, label_text, transform=ax_topo.transAxes,
                        fontsize=10, fontweight='bold', va='top', ha='left',
                        color='white', bbox=dict(boxstyle='round,pad=0.2',
                        facecolor='black', alpha=0.7))
        else:
            ax_topo.set_title("")
    ax_topo.set_xlabel("")
    ax_topo.set_ylabel("")
    ax_topo.set_xticks([])
    ax_topo.set_yticks([])

    # 2. Time series (first 2.5s)
    try:
        duration = 2.5
        max_samples = min(int(duration * sfreq), len(component_data))
        times_ms = (np.arange(max_samples) / sfreq) * 1000
        ax_ts.plot(times_ms, component_data[:max_samples], linewidth=0.5, color="dodgerblue")
        ax_ts.set_xlim(times_ms[0], times_ms[-1])
        if minimal or label_only:
            ax_ts.set_title("")
            ax_ts.set_xlabel("")
            ax_ts.set_ylabel("")
            ax_ts.set_xticks([])
            ax_ts.set_yticks([])
        else:
            ax_ts.set_title(f"Time Series (2.5s)", fontsize=8)
            ax_ts.set_xlabel("Time (ms)", fontsize=7)
            ax_ts.grid(True, linestyle=":", alpha=0.5)
            ax_ts.tick_params(axis="both", labelsize=6)
    except Exception as e:
        if not minimal and not label_only:
            ax_ts.text(0.5, 0.5, "TS failed", ha="center", va="center")

    # 3. ERP image (continuous data)
    try:
        comp_centered = component_data - np.mean(component_data)
        segment_duration = 1.5
        max_segments = 100
        segment_len = int(segment_duration * sfreq)
        if segment_len == 0:
            segment_len = 1

        samples_to_use = min(len(comp_centered), max_segments * segment_len)
        n_segments = samples_to_use // segment_len

        if n_segments > 0:
            erp_data = comp_centered[:n_segments * segment_len].reshape(n_segments, segment_len)
            if n_segments >= 3:
                erp_data = uniform_filter1d(erp_data, size=3, axis=0, mode="nearest")

            max_val = np.max(np.abs(erp_data))
            clim = (2/3) * max_val if max_val > 1e-9 else 1.0

            ax_erp.imshow(erp_data, aspect="auto", cmap="jet", vmin=-clim, vmax=clim)
            ax_erp.invert_yaxis()
            if minimal or label_only:
                ax_erp.set_title("")
                ax_erp.set_xlabel("")
                ax_erp.set_ylabel("")
                ax_erp.set_xticks([])
                ax_erp.set_yticks([])
            else:
                ax_erp.set_title("Continuous Data", fontsize=8)
                ax_erp.set_xlabel("Time", fontsize=7)
                ax_erp.set_ylabel("Segment", fontsize=7)
                ax_erp.tick_params(axis="both", labelsize=6)
        else:
            if not minimal and not label_only:
                ax_erp.text(0.5, 0.5, "No data", ha="center", va="center")
    except Exception as e:
        if not minimal and not label_only:
            ax_erp.text(0.5, 0.5, "ERP failed", ha="center", va="center")

    # 4. PSD (cut off at 55Hz to avoid notch filter dip at 60Hz)
    try:
        fmin, fmax = 1.0, min(55.0, sfreq / 2.0 - 0.5)
        n_fft = min(int(sfreq * 2), len(component_data))
        n_fft = max(n_fft, 256) if len(component_data) >= 256 else len(component_data)

        psds, freqs = psd_array_welch(
            component_data, sfreq=sfreq, fmin=fmin, fmax=fmax,
            n_fft=n_fft, n_overlap=n_fft // 2, verbose=False
        )
        psds_db = 10 * np.log10(np.maximum(psds, 1e-20))

        ax_psd.plot(freqs, psds_db, color="red", linewidth=0.8)
        ax_psd.set_xlim(freqs[0], freqs[-1])
        if minimal or label_only:
            ax_psd.set_title("")
            ax_psd.set_xlabel("")
            ax_psd.set_ylabel("")
            ax_psd.set_xticks([])
            ax_psd.set_yticks([])
        else:
            ax_psd.set_title(f"Power Spectrum (1-{int(fmax)}Hz)", fontsize=8)
            ax_psd.set_xlabel("Freq (Hz)", fontsize=7)
            ax_psd.set_ylabel("dB", fontsize=7)
            ax_psd.grid(True, linestyle="--", alpha=0.4)
            ax_psd.tick_params(axis="both", labelsize=6)
    except Exception as e:
        if not minimal and not label_only:
            ax_psd.text(0.5, 0.5, "PSD failed", ha="center", va="center")


def create_grid_image(ica_obj, raw_obj, component_indices, output_path, grid_size=None, compact=False, layout="grid"):
    """Create a grid of component plots.

    Args:
        ica_obj: MNE ICA object
        raw_obj: MNE Raw object
        component_indices: List of component indices
        output_path: Path to save the grid image
        grid_size: Tuple (rows, cols) or None for auto
        compact: If True, reduce whitespace aggressively
        layout: "grid" (2x2 per component), "label_only" (grid with labels), "strip" (horizontal row per component)

    Returns:
        Path to the saved image
    """
    n_components = len(component_indices)
    # Support up to 16 components (A-P)
    all_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    labels = all_labels[:n_components]

    # Handle strip layout (Option 4) - one row per component
    if layout == "strip":
        fig_width = 16
        fig_height = 2 * n_components
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=150)
        outer_gs = GridSpec(n_components, 4, figure=fig, hspace=0.1, wspace=0.05,
                            left=0.02, right=0.98, top=0.98, bottom=0.02)

        for i, comp_idx in enumerate(component_indices):
            label = labels[i]
            axes_dict = {
                'topo': fig.add_subplot(outer_gs[i, 0]),
                'ts': fig.add_subplot(outer_gs[i, 1]),
                'erp': fig.add_subplot(outer_gs[i, 2]),
                'psd': fig.add_subplot(outer_gs[i, 3]),
            }
            plot_single_component_subplot(ica_obj, raw_obj, comp_idx, axes_dict, label, minimal=True, label_only=True)

        plt.savefig(output_path, format="webp", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        logger.info(f"Saved grid image to {output_path}")
        return output_path

    # Standard grid layouts (grid, label_only, compact)
    # Determine grid layout
    if grid_size:
        n_rows, n_cols = grid_size
    else:
        # Auto layout based on component count
        if n_components <= 4:
            n_rows, n_cols = 2, 2
        elif n_components <= 6:
            n_rows, n_cols = 2, 3
        elif n_components <= 9:
            n_rows, n_cols = 3, 3
        else:
            n_cols = 4
            n_rows = (n_components + 3) // 4

    # Scale figure size based on grid
    if compact or layout == "label_only":
        fig_width = 6 * n_cols
        fig_height = 6 * n_rows
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=150)
        # Aggressive spacing reduction
        outer_gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.08, wspace=0.08,
                            left=0.02, right=0.98, top=0.97, bottom=0.02)
    else:
        fig_width = 10 * n_cols
        fig_height = 10 * n_rows
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=150)
        # Original spacing
        outer_gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2,
                            left=0.05, right=0.95, top=0.95, bottom=0.05)

    for i, comp_idx in enumerate(component_indices):
        if i >= n_rows * n_cols:
            break

        row, col = divmod(i, n_cols)
        label = labels[i]

        # Inner 2x2 grid for each component's plots
        if compact or layout == "label_only":
            inner_gs = outer_gs[row, col].subgridspec(2, 2, hspace=0.15, wspace=0.15)
        else:
            inner_gs = outer_gs[row, col].subgridspec(2, 2, hspace=0.4, wspace=0.3)

        axes_dict = {
            'topo': fig.add_subplot(inner_gs[0, 0]),
            'ts': fig.add_subplot(inner_gs[0, 1]),
            'erp': fig.add_subplot(inner_gs[1, 0]),
            'psd': fig.add_subplot(inner_gs[1, 1]),
        }

        # Determine plot mode
        if layout == "label_only":
            plot_single_component_subplot(ica_obj, raw_obj, comp_idx, axes_dict, label, minimal=False, label_only=True)
        elif compact:
            plot_single_component_subplot(ica_obj, raw_obj, comp_idx, axes_dict, label, minimal=True)
        else:
            plot_single_component_subplot(ica_obj, raw_obj, comp_idx, axes_dict, label)

    # Handle empty cells
    for i in range(len(component_indices), n_rows * n_cols):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(outer_gs[row, col])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    if not compact and layout == "grid":
        labels_str = ", ".join(labels)
        fig.suptitle(f"ICA Component Grid ({labels_str})", fontsize=16, fontweight="bold")

    plt.savefig(output_path, format="webp", bbox_inches="tight", pad_inches=0.02 if (compact or layout in ["label_only", "strip"]) else 0.1)
    plt.close(fig)

    logger.info(f"Saved grid image to {output_path}")
    return output_path


def classify_grid(image_path, api_key, base_url, model="gpt-5.2", n_components=4):
    """Send grid image to vision API for classification.

    Args:
        image_path: Path to grid image
        api_key: API key
        base_url: Base URL for API
        model: Model name
        n_components: Number of components in the grid

    Returns:
        List of classification results
    """
    import openai

    # Read and encode image
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    prompt = get_grid_prompt(n_components)
    logger.info(f"Sending {n_components}-component grid to {model} via {base_url}...")

    response = client.responses.create(
        model=model,
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

    # Parse response - find the message item
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
        logger.error("No valid response content")
        return []

    logger.info(f"Raw response: {message_content[:500]}...")

    # Parse JSON - handle markdown code blocks
    json_str = message_content.strip()
    if json_str.startswith("```"):
        # Remove markdown code block
        lines = json_str.split("\n")
        json_str = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    try:
        results = json.loads(json_str)
        return results
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Content: {message_content}")
        return []


def main():
    import os

    parser = argparse.ArgumentParser(description="Test 2x2 grid classification")
    parser.add_argument("raw_file", type=Path, help="Path to raw .fif file")
    parser.add_argument("ica_file", type=Path, help="Path to ICA .fif file")
    parser.add_argument("--components", type=str, default="0,1,2,3",
                        help="Comma-separated component indices (default: 0,1,2,3)")
    parser.add_argument("--model", type=str, default="gpt-5.2",
                        help="Model name (default: gpt-5.2)")
    parser.add_argument("--save-grid", type=Path, default=None,
                        help="Save grid image to this path (optional)")
    parser.add_argument("--compact", action="store_true",
                        help="Use compact layout with reduced whitespace (no labels)")
    parser.add_argument("--layout", type=str, default="grid",
                        choices=["grid", "label_only", "strip"],
                        help="Layout style: grid (2x2 with text), label_only (minimal with corner labels), strip (horizontal rows)")
    parser.add_argument("--no-classify", action="store_true",
                        help="Only generate the grid image, skip classification")
    args = parser.parse_args()

    # Get API config from environment
    api_key = os.environ.get("OPENAI_API_KEY", "dev-local-key")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://vision.autocleaneeg.org/v1")

    logger.info(f"Using API: {base_url}")
    logger.info(f"Model: {args.model}")

    # Parse component indices
    component_indices = [int(x.strip()) for x in args.components.split(",")]
    logger.info(f"Components to classify: {component_indices}")

    # Load data
    logger.info(f"Loading raw: {args.raw_file}")
    raw = mne.io.read_raw_fif(args.raw_file, preload=True, verbose=False)

    logger.info(f"Loading ICA: {args.ica_file}")
    ica = mne.preprocessing.read_ica(args.ica_file, verbose=False)

    logger.info(f"ICA has {ica.n_components_} components")

    # Validate component indices
    valid_indices = [i for i in component_indices if 0 <= i < ica.n_components_]
    if len(valid_indices) != len(component_indices):
        logger.warning(f"Some indices out of range, using: {valid_indices}")
        component_indices = valid_indices

    # Create grid image
    with tempfile.TemporaryDirectory() as tmpdir:
        if args.save_grid:
            grid_path = args.save_grid
        else:
            grid_path = Path(tmpdir) / "grid.webp"

        create_grid_image(ica, raw, component_indices, grid_path, compact=args.compact, layout=args.layout)

        if args.no_classify:
            print(f"\nGrid image saved to: {grid_path}")
            return 0

        # Classify
        results = classify_grid(grid_path, api_key, base_url, args.model, n_components=len(component_indices))

    # Display results
    if results:
        print("\n" + "=" * 60)
        print("CLASSIFICATION RESULTS")
        print("=" * 60)
        for r in results:
            comp = r.get("component", "?")
            label = r.get("label", "unknown")
            conf = r.get("confidence", 0.0)
            reason = r.get("reason", "")

            # Map letter to actual component index
            letter_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
            letter_to_idx = {letter_labels[i]: component_indices[i] for i in range(len(component_indices)) if i < len(letter_labels)}
            actual_idx = letter_to_idx.get(comp, "?")

            print(f"\n{comp} (IC{actual_idx}): {label.upper()}")
            print(f"   Confidence: {conf:.2f}")
            print(f"   Reason: {reason}")
        print("\n" + "=" * 60)
    else:
        print("No results returned")

    return 0


if __name__ == "__main__":
    sys.exit(main())
