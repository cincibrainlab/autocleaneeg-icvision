#!/usr/bin/env python3
"""
Generate component images for the human rater web app.

This script uses the EXACT same plotting code as test_grid_classify.py
to ensure consistency between API classification and human rating.

Usage:
    python generate_images.py --raw data.fif --ica ica.fif --output public/components/
    python generate_images.py --raw data.set --output public/components/  # Auto-detects ICA from .set
"""

import argparse
import json
import logging
import sys
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


def load_data(raw_path: str, ica_path: str = None) -> tuple:
    """Load raw data and ICA object."""
    raw_path = Path(raw_path)

    # Load raw data
    if raw_path.suffix == ".set":
        raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose=False)
        # Try to load ICA from .set file if no separate ICA provided
        if ica_path is None:
            try:
                ica = mne.preprocessing.read_ica_eeglab(raw_path)
                logger.info(f"Loaded ICA from EEGLAB .set file: {ica.n_components_} components")
            except Exception as e:
                raise ValueError(f"Could not extract ICA from .set file: {e}")
    else:
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)

    # Load separate ICA file if provided
    if ica_path is not None:
        ica_path = Path(ica_path)
        if ica_path.suffix == ".fif":
            ica = mne.preprocessing.read_ica(ica_path, verbose=False)
        else:
            raise ValueError(f"Unsupported ICA file format: {ica_path.suffix}")
        logger.info(f"Loaded ICA from {ica_path}: {ica.n_components_} components")

    logger.info(f"Loaded raw data: {len(raw.ch_names)} channels, {raw.n_times / raw.info['sfreq']:.1f}s")

    return raw, ica


def plot_single_component_strip(ica_obj, raw_obj, component_idx, output_path):
    """
    Plot a single component as a horizontal strip with 4 panels.

    This is the EXACT same plotting code from test_grid_classify.py
    to ensure visual consistency.

    Panels: Topography | Time Series (2.5s) | ERP Image | Power Spectrum (1-55Hz)
    """
    # Get component data
    sources = ica_obj.get_sources(raw_obj)
    sfreq = sources.info["sfreq"]
    component_data = sources.get_data(picks=[component_idx])[0]

    # Create figure with strip layout
    fig = plt.figure(figsize=(16, 2.5), dpi=150)
    gs = GridSpec(1, 4, figure=fig, wspace=0.08, left=0.02, right=0.98, top=0.92, bottom=0.08)

    ax_topo = fig.add_subplot(gs[0, 0])
    ax_ts = fig.add_subplot(gs[0, 1])
    ax_erp = fig.add_subplot(gs[0, 2])
    ax_psd = fig.add_subplot(gs[0, 3])

    label = f"IC{component_idx}"

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
        ax_topo.set_title("")
        # Add label in top-left corner
        ax_topo.text(0.05, 0.95, label, transform=ax_topo.transAxes,
                    fontsize=11, fontweight='bold', va='top', ha='left',
                    color='white', bbox=dict(boxstyle='round,pad=0.2',
                    facecolor='black', alpha=0.7))
    except Exception as e:
        ax_topo.text(0.5, 0.5, label, ha="center", va="center", fontsize=12, fontweight='bold')
        logger.warning(f"Could not plot topography for IC{component_idx}: {e}")

    ax_topo.set_xlabel("")
    ax_topo.set_ylabel("")
    ax_topo.set_xticks([])
    ax_topo.set_yticks([])

    # 2. Time series (first 2.5s) - exact same as test_grid_classify.py
    try:
        duration = 2.5
        max_samples = min(int(duration * sfreq), len(component_data))
        times_ms = (np.arange(max_samples) / sfreq) * 1000
        ax_ts.plot(times_ms, component_data[:max_samples], linewidth=0.5, color="dodgerblue")
        ax_ts.set_xlim(times_ms[0], times_ms[-1])
        ax_ts.set_title("")
        ax_ts.set_xlabel("")
        ax_ts.set_ylabel("")
        ax_ts.set_xticks([])
        ax_ts.set_yticks([])
    except Exception as e:
        logger.warning(f"Could not plot time series for IC{component_idx}: {e}")

    # 3. ERP image (continuous data) - exact same as test_grid_classify.py
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
            ax_erp.set_title("")
            ax_erp.set_xlabel("")
            ax_erp.set_ylabel("")
            ax_erp.set_xticks([])
            ax_erp.set_yticks([])
    except Exception as e:
        logger.warning(f"Could not plot ERP image for IC{component_idx}: {e}")

    # 4. PSD (cut off at 55Hz to avoid notch filter dip at 60Hz) - exact same as test_grid_classify.py
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
        ax_psd.set_title("")
        ax_psd.set_xlabel("")
        ax_psd.set_ylabel("")
        ax_psd.set_xticks([])
        ax_psd.set_yticks([])
    except Exception as e:
        logger.warning(f"Could not plot PSD for IC{component_idx}: {e}")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate component images for human rater app")
    parser.add_argument("--raw", required=True, help="Path to raw EEG data (.fif or .set)")
    parser.add_argument("--ica", help="Path to ICA file (.fif). Auto-detected for .set files.")
    parser.add_argument("--output", required=True, help="Output directory for images")
    parser.add_argument("--dataset", default="default", help="Dataset name for metadata")
    parser.add_argument("--format", default="png", choices=["png", "webp"], help="Image format")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    raw, ica = load_data(args.raw, args.ica)

    # Generate images for each component
    metadata = []
    n_components = ica.n_components_

    for idx in range(n_components):
        output_path = output_dir / f"ic_{idx:03d}.{args.format}"
        plot_single_component_strip(ica, raw, idx, output_path)

        info = {
            "ic_index": idx,
            "image_path": f"/components/{output_path.name}",
            "dataset": args.dataset,
            "model_label": None,
            "model_confidence": None,
        }
        metadata.append(info)
        logger.info(f"Saved IC{idx} to {output_path}")

    # Save metadata JSON for Rails seeding
    metadata_path = output_dir / "components.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Generated {len(metadata)} component images")
    logger.info(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
