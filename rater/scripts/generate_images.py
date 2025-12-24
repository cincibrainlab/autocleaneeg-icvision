#!/usr/bin/env python3
"""
Generate component images for the human rater web app.

This script:
1. Loads an ICA decomposition and raw EEG data
2. Generates strip images for each component
3. Exports metadata to JSON for Rails database seeding

Usage:
    python generate_images.py --raw data.fif --ica ica.fif --output public/components/
    python generate_images.py --raw data.set --output public/components/  # Auto-detects ICA from .set
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for icvision imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch

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


def plot_component_strip(
    ica: ICA,
    raw: mne.io.Raw,
    component_idx: int,
    output_path: Path,
    psd_fmax: float = 80.0,
) -> dict:
    """
    Generate a horizontal strip image for a single component.

    Layout: [Topography] [Time Series] [Power Spectrum]

    Returns metadata dict with component info.
    """
    # Get component data
    sources = ica.get_sources(raw).get_data()
    component_data = sources[component_idx, :]
    sfreq = raw.info["sfreq"]

    # Create figure with strip layout
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), gridspec_kw={"width_ratios": [1, 2, 1.5]})
    fig.patch.set_facecolor("white")

    # 1. Topography
    ax_topo = axes[0]
    try:
        ica.plot_components(picks=[component_idx], axes=ax_topo, show=False, colorbar=False)
        ax_topo.set_title(f"IC{component_idx}", fontsize=12, fontweight="bold")
    except Exception as e:
        ax_topo.text(0.5, 0.5, f"IC{component_idx}\n(topo unavailable)", ha="center", va="center", fontsize=10)
        ax_topo.set_xlim(0, 1)
        ax_topo.set_ylim(0, 1)
        ax_topo.axis("off")
        logger.warning(f"Could not plot topography for IC{component_idx}: {e}")

    # 2. Time series (show ~5 seconds centered in data)
    ax_ts = axes[1]
    n_samples = int(5 * sfreq)
    start_idx = max(0, len(component_data) // 2 - n_samples // 2)
    end_idx = min(len(component_data), start_idx + n_samples)
    time_segment = component_data[start_idx:end_idx]
    times = np.arange(len(time_segment)) / sfreq

    ax_ts.plot(times, time_segment, "b-", linewidth=0.5)
    ax_ts.set_xlabel("Time (s)", fontsize=9)
    ax_ts.set_ylabel("Amplitude", fontsize=9)
    ax_ts.set_title("Time Series", fontsize=10)
    ax_ts.set_xlim(times[0], times[-1])

    # Auto-scale y-axis, removing outliers
    q1, q99 = np.percentile(time_segment, [1, 99])
    margin = (q99 - q1) * 0.1
    ax_ts.set_ylim(q1 - margin, q99 + margin)

    # 3. Power spectrum
    ax_psd = axes[2]
    try:
        # Compute PSD
        nyquist = sfreq / 2
        fmax = min(psd_fmax, nyquist - 1)
        psds, freqs = psd_array_welch(
            component_data.reshape(1, -1),
            sfreq=sfreq,
            fmin=0.5,
            fmax=fmax,
            n_fft=int(sfreq * 2),
            n_overlap=int(sfreq),
            verbose=False,
        )
        psd = psds[0]

        # Plot in dB
        psd_db = 10 * np.log10(psd + 1e-20)
        ax_psd.plot(freqs, psd_db, "b-", linewidth=1)
        ax_psd.set_xlabel("Frequency (Hz)", fontsize=9)
        ax_psd.set_ylabel("Power (dB)", fontsize=9)
        ax_psd.set_title("Power Spectrum", fontsize=10)
        ax_psd.set_xlim(0.5, fmax)

        # Mark alpha band (8-12 Hz)
        ax_psd.axvspan(8, 12, alpha=0.2, color="green", label="Alpha")
        ax_psd.legend(loc="upper right", fontsize=8)

    except Exception as e:
        ax_psd.text(0.5, 0.5, f"PSD unavailable:\n{e}", ha="center", va="center", fontsize=8)
        ax_psd.set_xlim(0, 1)
        ax_psd.set_ylim(0, 1)
        logger.warning(f"Could not compute PSD for IC{component_idx}: {e}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Saved IC{component_idx} to {output_path}")

    return {
        "ic_index": component_idx,
        "image_path": f"/components/{output_path.name}",
    }


def main():
    parser = argparse.ArgumentParser(description="Generate component images for human rater app")
    parser.add_argument("--raw", required=True, help="Path to raw EEG data (.fif or .set)")
    parser.add_argument("--ica", help="Path to ICA file (.fif). Auto-detected for .set files.")
    parser.add_argument("--output", required=True, help="Output directory for images")
    parser.add_argument("--dataset", default="default", help="Dataset name for metadata")
    parser.add_argument("--psd-fmax", type=float, default=80.0, help="Max frequency for PSD")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    raw, ica = load_data(args.raw, args.ica)

    # Generate images for each component
    metadata = []
    n_components = ica.n_components_

    for idx in range(n_components):
        output_path = output_dir / f"ic_{idx:03d}.png"
        info = plot_component_strip(ica, raw, idx, output_path, psd_fmax=args.psd_fmax)
        info["dataset"] = args.dataset
        metadata.append(info)

    # Save metadata JSON for Rails seeding
    metadata_path = output_dir / "components.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Generated {len(metadata)} component images")
    logger.info(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
