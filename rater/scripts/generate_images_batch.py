#!/usr/bin/env python3
"""
Batch generate component images for the human rater web app.

This script processes a folder of EEG files, automatically pairing raw .set files
with their corresponding ICA .fif files based on naming patterns.

File Pairing Pattern:
    Raw:  {subject_id}_rest_ica_clean_raw.set
    ICA:  {subject_id}_rest-ica.fif

The script extracts the common prefix (e.g., "0079_rest") and pairs files accordingly.

Smart Path Detection:
    - If run from within the rater/ directory tree, auto-detects public/components/
    - Otherwise, requires --output to be specified

Usage:
    # From rater/scripts/ directory (auto-detects output)
    python generate_images_batch.py --input .
    
    # From anywhere with explicit output
    python generate_images_batch.py --input /path/to/data/ --output /path/to/output/
"""

import argparse
import json
import logging
import os
import re
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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_rater_root() -> Path | None:
    """
    Find the rater app root directory by checking common locations.
    
    Searches in this order:
    1. Current working directory (if it's the rater root)
    2. Parent directories of CWD (up to 5 levels)
    3. Parent directories of script location (up to 5 levels)
    
    Returns:
        Path to rater root if found, None otherwise
    """
    def is_rater_root(path: Path) -> bool:
        """Check if a directory is the rater Rails app root."""
        # Must have Gemfile and public/ directory
        return (path / "Gemfile").exists() and (path / "public").is_dir()
    
    # Check from current working directory
    cwd = Path.cwd().resolve()
    for i in range(6):  # Check up to 5 parent levels
        check_path = cwd
        for _ in range(i):
            check_path = check_path.parent
        if is_rater_root(check_path):
            return check_path
    
    # Check from script location
    script_dir = Path(__file__).resolve().parent
    for i in range(6):
        check_path = script_dir
        for _ in range(i):
            check_path = check_path.parent
        if is_rater_root(check_path):
            return check_path
    
    return None


def get_default_output_dir() -> Path | None:
    """
    Determine the default output directory based on current location.
    
    Returns:
        Path to public/components/ if rater root is found, None otherwise
    """
    rater_root = find_rater_root()
    if rater_root:
        return rater_root / "public" / "components"
    return None


def find_paired_files(input_dir: Path) -> list[tuple[Path, Path, str]]:
    """
    Find paired raw and ICA files in the input directory.
    
    Looks for files matching these patterns:
        Raw:  {prefix}_ica_clean_raw.set
        ICA:  {prefix}-ica.fif
    
    Args:
        input_dir: Directory containing raw and ICA files
        
    Returns:
        List of tuples: (raw_path, ica_path, subject_id)
    """
    pairs = []
    
    # Find all .set files matching the pattern *_ica_clean_raw.set
    set_files = list(input_dir.glob("*_ica_clean_raw.set"))
    
    if not set_files:
        logger.warning(f"No *_ica_clean_raw.set files found in {input_dir}")
        return pairs
    
    logger.info(f"Found {len(set_files)} .set files in {input_dir}")
    
    for set_file in set_files:
        # Extract the prefix (everything before _ica_clean_raw.set)
        # Example: 0079_rest_ica_clean_raw.set -> prefix = 0079_rest
        filename = set_file.stem  # Remove .set extension
        match = re.match(r"(.+)_ica_clean_raw$", filename)
        
        if not match:
            logger.warning(f"File {set_file.name} doesn't match expected pattern, skipping")
            continue
            
        prefix = match.group(1)
        
        # Look for the corresponding ICA file: {prefix}-ica.fif
        ica_file = input_dir / f"{prefix}-ica.fif"
        
        if ica_file.exists():
            # Extract subject ID (first part before underscore)
            subject_id = prefix.split("_")[0]
            pairs.append((set_file, ica_file, prefix))
            logger.info(f"Paired: {set_file.name} <-> {ica_file.name}")
        else:
            logger.warning(f"No matching ICA file found for {set_file.name}")
            logger.warning(f"  Expected: {ica_file.name}")
    
    return pairs


def load_data(raw_path: Path, ica_path: Path) -> tuple:
    """
    Load raw data (or epochs) and ICA object.
    
    Args:
        raw_path: Path to raw EEG data (.set file)
        ica_path: Path to ICA file (.fif file)
        
    Returns:
        Tuple of (data, ica) objects
    """
    data = None
    ica = None

    # Load raw data from .set file
    try:
        data = mne.io.read_raw_eeglab(raw_path, preload=True, verbose=False)
        logger.info(f"Loaded raw data: {len(data.ch_names)} channels, {data.n_times / data.info['sfreq']:.1f}s")
    except TypeError as e:
        if "trials" in str(e):
            # This is epoched data, load as epochs
            data = mne.io.read_epochs_eeglab(raw_path, verbose=False)
            logger.info(f"Loaded epoched data: {len(data)} epochs, {len(data.ch_names)} channels")
        else:
            raise

    # Load ICA from .fif file
    ica = mne.preprocessing.read_ica(ica_path, verbose=False)
    logger.info(f"Loaded ICA from {ica_path.name}: {ica.n_components_} components")

    return data, ica


def plot_single_component_strip(ica_obj, data_obj, component_idx, output_path):
    """
    Plot a single component as a horizontal strip with 4 panels.

    This is the EXACT same plotting code from test_grid_classify.py
    to ensure visual consistency.

    Panels: Topography | Time Series (2.5s) | ERP Image | Power Spectrum (1-55Hz)
    
    Args:
        ica_obj: MNE ICA object
        data_obj: MNE Raw or Epochs object
        component_idx: Index of the component to plot
        output_path: Path to save the output image
        
    Returns:
        Path to the saved image
    """
    # Get component data - handle both Raw and Epochs
    sources = ica_obj.get_sources(data_obj)
    sfreq = sources.info["sfreq"]

    # Handle Epochs vs Raw data
    if hasattr(sources, 'get_data') and callable(sources.get_data):
        source_data = sources.get_data(picks=[component_idx])
        if source_data.ndim == 3:
            # Epochs: shape is (n_epochs, n_components, n_times) -> concatenate epochs
            component_data = source_data[:, 0, :].flatten()
        else:
            # Raw: shape is (n_components, n_times)
            component_data = source_data[0]
    else:
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


def process_single_file(raw_path: Path, ica_path: Path, output_dir: Path, 
                        prefix: str, image_format: str = "png") -> list[dict]:
    """
    Process a single raw/ICA pair and generate component images.
    
    Args:
        raw_path: Path to raw EEG data
        ica_path: Path to ICA file
        output_dir: Base output directory
        prefix: File prefix (e.g., "0079_rest")
        image_format: Output image format ("png" or "webp")
        
    Returns:
        List of metadata dictionaries for each component
    """
    # Create subdirectory for this file's components
    file_output_dir = output_dir / prefix
    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    try:
        data, ica = load_data(raw_path, ica_path)
    except Exception as e:
        logger.error(f"Failed to load {prefix}: {e}")
        return []
    
    # Generate images for each component
    metadata = []
    n_components = ica.n_components_
    
    logger.info(f"Processing {prefix}: {n_components} components")
    
    for idx in range(n_components):
        output_path = file_output_dir / f"ic_{idx:03d}.{image_format}"
        
        try:
            plot_single_component_strip(ica, data, idx, output_path)
            
            info = {
                "ic_index": idx,
                "image_path": f"/components/{prefix}/ic_{idx:03d}.{image_format}",
                "dataset": prefix,
                "subject_id": prefix.split("_")[0],
                "model_label": None,
                "model_confidence": None,
            }
            metadata.append(info)
            
            # Log progress every 10 components
            if (idx + 1) % 10 == 0 or idx == n_components - 1:
                logger.info(f"  {prefix}: Processed {idx + 1}/{n_components} components")
                
        except Exception as e:
            logger.error(f"Failed to generate IC{idx} for {prefix}: {e}")
    
    # Save metadata JSON for this file
    metadata_path = file_output_dir / "components.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved {len(metadata)} component images for {prefix}")
    
    return metadata


def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch generate component images for human rater app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
File Pairing Pattern:
    Raw:  {subject_id}_rest_ica_clean_raw.set
    ICA:  {subject_id}_rest-ica.fif

Example:
    0079_rest_ica_clean_raw.set  <-->  0079_rest-ica.fif
    0080_rest_ica_clean_raw.set  <-->  0080_rest-ica.fif
        """
    )
    parser.add_argument(
        "--input", "-i", 
        required=True, 
        help="Input directory containing raw .set and ICA .fif files"
    )
    parser.add_argument(
        "--output", "-o", 
        default=None,
        help="Output directory (auto-detects public/components/ if in rater directory)"
    )
    parser.add_argument(
        "--format", "-f",
        default="png", 
        choices=["png", "webp"], 
        help="Image format (default: png)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    
    # Smart output directory detection
    if args.output is None:
        output_dir = get_default_output_dir()
        if output_dir is None:
            logger.error("Could not auto-detect output directory.")
            logger.error("Please run from within the rater/ directory, or specify --output")
            logger.error("")
            logger.error("Example:")
            logger.error("  cd rater/scripts")
            logger.error("  python generate_images_batch.py --input .")
            logger.error("")
            logger.error("Or specify output explicitly:")
            logger.error("  python generate_images_batch.py --input . --output /path/to/output/")
            sys.exit(1)
        
        rater_root = find_rater_root()
        logger.info(f"Detected rater app at: {rater_root}")
        logger.info(f"Output directory: {output_dir}")
    else:
        output_dir = Path(args.output).resolve()
    
    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Find all paired files
    pairs = find_paired_files(input_dir)
    
    if not pairs:
        logger.error("No valid file pairs found. Check the input directory and file naming.")
        sys.exit(1)
    
    logger.info(f"Found {len(pairs)} file pair(s) to process")
    
    # Process each pair
    all_metadata = []
    successful = 0
    failed = 0
    
    for raw_path, ica_path, prefix in pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {prefix}")
        logger.info(f"  Raw: {raw_path.name}")
        logger.info(f"  ICA: {ica_path.name}")
        logger.info(f"{'='*60}")
        
        try:
            metadata = process_single_file(
                raw_path, ica_path, output_dir, prefix, args.format
            )
            if metadata:
                all_metadata.extend(metadata)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Failed to process {prefix}: {e}")
            failed += 1
    
    # Save combined metadata for all files
    combined_metadata_path = output_dir / "all_components.json"
    with open(combined_metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Successful: {successful}/{len(pairs)} files")
    logger.info(f"Failed: {failed}/{len(pairs)} files")
    logger.info(f"Total components: {len(all_metadata)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Combined metadata: {combined_metadata_path}")


if __name__ == "__main__":
    main()

