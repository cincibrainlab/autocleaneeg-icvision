#!/usr/bin/env python3
"""
Batch generate component images for the human rater web app.

File Discovery (ICA-first approach):
    Looks for -ica.fif files first, then finds matching .set files.
    This ensures we only process data with valid ICA decompositions.

1. Paired files (preferred):
    ICA:  {prefix}-ica.fif  →  looks for  {prefix}_pre_ica_raw.set or {prefix}*.set
    Example: 0079_rest-ica.fif + 0079_rest_pre_ica_raw.set

2. Standalone .set files (fallback):
    If no -ica.fif files found, processes .set files with embedded ICA
    WARNING: Files with rejected components will have ICA matrix mismatch!

3. Clean data (no ICA):
    .set file without ICA → computes Infomax ICA on the fly

Usage:
    python generate_images_batch.py --input /path/to/data/ --output /path/to/output/
"""

import argparse
import json
import logging
import re
import sys
import warnings
from pathlib import Path

from filelock import FileLock

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


def find_paired_files(input_dir: Path) -> list[tuple[Path, Path | None, str]]:
    """
    Find -ica.fif files first, then locate matching .set files.
    Falls back to standalone .set files if no ICA files exist.
    
    Pattern: {prefix}-ica.fif → {prefix}_pre_ica_raw.set (or {prefix}*.set)
    """
    files = []
    ica_files = list(input_dir.glob("*-ica.fif"))
    
    if not ica_files:
        # Fallback: process standalone .set files
        for f in input_dir.glob("*.set"):
            files.append((f, None, f.stem))
        logger.info(f"Found {len(files)} standalone .set files (no -ica.fif)")
        return files
    
    logger.info(f"Found {len(ica_files)} ICA files")
    
    for ica_file in ica_files:
        prefix = ica_file.stem.replace("-ica", "")
        
        # Try specific patterns first, then glob fallback
        set_file = None
        for suffix in ["_pre_ica_raw.set", "_ica_clean_raw.set"]:
            if (input_dir / f"{prefix}{suffix}").exists():
                set_file = input_dir / f"{prefix}{suffix}"
                break
        
        if not set_file:
            matches = [f for f in input_dir.glob(f"{prefix}*.set")]
            set_file = next((f for f in matches if "pre" in f.stem.lower()), matches[0] if matches else None)
        
        if set_file:
            logger.info(f"Paired: {ica_file.name} <-> {set_file.name}")
            files.append((set_file, ica_file, prefix))
        else:
            logger.warning(f"No .set file for {ica_file.name}")
    
    return files


def load_data(raw_path: Path, ica_path: Path | None = None) -> tuple:
    """
    Load raw data (or epochs) and ICA object.
    
    Handles three cases:
    1. Paired .fif ICA file provided → load from .fif
    2. Embedded ICA in .set file → load embedded (check for mismatch)
    3. No ICA available → compute Infomax ICA on the fly
    
    Args:
        raw_path: Path to raw EEG data (.set file)
        ica_path: Path to ICA file (.fif), or None to try embedded/compute
        
    Returns:
        Tuple of (data, ica, has_mismatch, was_computed) where:
        - has_mismatch: True if embedded ICA has matrix inconsistency
        - was_computed: True if ICA was computed (not loaded)
    """
    data = None
    ica = None
    has_mismatch = False
    was_computed = False

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

    # Case 1: Load ICA from separate .fif file
    if ica_path is not None:
        ica = mne.preprocessing.read_ica(ica_path, verbose=False)
        logger.info(f"Loaded ICA from {ica_path.name}: {ica.n_components_} components")
    else:
        # Case 2: Try to load embedded ICA from .set file
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                ica = mne.preprocessing.read_ica_eeglab(raw_path)
                
                # Check for the mismatch warning
                for w in caught_warnings:
                    if "Mismatch between icawinv and icaweights" in str(w.message):
                        has_mismatch = True
                        break
            
            logger.info(f"Loaded embedded ICA from {raw_path.name}: {ica.n_components_} components")
        
        except Exception:
            # Case 3: No ICA in file - compute Infomax ICA
            logger.warning(f"\nNO ICA IN FILE: {raw_path.name} - computing Infomax ICA...")
            ica = compute_ica(data)
            was_computed = True
            logger.info(f"Computed ICA: {ica.n_components_} components")

    return data, ica, has_mismatch, was_computed


def compute_ica(data) -> mne.preprocessing.ICA:
    """
    Compute Infomax ICA on the data.
    
    Automatically estimates data rank to avoid unstable mixing matrices.
    
    Args:
        data: MNE Raw or Epochs object
        
    Returns:
        Fitted ICA object
    """
    # Check if data is high-pass filtered, apply 1 Hz filter if not
    highpass = data.info.get('highpass', 0)
    if highpass < 1.0:
        logger.info(f"Data highpass is {highpass} Hz - applying 1 Hz high-pass filter for ICA")
        data = data.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        logger.info("High-pass filter applied (1 Hz)")
    else:
        logger.info(f"Data already high-pass filtered at {highpass} Hz")

    # Create and fit ICA
    ica = mne.preprocessing.ICA(
        method='infomax',
        fit_params=dict(extended=True),  # Extended Infomax for sub-Gaussian sources
        verbose=False
    )
    
    # Fit ICA to data
    ica.fit(data, verbose=False)
    
    logger.info(f"ICA fitting complete: {ica.n_components_} components extracted")
    
    return ica


def plot_single_component_strip(ica_obj, component_idx, output_path, component_data, sfreq):
    """
    Plot a single component as a horizontal strip with 4 panels.

    This is the EXACT same plotting code from test_grid_classify.py
    to ensure visual consistency.

    Panels: Topography | Time Series (2.5s) | ERP Image | Power Spectrum (1-55Hz)
    
    Args:
        ica_obj: MNE ICA object
        component_idx: Index of the component to plot
        output_path: Path to save the output image
        component_data: Pre-computed component time series (1D numpy array)
        sfreq: Sampling frequency in Hz
        
    Returns:
        Path to the saved image
    """
    # Create figure with strip layout
    fig = plt.figure(figsize=(16, 2.5), dpi=150)
    gs = GridSpec(1, 4, figure=fig, wspace=0.08, left=0.02, right=0.98, top=0.92, bottom=0.08)

    ax_topo = fig.add_subplot(gs[0, 0])
    ax_ts = fig.add_subplot(gs[0, 1])
    ax_erp = fig.add_subplot(gs[0, 2])
    ax_psd = fig.add_subplot(gs[0, 3])
    label = f"IC{component_idx}"
    
    # Clear all axes labels/ticks upfront
    for ax in [ax_topo, ax_ts, ax_erp, ax_psd]:
        ax.set(xticks=[], yticks=[], xlabel="", ylabel="", title="")

    # 1. Topography
    try:
        ica_obj.plot_components(picks=component_idx, axes=ax_topo, ch_type="eeg",
            show=False, colorbar=False, cmap="jet", outlines="head", sensors=True, contours=6)
        ax_topo.set_title("")
        ax_topo.text(0.05, 0.95, label, transform=ax_topo.transAxes, fontsize=11, 
                     fontweight='bold', va='top', ha='left', color='white',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    except Exception as e:
        ax_topo.text(0.5, 0.5, label, ha="center", va="center", fontsize=12, fontweight='bold')
        logger.warning(f"Could not plot topography for IC{component_idx}: {e}")

    # 2. Time series (first 2.5s)
    try:
        max_samples = min(int(2.5 * sfreq), len(component_data))
        times_ms = (np.arange(max_samples) / sfreq) * 1000
        ax_ts.plot(times_ms, component_data[:max_samples], linewidth=0.5, color="dodgerblue")
        ax_ts.set_xlim(times_ms[0], times_ms[-1])
    except Exception as e:
        logger.warning(f"Could not plot time series for IC{component_idx}: {e}")

    # 3. ERP image
    try:
        comp_centered = component_data - np.mean(component_data)
        segment_len = max(1, int(1.5 * sfreq))
        n_segments = min(len(comp_centered) // segment_len, 100)

        if n_segments > 0:
            erp_data = comp_centered[:n_segments * segment_len].reshape(n_segments, segment_len)
            if n_segments >= 3:
                erp_data = uniform_filter1d(erp_data, size=3, axis=0, mode="nearest")
            max_val = np.max(np.abs(erp_data))
            clim = (2/3) * max_val if max_val > 1e-9 else 1.0
            ax_erp.imshow(erp_data, aspect="auto", cmap="jet", vmin=-clim, vmax=clim)
            ax_erp.invert_yaxis()
    except Exception as e:
        logger.warning(f"Could not plot ERP image for IC{component_idx}: {e}")

    # 4. PSD (1-55Hz)
    try:
        fmin, fmax = 1.0, min(55.0, sfreq / 2.0 - 0.5)
        n_fft = max(256, min(int(sfreq * 2), len(component_data))) if len(component_data) >= 256 else len(component_data)
        psds, freqs = psd_array_welch(component_data, sfreq=sfreq, fmin=fmin, fmax=fmax,
                                       n_fft=n_fft, n_overlap=n_fft // 2, verbose=False)
        ax_psd.plot(freqs, 10 * np.log10(np.maximum(psds, 1e-20)), color="red", linewidth=0.8)
        ax_psd.set_xlim(freqs[0], freqs[-1])
    except Exception as e:
        logger.warning(f"Could not plot PSD for IC{component_idx}: {e}")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)

    return output_path


def process_single_file(
    raw_path: Path, 
    ica_path: Path | None, 
    output_dir: Path, 
    dataset_name: str, 
    image_format: str = "webp",
    skip_mismatched: bool = False,
    skip_existing: bool = True
) -> tuple[list[dict], bool, bool, bool]:
    """
    Process a single .set file and generate component images.
    
    Args:
        raw_path: Path to raw EEG data (.set file)
        ica_path: Path to ICA file (.fif), or None to use embedded ICA
        output_dir: Base output directory
        dataset_name: Dataset identifier for organizing output
        image_format: Output image format ("png" or "webp")
        skip_mismatched: If True, skip files with ICA matrix mismatch
        skip_existing: If True, skip components that already have images
        
    Returns:
        Tuple of (metadata_list, was_skipped, had_mismatch, was_computed):
        - metadata_list: List of component metadata dicts
        - was_skipped: True if file was skipped due to mismatch
        - had_mismatch: True if ICA matrix mismatch was detected
        - was_computed: True if ICA was computed on the fly
    """
    # Load data first to check for mismatches
    try:
        data, ica, has_mismatch, was_computed = load_data(raw_path, ica_path)
    except Exception as e:
        logger.error(f"Failed to load {dataset_name}: {e}")
        return [], False, False, False
    
    # Handle ICA matrix mismatch (likely from component rejection)
    if has_mismatch:
        logger.warning(f"\nICA MISMATCH: {dataset_name} - likely has rejected components")
        logger.warning("Topographies may NOT match time series! Use pre-rejection data for accuracy.")
        if skip_mismatched:
            logger.warning(f"SKIPPING (--skip-mismatched)")
            return [], True, True, False
        logger.warning("Proceeding anyway...")
    
    # Create subdirectory for this file's components
    file_output_dir = output_dir / dataset_name
    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-compute ALL component sources at once (major optimization)
    # This avoids calling get_sources() for each component individually
    logger.info("Extracting ICA sources...")
    sources = ica.get_sources(data)
    sfreq = sources.info["sfreq"]
    
    # Get all source data in one call
    all_source_data = sources.get_data()
    if all_source_data.ndim == 3:
        # Epochs: shape (n_epochs, n_components, n_times) -> concatenate epochs
        all_source_data = all_source_data.transpose(1, 0, 2).reshape(all_source_data.shape[1], -1)
    
    # Generate images for each component
    metadata = []
    n_components = ica.n_components_
    subject_id = re.split(r'[_-]', dataset_name)[0]
    
    def make_metadata(idx):
        return {"ic_index": idx, "image_path": f"/components/{dataset_name}/ic_{idx:03d}.{image_format}",
                "dataset": dataset_name, "subject_id": subject_id, "model_label": None, "model_confidence": None}
    
    logger.info(f"Generating {n_components} component images...")
    skipped_count = 0
    for idx in range(n_components):
        output_path = file_output_dir / f"ic_{idx:03d}.{image_format}"
        
        # Skip image generation if it already exists
        if skip_existing and output_path.exists():
            metadata.append(make_metadata(idx))  # Still track metadata
            skipped_count += 1
            continue
        
        try:
            plot_single_component_strip(ica, idx, output_path, all_source_data[idx], sfreq)
            metadata.append(make_metadata(idx))
            if (idx + 1) % 10 == 0 or idx == n_components - 1:
                logger.info(f"  {dataset_name}: {idx + 1}/{n_components} done")
        except Exception as e:
            logger.error(f"Failed IC{idx} for {dataset_name}: {e}")
    
    if skipped_count > 0:
        logger.info(f"  Skipped {skipped_count} existing images")
    
    # Save metadata JSON for this file
    metadata_path = file_output_dir / "components.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved {len(metadata)} component images for {dataset_name}")
    
    return metadata, False, has_mismatch, was_computed


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
        required=True,
        help="Output directory for generated component images"
    )
    parser.add_argument(
        "--format", "-f",
        default="webp", 
        choices=["png", "webp"], 
        help="Image format (default: webp)"
    )
    parser.add_argument(
        "--skip-mismatched",
        action="store_true",
        help="Skip files with ICA matrix mismatch (likely post-rejection data)"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate all images even if they already exist (default: skip existing)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
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
    
    # Find all .set files (with or without paired ICA files)
    files = find_paired_files(input_dir)
    
    if not files:
        logger.error("No .set files found in the input directory.")
        sys.exit(1)
    
    logger.info(f"Found {len(files)} file(s) to process")
    
    # Setup for incremental metadata writes (with file locking for concurrency)
    combined_metadata_path = output_dir / "all_components.json"
    lock_path = output_dir / "all_components.json.lock"
    metadata_lock = FileLock(lock_path)
    
    if combined_metadata_path.exists():
        logger.info(f"Appending to existing all_components.json")
    
    # Process each file
    successful = 0
    failed = 0
    skipped = 0
    computed = 0
    mismatched_files = []
    computed_files = []
    
    for raw_path, ica_path, dataset_name in files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {dataset_name}")
        logger.info(f"  Raw: {raw_path.name}")
        logger.info(f"  ICA: {ica_path.name if ica_path else 'embedded or computed'}")
        logger.info(f"{'='*60}")
        
        try:
            metadata, was_skipped, had_mismatch, was_computed = process_single_file(
                raw_path, ica_path, output_dir, dataset_name, args.format,
                skip_mismatched=args.skip_mismatched,
                skip_existing=not args.regenerate
            )
            if had_mismatch:
                mismatched_files.append(dataset_name)
            if was_computed:
                computed_files.append(dataset_name)
                computed += 1
            
            if was_skipped:
                skipped += 1
            elif metadata:
                # Thread-safe incremental write: lock, re-read, merge (dedupe), write
                with metadata_lock:
                    if combined_metadata_path.exists():
                        with open(combined_metadata_path, "r", encoding="utf-8") as f:
                            all_metadata = json.load(f)
                    else:
                        all_metadata = []
                    # Deduplicate by (dataset, ic_index) - new entries override existing
                    existing_keys = {(m["dataset"], m["ic_index"]) for m in all_metadata}
                    for m in metadata:
                        key = (m["dataset"], m["ic_index"])
                        if key not in existing_keys:
                            all_metadata.append(m)
                            existing_keys.add(key)
                    with open(combined_metadata_path, "w", encoding="utf-8") as f:
                        json.dump(all_metadata, f, indent=2)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}")
            failed += 1
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Successful: {successful}/{len(files)} files")
    logger.info(f"Failed: {failed}/{len(files)} files")
    if skipped > 0:
        logger.info(f"Skipped (ICA mismatch): {skipped}/{len(files)} files")
    if computed > 0:
        logger.info(f"ICA computed on-the-fly: {computed}/{len(files)} files")
    # Get final component count from file
    if combined_metadata_path.exists():
        with open(combined_metadata_path, "r", encoding="utf-8") as f:
            total_components = len(json.load(f))
    else:
        total_components = 0
    logger.info(f"Total components: {total_components}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Combined metadata: {combined_metadata_path}")
    
    # List files where ICA was computed
    if computed_files:
        logger.info("")
        logger.info("=" * 60)
        logger.info("FILES WHERE ICA WAS COMPUTED (no existing ICA found):")
        logger.info("=" * 60)
        for name in computed_files:
            logger.info(f"  - {name}")
        logger.info("=" * 60)
    
    if mismatched_files and not args.skip_mismatched:
        logger.warning(f"\nMISMATCHED FILES: {', '.join(mismatched_files)}")
        logger.warning("Use --skip-mismatched to exclude these.")


if __name__ == "__main__":
    main()

