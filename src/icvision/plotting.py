"""
Plotting functions for ICVision.

This module contains functions for visualizing ICA components and saving
ICA data, adapted from the original ica.py functionality.
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.gridspec import GridSpec
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch
from scipy.ndimage import uniform_filter1d

# Set up logging for the module
logger = logging.getLogger("icvision.plotting")


def plot_component_for_classification(
    ica_obj: ICA,
    raw_obj: mne.io.Raw,
    component_idx: int,
    output_dir: Path,
    # Parameters for PDF report generation (optional, not used for API image)
    classification_label: Optional[str] = None,
    classification_confidence: Optional[float] = None,
    classification_reason: Optional[str] = None,
    return_fig_object: bool = False,
    source_filename: Optional[str] = None,
) -> Union[Path, plt.Figure, None]:
    """
    Creates a standardized plot for an ICA component.

    This plot is used for OpenAI Vision API classification and can also be
    included in PDF reports with classification details.
    The layout includes: Topography, Scrolling IC Activity, Continuous Data (ERP image),
    and Power Spectrum.

    Args:
        ica_obj: The MNE ICA object.
        raw_obj: The MNE Raw object used for ICA.
        component_idx: Index of the component to plot.
        output_dir: Directory to save the plot (if not returning a figure object).
        classification_label: Vision API label (for PDF report).
        classification_confidence: Vision API confidence (for PDF report).
        classification_reason: Vision API reason (for PDF report).
        return_fig_object: If True, returns matplotlib Figure object instead of saving.

    Returns:
        Path to saved image file (if return_fig_object is False).
        matplotlib Figure object (if return_fig_object is True).
        None on failure.

    Raises:
        ValueError: If output_dir is None and return_fig_object is False.
    """
    # Ensure non-interactive backend for scripts/batch processing
    matplotlib.use("Agg", force=True)

    # Clear any existing plots to prevent interference
    plt.close("all")

    fig_height = 9.5
    gridspec_bottom = 0.05

    # Adjust figure height and bottom margin if classification reason is provided (for PDF)
    if return_fig_object and classification_reason:
        fig_height = 11  # Increased height for reasoning text
        gridspec_bottom = 0.18

    fig = plt.figure(figsize=(12, fig_height), dpi=120)
    main_plot_title_text = f"ICA Component IC{component_idx} Analysis"
    gridspec_top = 0.95
    suptitle_y_pos = 0.98

    # Adjust top margin and title position if classification label is present (for PDF)
    if return_fig_object and classification_label is not None:
        gridspec_top = 0.90
        suptitle_y_pos = 0.96

    # Define GridSpec for plot layout
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[0.915, 0.572, 2.213],  # Adjusted for general look
        width_ratios=[0.9, 1],
        hspace=0.7,
        wspace=0.35,
        left=0.05,
        right=0.95,
        top=gridspec_top,
        bottom=gridspec_bottom,
    )

    # Add subplots to the grid
    ax_topo = fig.add_subplot(gs[0:2, 0])
    ax_cont_data = fig.add_subplot(gs[2, 0])
    ax_ts_scroll = fig.add_subplot(gs[0, 1])
    ax_psd = fig.add_subplot(gs[2, 1])

    try:
        # Get ICA sources with error handling
        sources = ica_obj.get_sources(raw_obj)
        sfreq = sources.info["sfreq"]
        component_data_array = sources.get_data(picks=[component_idx])[0]

        # Validate that we got valid data
        if len(component_data_array) == 0:
            logger.error("No data available for IC%d", component_idx)
            plt.close(fig)
            return None

    except Exception as e:
        logger.error("Failed to get ICA sources for IC%d: %s", component_idx, e)
        plt.close(fig)
        return None

    # 1. Topography Plot
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
        ax_topo.set_title(f"IC{component_idx} Topography", fontsize=12, loc="center")
        ax_topo.set_xlabel("")
        ax_topo.set_ylabel("")
        ax_topo.set_xticks([])
        ax_topo.set_yticks([])
    except Exception as e:
        logger.error("Error plotting topography for IC%d: %s", component_idx, e)
        ax_topo.text(0.5, 0.5, "Topography plot failed", ha="center", va="center")

    # 2. Scrolling IC Activity (Time Series)
    try:
        duration_segment_ts = 2.5  # seconds
        max_samples_ts = min(int(duration_segment_ts * sfreq), len(component_data_array))
        times_ts_ms = (np.arange(max_samples_ts) / sfreq) * 1000  # convert to ms

        ax_ts_scroll.plot(
            times_ts_ms,
            component_data_array[:max_samples_ts],
            linewidth=0.8,
            color="dodgerblue",
        )
        ax_ts_scroll.set_title("Scrolling IC Activity (First 2.5s)", fontsize=10)
        ax_ts_scroll.set_xlabel("Time (ms)", fontsize=9)
        ax_ts_scroll.set_ylabel("Amplitude (a.u.)", fontsize=9)
        if max_samples_ts > 0 and times_ts_ms.size > 0:
            ax_ts_scroll.set_xlim(times_ts_ms[0], times_ts_ms[-1])
        ax_ts_scroll.grid(True, linestyle=":", alpha=0.6)
        ax_ts_scroll.tick_params(axis="both", which="major", labelsize=8)
    except Exception as e:
        logger.error("Error plotting scrolling IC activity for IC%d: %s", component_idx, e)
        ax_ts_scroll.text(0.5, 0.5, "Time series plot failed", ha="center", va="center")

    # 3. Continuous Data (EEGLAB-style ERP image)
    try:
        comp_data_offset_corrected = component_data_array - np.mean(component_data_array)
        target_segment_duration_s = 1.5
        target_max_segments = 200  # Limit segments for manageable plot
        segment_len_samples_cd = int(target_segment_duration_s * sfreq)
        if segment_len_samples_cd == 0:
            segment_len_samples_cd = 1  # Avoid division by zero

        available_samples = comp_data_offset_corrected.shape[0]
        max_total_samples_for_plot = int(target_max_segments * segment_len_samples_cd)
        samples_to_use = min(available_samples, max_total_samples_for_plot)

        n_segments_cd = 0
        current_segment_len = 1

        if segment_len_samples_cd > 0 and samples_to_use >= segment_len_samples_cd:
            n_segments_cd = math.floor(samples_to_use / segment_len_samples_cd)

        if n_segments_cd > 0:
            current_segment_len = segment_len_samples_cd
            final_samples_for_reshape = n_segments_cd * current_segment_len
            erp_image_data = comp_data_offset_corrected[:final_samples_for_reshape].reshape(
                n_segments_cd, current_segment_len
            )
        elif samples_to_use > 0:  # Handle less than one segment of data
            n_segments_cd = 1
            current_segment_len = samples_to_use
            erp_image_data = comp_data_offset_corrected[:current_segment_len].reshape(1, current_segment_len)
        else:  # No data to plot
            erp_image_data = np.zeros((1, 1))
            current_segment_len = 1  # For placeholder ticks

        # Apply smoothing if enough segments
        if n_segments_cd >= 3 and erp_image_data.shape[0] >= 3:
            erp_image_smoothed = uniform_filter1d(erp_image_data, size=3, axis=0, mode="nearest")
        else:
            erp_image_smoothed = erp_image_data

        # Determine color limits
        if erp_image_smoothed.size > 0:
            max_abs_val: float = np.max(np.abs(erp_image_smoothed))
            clim_val = (2 / 3) * max_abs_val if max_abs_val > 1e-9 else 1.0
        else:
            clim_val = 1.0
        clim_val = max(clim_val, 1e-9)  # Avoid clim_val being zero
        vmin_cd, vmax_cd = -clim_val, clim_val

        im = ax_cont_data.imshow(
            erp_image_smoothed,
            aspect="auto",
            cmap="jet",
            interpolation="nearest",
            vmin=vmin_cd,
            vmax=vmax_cd,
        )

        ax_cont_data.set_title(f"Continuous Data Segments (Max {target_max_segments})", fontsize=10)
        ax_cont_data.set_xlabel("Time (ms)", fontsize=9)
        if current_segment_len > 1:
            num_xticks = min(4, current_segment_len)
            xtick_positions_samples = np.linspace(0, current_segment_len - 1, num_xticks)
            xtick_labels_ms = (xtick_positions_samples / sfreq * 1000).astype(int)
            ax_cont_data.set_xticks(xtick_positions_samples)
            ax_cont_data.set_xticklabels(xtick_labels_ms)
        else:
            ax_cont_data.set_xticks([])

        ax_cont_data.set_ylabel("Trials (Segments)", fontsize=9)
        if n_segments_cd > 1:
            num_yticks = min(5, n_segments_cd)
            ytick_positions = np.linspace(0, n_segments_cd - 1, num_yticks).astype(int)
            ax_cont_data.set_yticks(ytick_positions)
            ax_cont_data.set_yticklabels(ytick_positions)
        elif n_segments_cd == 1:
            ax_cont_data.set_yticks([0])
            ax_cont_data.set_yticklabels(["0"])
        else:
            ax_cont_data.set_yticks([])

        if n_segments_cd > 0:
            ax_cont_data.invert_yaxis()

        cbar_cont = fig.colorbar(im, ax=ax_cont_data, orientation="vertical", fraction=0.046, pad=0.1)
        cbar_cont.set_label("Activation (a.u.)", fontsize=8)
        cbar_cont.ax.tick_params(labelsize=7)
    except Exception as e_cont:
        logger.error("Error plotting continuous data for IC%d: %s", component_idx, e_cont)
        ax_cont_data.text(0.5, 0.5, "Continuous data plot failed", ha="center", va="center")

    # 4. IC Activity Power Spectrum
    try:
        fmin_psd = 1.0
        fmax_psd = min(80.0, sfreq / 2.0 - 0.51)  # Cap at 80Hz or Nyquist
        n_fft_psd = int(sfreq * 2.0)  # 2-second window
        if n_fft_psd > len(component_data_array):
            n_fft_psd = len(component_data_array)
        # Ensure n_fft is at least 256 if data is long enough
        n_fft_psd = max(
            n_fft_psd,
            (
                256
                if len(component_data_array) >= 256
                else (len(component_data_array) if len(component_data_array) > 0 else 1)
            ),
        )

        if n_fft_psd == 0 or fmax_psd <= fmin_psd:
            raise ValueError(
                "Cannot compute PSD for IC%d: Invalid params (n_fft=%d, fmin=%s, fmax=%s)"
                % (component_idx, n_fft_psd, fmin_psd, fmax_psd)
            )

        psds, freqs = psd_array_welch(
            component_data_array,
            sfreq=sfreq,
            fmin=fmin_psd,
            fmax=fmax_psd,
            n_fft=n_fft_psd,
            n_overlap=int(n_fft_psd * 0.5),
            verbose=False,
            average="mean",
        )
        if psds.size == 0:
            raise ValueError("PSD computation returned empty array.")

        psds_db = 10 * np.log10(np.maximum(psds, 1e-20))  # Avoid log(0)

        ax_psd.plot(freqs, psds_db, color="red", linewidth=1.2)
        ax_psd.set_title(f"IC{component_idx} Power Spectrum (1-80Hz)", fontsize=10)
        ax_psd.set_xlabel("Frequency (Hz)", fontsize=9)
        ax_psd.set_ylabel("Power (dB)", fontsize=9)
        if len(freqs) > 0:
            ax_psd.set_xlim(freqs[0], freqs[-1])
        ax_psd.grid(True, linestyle="--", alpha=0.5)
        ax_psd.tick_params(axis="both", which="major", labelsize=8)
    except Exception as e_psd:
        logger.error("Error plotting PSD for IC%d: %s", component_idx, e_psd)
        ax_psd.text(0.5, 0.5, "PSD plot failed", ha="center", va="center")

    if return_fig_object and classification_label is not None and classification_confidence is not None:
        # Create dual-title layout: IC Component title on left, Classification on right
        from .config import (  # Local import to avoid circular dependency if any
            COLOR_MAP,
        )

        # Left-justified main title
        fig.text(
            0.05,
            suptitle_y_pos,
            main_plot_title_text,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            transform=fig.transFigure,
        )

        # Right-justified classification subtitle
        subtitle_color = COLOR_MAP.get(classification_label.lower(), "black")
        classification_subtitle = (
            f"Vision Classification: {str(classification_label).title()} "
            f"(Confidence: {classification_confidence:.2f})"
        )
        fig.text(
            0.95,
            suptitle_y_pos,
            classification_subtitle,
            ha="right",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=subtitle_color,
            transform=fig.transFigure,
        )
    else:
        # Standard centered title when no classification
        fig.suptitle(main_plot_title_text, fontsize=14, y=suptitle_y_pos)

    if return_fig_object:
        # Add reasoning if provided (for PDF report)
        if classification_reason:
            reasoning_text = f"Rationale: {classification_reason}"
            reason_text_y = gridspec_bottom - 0.03

            fig.text(
                0.05,
                reason_text_y,
                reasoning_text,
                ha="left",
                va="top",
                fontsize=8,
                wrap=True,
                transform=fig.transFigure,
                bbox=dict(boxstyle="round,pad=0.4", fc="aliceblue", alpha=0.75, ec="lightgrey"),
            )

        # Add footer with source filename if provided (for PDF report)
        if source_filename:
            fig.text(
                0.5,
                0.01,
                f"Autoclean ICVision | https://github.com/cincibrainlab | Source: {source_filename}",
                ha="center",
                va="bottom",
                fontsize=8,
                style="italic",
                color="gray",
                transform=fig.transFigure,
            )

        # No additional layout adjustment needed - GridSpec already handles proper spacing
        return fig
    else:
        # Save as .webp for OpenAI API (no classification text on image itself)
        if output_dir is None:
            raise ValueError("output_dir must be provided if not returning figure object.")

        filename = f"component_IC{component_idx}_vision_analysis.webp"
        filepath = output_dir / filename
        try:
            # Ensure tight layout for API image
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.93, hspace=0.7, wspace=0.35)
            plt.savefig(filepath, format="webp", bbox_inches="tight", pad_inches=0.1)
            logger.debug("Saved component plot for API to %s", filepath)
        except Exception as e_save:
            logger.error("Error saving API figure for IC%d: %s", component_idx, e_save)
            plt.close(fig)
            return None
        finally:
            plt.close(fig)  # Ensure figure is closed
        return filepath


def plot_components_batch(
    ica_obj: mne.preprocessing.ICA,
    raw_obj: mne.io.Raw,
    component_indices: List[int],
    output_dir: Path,
    batch_size: int = 1,
) -> Dict[int, Optional[Path]]:
    """
    Generate component plots with improved error handling and memory management.

    This function processes components sequentially with proper cleanup to avoid
    matplotlib threading issues while maintaining reasonable performance.

    Args:
        ica_obj: The MNE ICA object.
        raw_obj: The MNE Raw object used for ICA.
        component_indices: List of component indices to plot.
        output_dir: Directory to save the component images.
        batch_size: Number of components to process before cleanup (default: 1).

    Returns:
        Dictionary mapping component_idx to image path (or None if plotting failed).

    Example:
        >>> indices = list(range(ica_obj.n_components_))
        >>> results = plot_components_batch(ica, raw, indices, output_dir)
        >>> successful_plots = {k: v for k, v in results.items() if v is not None}
        >>> print(f"Successfully plotted {len(successful_plots)} components")
    """
    if not component_indices:
        logger.info("No components to plot.")
        return {}

    # Ensure matplotlib backend is set properly
    matplotlib.use("Agg", force=True)

    logger.debug(
        "Plotting %d components sequentially with enhanced error handling",
        len(component_indices),
    )

    results_dict = {}
    completed_count = 0

    for i, component_idx in enumerate(component_indices):
        try:
            # Clear any existing plots to prevent memory issues
            plt.close("all")

            # Plot the component with enhanced error handling
            image_path = plot_component_for_classification(
                ica_obj, raw_obj, component_idx, output_dir, return_fig_object=False
            )

            results_dict[component_idx] = image_path
            completed_count += 1

            # Log progress every 10% or at completion
            if completed_count % max(1, len(component_indices) // 10) == 0 or completed_count == len(component_indices):
                logger.info(
                    "Plotting progress: %d/%d components completed",
                    completed_count,
                    len(component_indices),
                )

            # Periodic cleanup to prevent memory accumulation
            if (i + 1) % batch_size == 0:
                plt.close("all")
                # Force garbage collection if needed
                import gc

                gc.collect()

        except Exception as e:
            logger.warning("Failed to plot component IC%d: %s", component_idx, e)
            results_dict[component_idx] = None
            # Ensure cleanup even on failure
            plt.close("all")

    # Final cleanup
    plt.close("all")

    successful_plots = sum(1 for path in results_dict.values() if path is not None)
    logger.info(
        "Plotting completed: %d/%d components plotted successfully",
        successful_plots,
        len(component_indices),
    )

    return results_dict


def save_ica_data(
    ica_obj: mne.preprocessing.ICA,
    output_dir: Path,
    input_basename: Optional[str] = None,
    filename_prefix: Optional[str] = None,
) -> Path:
    """
    Save the updated MNE ICA object to a .fif file.

    Args:
        ica_obj: The MNE ICA object to save.
        output_dir: Directory to save the file.
        input_basename: Basename from input file for default naming.
        filename_prefix: Custom prefix for the output filename. If None, uses basename_icvis_classified.

    Returns:
        Path to the saved ICA file.
    """
    # Set default filename prefix based on basename
    if filename_prefix is None:
        if input_basename is None:
            filename_prefix = "icvision_classified"
        else:
            filename_prefix = f"{input_basename}_icvis_classified"

    output_filename = f"{filename_prefix}_ica.fif"
    output_path = output_dir / output_filename

    try:
        ica_obj.save(output_path, overwrite=True)
        logger.info("Updated ICA object saved to: %s", output_path)
        return output_path
    except Exception as e:
        logger.error("Failed to save ICA object to %s: %s", output_path, e)
        raise RuntimeError("Failed to save ICA object: {}".format(e))


def plot_ica_topographies_overview(
    ica_obj: mne.preprocessing.ICA,
    indices_to_plot: Optional[list] = None,
    max_plots_per_fig: int = 25,
) -> list:
    """
    Generate figures showing an overview of ICA component topographies.

    Args:
        ica_obj: The MNE ICA object.
        indices_to_plot: List of component indices to plot. If None, plots all.
        max_plots_per_fig: Maximum number of topographies per figure.

    Returns:
        List of matplotlib Figure objects.
    """
    matplotlib.use("Agg")  # Ensure non-interactive backend
    figures: List[plt.Figure] = []

    if indices_to_plot is None:
        indices_to_plot = list(range(ica_obj.n_components_))

    if not indices_to_plot:
        logger.info("No component topographies to plot for overview.")
        return figures

    for i in range(0, len(indices_to_plot), max_plots_per_fig):
        batch_indices = indices_to_plot[i : i + max_plots_per_fig]
        if not batch_indices:
            continue

        n_batch = len(batch_indices)
        # Calculate layout for a grid of topomaps
        ncols = math.ceil(math.sqrt(n_batch / 1.5))  # Aim for a wider aspect ratio
        nrows = math.ceil(n_batch / ncols)

        fig_batch, axes_batch = plt.subplots(
            nrows,
            ncols,
            figsize=(min(ncols * 2.5, 14), min(nrows * 2.5, 18)),
            squeeze=False,
        )
        fig_batch.suptitle(f"ICA Topographies Overview (Batch {i//max_plots_per_fig + 1})", fontsize=14)

        for ax_idx, comp_idx_topo in enumerate(batch_indices):
            r, c = divmod(ax_idx, ncols)
            ax_curr = axes_batch[r, c]
            try:
                ica_obj.plot_components(
                    picks=comp_idx_topo,
                    axes=ax_curr,
                    show=False,
                    colorbar=False,
                    cmap="jet",
                    outlines="head",
                    sensors=False,
                    contours=4,
                )
                ax_curr.set_title(f"IC{comp_idx_topo}", fontsize=9)
            except Exception as e_single_topo:
                logger.warning(
                    "Could not plot topography for IC%d in overview: %s",
                    comp_idx_topo,
                    e_single_topo,
                )
                ax_curr.text(0.5, 0.5, "Error", ha="center", va="center")
                ax_curr.set_title(f"IC{comp_idx_topo} (Err)", fontsize=9)
            ax_curr.set_xlabel("")
            ax_curr.set_ylabel("")
            ax_curr.set_xticks([])
            ax_curr.set_yticks([])

        # Hide unused axes
        for ax_idx_hide in range(n_batch, nrows * ncols):
            r, c = divmod(ax_idx_hide, ncols)
            fig_batch.delaxes(axes_batch[r, c])

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Space for suptitle
        figures.append(fig_batch)
        # plt.close(fig_batch) # Figure should be closed by the caller (e.g., PdfPages)

    return figures
