"""
PDF Report generation for ICVision.

This module handles the creation of comprehensive PDF reports summarizing
the ICA component classification results, including visualizations.
"""

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import mne
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .config import COLOR_MAP
from .plotting import plot_component_for_classification, plot_ica_topographies_overview

# Set up logging for the module
logger = logging.getLogger("icvision.reports")


def _create_summary_table_page(
    pdf_pages: PdfPages,
    results_df: pd.DataFrame,
    component_indices: List[int],
    report_title_prefix: str = "AutocleanEEG ICVision Report",
    source_filename: Optional[str] = None,
) -> None:
    """
    Creates summary table pages for the PDF report.

    Args:
        pdf_pages: The PdfPages object to save figures to.
        results_df: DataFrame with classification results.
        component_indices: List of component indices to include in this summary.
        report_title_prefix: Prefix for the report title.
    """
    matplotlib.use("Agg")  # Ensure non-interactive backend
    components_per_page = 20  # Fit more components by reducing font and adjusting layout
    num_total_components = len(component_indices)

    if num_total_components == 0:
        logger.debug("No components to summarize in table page.")
        return

    num_summary_pages = math.ceil(num_total_components / components_per_page)

    for page_num in range(num_summary_pages):
        start_idx = page_num * components_per_page
        end_idx = min((page_num + 1) * components_per_page, num_total_components)
        page_component_indices = component_indices[start_idx:end_idx]

        if not page_component_indices:
            continue

        fig_table = plt.figure(figsize=(11, 8.5))  # US Letter size
        ax_table = fig_table.add_subplot(111)
        ax_table.axis("off")

        table_data = []
        table_cell_colors = []

        for comp_idx in page_component_indices:
            if comp_idx not in results_df.index:
                logger.warning(
                    "Component IC%d not in results. Skipping in summary table.",
                    comp_idx,
                )
                continue

            comp_info = results_df.loc[comp_idx]
            label = comp_info.get("label", "N/A")
            mne_label = comp_info.get("mne_label", "N/A")
            confidence = comp_info.get("confidence", 0.0)
            excluded_text = "Yes" if comp_info.get("exclude_vision", False) else "No"
            reason_snippet = (
                str(comp_info.get("reason", ""))[:45] + "..."
                if len(str(comp_info.get("reason", ""))) > 45
                else str(comp_info.get("reason", ""))
            )

            table_data.append(
                [
                    f"IC{comp_idx}",
                    str(label).title(),
                    str(mne_label).title(),
                    f"{confidence:.2f}",
                    excluded_text,
                    reason_snippet,
                ]
            )

            row_color = COLOR_MAP.get(label, "#ffffff")  # Default to white
            table_cell_colors.append([row_color] * 6)  # 6 columns

        if not table_data:
            plt.close(fig_table)
            continue

        table = ax_table.table(
            cellText=table_data,
            colLabels=[
                "Component",
                "Vision Label",
                "MNE Label",
                "Confidence",
                "Excluded?",
                "Reason (Brief)",
            ],
            loc="center",
            cellLoc="left",
            cellColours=table_cell_colors,
            colWidths=[0.1, 0.15, 0.15, 0.12, 0.12, 0.36],  # Adjusted for content
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)  # Smaller font for more rows
        table.scale(1.0, 1.1)  # Adjust scale

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        page_info = f"(Page {page_num + 1} of {num_summary_pages})"
        fig_table.suptitle(
            f"{report_title_prefix} - Classification Summary\n" f"{page_info} - Generated: {timestamp}",
            fontsize=10,
            y=0.97,  # Adjusted y for smaller font
        )

        # Add legend for label colors
        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color, label=label.title()) for label, color in COLOR_MAP.items()
        ]
        if legend_patches:
            ax_table.legend(
                handles=legend_patches,
                loc="upper right",
                bbox_to_anchor=(1.02, 0.92),  # Adjusted position
                title="Vision Labels",
                fontsize=6,
            )

        # Add footer with source filename if provided
        if source_filename:
            fig_table.text(
                0.5,
                0.02,
                f"Autoclean ICVision | https://github.com/cincibrainlab | Source: {source_filename}",
                ha="center",
                va="bottom",
                fontsize=8,
                style="italic",
                color="gray",
                transform=fig_table.transFigure,
            )

        plt.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.08)
        pdf_pages.savefig(fig_table, bbox_inches="tight")
        plt.close(fig_table)


def generate_classification_report(
    ica_obj: mne.preprocessing.ICA,
    raw_obj: mne.io.Raw,
    results_df: pd.DataFrame,
    output_dir: Path,
    input_basename: Optional[str] = None,
    report_filename_prefix: Optional[str] = None,
    components_to_detail: str = "all",  # "all" or "artifacts_only"
    source_filename: Optional[str] = None,
    psd_fmax: Optional[float] = None,
) -> Optional[Path]:
    """
    Generates a comprehensive PDF report for ICA component classifications.

    Args:
        ica_obj: The MNE ICA object (updated with classifications).
        raw_obj: The MNE Raw object (cleaned or original, for context).
        results_df: DataFrame with classification results from ICVision.
        output_dir: Directory to save the PDF report.
        input_basename: Basename from input file for default naming.
        report_filename_prefix: Custom prefix for the PDF report filename. If None, uses basename_icvis_report.
        components_to_detail: Which components to include detail pages for:
                             "all" or "artifacts_only" (where 'exclude_vision' is True).
        source_filename: Original filename for PDF footer.
        psd_fmax: Maximum frequency for PSD plots in Hz (default: None for auto).

    Returns:
        Path to the generated PDF report, or None if generation failed.
    """
    matplotlib.use("Agg")  # Ensure non-interactive backend

    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e_mkdir:
            logger.error("Could not create output directory %s: %s", output_dir, e_mkdir)
            return None

    # Set default filename prefix based on basename
    if report_filename_prefix is None:
        if input_basename is None:
            report_filename_prefix = "icvision_report"
        else:
            report_filename_prefix = f"{input_basename}_icvis_report"

    report_type_suffix = "all_comps" if components_to_detail == "all" else "artifacts_only"
    pdf_filename = f"{report_filename_prefix}_{report_type_suffix}.pdf"
    pdf_path = output_dir / pdf_filename

    if pdf_path.exists():
        try:
            pdf_path.unlink()
            logger.debug("Removed existing PDF: %s", pdf_path)
        except OSError as e_unlink:
            logger.warning("Could not delete existing PDF %s: %s", pdf_path, e_unlink)

    # Determine which component indices to plot in detail
    detail_plot_indices = []
    if not results_df.empty:
        if components_to_detail == "all":
            detail_plot_indices = list(results_df.index)
        elif components_to_detail == "artifacts_only":
            if "exclude_vision" in results_df.columns:
                detail_plot_indices = list(results_df[results_df["exclude_vision"]].index)
            else:
                logger.warning("'exclude_vision' column not in results. " "Cannot filter artifacts for report.")
                detail_plot_indices = list(results_df.index)  # Fallback to all

    # Ensure indices are valid and sorted
    valid_indices = [idx for idx in detail_plot_indices if idx < ica_obj.n_components_]
    detail_plot_indices = sorted(list(set(valid_indices)))

    if not detail_plot_indices and results_df.empty:
        logger.info("No classification results available. Skipping PDF report generation.")
        return None
    if not detail_plot_indices and not results_df.empty:
        logger.info(
            f"No components meet criteria '{components_to_detail}' for detail plots. "
            f"Report will contain summary only."
        )

    logger.info("Generating PDF report ('%s') to %s...", report_type_suffix, pdf_path)

    try:
        with PdfPages(pdf_path) as pdf:
            report_title = f"ICVision Report - {report_filename_prefix}"

            # 1. Summary Table Page(s)
            # Include all classified components in the summary table, regardless of detail_plot_indices
            all_classified_indices = list(results_df.index) if not results_df.empty else []
            all_classified_indices = sorted([idx for idx in all_classified_indices if idx < ica_obj.n_components_])
            if all_classified_indices:
                _create_summary_table_page(pdf, results_df, all_classified_indices, report_title, source_filename)
            else:
                logger.info("No components classified to include in summary table.")

            # 2. Component Topographies Overview Page (for components in detail_plot_indices)
            if detail_plot_indices:
                logger.debug(
                    "Plotting topographies overview for %d components.",
                    len(detail_plot_indices),
                )
                topo_overview_figs = plot_ica_topographies_overview(ica_obj, detail_plot_indices)
                for fig_topo_batch in topo_overview_figs:
                    # Add footer with source filename if provided
                    if source_filename:
                        fig_topo_batch.text(
                            0.5,
                            0.02,
                            f"Source: {source_filename}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            style="italic",
                            color="gray",
                            transform=fig_topo_batch.transFigure,
                        )
                    pdf.savefig(fig_topo_batch)
                    plt.close(fig_topo_batch)
            else:
                logger.info("No components selected for topography overview page.")

            # 3. Individual Component Detail Pages (for components in detail_plot_indices)
            if detail_plot_indices:
                logger.debug(
                    "Generating detail pages for %d components.",
                    len(detail_plot_indices),
                )
                for comp_idx_detail in detail_plot_indices:
                    if comp_idx_detail not in results_df.index:
                        logger.warning(
                            "Skipping IC%d detail: not in classification results.",
                            comp_idx_detail,
                        )
                        continue

                    comp_info = results_df.loc[comp_idx_detail]
                    label = comp_info.get("label", "N/A")
                    conf = comp_info.get("confidence", 0.0)
                    reason = comp_info.get("reason", "N/A")

                    fig_detail = plot_component_for_classification(
                        ica_obj=ica_obj,
                        raw_obj=raw_obj,  # Pass raw for context
                        component_idx=comp_idx_detail,
                        output_dir=output_dir,  # Needed for temp saving if not returning fig
                        classification_label=label,
                        classification_confidence=conf,
                        classification_reason=reason,
                        return_fig_object=True,
                        source_filename=source_filename,
                        psd_fmax=psd_fmax,  # Pass through PSD frequency limit
                    )

                    if fig_detail:
                        try:
                            pdf.savefig(fig_detail)
                        except Exception as e_save_detail:
                            logger.error(
                                "Error saving detail page for IC%d to PDF: %s",
                                comp_idx_detail,
                                e_save_detail,
                            )
                            # Create a fallback error page in PDF for this component
                            fig_err_s = plt.figure()
                            ax_err_s = fig_err_s.add_subplot(111)
                            ax_err_s.text(
                                0.5,
                                0.5,
                                "Plot save for IC{}\nfailed.".format(comp_idx_detail),
                                ha="center",
                                va="center",
                            )
                            pdf.savefig(fig_err_s)
                            plt.close(fig_err_s)
                        finally:
                            plt.close(fig_detail)  # Ensure figure is always closed
                    else:
                        logger.warning(
                            "Failed to generate plot object for IC%d for PDF detail page.",
                            comp_idx_detail,
                        )
                        fig_err_g = plt.figure()
                        ax_err_g = fig_err_g.add_subplot(111)
                        ax_err_g.text(
                            0.5,
                            0.5,
                            "Plot gen for IC{}\nfailed.".format(comp_idx_detail),
                            ha="center",
                            va="center",
                        )
                        pdf.savefig(fig_err_g)
                        plt.close(fig_err_g)
            else:
                logger.info("No components selected for individual detail pages.")

            # Add metadata to PDF
            d = pdf.infodict()
            d["Title"] = report_title
            d["Author"] = "ICVision Package"
            d["Subject"] = "ICA Component Classification Report"
            d["Keywords"] = "EEG ICA OpenAI Vision Classification"
            d["CreationDate"] = datetime.now()
            d["ModDate"] = datetime.now()

        logger.info("Successfully generated ICVision PDF report: %s", pdf_path)
        return pdf_path

    except ImportError:
        logger.error("Matplotlib PdfPages not available. PDF report generation failed.")
        return None
    except Exception as e_pdf_main:
        logger.error("Major error during PDF report generation: %s", e_pdf_main)
        # import traceback
        # logger.error(traceback.format_exc()) # For detailed debugging if needed
        return None
