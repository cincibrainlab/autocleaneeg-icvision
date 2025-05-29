"""
Command-Line Interface for ICVision.

This module provides a CLI for classifying ICA components using the ICVision package.
It allows users to specify input data (Raw and ICA), API key, and various
classification parameters directly from the command line.
"""

import argparse
import logging
import sys
from pathlib import Path

import openai

from . import __version__
from .cli_formatter import CLIFormatter, print_error, print_success, print_info
from .config import DEFAULT_CONFIG
from .core import label_components
from .utils import format_summary_stats

# Set up basic logging for CLI; can be overridden by verbose flag
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("icvision_cli")


def setup_cli_logging(verbose: bool = False) -> None:
    """Set up more detailed logging if verbose flag is used."""
    level = logging.DEBUG if verbose else logging.INFO
    # Get root logger to change level for all package loggers
    logging.getLogger("icvision").setLevel(level)
    # Also set CLI logger level if needed, though package level should cover it
    logger.setLevel(level)
    if verbose:
        logger.info("Verbose logging enabled.")


def main() -> None:
    """Main CLI entry point for ICVision."""
    parser = argparse.ArgumentParser(
        prog="autoclean-icvision",
        description=(
            f"ICVision v{__version__}: Automated ICA component classification " "using OpenAI Vision API for EEG data."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            """
Examples:
  Basic usage with EEGLAB .set file (auto-detects ICA):
    autoclean-icvision path/to/your_raw.set

  Basic usage with separate files:
    autoclean-icvision path/to/your_raw.set path/to/your_ica.fif

  With API key and custom output directory:
    autoclean-icvision raw_data.set --api-key YOUR_API_KEY --output-dir results/

  Using separate ICA file:
    autoclean-icvision raw_data.set ica_data.fif --api-key YOUR_API_KEY --output-dir results/

  Adjusting classification parameters:
    autoclean-icvision raw.set -ct 0.7 --model gpt-4-vision-preview --batch-size 5

  Using a custom prompt file:
    autoclean-icvision raw.set --prompt-file my_custom_prompt.txt

  Disabling report generation:
    autoclean-icvision raw.set --no-report

  For more help on a specific command or option, use: autoclean-icvision <command> --help
"""
        ),
    )

    # Positional arguments for data paths
    parser.add_argument(
        "raw_data_path",
        type=str,
        help="Path to the raw EEG data file (e.g., .set, .fif).",
    )
    parser.add_argument(
        "ica_data_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the ICA decomposition file (e.g., .fif, .set). "
        "Optional if raw_data_path is an EEGLAB .set file containing ICA data.",
    )

    # API and Model configuration
    api_group = parser.add_argument_group("API and Model Configuration")
    api_group.add_argument(
        "-k",
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key. If not provided, uses OPENAI_API_KEY env variable.",
    )
    api_group.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_CONFIG["model_name"],
        help=f"OpenAI model to use (default: {DEFAULT_CONFIG['model_name']}).",
    )
    api_group.add_argument(
        "-p",
        "--prompt-file",
        type=str,
        default=None,
        help="Path to a custom text file containing the classification prompt. "
        "If not provided, uses the default internal prompt.",
    )

    # Classification parameters
    class_group = parser.add_argument_group("Classification Parameters")
    class_group.add_argument(
        "-ct",
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIG["confidence_threshold"],
        help="Minimum confidence for auto-exclusion (0.0-1.0; " f"default: {DEFAULT_CONFIG['confidence_threshold']}).",
    )
    class_group.add_argument(
        "--no-auto-exclude",
        action="store_false",
        dest="auto_exclude",
        help="Disable automatic exclusion of classified artifact components. "
        "If set, components are labeled but not excluded.",
    )
    parser.set_defaults(auto_exclude=DEFAULT_CONFIG["auto_exclude"])
    class_group.add_argument(
        "--labels-to-exclude",
        type=str,
        nargs="+",
        default=None,  # Uses default from core.py if None
        help="List of component labels to consider for auto-exclusion "
        "(e.g., eye muscle heart). Defaults to all non-brain types.",
    )

    # Output and Reporting
    output_group = parser.add_argument_group("Output and Reporting")
    output_group.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (images, CSV, report). "
        "If None, creates 'icvision_results' in current directory.",
    )
    output_group.add_argument(
        "--no-report",
        action="store_false",
        dest="generate_report",
        help="Disable generation of the PDF report.",
    )
    parser.set_defaults(generate_report=DEFAULT_CONFIG["generate_report"])

    # Performance parameters
    perf_group = parser.add_argument_group("Performance Parameters")
    perf_group.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help="Number of components to process concurrently for API calls "
        f"(default: {DEFAULT_CONFIG['batch_size']}).",
    )
    perf_group.add_argument(
        "-mc",
        "--max-concurrency",
        type=int,
        default=DEFAULT_CONFIG["max_concurrency"],
        help="Maximum number of parallel API requests " f"(default: {DEFAULT_CONFIG['max_concurrency']}).",
    )

    # General options
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging for detailed output.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    # Setup logging level based on verbose flag
    setup_cli_logging(args.verbose)

    # Load custom prompt if provided
    custom_prompt_text = None
    if args.prompt_file:
        try:
            prompt_path = Path(args.prompt_file)
            if not prompt_path.is_file():
                if args.verbose:
                    logger.error("Custom prompt file not found: %s", prompt_path)
                else:
                    print_error(
                        "Custom prompt file not found",
                        details=f"Could not find file: {prompt_path}",
                        suggestion="Check the file path is correct:\n"
                                  f"  ls -la {prompt_path}"
                    )
                sys.exit(1)
            custom_prompt_text = prompt_path.read_text(encoding="utf-8")
            if args.verbose:
                logger.info("Using custom prompt from: %s", prompt_path)
        except Exception as e:
            if args.verbose:
                logger.error("Failed to read custom prompt file: %s", e)
            else:
                print_error(
                    "Failed to read custom prompt file",
                    details=str(e),
                    suggestion="Check file permissions and encoding:\n"
                              f"  cat {args.prompt_file}"
                )
            sys.exit(1)

    # Show welcome message
    if not args.verbose:
        CLIFormatter.print_welcome(__version__)
        print_info(f"Processing: {args.raw_data_path}" + 
                   (f" + {args.ica_data_path}" if args.ica_data_path else " (auto-detecting ICA)"))
    
    logger.info("Starting ICVision CLI v%s", __version__)
    logger.info("Processing Raw: %s, ICA: %s", args.raw_data_path, args.ica_data_path)

    try:
        raw_cleaned, ica_updated, results_df = label_components(
            raw_data=args.raw_data_path,
            ica_data=args.ica_data_path,
            api_key=args.api_key,
            confidence_threshold=args.confidence_threshold,
            auto_exclude=args.auto_exclude,
            labels_to_exclude=args.labels_to_exclude,
            output_dir=args.output_dir,
            generate_report=args.generate_report,
            batch_size=args.batch_size,
            max_concurrency=args.max_concurrency,
            model_name=args.model,
            custom_prompt=custom_prompt_text,
        )

        # Determine output path
        if args.output_dir:
            output_path = Path(args.output_dir)
        else:
            output_path = Path.cwd() / "icvision_results"
        
        # Print formatted success message and summary
        if not args.verbose:
            excluded_count = results_df.get("exclude_vision", 0).sum() if not results_df.empty else 0
            total_count = len(results_df) if not results_df.empty else 0
            
            print_success(
                f"Processing completed! Classified {total_count} components, excluded {excluded_count} artifacts.",
                details=f"Results saved to: {output_path.resolve()}"
            )
            
            # Print formatted summary stats
            if not results_df.empty:
                label_counts = results_df["label"].value_counts().to_dict()
                CLIFormatter.print_summary_stats({
                    "total_components": total_count,
                    "excluded_artifacts": excluded_count,
                    **{f"{label}_components": count for label, count in label_counts.items()}
                })
        else:
            # Verbose mode - use traditional logging
            logger.info("ICVision processing completed successfully.")
            summary = format_summary_stats(results_df)
            print("\n" + summary)
            logger.info("All results, logs, and reports (if enabled) are in: %s", output_path.resolve())

    except FileNotFoundError as e:
        if args.verbose:
            logger.error("Input file not found: %s", e)
        else:
            print_error(
                "Input file not found",
                details=str(e),
                suggestion="Check that the file path is correct and the file exists:\n"
                          f"  ls -la {args.raw_data_path}"
            )
        sys.exit(1)
    except ValueError as e:
        if args.verbose:
            logger.error("Invalid input or configuration: %s", e)
        else:
            error_str = str(e)
            if "No ICA components found" in error_str:
                print_error(
                    "No ICA decomposition found in your data file",
                    details=error_str,
                    suggestion="Either run ICA decomposition in EEGLAB first, or provide a separate ICA file:\n"
                              f"  autoclean-icvision {args.raw_data_path} your_ica_file.fif"
                )
            elif "API key" in error_str:
                print_error(
                    "OpenAI API key not found",
                    details=error_str,
                    suggestion="Set your API key as an environment variable:\n"
                              "  export OPENAI_API_KEY='sk-your-api-key-here'\n"
                              "Or provide it directly:\n"
                              f"  autoclean-icvision {args.raw_data_path} --api-key sk-your-key"
                )
            else:
                print_error("Invalid input or configuration", details=error_str)
        sys.exit(1)
    except RuntimeError as e:
        if args.verbose:
            logger.error("Processing error: %s", e)
        else:
            print_error("Processing failed", details=str(e))
        sys.exit(1)
    except openai.AuthenticationError as e:
        if args.verbose:
            logger.error("OpenAI Authentication Error: %s. Please check your API key.", e)
        else:
            print_error(
                "OpenAI authentication failed",
                details="Your API key was rejected by OpenAI",
                suggestion="Check your API key is correct and has sufficient credits:\n"
                          "1. Visit https://platform.openai.com/api-keys\n"
                          "2. Verify your key is active\n"
                          "3. Check your billing settings"
            )
        sys.exit(1)
    except Exception as e:
        if args.verbose:
            logger.error("An unexpected error occurred: %s", e)
            # For debugging, you might want to re-raise or print traceback
            # import traceback
            # logger.error(traceback.format_exc())
        else:
            print_error(
                "An unexpected error occurred",
                details=str(e),
                suggestion="Try running with --verbose for more details:\n"
                          f"  autoclean-icvision {args.raw_data_path} --verbose"
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
