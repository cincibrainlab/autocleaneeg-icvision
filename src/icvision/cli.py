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

from . import __version__
from .config import DEFAULT_CONFIG, OPENAI_ICA_PROMPT
from .core import label_components
from .utils import format_summary_stats

# Set up basic logging for CLI; can be overridden by verbose flag
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("icvision_cli")


def setup_cli_logging(verbose: bool = False) -> None:
    """ Set up more detailed logging if verbose flag is used. """
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
        prog="icvision",
        description=(
            f"ICVision v{__version__}: Automated ICA component classification "
            "using OpenAI Vision API for EEG data."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=("""
Examples:
  Basic usage:
    icvision path/to/your_raw.set path/to/your_ica.fif

  With API key and custom output directory:
    icvision raw_data.set ica_data.fif --api-key YOUR_API_KEY --output-dir results/

  Adjusting classification parameters:
    icvision raw.set ica.fif -ct 0.7 --model gpt-4-vision-preview --batch-size 5

  Using a custom prompt file:
    icvision raw.set ica.fif --prompt-file my_custom_prompt.txt

  Disabling report generation:
    icvision raw.set ica.fif --no-report

  For more help on a specific command or option, use: icvision <command> --help
""")
    )

    # Positional arguments for data paths
    parser.add_argument(
        "raw_data_path",
        type=str,
        help="Path to the raw EEG data file (e.g., .set, .fif)."
    )
    parser.add_argument(
        "ica_data_path",
        type=str,
        help="Path to the ICA decomposition file (e.g., .fif)."
    )

    # API and Model configuration
    api_group = parser.add_argument_group("API and Model Configuration")
    api_group.add_argument(
        "-k", "--api-key",
        type=str,
        default=None,
        help="OpenAI API key. If not provided, uses OPENAI_API_KEY env variable."
    )
    api_group.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_CONFIG["model_name"],
        help=f"OpenAI model to use (default: {DEFAULT_CONFIG['model_name']})."
    )
    api_group.add_argument(
        "-p", "--prompt-file",
        type=str,
        default=None,
        help="Path to a custom text file containing the classification prompt. "
             "If not provided, uses the default internal prompt."
    )

    # Classification parameters
    class_group = parser.add_argument_group("Classification Parameters")
    class_group.add_argument(
        "-ct", "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIG["confidence_threshold"],
        help="Minimum confidence for auto-exclusion (0.0-1.0; "
             f"default: {DEFAULT_CONFIG['confidence_threshold']})."
    )
    class_group.add_argument(
        "--no-auto-exclude",
        action="store_false",
        dest="auto_exclude",
        help="Disable automatic exclusion of classified artifact components. "
             "If set, components are labeled but not excluded."
    )
    parser.set_defaults(auto_exclude=DEFAULT_CONFIG["auto_exclude"])
    class_group.add_argument(
        "--labels-to-exclude",
        type=str,
        nargs="+",
        default=None, # Uses default from core.py if None
        help="List of component labels to consider for auto-exclusion "
             "(e.g., eye muscle heart). Defaults to all non-brain types."
    )

    # Output and Reporting
    output_group = parser.add_argument_group("Output and Reporting")
    output_group.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (images, CSV, report). "
             "If None, creates 'icvision_results' in current directory."
    )
    output_group.add_argument(
        "--no-report",
        action="store_false",
        dest="generate_report",
        help="Disable generation of the PDF report."
    )
    parser.set_defaults(generate_report=DEFAULT_CONFIG["generate_report"])

    # Performance parameters
    perf_group = parser.add_argument_group("Performance Parameters")
    perf_group.add_argument(
        "-bs", "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help="Number of components to process concurrently for API calls "
             f"(default: {DEFAULT_CONFIG['batch_size']})."
    )
    perf_group.add_argument(
        "-mc", "--max-concurrency",
        type=int,
        default=DEFAULT_CONFIG["max_concurrency"],
        help="Maximum number of parallel API requests "
             f"(default: {DEFAULT_CONFIG['max_concurrency']})."
    )
    
    # General options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging for detailed output."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    args = parser.parse_args()

    # Setup logging level based on verbose flag
    setup_cli_logging(args.verbose)

    # Load custom prompt if provided
    custom_prompt_text = None
    if args.prompt_file:
        try:
            prompt_path = Path(args.prompt_file)
            if not prompt_path.is_file():
                logger.error(f"Custom prompt file not found: {prompt_path}")
                sys.exit(1)
            custom_prompt_text = prompt_path.read_text(encoding='utf-8')
            logger.info(f"Using custom prompt from: {prompt_path}")
        except Exception as e:
            logger.error(f"Failed to read custom prompt file: {e}")
            sys.exit(1)

    logger.info(f"Starting ICVision CLI v{__version__}")
    logger.info(f"Processing Raw: {args.raw_data_path}, ICA: {args.ica_data_path}")

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
        
        logger.info("ICVision processing completed successfully.")
        
        # Print summary statistics to console
        summary = format_summary_stats(results_df)
        print("\n" + summary) # Add newline for better console output
        
        if args.output_dir:
            output_path = Path(args.output_dir)
        else:
            output_path = Path.cwd() / "icvision_results"
        logger.info(f"All results, logs, and reports (if enabled) are in: {output_path.resolve()}")

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid input or configuration: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Processing error: {e}")
        sys.exit(1)
    except openai.AuthenticationError as e: # Catch specific OpenAI auth error
        logger.error(f"OpenAI Authentication Error: {e}. Please check your API key.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        # For debugging, you might want to re-raise or print traceback
        # import traceback
        # logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 