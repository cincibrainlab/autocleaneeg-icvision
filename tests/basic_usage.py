"""
Basic usage example for ICVision Python API.

This script demonstrates how to use ICVision to classify ICA components
using user-provided EEG data files.
"""

import os
from pathlib import Path

from icvision import label_components


def main() -> None:
    """Run basic ICVision classification with user files."""
    print("ICVision Basic Usage Example")
    print("=" * 40)

    # Set your file paths here
    raw_file = Path("path/to/your_raw_data.fif")  # Change this to your raw data file
    ica_file = Path("path/to/your_ica_data.fif")  # Change this to your ICA file

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return

    # Validate files exist
    if not raw_file.exists():
        print(f"ERROR: Raw data file not found: {raw_file}")
        print("Please update the raw_file path in this script")
        return

    if not ica_file.exists():
        print(f"ERROR: ICA file not found: {ica_file}")
        print("Please update the ica_file path in this script")
        return

    # Create output directory
    output_dir = Path("basic_usage_results")
    output_dir.mkdir(exist_ok=True)

    try:
        print(f"Raw data file: {raw_file}")
        print(f"ICA file: {ica_file}")
        print(f"Output directory: {output_dir}")

        # Run ICVision classification
        print("\nRunning ICVision classification...")
        cleaned_raw, updated_ica, results_df = label_components(
            raw_data=raw_file,
            ica_data=ica_file,
            api_key=api_key,
            output_dir=output_dir,
            confidence_threshold=0.8,
            generate_report=True,
        )

        # Display results
        print("\n" + "=" * 40)
        print("CLASSIFICATION RESULTS")
        print("=" * 40)
        print(f"Components processed: {len(results_df)}")
        print(f"Components excluded: {results_df['exclude_vision'].sum()}")

        print("\nClassification breakdown:")
        label_counts = results_df["label"].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

        print(f"\nResults saved to: {output_dir.resolve()}")

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
