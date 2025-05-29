# Autoclean EEG ICVision (Standalone)

[![PyPI version](https://badge.fury.io/py/icvision.svg)](https://badge.fury.io/py/icvision)
[![Python versions](https://img.shields.io/pypi/pyversions/icvision.svg)](https://pypi.org/project/icvision/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Automated ICA component classification for EEG data using OpenAI's Vision API.

## Overview

ICVision automates the tedious process of classifying ICA components from EEG data by generating component visualizations and sending them to OpenAI's Vision API for intelligent artifact identification.

**Workflow**: Raw EEG + ICA → Generate component plots → OpenAI Vision classification → Automated artifact removal → Clean EEG data

**Key Features**:
- Automated classification of 7 component types (brain, eye, muscle, heart, line noise, channel noise, other)
- Multi-panel component plots (topography, time series, PSD, ERP-image)
- MNE-Python integration with `.fif` and `.set` file support
- **EEGLAB .set file auto-detection**: Single file input with automatic ICA detection
- Parallel processing with configurable batch sizes
- Command-line and Python API interfaces
- Comprehensive PDF reports and CSV results

## Installation

```bash
pip install icvision
```

**Requirements**: Python 3.8+ and OpenAI API key with vision model access (e.g., `gpt-4.1`)

```bash
export OPENAI_API_KEY='your_api_key_here'
```

## Usage

### Command-Line Interface (CLI)

The primary way to use ICVision is through its command-line interface.

**Basic Usage:**

**Single EEGLAB .set file (Recommended):**
```bash
icvision /path/to/your_data.set
```

**Separate files:**
```bash
icvision /path/to/your_raw_data.set /path/to/your_ica_decomposition.fif
```

ICVision can automatically detect and read ICA data from EEGLAB `.set` files, making single-file usage possible when your `.set` file contains both raw data and ICA decomposition.

This command will:
1.  Load the raw EEG data and ICA solution (auto-detected from `.set` file or from separate files).
2.  Classify components using the default settings.
3.  Create an `icvision_results/` directory in your current working directory.
4.  Save the following into the output directory:
    *   Cleaned raw data (artifacts removed).
    *   Updated ICA object with component labels.
    *   `icvision_results.csv` detailing classifications for each component.
    *   `classification_summary.txt` with overall statistics.
    *   `icvision_report_all_comps.pdf` (if report generation is enabled by default).

**Common Options (with defaults):**

*   `--api-key YOUR_API_KEY`: Specify OpenAI API key (default: `OPENAI_API_KEY` env variable)
*   `--output-dir /path/to/output/`: Output directory (default: `./icvision_results`)
*   `--model MODEL_NAME`: OpenAI model (default: `gpt-4.1`)
*   `--confidence-threshold 0.8`: Confidence threshold for auto-exclusion (default: `0.8`)
*   `--labels-to-exclude eye muscle heart`: Artifact labels to exclude (default: all non-brain types)
*   `--batch-size 10`: Components per API request (default: `10`)
*   `--max-concurrency 4`: Max parallel requests (default: `4`)
*   `--no-auto-exclude`: Disable auto-exclusion (default: auto-exclude enabled)
*   `--prompt-file /path/to/prompt.txt`: Custom classification prompt (default: built-in prompt)
*   `--no-report`: Disable PDF report (default: report generation enabled)
*   `--verbose`: Enable detailed logging (default: standard logging)
*   `--version`: Show ICVision version
*   `--help`: Show full list of commands and options

**Examples with options:**

Single .set file usage:
```bash
icvision data/subject01_eeg.set \
    --api-key sk-xxxxxxxxxxxxxxxxxxxx \
    --output-dir analysis_results/subject01_icvision \
    --confidence-threshold 0.9 \
    --verbose
```

Traditional separate files:
```bash
icvision data/subject01_raw.fif data/subject01_ica.fif \
    --api-key sk-xxxxxxxxxxxxxxxxxxxx \
    --output-dir analysis_results/subject01_icvision \
    --model gpt-4.1 \
    --confidence-threshold 0.8 \
    --labels-to-exclude eye muscle line_noise channel_noise \
    --batch-size 8 \
    --verbose
```

### Python API

You can also use ICVision programmatically within your Python scripts.

**Single .set file usage (NEW):**
```python
from pathlib import Path
from icvision.core import label_components

# --- Configuration ---
API_KEY = "your_openai_api_key"  # Or set as environment variable OPENAI_API_KEY
DATA_PATH = "path/to/your_data.set"  # EEGLAB .set file with ICA
OUTPUT_DIR = Path("icvision_output")

# --- Run ICVision (ICA auto-detected from .set file) ---
try:
    raw_cleaned, ica_updated, results_df = label_components(
        raw_data=DATA_PATH,              # EEGLAB .set file path
        # ica_data parameter is optional - auto-detected from .set file
        api_key=API_KEY,                 # Optional if OPENAI_API_KEY env var is set
        output_dir=OUTPUT_DIR,
    )
```

**Traditional separate files:**
```python
from pathlib import Path
from icvision.core import label_components

# --- Configuration ---
API_KEY = "your_openai_api_key"  # Or set as environment variable OPENAI_API_KEY
RAW_DATA_PATH = "path/to/your_raw_data.set"
ICA_DATA_PATH = "path/to/your_ica_data.fif"
OUTPUT_DIR = Path("icvision_output")

# --- Run ICVision with all parameters ---
try:
    raw_cleaned, ica_updated, results_df = label_components(
        raw_data=RAW_DATA_PATH,          # Can be MNE object or path string/Path object
        ica_data=ICA_DATA_PATH,          # Can be MNE object, path, or None for auto-detection
        api_key=API_KEY,                 # Optional if OPENAI_API_KEY env var is set
        output_dir=OUTPUT_DIR,
        model_name="gpt-4.1",            # Default: "gpt-4.1"
        confidence_threshold=0.80,       # Default: 0.8
        labels_to_exclude=["eye", "muscle", "heart", "line_noise", "channel_noise"],  # Default: all non-brain
        generate_report=True,            # Default: True
        batch_size=5,                    # Default: 10
        max_concurrency=3,               # Default: 4
        auto_exclude=True,               # Default: True
        custom_prompt=None               # Default: None (uses built-in prompt)
    )

    print("\n--- ICVision Processing Complete ---")
    print(f"Cleaned raw data channels: {raw_cleaned.info['nchan']}")
    print(f"Updated ICA components: {ica_updated.n_components_}")
    print(f"Number of components classified: {len(results_df)}")

    if not results_df.empty:
        print(f"Number of components marked for exclusion: {results_df['exclude_vision'].sum()}")
        print("\nClassification Summary:")
        print(results_df[['component_name', 'label', 'confidence', 'exclude_vision']].head())

    print(f"\nResults saved in: {OUTPUT_DIR.resolve()}")

except Exception as e:
    print(f"An error occurred: {e}")

```

## Configuration Details

### Input File Support

**EEGLAB .set files:**
- **Raw data**: Supports EEGLAB `.set` files for raw EEG data
- **ICA data**: Now supports automatic ICA detection from `.set` files using `mne.preprocessing.read_ica_eeglab()`
- **Single file mode**: Use just a `.set` file when it contains both raw data and ICA decomposition

**MNE formats:**
Other supported formats include:
- **Raw data**: `.fif`, `.edf`, `.raw`
- **ICA data**: `.fif` files containing MNE ICA objects

### Default Parameter Values

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `model_name` | `"gpt-4.1"` | OpenAI model for classification |
| `confidence_threshold` | `0.8` | Minimum confidence for auto-exclusion |
| `auto_exclude` | `True` | Automatically exclude artifact components |
| `labels_to_exclude` | `["eye", "muscle", "heart", "line_noise", "channel_noise", "other_artifact"]` | Labels to exclude (all non-brain) |
| `output_dir` | `"./icvision_results"` | Output directory for results |
| `generate_report` | `True` | Generate PDF report |
| `batch_size` | `10` | Components per API request |
| `max_concurrency` | `4` | Maximum parallel API requests |
| `api_key` | `None` | Uses `OPENAI_API_KEY` environment variable |
| `custom_prompt` | `None` | Uses built-in classification prompt |

### Component Labels

The standard set of labels ICVision uses (and expects from the API) are:
- `brain` - Neural brain activity (retained)
- `eye` - Eye movement artifacts
- `muscle` - Muscle artifacts
- `heart` - Cardiac artifacts
- `line_noise` - Electrical line noise
- `channel_noise` - Channel-specific noise
- `other_artifact` - Other artifacts

These are defined in `src/icvision/config.py`.

### Output Files

*   `{output_dir}/icvision_classified_ica.fif`: MNE ICA object with labels and exclusions
*   `{output_dir}/icvision_results.csv`: Detailed classification results per component
*   `{output_dir}/classification_summary.txt`: Summary statistics by label type
*   `{output_dir}/icvision_report_all_comps.pdf`: Comprehensive PDF report (if enabled)

### Custom Classification Prompt

The default prompt is optimized for EEG component classification on EGI128 nets. You can customize it by:
- **CLI**: `--prompt-file /path/to/custom_prompt.txt`
- **Python API**: `custom_prompt="Your custom prompt here"`
- **View default**: Check `src/icvision/config.py`

## Development

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Citation

If you use ICVision in your research, please consider citing it (details to be added upon publication/DOI generation).

## Acknowledgements

*   This project relies heavily on the [MNE-Python](https://mne.tools/) library.
*   Utilizes the [OpenAI API](https://openai.com/api/).
