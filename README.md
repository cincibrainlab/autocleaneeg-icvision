"""
ICVision - Automated ICA component classification using OpenAI Vision API for EEG data.

ICVision is a Python package that automates the classification of Independent Component Analysis (ICA)
components from EEG data. It generates standardized visualizations of each component, sends them to
the OpenAI Vision API for artifact classification, and then updates the MNE ICA object with these
labels. The package also provides functionality to automatically reject identified artifactual
components from the raw EEG data and generate comprehensive PDF reports.

Key Features:
- Automated classification of ICA components (Brain, Eye, Muscle, Heart, Line Noise, Channel Noise, Other).
- Utilizes OpenAI's multimodal (Vision) API for robust component identification.
- Generates standardized, multi-panel plots for each ICA component.
- Supports common EEG data formats (MNE .fif, EEGLAB .set) via MNE-Python.
- Flexible configuration for API model, confidence thresholds, and component exclusion.
- Command-line interface (CLI) for easy integration into analysis pipelines.
- Generates detailed PDF reports with component visualizations and classification summaries.
- Parallel processing for faster API communication.

Workflow:
1. Load Raw EEG data and pre-computed ICA decomposition.
2. For each ICA component:
   a. Generate a standardized image (topography, time series, PSD, ERP image).
   b. Send the image to the OpenAI Vision API with a specialized prompt.
3. Parse the API's classification (label, confidence, reason).
4. Update the MNE ICA object with these labels and mark components for exclusion based on configuration.
5. Apply the updated ICA to the Raw data to remove artifactual components.
6. Save the cleaned Raw data, updated ICA object, classification results (CSV), and a summary report (PDF).
"""

# ICVision

[![PyPI version](https://badge.fury.io/py/icvision.svg)](https://badge.fury.io/py/icvision)
[![Python versions](https://img.shields.io/pypi/pyversions/icvision.svg)](https://pypi.org/project/icvision/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/yourusername/icvision/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/icvision/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/yourusername/icvision/badge.svg?branch=main)](https://coveralls.io/github/yourusername/icvision?branch=main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Automated ICA component classification using OpenAI Vision API for EEG data.

## Overview

ICVision is a Python package designed to streamline and automate the often tedious process of classifying Independent Component Analysis (ICA) components derived from EEG (Electroencephalography) data. By leveraging the power of OpenAI's multimodal Vision API, ICVision provides a robust and efficient way to identify common biological and non-biological artifacts.

The typical workflow involves:
1.  **Loading Data**: Input your preprocessed EEG data (e.g., MNE .fif or EEGLAB .set files) and the corresponding pre-computed ICA decomposition.
2.  **Component Visualization**: For each independent component, ICVision generates a standardized multi-panel plot. This plot includes:
    *   Topography map
    *   Scrolling time series (first few seconds)
    *   Power Spectral Density (PSD)
    *   An ERP-image like view of continuous data segments
3.  **AI-Powered Classification**: These component images are sent to an OpenAI Vision model (e.g., GPT-4.1, GPT-4 Vision Preview) along with a detailed prompt engineered to guide the model in distinguishing between brain activity and various artifact types (eye movements, muscle activity, heart signals, line noise, channel noise, etc.).
4.  **Results Integration**: The classifications (label, confidence score, and reasoning) from the API are parsed and used to update the MNE `ICA` object. Components are automatically marked for exclusion based on user-configurable criteria (e.g., label type and confidence threshold).
5.  **Data Cleaning**: The updated `ICA` object is then applied to the raw EEG data to remove the identified artifactual components.
6.  **Reporting**: ICVision saves the cleaned data, the updated ICA object, a CSV file with detailed classification results, and can generate a comprehensive PDF report. The report includes summary tables and individual pages for each component, displaying its visualizations and the AI's classification.

## Key Features

*   **Automated Artifact Identification**: Significantly reduces manual effort in ICA component review.
*   **OpenAI Vision API Integration**: Utilizes state-of-the-art AI for component classification.
*   **MNE Python Compatibility**: Works seamlessly with MNE-Python data structures (`Raw`, `ICA`).
*   **Flexible Input**: Supports MNE `.fif` files and EEGLAB `.set` files (via MNE).
*   **Configurable**: Control API model, confidence thresholds, labels to exclude, batch processing parameters, and more.
*   **Command-Line Interface (CLI)**: Easy to use from the terminal for batch processing and pipeline integration.
*   **Detailed Reporting**: Generates PDF reports with visualizations and classification summaries for review and documentation.
*   **Parallel Processing**: Efficiently handles multiple components by making concurrent API requests.
*   **Extensible**: Default prompt and component labels can be customized.

## Installation

```bash
pip install icvision
```

## Prerequisites

*   Python 3.8 or higher.
*   An OpenAI API key with access to a vision-capable model (e.g., `gpt-4.1`, `gpt-4-vision-preview`).
    You can set your API key as an environment variable:
    ```bash
    export OPENAI_API_KEY='your_api_key_here'
    ```
    Alternatively, you can pass it directly via the `--api-key` argument in the CLI or as a parameter in the Python API.

## Usage

### Command-Line Interface (CLI)

The primary way to use ICVision is through its command-line interface.

**Basic Usage:**

```bash
icvision /path/to/your_raw_data.set /path/to/your_ica_decomposition.fif
```

This command will:
1.  Load the raw EEG data and the ICA solution.
2.  Classify components using the default settings (OpenAI model, prompt, thresholds).
3.  Create an `icvision_results/` directory in your current working directory.
4.  Save the following into the output directory:
    *   Cleaned raw data (artifacts removed).
    *   Updated ICA object with component labels.
    *   `icvision_results.csv` detailing classifications for each component.
    *   `classification_summary.txt` with overall statistics.
    *   `icvision_report_all_comps.pdf` (if report generation is enabled by default, which it usually is).

**Common Options:**

*   `--api-key YOUR_API_KEY`: Specify your OpenAI API key.
*   `--output-dir /path/to/output/`: Specify a custom output directory.
*   `--model gpt-4-vision-preview`: Use a specific OpenAI model.
*   `--confidence-threshold 0.75`: Set the minimum confidence for a component to be auto-excluded.
*   `--labels-to-exclude eye muscle heart`: Specify which artifact labels should lead to exclusion.
*   `--no-auto-exclude`: Label components but do not automatically exclude them.
*   `--prompt-file /path/to/custom_prompt.txt`: Use a custom classification prompt.
*   `--no-report`: Disable PDF report generation.
*   `--verbose`: Enable more detailed logging output.
*   `--version`: Show ICVision version.
*   `--help`: Show the full list of commands and options.

**Example with more options:**

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

```python
from pathlib import Path
import mne
from icvision.core import label_components
from icvision.utils import load_raw_data, load_ica_data

# --- Configuration ---
API_KEY = "your_openai_api_key"  # Or set as environment variable OPENAI_API_KEY
RAW_DATA_PATH = Path("path/to/your_raw_data.set")
ICA_DATA_PATH = Path("path/to/your_ica_data.fif")
OUTPUT_DIR = Path("icvision_output")

# --- Load Data (Example) ---
# In a real scenario, load your actual MNE Raw and ICA objects or paths to them.
# For demonstration, we create dummy data if files don't exist.

# Create dummy raw data if it doesn't exist (replace with your actual loading)
def get_or_create_dummy_raw(raw_path, sfreq=250, n_ch=10, n_sec=20):
    if raw_path.exists():
        return load_raw_data(raw_path)
    print(f"Dummy raw data not found at {raw_path}, creating one.")
    ch_names = [f"EEG {i:03}" for i in range(n_ch)]
    data = np.random.randn(n_ch, n_sec * sfreq)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_ch)
    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1020")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(raw_path, overwrite=True)
    return raw

# Create dummy ICA data if it doesn't exist (replace with your actual loading)
def get_or_create_dummy_ica(ica_path, raw_obj, n_comps=5):
    if ica_path.exists():
        return load_ica_data(ica_path)
    print(f"Dummy ICA data not found at {ica_path}, creating one.")
    ica = mne.preprocessing.ICA(n_components=n_comps, random_state=42, max_iter='auto')
    ica.fit(raw_obj.copy().pick_types(eeg=True))
    ica_path.parent.mkdir(parents=True, exist_ok=True)
    ica.save(ica_path, overwrite=True)
    return ica

# Ensure parent directory for dummy data exists for the example
if not RAW_DATA_PATH.parent.exists(): RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
if not ICA_DATA_PATH.parent.exists(): ICA_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

raw_input_data = get_or_create_dummy_raw(RAW_DATA_PATH)
ica_input_data = get_or_create_dummy_ica(ICA_DATA_PATH, raw_input_data)

# --- Run ICVision ---
try:
    raw_cleaned, ica_updated, results_df = label_components(
        raw_data=raw_input_data,         # Can be MNE object or path string/Path object
        ica_data=ica_input_data,         # Can be MNE object or path string/Path object
        api_key=API_KEY,                 # Optional if OPENAI_API_KEY env var is set
        output_dir=OUTPUT_DIR,
        model_name="gpt-4.1",          # Specify the model
        confidence_threshold=0.80,       # Components with confidence >= 0.8 for specified labels will be excluded
        labels_to_exclude=["eye", "muscle", "heart", "line_noise", "channel_noise"],
        generate_report=True,            # Generate a PDF report
        batch_size=5,                    # Number of components to process concurrently
        max_concurrency=3                # Max parallel API requests
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
    # import traceback
    # print(traceback.format_exc()) # For detailed debugging

```

## Configuration Details

*   **Prompt**: The default prompt sent to the OpenAI API is designed to be comprehensive. You can view/modify it in `src/icvision/config.py` or provide your own prompt via a text file using `--prompt-file` (CLI) or `custom_prompt` (Python API).
*   **Component Labels**: The standard set of labels ICVision uses (and expects from the API) are: `brain`, `eye`, `muscle`, `heart`, `line_noise`, `channel_noise`, `other_artifact`. These are defined in `src/icvision/config.py`.
*   **Output Files**:
    *   `{output_dir}/icvision_classified_ica.fif`: The MNE ICA object updated with labels and exclusions.
    *   `{output_dir}/icvision_cleaned_raw.fif` (if raw was modified, name may vary or not be saved by default - `label_components` returns the cleaned raw object).
    *   `{output_dir}/icvision_results.csv`: A CSV file detailing the classification for each component (index, name, label, confidence, reason, exclude status).
    *   `{output_dir}/classification_summary.txt`: A text file with a summary of how many components were classified into each category.
    *   `{output_dir}/{report_filename_prefix}_{report_type}.pdf`: A PDF report (if `generate_report=True`).

## Development

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

To set up a development environment:

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/icvision.git
    cd icvision
    ```
2.  Create a virtual environment and activate it:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  Install dependencies, including development tools:
    ```bash
    pip install -e ".[dev,test,docs]"
    ```
4.  Install pre-commit hooks:
    ```bash
    pre-commit install
    ```

### Running Tests

```bash
pytest
```
Or using tox for multi-environment testing:
```bash
tox
```

### Building Documentation

(Assuming Sphinx is set up in the `docs/` directory)
```bash
cd docs
make html
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Citation

If you use ICVision in your research, please consider citing it (details to be added upon publication/DOI generation).

## Acknowledgements

*   This project relies heavily on the [MNE-Python](https://mne.tools/) library.
*   Utilizes the [OpenAI API](https://openai.com/api/).

*(Please replace `yourusername` with your actual GitHub username in URLs and examples.)*
