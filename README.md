# ICVision

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
