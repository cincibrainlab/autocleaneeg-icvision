# Autoclean EEG ICVision

[![PyPI version](https://badge.fury.io/py/autoclean-icvision.svg)](https://badge.fury.io/py/autoclean-icvision)
[![Python versions](https://img.shields.io/pypi/pyversions/autoclean-icvision.svg)](https://pypi.org/project/autoclean-icvision/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Automated ICA component classification for EEG data using OpenAI Vision API.**

Internal tool for CinciBrain Lab - classifies ICA components from EEG data by sending component visualizations to OpenAI's Vision API for intelligent artifact identification.

## Quick Start

```bash
pip install autocleaneeg-icvision
```

### Using CinciBrain Lab Proxy (Recommended)

```bash
export OPENAI_BASE_URL=https://vision.autocleaneeg.org/v1
export OPENAI_API_KEY=dev-local-key

autoclean-icvision your_data.set
```

Or pass directly:

```bash
autoclean-icvision your_data.set \
    --base-url https://vision.autocleaneeg.org/v1 \
    --api-key dev-local-key \
    --model gpt-5.1
```

### Using OpenAI Directly

```bash
export OPENAI_API_KEY=sk-your-api-key
autoclean-icvision your_data.set
```

## Available Models

**Via CinciBrain Lab Proxy** (`vision.autocleaneeg.org`):
- `gpt-5.1` (default)
- `gpt-5.2-codex`
- `gpt-5`
- `gpt-5-codex`
- `gpt-5-codex-mini`
- `gpt-5.1-codex`
- `gpt-5.1-codex-mini`
- `gpt-5.1-codex-max`
- `gpt-5.2`

**Via OpenAI Direct**:
- `gpt-4.1`
- `gpt-4.1-mini`
- `gpt-4.1-nano`

## CLI Usage

```bash
# Basic usage - single EEGLAB .set file
autoclean-icvision data.set

# With separate ICA file
autoclean-icvision raw_data.set ica_data.fif

# Full options
autoclean-icvision data.set \
    --base-url https://vision.autocleaneeg.org/v1 \
    --api-key dev-local-key \
    --model gpt-5.1 \
    --confidence-threshold 0.8 \
    --output-dir results/ \
    --verbose

# For ERP studies with low-pass filtered data
autoclean-icvision erp_data.set --psd-fmax 40

# Batch processing
for f in data/sub-*.set; do
    autoclean-icvision "$f" --base-url https://vision.autocleaneeg.org/v1 --api-key dev-local-key
done
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--base-url` | OpenAI | Custom API endpoint (e.g., `https://vision.autocleaneeg.org/v1`) |
| `--api-key` | `$OPENAI_API_KEY` | API key |
| `--model` | `gpt-4.1` | Model name |
| `--confidence-threshold` | `0.8` | Min confidence for exclusion |
| `--output-dir` | `./autoclean_icvision_results` | Output directory |
| `--psd-fmax` | `80` | Max frequency for PSD plots |
| `--batch-size` | `10` | Components per batch |
| `--max-concurrency` | `4` | Parallel requests |
| `--no-auto-exclude` | - | Disable auto-exclusion |
| `--no-report` | - | Disable PDF report |
| `--verbose` | - | Detailed logging |

## Python API

```python
from icvision.core import label_components

# Using CinciBrain Lab proxy
raw_cleaned, ica_updated, results_df = label_components(
    raw_data="data.set",
    base_url="https://vision.autocleaneeg.org/v1",
    api_key="dev-local-key",
    model_name="gpt-5.1",
    confidence_threshold=0.8,
    output_dir="results/",
)

# Results
print(f"Classified {len(results_df)} components")
print(f"Excluded {results_df['exclude_vision'].sum()} artifacts")
print(results_df[['component_name', 'label', 'confidence', 'exclude_vision']])
```

### ICLabel Drop-in Replacement

```python
from icvision.compat import label_components

# Same API as mne_icalabel
result = label_components(raw, ica, method='icvision')
print(result['labels'])  # ['brain', 'eye blink', 'other', ...]
```

## Component Labels

| Label | Description |
|-------|-------------|
| `brain` | Neural activity (retained) |
| `eye` | Eye movement artifacts |
| `muscle` | Muscle artifacts |
| `heart` | Cardiac artifacts |
| `line_noise` | 50/60Hz line noise |
| `channel_noise` | Bad channel noise |
| `other_artifact` | Other artifacts |

## Output Files

Processing `sub-01_eeg.set` creates:

```
autoclean_icvision_results/
├── sub-01_eeg_icvis_results.csv          # Classification results
├── sub-01_eeg_icvis_classified_ica.fif   # Updated ICA object
├── sub-01_eeg_icvis_cleaned_raw.set      # Cleaned EEG data
├── sub-01_eeg_icvis_summary.txt          # Summary statistics
└── sub-01_eeg_icvis_report_all_comps.pdf # PDF report
```

## Supported Formats

- **EEGLAB**: `.set` files (auto-detects ICA)
- **MNE**: `.fif`, `.edf`, `.raw`

## License

MIT License - CinciBrain Lab
