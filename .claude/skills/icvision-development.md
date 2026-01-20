# ICVision Development Skill

## Overview

ICVision is an automated ICA component classification tool for EEG data using OpenAI's Vision API. This skill provides context for developing, testing, and maintaining the ICVision package.

## Package Structure

```
autocleaneeg-icvision/
├── src/icvision/
│   ├── api.py          # OpenAI API calls, strip classification
│   ├── cli.py          # Command-line interface
│   ├── compat.py       # MNE-ICALabel drop-in compatibility
│   ├── config.py       # Prompts, labels, constants
│   ├── core.py         # Main orchestration (label_components)
│   ├── plotting.py     # Component visualization, strip images
│   ├── reports.py      # PDF report generation
│   └── utils.py        # Data loading, file handling
├── tests/              # Test suites
├── plan/               # RFC documents (Quarto)
└── .env                # API credentials (not committed)
```

## Key Concepts

### Classification Modes

| Mode | Description | API Calls | Use Case |
|------|-------------|-----------|----------|
| `single` | One image per component | N | Default, most accurate |
| `strip` | 9 components per image | N/9 | Production, 88% cost reduction |

### API Surface

```python
# Main API (full output)
from icvision.core import label_components
raw_cleaned, ica_updated, results_df = label_components(
    raw_data="data.set",
    layout="strip",           # "single" or "strip"
    base_url="https://...",   # Custom endpoint
    reasoning_effort="none",  # none/low/medium/high
)

# ICLabel-compatible API (drop-in replacement)
from icvision.compat import label_components
result = label_components(raw, ica, method='icvision')
```

### CLI Usage

```bash
# Basic usage
autoclean-icvision data.set

# Strip mode with custom endpoint
autoclean-icvision data.set \
    --layout strip \
    --base-url https://your-proxy.com/v1 \
    --model gpt-5.2 \
    --reasoning-effort none
```

## CLIProxy Integration

### Endpoints

| Environment | Base URL | Notes |
|-------------|----------|-------|
| Production | `https://openai.cincibrainlab.com/v1` | Main endpoint |
| Local test | `http://localhost:28080/v1` | For testing fixes |

### Reasoning Effort Behavior

CLIProxy translates `reasoning_effort` to OpenAI's `reasoning.effort` field:

| Setting | Time | Artifacts | Notes |
|---------|------|-----------|-------|
| `none` | ~50s | 7 | **Fastest** |
| Default (→medium) | ~56s | 5 | Proxy default |
| `low` | ~81s | 4 | Paradoxically slower |

**Recommendation**: Use `none` for speed, or omit parameter for balanced results.

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_strip_compatibility.py

# Test with coverage
uv run pytest --cov=icvision
```

### Test Data Location

- Pre-ICA raw: `/Users/ernie/sandbox/Autoclean-EEG/output/BiotrialResting1020/bids/derivatives/09_pre_ica/`
- Classified ICA: `/Users/ernie/sandbox/Autoclean-EEG/output/BiotrialResting1020/bids/derivatives/icvision_icvis_classified_ica.fif`

## Development Workflow

### RFC Documentation

The project uses Quarto for RFC-style documentation:

```bash
# Render RFC document
cd plan && quarto render multi-tracing-production.qmd

# View rendered HTML
open plan/multi-tracing-production.html
```

### Git Workflow

1. Check status: `git status`
2. Stage changes: `git add <files>`
3. Commit with co-author:
   ```bash
   git commit -m "Message

   Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
   ```
4. Push: `git push`

### Publishing to PyPI

```bash
uv build
uvx uv-publish
```

## Pipeline Integration

ICVision integrates with `autocleaneeg_pipeline` in `ica_processing.py`:

```python
# Pipeline uses strip mode by default
icvision_kwargs = {"layout": "strip", **kwargs}
label_components(raw, ica, **icvision_kwargs)
```

### Fallback Behavior

If ICVision fails, pipeline falls back to ICLabel automatically.

## Common Issues

### "No ICA components found"
- File is pre-ICA (before decomposition)
- Use a file with ICA data or provide separate ICA file

### API errors with reasoning_effort
- Supported values: `none`, `low`, `medium`, `high`, `xhigh`
- `minimal` is NOT supported by gpt-5.2

### Slow with low reasoning
- This is expected CLIProxy/OpenAI behavior
- Use `none` or default for best speed

## Environment Variables

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://openai.cincibrainlab.com/v1"  # Optional
```

## Related Repositories

- `autocleaneeg_pipeline`: Main processing pipeline
- `cliproxy`: OpenAI API proxy (internal)
