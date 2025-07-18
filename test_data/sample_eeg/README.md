# Sample EEG Test Data

This folder contains sample data for testing the ICVision web component browser.

## Files

- `subject_001_icvis_results.csv` - Pre-existing classification results for 10 ICA components
- `README.md` - This file

## Missing Files (To Be Added)

For a complete test, you would need:

- `subject_001.set` - Raw EEG data in EEGLAB format
- `subject_001.fif` - ICA decomposition in MNE format

## Creating Test Data

Since we cannot include actual EEG data in the repository, you can create test data using:

### Option 1: Use MNE Sample Data

```python
import mne
from mne.datasets import sample
from mne.preprocessing import ICA

# Download sample data
data_path = sample.data_path()
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'

# Load and prepare data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.pick_types(eeg=True, exclude='bads')
raw.filter(1, 40)

# Run ICA
ica = ICA(n_components=10, random_state=42)
ica.fit(raw)

# Save test files
raw.save('test_data/sample_eeg/subject_001_raw.fif', overwrite=True)
ica.save('test_data/sample_eeg/subject_001_ica.fif', overwrite=True)
```

### Option 2: Generate Synthetic Data

```python
import numpy as np
import mne
from mne.preprocessing import ICA

# Create synthetic EEG data
n_channels = 32
n_times = 10000
sfreq = 250

# Generate random data
data = np.random.randn(n_channels, n_times) * 1e-6

# Create info structure
ch_names = [f'EEG{i:03d}' for i in range(1, n_channels + 1)]
info = mne.create_info(ch_names, sfreq, ch_types='eeg')

# Create Raw object
raw = mne.io.RawArray(data, info)

# Run ICA
ica = ICA(n_components=10, random_state=42)
ica.fit(raw)

# Save files
raw.save('test_data/sample_eeg/subject_001_raw.fif', overwrite=True)
ica.save('test_data/sample_eeg/subject_001_ica.fif', overwrite=True)
```

## Testing the Web Interface

1. Create the missing EEG files using one of the methods above
2. Start the web server:
   ```bash
   uvicorn icvision.web.app:app --reload --port 8000
   ```
3. Open http://localhost:8000
4. Enter folder path: `/path/to/icvision/test_data/sample_eeg`
5. Test the component browser functionality

## Expected Behavior

The web interface should:
1. Load the 10 components from the ICA file
2. Display component plots with existing classifications
3. Allow override of classifications
4. Export override CSV files

## Component Classifications

The sample CSV includes:
- 4 brain components (IC0, IC3, IC6, IC8)
- 2 eye artifacts (IC1)
- 1 muscle artifact (IC2)
- 1 heart artifact (IC5)
- 1 line noise (IC4)
- 1 channel noise (IC9)
- 1 other artifact (IC7)

This provides a good mix for testing the override functionality.