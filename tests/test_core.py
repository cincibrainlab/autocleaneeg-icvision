"""
Unit and integration tests for the core ICVision functionality.

This module tests the `label_components` function and its helper functions
within `src.icvision.core` to ensure robust and correct operation under various
conditions, including different input types, API interactions (mocked), and
output validation.
"""

import logging
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pandas as pd
import pytest

from icvision.config import COMPONENT_LABELS, DEFAULT_CONFIG, DEFAULT_EXCLUDE_LABELS
from icvision.core import label_components, _update_ica_with_classifications, _apply_artifact_rejection
from icvision.utils import load_raw_data, load_ica_data

# Configure logging for tests
logger = logging.getLogger("icvision_tests_core")
logger.setLevel(logging.DEBUG) # Show detailed logs during testing

# --- Test Data Setup ---

# Create a fixture for a temporary test directory
@pytest.fixture(scope="module")
def temp_test_dir(tmp_path_factory):
    """Create a temporary directory for test artifacts."""
    tdir = tmp_path_factory.mktemp("icvision_core_tests")
    logger.info(f"Created temporary test directory: {tdir}")
    yield tdir
    # No explicit shutil.rmtree(tdir) needed due to tmp_path_factory
    logger.info(f"Temporary test directory {tdir} will be cleaned up.")

# Create a fixture for dummy raw data
@pytest.fixture(scope="module")
def dummy_raw_data(temp_test_dir) -> mne.io.Raw:
    """Generate a simple MNE Raw object for testing."""
    sfreq = 250
    n_channels = 5
    n_seconds = 10
    ch_names = [f"EEG {i:03}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    data = np.random.randn(n_channels, n_seconds * sfreq)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1020") # Add a montage for plotting
    # Save to a file to also test file loading path
    raw_path = temp_test_dir / "dummy_raw.fif"
    raw.save(raw_path, overwrite=True)
    logger.debug(f"Created and saved dummy raw data to {raw_path}")
    return raw

# Create a fixture for a dummy ICA object
@pytest.fixture(scope="module")
def dummy_ica_data(dummy_raw_data, temp_test_dir) -> mne.preprocessing.ICA:
    """Generate a simple MNE ICA object for testing."""
    n_components = 3
    ica = mne.preprocessing.ICA(
        n_components=n_components, 
        random_state=42, 
        max_iter="auto"
    )
    ica.fit(dummy_raw_data)
    # Save to a file to also test file loading path
    ica_path = temp_test_dir / "dummy_ica.fif"
    ica.save(ica_path, overwrite=True)
    logger.debug(f"Created and saved dummy ICA data to {ica_path}")
    return ica

# --- Mocked API Responses ---

@pytest.fixture
def mock_openai_classify_success():
    """Mock a successful OpenAI API classification response."""
    # This mock function will be called by classify_components_batch
    # It needs to return a DataFrame similar to what classify_components_batch would produce.
    def mock_classify_batch(*args, **kwargs):
        ica_obj = kwargs.get('ica_obj')
        n_comps = ica_obj.n_components_
        results = []
        for i in range(n_comps):
            label = COMPONENT_LABELS[i % len(COMPONENT_LABELS)] # Cycle through labels
            results.append({
                'component_index': i,
                'component_name': f'IC{i}',
                'label': label,
                'mne_label': label, # Simplified for mock
                'confidence': 0.95,
                'reason': f'Mocked reason for {label} component IC{i}',
                'exclude_vision': label != 'brain' # Exclude if not brain
            })
        df = pd.DataFrame(results)
        df = df.set_index('component_index', drop=False)
        return df
    return mock_classify_batch

@pytest.fixture
def mock_openai_classify_failure():
    """Mock a failing OpenAI API classification response."""
    def mock_classify_batch_fail(*args, **kwargs):
        raise openai.APIError("Mocked API Error")
    return mock_classify_batch_fail

# --- Tests for label_components --- 

@patch('icvision.core.classify_components_batch')
@patch('icvision.core.generate_classification_report')
def test_label_components_successful_run(
    mock_gen_report: MagicMock,
    mock_classify_batch_api: MagicMock,
    dummy_raw_data: mne.io.Raw,
    dummy_ica_data: mne.preprocessing.ICA,
    mock_openai_classify_success,
    temp_test_dir: Path
):
    """Test a full successful run of label_components with mocked API."""
    logger.info("Testing successful run of label_components...")
    mock_classify_batch_api.side_effect = mock_openai_classify_success
    
    raw_path = temp_test_dir / "dummy_raw.fif"
    ica_path = temp_test_dir / "dummy_ica.fif"

    raw_cleaned, ica_updated, results_df = label_components(
        raw_data=raw_path, # Test with file paths
        ica_data=ica_path,
        api_key="FAKE_API_KEY",
        output_dir=temp_test_dir,
        generate_report=True
    )

    assert isinstance(raw_cleaned, mne.io.Raw), "Cleaned raw should be an MNE Raw object"
    assert isinstance(ica_updated, mne.preprocessing.ICA), "Updated ICA should be an MNE ICA object"
    assert isinstance(results_df, pd.DataFrame), "Results should be a Pandas DataFrame"
    
    assert not results_df.empty, "Results DataFrame should not be empty"
    assert 'label' in results_df.columns, "'label' column missing in results"
    assert 'exclude_vision' in results_df.columns, "'exclude_vision' column missing"

    # Check if API mock was called
    mock_classify_batch_api.assert_called_once()
    
    # Check if report generation was called
    mock_gen_report.assert_called_once()
    
    # Check if files were created in output_dir
    assert (temp_test_dir / "icvision_results.csv").exists(), "Results CSV not created"
    assert (temp_test_dir / "icvision_classified_ica.fif").exists(), "Updated ICA FIF not created"
    assert (temp_test_dir / "classification_summary.txt").exists(), "Summary TXT not created"

    # Verify ICA object update
    assert ica_updated.labels_ is not None, "ICA labels_ should be set"
    assert ica_updated.exclude is not None, "ICA exclude should be set"
    if dummy_ica_data.n_components_ > 0:
         # Example: check if at least one component was marked for exclusion (if not all brain)
        if any(r['label'] != 'brain' for r in results_df.to_dict(orient='records')):
            assert len(ica_updated.exclude) > 0, "Expected some components to be excluded"
    
    logger.info("Successful run test completed.")

@patch('icvision.core.classify_components_batch')
def test_label_components_api_failure(
    mock_classify_batch_api: MagicMock,
    dummy_raw_data: mne.io.Raw, 
    dummy_ica_data: mne.preprocessing.ICA,
    mock_openai_classify_failure,
    temp_test_dir: Path
):
    """Test label_components handling of API call failures."""
    logger.info("Testing API failure handling in label_components...")
    mock_classify_batch_api.side_effect = mock_openai_classify_failure
    
    with pytest.raises(RuntimeError, match="Failed to classify components: Mocked API Error"):
        label_components(
            raw_data=dummy_raw_data, # Test with MNE objects
            ica_data=dummy_ica_data,
            api_key="FAKE_API_KEY",
            output_dir=temp_test_dir,
            generate_report=False # Disable report to isolate API error
        )
    logger.info("API failure handling test completed.")


@patch('icvision.core.classify_components_batch')
def test_label_components_no_report(
    mock_classify_batch_api: MagicMock,
    dummy_raw_data: mne.io.Raw, 
    dummy_ica_data: mne.preprocessing.ICA,
    mock_openai_classify_success,
    temp_test_dir: Path
):
    """Test label_components with report generation disabled."""
    logger.info("Testing label_components with no report generation...")
    mock_classify_batch_api.side_effect = mock_openai_classify_success

    # Use a sub-directory for this test to avoid conflicts
    no_report_output_dir = temp_test_dir / "no_report_test"
    no_report_output_dir.mkdir(exist_ok=True)

    with patch('icvision.core.generate_classification_report') as mock_gen_report_local:
        label_components(
            raw_data=dummy_raw_data,
            ica_data=dummy_ica_data,
            api_key="FAKE_API_KEY",
            output_dir=no_report_output_dir,
            generate_report=False
        )
        mock_gen_report_local.assert_not_called() 

    # Check that report file does NOT exist
    # (Note: generate_classification_report itself creates the file, so if it's not called, file won't exist)
    # This test primarily ensures the function is not called.
    report_files = list(no_report_output_dir.glob("*.pdf"))
    assert not report_files, "PDF report should not be created when generate_report is False"
    logger.info("No report generation test completed.")


def test_label_components_invalid_inputs():
    """Test label_components with various invalid inputs."""
    logger.info("Testing invalid inputs for label_components...")
    # Invalid raw_data path
    with pytest.raises(FileNotFoundError):
        label_components("invalid_path.set", "some_ica.fif", api_key="key")

    # Invalid ica_data path (using a valid raw for this check)
    raw_file = mne.io.RawArray(np.random.rand(1,100), mne.create_info(1,100,'eeg'))
    with pytest.raises(FileNotFoundError):
        label_components(raw_file, "invalid_ica.fif", api_key="key")
    
    # Missing API key (if not in env)
    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}): # Ensure env var is empty
        with pytest.raises(ValueError, match="No OpenAI API key provided"):
            label_components(raw_file, raw_file, api_key=None) # raw_file for ica to pass load_ica_data check if it was ICA

    logger.info("Invalid inputs test completed.")

# --- Tests for helper functions in core.py ---

def test_update_ica_with_classifications(
    dummy_ica_data: mne.preprocessing.ICA
):
    """Test the _update_ica_with_classifications helper function."""
    logger.info("Testing _update_ica_with_classifications...")
    ica = dummy_ica_data.copy()
    n_comps = ica.n_components_

    # Create sample classification results
    results_data = []
    for i in range(n_comps):
        label = COMPONENT_LABELS[i % len(COMPONENT_LABELS)]
        results_data.append({
            'component_index': i,
            'label': label,
            'confidence': 0.9, 
            'reason': 'test reason',
            'exclude_vision': label != 'brain' # Exclude if not brain
        })
    results_df = pd.DataFrame(results_data)
    results_df = results_df.set_index('component_index', drop=False)

    ica_updated = _update_ica_with_classifications(ica, results_df)

    assert ica_updated is not ica, "Should return a new ICA object (or a copy)"
    assert hasattr(ica_updated, 'labels_'), "Updated ICA should have 'labels_' attribute"
    assert hasattr(ica_updated, 'labels_scores_'), "Updated ICA should have 'labels_scores_' attribute"
    assert hasattr(ica_updated, 'exclude'), "Updated ICA should have 'exclude' attribute"

    expected_excluded_count = sum(1 for lbl in results_df['label'] if lbl != 'brain')
    assert len(ica_updated.exclude) == expected_excluded_count, "Mismatch in excluded components count"

    # Check labels_scores_ shape and content (basic check)
    assert ica_updated.labels_scores_.shape == (n_comps, len(COMPONENT_LABELS))
    for i in range(n_comps):
        label_idx = COMPONENT_LABELS.index(results_df.loc[i, 'label'])
        assert ica_updated.labels_scores_[i, label_idx] == pytest.approx(0.9)
    
    logger.info("_update_ica_with_classifications test completed.")


def test_apply_artifact_rejection(
    dummy_raw_data: mne.io.Raw,
    dummy_ica_data: mne.preprocessing.ICA
):
    """Test the _apply_artifact_rejection helper function."""
    logger.info("Testing _apply_artifact_rejection...")
    raw = dummy_raw_data.copy()
    ica = dummy_ica_data.copy()

    # Scenario 1: No components excluded
    ica.exclude = []
    raw_cleaned_none_excluded = _apply_artifact_rejection(raw.copy(), ica)
    assert np.allclose(raw.get_data(), raw_cleaned_none_excluded.get_data()), \
        "Data should be unchanged if no components are excluded"

    # Scenario 2: Exclude one component
    if ica.n_components_ > 0:
        ica.exclude = [0] # Exclude the first component
        raw_cleaned_one_excluded = _apply_artifact_rejection(raw.copy(), ica)
        # Data should change if a component is applied (unless it has zero contribution)
        # This is a basic check; exact data change depends on the component
        assert not np.allclose(raw.get_data(), raw_cleaned_one_excluded.get_data()), \
            "Data should change if a component is excluded (assuming non-zero component)"
    else:
        logger.warning("Skipping part of _apply_artifact_rejection test due to 0 ICA components.")

    logger.info("_apply_artifact_rejection test completed.")


# --- More specific scenarios for label_components ---

@patch('icvision.core.classify_components_batch')
def test_label_components_custom_params(
    mock_classify_batch_api: MagicMock,
    dummy_raw_data: mne.io.Raw,
    dummy_ica_data: mne.preprocessing.ICA,
    mock_openai_classify_success,
    temp_test_dir: Path
):
    """Test label_components with custom parameters like threshold, model, etc."""
    logger.info("Testing label_components with custom parameters...")
    mock_classify_batch_api.side_effect = mock_openai_classify_success
    
    custom_output_dir = temp_test_dir / "custom_params_output"
    custom_output_dir.mkdir(exist_ok=True)
    
    custom_labels_to_exclude = ["eye", "muscle"]
    custom_confidence = 0.75
    custom_model = "gpt-4-turbo"
    custom_batch_size = 3

    _, _, results_df = label_components(
        raw_data=dummy_raw_data,
        ica_data=dummy_ica_data,
        api_key="FAKE_KEY",
        output_dir=custom_output_dir,
        generate_report=False,
        confidence_threshold=custom_confidence,
        labels_to_exclude=custom_labels_to_exclude,
        model_name=custom_model,
        batch_size=custom_batch_size
    )

    # Check if classify_components_batch was called with custom params
    call_args = mock_classify_batch_api.call_args[1] # Get kwargs of the call
    assert call_args['confidence_threshold'] == custom_confidence
    assert call_args['labels_to_exclude'] == custom_labels_to_exclude
    assert call_args['model_name'] == custom_model
    assert call_args['batch_size'] == custom_batch_size
    
    # Check results based on custom exclusion (simplified)
    # This depends on the mock_openai_classify_success logic
    for _, row in results_df.iterrows():
        should_exclude = row['label'] in custom_labels_to_exclude and row['confidence'] >= custom_confidence
        # The mock always uses 0.95 confidence, so if label is in custom_labels_to_exclude, it should be excluded
        if row['label'] in custom_labels_to_exclude:
             assert row['exclude_vision'] == True, f"Component {row['component_index']} with label {row['label']} should be excluded"
        elif row['label'] == 'brain': # Brain is never in custom_labels_to_exclude by default
             assert row['exclude_vision'] == False, f"Brain component {row['component_index']} should not be excluded"

    logger.info("Custom parameters test completed.")


# Add more tests as needed: e.g., different file types for raw/ica, empty raw/ica, etc. 