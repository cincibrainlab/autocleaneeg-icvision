"""
Unit tests for utility functions in ICVision.

This module tests the functions in `src.icvision.utils` to ensure they
handle various scenarios correctly, including file loading, input validation,
API key handling, output directory creation, results saving/formatting, and classification results validation.
It uses mocked data and temporary directories for isolated testing.
"""

import logging
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd
import pytest

from icvision.utils import (
    load_raw_data,
    load_ica_data,
    validate_inputs,
    create_output_directory,
    validate_api_key,
    save_results,
    format_summary_stats,
    validate_classification_results,
)
from icvision.config import COMPONENT_LABELS

# Configure logging for tests
logger = logging.getLogger("icvision_tests_utils")
logger.setLevel(logging.DEBUG)

# --- Test Data Setup ---

@pytest.fixture(scope="module")
def temp_utils_test_dir(tmp_path_factory):
    """Create a temporary directory for utility function test artifacts."""
    tdir = tmp_path_factory.mktemp("icvision_utils_tests")
    logger.info(f"Created temporary utils test directory: {tdir}")
    yield tdir
    logger.info(f"Temporary utils test directory {tdir} will be cleaned up.")

@pytest.fixture(scope="module")
def dummy_raw_object() -> mne.io.Raw:
    """Generate a simple MNE Raw object for direct use."""
    sfreq = 100
    n_channels = 3
    n_seconds = 5
    ch_names = [f"CH{i}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    data = np.random.randn(n_channels, n_seconds * sfreq)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return mne.io.RawArray(data, info)

@pytest.fixture(scope="module")
def dummy_ica_object(dummy_raw_object) -> mne.preprocessing.ICA:
    """Generate a simple MNE ICA object for direct use."""
    ica = mne.preprocessing.ICA(n_components=2, random_state=0, max_iter="auto")
    ica.fit(dummy_raw_object.copy().pick_types(eeg=True)) # Fit on EEG channels
    return ica

# --- Tests for load_raw_data ---

def test_load_raw_data_from_object(dummy_raw_object):
    """Test loading raw data when an MNE Raw object is passed."""
    raw = load_raw_data(dummy_raw_object)
    assert raw is dummy_raw_object, "Should return the same object if Raw is passed"

def test_load_raw_data_from_fif(temp_utils_test_dir, dummy_raw_object):
    """Test loading raw data from a .fif file."""
    raw_path = temp_utils_test_dir / "test_raw.fif"
    dummy_raw_object.save(raw_path, overwrite=True)
    
    loaded_raw = load_raw_data(raw_path)
    assert isinstance(loaded_raw, mne.io.BaseRaw), "Loaded data should be MNE Raw"
    assert len(loaded_raw.ch_names) == len(dummy_raw_object.ch_names)

    # Test loading from string path as well
    loaded_raw_str_path = load_raw_data(str(raw_path))
    assert isinstance(loaded_raw_str_path, mne.io.BaseRaw)

def test_load_raw_data_file_not_found():
    """Test loading raw data from a non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("non_existent_raw.fif")

def test_load_raw_data_unsupported_format(temp_utils_test_dir):
    """Test loading raw data from an unsupported file format."""
    unsupported_file = temp_utils_test_dir / "test.txt"
    unsupported_file.write_text("dummy content")
    with pytest.raises(ValueError, match="Unsupported file format: .txt"):
        load_raw_data(unsupported_file)

# Note: Testing EEGLAB .set would require a sample .set and .fdt file.
# For simplicity, we assume MNE's internal eeglab loader is tested by MNE.

# --- Tests for load_ica_data ---

def test_load_ica_data_from_object(dummy_ica_object):
    """Test loading ICA data when an MNE ICA object is passed."""
    ica = load_ica_data(dummy_ica_object)
    assert ica is dummy_ica_object, "Should return the same object if ICA is passed"

def test_load_ica_data_from_fif(temp_utils_test_dir, dummy_ica_object):
    """Test loading ICA data from a .fif file."""
    ica_path = temp_utils_test_dir / "test_ica.fif"
    dummy_ica_object.save(ica_path, overwrite=True)
    
    loaded_ica = load_ica_data(ica_path)
    assert isinstance(loaded_ica, mne.preprocessing.ICA), "Loaded data should be MNE ICA"
    assert loaded_ica.n_components_ == dummy_ica_object.n_components_

    # Test loading from string path
    loaded_ica_str_path = load_ica_data(str(ica_path))
    assert isinstance(loaded_ica_str_path, mne.preprocessing.ICA)

def test_load_ica_data_file_not_found():
    """Test loading ICA data from a non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_ica_data("non_existent_ica.fif")

def test_load_ica_data_unsupported_format(temp_utils_test_dir):
    """Test loading ICA data from an unsupported file format."""
    unsupported_file = temp_utils_test_dir / "test_ica.txt"
    unsupported_file.write_text("dummy ica content")
    with pytest.raises(ValueError, match="Unsupported ICA file format: .txt"):
        load_ica_data(unsupported_file)

# --- Tests for validate_inputs ---

def test_validate_inputs_compatible(dummy_raw_object, dummy_ica_object):
    """Test input validation with compatible Raw and ICA objects."""
    # This should not raise any exception
    try:
        validate_inputs(dummy_raw_object, dummy_ica_object)
    except ValueError:
        pytest.fail("validate_inputs raised ValueError unexpectedly for compatible inputs")

def test_validate_inputs_ica_not_fitted(dummy_raw_object):
    """Test input validation when ICA is not fitted."""
    unfitted_ica = mne.preprocessing.ICA(n_components=2)
    with pytest.raises(ValueError, match="ICA object appears to not be fitted"):
        validate_inputs(dummy_raw_object, unfitted_ica)

def test_validate_inputs_channel_mismatch(dummy_raw_object, dummy_ica_object):
    """Test input validation with channel mismatch (warning expected)."""
    # Create a raw object with different channels than ICA was fit on
    raw_mismatch = mne.io.RawArray(
        np.random.rand(dummy_raw_object.info['nchan'] + 1, 100),
        mne.create_info(dummy_raw_object.info['nchan'] + 1, 100, 'eeg')
    )
    # This should log a warning but not raise an error if we only check basics
    # Current validate_inputs only checks for channel count equality for a warning.
    with pytest.warns(UserWarning, match="Channel count mismatch"):
         validate_inputs(raw_mismatch, dummy_ica_object)

# --- Tests for create_output_directory ---

def test_create_output_directory_none(temp_utils_test_dir):
    """Test creating output directory when None is passed (should use default)."""
    default_dir_name = "icvision_results"
    # Temporarily change CWD for this test to isolate default dir creation
    original_cwd = Path.cwd()
    os.chdir(temp_utils_test_dir)
    try:
        output_path = create_output_directory(None)
        assert output_path.name == default_dir_name
        assert output_path.is_dir(), "Default output directory was not created"
        assert output_path.parent == temp_utils_test_dir
    finally:
        os.chdir(original_cwd) # Change back CWD
        if (temp_utils_test_dir / default_dir_name).exists():
            shutil.rmtree(temp_utils_test_dir / default_dir_name)

def test_create_output_directory_specific_path(temp_utils_test_dir):
    """Test creating output directory with a specific path."""
    specific_dir = temp_utils_test_dir / "my_custom_output"
    output_path = create_output_directory(specific_dir)
    assert output_path == specific_dir
    assert specific_dir.is_dir(), "Specific output directory was not created"

    # Test with string path
    specific_dir_str = str(temp_utils_test_dir / "my_custom_output_str")
    output_path_str = create_output_directory(specific_dir_str)
    assert output_path_str == Path(specific_dir_str)
    assert Path(specific_dir_str).is_dir()

# --- Tests for validate_api_key ---

def test_validate_api_key_provided():
    """Test API key validation when key is directly provided."""
    api_key = "test_key_123"
    assert validate_api_key(api_key) == api_key

@patch.dict(os.environ, {"OPENAI_API_KEY": "env_key_456"})
def test_validate_api_key_from_env():
    """Test API key validation when key is from environment variable."""
    assert validate_api_key(None) == "env_key_456"

@patch.dict(os.environ, {"OPENAI_API_KEY": ""}) # Ensure env var is empty or not set
def test_validate_api_key_missing():
    """Test API key validation when key is missing."""
    with pytest.raises(ValueError, match="No OpenAI API key provided"):
        validate_api_key(None)

# --- Tests for save_results ---

def test_save_results(temp_utils_test_dir):
    """Test saving classification results to CSV."""
    results_data = [
        {'component_index': 0, 'label': 'brain', 'confidence': 0.99, 'reason': 'Looks like brain'},
        {'component_index': 1, 'label': 'eye', 'confidence': 0.85, 'reason': 'Typical eye movements'}
    ]
    results_df = pd.DataFrame(results_data)
    output_dir = temp_utils_test_dir / "results_output"
    output_dir.mkdir(exist_ok=True)

    filename = "test_classification.csv"
    saved_path = save_results(results_df, output_dir, filename)

    assert saved_path.exists(), "Results CSV file was not created"
    assert saved_path.name == filename
    assert saved_path.parent == output_dir

    # Verify content
    loaded_df = pd.read_csv(saved_path)
    pd.testing.assert_frame_equal(loaded_df, results_df, check_dtype=False)

# --- Tests for format_summary_stats ---

def test_format_summary_stats_empty():
    """Test formatting summary stats for empty results."""
    empty_df = pd.DataFrame(columns=['component_index', 'label', 'confidence', 'reason', 'exclude_vision'])
    summary = format_summary_stats(empty_df)
    assert "No classification results available" in summary

def test_format_summary_stats_with_data():
    """Test formatting summary stats with sample classification data."""
    results_data = [
        {'component_index': 0, 'label': 'brain', 'confidence': 0.9, 'reason': 'r1', 'exclude_vision': False},
        {'component_index': 1, 'label': 'eye', 'confidence': 0.8, 'reason': 'r2', 'exclude_vision': True},
        {'component_index': 2, 'label': 'muscle', 'confidence': 0.7, 'reason': 'r3', 'exclude_vision': True},
        {'component_index': 3, 'label': 'brain', 'confidence': 0.95, 'reason': 'r4', 'exclude_vision': False},
    ]
    results_df = pd.DataFrame(results_data)
    summary = format_summary_stats(results_df)

    assert "Total components analyzed: 4" in summary
    assert "Components marked for exclusion: 2" in summary
    assert "brain           :   2 ( 50.0%)" in summary # Check formatting and counts
    assert "eye             :   1 ( 25.0%)" in summary
    assert "muscle          :   1 ( 25.0%)" in summary

# --- Tests for validate_classification_results ---

def test_validate_classification_results_valid():
    """Test validation of a correctly formatted results DataFrame."""
    valid_data = [
        {'component_index': 0, 'label': 'brain', 'confidence': 0.9, 'reason': 'Valid reason'}
    ]
    valid_df = pd.DataFrame(valid_data)
    assert validate_classification_results(valid_df) is True

def test_validate_classification_results_missing_columns():
    """Test validation when required columns are missing."""
    invalid_df = pd.DataFrame([{'label': 'brain'}]) # Missing component_index, confidence, reason
    assert validate_classification_results(invalid_df) is False

def test_validate_classification_results_invalid_label():
    """Test validation with an invalid component label."""
    invalid_data = [
        {'component_index': 0, 'label': 'unknown_artifact', 'confidence': 0.9, 'reason': 'r'}
    ]
    invalid_df = pd.DataFrame(invalid_data)
    assert validate_classification_results(invalid_df) is False

def test_validate_classification_results_invalid_confidence():
    """Test validation with confidence outside the 0-1 range."""
    invalid_data_high = [
        {'component_index': 0, 'label': 'brain', 'confidence': 1.1, 'reason': 'r'}
    ]
    invalid_df_high = pd.DataFrame(invalid_data_high)
    assert validate_classification_results(invalid_df_high) is False

    invalid_data_low = [
        {'component_index': 0, 'label': 'eye', 'confidence': -0.1, 'reason': 'r'}
    ]
    invalid_df_low = pd.DataFrame(invalid_data_low)
    assert validate_classification_results(invalid_df_low) is False 