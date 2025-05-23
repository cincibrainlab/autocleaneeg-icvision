"""
Unit tests for utility functions in ICVision.

This module tests the functions in `src.icvision.utils` to ensure they
handle various scenarios correctly, including file loading, input validation,
API key handling, output directory creation, results saving/formatting,
and classification results validation.
It uses mocked data and temporary directories for isolated testing.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd
import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.tmpdir import TempPathFactory

from icvision.config import COMPONENT_LABELS
from icvision.utils import (
    create_output_directory,
    format_summary_stats,
    load_ica_data,
    load_raw_data,
    save_results,
    validate_api_key,
    validate_classification_results,
    validate_inputs,
)

# Configure logging for tests
logger = logging.getLogger("icvision_tests_utils")
logger.setLevel(logging.DEBUG)

# --- Test Data Setup ---


@pytest.fixture(scope="module")  # type: ignore[misc]
def temp_utils_test_dir(tmp_path_factory: TempPathFactory) -> Iterator[Path]:
    """Create a temporary directory for utility function test artifacts."""
    tdir = tmp_path_factory.mktemp("icvision_utils_tests")
    logger.info(f"Created temporary utils test directory: {tdir}")
    yield tdir
    logger.info(f"Temporary utils test directory {tdir} will be cleaned up.")


@pytest.fixture(scope="module")  # type: ignore[misc]
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


@pytest.fixture(scope="module")  # type: ignore[misc]
def dummy_ica_object(dummy_raw_object: mne.io.Raw) -> mne.preprocessing.ICA:
    """Generate a simple MNE ICA object for direct use."""
    ica = mne.preprocessing.ICA(n_components=2, random_state=0, max_iter="auto")
    ica.fit(dummy_raw_object.copy().pick_types(eeg=True))  # Fit on EEG channels
    return ica


# --- Tests for load_raw_data ---


def test_load_raw_data_from_object(dummy_raw_object: mne.io.Raw) -> None:
    """Test loading raw data when an MNE Raw object is passed."""
    raw = load_raw_data(dummy_raw_object)
    assert raw is dummy_raw_object, "Should return the same object if Raw is passed"


def test_load_raw_data_from_fif(
    temp_utils_test_dir: Path, dummy_raw_object: mne.io.Raw
) -> None:
    """Test loading raw data from a .fif file."""
    raw_path = temp_utils_test_dir / "test_raw.fif"
    dummy_raw_object.save(raw_path, overwrite=True)

    loaded_raw = load_raw_data(raw_path)
    assert isinstance(loaded_raw, mne.io.BaseRaw), "Loaded data should be MNE Raw"
    assert len(loaded_raw.ch_names) == len(dummy_raw_object.ch_names)

    # Test loading from string path as well
    loaded_raw_str_path = load_raw_data(str(raw_path))
    assert isinstance(loaded_raw_str_path, mne.io.BaseRaw)


def test_load_raw_data_file_not_found() -> None:
    """Test loading raw data from a non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_raw_data(Path("non_existent_raw.fif"))


def test_load_raw_data_unsupported_format(temp_utils_test_dir: Path) -> None:
    """Test loading raw data from an unsupported file format."""
    unsupported_file = temp_utils_test_dir / "test.txt"
    unsupported_file.write_text("dummy content")
    with pytest.raises(ValueError, match="Unsupported file format: .txt"):
        load_raw_data(unsupported_file)


# Note: Testing EEGLAB .set would require a sample .set and .fdt file.
# For simplicity, we assume MNE's internal eeglab loader is tested by MNE.

# --- Tests for load_ica_data ---


def test_load_ica_data_from_object(dummy_ica_object: mne.preprocessing.ICA) -> None:
    """Test loading ICA data when an MNE ICA object is passed."""
    ica = load_ica_data(dummy_ica_object)
    assert ica is dummy_ica_object, "Should return the same object if ICA is passed"


def test_load_ica_data_from_fif(
    temp_utils_test_dir: Path, dummy_ica_object: mne.preprocessing.ICA
) -> None:
    """Test loading ICA data from a .fif file."""
    ica_path = temp_utils_test_dir / "test_ica.fif"
    dummy_ica_object.save(ica_path, overwrite=True)

    loaded_ica = load_ica_data(ica_path)
    assert isinstance(
        loaded_ica, mne.preprocessing.ICA
    ), "Loaded data should be MNE ICA"
    assert loaded_ica.n_components_ == dummy_ica_object.n_components_

    # Test loading from string path
    loaded_ica_str_path = load_ica_data(str(ica_path))
    assert isinstance(loaded_ica_str_path, mne.preprocessing.ICA)


def test_load_ica_data_file_not_found() -> None:
    """Test loading ICA data from a non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_ica_data(Path("non_existent_ica.fif"))


def test_load_ica_data_unsupported_format(temp_utils_test_dir: Path) -> None:
    """Test loading ICA data from an unsupported file format."""
    unsupported_file = temp_utils_test_dir / "test_ica.txt"
    unsupported_file.write_text("dummy ica content")
    with pytest.raises(ValueError, match="Unsupported ICA file format: .txt"):
        load_ica_data(unsupported_file)


# --- Tests for validate_inputs ---


def test_validate_inputs_compatible(
    dummy_raw_object: mne.io.Raw, dummy_ica_object: mne.preprocessing.ICA
) -> None:
    """Test input validation with compatible Raw and ICA objects."""
    # This should not raise any exception
    try:
        validate_inputs(dummy_raw_object, dummy_ica_object)
    except ValueError:
        pytest.fail(
            "validate_inputs raised ValueError unexpectedly for compatible inputs"
        )


def test_validate_inputs_ica_not_fitted(dummy_raw_object: mne.io.Raw) -> None:
    """Test input validation when ICA is not fitted."""
    unfitted_ica = mne.preprocessing.ICA(n_components=2)
    with pytest.raises(ValueError, match="ICA object appears to not be fitted"):
        validate_inputs(dummy_raw_object, unfitted_ica)


def test_validate_inputs_channel_mismatch(
    dummy_raw_object: mne.io.Raw,
    dummy_ica_object: mne.preprocessing.ICA,
    caplog: LogCaptureFixture,
) -> None:
    """Test input validation with channel mismatch (warning expected)."""
    # Create a raw object with different channels than ICA was fit on
    raw_mismatch = mne.io.RawArray(
        np.random.rand(dummy_raw_object.info["nchan"] + 1, 100),
        mne.create_info(dummy_raw_object.info["nchan"] + 1, 100, "eeg"),
    )
    # This should log a warning but not raise an error if we only check basics
    # Current validate_inputs only checks for channel count equality for a warning.
    # with pytest.warns(UserWarning, match="Channel count mismatch"):
    #      validate_inputs(raw_mismatch, dummy_ica_object)

    # Use caplog to check for logged warnings
    validate_inputs(raw_mismatch, dummy_ica_object)  # Call the function that should log
    assert any(
        "Channel count mismatch" in record.message and record.levelname == "WARNING"
        for record in caplog.records
    )
    # Optionally, clear the log if other tests in the same function might log warnings
    caplog.clear()


# --- Tests for create_output_directory ---


def test_create_output_directory_none(temp_utils_test_dir: Path) -> None:
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
        os.chdir(original_cwd)  # Change back CWD
        if (temp_utils_test_dir / default_dir_name).exists():
            shutil.rmtree(temp_utils_test_dir / default_dir_name)


def test_create_output_directory_specific_path(temp_utils_test_dir: Path) -> None:
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


def test_validate_api_key_provided() -> None:
    """Test API key validation when key is directly provided."""
    api_key = "test_key_123"
    assert validate_api_key(api_key) == api_key


@patch.dict(os.environ, {"OPENAI_API_KEY": "env_key_456"})
def test_validate_api_key_from_env() -> None:
    """Test API key validation when key is from environment variable."""
    assert validate_api_key(None) == "env_key_456"


@patch.dict(os.environ, {"OPENAI_API_KEY": ""})  # Ensure env var is empty or not set
def test_validate_api_key_missing() -> None:
    """Test API key validation when key is missing."""
    with pytest.raises(ValueError, match="No OpenAI API key provided"):
        validate_api_key(None)


# --- Tests for save_results ---


def test_save_results(temp_utils_test_dir: Path) -> None:
    """Test saving classification results to CSV."""
    results_data = [
        {
            "component": 0,
            "label": "brain",
            "confidence": 0.9,
            "reason": "Looks like brain",
        },
        {"component": 1, "label": "eye", "confidence": 0.8, "reason": "Looks like eye"},
    ]
    results_df = pd.DataFrame(results_data)
    output_filename = "test_results.csv"

    file_path = save_results(results_df, temp_utils_test_dir, output_filename)

    assert file_path.exists(), "CSV file was not created"
    assert file_path.name == output_filename, "CSV file has incorrect name"

    loaded_df = pd.read_csv(file_path)
    assert len(loaded_df) == len(results_df), "Saved CSV has incorrect number of rows"
    assert list(loaded_df.columns) == list(
        results_df.columns
    ), "Saved CSV has incorrect columns"


def test_save_results_empty(temp_utils_test_dir: Path) -> None:
    """Test saving empty classification results."""
    empty_df = pd.DataFrame()
    output_filename = "empty_results.csv"

    file_path = save_results(empty_df, temp_utils_test_dir, output_filename)
    assert file_path.exists(), "CSV file for empty results was not created"
    loaded_df = pd.read_csv(file_path)
    assert len(loaded_df) == 0, "Empty CSV should have zero rows"


# --- Tests for format_summary_stats ---


def test_format_summary_stats_empty() -> None:
    """Test formatting summary statistics with no results."""
    empty_df = pd.DataFrame(columns=["label", "exclude_vision"])
    summary = format_summary_stats(empty_df)
    assert "Total components classified: 0" in summary
    assert "Components marked for exclusion: 0" in summary
    for label in COMPONENT_LABELS:
        assert f"- {label.title()}: 0" in summary


def test_format_summary_stats_with_data() -> None:
    """Test formatting summary statistics with sample data."""
    results_data = [
        {"label": "brain", "exclude_vision": False, "confidence": 0.9, "reason": "-"},
        {"label": "eye", "exclude_vision": True, "confidence": 0.8, "reason": "-"},
        {"label": "muscle", "exclude_vision": True, "confidence": 0.85, "reason": "-"},
        {"label": "brain", "exclude_vision": False, "confidence": 0.95, "reason": "-"},
        {
            "label": "other_artifact",
            "exclude_vision": False,
            "confidence": 0.7,
            "reason": "-",
        },
    ]
    # Add dummy columns that might exist in a real df but are not used by format_summary_stats
    full_results_df = pd.DataFrame(results_data)
    full_results_df["component"] = range(len(results_data))
    full_results_df["mne_label"] = ["brain", "eog", "muscle", "brain", "other"]

    summary = format_summary_stats(full_results_df)

    assert "Total components classified: 5" in summary
    assert "Components marked for exclusion: 2" in summary
    assert "- Brain: 2" in summary
    assert "- Eye: 1" in summary
    assert "- Muscle: 1" in summary
    assert "- Heart: 0" in summary  # Assuming COMPONENT_LABELS includes heart
    assert "- Line_Noise: 0" in summary
    assert "- Channel_Noise: 0" in summary
    assert "- Other_Artifact: 1" in summary


# --- Tests for validate_classification_results ---


@pytest.fixture  # type: ignore[misc]
def valid_classification_df() -> pd.DataFrame:
    """Create a valid DataFrame for classification results."""
    data = {
        "component_index": [0, 1, 2],
        "component_name": ["IC0", "IC1", "IC2"],
        "label": ["brain", "eye", "muscle"],
        "mne_label": ["brain", "eog", "muscle"],
        "confidence": [0.9, 0.95, 0.88],
        "reason": ["Looks like brain", "Clear eye movements", "EMG activity"],
        "exclude_vision": [False, True, True],
    }
    return pd.DataFrame(data)


def test_validate_classification_results_valid(
    valid_classification_df: pd.DataFrame,
) -> None:
    """Test validation with a perfectly valid classification DataFrame."""
    try:
        validate_classification_results(valid_classification_df)
    except ValueError as e:
        pytest.fail(f"Validation failed for a valid DataFrame: {e}")


def test_validate_classification_results_missing_columns(
    valid_classification_df: pd.DataFrame,
) -> None:
    """Test validation when required columns are missing."""
    for col in ["label", "confidence", "exclude_vision"]:
        df_missing_col = valid_classification_df.drop(columns=[col])
        with pytest.raises(ValueError, match=f"Missing required column: {col}"):
            validate_classification_results(df_missing_col)


def test_validate_classification_results_invalid_label(
    valid_classification_df: pd.DataFrame,
) -> None:
    """Test validation with an invalid component label."""
    invalid_df = valid_classification_df.copy()
    invalid_df.loc[0, "label"] = "not_a_real_label"
    with pytest.raises(ValueError, match="Invalid label 'not_a_real_label' found"):
        validate_classification_results(invalid_df)


def test_validate_classification_results_invalid_confidence(
    valid_classification_df: pd.DataFrame,
) -> None:
    """Test validation with invalid confidence scores."""
    invalid_df_low = valid_classification_df.copy()
    invalid_df_low.loc[0, "confidence"] = -0.1
    with pytest.raises(
        ValueError, match="Confidence score -0.10 is outside the valid range"
    ):
        validate_classification_results(invalid_df_low)

    invalid_df_high = valid_classification_df.copy()
    invalid_df_high.loc[0, "confidence"] = 1.1
    with pytest.raises(
        ValueError, match="Confidence score 1.10 is outside the valid range"
    ):
        validate_classification_results(invalid_df_high)

    invalid_df_type = valid_classification_df.copy()
    invalid_df_type.loc[0, "confidence"] = "not_a_float"
    with pytest.raises(
        ValueError, match="Confidence score 'not_a_float' is not a float"
    ):
        validate_classification_results(invalid_df_type)
