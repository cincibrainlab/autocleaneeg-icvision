"""
Unit and integration tests for the core ICVision functionality.

This module tests the `label_components` function and its helper functions
within `src.icvision.core` to ensure robust and correct operation under various
conditions, including different input types, API interactions (mocked), and
output validation.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pandas as pd
import pytest
from _pytest.tmpdir import TempPathFactory

from icvision.config import (  # Removed DEFAULT_CONFIG, DEFAULT_EXCLUDE_LABELS
    COMPONENT_LABELS,
    ICVISION_TO_MNE_LABEL_MAP,
)
from icvision.core import (
    _apply_artifact_rejection,
    _update_ica_with_classifications,
    label_components,
)

# from icvision.utils import load_ica_data, load_raw_data # F401

# Configure logging for tests
logger = logging.getLogger("icvision_tests_core")
logger.setLevel(logging.DEBUG)  # Show detailed logs during testing

# --- Test Data Setup ---


# Create a fixture for a temporary test directory
@pytest.fixture(scope="module")  # type: ignore[misc]
def temp_test_dir(tmp_path_factory: TempPathFactory) -> Iterator[Path]:
    """Create a temporary directory for test artifacts."""
    tdir = tmp_path_factory.mktemp("icvision_core_tests")
    logger.info("Created temporary test directory: %s", tdir)
    yield tdir
    # No explicit shutil.rmtree(tdir) needed due to tmp_path_factory
    logger.info("Temporary test directory %s will be cleaned up.", tdir)


# Create a fixture for dummy raw data
@pytest.fixture(scope="module")  # type: ignore[misc]
def dummy_raw_data(temp_test_dir: Path) -> mne.io.Raw:
    """Generate a simple MNE Raw object for testing."""
    sfreq = 250
    n_channels = 5
    n_seconds = 10
    ch_names = [f"EEG {i:03}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    data = np.random.randn(n_channels, n_seconds * sfreq)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1020", on_missing="warn")  # Add a montage for plotting
    # Save to a file to also test file loading path
    raw_path = temp_test_dir / "dummy_raw.fif"
    raw.save(raw_path, overwrite=True)
    logger.debug("Created and saved dummy raw data to %s", raw_path)
    return raw


# Create a fixture for a dummy ICA object
@pytest.fixture(scope="module")  # type: ignore[misc]
def dummy_ica_data(dummy_raw_data: mne.io.Raw, temp_test_dir: Path) -> mne.preprocessing.ICA:
    """Generate a simple MNE ICA object for testing."""
    n_components = 3
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, max_iter="auto")
    ica.fit(dummy_raw_data)
    # Save to a file to also test file loading path
    ica_path = temp_test_dir / "dummy_ica.fif"
    ica.save(ica_path, overwrite=True)
    logger.debug("Created and saved dummy ICA data to %s", ica_path)
    return ica


# --- Mocked API Responses ---


@pytest.fixture  # type: ignore[misc]
def mock_openai_classify_success() -> Callable[..., pd.DataFrame]:
    """Mock a successful OpenAI API classification response."""

    # This mock function will be called by classify_components_batch
    # It needs to return a DataFrame similar to what classify_components_batch would produce.
    def mock_classify_batch(*args: Any, **kwargs: Any) -> pd.DataFrame:
        ica_obj = kwargs.get("ica_obj")
        assert ica_obj is not None, "ICA object must be provided to mock_classify_batch"
        n_comps = ica_obj.n_components_
        results = []
        for i in range(n_comps):
            label = COMPONENT_LABELS[i % len(COMPONENT_LABELS)]  # Cycle through labels
            results.append(
                {
                    "component_index": i,
                    "component_name": f"IC{i}",
                    "label": label,
                    "mne_label": label,  # Simplified for mock
                    "confidence": 0.95,
                    "reason": f"Mocked reason for {label} component IC{i}",
                    "exclude_vision": label != "brain",  # Exclude if not brain
                }
            )
        df = pd.DataFrame(results)
        df = df.set_index("component_index", drop=False)
        return df

    return mock_classify_batch


@pytest.fixture  # type: ignore[misc]
def mock_openai_classify_failure() -> Callable[..., Any]:
    """Mock a failing OpenAI API classification response."""

    def mock_classify_batch_fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Mocked API Error")

    return mock_classify_batch_fail


# --- Tests for label_components ---


@patch("icvision.core.classify_components_batch")
@patch("icvision.core.generate_classification_report")
def test_label_components_successful_run(
    mock_gen_report: MagicMock,
    mock_classify_batch_api: MagicMock,
    dummy_raw_data: mne.io.Raw,
    dummy_ica_data: mne.preprocessing.ICA,
    mock_openai_classify_success: Callable[..., pd.DataFrame],
    temp_test_dir: Path,
) -> None:
    """Test a full successful run of label_components with mocked API."""
    logger.info("Testing successful run of label_components...")
    mock_classify_batch_api.side_effect = mock_openai_classify_success

    raw_path = temp_test_dir / "dummy_raw.fif"
    ica_path = temp_test_dir / "dummy_ica.fif"

    raw_cleaned, ica_updated, results_df = label_components(
        raw_data=raw_path,  # Test with file paths
        ica_data=ica_path,
        api_key="FAKE_API_KEY",
        output_dir=temp_test_dir,
        generate_report=True,
    )

    assert isinstance(raw_cleaned, mne.io.Raw), "Cleaned raw should be an MNE Raw object"
    assert isinstance(ica_updated, mne.preprocessing.ICA), "Updated ICA should be an MNE ICA object"
    assert isinstance(results_df, pd.DataFrame), "Results should be a Pandas DataFrame"

    assert not results_df.empty, "Results DataFrame should not be empty"
    assert "label" in results_df.columns, "'label' column missing in results"
    assert "exclude_vision" in results_df.columns, "'exclude_vision' column missing"

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
        if any(r["label"] != "brain" for r in results_df.to_dict(orient="records")):
            assert len(ica_updated.exclude) > 0, "Expected some components to be excluded"

    logger.info("Successful run test completed.")


@patch("icvision.core.classify_components_batch")
def test_label_components_api_failure(
    mock_classify_batch_api: MagicMock,
    dummy_raw_data: mne.io.Raw,
    dummy_ica_data: mne.preprocessing.ICA,
    mock_openai_classify_failure: Callable[..., Any],
    temp_test_dir: Path,
) -> None:
    """Test label_components handling of API call failures."""
    logger.info("Testing API failure handling in label_components...")
    mock_classify_batch_api.side_effect = mock_openai_classify_failure

    with pytest.raises(RuntimeError, match="Failed to classify components: Mocked API Error"):
        label_components(
            raw_data=dummy_raw_data,  # Test with MNE objects
            ica_data=dummy_ica_data,
            api_key="FAKE_API_KEY",
            output_dir=temp_test_dir,
            generate_report=False,  # Disable report to isolate API error
        )
    logger.info("API failure handling test completed.")


@patch("icvision.core.classify_components_batch")
def test_label_components_no_report(
    mock_classify_batch_api: MagicMock,
    dummy_raw_data: mne.io.Raw,
    dummy_ica_data: mne.preprocessing.ICA,
    mock_openai_classify_success: Callable[..., pd.DataFrame],
    temp_test_dir: Path,
) -> None:
    """Test label_components with report generation disabled."""
    logger.info("Testing label_components with no report generation...")
    mock_classify_batch_api.side_effect = mock_openai_classify_success

    # Use a sub-directory for this test to avoid conflicts
    no_report_output_dir = temp_test_dir / "no_report_test"
    no_report_output_dir.mkdir(exist_ok=True)

    with patch("icvision.core.generate_classification_report") as mock_gen_report_local:
        label_components(
            raw_data=dummy_raw_data,
            ica_data=dummy_ica_data,
            api_key="FAKE_API_KEY",
            output_dir=no_report_output_dir,
            generate_report=False,
        )
        mock_gen_report_local.assert_not_called()

    # Check that report file does NOT exist
    # (Note: generate_classification_report itself creates the file, so if it's not called, file won't exist)
    # This test primarily ensures the function is not called.
    report_files = list(no_report_output_dir.glob("*.pdf"))
    assert len(report_files) == 0, f"PDF report was created in {no_report_output_dir} when generate_report=False"
    logger.info("No report generation test completed.")


@patch("icvision.core.classify_components_batch")  # Mock to prevent actual API calls
def test_label_components_invalid_inputs(mock_classify_api: MagicMock, temp_test_dir: Path) -> None:
    """Test label_components with various invalid inputs."""
    logger.info("Testing invalid inputs for label_components...")
    mock_classify_api.return_value = pd.DataFrame()  # Prevent issues if called

    # 1. Non-existent raw data file
    with pytest.raises(FileNotFoundError):
        label_components(raw_data="non_existent_raw.fif", ica_data="dummy_ica.fif", api_key="key")

    # 2. Non-existent ICA data file
    raw_dummy_file = temp_test_dir / "temp_raw.fif"
    mne.io.RawArray(np.random.rand(1, 100), mne.create_info(1, 100, "eeg")).save(raw_dummy_file, overwrite=True)
    with pytest.raises(FileNotFoundError):
        label_components(raw_data=raw_dummy_file, ica_data="non_existent_ica.fif", api_key="key")

    # 3. Missing API key (if not in env)
    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):  # Ensure env var is not set
        with pytest.raises(ValueError, match="No OpenAI API key provided"):
            label_components(
                raw_data=raw_dummy_file,
                ica_data=raw_dummy_file,
                api_key=None,  # dummy ica path
            )
    logger.info("Invalid inputs test completed.")


# --- Tests for helper functions in core.py ---


def test_update_ica_with_classifications(dummy_ica_data: mne.preprocessing.ICA) -> None:
    """Test updating an ICA object with classification results."""
    logger.info("Testing _update_ica_with_classifications...")
    ica_to_update = dummy_ica_data.copy()
    n_comps = ica_to_update.n_components_
    assert n_comps is not None, "Number of components is None"

    # Create sample classification results
    results_data = []
    for i in range(n_comps):
        label = COMPONENT_LABELS[i % len(COMPONENT_LABELS)]
        results_data.append(
            {
                "component_index": i,
                "label": label,
                "confidence": 0.9,
                "exclude_vision": (label != "brain"),  # Exclude if not brain
                # Add other necessary columns that _update_ica_with_classifications might expect
                "component_name": f"IC{i}",
                "mne_label": label,  # Simplified for this test
                "reason": "Test reason",
            }
        )
    results_df = pd.DataFrame(results_data).set_index("component_index")

    updated_ica = _update_ica_with_classifications(ica_to_update, results_df)

    assert updated_ica is ica_to_update, "ICA object should be updated in-place"
    assert hasattr(updated_ica, "labels_"), "ICA object should have 'labels_' attribute"
    assert hasattr(updated_ica, "labels_scores_"), "ICA object should have 'labels_scores_' attribute"
    assert updated_ica.labels_ is not None
    assert updated_ica.labels_scores_ is not None

    # Check if labels and scores are set correctly based on 'brain' vs other
    for i in range(n_comps):
        expected_label_for_comp = COMPONENT_LABELS[i % len(COMPONENT_LABELS)]
        actual_mne_label_assigned = ""
        for mne_label_cat, comp_indices in updated_ica.labels_.items():
            if i in comp_indices:
                actual_mne_label_assigned = mne_label_cat
                break

        # Check mapping using ICVISION_TO_MNE_LABEL_MAP
        expected_mne_label = ICVISION_TO_MNE_LABEL_MAP.get(expected_label_for_comp, "other")
        assert (
            actual_mne_label_assigned == expected_mne_label
        ), f"IC{i} expected {expected_mne_label}, got {actual_mne_label_assigned}"

        # Check scores (simplified check: score is 0.9 for its assigned OpenAI label)
        label_idx_in_openai_order = COMPONENT_LABELS.index(expected_label_for_comp)
        if updated_ica.labels_scores_ is not None:  # mypy check
            assert updated_ica.labels_scores_[i, label_idx_in_openai_order] == 0.9

    logger.info("_update_ica_with_classifications test completed.")


def test_apply_artifact_rejection(dummy_raw_data: mne.io.Raw, dummy_ica_data: mne.preprocessing.ICA) -> None:
    """Test applying artifact rejection to Raw data based on ICA exclude list."""
    logger.info("Testing _apply_artifact_rejection...")
    raw_to_clean = dummy_raw_data.copy()
    ica_with_exclusions = dummy_ica_data.copy()

    # Mark a component for exclusion (e.g., the first one)
    if ica_with_exclusions.n_components_ is not None and ica_with_exclusions.n_components_ > 0:
        ica_with_exclusions.exclude = [0]
    else:
        logger.warning("No components in dummy ICA to mark for exclusion. Test might be trivial.")
        ica_with_exclusions.exclude = []

    # Capture data before applying rejection to compare
    # Note: .apply() modifies data in-place for Raw, but returns a new Epochs object.
    # We are testing with Raw.
    data_before, _ = raw_to_clean[:, :]

    cleaned_raw = _apply_artifact_rejection(raw_to_clean, ica_with_exclusions)

    assert cleaned_raw is raw_to_clean, "Raw object should be modified in-place"

    if ica_with_exclusions.exclude and len(ica_with_exclusions.exclude) > 0:
        data_after, _ = cleaned_raw[:, :]
        # Check if data has changed (artifact removal should alter the data)
        assert not np.array_equal(data_before, data_after), "Data should change after applying ICA with exclusions"
    else:
        logger.info("No components were excluded, data should remain unchanged.")
        data_after, _ = cleaned_raw[:, :]
        assert np.array_equal(data_before, data_after), "Data should not change if no components are excluded"

    logger.info("_apply_artifact_rejection test completed.")


# --- More specific scenarios for label_components ---


@patch("icvision.core.classify_components_batch")
def test_label_components_custom_params(
    mock_classify_batch_api: MagicMock,
    dummy_raw_data: mne.io.Raw,
    dummy_ica_data: mne.preprocessing.ICA,
    mock_openai_classify_success: Callable[..., pd.DataFrame],
    temp_test_dir: Path,
) -> None:
    """Test label_components with custom model, prompt, and exclusion settings."""
    logger.info("Testing label_components with custom parameters...")
    mock_classify_batch_api.side_effect = mock_openai_classify_success

    custom_model = "gpt-custom-model"
    custom_prompt_text = "Classify this component: {component_image}"
    custom_exclude = ["eye", "muscle"]

    output_subdir = temp_test_dir / "custom_params_run"
    output_subdir.mkdir(exist_ok=True)

    _, _, results_df = label_components(
        raw_data=dummy_raw_data,
        ica_data=dummy_ica_data,
        api_key="FAKE_KEY",
        output_dir=output_subdir,
        model_name=custom_model,
        custom_prompt=custom_prompt_text,
        labels_to_exclude=custom_exclude,
        generate_report=False,  # Keep test focused on params
    )

    # Check if classify_components_batch was called with custom params
    mock_classify_batch_api.assert_called_once()
    call_args = mock_classify_batch_api.call_args[1]  # Get kwargs
    assert call_args.get("model_name") == custom_model
    assert call_args.get("custom_prompt") == custom_prompt_text
    assert call_args.get("labels_to_exclude") == custom_exclude

    # Check custom filenames
    assert (output_subdir / "icvision_results.csv").exists()
    assert (output_subdir / "icvision_classified_ica.fif").exists()
    assert (output_subdir / "classification_summary.txt").exists()

    # Check exclusion logic based on custom_exclude
    # Assuming mock_openai_classify_success labels components cyclically (brain, eye, muscle, ...)
    # and excludes if label is in custom_exclude AND is not 'brain' (implicit in mock)
    # For this to be robust, mock should be more explicit or results_df examined carefully
    if not results_df.empty:
        for _, row in results_df.iterrows():
            if row["label"] in custom_exclude and row["label"] != "brain":
                assert (
                    row["exclude_vision"] is True
                ), f"Component {row['component_index']} ({row['label']}) should be excluded"
            elif row["label"] == "brain":  # Brain is never in custom_exclude from DEFAULT_EXCLUDE_LABELS logic
                assert (
                    row["exclude_vision"] is False
                ), f"Component {row['component_index']} (brain) should not be excluded"

    logger.info("Custom parameters test completed.")


# Add more tests as needed: e.g., different file types for raw/ica, empty raw/ica, etc.
