"""
TDD Tests for Phase 2: Strip Layout Output Compatibility

These tests verify that strip layout classification produces outputs
compatible with existing pipeline functions:
1. DataFrame schema matches single-image mode
2. save_results() accepts strip-generated DataFrames
3. _update_ica_with_classifications() accepts strip results
4. Output file generation works correctly

Test-Driven Development: Tests written FIRST, then code fixed to pass.
"""

import logging
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pandas as pd
import pytest
from _pytest.tmpdir import TempPathFactory

logger = logging.getLogger("icvision_tests_strip")
logger.setLevel(logging.DEBUG)


# --- Fixtures ---


@pytest.fixture(scope="module")
def temp_test_dir(tmp_path_factory: TempPathFactory) -> Iterator[Path]:
    """Create a temporary directory for test artifacts."""
    tdir = tmp_path_factory.mktemp("icvision_strip_tests")
    yield tdir


@pytest.fixture(scope="module")
def dummy_raw_data(temp_test_dir: Path) -> mne.io.Raw:
    """Generate a simple MNE Raw object for testing."""
    sfreq = 250
    n_channels = 20  # Increased for 12 ICA components
    n_seconds = 20
    ch_names = [f"EEG {i:03}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    np.random.seed(42)
    data = np.random.randn(n_channels, n_seconds * sfreq) * 1e-6  # Realistic EEG scale
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1020", on_missing="ignore")
    # High-pass filter for ICA (avoid warning)
    raw.filter(l_freq=1.0, h_freq=None, verbose=False)
    return raw


@pytest.fixture(scope="module")
def dummy_ica_data(dummy_raw_data: mne.io.Raw) -> mne.preprocessing.ICA:
    """Generate a simple MNE ICA object for testing (12 components)."""
    n_components = 12  # Not divisible by 9 to test remainder handling
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, max_iter=200)
    ica.fit(dummy_raw_data, verbose=False)
    return ica


# --- Test: DataFrame Schema Parity ---


class TestDataFrameSchemaParity:
    """Tests to verify strip DataFrame matches single-image schema."""

    EXPECTED_COLUMNS = {
        "component_index",
        "component_name",
        "label",
        "confidence",
        "reason",
        "exclude_vision",
        "mne_label",
    }

    def test_strip_dataframe_has_required_columns(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """Strip batch results must have all columns expected by downstream functions."""
        from icvision.api import classify_components_strip_batch

        # Mock the API call to avoid real network requests
        mock_results = [
            {"component_idx": i, "label": "brain", "confidence": 0.95, "reason": "Test"}
            for i in range(dummy_ica_data.n_components_)
        ]

        with patch("icvision.api.classify_strip_image", return_value=mock_results[:9]):
            with patch("icvision.api.create_strip_image"):
                results_df, metadata = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(dummy_ica_data.n_components_)),
                    output_dir=temp_test_dir,
                )

        missing_columns = self.EXPECTED_COLUMNS - set(results_df.columns)
        assert not missing_columns, f"Strip DataFrame missing columns: {missing_columns}"

    def test_strip_dataframe_column_types(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """Strip DataFrame columns must have correct types for downstream compatibility."""
        from icvision.api import classify_components_strip_batch

        mock_results = [
            {"component_idx": i, "label": "brain", "confidence": 0.95, "reason": "Test"}
            for i in range(9)
        ]

        with patch("icvision.api.classify_strip_image", return_value=mock_results):
            with patch("icvision.api.create_strip_image"):
                results_df, _ = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(9)),
                    output_dir=temp_test_dir,
                )

        # Check types
        assert results_df["component_index"].dtype in [np.int64, np.int32, int]
        assert results_df["component_name"].dtype == object  # string
        assert results_df["label"].dtype == object  # string
        assert results_df["confidence"].dtype in [np.float64, float]
        assert results_df["exclude_vision"].dtype == bool

    def test_strip_dataframe_component_name_format(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """Component names must follow 'ICn' format."""
        from icvision.api import classify_components_strip_batch

        mock_results = [
            {"component_idx": i, "label": "brain", "confidence": 0.95, "reason": "Test"}
            for i in range(9)
        ]

        with patch("icvision.api.classify_strip_image", return_value=mock_results):
            with patch("icvision.api.create_strip_image"):
                results_df, _ = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(9)),
                    output_dir=temp_test_dir,
                )

        for idx, row in results_df.iterrows():
            expected_name = f"IC{row['component_index']}"
            assert row["component_name"] == expected_name, f"Expected {expected_name}, got {row['component_name']}"

    def test_strip_dataframe_index_is_component_index(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """DataFrame index must be component_index for _update_ica_with_classifications."""
        from icvision.api import classify_components_strip_batch

        mock_results = [
            {"component_idx": i, "label": "brain", "confidence": 0.95, "reason": "Test"}
            for i in range(9)
        ]

        with patch("icvision.api.classify_strip_image", return_value=mock_results):
            with patch("icvision.api.create_strip_image"):
                results_df, _ = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(9)),
                    output_dir=temp_test_dir,
                )

        # Index should be component_index values
        assert results_df.index.name == "component_index" or "component_index" in results_df.columns


# --- Test: Integration with save_results ---


class TestSaveResultsIntegration:
    """Tests to verify strip results work with save_results()."""

    def test_save_results_accepts_strip_dataframe(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """save_results() must accept strip-generated DataFrame without errors."""
        from icvision.api import classify_components_strip_batch
        from icvision.utils import save_results

        mock_results = [
            {"component_idx": i, "label": "brain", "confidence": 0.95, "reason": "Test"}
            for i in range(9)
        ]

        with patch("icvision.api.classify_strip_image", return_value=mock_results):
            with patch("icvision.api.create_strip_image"):
                results_df, _ = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(9)),
                    output_dir=temp_test_dir,
                )

        # This should not raise an exception
        output_path = save_results(results_df, temp_test_dir, input_basename="test")
        assert output_path.exists()

        # Verify CSV can be read back
        loaded_df = pd.read_csv(output_path)
        assert "component_index" in loaded_df.columns


# --- Test: Integration with _update_ica_with_classifications ---


class TestUpdateICAIntegration:
    """Tests to verify strip results work with _update_ica_with_classifications()."""

    def test_update_ica_accepts_strip_dataframe(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """_update_ica_with_classifications() must accept strip-generated DataFrame."""
        from icvision.api import classify_components_strip_batch
        from icvision.core import _update_ica_with_classifications

        mock_results = [
            {"component_idx": i, "label": "eye" if i == 0 else "brain", "confidence": 0.95, "reason": "Test"}
            for i in range(dummy_ica_data.n_components_)
        ]

        with patch("icvision.api.classify_strip_image", side_effect=[mock_results[:9], mock_results[9:]]):
            with patch("icvision.api.create_strip_image"):
                results_df, _ = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(dummy_ica_data.n_components_)),
                    output_dir=temp_test_dir,
                )

        # This should not raise an exception
        ica_updated = _update_ica_with_classifications(dummy_ica_data.copy(), results_df)

        # Verify ICA was updated
        assert hasattr(ica_updated, "labels_")
        assert hasattr(ica_updated, "labels_scores_")

    def test_update_ica_sets_exclusions_from_strip(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """Components marked exclude_vision=True must be added to ica.exclude."""
        from icvision.api import classify_components_strip_batch
        from icvision.core import _update_ica_with_classifications

        # Mark component 0 and 2 as eye (should be excluded)
        mock_results = [
            {"component_idx": i, "label": "eye" if i in [0, 2] else "brain", "confidence": 0.95, "reason": "Test"}
            for i in range(9)
        ]

        with patch("icvision.api.classify_strip_image", return_value=mock_results):
            with patch("icvision.api.create_strip_image"):
                results_df, _ = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(9)),
                    output_dir=temp_test_dir,
                    labels_to_exclude=["eye"],
                )

        ica_copy = dummy_ica_data.copy()
        ica_copy.exclude = []
        ica_updated = _update_ica_with_classifications(ica_copy, results_df)

        # Components 0 and 2 should be in exclude list
        assert 0 in ica_updated.exclude, "Component 0 (eye) should be excluded"
        assert 2 in ica_updated.exclude, "Component 2 (eye) should be excluded"


# --- Test: Remainder Handling ---


class TestRemainderHandling:
    """Tests for handling component counts not divisible by 9."""

    def test_remainder_batch_produces_valid_results(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """Final partial batch (< 9 components) must produce valid DataFrame rows."""
        from icvision.api import classify_components_strip_batch

        # 12 components: batch 1 = [0-8], batch 2 = [9-11]
        n_components = dummy_ica_data.n_components_  # 12

        mock_batch1 = [
            {"component_idx": i, "label": "brain", "confidence": 0.95, "reason": "Test"}
            for i in range(9)
        ]
        mock_batch2 = [
            {"component_idx": i, "label": "brain", "confidence": 0.95, "reason": "Test"}
            for i in range(9, n_components)
        ]

        with patch("icvision.api.classify_strip_image", side_effect=[mock_batch1, mock_batch2]):
            with patch("icvision.api.create_strip_image"):
                results_df, metadata = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(n_components)),
                    output_dir=temp_test_dir,
                )

        # Should have results for all 12 components
        assert len(results_df) == n_components
        assert metadata["n_batches"] == 2

        # All component indices should be present
        result_indices = set(results_df["component_index"].tolist())
        expected_indices = set(range(n_components))
        assert result_indices == expected_indices


# --- Test: MNE Label Mapping ---


class TestMNELabelMapping:
    """Tests for correct mne_label mapping."""

    def test_mne_label_column_present(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """mne_label column must be present for ICA update compatibility."""
        from icvision.api import classify_components_strip_batch

        mock_results = [
            {"component_idx": i, "label": "eye", "confidence": 0.95, "reason": "Test"}
            for i in range(9)
        ]

        with patch("icvision.api.classify_strip_image", return_value=mock_results):
            with patch("icvision.api.create_strip_image"):
                results_df, _ = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(9)),
                    output_dir=temp_test_dir,
                )

        assert "mne_label" in results_df.columns

    def test_mne_label_mapping_correct(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """mne_label must map correctly from icvision labels."""
        from icvision.api import classify_components_strip_batch
        from icvision.config import ICVISION_TO_MNE_LABEL_MAP

        mock_results = [
            {"component_idx": 0, "label": "eye", "confidence": 0.95, "reason": "Test"},
            {"component_idx": 1, "label": "brain", "confidence": 0.95, "reason": "Test"},
            {"component_idx": 2, "label": "muscle", "confidence": 0.95, "reason": "Test"},
        ]

        with patch("icvision.api.classify_strip_image", return_value=mock_results):
            with patch("icvision.api.create_strip_image"):
                results_df, _ = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=[0, 1, 2],
                    output_dir=temp_test_dir,
                )

        # Verify mappings
        for _, row in results_df.iterrows():
            expected_mne = ICVISION_TO_MNE_LABEL_MAP.get(row["label"], "other")
            assert row["mne_label"] == expected_mne


# --- Test: PDF Report Integration (Option A) ---


class TestPDFReportIntegration:
    """Tests to verify PDF report generation works with strip-mode DataFrames (Option A)."""

    def test_generate_report_accepts_strip_dataframe(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """generate_classification_report() must accept strip-generated DataFrame."""
        from icvision.api import classify_components_strip_batch
        from icvision.reports import generate_classification_report

        mock_results = [
            {"component_idx": i, "label": "eye" if i == 0 else "brain", "confidence": 0.95, "reason": "Test reason"}
            for i in range(3)
        ]

        with patch("icvision.api.classify_strip_image", return_value=mock_results):
            with patch("icvision.api.create_strip_image"):
                results_df, _ = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=[0, 1, 2],
                    output_dir=temp_test_dir,
                )

        # This should not raise an exception
        pdf_path = generate_classification_report(
            ica_obj=dummy_ica_data,
            raw_obj=dummy_raw_data,
            results_df=results_df,
            output_dir=temp_test_dir,
            input_basename="strip_test",
            components_to_detail="all",
        )

        assert pdf_path is not None
        assert pdf_path.exists()
        assert pdf_path.suffix == ".pdf"

    def test_generate_report_artifacts_only_with_strip_dataframe(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """PDF report 'artifacts_only' mode must work with strip DataFrame."""
        from icvision.api import classify_components_strip_batch
        from icvision.reports import generate_classification_report

        # Mark component 0 as eye (artifact), others as brain
        mock_results = [
            {"component_idx": 0, "label": "eye", "confidence": 0.95, "reason": "Blink pattern"},
            {"component_idx": 1, "label": "brain", "confidence": 0.90, "reason": "Neural"},
            {"component_idx": 2, "label": "brain", "confidence": 0.85, "reason": "Neural"},
        ]

        with patch("icvision.api.classify_strip_image", return_value=mock_results):
            with patch("icvision.api.create_strip_image"):
                results_df, _ = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=[0, 1, 2],
                    output_dir=temp_test_dir,
                    labels_to_exclude=["eye"],
                )

        # Verify exclude_vision is set correctly
        assert results_df.loc[0, "exclude_vision"] == True  # noqa: E712 (numpy bool)

        # Generate artifacts-only report
        pdf_path = generate_classification_report(
            ica_obj=dummy_ica_data,
            raw_obj=dummy_raw_data,
            results_df=results_df,
            output_dir=temp_test_dir,
            input_basename="strip_artifacts",
            components_to_detail="artifacts_only",
        )

        assert pdf_path is not None
        assert pdf_path.exists()
        assert "artifacts_only" in pdf_path.name
