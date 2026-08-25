"""
TDD Tests for Phase 4: Retry Logic with Exponential Backoff

Tests verify:
1. Failed batches are retried
2. Exponential backoff is applied between retries
3. Maximum retry count is respected
4. Successful retries produce valid results
5. All retries exhausted fails closed without synthetic labels
"""

import time
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch, call

import mne
import numpy as np
import pandas as pd
import pytest
from _pytest.tmpdir import TempPathFactory


# --- Fixtures ---


@pytest.fixture(scope="module")
def temp_test_dir(tmp_path_factory: TempPathFactory) -> Iterator[Path]:
    """Create a temporary directory for test artifacts."""
    tdir = tmp_path_factory.mktemp("icvision_phase4_tests")
    yield tdir


@pytest.fixture(scope="module")
def dummy_raw_data(temp_test_dir: Path) -> mne.io.Raw:
    """Generate a simple MNE Raw object for testing."""
    sfreq = 250
    n_channels = 20
    n_seconds = 20
    ch_names = [f"EEG {i:03}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    np.random.seed(42)
    data = np.random.randn(n_channels, n_seconds * sfreq) * 1e-6
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1020", on_missing="ignore")
    raw.filter(l_freq=1.0, h_freq=None, verbose=False)
    return raw


@pytest.fixture(scope="module")
def dummy_ica_data(dummy_raw_data: mne.io.Raw) -> mne.preprocessing.ICA:
    """Generate a simple MNE ICA object for testing."""
    n_components = 12  # Test with batches
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, max_iter=200)
    ica.fit(dummy_raw_data, verbose=False)
    return ica


# --- Test: Retry Configuration ---


class TestRetryConfiguration:
    """Tests for retry configuration parameters."""

    def test_classify_strip_image_has_max_retries_param(self):
        """classify_strip_image() must accept max_retries parameter."""
        from icvision.api import classify_strip_image
        import inspect

        sig = inspect.signature(classify_strip_image)
        assert "max_retries" in sig.parameters, "max_retries parameter must exist"

    def test_classify_strip_image_default_retries_is_3(self):
        """Default max_retries should be 3."""
        from icvision.api import classify_strip_image
        import inspect

        sig = inspect.signature(classify_strip_image)
        max_retries_param = sig.parameters.get("max_retries")
        assert max_retries_param is not None
        assert max_retries_param.default == 3


# --- Test: Retry Behavior ---


class TestRetryBehavior:
    """Tests for retry logic implementation."""

    def test_retry_on_api_failure(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """Failed API calls should be retried."""
        import json
        from icvision.api import classify_strip_image

        # Mock returns JSON string (as _call_openai_api does)
        # Labels A-I map to indices 0-8
        success_json = json.dumps([
            {"component": chr(ord("A") + i), "label": "brain", "confidence": 0.95, "reason": "Test"}
            for i in range(9)
        ])

        call_count = [0]

        def mock_api_call(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("API temporarily unavailable")
            return success_json  # Return JSON string

        with patch("icvision.api._call_openai_api", side_effect=mock_api_call):
            # Create a mock image path
            mock_path = temp_test_dir / "test_strip.webp"
            mock_path.write_bytes(b"fake image data")

            result = classify_strip_image(
                image_path=mock_path,
                component_indices=list(range(9)),
                api_key="test-key",
                max_retries=3,
            )

        # Should have retried and eventually succeeded
        assert call_count[0] == 3
        assert len(result) == 9

    def test_max_retries_respected(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """Retry count should not exceed max_retries."""
        from icvision.api import classify_strip_image

        call_count = [0]

        def always_fail(*args, **kwargs):
            call_count[0] += 1
            raise Exception("Persistent failure")

        with patch("icvision.api._call_openai_api", side_effect=always_fail):
            mock_path = temp_test_dir / "test_strip2.webp"
            mock_path.write_bytes(b"fake image data")

            with pytest.raises(RuntimeError, match="All 3 API attempts failed"):
                classify_strip_image(
                    image_path=mock_path,
                    component_indices=list(range(9)),
                    api_key="test-key",
                    max_retries=3,
                )

        # Should have tried exactly max_retries times (3)
        assert call_count[0] == 3


# --- Test: Exponential Backoff ---


class TestExponentialBackoff:
    """Tests for exponential backoff timing."""

    def test_backoff_delays_increase(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """Retry delays should increase exponentially."""
        import json
        from icvision.api import classify_strip_image

        delays = []
        call_count = [0]
        last_call_time = [time.time()]

        # JSON string response
        success_json = json.dumps([{"component": "A", "label": "brain", "confidence": 0.9, "reason": "Ok"}])

        def track_timing(*args, **kwargs):
            current_time = time.time()
            if call_count[0] > 0:
                delays.append(current_time - last_call_time[0])
            last_call_time[0] = current_time
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("API error")
            return success_json

        with patch("icvision.api._call_openai_api", side_effect=track_timing):
            mock_path = temp_test_dir / "test_backoff.webp"
            mock_path.write_bytes(b"fake image data")

            classify_strip_image(
                image_path=mock_path,
                component_indices=[0],
                api_key="test-key",
                max_retries=3,
            )

        # Should have 2 delays (between retry 1-2 and 2-3)
        assert len(delays) == 2
        # Second delay should be longer than first (exponential backoff)
        # Allow some tolerance for timing variations
        assert delays[1] >= delays[0] * 1.5, f"Delays should increase: {delays}"


class TestStrictStripParsing:
    """Tests for strict lower-level strip response parsing."""

    def test_missing_required_key_raises(self, temp_test_dir: Path):
        """Strip responses must include component, label, confidence, and reason."""
        import json
        from icvision.api import classify_strip_image

        mock_path = temp_test_dir / "missing_key.webp"
        mock_path.write_bytes(b"fake image data")
        response = json.dumps([{"component": "A", "confidence": 0.95, "reason": "Missing label"}])

        with patch("icvision.api._call_openai_api", return_value=response):
            with pytest.raises(ValueError, match="missing required keys: label"):
                classify_strip_image(
                    image_path=mock_path,
                    component_indices=[0],
                    api_key="test-key",
                    max_retries=1,
                )

    def test_missing_reason_raises(self, temp_test_dir: Path):
        """Reason is required and must not default to an empty string."""
        import json
        from icvision.api import classify_strip_image

        mock_path = temp_test_dir / "missing_reason.webp"
        mock_path.write_bytes(b"fake image data")
        response = json.dumps([{"component": "A", "label": "brain", "confidence": 0.95}])

        with patch("icvision.api._call_openai_api", return_value=response):
            with pytest.raises(ValueError, match="missing required keys: reason"):
                classify_strip_image(
                    image_path=mock_path,
                    component_indices=[0],
                    api_key="test-key",
                    max_retries=1,
                )

    def test_non_string_reason_raises(self, temp_test_dir: Path):
        """Reason must be a string for auditability."""
        import json
        from icvision.api import classify_strip_image

        mock_path = temp_test_dir / "bad_reason.webp"
        mock_path.write_bytes(b"fake image data")
        response = json.dumps([{"component": "A", "label": "brain", "confidence": 0.95, "reason": 123}])

        with patch("icvision.api._call_openai_api", return_value=response):
            with pytest.raises(ValueError, match="Invalid reason"):
                classify_strip_image(
                    image_path=mock_path,
                    component_indices=[0],
                    api_key="test-key",
                    max_retries=1,
                )


# --- Test: Batch Integration ---


class TestBatchRetryIntegration:
    """Tests for retry logic in batch processing."""

    def test_strip_batch_retries_failed_batches(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """classify_components_strip_batch() should retry failed batches."""
        from icvision.api import classify_components_strip_batch

        # First batch succeeds once; second batch fails then succeeds on retry.
        batch_results = [
            [{"component_idx": i, "label": "brain", "confidence": 0.95, "reason": "Test"} for i in range(9)],
            [],  # First attempt at batch 2 fails
            [{"component_idx": i, "label": "brain", "confidence": 0.95, "reason": "Test"} for i in range(9, 12)],
        ]
        call_idx = [0]

        def mock_classify(*args, **kwargs):
            result = batch_results[call_idx[0]] if call_idx[0] < len(batch_results) else []
            call_idx[0] += 1
            return result

        with patch("icvision.api.classify_strip_image", side_effect=mock_classify):
            with patch("icvision.api.create_strip_image"):
                results_df, metadata = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(12)),
                    output_dir=temp_test_dir,
                )

        # All 12 components should have results
        assert len(results_df) == 12
        assert call_idx[0] == 3
        assert metadata["status"] == "complete"


# --- Test: Fail Closed on Exhausted Retries ---


class TestFailClosedBehavior:
    """Strip classification must fail closed on incomplete results."""

    def test_empty_batch_result_fails_closed(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """An empty classifier response must not synthesize labels."""
        from icvision.api import classify_components_strip_batch

        with patch("icvision.api.classify_strip_image", return_value=[]):
            with patch("icvision.api.create_strip_image"):
                results_df, metadata = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(9)),
                    output_dir=temp_test_dir,
        )

        assert results_df.empty
        assert metadata["status"] == "unavailable"
        assert metadata["failed_batches"][0]["component_indices"] == list(range(9))

    def test_failed_batch_attempts_are_capped_at_five(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """A failed strip batch must make at most five API attempts."""
        from icvision.api import classify_components_strip_batch

        call_count = [0]

        def empty_result(*args, **kwargs):
            call_count[0] += 1
            return []

        with patch("icvision.api.classify_strip_image", side_effect=empty_result):
            with patch("icvision.api.create_strip_image"):
                results_df, metadata = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(9)),
                    output_dir=temp_test_dir,
                )

        assert call_count[0] == 5
        assert results_df.empty
        assert metadata["status"] == "unavailable"
        assert metadata["failed_batches"][0]["attempts"] == 5

    def test_malformed_batch_result_fails_closed(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """Lower-layer malformed rows must not be converted into synthetic labels."""
        from icvision.api import classify_components_strip_batch

        malformed_results = [{"label": "brain", "confidence": 0.95, "reason": "missing component_idx"}]

        with patch("icvision.api.classify_strip_image", return_value=malformed_results):
            with patch("icvision.api.create_strip_image"):
                results_df, metadata = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(9)),
                    output_dir=temp_test_dir,
                )

        assert results_df.empty
        assert metadata["status"] == "unavailable"

    def test_partial_batch_result_fails_closed(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """Partial lower-layer results must not create placeholder rows."""
        from icvision.api import classify_components_strip_batch

        partial_results = [
            {"component_idx": i, "label": "brain", "confidence": 0.95, "reason": "Test"}
            for i in range(8)
        ]

        with patch("icvision.api.classify_strip_image", return_value=partial_results):
            with patch("icvision.api.create_strip_image"):
                results_df, metadata = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(9)),
                    output_dir=temp_test_dir,
                )

        assert results_df.empty
        assert metadata["status"] == "unavailable"

    def test_duplicate_batch_result_fails_closed(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """Duplicate lower-layer component results must not be accepted."""
        from icvision.api import classify_components_strip_batch

        duplicate_results = [
            {"component_idx": i, "label": "brain", "confidence": 0.95, "reason": "Test"}
            for i in range(8)
        ]
        duplicate_results.append(
            {"component_idx": 7, "label": "brain", "confidence": 0.95, "reason": "Duplicate"}
        )

        with patch("icvision.api.classify_strip_image", return_value=duplicate_results):
            with patch("icvision.api.create_strip_image"):
                results_df, metadata = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(9)),
                    output_dir=temp_test_dir,
                )

        assert results_df.empty
        assert metadata["status"] == "unavailable"

    def test_successful_prior_batches_are_preserved_when_later_batch_fails(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """A later failed batch must not erase or rerun earlier successful rows."""
        from icvision.api import classify_components_strip_batch

        first_batch = [
            {"component_idx": i, "label": "brain", "confidence": 0.95, "reason": "Test"}
            for i in range(9)
        ]
        calls = []

        def mock_classify(_path, batch_indices, *args, **kwargs):
            calls.append(list(batch_indices))
            if list(batch_indices) == list(range(9)):
                return first_batch
            return []

        with patch("icvision.api.classify_strip_image", side_effect=mock_classify):
            with patch("icvision.api.create_strip_image"):
                results_df, metadata = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(12)),
                    output_dir=temp_test_dir,
                )

        assert len(results_df) == 9
        assert list(results_df["component_index"]) == list(range(9))
        assert calls.count(list(range(9))) == 1
        assert calls.count([9, 10, 11]) == 5
        assert metadata["status"] == "partial"
        assert metadata["failed_batches"][0]["component_indices"] == [9, 10, 11]
