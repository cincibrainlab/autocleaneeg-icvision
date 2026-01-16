"""
TDD Tests for Phase 4: Retry Logic with Exponential Backoff

Tests verify:
1. Failed batches are retried
2. Exponential backoff is applied between retries
3. Maximum retry count is respected
4. Successful retries produce valid results
5. All retries exhausted falls back appropriately
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

            result = classify_strip_image(
                image_path=mock_path,
                component_indices=list(range(9)),
                api_key="test-key",
                max_retries=3,
            )

        # Should have tried exactly max_retries times (3)
        assert call_count[0] == 3
        # Should return empty list on complete failure
        assert result == []


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


# --- Test: Batch Integration ---


class TestBatchRetryIntegration:
    """Tests for retry logic in batch processing."""

    def test_strip_batch_retries_failed_batches(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """classify_components_strip_batch() should retry failed batches."""
        from icvision.api import classify_components_strip_batch

        # First batch succeeds, second batch fails then succeeds on retry
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


# --- Test: Fallback on Exhausted Retries ---


class TestFallbackBehavior:
    """Tests for fallback behavior when all retries exhausted."""

    def test_fallback_to_other_artifact_on_exhausted_retries(
        self, dummy_ica_data: mne.preprocessing.ICA, dummy_raw_data: mne.io.Raw, temp_test_dir: Path
    ):
        """Components should be labeled 'other_artifact' when all retries fail."""
        from icvision.api import classify_components_strip_batch

        # All classification attempts fail
        with patch("icvision.api.classify_strip_image", return_value=[]):
            with patch("icvision.api.create_strip_image"):
                results_df, metadata = classify_components_strip_batch(
                    ica_obj=dummy_ica_data,
                    raw_obj=dummy_raw_data,
                    api_key="test-key",
                    component_indices=list(range(9)),
                    output_dir=temp_test_dir,
                )

        # All components should be labeled as other_artifact
        assert len(results_df) == 9
        assert all(results_df["label"] == "other_artifact")
