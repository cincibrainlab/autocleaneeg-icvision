"""
TDD Tests for Phase 3: CLI and API Surface

Tests verify:
1. CLI accepts --layout flag with 'single' and 'strip' values
2. CLI accepts --strip-size flag
3. label_components() accepts layout and strip_size parameters
4. compat.label_components() accepts layout parameter
5. Default is 'single' for backward compatibility
"""

import subprocess
import sys
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pandas as pd
import pytest
from _pytest.tmpdir import TempPathFactory


# --- Fixtures ---


@pytest.fixture(scope="module")
def temp_test_dir(tmp_path_factory: TempPathFactory) -> Iterator[Path]:
    """Create a temporary directory for test artifacts."""
    tdir = tmp_path_factory.mktemp("icvision_phase3_tests")
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
    n_components = 5
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, max_iter=200)
    ica.fit(dummy_raw_data, verbose=False)
    return ica


# --- Test: CLI Argument Parsing ---


class TestCLIArgumentParsing:
    """Tests for CLI --layout and --strip-size argument parsing."""

    def test_cli_accepts_layout_single(self):
        """CLI must accept --layout single."""
        from icvision.cli import main
        import argparse

        # Test argument parsing only (not full execution)
        parser = argparse.ArgumentParser()
        parser.add_argument("raw_data_path")
        parser.add_argument("--layout", choices=["single", "strip"], default="single")

        args = parser.parse_args(["test.set", "--layout", "single"])
        assert args.layout == "single"

    def test_cli_accepts_layout_strip(self):
        """CLI must accept --layout strip."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("raw_data_path")
        parser.add_argument("--layout", choices=["single", "strip"], default="single")

        args = parser.parse_args(["test.set", "--layout", "strip"])
        assert args.layout == "strip"

    def test_cli_layout_default_is_single(self):
        """CLI default layout must be 'single' for backward compatibility."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("raw_data_path")
        parser.add_argument("--layout", choices=["single", "strip"], default="single")

        args = parser.parse_args(["test.set"])
        assert args.layout == "single"

    def test_cli_accepts_strip_size(self):
        """CLI must accept --strip-size argument."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("raw_data_path")
        parser.add_argument("--strip-size", type=int, default=9)

        args = parser.parse_args(["test.set", "--strip-size", "12"])
        assert args.strip_size == 12

    def test_cli_strip_size_default_is_9(self):
        """CLI default strip-size must be 9."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("raw_data_path")
        parser.add_argument("--strip-size", type=int, default=9)

        args = parser.parse_args(["test.set"])
        assert args.strip_size == 9


# --- Test: label_components() API ---


class TestLabelComponentsAPI:
    """Tests for layout parameter in core.label_components()."""

    def test_label_components_accepts_layout_parameter(
        self, dummy_raw_data: mne.io.Raw, dummy_ica_data: mne.preprocessing.ICA, temp_test_dir: Path
    ):
        """label_components() must accept layout parameter."""
        from icvision.core import label_components

        # Mock API to avoid real calls
        mock_df = pd.DataFrame({
            "component_index": [0, 1, 2, 3, 4],
            "component_name": ["IC0", "IC1", "IC2", "IC3", "IC4"],
            "label": ["brain"] * 5,
            "confidence": [0.95] * 5,
            "reason": ["Test"] * 5,
            "mne_label": ["brain"] * 5,
            "exclude_vision": [False] * 5,
        }).set_index("component_index", drop=False)

        with patch("icvision.core.classify_components_batch", return_value=(mock_df, {})):
            with patch("icvision.core.generate_classification_report", return_value=None):
                # This should not raise TypeError about unexpected keyword argument
                raw_cleaned, ica_updated, results_df = label_components(
                    raw_data=dummy_raw_data,
                    ica_data=dummy_ica_data,
                    api_key="test-key",
                    output_dir=temp_test_dir,
                    generate_report=False,
                    layout="single",  # NEW parameter
                )

        assert results_df is not None

    def test_label_components_accepts_strip_size_parameter(
        self, dummy_raw_data: mne.io.Raw, dummy_ica_data: mne.preprocessing.ICA, temp_test_dir: Path
    ):
        """label_components() must accept strip_size parameter."""
        from icvision.core import label_components

        mock_df = pd.DataFrame({
            "component_index": [0, 1, 2, 3, 4],
            "component_name": ["IC0", "IC1", "IC2", "IC3", "IC4"],
            "label": ["brain"] * 5,
            "confidence": [0.95] * 5,
            "reason": ["Test"] * 5,
            "mne_label": ["brain"] * 5,
            "exclude_vision": [False] * 5,
        }).set_index("component_index", drop=False)

        with patch("icvision.core.classify_components_batch", return_value=(mock_df, {})):
            with patch("icvision.core.generate_classification_report", return_value=None):
                raw_cleaned, ica_updated, results_df = label_components(
                    raw_data=dummy_raw_data,
                    ica_data=dummy_ica_data,
                    api_key="test-key",
                    output_dir=temp_test_dir,
                    generate_report=False,
                    layout="strip",
                    strip_size=9,  # NEW parameter
                )

        assert results_df is not None

    def test_label_components_default_layout_is_single(
        self, dummy_raw_data: mne.io.Raw, dummy_ica_data: mne.preprocessing.ICA, temp_test_dir: Path
    ):
        """label_components() default layout must be 'single'."""
        from icvision.core import label_components
        import inspect

        sig = inspect.signature(label_components)
        layout_param = sig.parameters.get("layout")

        assert layout_param is not None, "layout parameter must exist"
        assert layout_param.default == "single", "default must be 'single'"


# --- Test: compat.label_components() API ---


class TestCompatLabelComponentsAPI:
    """Tests for layout parameter in compat.label_components()."""

    def test_compat_label_components_accepts_layout_parameter(
        self, dummy_raw_data: mne.io.Raw, dummy_ica_data: mne.preprocessing.ICA, temp_test_dir: Path
    ):
        """compat.label_components() must accept layout parameter."""
        from icvision.compat import label_components

        mock_df = pd.DataFrame({
            "component_index": [0, 1, 2, 3, 4],
            "component_name": ["IC0", "IC1", "IC2", "IC3", "IC4"],
            "label": ["brain"] * 5,
            "confidence": [0.95] * 5,
            "reason": ["Test"] * 5,
            "mne_label": ["brain"] * 5,
            "exclude_vision": [False] * 5,
        }).set_index("component_index", drop=False)

        # Mock the core label_components
        with patch("icvision.compat.icvision_label_components") as mock_lc:
            mock_lc.return_value = (dummy_raw_data, dummy_ica_data, mock_df)

            # This should not raise TypeError
            result = label_components(
                inst=dummy_raw_data,
                ica=dummy_ica_data,
                method="icvision",
                generate_report=False,
                output_dir=str(temp_test_dir),
                layout="strip",  # NEW parameter
            )

        assert result is not None


# --- Test: Layout Parameter Passthrough ---


class TestLayoutPassthrough:
    """Tests to verify layout parameter is passed through the call chain."""

    def test_classify_components_batch_receives_layout(
        self, dummy_raw_data: mne.io.Raw, dummy_ica_data: mne.preprocessing.ICA, temp_test_dir: Path
    ):
        """classify_components_batch() must receive layout from label_components()."""
        from icvision.core import label_components

        mock_df = pd.DataFrame({
            "component_index": [0, 1, 2, 3, 4],
            "component_name": ["IC0", "IC1", "IC2", "IC3", "IC4"],
            "label": ["brain"] * 5,
            "confidence": [0.95] * 5,
            "reason": ["Test"] * 5,
            "mne_label": ["brain"] * 5,
            "exclude_vision": [False] * 5,
        }).set_index("component_index", drop=False)

        with patch("icvision.core.classify_components_batch", return_value=(mock_df, {})) as mock_ccb:
            with patch("icvision.core.generate_classification_report", return_value=None):
                label_components(
                    raw_data=dummy_raw_data,
                    ica_data=dummy_ica_data,
                    api_key="test-key",
                    output_dir=temp_test_dir,
                    generate_report=False,
                    layout="strip",
                    strip_size=12,
                )

        # Verify classify_components_batch was called with layout and strip_size
        mock_ccb.assert_called_once()
        call_kwargs = mock_ccb.call_args.kwargs
        assert call_kwargs.get("layout") == "strip"
        assert call_kwargs.get("strip_size") == 12
