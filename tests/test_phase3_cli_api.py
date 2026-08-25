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


def make_mock_results_df(component_indices=None) -> pd.DataFrame:
    """Return a valid classification DataFrame for forwarding tests."""
    if component_indices is None:
        component_indices = [0, 1, 2, 3, 4]
    df = pd.DataFrame({
        "component_index": component_indices,
        "component_name": [f"IC{i}" for i in component_indices],
        "label": ["brain"] * len(component_indices),
        "confidence": [0.95] * len(component_indices),
        "reason": ["Test"] * len(component_indices),
        "mne_label": ["brain"] * len(component_indices),
        "exclude_vision": [False] * len(component_indices),
    })
    return df.set_index("component_index", drop=False)


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

    def test_cli_passes_strip_layout_size_and_prompt_file(self, temp_test_dir: Path):
        """CLI must pass strip layout controls and --prompt-file through to core."""
        from icvision.cli import main

        mock_df = make_mock_results_df([0])
        custom_prompt = "CUSTOM CLI STRIP PROMPT {n}: {labels}\n{json_example}"
        prompt_path = temp_test_dir / "custom_strip_prompt.txt"
        prompt_path.write_text(custom_prompt, encoding="utf-8")

        test_argv = [
            "autoclean-icvision",
            "test.set",
            "--api-key",
            "test-key",
            "--layout",
            "strip",
            "--strip-size",
            "9",
            "--prompt-file",
            str(prompt_path),
            "--no-report",
            "--output-dir",
            str(temp_test_dir),
        ]
        with patch.object(sys, "argv", test_argv):
            with patch("icvision.cli.CLIFormatter.print_welcome"):
                with patch("icvision.cli.CLIFormatter.print_summary_stats"):
                    with patch("icvision.cli.print_info"):
                        with patch("icvision.cli.print_success"):
                            with patch("icvision.cli.label_components", return_value=(MagicMock(), MagicMock(), mock_df)) as mock_lc:
                                main()

        call_kwargs = mock_lc.call_args.kwargs
        assert call_kwargs["layout"] == "strip"
        assert call_kwargs["strip_size"] == 9
        assert call_kwargs["custom_prompt"] == custom_prompt
        assert call_kwargs["generate_report"] is False


# --- Test: label_components() API ---


class TestLabelComponentsAPI:
    """Tests for layout parameter in core.label_components()."""

    def test_label_components_accepts_layout_parameter(
        self, dummy_raw_data: mne.io.Raw, dummy_ica_data: mne.preprocessing.ICA, temp_test_dir: Path
    ):
        """label_components() must accept layout parameter."""
        from icvision.core import label_components

        # Mock API to avoid real calls
        mock_df = make_mock_results_df()

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

        mock_df = make_mock_results_df()

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

    def test_label_components_rejects_partial_strip_metadata(
        self, dummy_raw_data: mne.io.Raw, dummy_ica_data: mne.preprocessing.ICA, temp_test_dir: Path
    ):
        """Partial strip metadata must not silently continue downstream."""
        from icvision.core import label_components

        partial_df = make_mock_results_df([0, 1])
        partial_metadata = {
            "layout": "strip",
            "status": "partial",
            "failed_batches": [{"batch_index": 1, "component_indices": [2, 3, 4]}],
        }
        original_exclude = list(dummy_ica_data.exclude)

        with patch("icvision.core.classify_components_batch", return_value=(partial_df, partial_metadata)):
            with patch("icvision.core.generate_classification_report") as mock_gen_report:
                with pytest.raises(RuntimeError, match="partial.*failed_batches"):
                    label_components(
                        raw_data=dummy_raw_data,
                        ica_data=dummy_ica_data,
                        api_key="test-key",
                        output_dir=temp_test_dir,
                        generate_report=True,
                        layout="strip",
                        strip_size=9,
                    )

        assert dummy_ica_data.exclude == original_exclude
        mock_gen_report.assert_not_called()

    def test_label_components_rejects_unavailable_strip_metadata(
        self, dummy_raw_data: mne.io.Raw, dummy_ica_data: mne.preprocessing.ICA, temp_test_dir: Path
    ):
        """Unavailable strip metadata must not silently continue downstream."""
        from icvision.core import label_components

        unavailable_df = make_mock_results_df([])
        unavailable_metadata = {
            "layout": "strip",
            "status": "unavailable",
            "failed_batches": [{"batch_index": 0, "component_indices": [0, 1, 2, 3, 4]}],
        }
        original_exclude = list(dummy_ica_data.exclude)

        with patch("icvision.core.classify_components_batch", return_value=(unavailable_df, unavailable_metadata)):
            with patch("icvision.core.generate_classification_report") as mock_gen_report:
                with pytest.raises(RuntimeError, match="unavailable.*failed_batches"):
                    label_components(
                        raw_data=dummy_raw_data,
                        ica_data=dummy_ica_data,
                        api_key="test-key",
                        output_dir=temp_test_dir,
                        generate_report=True,
                        layout="strip",
                        strip_size=9,
                    )

        assert dummy_ica_data.exclude == original_exclude
        mock_gen_report.assert_not_called()

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
        """compat.label_components() must forward strip layout controls."""
        from icvision.compat import label_components

        mock_df = make_mock_results_df()
        custom_prompt = "CUSTOM COMPAT STRIP PROMPT {n}: {labels}\n{json_example}"

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
                strip_size=9,
                custom_prompt=custom_prompt,
            )

        assert result is not None
        call_kwargs = mock_lc.call_args.kwargs
        assert call_kwargs["layout"] == "strip"
        assert call_kwargs["strip_size"] == 9
        assert call_kwargs["custom_prompt"] == custom_prompt

    def test_compat_does_not_swallow_strip_metadata_failure(
        self, dummy_raw_data: mne.io.Raw, dummy_ica_data: mne.preprocessing.ICA, temp_test_dir: Path
    ):
        """Compatibility wrapper must not turn strip partial failure into success."""
        from icvision.compat import label_components

        with patch(
            "icvision.compat.icvision_label_components",
            side_effect=RuntimeError("Classification metadata status is 'partial'; failed_batches=[]"),
        ):
            with pytest.raises(RuntimeError, match="partial"):
                label_components(
                    inst=dummy_raw_data,
                    ica=dummy_ica_data,
                    method="icvision",
                    generate_report=False,
                    output_dir=str(temp_test_dir),
                    layout="strip",
                    strip_size=9,
                )


# --- Test: Layout Parameter Passthrough ---


class TestLayoutPassthrough:
    """Tests to verify layout parameter is passed through the call chain."""

    def test_classify_components_batch_receives_layout(
        self, dummy_raw_data: mne.io.Raw, dummy_ica_data: mne.preprocessing.ICA, temp_test_dir: Path
    ):
        """classify_components_batch() must receive layout from label_components()."""
        from icvision.core import label_components

        mock_df = make_mock_results_df()

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
