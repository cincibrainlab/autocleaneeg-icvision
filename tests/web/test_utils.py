"""
Unit tests for ICVision web utilities.

This module tests the utility functions used by the web interface.
"""

import pytest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch


def test_validate_folder_path():
    """Test folder path validation."""
    # Import here to handle optional dependencies
    try:
        from icvision.web.utils import validate_folder_path
        from fastapi import HTTPException
        
        # Test with valid directory
        with tempfile.TemporaryDirectory() as temp_dir:
            result = validate_folder_path(temp_dir)
            assert isinstance(result, Path)
            assert result.exists()
            
        # Test with non-existent directory
        with pytest.raises(HTTPException) as exc_info:
            validate_folder_path("/nonexistent/path")
        assert exc_info.value.status_code == 400
        assert "does not exist" in str(exc_info.value.detail)
        
        # Test with file instead of directory
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(HTTPException) as exc_info:
                validate_folder_path(temp_file.name)
            assert exc_info.value.status_code == 400
            assert "not a directory" in str(exc_info.value.detail)
            
    except ImportError:
        pytest.skip("Web dependencies not available")


def test_validate_file_extension():
    """Test file extension validation."""
    try:
        from icvision.web.utils import validate_file_extension
        
        # Test valid extensions
        assert validate_file_extension(Path("test.set")) == True
        assert validate_file_extension(Path("test.fif")) == True
        assert validate_file_extension(Path("test.edf")) == True
        assert validate_file_extension(Path("test.csv")) == True
        
        # Test invalid extensions
        assert validate_file_extension(Path("test.txt")) == False
        assert validate_file_extension(Path("test.doc")) == False
        assert validate_file_extension(Path("test")) == False
        
        # Test case insensitivity
        assert validate_file_extension(Path("test.SET")) == True
        assert validate_file_extension(Path("test.FIF")) == True
        
    except ImportError:
        pytest.skip("Web dependencies not available")


def test_export_subject_override_csv():
    """Test CSV export functionality."""
    try:
        from icvision.web.utils import export_subject_override_csv
        
        # Create mock subject with override data
        mock_subject = Mock()
        mock_subject.subject_id = "test_subject"
        mock_subject.raw_file_path = "/tmp/test_subject.fif"
        
        # Create mock components with overrides
        mock_component1 = Mock()
        mock_component1.component_index = 0
        mock_component1.component_name = "IC0"
        mock_component1.override_label = "brain"
        mock_component1.override_confidence = 0.95
        mock_component1.override_reason = "Manual override"
        
        mock_component2 = Mock()
        mock_component2.component_index = 1
        mock_component2.component_name = "IC1"
        mock_component2.override_label = None  # No override
        
        mock_subject.components = [mock_component1, mock_component2]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update the mock path to use temp directory
            mock_subject.raw_file_path = str(Path(temp_dir) / "test_subject.fif")
            
            result = export_subject_override_csv(mock_subject)
            
            # Should return path since there are overrides
            assert result is not None
            assert "test_subject_icvis_results_override.csv" in result
            
            # Check that CSV file was created
            csv_path = Path(result)
            assert csv_path.exists()
            
            # Check CSV content
            df = pd.read_csv(csv_path)
            assert len(df) == 1  # Only component with override
            assert df.iloc[0]['component_index'] == 0
            assert df.iloc[0]['label'] == "brain"
            assert df.iloc[0]['confidence'] == 0.95
            
    except ImportError:
        pytest.skip("Web dependencies not available")


def test_export_subject_override_csv_no_overrides():
    """Test CSV export with no overrides."""
    try:
        from icvision.web.utils import export_subject_override_csv
        
        # Create mock subject with no overrides
        mock_subject = Mock()
        mock_subject.subject_id = "test_subject"
        mock_subject.raw_file_path = "/tmp/test_subject.fif"
        
        mock_component = Mock()
        mock_component.override_label = None
        mock_subject.components = [mock_component]
        
        result = export_subject_override_csv(mock_subject)
        
        # Should return None since no overrides
        assert result is None
        
    except ImportError:
        pytest.skip("Web dependencies not available")


@patch('icvision.web.utils.load_raw_data')
@patch('icvision.web.utils.load_ica_data')
def test_scan_folder_for_eeg_files(mock_load_ica, mock_load_raw):
    """Test folder scanning for EEG files."""
    try:
        from icvision.web.utils import scan_folder_for_eeg_files
        
        # Mock successful loading
        mock_load_raw.return_value = Mock()
        mock_load_ica.return_value = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "subject_001.set").touch()
            (temp_path / "subject_001.fif").touch()
            (temp_path / "subject_001_icvis_results.csv").touch()
            (temp_path / "invalid.txt").touch()  # Should be ignored
            
            result = scan_folder_for_eeg_files(temp_path)
            
            # Should find one subject
            assert len(result) == 1
            subject = result[0]
            assert subject.subject_id == "subject_001"
            assert subject.raw_file_path.endswith("subject_001.set")
            assert subject.ica_file_path.endswith("subject_001.fif")
            assert subject.results_file_path.endswith("subject_001_icvis_results.csv")
            
    except ImportError:
        pytest.skip("Web dependencies not available")


def test_create_mock_eeg_data():
    """Test mock EEG data generation for testing."""
    import numpy as np
    
    try:
        import mne
        from mne.preprocessing import ICA
        
        # Create minimal synthetic data
        n_channels = 10
        n_times = 1000
        sfreq = 250
        
        # Generate random data
        data = np.random.randn(n_channels, n_times) * 1e-6
        
        # Create info structure
        ch_names = [f'EEG{i:03d}' for i in range(1, n_channels + 1)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        
        # Create Raw object
        raw = mne.io.RawArray(data, info)
        
        # Run ICA
        ica = ICA(n_components=5, random_state=42)
        ica.fit(raw)
        
        # Verify objects were created
        assert raw.n_times == n_times
        assert len(raw.ch_names) == n_channels
        assert ica.n_components_ == 5
        
        return raw, ica
        
    except ImportError:
        pytest.skip("MNE not available for mock data generation")


def generate_mock_eeg_data(output_dir):
    """
    Generate mock EEG data for testing.
    
    Args:
        output_dir: Directory to save mock data files
    """
    try:
        raw, ica = test_create_mock_eeg_data()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save files
        raw_path = output_path / "subject_001_raw.fif"
        ica_path = output_path / "subject_001_ica.fif"
        
        raw.save(raw_path, overwrite=True)
        ica.save(ica_path, overwrite=True)
        
        # Create sample CSV
        csv_data = {
            'component_index': [0, 1, 2, 3, 4],
            'component_name': ['IC0', 'IC1', 'IC2', 'IC3', 'IC4'],
            'label': ['brain', 'eye', 'muscle', 'brain', 'other_artifact'],
            'confidence': [0.95, 0.88, 0.92, 0.87, 0.75],
            'reason': [
                'Clear brain activity',
                'Eye movement artifact',
                'Muscle artifact',
                'Neural activity',
                'Unclear artifact'
            ],
            'exclude_vision': [False, True, True, False, True]
        }
        
        csv_path = output_path / "subject_001_icvis_results.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        
        print(f"Mock data generated in {output_path}")
        return str(output_path)
        
    except ImportError:
        print("MNE not available - cannot generate mock data")
        return None


if __name__ == "__main__":
    # Generate mock data if run directly
    import sys
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "test_data/mock_data"
    
    generate_mock_eeg_data(output_dir)