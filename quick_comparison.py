#!/usr/bin/env python3
"""
Quick comparison script for ICA classifiers.
Supports ICVision vs ICLabel, and optionally vs human labels from CSV/Excel.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import mne

# Add src to path for ICVision imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def load_human_labels(filepath: str) -> dict:
    """Load human labels from CSV or Excel file."""
    file_path = Path(filepath)
    
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported format: {file_path.suffix}")
    
    # Expected columns: component_index, label, confidence (optional)
    if 'component_index' in df.columns and 'label' in df.columns:
        labels = df.set_index('component_index')['label'].to_dict()
    else:
        # Try alternative format with IC0, IC1, etc. as columns
        component_cols = [col for col in df.columns if col.startswith('IC')]
        if component_cols:
            labels = {}
            for col in component_cols:
                comp_idx = int(col[2:])  # Extract number from IC0, IC1, etc.
                labels[comp_idx] = df[col].iloc[0]  # Assume first row has the labels
        else:
            raise ValueError("Could not parse human labels. Expected 'component_index' and 'label' columns or 'IC0', 'IC1', etc. columns")
    
    return labels

def quick_compare(raw_path: str, ica_path: str = None, human_labels_path: str = None):
    """
    Quick comparison between ICVision and ICLabel (and optionally human labels).
    
    Args:
        raw_path: Path to raw EEG data
        ica_path: Path to ICA data (optional if raw_path contains ICA)
        human_labels_path: Path to human labels CSV/Excel (optional)
    """
    print("🔬 Quick ICA Classifier Comparison")
    print("=" * 40)
    
    # Load data
    print(f"📁 Loading data from: {raw_path}")
    raw_path = Path(raw_path)
    
    if raw_path.suffix.lower() == '.set':
        raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose=False)
        if ica_path is None:
            ica = mne.preprocessing.read_ica_eeglab(raw_path)
        else:
            ica = mne.preprocessing.read_ica(ica_path)
    else:
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        ica = mne.preprocessing.read_ica(ica_path)
    
    # Prepare data
    raw.set_eeg_reference('average', projection=True, verbose=False)
    raw.apply_proj(verbose=False)
    raw_filtered = raw.copy().filter(1., 100., verbose=False)
    
    print(f"✓ Loaded: {raw.info['nchan']} channels, {ica.n_components_} components")
    
    results = {}
    
    # Run ICLabel
    print("\n🧠 Running MNE-ICALabel...")
    icalabel_success = False
    try:
        from mne_icalabel import label_components
        ica_copy = ica.copy()
        
        # Check ICA algorithm and warn if not ideal for ICLabel
        if hasattr(ica_copy, 'fit_params_') and ica_copy.fit_params_.get('algorithm') == 'imported_eeglab':
            print("⚠️  Warning: ICA was imported from EEGLAB. ICLabel works best with extended infomax.")
        
        icalabel_result = label_components(raw_filtered, ica_copy, method='iclabel')
        results['ICLabel'] = icalabel_result['labels']
        print(f"✓ ICLabel completed: {len(results['ICLabel'])} components classified")
        icalabel_success = True
    except ImportError as e:
        print(f"❌ MNE-ICALabel import failed: {e}")
        print("   💡 Install with: pip install mne-icalabel")
        results['ICLabel'] = None
    except Exception as e:
        print(f"❌ ICLabel classification failed: {e}")
        print(f"   Full error: {repr(e)}")
        # Try to provide more helpful error messages  
        error_str = str(e).lower()
        if 'ica' in error_str and ('algorithm' in error_str or 'infomax' in error_str):
            print("   💡 Tip: ICLabel works best with extended infomax ICA, but EEGLAB ICA should still work")
        elif 'reference' in error_str:
            print("   💡 Tip: ICLabel prefers common average reference, but will still classify")
        elif 'result' in error_str:
            print("   💡 ICLabel may have completed but returned unexpected format")
        results['ICLabel'] = None
    
    # Run ICVision (only if ICLabel succeeded)
    if icalabel_success:
        print("\n🤖 Running ICVision...")
        try:
            from icvision.compat import label_components
            ica_copy = ica.copy()
            # Disable ICVision PDF report during comparison  
            icvision_result = label_components(raw_filtered, ica_copy, method='icvision',
                                             generate_report=False, output_dir=None)
            results['ICVision'] = icvision_result['labels']
            print(f"✓ ICVision completed: {len(results['ICVision'])} components classified")
        except ImportError as e:
            print(f"❌ ICVision import failed: {e}")
            print("   💡 Make sure you're running from the ICVision directory")
            results['ICVision'] = None
        except Exception as e:
            print(f"❌ ICVision failed: {e}")
            error_str = str(e).lower()
            if 'api' in error_str and 'key' in error_str:
                print("   💡 Set your OpenAI API key: export OPENAI_API_KEY='your_key'")
            elif 'openai' in error_str:
                print("   💡 Check your OpenAI API key and internet connection")
            elif 'any' in error_str and 'not defined' in error_str:
                print("   💡 Import issue in compat module - checking fix...")
            results['ICVision'] = None
    else:
        print("\n🤖 Skipping ICVision (ICLabel failed)")
        results['ICVision'] = None
    
    # Load human labels if provided
    if human_labels_path:
        print(f"\n👤 Loading human labels from: {human_labels_path}")
        try:
            human_dict = load_human_labels(human_labels_path)
            # Convert to list format matching component order
            human_labels = [human_dict.get(i, 'unknown') for i in range(ica.n_components_)]
            results['Human'] = human_labels
            print(f"✓ Human labels loaded: {len(human_labels)} components")
        except Exception as e:
            print(f"❌ Failed to load human labels: {e}")
            results['Human'] = None
    
    # Compare results
    print("\n📊 COMPARISON RESULTS")
    print("=" * 40)
    
    # Create comparison table
    methods = [key for key, value in results.items() if value is not None]
    
    if len(methods) < 2:
        print("❌ Need at least 2 successful classifications to compare")
        return
    
    # Print component-by-component comparison
    print(f"{'Component':<10}", end="")
    for method in methods:
        print(f"{method:<15}", end="")
    print("Agreement")
    print("-" * (10 + 15 * len(methods) + 10))
    
    agreements = []
    for i in range(ica.n_components_):
        labels_for_component = []
        print(f"IC{i:02d}       ", end="")
        
        for method in methods:
            label = results[method][i] if i < len(results[method]) else 'N/A'
            print(f"{label:<15}", end="")
            labels_for_component.append(label)
        
        # Check agreement (all labels the same)
        unique_labels = set(labels_for_component)
        is_agreement = len(unique_labels) == 1 and 'N/A' not in unique_labels
        agreement_symbol = "✓" if is_agreement else "✗"
        print(f"{agreement_symbol}")
        
        agreements.append(is_agreement)
    
    # Summary statistics
    print("\n📈 SUMMARY STATISTICS")
    print("-" * 25)
    
    total_components = len(agreements)
    total_agreements = sum(agreements)
    agreement_rate = total_agreements / total_components if total_components > 0 else 0
    
    print(f"Total components: {total_components}")
    print(f"Full agreement: {total_agreements}")
    print(f"Agreement rate: {agreement_rate:.1%}")
    
    # Label distribution
    print(f"\n🏷️  LABEL DISTRIBUTION")
    print("-" * 20)
    
    all_labels = set()
    for method_labels in results.values():
        if method_labels:
            all_labels.update(method_labels)
    
    print(f"{'Label':<15}", end="")
    for method in methods:
        print(f"{method:<10}", end="")
    print()
    print("-" * (15 + 10 * len(methods)))
    
    for label in sorted(all_labels):
        print(f"{label:<15}", end="")
        for method in methods:
            count = results[method].count(label) if results[method] else 0
            print(f"{count:<10}", end="")
        print()
    
    # Pairwise agreement for multiple methods
    if len(methods) > 2:
        print(f"\n🤝 PAIRWISE AGREEMENT")
        print("-" * 20)
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                if results[method1] and results[method2]:
                    pairwise_agreements = [
                        results[method1][k] == results[method2][k] 
                        for k in range(min(len(results[method1]), len(results[method2])))
                    ]
                    pairwise_rate = sum(pairwise_agreements) / len(pairwise_agreements)
                    print(f"{method1} vs {method2}: {pairwise_rate:.1%}")

def main():
    """Main function with simple command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Quick comparison between ICA classifiers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare ICVision vs ICLabel on EEGLAB file
  python quick_comparison.py data/subject01.set
  
  # Include human labels from CSV
  python quick_comparison.py data/subject01.set --human labels.csv
  
  # Use separate ICA file
  python quick_comparison.py raw.fif --ica ica.fif --human human_labels.xlsx
        """
    )
    
    parser.add_argument('raw_data', help='Path to raw EEG data')
    parser.add_argument('--ica', help='Path to ICA data (if not in raw_data)')
    parser.add_argument('--human', help='Path to human labels (CSV or Excel)')
    
    args = parser.parse_args()
    
    try:
        quick_compare(args.raw_data, args.ica, args.human)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()