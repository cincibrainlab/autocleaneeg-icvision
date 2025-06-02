#!/usr/bin/env python3
"""
Comparison script for ICVision vs MNE-ICALabel classification results.

This script runs both ICVision and MNE-ICALabel on the same EEG data and 
compares their classification results with detailed analysis and visualizations.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import mne


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_and_prepare_data(raw_path: str, ica_path: Optional[str] = None) -> Tuple[mne.io.Raw, mne.preprocessing.ICA]:
    """
    Load and prepare EEG data and ICA for classification.
    
    Args:
        raw_path: Path to raw EEG data
        ica_path: Path to ICA data (optional if raw_path contains ICA)
        
    Returns:
        Tuple of (raw, ica) objects
    """
    logging.info(f"Loading raw data from: {raw_path}")
    
    # Load raw data
    raw_path = Path(raw_path)
    if raw_path.suffix.lower() == '.set':
        raw = mne.io.read_raw_eeglab(raw_path, preload=True)
    elif raw_path.suffix.lower() == '.fif':
        raw = mne.io.read_raw_fif(raw_path, preload=True)
    else:
        raise ValueError(f"Unsupported raw data format: {raw_path.suffix}")
    
    # Load ICA
    if ica_path is None:
        if raw_path.suffix.lower() == '.set':
            logging.info("Attempting to load ICA from EEGLAB .set file")
            try:
                ica = mne.preprocessing.read_ica_eeglab(raw_path)
            except Exception as e:
                raise RuntimeError(f"Could not load ICA from .set file: {e}")
        else:
            raise ValueError("ICA path required for non-.set files")
    else:
        logging.info(f"Loading ICA from: {ica_path}")
        ica_path = Path(ica_path)
        if ica_path.suffix.lower() == '.fif':
            ica = mne.preprocessing.read_ica(ica_path)
        elif ica_path.suffix.lower() == '.set':
            ica = mne.preprocessing.read_ica_eeglab(ica_path)
        else:
            raise ValueError(f"Unsupported ICA format: {ica_path.suffix}")
    
    # Prepare data for classification
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    raw_filtered = raw.copy().filter(1., 100.)
    
    logging.info(f"Data loaded: {raw.info['nchan']} channels, {ica.n_components_} ICA components")
    return raw_filtered, ica


def run_icalabel_classification(raw: mne.io.Raw, ica: mne.preprocessing.ICA) -> Dict:
    """
    Run MNE-ICALabel classification.
    
    Args:
        raw: Preprocessed raw data
        ica: ICA object
        
    Returns:
        Dictionary with classification results
    """
    try:
        from mne_icalabel import label_components
        logging.info("Running MNE-ICALabel classification...")
        
        # Reset ICA object to clean state
        ica_copy = ica.copy()
        
        result = label_components(raw, ica_copy, method='iclabel')
        
        return {
            'method': 'MNE-ICALabel',
            'labels': result['labels'],
            'y_pred_proba': result['y_pred_proba'],
            'ica_labels_scores_': ica_copy.labels_scores_.copy(),
            'ica_labels_': ica_copy.labels_.copy(),
            'success': True,
            'error': None
        }
        
    except ImportError:
        logging.error("MNE-ICALabel not available. Install with: pip install mne-icalabel")
        return {
            'method': 'MNE-ICALabel',
            'success': False,
            'error': 'MNE-ICALabel not installed'
        }
    except Exception as e:
        logging.error(f"MNE-ICALabel classification failed: {e}")
        return {
            'method': 'MNE-ICALabel',
            'success': False,
            'error': str(e)
        }


def run_icvision_classification(raw: mne.io.Raw, ica: mne.preprocessing.ICA) -> Dict:
    """
    Run ICVision classification using compatibility layer.
    
    Args:
        raw: Preprocessed raw data
        ica: ICA object
        
    Returns:
        Dictionary with classification results
    """
    try:
        # Add src to path for imports
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from icvision.compat import label_components
        
        logging.info("Running ICVision classification (compatibility mode)...")
        
        # Reset ICA object to clean state
        ica_copy = ica.copy()
        
        result = label_components(raw, ica_copy, method='icvision')
        
        return {
            'method': 'ICVision',
            'labels': result['labels'],
            'y_pred_proba': result['y_pred_proba'], 
            'ica_labels_scores_': ica_copy.labels_scores_.copy(),
            'ica_labels_': ica_copy.labels_.copy(),
            'success': True,
            'error': None
        }
        
    except Exception as e:
        logging.error(f"ICVision classification failed: {e}")
        return {
            'method': 'ICVision',
            'success': False,
            'error': str(e)
        }


def harmonize_labels(labels: List[str], method: str) -> List[str]:
    """
    Harmonize labels to common vocabulary for comparison.
    
    Args:
        labels: Original labels
        method: Classification method name
        
    Returns:
        Harmonized labels
    """
    # Common label mapping
    label_mapping = {
        'eye blink': 'eye',
        'muscle artifact': 'muscle', 
        'heart beat': 'heart',
        'line noise': 'line_noise',
        'channel noise': 'channel_noise',
        'other': 'other_artifact'
    }
    
    harmonized = []
    for label in labels:
        harmonized_label = label_mapping.get(label, label)
        harmonized.append(harmonized_label)
    
    return harmonized


def calculate_agreement_metrics(labels1: List[str], labels2: List[str], 
                              method1: str, method2: str) -> Dict:
    """
    Calculate agreement metrics between two classification methods.
    
    Args:
        labels1: Labels from first method
        labels2: Labels from second method
        method1: Name of first method
        method2: Name of second method
        
    Returns:
        Dictionary with agreement metrics
    """
    # Harmonize labels for fair comparison
    harmonized1 = harmonize_labels(labels1, method1)
    harmonized2 = harmonize_labels(labels2, method2)
    
    # Calculate metrics
    accuracy = accuracy_score(harmonized1, harmonized2)
    
    # Get unique labels
    all_labels = sorted(set(harmonized1 + harmonized2))
    
    # Confusion matrix
    cm = confusion_matrix(harmonized1, harmonized2, labels=all_labels)
    
    # Classification report
    try:
        class_report = classification_report(
            harmonized1, harmonized2, 
            labels=all_labels, 
            output_dict=True,
            zero_division=0
        )
    except:
        class_report = None
    
    # Component-wise agreement
    agreements = []
    for i, (l1, l2) in enumerate(zip(harmonized1, harmonized2)):
        agreements.append({
            'component': f'IC{i:02d}',
            'method1_label': l1,
            'method2_label': l2,
            'agreement': l1 == l2
        })
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'labels': all_labels,
        'classification_report': class_report,
        'component_agreements': agreements,
        'harmonized_labels1': harmonized1,
        'harmonized_labels2': harmonized2
    }


def create_comparison_visualizations(results1: Dict, results2: Dict, 
                                   agreement_metrics: Dict, output_dir: Path) -> None:
    """
    Create visualizations comparing the two classification methods.
    
    Args:
        results1: Results from first method
        results2: Results from second method
        agreement_metrics: Agreement analysis results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Label distribution comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count labels for each method
    labels1 = agreement_metrics['harmonized_labels1']
    labels2 = agreement_metrics['harmonized_labels2']
    
    unique_labels = agreement_metrics['labels']
    counts1 = [labels1.count(label) for label in unique_labels]
    counts2 = [labels2.count(label) for label in unique_labels]
    
    x = np.arange(len(unique_labels))
    width = 0.35
    
    ax1.bar(x - width/2, counts1, width, label=results1['method'], alpha=0.8)
    ax1.bar(x + width/2, counts2, width, label=results2['method'], alpha=0.8)
    ax1.set_xlabel('Component Labels')
    ax1.set_ylabel('Count')
    ax1.set_title('Label Distribution Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(unique_labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Confusion Matrix
    cm = agreement_metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels, ax=ax2)
    ax2.set_xlabel(f'{results2["method"]} Labels')
    ax2.set_ylabel(f'{results1["method"]} Labels')
    ax2.set_title('Confusion Matrix')
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'label_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Component-by-component agreement
    agreements = agreement_metrics['component_agreements']
    n_components = len(agreements)
    
    fig, ax = plt.subplots(figsize=(max(12, n_components * 0.5), 8))
    
    colors = ['green' if agree['agreement'] else 'red' for agree in agreements]
    components = [agree['component'] for agree in agreements]
    
    y_pos = np.arange(len(components))
    
    # Create horizontal bar chart showing agreements/disagreements
    bars = ax.barh(y_pos, [1] * len(components), color=colors, alpha=0.7)
    
    # Add labels showing what each method predicted
    for i, agree in enumerate(agreements):
        method1_label = agree['method1_label']
        method2_label = agree['method2_label']
        ax.text(0.5, i, f'{method1_label} | {method2_label}', 
                ha='center', va='center', fontweight='bold', fontsize=9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(components)
    ax.set_xlabel('Agreement')
    ax.set_title(f'Component-by-Component Agreement\n'
                f'{results1["method"]} vs {results2["method"]}')
    ax.set_xlim(0, 1)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='Agreement'),
                      Patch(facecolor='red', alpha=0.7, label='Disagreement')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'component_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Probability comparison (if available)
    if results1['success'] and results2['success']:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Max probabilities comparison
        proba1 = results1['y_pred_proba']
        proba2 = results2['y_pred_proba']
        
        x = np.arange(len(proba1))
        
        axes[0].plot(x, proba1, 'o-', label=results1['method'], alpha=0.7)
        axes[0].plot(x, proba2, 's-', label=results2['method'], alpha=0.7)
        axes[0].set_xlabel('Component Index')
        axes[0].set_ylabel('Max Probability')
        axes[0].set_title('Maximum Classification Probability Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Probability difference
        prob_diff = np.array(proba1) - np.array(proba2)
        colors = ['red' if diff < 0 else 'blue' for diff in prob_diff]
        
        axes[1].bar(x, prob_diff, color=colors, alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Component Index')
        axes[1].set_ylabel(f'{results1["method"]} - {results2["method"]} Probability')
        axes[1].set_title('Probability Difference (Positive = ICVision Higher)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'probability_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Visualizations saved to: {output_dir}")


def generate_comparison_report(results1: Dict, results2: Dict, 
                             agreement_metrics: Dict, output_dir: Path) -> None:
    """
    Generate a detailed comparison report.
    
    Args:
        results1: Results from first method
        results2: Results from second method  
        agreement_metrics: Agreement analysis results
        output_dir: Directory to save report
    """
    report_path = output_dir / 'comparison_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ICA Component Classification Comparison Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Methods compared
        f.write("Methods Compared:\n")
        f.write(f"- {results1['method']}: {'[SUCCESS]' if results1['success'] else '[FAILED]'}\n")
        f.write(f"- {results2['method']}: {'[SUCCESS]' if results2['success'] else '[FAILED]'}\n\n")
        
        if not (results1['success'] and results2['success']):
            f.write("Comparison could not be completed due to classification failures.\n")
            if not results1['success']:
                f.write(f"{results1['method']} error: {results1['error']}\n")
            if not results2['success']:
                f.write(f"{results2['method']} error: {results2['error']}\n")
            return
        
        # Overall agreement
        f.write(f"Overall Agreement: {agreement_metrics['accuracy']:.3f} "
                f"({agreement_metrics['accuracy']*100:.1f}%)\n\n")
        
        # Label distribution
        f.write("Label Distribution:\n")
        f.write("-" * 20 + "\n")
        labels = agreement_metrics['labels']
        labels1 = agreement_metrics['harmonized_labels1']
        labels2 = agreement_metrics['harmonized_labels2']
        
        f.write(f"{'Label':<15} {results1['method']:<12} {results2['method']:<12} {'Difference':<10}\n")
        f.write("-" * 60 + "\n")
        
        for label in labels:
            count1 = labels1.count(label)
            count2 = labels2.count(label)
            diff = count1 - count2
            f.write(f"{label:<15} {count1:<12} {count2:<12} {diff:+d}\n")
        
        f.write("\n")
        
        # Component-by-component breakdown
        f.write("Component-by-Component Analysis:\n")
        f.write("-" * 35 + "\n")
        f.write(f"{'Component':<10} {results1['method']:<15} {results2['method']:<15} {'Agreement':<10}\n")
        f.write("-" * 60 + "\n")
        
        for agree in agreement_metrics['component_agreements']:
            status = "[AGREE]" if agree['agreement'] else "[DISAGREE]"
            f.write(f"{agree['component']:<10} {agree['method1_label']:<15} "
                   f"{agree['method2_label']:<15} {status:<10}\n")
        
        f.write("\n")
        
        # Classification metrics
        if agreement_metrics['classification_report']:
            f.write("Detailed Classification Metrics:\n")
            f.write("-" * 35 + "\n")
            
            class_report = agreement_metrics['classification_report']
            f.write(f"{'Label':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
            f.write("-" * 50 + "\n")
            
            for label in labels:
                if label in class_report:
                    metrics = class_report[label]
                    f.write(f"{label:<15} {metrics['precision']:<10.3f} "
                           f"{metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f}\n")
        
        # Summary
        f.write("\nSummary:\n")
        f.write("-" * 10 + "\n")
        
        disagreements = [a for a in agreement_metrics['component_agreements'] if not a['agreement']]
        n_components = len(agreement_metrics['component_agreements'])
        
        f.write(f"Total components analyzed: {n_components}\n")
        f.write(f"Components in agreement: {n_components - len(disagreements)}\n")
        f.write(f"Components in disagreement: {len(disagreements)}\n")
        f.write(f"Agreement rate: {agreement_metrics['accuracy']*100:.1f}%\n")
        
        if disagreements:
            f.write(f"\nComponents with disagreements:\n")
            for d in disagreements:
                f.write(f"- {d['component']}: {d['method1_label']} vs {d['method2_label']}\n")
    
    logging.info(f"Comparison report saved to: {report_path}")


def save_detailed_results(results1: Dict, results2: Dict, 
                         agreement_metrics: Dict, output_dir: Path) -> None:
    """
    Save detailed results to CSV files.
    
    Args:
        results1: Results from first method
        results2: Results from second method
        agreement_metrics: Agreement analysis results
        output_dir: Directory to save files
    """
    if not (results1['success'] and results2['success']):
        return
    
    # Component-level comparison
    comparison_data = []
    agreements = agreement_metrics['component_agreements']
    
    for i, agree in enumerate(agreements):
        row = {
            'component': agree['component'],
            'component_index': i,
            f'{results1["method"]}_label': agree['method1_label'],
            f'{results2["method"]}_label': agree['method2_label'],
            f'{results1["method"]}_probability': results1['y_pred_proba'][i],
            f'{results2["method"]}_probability': results2['y_pred_proba'][i],
            'agreement': agree['agreement'],
            'probability_difference': results1['y_pred_proba'][i] - results2['y_pred_proba'][i]
        }
        comparison_data.append(row)
    
    # Save to CSV
    df = pd.DataFrame(comparison_data)
    csv_path = output_dir / 'detailed_comparison.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    logging.info(f"Detailed results saved to: {csv_path}")


def main():
    """Main function to run the comparison."""
    parser = argparse.ArgumentParser(
        description="Compare ICVision and MNE-ICALabel classification results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare using EEGLAB .set file with embedded ICA
  python compare_ica_classifiers.py data/subject01.set
  
  # Compare using separate raw and ICA files  
  python compare_ica_classifiers.py data/raw.fif --ica data/ica.fif
  
  # Save results to custom directory
  python compare_ica_classifiers.py data/subject01.set --output results/comparison/
        """
    )
    
    parser.add_argument('raw_data', help='Path to raw EEG data (.fif or .set)')
    parser.add_argument('--ica', help='Path to ICA data (optional if raw_data contains ICA)')
    parser.add_argument('--output', '-o', default='comparison_results', 
                       help='Output directory for results (default: comparison_results)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    logging.info("Starting ICA classification comparison")
    logging.info(f"Raw data: {args.raw_data}")
    logging.info(f"ICA data: {args.ica or 'auto-detect from raw'}")
    logging.info(f"Output directory: {output_dir}")
    
    try:
        # Load data
        raw, ica = load_and_prepare_data(args.raw_data, args.ica)
        
        # Run classifications
        logging.info("Running classifications...")
        
        # Run MNE-ICALabel first (modifies ICA object)
        icalabel_results = run_icalabel_classification(raw, ica)
        
        # Run ICVision second (on fresh copy)
        icvision_results = run_icvision_classification(raw, ica)
        
        # Check if both succeeded
        if not icalabel_results['success']:
            logging.error(f"MNE-ICALabel failed: {icalabel_results['error']}")
        
        if not icvision_results['success']:
            logging.error(f"ICVision failed: {icvision_results['error']}")
        
        if not (icalabel_results['success'] and icvision_results['success']):
            logging.error("Cannot perform comparison due to classification failures")
            sys.exit(1)
        
        # Analyze agreement
        logging.info("Analyzing agreement between methods...")
        agreement_metrics = calculate_agreement_metrics(
            icalabel_results['labels'], icvision_results['labels'],
            icalabel_results['method'], icvision_results['method']
        )
        
        # Generate outputs
        logging.info("Generating comparison outputs...")
        
        # Save detailed results
        save_detailed_results(icalabel_results, icvision_results, agreement_metrics, output_dir)
        
        # Generate report
        generate_comparison_report(icalabel_results, icvision_results, agreement_metrics, output_dir)
        
        # Create visualizations
        if not args.no_plots:
            create_comparison_visualizations(icalabel_results, icvision_results, 
                                           agreement_metrics, output_dir)
        
        # Print summary
        n_components = len(agreement_metrics['component_agreements'])
        accuracy = agreement_metrics['accuracy']
        
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Components analyzed: {n_components}")
        print(f"Overall agreement: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
        
        logging.info("Comparison completed successfully!")
        
    except Exception as e:
        logging.error(f"Comparison failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()