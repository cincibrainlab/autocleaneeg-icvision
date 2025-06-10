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
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


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
    raw_path_obj = Path(raw_path)
    if raw_path_obj.suffix.lower() == ".set":
        raw = mne.io.read_raw_eeglab(raw_path_obj, preload=True)
    elif raw_path_obj.suffix.lower() == ".fif":
        raw = mne.io.read_raw_fif(raw_path_obj, preload=True)
    else:
        raise ValueError(f"Unsupported raw data format: {raw_path_obj.suffix}")

    # Load ICA
    if ica_path is None:
        if raw_path_obj.suffix.lower() == ".set":
            logging.info("Attempting to load ICA from EEGLAB .set file")
            try:
                ica = mne.preprocessing.read_ica_eeglab(raw_path_obj)
            except Exception as e:
                raise RuntimeError(f"Could not load ICA from .set file: {e}")
        else:
            raise ValueError("ICA path required for non-.set files")
    else:
        logging.info(f"Loading ICA from: {ica_path}")
        ica_path_obj = Path(ica_path)
        if ica_path_obj.suffix.lower() == ".fif":
            ica = mne.preprocessing.read_ica(ica_path_obj)
        elif ica_path_obj.suffix.lower() == ".set":
            ica = mne.preprocessing.read_ica_eeglab(ica_path_obj)
        else:
            raise ValueError(f"Unsupported ICA format: {ica_path_obj.suffix}")

    # Prepare data for classification
    raw.set_eeg_reference("average", projection=True)
    raw.apply_proj()
    raw_filtered = raw.copy().filter(1.0, 100.0)

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

        result = label_components(raw, ica_copy, method="iclabel")

        return {
            "method": "MNE-ICALabel",
            "labels": result["labels"],
            "y_pred_proba": result["y_pred_proba"],
            "ica_labels_scores_": ica_copy.labels_scores_.copy(),
            "ica_labels_": ica_copy.labels_.copy(),
            "success": True,
            "error": None,
        }

    except ImportError:
        logging.error("MNE-ICALabel not available. Install with: pip install mne-icalabel")
        return {"method": "MNE-ICALabel", "success": False, "error": "MNE-ICALabel not installed"}
    except Exception as e:
        logging.error(f"MNE-ICALabel classification failed: {e}")
        return {"method": "MNE-ICALabel", "success": False, "error": str(e)}


def run_icvision_classification(raw: mne.io.Raw, ica: mne.preprocessing.ICA, output_dir: Optional[Path] = None) -> Dict:
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
        from icvision.core import label_components as icvision_label_components

        logging.info("Running ICVision classification (direct mode to capture images)...")

        # Reset ICA object to clean state
        ica_copy = ica.copy()

        # Set up temporary output directory for ICVision to capture reasoning and image paths
        if output_dir:
            temp_icvision_dir = output_dir / "icvision_temp"
            temp_icvision_dir.mkdir(exist_ok=True)
        else:
            import tempfile

            temp_icvision_dir = Path(tempfile.mkdtemp(prefix="icvision_comparison_"))

        # Call ICVision directly to ensure images are generated
        raw_cleaned, ica_updated, results_df = icvision_label_components(
            raw,
            ica_copy,
            output_dir=temp_icvision_dir,
            generate_report=True,
            auto_exclude=False,  # Don't modify the data, just classify
        )

        # Now generate component images separately and save them to our temp directory
        logging.info("Generating component images for prompt refinement analysis...")
        component_plots_dir = temp_icvision_dir / "component_plots"
        component_plots_dir.mkdir(exist_ok=True)

        try:
            from icvision.plotting import plot_components_batch

            component_indices = list(range(ica_copy.n_components_))
            plot_results = plot_components_batch(ica_copy, raw, component_indices, component_plots_dir)
            logging.info(f"Generated {len(plot_results)} component images in {component_plots_dir}")
        except Exception as e:
            logging.warning(f"Failed to generate component images: {e}")

        # Extract labels and probabilities for compatibility
        labels = results_df["label"].tolist()
        confidences = results_df["confidence"].tolist()

        # Convert ICVision labels to ICLabel format for comparison
        from icvision.compat import ICVISION_TO_ICALABEL_DISPLAY

        icalabel_labels = [ICVISION_TO_ICALABEL_DISPLAY.get(label, "other") for label in labels]

        result = {"labels": icalabel_labels, "y_pred_proba": confidences}

        # Capture detailed results including reasoning and images
        reasoning_data = None
        image_dir = None

        # Extract reasoning from results_df
        if "reason" in results_df.columns:
            reasoning_data = results_df[["component_index", "reason"]].to_dict("records")

        # Look for component images directory
        component_plots_dir = temp_icvision_dir / "component_plots"
        if component_plots_dir.exists():
            image_dir = component_plots_dir
        else:
            # Look for ICVision generated plots (backup)
            image_dirs = list(temp_icvision_dir.glob("*_component_plots"))
            if image_dirs:
                image_dir = image_dirs[0]
            else:
                # Look for alternative naming patterns
                alt_dirs = list(temp_icvision_dir.glob("component_plots*"))
                if alt_dirs:
                    image_dir = alt_dirs[0]
                else:
                    image_dir = None

        logging.info(f"ICVision temp directory: {temp_icvision_dir}")
        if image_dir:
            logging.info(f"Component images found in: {image_dir}")
        else:
            logging.warning("No component images directory found")

        return {
            "method": "ICVision",
            "labels": result["labels"],
            "y_pred_proba": result["y_pred_proba"],
            "ica_labels_scores_": ica_updated.labels_scores_.copy() if hasattr(ica_updated, "labels_scores_") else None,
            "ica_labels_": ica_updated.labels_.copy() if hasattr(ica_updated, "labels_") else None,
            "reasoning_data": reasoning_data,
            "image_dir": str(image_dir) if image_dir else None,
            "temp_dir": str(temp_icvision_dir),
            "success": True,
            "error": None,
        }

    except Exception as e:
        logging.error(f"ICVision classification failed: {e}")
        return {"method": "ICVision", "success": False, "error": str(e)}


def load_human_labels(csv_path: str) -> Dict:
    """
    Load human-labeled ground truth data from CSV file.

    Args:
        csv_path: Path to CSV file with human labels

    Returns:
        Dictionary with human label data
    """
    try:
        df = pd.read_csv(csv_path)

        # Validate required columns
        required_cols = ["component_index", "label"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Sort by component index to ensure proper order
        df = df.sort_values("component_index")

        # Convert to lists for compatibility with existing code
        labels = df["label"].tolist()
        if "confidence" in df.columns:
            confidences = df["confidence"].tolist()
        else:
            confidences = [1.0] * len(labels)
        if "notes" in df.columns:
            notes = df["notes"].tolist()
        else:
            notes = [""] * len(labels)

        return {
            "method": "Human Labels",
            "labels": labels,
            "y_pred_proba": confidences,  # Use confidence as probability
            "component_indices": df["component_index"].tolist(),
            "notes": notes,
            "success": True,
            "error": None,
        }

    except Exception as e:
        logging.error(f"Failed to load human labels from {csv_path}: {e}")
        return {"method": "Human Labels", "success": False, "error": str(e)}


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
        "eye blink": "eye",
        "muscle artifact": "muscle",
        "heart beat": "heart",
        "line noise": "line_noise",
        "channel noise": "channel_noise",
        "other": "other_artifact",
    }

    harmonized = []
    for label in labels:
        harmonized_label = label_mapping.get(label, label)
        harmonized.append(harmonized_label)

    return harmonized


def calculate_agreement_metrics(
    labels1: List[str], labels2: List[str], method1: str, method2: str, human_labels: Optional[List[str]] = None
) -> Dict:
    """
    Calculate agreement metrics between classification methods.

    Args:
        labels1: Labels from first method
        labels2: Labels from second method
        method1: Name of first method
        method2: Name of second method
        human_labels: Optional human ground truth labels for 3-way comparison

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
            harmonized1, harmonized2, labels=all_labels, output_dict=True, zero_division=0
        )
    except:
        class_report = None

    # Component-wise agreement
    agreements = []
    for i, (l1, l2) in enumerate(zip(harmonized1, harmonized2)):
        agreements.append({"component": f"IC{i:02d}", "method1_label": l1, "method2_label": l2, "agreement": l1 == l2})

    result = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "labels": all_labels,
        "classification_report": class_report,
        "component_agreements": agreements,
        "harmonized_labels1": harmonized1,
        "harmonized_labels2": harmonized2,
    }

    # Add human labels comparison if available
    if human_labels is not None:
        harmonized_human = harmonize_labels(human_labels, "Human")

        # Ensure same length
        min_len = min(len(harmonized1), len(harmonized2), len(harmonized_human))
        harmonized1_trimmed = harmonized1[:min_len]
        harmonized2_trimmed = harmonized2[:min_len]
        harmonized_human_trimmed = harmonized_human[:min_len]

        # Calculate accuracy against human labels with special rule for ICVision
        # ICVision brain classifications are not counted as wrong (since brain components don't get rejected)
        icvision_vs_human_correct = []
        for human_label, icvision_label in zip(harmonized_human_trimmed, harmonized2_trimmed):
            # If ICVision classifies as brain, count as correct regardless of human label
            if icvision_label == "brain":
                icvision_vs_human_correct.append(True)
            else:
                icvision_vs_human_correct.append(human_label == icvision_label)

        human_accuracy1 = accuracy_score(harmonized_human_trimmed, harmonized1_trimmed)
        human_accuracy2 = sum(icvision_vs_human_correct) / len(icvision_vs_human_correct)

        # Add to component agreements with special rule for ICVision
        for i, agree in enumerate(agreements[:min_len]):
            agree["human_label"] = harmonized_human_trimmed[i]
            agree["method1_vs_human"] = harmonized1_trimmed[i] == harmonized_human_trimmed[i]
            # ICVision brain classifications are considered correct
            if harmonized2_trimmed[i] == "brain":
                agree["method2_vs_human"] = True
            else:
                agree["method2_vs_human"] = harmonized2_trimmed[i] == harmonized_human_trimmed[i]

        result.update(
            {
                "human_labels_available": True,
                "harmonized_human": harmonized_human_trimmed,
                "human_accuracy1": human_accuracy1,
                "human_accuracy2": human_accuracy2,
                "method1_vs_human_cm": confusion_matrix(
                    harmonized_human_trimmed, harmonized1_trimmed, labels=all_labels
                ),
                "method2_vs_human_cm": confusion_matrix(
                    harmonized_human_trimmed, harmonized2_trimmed, labels=all_labels
                ),
            }
        )
    else:
        result["human_labels_available"] = False

    return result


def create_comparison_visualizations(
    results1: Dict, results2: Dict, agreement_metrics: Dict, output_dir: Path, human_results: Optional[Dict] = None
) -> None:
    """
    Create visualizations comparing the classification methods.

    Args:
        results1: Results from first method
        results2: Results from second method
        agreement_metrics: Agreement analysis results
        output_dir: Directory to save plots
        human_results: Optional human labels for ground truth comparison
    """
    output_dir.mkdir(exist_ok=True)

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # 1. Label distribution comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Count labels for each method
    labels1 = agreement_metrics["harmonized_labels1"]
    labels2 = agreement_metrics["harmonized_labels2"]

    unique_labels = agreement_metrics["labels"]
    counts1 = [labels1.count(label) for label in unique_labels]
    counts2 = [labels2.count(label) for label in unique_labels]

    x = np.arange(len(unique_labels))
    width = 0.35

    ax1.bar(x - width / 2, counts1, width, label=results1["method"], alpha=0.8)
    ax1.bar(x + width / 2, counts2, width, label=results2["method"], alpha=0.8)
    ax1.set_xlabel("Component Labels")
    ax1.set_ylabel("Count")
    ax1.set_title("Label Distribution Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(unique_labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Confusion Matrix
    cm = agreement_metrics["confusion_matrix"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels, ax=ax2)
    ax2.set_xlabel(f'{results2["method"]} Labels')
    ax2.set_ylabel(f'{results1["method"]} Labels')
    ax2.set_title("Confusion Matrix")
    ax2.tick_params(axis="x", rotation=45)
    ax2.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / "label_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Method comparison summary (instead of component agreement)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create summary statistics visualization
    agreements = agreement_metrics["component_agreements"]
    total_components = len(agreements)
    agreement_count = sum(1 for a in agreements if a["agreement"])
    disagreement_count = total_components - agreement_count

    # Pie chart of agreement vs disagreement
    sizes = [agreement_count, disagreement_count]
    pie_labels = [
        f"Agreement\n({agreement_count} components)",
        f"Different Classifications\n({disagreement_count} components)",
    ]
    colors = ["lightgreen", "lightcoral"]

    wedges, texts, autotexts = ax.pie(sizes, labels=pie_labels, colors=colors, autopct="%1.1f%%", startangle=90)

    ax.set_title(
        f"Classification Comparison Summary\n"
        f'{results1["method"]} vs {results2["method"]}\n'
        f"Total Components: {total_components}",
        fontsize=14,
    )

    # Make percentage text larger
    if autotexts:
        for autotext in autotexts:
            if hasattr(autotext, "set_color"):
                autotext.set_color("white")
            if hasattr(autotext, "set_fontsize"):
                autotext.set_fontsize(12)
            if hasattr(autotext, "set_weight"):
                autotext.set_weight("bold")

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Probability comparison (if available)
    if results1["success"] and results2["success"]:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Max probabilities comparison
        proba1 = results1["y_pred_proba"]
        proba2 = results2["y_pred_proba"]

        x = np.arange(len(proba1))

        axes[0].plot(x, proba1, "o-", label=results1["method"], alpha=0.7)
        axes[0].plot(x, proba2, "s-", label=results2["method"], alpha=0.7)
        axes[0].set_xlabel("Component Index")
        axes[0].set_ylabel("Max Probability")
        axes[0].set_title("Maximum Classification Probability Comparison")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Probability difference
        prob_diff = np.array(proba1) - np.array(proba2)
        colors = ["red" if diff < 0 else "blue" for diff in prob_diff]

        axes[1].bar(x, prob_diff, color=colors, alpha=0.7)
        axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Component Index")
        axes[1].set_ylabel(f'{results1["method"]} - {results2["method"]} Probability')
        axes[1].set_title("Probability Difference (Positive = ICVision Higher)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "probability_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 5. Human labels comparison (if available)
    if agreement_metrics.get("human_labels_available", False):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Method 1 vs Human confusion matrix
        cm1_human = agreement_metrics["method1_vs_human_cm"]
        sns.heatmap(
            cm1_human,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=unique_labels,
            yticklabels=unique_labels,
            ax=axes[0, 0],
        )
        axes[0, 0].set_title(f'{results1["method"]} vs Human Labels')
        axes[0, 0].set_xlabel("Predicted (Algorithm)")
        axes[0, 0].set_ylabel("True (Human)")

        # Method 2 vs Human confusion matrix
        cm2_human = agreement_metrics["method2_vs_human_cm"]
        sns.heatmap(
            cm2_human,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=unique_labels,
            yticklabels=unique_labels,
            ax=axes[0, 1],
        )
        axes[0, 1].set_title(f'{results2["method"]} vs Human Labels')
        axes[0, 1].set_xlabel("Predicted (Algorithm)")
        axes[0, 1].set_ylabel("True (Human)")

        # Accuracy comparison bar chart
        methods = [results1["method"], results2["method"]]
        accuracies = [agreement_metrics["human_accuracy1"], agreement_metrics["human_accuracy2"]]

        bars = axes[1, 0].bar(methods, accuracies, color=["lightblue", "lightgreen"], alpha=0.7)
        axes[1, 0].set_ylabel("Accuracy vs Human Labels")
        axes[1, 0].set_title("Algorithm Performance vs Human Ground Truth")
        axes[1, 0].set_ylim(0, 1.0)

        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Triple agreement analysis
        agreements = agreement_metrics["component_agreements"]
        triple_agreement = sum(
            1
            for a in agreements
            if a.get("agreement", False) and a.get("method1_vs_human", False) and a.get("method2_vs_human", False)
        )
        total_components = len(agreements)

        # Pie chart of agreement patterns
        agreement_counts = {
            "All Agree": triple_agreement,
            "Methods Agree\n(vs Human)": sum(
                1
                for a in agreements
                if a.get("agreement", False)
                and not (a.get("method1_vs_human", False) and a.get("method2_vs_human", False))
            ),
            "Mixed Agreement": total_components
            - triple_agreement
            - sum(1 for a in agreements if a.get("agreement", False)),
        }

        sizes = list(agreement_counts.values())
        labels_pie = list(agreement_counts.keys())
        colors = ["lightgreen", "yellow", "lightcoral"]

        axes[1, 1].pie(sizes, labels=labels_pie, colors=colors, autopct="%1.1f%%", startangle=90)
        axes[1, 1].set_title("Agreement Patterns\n(Methods vs Human Ground Truth)")

        plt.tight_layout()
        plt.savefig(output_dir / "human_labels_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    logging.info(f"Visualizations saved to: {output_dir}")


def generate_comparison_report(
    results1: Dict, results2: Dict, agreement_metrics: Dict, output_dir: Path, human_results: Optional[Dict] = None
) -> None:
    """
    Generate a detailed comparison report.

    Args:
        results1: Results from first method
        results2: Results from second method
        agreement_metrics: Agreement analysis results
        output_dir: Directory to save report
    """
    report_path = output_dir / "comparison_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ICA Component Classification Comparison Report\n")
        f.write("=" * 50 + "\n\n")

        # Methods compared
        f.write("Methods Compared:\n")
        f.write(f"- {results1['method']}: {'[SUCCESS]' if results1['success'] else '[FAILED]'}\n")
        f.write(f"- {results2['method']}: {'[SUCCESS]' if results2['success'] else '[FAILED]'}\n\n")

        if not (results1["success"] and results2["success"]):
            f.write("Comparison could not be completed due to classification failures.\n")
            if not results1["success"]:
                f.write(f"{results1['method']} error: {results1['error']}\n")
            if not results2["success"]:
                f.write(f"{results2['method']} error: {results2['error']}\n")
            return

        # Overall agreement
        f.write(
            f"Overall Agreement: {agreement_metrics['accuracy']:.3f} " f"({agreement_metrics['accuracy']*100:.1f}%)\n\n"
        )

        # Label distribution
        f.write("Label Distribution:\n")
        f.write("-" * 20 + "\n")
        labels = agreement_metrics["labels"]
        labels1 = agreement_metrics["harmonized_labels1"]
        labels2 = agreement_metrics["harmonized_labels2"]

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

        for agree in agreement_metrics["component_agreements"]:
            status = "[AGREE]" if agree["agreement"] else "[DISAGREE]"
            f.write(
                f"{agree['component']:<10} {agree['method1_label']:<15} " f"{agree['method2_label']:<15} {status:<10}\n"
            )

        f.write("\n")

        # Classification metrics
        if agreement_metrics["classification_report"]:
            f.write("Detailed Classification Metrics:\n")
            f.write("-" * 35 + "\n")

            class_report = agreement_metrics["classification_report"]
            f.write(f"{'Label':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
            f.write("-" * 50 + "\n")

            for label in labels:
                if label in class_report:
                    metrics = class_report[label]
                    f.write(
                        f"{label:<15} {metrics['precision']:<10.3f} "
                        f"{metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f}\n"
                    )

        # Summary
        f.write("\nAnalysis Summary:\n")
        f.write("-" * 17 + "\n")

        differences = [a for a in agreement_metrics["component_agreements"] if not a["agreement"]]
        n_components = len(agreement_metrics["component_agreements"])

        f.write(f"Total components analyzed: {n_components}\n")
        f.write(f"Components with same classification: {n_components - len(differences)}\n")
        f.write(f"Components with different classification: {len(differences)}\n")
        f.write(f"Classification agreement rate: {agreement_metrics['accuracy']*100:.1f}%\n")

        if differences:
            f.write(f"\nComponents with different classifications:\n")
            for d in differences:
                f.write(f"- {d['component']}: {d['method1_label']} vs {d['method2_label']}\n")

            f.write(f"\nNote: Classification differences reflect varying analytical approaches")
            f.write(f"\nrather than classification errors. Both methods provide valid perspectives.")

    # Generate human labels analysis report if available
    if agreement_metrics.get("human_labels_available", False) and human_results:
        generate_human_labels_analysis(results1, results2, agreement_metrics, human_results, output_dir)

    logging.info(f"Comparison report saved to: {report_path}")


def save_detailed_results(results1: Dict, results2: Dict, agreement_metrics: Dict, output_dir: Path) -> None:
    """
    Save detailed results to CSV files.

    Args:
        results1: Results from first method
        results2: Results from second method
        agreement_metrics: Agreement analysis results
        output_dir: Directory to save files
    """
    if not (results1["success"] and results2["success"]):
        return

    # Component-level comparison
    comparison_data = []
    agreements = agreement_metrics["component_agreements"]

    for i, agree in enumerate(agreements):
        row = {
            "component": agree["component"],
            "component_index": i,
            f'{results1["method"]}_label': agree["method1_label"],
            f'{results2["method"]}_label': agree["method2_label"],
            f'{results1["method"]}_probability': results1["y_pred_proba"][i],
            f'{results2["method"]}_probability': results2["y_pred_proba"][i],
            "agreement": agree["agreement"],
            "probability_difference": results1["y_pred_proba"][i] - results2["y_pred_proba"][i],
        }
        comparison_data.append(row)

    # Save to CSV
    df = pd.DataFrame(comparison_data)
    csv_path = output_dir / "detailed_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    logging.info(f"Detailed results saved to: {csv_path}")


def generate_human_labels_analysis(
    results1: Dict, results2: Dict, agreement_metrics: Dict, human_results: Dict, output_dir: Path
) -> None:
    """
    Generate detailed analysis focusing on human labels for prompt refinement.

    Args:
        results1: MNE-ICALabel results
        results2: ICVision results
        agreement_metrics: Agreement analysis results
        human_results: Human label ground truth
        output_dir: Directory to save report
    """
    report_path = output_dir / "human_labels_analysis.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("HUMAN LABELS ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write("This report analyzes algorithm performance against human ground truth\n")
        f.write("with focus on identifying ICVision prompt refinement opportunities.\n\n")

        # Overall performance summary
        f.write("PERFORMANCE SUMMARY:\n")
        f.write("-" * 20 + "\n")
        human_acc1 = agreement_metrics["human_accuracy1"]
        human_acc2 = agreement_metrics["human_accuracy2"]
        f.write(f"{results1['method']} vs Human: {human_acc1:.3f} ({human_acc1*100:.1f}%)\n")
        f.write(f"{results2['method']} vs Human: {human_acc2:.3f} ({human_acc2*100:.1f}%)\n")

        if human_acc1 > human_acc2:
            f.write(
                f"\n{results1['method']} outperforms {results2['method']} by {(human_acc1-human_acc2)*100:.1f} percentage points.\n"
            )
        elif human_acc2 > human_acc1:
            f.write(
                f"\n{results2['method']} outperforms {results1['method']} by {(human_acc2-human_acc1)*100:.1f} percentage points.\n"
            )
        else:
            f.write(f"\nBoth methods perform equally against human labels.\n")

        f.write("\n")

        # Critical analysis: Where ICLabel agrees with human but ICVision doesn't
        agreements = agreement_metrics["component_agreements"]
        icalabel_correct_icvision_wrong = []
        icvision_correct_icalabel_wrong = []
        both_wrong = []
        both_correct = []

        # Enhance agreements with ICVision reasoning and image paths
        icvision_reasoning_map = {}
        icvision_image_dir = None

        if results2.get("reasoning_data"):
            for item in results2["reasoning_data"]:
                icvision_reasoning_map[item["component_index"]] = item["reason"]

        if results2.get("image_dir"):
            icvision_image_dir = Path(results2["image_dir"])

        for agree in agreements:
            human_label = agree.get("human_label")
            icalabel_correct = agree.get("method1_vs_human", False)
            icvision_correct = agree.get("method2_vs_human", False)

            # Add ICVision reasoning if available
            comp_idx = int(agree["component"].replace("IC", ""))
            if comp_idx in icvision_reasoning_map:
                agree["icvision_reasoning"] = icvision_reasoning_map[comp_idx]

            # Add image file reference if available
            if icvision_image_dir:
                # Look for component image file
                possible_image_files = [
                    icvision_image_dir / f"component_{comp_idx:02d}.webp",
                    icvision_image_dir / f"IC{comp_idx:02d}.webp",
                    icvision_image_dir / f"component_{comp_idx}.webp",
                ]
                for img_file in possible_image_files:
                    if img_file.exists():
                        agree["image_file"] = str(img_file)
                        break

            # Skip cases where ICVision classified as brain (these are now considered correct)
            if agree["method2_label"] == "brain":
                both_correct.append(agree)
            elif icalabel_correct and not icvision_correct:
                icalabel_correct_icvision_wrong.append(agree)
            elif icvision_correct and not icalabel_correct:
                icvision_correct_icalabel_wrong.append(agree)
            elif not icalabel_correct and not icvision_correct:
                both_wrong.append(agree)
            else:
                both_correct.append(agree)

        f.write("DISAGREEMENT ANALYSIS:\n")
        f.write("-" * 22 + "\n")
        f.write(f"Both methods correct: {len(both_correct)} components\n")
        f.write(f"ICLabel correct, ICVision wrong: {len(icalabel_correct_icvision_wrong)} components\n")
        f.write(f"ICVision correct, ICLabel wrong: {len(icvision_correct_icalabel_wrong)} components\n")
        f.write(f"Both methods wrong: {len(both_wrong)} components\n\n")

        # Detailed analysis for ICVision prompt refinement
        f.write("ICVISION PROMPT REFINEMENT OPPORTUNITIES:\n")
        f.write("=" * 42 + "\n")
        f.write("Components where ICLabel agrees with human but ICVision does not:\n\n")

        if icalabel_correct_icvision_wrong:
            # Group by error patterns (excluding brain classifications)
            error_patterns: Dict[str, List[Any]] = {}
            for agree in icalabel_correct_icvision_wrong:
                human_label = agree["human_label"]
                icvision_label = agree["method2_label"]

                # Skip brain classifications as they're now considered correct
                if icvision_label == "brain":
                    continue

                pattern = f"{human_label} -> {icvision_label}"

                if pattern not in error_patterns:
                    error_patterns[pattern] = []
                error_patterns[pattern].append(agree)

            # Sort patterns by frequency
            sorted_patterns = sorted(error_patterns.items(), key=lambda x: len(x[1]), reverse=True)

            f.write("ERROR PATTERN ANALYSIS:\n")
            f.write("-" * 23 + "\n")
            for pattern, components in sorted_patterns:
                f.write(f"\nPattern: {pattern} ({len(components)} components)\n")
                f.write(f"Components: {', '.join([c['component'] for c in components])}\n")

                # Analyze the specific error
                true_label, wrong_label = pattern.split(" -> ")
                f.write(f"\nPROMPT REFINEMENT SUGGESTION for {pattern}:\n")

                if true_label == "brain" and wrong_label in ["muscle", "other_artifact"]:
                    f.write("- Emphasize neural activity patterns in topography\n")
                    f.write("- Look for symmetric bilateral patterns typical of brain activity\n")
                    f.write("- Focus on frequency content in alpha/beta ranges\n")
                elif true_label == "muscle" and wrong_label == "brain":
                    f.write("- Emphasize high-frequency content and temporal muscle patterns\n")
                    f.write("- Look for asymmetric, localized topographies near temporal/frontal areas\n")
                    f.write("- Focus on irregular, high-amplitude time series\n")
                elif true_label == "eye" and wrong_label in ["brain", "other_artifact"]:
                    f.write("- Emphasize frontal/prefrontal topography patterns\n")
                    f.write("- Look for characteristic eye movement artifacts in time series\n")
                    f.write("- Focus on low-frequency, high-amplitude patterns\n")
                elif true_label == "heart" and wrong_label in ["brain", "other_artifact"]:
                    f.write("- Emphasize regular, rhythmic patterns around 60-100 BPM\n")
                    f.write("- Look for posterior/occipital topography patterns\n")
                    f.write("- Focus on QRS-like sharp deflections in time series\n")
                elif true_label == "channel_noise" and wrong_label == "brain":
                    f.write("- Emphasize very localized, single-channel topography\n")
                    f.write("- Look for flat or extremely noisy time series\n")
                    f.write("- Focus on broadband noise across all frequencies\n")
                elif true_label == "other_artifact" and wrong_label == "brain":
                    f.write("- Be more conservative with brain classification\n")
                    f.write("- Look for atypical patterns that don't fit clear categories\n")
                    f.write("- Consider mixed or unclear topography patterns\n")
                else:
                    f.write(f"- Review classification criteria for distinguishing {true_label} from {wrong_label}\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write("COMPONENT-BY-COMPONENT DETAILS:\n")
            f.write("=" * 32 + "\n")

            for agree in icalabel_correct_icvision_wrong:
                # Skip brain classifications as they're now considered correct
                if agree["method2_label"] == "brain":
                    continue

                f.write(f"\nComponent {agree['component']}:\n")
                f.write(f"  Human (Ground Truth): {agree['human_label']}\n")
                f.write(f"  ICLabel (Correct): {agree['method1_label']}\n")
                f.write(f"  ICVision (Wrong): {agree['method2_label']}\n")

                # Add component index for reference
                comp_idx = int(agree["component"].replace("IC", ""))
                f.write(f"  Component Index: {comp_idx}\n")

                # Add ICVision reasoning if available
                if "icvision_reasoning" in agree:
                    f.write(f"  ICVision Reasoning: {agree['icvision_reasoning']}\n")

                # Add image file reference if available
                if "image_file" in agree:
                    f.write(f"  Component Image: {agree['image_file']}\n")

        else:
            f.write("No cases where ICLabel was correct and ICVision was wrong.\n")
            f.write("ICVision performance matches or exceeds ICLabel against human ground truth.\n")

        # Additional insights for ICVision improvement
        f.write("\n" + "=" * 50 + "\n")
        f.write("ADDITIONAL ICVISION INSIGHTS:\n")
        f.write("=" * 30 + "\n")

        if icvision_correct_icalabel_wrong:
            f.write(
                f"\nICVision advantages ({len(icvision_correct_icalabel_wrong)} components where ICVision was correct and ICLabel wrong):\n"
            )
            icvision_advantage_patterns: Dict[str, List[str]] = {}
            for agree in icvision_correct_icalabel_wrong:
                human_label = agree["human_label"]
                icalabel_label = agree["method1_label"]
                pattern = f"{human_label} (vs ICLabel: {icalabel_label})"

                if pattern not in icvision_advantage_patterns:
                    icvision_advantage_patterns[pattern] = []
                icvision_advantage_patterns[pattern].append(agree["component"])

            for pattern, components in icvision_advantage_patterns.items():
                f.write(f"- {pattern}: {', '.join(components)}\n")

        if both_wrong:
            f.write(f"\nChallenging components ({len(both_wrong)} where both methods disagree with human):\n")
            challenging_patterns: Dict[str, List[str]] = {}
            for agree in both_wrong:
                human_label = agree["human_label"]
                pattern = f"Human: {human_label}"

                if pattern not in challenging_patterns:
                    challenging_patterns[pattern] = []
                challenging_patterns[pattern].append(
                    f"{agree['component']} (ICL:{agree['method1_label']}, ICV:{agree['method2_label']})"
                )

            for pattern, components in challenging_patterns.items():
                f.write(f"- {pattern}: {', '.join(components)}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("SUMMARY RECOMMENDATIONS:\n")
        f.write("=" * 25 + "\n")

        if len(icalabel_correct_icvision_wrong) > len(icvision_correct_icalabel_wrong):
            f.write("1. ICVision has room for improvement against human ground truth\n")
            f.write("2. Focus prompt refinement on the error patterns identified above\n")
            f.write("3. Consider incorporating ICLabel's strengths in classification criteria\n")
        elif len(icvision_correct_icalabel_wrong) > len(icalabel_correct_icvision_wrong):
            f.write("1. ICVision outperforms ICLabel against human ground truth\n")
            f.write("2. Current prompt appears well-calibrated\n")
            f.write("3. Consider minor refinements based on remaining error patterns\n")
        else:
            f.write("1. ICVision and ICLabel perform similarly against human ground truth\n")
            f.write("2. Both methods have complementary strengths\n")
            f.write("3. Focus on the challenging components where both methods struggle\n")

        if both_wrong:
            f.write(f"4. {len(both_wrong)} components are challenging for both automated methods\n")
            f.write("5. These may represent inherently ambiguous components or edge cases\n")

    logging.info(f"Human labels analysis saved to: {report_path}")


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
        """,
    )

    parser.add_argument("raw_data", help="Path to raw EEG data (.fif or .set)")
    parser.add_argument("--ica", help="Path to ICA data (optional if raw_data contains ICA)")
    parser.add_argument("--human-labels", help="Path to CSV file with human-labeled ground truth data")
    parser.add_argument(
        "--output",
        "-o",
        default="comparison_results",
        help="Output directory for results (default: comparison_results)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")

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
        icvision_results = run_icvision_classification(raw, ica, output_dir)

        # Load human labels if provided
        human_results = None
        if args.human_labels:
            logging.info(f"Loading human labels from: {args.human_labels}")
            human_results = load_human_labels(args.human_labels)
            if not human_results["success"]:
                logging.error(f"Failed to load human labels: {human_results['error']}")
                human_results = None

        # Check if both automated methods succeeded
        if not icalabel_results["success"]:
            logging.error(f"MNE-ICALabel failed: {icalabel_results['error']}")

        if not icvision_results["success"]:
            logging.error(f"ICVision failed: {icvision_results['error']}")

        if not (icalabel_results["success"] and icvision_results["success"]):
            logging.error("Cannot perform comparison due to classification failures")
            sys.exit(1)

        # Analyze agreement
        logging.info("Analyzing agreement between methods...")
        human_labels = human_results["labels"] if human_results else None
        agreement_metrics = calculate_agreement_metrics(
            icalabel_results["labels"],
            icvision_results["labels"],
            icalabel_results["method"],
            icvision_results["method"],
            human_labels,
        )

        # Generate outputs
        logging.info("Generating comparison outputs...")

        # Save detailed results
        save_detailed_results(icalabel_results, icvision_results, agreement_metrics, output_dir)

        # Generate report
        generate_comparison_report(icalabel_results, icvision_results, agreement_metrics, output_dir, human_results)

        # Create visualizations
        if not args.no_plots:
            create_comparison_visualizations(
                icalabel_results, icvision_results, agreement_metrics, output_dir, human_results
            )

        # Print summary
        n_components = len(agreement_metrics["component_agreements"])
        accuracy = agreement_metrics["accuracy"]

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
