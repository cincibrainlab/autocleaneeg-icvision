#!/usr/bin/env python3
"""Analyze accuracy sweep results across models."""

import json
import pandas as pd
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path("/tmp/accuracy_sweep_extracted/accuracy_sweep")
MANIFEST = Path("/Users/syex8x/Documents/Github/autocleaneeg-icvision/experiments/manifests/accuracy_screen_120.csv")

MODELS = [
    "gpt-5.6-sol",
    "gpt-5.6-terra", 
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex-spark",
    "gpt-daybreak-blue-latest"
]

def load_manifest():
    """Load ground truth manifest."""
    df = pd.read_csv(MANIFEST)
    return df

def load_results():
    """Load all model results."""
    results = {}
    for model in MODELS:
        csv_file = RESULTS_DIR / f"{model}_medium.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            results[model] = df
            print(f"Loaded {model}: {len(df)} rows")
        else:
            print(f"MISSING: {model}")
    return results

def calculate_accuracy(df, manifest):
    """Calculate accuracy metrics."""
    merged = manifest.merge(df, on=['set_path', 'component_index'], suffixes=('_true', '_pred'))
    
    # Overall accuracy
    correct = (merged['true_label_norm_true'] == merged['predicted_label']).sum()
    total = len(merged)
    accuracy = correct / total
    
    # Per-class accuracy
    per_class = {}
    for label in merged['true_label_norm_true'].unique():
        mask = merged['true_label_norm_true'] == label
        class_correct = (merged.loc[mask, 'true_label_norm_true'] == merged.loc[mask, 'predicted_label']).sum()
        class_total = mask.sum()
        per_class[label] = {
            'correct': class_correct,
            'total': class_total,
            'accuracy': class_correct / class_total if class_total > 0 else 0
        }
    
    # Confusion matrix
    confusion = pd.crosstab(merged['true_label_norm_true'], merged['predicted_label'])
    
    return {
        'overall': accuracy,
        'correct': correct,
        'total': total,
        'per_class': per_class,
        'confusion': confusion,
        'merged': merged
    }

def analyze_low_accuracy(merged, threshold=0.5):
    """Find components with poor performance."""
    merged = merged.copy()
    merged['correct'] = merged['true_label_norm_true'] == merged['predicted_label']
    
    # Group by file and check accuracy
    file_stats = merged.groupby('set_path').agg({
        'correct': ['sum', 'count', 'mean']
    }).round(3)
    file_stats.columns = ['correct', 'total', 'accuracy']
    file_stats = file_stats.sort_values('accuracy')
    
    # Files below threshold
    bad_files = file_stats[file_stats['accuracy'] < threshold]
    
    return file_stats, bad_files

def analyze_by_component_number(merged):
    """Check if early components (0-30) perform better."""
    merged = merged.copy()
    merged['correct'] = merged['true_label_norm_true'] == merged['predicted_label']
    merged['early_component'] = merged['component_index'] <= 30
    
    early = merged[merged['early_component']]
    late = merged[merged['early_component'] == False]
    
    early_acc = early['correct'].mean() if len(early) > 0 else 0
    late_acc = late['correct'].mean() if len(late) > 0 else 0
    
    return {
        'early': {'count': len(early), 'accuracy': early_acc},
        'late': {'count': len(late), 'accuracy': late_acc}
    }

def main():
    print("=" * 80)
    print("ACCURACY SWEEP ANALYSIS")
    print("=" * 80)
    
    manifest = load_manifest()
    print(f"\nManifest: {len(manifest)} components from {manifest['set_path'].nunique()} files")
    print(f"Class distribution:\n{manifest['true_label_norm'].value_counts()}\n")
    
    results = load_results()
    print()
    
    # Summary table
    summary = []
    for model, df in results.items():
        metrics = calculate_accuracy(df, manifest)
        summary.append({
            'model': model,
            'accuracy': metrics['overall'],
            'correct': metrics['correct'],
            'total': metrics['total']
        })
    
    summary_df = pd.DataFrame(summary).sort_values('accuracy', ascending=False)
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    # Detailed per-model analysis
    print("\n" + "=" * 80)
    print("PER-MODEL DETAILED ANALYSIS")
    print("=" * 80)
    
    for model, df in results.items():
        metrics = calculate_accuracy(df, manifest)
        print(f"\n{model}: {metrics['overall']:.2%} ({metrics['correct']}/{metrics['total']})")
        print("-" * 60)
        
        # Per-class performance
        print("Per-class accuracy:")
        for label, stats in sorted(metrics['per_class'].items()):
            print(f"  {label:20s}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        
        # Component number analysis
        comp_analysis = analyze_by_component_number(metrics['merged'])
        print(f"\nEarly components (≤30): {comp_analysis['early']['accuracy']:.2%} (n={comp_analysis['early']['count']})")
        print(f"Late components (>30):  {comp_analysis['late']['accuracy']:.2%} (n={comp_analysis['late']['count']})")
        
        # Find worst performing files
        file_stats, bad_files = analyze_low_accuracy(metrics['merged'], threshold=0.5)
        if len(bad_files) > 0:
            print(f"\nFiles with <50% accuracy ({len(bad_files)} total):")
            for file, row in bad_files.iterrows():
                print(f"  {file}: {row['accuracy']:.2%} ({int(row['correct'])}/{int(row['total'])})")
        
        # Show confusion matrix for top model
        if model == summary_df.iloc[0]['model']:
            print(f"\nConfusion matrix (rows=true, cols=predicted):")
            print(metrics['confusion'].to_string())
    
    # Find systematic biases
    print("\n" + "=" * 80)
    print("SYSTEMATIC BIAS ANALYSIS")
    print("=" * 80)
    
    # Aggregate all predictions
    all_merged = []
    for model, df in results.items():
        m = calculate_accuracy(df, manifest)['merged']
        m['model'] = model
        all_merged.append(m)
    
    all_data = pd.concat(all_merged)
    
    # Check class distribution skew
    print("\nPrediction distribution by model:")
    pred_dist = all_data.groupby(['model', 'predicted_label']).size().unstack(fill_value=0)
    print(pred_dist.to_string())
    
    print("\nTrue class distribution (constant):")
    print(all_data['true_label_norm_true'].value_counts().to_string())

if __name__ == "__main__":
    main()
