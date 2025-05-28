Examples
========

This page provides practical examples of using ICVision in different scenarios.

Basic Usage Examples
--------------------

Single Subject Processing
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from icvision.core import label_components
   from pathlib import Path

   # Process a single subject
   subject_id = "sub-001"
   raw_path = f"data/{subject_id}_raw.fif"
   ica_path = f"data/{subject_id}_ica.fif"
   output_dir = Path(f"results/{subject_id}")

   raw_cleaned, ica_updated, results_df = label_components(
       raw_data=raw_path,
       ica_data=ica_path,
       output_dir=output_dir,
       confidence_threshold=0.8,
       generate_report=True
   )

   print(f"Processed {subject_id}")
   print(f"Components: {len(results_df)}")
   print(f"Excluded: {results_df['exclude_vision'].sum()}")

Batch Processing Multiple Subjects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from icvision.core import label_components
   from pathlib import Path
   import pandas as pd

   subjects = ["sub-001", "sub-002", "sub-003", "sub-004"]
   results_summary = []

   for subject_id in subjects:
       print(f"Processing {subject_id}...")
       
       try:
           raw_path = f"data/{subject_id}_raw.fif"
           ica_path = f"data/{subject_id}_ica.fif"
           output_dir = Path(f"results/{subject_id}")
           
           raw_cleaned, ica_updated, results_df = label_components(
               raw_data=raw_path,
               ica_data=ica_path,
               output_dir=output_dir,
               confidence_threshold=0.8
           )
           
           # Collect summary statistics
           summary = {
               'subject': subject_id,
               'total_components': len(results_df),
               'excluded_components': results_df['exclude_vision'].sum(),
               'brain_components': (results_df['label'] == 'brain').sum(),
               'eye_components': (results_df['label'] == 'eye').sum(),
               'muscle_components': (results_df['label'] == 'muscle').sum(),
               'status': 'success'
           }
           
       except Exception as e:
           summary = {
               'subject': subject_id,
               'status': f'failed: {str(e)}'
           }
       
       results_summary.append(summary)

   # Create summary report
   summary_df = pd.DataFrame(results_summary)
   summary_df.to_csv("batch_processing_summary.csv", index=False)
   print("\nBatch processing complete!")
   print(summary_df)

Advanced Configuration Examples
-------------------------------

Conservative Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use higher confidence thresholds and exclude fewer artifact types:

.. code-block:: python

   # Very conservative - only exclude high-confidence artifacts
   raw_cleaned, ica_updated, results_df = label_components(
       raw_data="data.set",
       ica_data="ica.fif",
       confidence_threshold=0.95,  # Very high confidence required
       labels_to_exclude=["eye", "muscle"],  # Only obvious artifacts
       auto_exclude=True
   )

Aggressive Artifact Removal
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lower thresholds to catch more artifacts:

.. code-block:: python

   # Aggressive - exclude more potential artifacts
   raw_cleaned, ica_updated, results_df = label_components(
       raw_data="data.set",
       ica_data="ica.fif",
       confidence_threshold=0.7,  # Lower threshold
       labels_to_exclude=["eye", "muscle", "heart", "line_noise", "channel_noise", "other_artifact"],
       auto_exclude=True
   )

Custom Classification Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from icvision.core import label_components
   import pandas as pd

   # Step 1: Run classification without auto-exclusion
   raw_cleaned, ica_updated, results_df = label_components(
       raw_data="data.set",
       ica_data="ica.fif",
       auto_exclude=False,  # Don't exclude automatically
       generate_report=True
   )

   # Step 2: Custom exclusion logic
   def custom_exclusion_rules(df):
       """Apply custom rules for component exclusion."""
       exclude_mask = (
           # High confidence artifacts
           ((df['confidence'] >= 0.9) & (df['label'].isin(['eye', 'muscle', 'heart']))) |
           # Very high confidence line noise
           ((df['confidence'] >= 0.95) & (df['label'] == 'line_noise')) |
           # Any muscle artifact above 0.8
           ((df['confidence'] >= 0.8) & (df['label'] == 'muscle'))
       )
       return exclude_mask

   # Apply custom rules
   custom_exclude = custom_exclusion_rules(results_df)
   results_df['custom_exclude'] = custom_exclude

   # Manually apply exclusions to ICA
   exclude_indices = results_df[custom_exclude].index.tolist()
   ica_updated.exclude = exclude_indices

   print(f"Custom exclusion: {len(exclude_indices)} components")

Working with Different File Formats
-----------------------------------

EEGLAB .set Files
~~~~~~~~~~~~~~~~

.. code-block:: python

   # EEGLAB format
   raw_cleaned, ica_updated, results_df = label_components(
       raw_data="data/subject01.set",  # EEGLAB .set file
       ica_data="data/subject01_ica.fif",  # MNE .fif file
       output_dir="results/"
   )

MNE .fif Files
~~~~~~~~~~~~~

.. code-block:: python

   # MNE format
   raw_cleaned, ica_updated, results_df = label_components(
       raw_data="data/subject01_raw.fif",
       ica_data="data/subject01_ica.fif", 
       output_dir="results/"
   )

Mixed Formats
~~~~~~~~~~~~

.. code-block:: python

   import mne
   from icvision.utils import load_raw_data, load_ica_data

   # Load data explicitly
   raw = load_raw_data("data/subject01.set")  # Will handle .set format
   ica = load_ica_data("data/subject01_ica.fif")  # Will handle .fif format

   # Process with loaded objects
   raw_cleaned, ica_updated, results_df = label_components(
       raw_data=raw,
       ica_data=ica,
       output_dir="results/"
   )

Integration Examples
-------------------

MNE-Python Pipeline
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import mne
   from icvision.core import label_components

   # Load raw data
   raw = mne.io.read_raw_fif("data.fif", preload=True)

   # Preprocessing
   raw.filter(1, 40)  # Band-pass filter
   raw.set_eeg_reference('average')  # Average reference

   # ICA decomposition
   ica = mne.preprocessing.ICA(n_components=20, random_state=42)
   ica.fit(raw)

   # ICVision classification
   raw_cleaned, ica_updated, results_df = label_components(
       raw_data=raw,
       ica_data=ica,
       output_dir="results/"
   )

   # Continue with cleaned data
   epochs = mne.Epochs(raw_cleaned, events, event_id, tmin=-0.2, tmax=0.8)

Custom Reporting
~~~~~~~~~~~~~~~

.. code-block:: python

   from icvision.core import label_components
   import matplotlib.pyplot as plt
   import seaborn as sns

   # Run ICVision
   raw_cleaned, ica_updated, results_df = label_components(
       raw_data="data.set",
       ica_data="ica.fif",
       output_dir="results/"
   )

   # Custom analysis and plotting
   fig, axes = plt.subplots(2, 2, figsize=(12, 8))

   # Distribution of labels
   results_df['label'].value_counts().plot(kind='bar', ax=axes[0,0])
   axes[0,0].set_title('Component Labels')

   # Confidence distribution
   results_df['confidence'].hist(bins=20, ax=axes[0,1])
   axes[0,1].set_title('Confidence Distribution')

   # Confidence by label
   sns.boxplot(data=results_df, x='label', y='confidence', ax=axes[1,0])
   axes[1,0].tick_params(axis='x', rotation=45)
   axes[1,0].set_title('Confidence by Label')

   # Exclusion summary
   exclusion_summary = results_df.groupby('label')['exclude_vision'].agg(['count', 'sum'])
   exclusion_summary.plot(kind='bar', ax=axes[1,1])
   axes[1,1].set_title('Exclusions by Label')

   plt.tight_layout()
   plt.savefig("results/custom_analysis.png", dpi=300, bbox_inches='tight')

Error Handling and Debugging
----------------------------

Robust Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from icvision.core import label_components
   import logging

   # Set up logging
   logging.basicConfig(level=logging.INFO)

   def process_subject_safely(subject_id, max_retries=3):
       """Process a subject with error handling and retries."""
       for attempt in range(max_retries):
           try:
               raw_path = f"data/{subject_id}_raw.fif"
               ica_path = f"data/{subject_id}_ica.fif"
               output_dir = f"results/{subject_id}"
               
               raw_cleaned, ica_updated, results_df = label_components(
                   raw_data=raw_path,
                   ica_data=ica_path,
                   output_dir=output_dir,
                   batch_size=5,  # Smaller batches for stability
                   max_concurrency=2  # Conservative concurrency
               )
               
               print(f"✓ Successfully processed {subject_id}")
               return results_df
               
           except Exception as e:
               print(f"✗ Attempt {attempt + 1} failed for {subject_id}: {str(e)}")
               if attempt == max_retries - 1:
                   print(f"✗ Failed to process {subject_id} after {max_retries} attempts")
                   return None
               else:
                   print(f"  Retrying in 5 seconds...")
                   time.sleep(5)

   # Use the robust function
   subjects = ["sub-001", "sub-002", "sub-003"]
   for subject in subjects:
       result = process_subject_safely(subject)

Performance Optimization
-----------------------

High-Throughput Processing
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimized for speed
   raw_cleaned, ica_updated, results_df = label_components(
       raw_data="data.set",
       ica_data="ica.fif",
       batch_size=25,        # Large batches
       max_concurrency=8,    # High concurrency
       generate_report=False # Skip PDF generation for speed
   )

Memory-Conscious Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimized for low memory usage
   raw_cleaned, ica_updated, results_df = label_components(
       raw_data="data.set",
       ica_data="ica.fif", 
       batch_size=5,         # Small batches
       max_concurrency=2,    # Low concurrency
       generate_report=False # Reduce memory usage
   )