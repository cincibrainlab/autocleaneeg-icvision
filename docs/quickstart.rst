Quick Start Guide
=================

This guide will get you up and running with ICVision in just a few minutes.

Prerequisites
-------------

Before starting, ensure you have:

1. ICVision installed (see :doc:`installation`)
2. An OpenAI API key configured
3. EEG data in MNE-compatible format (.fif, .set)
4. An ICA decomposition file (.fif)

Basic Workflow
--------------

ICVision follows a simple workflow:

1. **Load** your raw EEG data and ICA decomposition
2. **Generate** component visualizations
3. **Classify** components using OpenAI's Vision API
4. **Apply** artifact removal based on classifications
5. **Save** cleaned data and results

Command-Line Usage
------------------

The easiest way to use ICVision is through the command-line interface:

Basic Command
~~~~~~~~~~~~~

.. code-block:: bash

   icvision /path/to/raw_data.set /path/to/ica_data.fif

This will:

- Create an ``icvision_results/`` directory
- Save cleaned raw data and updated ICA object
- Generate a CSV file with classification results
- Create a PDF report with component visualizations

Advanced Options
~~~~~~~~~~~~~~~~

.. code-block:: bash

   icvision data/subject01_raw.fif data/subject01_ica.fif \
       --output-dir results/subject01/ \
       --model gpt-4.1 \
       --confidence-threshold 0.8 \
       --labels-to-exclude eye muscle line_noise \
       --batch-size 8 \
       --verbose

Python API Usage
----------------

For programmatic use or integration into existing pipelines:

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   from icvision.core import label_components
   from pathlib import Path

   # Define paths
   raw_path = "data/subject01_raw.set"
   ica_path = "data/subject01_ica.fif"
   output_dir = Path("results/subject01")

   # Run ICVision
   raw_cleaned, ica_updated, results_df = label_components(
       raw_data=raw_path,
       ica_data=ica_path,
       output_dir=output_dir,
       confidence_threshold=0.8,
       generate_report=True
   )

   # Use the results
   print(f"Components classified: {len(results_df)}")
   print(f"Components excluded: {results_df['exclude_vision'].sum()}")

Working with MNE Objects
~~~~~~~~~~~~~~~~~~~~~~~~

You can also pass MNE objects directly:

.. code-block:: python

   import mne
   from icvision.core import label_components

   # Load your data
   raw = mne.io.read_raw_fif("data.fif", preload=True)
   ica = mne.preprocessing.read_ica("ica.fif")

   # Run ICVision
   raw_cleaned, ica_updated, results_df = label_components(
       raw_data=raw,
       ica_data=ica,
       output_dir="results/"
   )

Understanding Results
--------------------

CSV Output
~~~~~~~~~~

The results CSV contains:

- ``component_name``: ICA component identifier (e.g., "IC0", "IC1")
- ``label``: Classified component type (brain, eye, muscle, etc.)
- ``confidence``: Classification confidence (0.0-1.0)
- ``reason``: Explanation for the classification
- ``exclude_vision``: Whether component was marked for exclusion

Example:

.. code-block:: text

   component_name,label,confidence,reason,exclude_vision
   IC0,brain,0.95,Clear brain activity pattern,False
   IC1,eye,0.87,Frontal topography with blink artifacts,True
   IC2,muscle,0.92,High frequency noise pattern,True

PDF Report
~~~~~~~~~~

The PDF report includes:

- Summary statistics
- Individual component plots for each classified component
- Topography, time series, power spectral density, and ERP-image views

Next Steps
----------

- Learn about :doc:`configuration` options
- Explore :doc:`examples` for advanced usage
- Check the :doc:`api/core` for detailed API documentation