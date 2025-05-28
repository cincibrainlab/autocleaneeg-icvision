Utils Module
============

The utils module provides utility functions for data loading, file handling, and validation.

.. automodule:: icvision.utils
   :members:
   :undoc-members:
   :show-inheritance:

Data Loading Functions
---------------------

load_raw_data
~~~~~~~~~~~~~

.. autofunction:: icvision.utils.load_raw_data

Loads raw EEG data from various file formats with automatic format detection:

- **MNE .fif files**: Native MNE format
- **EEGLAB .set files**: EEGLAB format with automatic conversion
- **Other formats**: Any format supported by MNE-Python

**Example Usage:**

.. code-block:: python

   from icvision.utils import load_raw_data
   
   # Load different file formats
   raw_fif = load_raw_data("data/subject01.fif")
   raw_set = load_raw_data("data/subject01.set") 
   
   # Returns MNE Raw objects
   print(f"Sampling rate: {raw_fif.info['sfreq']} Hz")
   print(f"Channels: {raw_fif.info['nchan']}")

load_ica_data
~~~~~~~~~~~~~

.. autofunction:: icvision.utils.load_ica_data

Loads ICA decomposition data from MNE .fif files.

**Example Usage:**

.. code-block:: python

   from icvision.utils import load_ica_data
   
   # Load ICA decomposition
   ica = load_ica_data("data/subject01_ica.fif")
   
   print(f"Number of components: {ica.n_components_}")
   print(f"Excluded components: {ica.exclude}")

File and Directory Functions
----------------------------

create_output_directory
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: icvision.utils.create_output_directory

Creates output directories with proper error handling.

**Example Usage:**

.. code-block:: python

   from icvision.utils import create_output_directory
   
   # Create output directory structure
   output_dir = "results/subject01/"
   create_output_directory(output_dir)

save_results
~~~~~~~~~~~~

.. autofunction:: icvision.utils.save_results

Saves classification results to CSV format.

**Example Usage:**

.. code-block:: python

   from icvision.utils import save_results
   
   # Save results to CSV
   save_results(results_df, "results/", "classification_results.csv")

Validation Functions
-------------------

validate_inputs
~~~~~~~~~~~~~~~

.. autofunction:: icvision.utils.validate_inputs

Validates that raw EEG data and ICA decomposition are compatible for processing.

**Example Usage:**

.. code-block:: python

   from icvision.utils import validate_inputs, load_raw_data, load_ica_data
   
   # Load and validate data
   raw = load_raw_data("data.fif")
   ica = load_ica_data("ica.fif")
   
   try:
       validate_inputs(raw, ica)
       print("✓ Data validation passed")
   except ValueError as e:
       print(f"✗ Data validation failed: {e}")

validate_api_key
~~~~~~~~~~~~~~~~

.. autofunction:: icvision.utils.validate_api_key

Validates OpenAI API key format and accessibility.

**Example Usage:**

.. code-block:: python

   from icvision.utils import validate_api_key
   
   try:
       validate_api_key("your-api-key")
       print("✓ API key is valid")
   except ValueError as e:
       print(f"✗ API key validation failed: {e}")

validate_classification_results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: icvision.utils.validate_classification_results

Validates that classification results are properly formatted and complete.

**Example Usage:**

.. code-block:: python

   from icvision.utils import validate_classification_results
   
   try:
       validate_classification_results(results_df)
       print("✓ Results validation passed")
   except ValueError as e:
       print(f"✗ Results validation failed: {e}")

Summary and Formatting
----------------------

format_summary_stats
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: icvision.utils.format_summary_stats

Creates formatted summary statistics from classification results.

**Example Usage:**

.. code-block:: python

   from icvision.utils import format_summary_stats
   
   # Generate summary statistics
   summary = format_summary_stats(results_df)
   print(summary)

Integration Examples
-------------------

**Complete Workflow with Utilities:**

.. code-block:: python

   from icvision.utils import (
       load_raw_data, load_ica_data, validate_inputs,
       validate_api_key, create_output_directory, save_results
   )
   
   # Setup
   output_dir = "results/"
   create_output_directory(output_dir)
   
   # Load and validate data
   raw = load_raw_data("data/raw.fif")
   ica = load_ica_data("data/ica.fif")
   validate_inputs(raw, ica)
   
   # Validate API key
   validate_api_key("your-api-key")
   
   print("✓ All validations passed - ready for processing")

**Error-Resilient Processing:**

.. code-block:: python

   from icvision.utils import load_raw_data
   
   def process_subject(subject_id):
       try:
           raw = load_raw_data(f"data/{subject_id}.fif")
           # ... processing ...
           return True
       except Exception as e:
           print(f"Failed to process {subject_id}: {e}")
           return False
   
   # Process multiple subjects safely
   subjects = ["sub-001", "sub-002", "sub-003"]
   successful = [s for s in subjects if process_subject(s)]
   
   print(f"Successfully processed: {len(successful)}/{len(subjects)}")