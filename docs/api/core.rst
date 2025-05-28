Core Module
===========

The core module contains the main functionality for ICVision, including the primary ``label_components`` function that orchestrates the entire workflow.

.. automodule:: icvision.core
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

label_components
~~~~~~~~~~~~~~~~

.. autofunction:: icvision.core.label_components

This is the main entry point for ICVision. It coordinates the entire workflow:

1. Loads raw EEG data and ICA decomposition
2. Generates component visualizations  
3. Classifies components using OpenAI's Vision API
4. Updates the ICA object with labels and exclusions
5. Applies artifact removal to the raw data
6. Saves results and generates reports

**Example Usage:**

.. code-block:: python

   from icvision.core import label_components

   # Basic usage
   raw_cleaned, ica_updated, results_df = label_components(
       raw_data="path/to/raw.fif",
       ica_data="path/to/ica.fif",
       output_dir="results/"
   )

   # Advanced usage with custom parameters
   raw_cleaned, ica_updated, results_df = label_components(
       raw_data="path/to/raw.fif",
       ica_data="path/to/ica.fif",
       output_dir="results/",
       model_name="gpt-4.1",
       confidence_threshold=0.8,
       labels_to_exclude=["eye", "muscle", "heart"],
       batch_size=10,
       max_concurrency=4,
       generate_report=True
   )

**Parameters:**

- **raw_data** (*str, Path, or MNE Raw object*): Input raw EEG data
- **ica_data** (*str, Path, or MNE ICA object*): Input ICA decomposition  
- **output_dir** (*str or Path*): Directory to save results
- **api_key** (*str, optional*): OpenAI API key (uses OPENAI_API_KEY env var if not provided)
- **model_name** (*str, default: "gpt-4.1"*): OpenAI model for classification
- **confidence_threshold** (*float, default: 0.8*): Minimum confidence for auto-exclusion
- **labels_to_exclude** (*list, default: ["eye", "muscle", "heart", "line_noise", "channel_noise"]*): Labels to exclude
- **auto_exclude** (*bool, default: True*): Whether to automatically exclude components
- **batch_size** (*int, default: 10*): Number of components per visualization batch
- **max_concurrency** (*int, default: 4*): Maximum parallel API requests
- **generate_report** (*bool, default: True*): Whether to generate PDF report
- **custom_prompt** (*str, optional*): Custom classification prompt

**Returns:**

- **raw_cleaned** (*MNE Raw object*): Cleaned raw data with artifacts removed
- **ica_updated** (*MNE ICA object*): ICA object with updated labels and exclusions  
- **results_df** (*pandas DataFrame*): Classification results for each component

Internal Helper Functions
-------------------------

.. autofunction:: icvision.core._update_ica_with_classifications

.. autofunction:: icvision.core._apply_artifact_rejection