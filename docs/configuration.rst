Configuration
=============

ICVision provides many configuration options to customize the classification behavior and output.

Default Configuration
---------------------

All default settings are defined in ``icvision.config.DEFAULT_CONFIG``:

.. code-block:: python

   from icvision.config import DEFAULT_CONFIG
   print(DEFAULT_CONFIG)

Core Parameters
---------------

Model Settings
~~~~~~~~~~~~~~

``model_name`` (str, default: "gpt-4.1")
   OpenAI model to use for classification. Recommended models:
   
   - ``gpt-4.1``: Latest vision model with best performance
   - ``gpt-4-vision-preview``: Alternative vision model

``confidence_threshold`` (float, default: 0.8)
   Minimum confidence required for automatic component exclusion.
   Components with confidence >= threshold will be excluded if their
   label is in ``labels_to_exclude``.

``custom_prompt`` (str, optional)
   Custom classification prompt. If not provided, uses the default
   prompt from ``icvision.config.OPENAI_ICA_PROMPT``.

Classification Labels
~~~~~~~~~~~~~~~~~~~~

``labels_to_exclude`` (list, default: ["eye", "muscle", "heart", "line_noise", "channel_noise"])
   Component labels that should be automatically excluded when
   confidence >= threshold.

Available labels:

- ``brain``: Neural brain activity (keep)
- ``eye``: Eye movement artifacts (exclude)
- ``muscle``: Muscle artifacts (exclude)  
- ``heart``: Cardiac artifacts (exclude)
- ``line_noise``: Electrical line noise (exclude)
- ``channel_noise``: Bad channel noise (exclude)
- ``other_artifact``: Other artifacts (exclude)

Processing Settings
~~~~~~~~~~~~~~~~~~

``batch_size`` (int, default: 10)
   Number of components to process in each batch for visualization.
   Larger batches use more memory but may be faster.

``max_concurrency`` (int, default: 4)
   Maximum number of parallel API requests to OpenAI.
   Higher values speed up processing but may hit rate limits.

``auto_exclude`` (bool, default: True)
   Whether to automatically exclude components based on
   classification and confidence threshold.

Output Settings
~~~~~~~~~~~~~~

``generate_report`` (bool, default: True)
   Whether to generate a PDF report with component visualizations.

``report_filename_prefix`` (str, default: "icvision_report")
   Prefix for the generated PDF report filename.

``save_classified_ica`` (bool, default: True)
   Whether to save the ICA object with updated labels and exclusions.

``save_results_csv`` (bool, default: True)
   Whether to save classification results to CSV.

Configuration Examples
----------------------

Command-Line Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # High confidence threshold
   icvision data.set ica.fif --confidence-threshold 0.9

   # Exclude only eye artifacts
   icvision data.set ica.fif --labels-to-exclude eye

   # Faster processing with larger batches
   icvision data.set ica.fif --batch-size 20 --max-concurrency 8

   # Custom output directory and no report
   icvision data.set ica.fif --output-dir custom_results/ --no-report

Python API Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from icvision.core import label_components

   # Conservative classification
   results = label_components(
       raw_data="data.set",
       ica_data="ica.fif",
       confidence_threshold=0.95,
       labels_to_exclude=["eye", "muscle"],
       auto_exclude=True
   )

   # High-throughput processing
   results = label_components(
       raw_data="data.set", 
       ica_data="ica.fif",
       batch_size=25,
       max_concurrency=10,
       generate_report=False
   )

   # Custom model and prompt
   custom_prompt = "Classify this EEG component as brain or artifact..."
   results = label_components(
       raw_data="data.set",
       ica_data="ica.fif", 
       model_name="gpt-4-vision-preview",
       custom_prompt=custom_prompt
   )

Custom Prompts
--------------

You can provide custom classification prompts either as a string or from a file:

From String
~~~~~~~~~~~

.. code-block:: python

   custom_prompt = """
   Analyze this EEG ICA component visualization and classify it as one of:
   - brain: Clear neural activity
   - artifact: Any non-brain signal
   
   Provide confidence score and brief reasoning.
   """

   results = label_components(
       raw_data="data.set",
       ica_data="ica.fif",
       custom_prompt=custom_prompt
   )

From File
~~~~~~~~~

.. code-block:: bash

   # Command line
   icvision data.set ica.fif --prompt-file my_prompt.txt

.. code-block:: python

   # Python API
   with open("my_prompt.txt", "r") as f:
       custom_prompt = f.read()
   
   results = label_components(
       raw_data="data.set",
       ica_data="ica.fif", 
       custom_prompt=custom_prompt
   )

Environment Variables
--------------------

``OPENAI_API_KEY``
   Your OpenAI API key. Required unless passed directly to functions.

``ICVISION_DEFAULT_MODEL`` 
   Override the default model globally.

``ICVISION_DEFAULT_CONFIDENCE``
   Override the default confidence threshold globally.

Best Practices
--------------

Model Selection
~~~~~~~~~~~~~~~

- Use ``gpt-4.1`` for best accuracy
- Use ``gpt-4-vision-preview`` if gpt-4.1 is unavailable
- Test different models on a subset of your data

Confidence Thresholds
~~~~~~~~~~~~~~~~~~~~

- Start with 0.8 (default) for balanced precision/recall
- Increase to 0.9+ for higher precision (fewer false exclusions)
- Decrease to 0.7 for higher recall (catch more artifacts)

Batch Processing
~~~~~~~~~~~~~~~

- Use larger batch sizes (15-25) for faster processing
- Reduce batch size if you encounter memory issues
- Monitor API rate limits with high concurrency