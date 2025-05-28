Config Module
=============

The config module contains default settings, classification labels, and prompts for ICVision.

.. automodule:: icvision.config
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Constants
-----------------------

DEFAULT_CONFIG
~~~~~~~~~~~~~~

.. autodata:: icvision.config.DEFAULT_CONFIG

The default configuration dictionary containing all standard settings for ICVision.

**Usage:**

.. code-block:: python

   from icvision.config import DEFAULT_CONFIG
   
   # Use default settings
   config = DEFAULT_CONFIG.copy()
   
   # Modify specific settings
   config['confidence_threshold'] = 0.9
   config['batch_size'] = 15

COMPONENT_LABELS
~~~~~~~~~~~~~~~

.. autodata:: icvision.config.COMPONENT_LABELS

List of valid component classification labels:

- **brain**: Neural brain activity (typically preserved)
- **eye**: Eye movement artifacts (blinks, saccades)
- **muscle**: Muscle tension artifacts (jaw, neck, facial)
- **heart**: Cardiac artifacts (ECG contamination)
- **line_noise**: Electrical line noise (50/60 Hz and harmonics)
- **channel_noise**: Bad channel or electrode artifacts
- **other_artifact**: Any other non-brain activity

**Usage:**

.. code-block:: python

   from icvision.config import COMPONENT_LABELS
   
   # Validate label
   def is_valid_label(label):
       return label in COMPONENT_LABELS
   
   # Get artifact labels (exclude 'brain')
   artifact_labels = [label for label in COMPONENT_LABELS if label != 'brain']

OPENAI_ICA_PROMPT
~~~~~~~~~~~~~~~~

.. autodata:: icvision.config.OPENAI_ICA_PROMPT

The default prompt used for OpenAI Vision API classification.

**Customizing the Prompt:**

.. code-block:: python

   from icvision.config import OPENAI_ICA_PROMPT
   
   # Use default prompt
   default_prompt = OPENAI_ICA_PROMPT
   
   # Create custom prompt
   custom_prompt = """
   Classify this EEG component as either:
   - brain: Neural activity
   - artifact: Non-neural activity
   
   Format: Label: [brain/artifact], Confidence: [0.0-1.0], Reason: [explanation]
   """

Model and Processing Configuration
---------------------------------

DEFAULT_MODEL
~~~~~~~~~~~~

.. autodata:: icvision.config.DEFAULT_MODEL

Default OpenAI model used for component classification.

DEFAULT_EXCLUDE_LABELS
~~~~~~~~~~~~~~~~~~~~~

.. autodata:: icvision.config.DEFAULT_EXCLUDE_LABELS

Default list of component labels that should be excluded from the cleaned data.

Label Mapping
------------

ICVISION_TO_MNE_LABEL_MAP
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autodata:: icvision.config.ICVISION_TO_MNE_LABEL_MAP

Mapping dictionary from ICVision classification labels to MNE-Python standard labels.

**Usage:**

.. code-block:: python

   from icvision.config import ICVISION_TO_MNE_LABEL_MAP
   
   # Convert ICVision label to MNE format
   icvision_label = "eye"
   mne_label = ICVISION_TO_MNE_LABEL_MAP.get(icvision_label, "misc")
   print(f"ICVision: {icvision_label} -> MNE: {mne_label}")

Visualization Settings
---------------------

COLOR_MAP
~~~~~~~~~

.. autodata:: icvision.config.COLOR_MAP

Color mapping for different component types used in visualizations.

**Usage:**

.. code-block:: python

   from icvision.config import COLOR_MAP
   
   # Get color for component type
   brain_color = COLOR_MAP.get('brain', 'blue')
   eye_color = COLOR_MAP.get('eye', 'red')

Usage Examples
--------------

**Complete Configuration Setup:**

.. code-block:: python

   from icvision.config import (
       DEFAULT_CONFIG, COMPONENT_LABELS, OPENAI_ICA_PROMPT,
       DEFAULT_MODEL, DEFAULT_EXCLUDE_LABELS
   )
   import os
   
   # Start with defaults
   config = DEFAULT_CONFIG.copy()
   
   # Apply environment overrides
   if 'ICVISION_MODEL' in os.environ:
       config['model_name'] = os.environ['ICVISION_MODEL']
   else:
       config['model_name'] = DEFAULT_MODEL
   
   # Apply custom settings
   config.update({
       'confidence_threshold': 0.9,
       'labels_to_exclude': DEFAULT_EXCLUDE_LABELS.copy(),
       'batch_size': 15
   })
   
   print("Configuration ready:")
   for key, value in config.items():
       print(f"  {key}: {value}")

**Custom Prompt Development:**

.. code-block:: python

   from icvision.config import OPENAI_ICA_PROMPT, COMPONENT_LABELS
   
   # Build custom prompt
   labels_text = ", ".join(COMPONENT_LABELS)
   
   custom_prompt = f"""
   Classify this EEG component into one of: {labels_text}
   
   Base your classification on:
   1. Spatial topography patterns
   2. Temporal characteristics  
   3. Frequency content
   4. Trial-to-trial consistency
   
   Respond with: Label: [label], Confidence: [0.0-1.0], Reason: [explanation]
   """
   
   print("Custom prompt ready for use")