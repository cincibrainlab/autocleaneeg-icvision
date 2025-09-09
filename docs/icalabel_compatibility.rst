ICLabel Drop-in Replacement
===========================

ICVision can serve as a **drop-in replacement** for MNE-ICALabel with identical API and output format. This means you can upgrade existing ICLabel workflows to use OpenAI Vision API without changing any other code.

Overview
--------

Traditional ICA component classification workflows use MNE-ICALabel, which provides automated classification using a fixed neural network trained in 2019. ICVision enhances this approach by leveraging OpenAI's latest Vision API while maintaining complete compatibility with existing code.

Quick Migration
---------------

**Before (using MNE-ICALabel):**

.. code-block:: python

   from mne_icalabel import label_components

   # Classify components with ICLabel
   result = label_components(raw, ica, method='iclabel')
   print(result['labels'])  # ['brain', 'eye blink', 'other', ...]
   print(ica.labels_scores_.shape)  # (n_components, 7)

**After (using ICVision):**

.. code-block:: python

   from icvision.compat import label_components  # <-- Only line that changes!

   # Classify components with ICVision (same API!)
   result = label_components(raw, ica, method='icvision')
   print(result['labels'])  # Same format: ['brain', 'eye blink', 'other', ...]
   print(ica.labels_scores_.shape)  # Same shape: (n_components, 7)

What You Get
------------

- **🎯 Identical API**: Same function signature, same return format
- **📊 Same Output**: Returns dict with ``'y_pred_proba'`` and ``'labels'`` keys  
- **⚙️ Same ICA Modifications**: Sets ``ica.labels_scores_`` and ``ica.labels_`` exactly like ICLabel
- **🚀 Enhanced Intelligence**: OpenAI Vision API instead of fixed neural network
- **💡 Detailed Reasoning**: Each classification includes explanation (available in full API)

Why Use ICVision over ICLabel?
------------------------------

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Feature
     - ICLabel
     - ICVision
   * - **Classification Method**
     - Fixed neural network (2019)
     - OpenAI Vision API (latest models)
   * - **Accuracy**
     - Good on typical datasets
     - Enhanced with modern vision AI
   * - **Reasoning**
     - No explanations
     - Detailed reasoning for each decision
   * - **Customization**
     - Fixed model
     - Customizable prompts and models
   * - **Updates**
     - Static model
     - Benefits from OpenAI improvements
   * - **API Compatibility**
     - ✅ Original
     - ✅ Drop-in replacement

API Reference
-------------

``label_components(inst, ica, method='icvision')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Drop-in replacement for ``mne_icalabel.label_components``.

**Parameters:**

- ``inst`` : ``mne.io.Raw`` or ``mne.BaseEpochs``
    The data used to fit the ICA
- ``ica`` : ``mne.preprocessing.ICA``
    Fitted ICA object  
- ``method`` : ``str``
    Must be ``'icvision'``

**Returns:**

- ``dict`` with keys:
    - ``'y_pred_proba'`` : ``np.ndarray(n_components,)`` - Max probability per component
    - ``'labels'`` : ``list[str]`` - Predicted class names using ICLabel vocabulary

**ICA Object Modifications:**

- Sets ``ica.labels_scores_`` : ``np.ndarray(n_components, 7)`` - Full probability matrix
- Sets ``ica.labels_`` : ``dict[str, list[int]]`` - Component indices grouped by class

Output Format Compatibility
----------------------------

Return Dictionary
~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = {
       'y_pred_proba': np.array([0.92, 0.78, 0.65, ...]),  # Max probabilities
       'labels': ['brain', 'eye blink', 'other', ...]      # ICLabel class names
   }

ICA Object Attributes
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Full 7-class probability matrix (same as ICLabel)
   ica.labels_scores_  # shape (n_components, 7), dtype float32
   # Classes: brain, muscle_artifact, eye_blink, heart_beat, line_noise, channel_noise, other

   # Component indices grouped by class (same keys as ICLabel)  
   ica.labels_  # dict with keys: brain, muscle, eog, ecg, line_noise, ch_noise, other

Class Mapping
-------------

ICVision classes are automatically mapped to ICLabel vocabulary:

.. list-table::
   :widths: 33 33 34
   :header-rows: 1

   * - ICVision
     - ICLabel Display
     - MNE Key
   * - ``brain``
     - ``brain``
     - ``brain``
   * - ``eye``
     - ``eye blink``
     - ``eog``
   * - ``muscle``
     - ``muscle artifact``
     - ``muscle``
   * - ``heart``
     - ``heart beat``
     - ``ecg``
   * - ``line_noise``
     - ``line noise``
     - ``line_noise``
   * - ``channel_noise``
     - ``channel noise``
     - ``ch_noise``
   * - ``other_artifact``
     - ``other``
     - ``other``

Usage Examples
---------------

Basic Classification
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import mne
   from icvision.compat import label_components

   # Load your data
   raw = mne.io.read_raw_fif('your_data.fif')
   ica = mne.preprocessing.read_ica('your_ica.fif')

   # Classify with ICVision (same API as ICLabel)
   result = label_components(raw, ica, method='icvision')

   print(f"Classified {len(result['labels'])} components")
   print(f"Labels: {result['labels']}")
   print(f"Max probabilities: {result['y_pred_proba']}")

Integration with Existing Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The compatibility layer works seamlessly with existing MNE workflows:

.. code-block:: python

   def analyze_ica_components(raw, ica, method='icvision'):
       """Generic function that works with both ICLabel and ICVision"""
       
       if method == 'icvision':
           from icvision.compat import label_components
       else:
           from mne_icalabel import label_components
       
       # Same API for both!
       result = label_components(raw, ica, method=method)
       
       # Same return format for both
       print(f"Classified {len(result['labels'])} components")
       
       # Same ICA object modifications for both
       brain_components = ica.labels_['brain']
       artifact_components = [idx for key, indices in ica.labels_.items() 
                             if key != 'brain' for idx in indices]
       
       print(f"Brain components: {brain_components}")
       print(f"Artifact components: {artifact_components}")
       
       return result

   # Works with either classifier
   result = analyze_ica_components(raw, ica, method='icvision')

Accessing Full Probability Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # After classification, ICA object has full probability matrix
   result = label_components(raw, ica, method='icvision')

   # Access full 7-class probabilities
   prob_matrix = ica.labels_scores_
   print(f"Probability matrix shape: {prob_matrix.shape}")  # (n_components, 7)

   # Show probabilities for first component
   print("IC0 probabilities:")
   classes = ['brain', 'muscle', 'eye_blink', 'heart', 'line_noise', 'ch_noise', 'other']
   for i, cls in enumerate(classes):
       print(f"  {cls}: {prob_matrix[0, i]:.3f}")

Working with Component Groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # After classification, access components by type
   result = label_components(raw, ica, method='icvision')

   print("Component groupings:")
   for label_type, components in ica.labels_.items():
       if components:  # Only show non-empty groups
           print(f"  {label_type}: {components}")

   # Example: exclude all non-brain components
   non_brain_components = []
   for label_type, components in ica.labels_.items():
       if label_type != 'brain':
           non_brain_components.extend(components)

   ica.exclude = non_brain_components
   print(f"Excluding {len(non_brain_components)} non-brain components")

Two APIs, Same Power
--------------------

ICVision provides **two complementary interfaces**:

1. **Original ICVision API**: Rich output with detailed results and file generation

   .. code-block:: python

      from icvision.core import label_components
      raw_cleaned, ica_updated, results_df = label_components(...)

2. **ICLabel-Compatible API**: Simple output matching ICLabel exactly

   .. code-block:: python

      from icvision.compat import label_components  
      result = label_components(raw, ica, method='icvision')

Choose the API that best fits your workflow - both use the same underlying OpenAI Vision classification.

Probability Matrix Details
---------------------------

ICVision provides single confidence values, which are converted to ICLabel's 7-class probability format:

.. code-block:: python

   # ICVision: brain, confidence=0.9
   # Becomes ICLabel probability matrix row:
   [0.9, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017]
   #  ^     ^      ^      ^      ^      ^      ^
   # brain muscle  eye   heart  line   chan   other

**Algorithm:**

1. Predicted class gets the ICVision confidence value
2. Remaining probability (1 - confidence) is distributed equally across other 6 classes  
3. All rows sum to exactly 1.0

Configuration
-------------

API Key Setup
~~~~~~~~~~~~~

ICVision requires an OpenAI API key:

.. code-block:: bash

   # Set environment variable
   export OPENAI_API_KEY='your_api_key_here'

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

For advanced ICVision features, use the full API:

.. code-block:: python

   # Full ICVision API (not compatible with ICLabel)
   from icvision.core import label_components as icvision_full

   raw_cleaned, ica_updated, results_df = icvision_full(
       raw_data=raw,
       ica_data=ica,
       confidence_threshold=0.9,
       model_name='gpt-4.1-mini',
       custom_prompt='Your custom prompt here'
   )

   # Then convert to ICLabel format if needed
   from icvision.compat import create_probability_matrix, update_ica_with_icalabel_format

   labels = results_df['label'].tolist()
   confidences = results_df['confidence'].tolist()
   prob_matrix = create_probability_matrix(labels, confidences)
   update_ica_with_icalabel_format(ica_updated, labels, confidences, prob_matrix)

Validation and Testing
----------------------

Validate Compatibility
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from icvision.compat import label_components, validate_icalabel_compatibility

   # Run classification
   result = label_components(raw, ica, method='icvision')

   # Validate that ICA object is properly formatted
   is_compatible = validate_icalabel_compatibility(ica)
   print(f"ICLabel compatibility: {'✅ PASS' if is_compatible else '❌ FAIL'}")

Check Output Format
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Verify return format matches ICLabel
   assert isinstance(result, dict)
   assert 'y_pred_proba' in result
   assert 'labels' in result
   assert result['y_pred_proba'].shape == (ica.n_components_,)
   assert len(result['labels']) == ica.n_components_

   # Verify ICA modifications match ICLabel
   assert hasattr(ica, 'labels_scores_')
   assert hasattr(ica, 'labels_')
   assert ica.labels_scores_.shape == (ica.n_components_, 7)
   assert set(ica.labels_.keys()) == {'brain', 'muscle', 'eog', 'ecg', 'line_noise', 'ch_noise', 'other'}

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Missing API Key:**

.. code-block::

   RuntimeError: No OpenAI API key provided

Solution: Set ``OPENAI_API_KEY`` environment variable

**Wrong Method:**

.. code-block::

   ValueError: Unsupported method 'iclabel'

Solution: Use ``method='icvision'`` with ICVision compatibility layer

**Missing Dependencies:**

.. code-block::

   ImportError: No module named 'icvision'

Solution: Install ICVision: ``pip install autocleaneeg-icvision``

Getting Help
~~~~~~~~~~~~

1. Check that your data is properly formatted (same requirements as ICLabel)
2. Ensure OpenAI API key is set and valid
3. Verify that ICVision is properly installed
4. See demo scripts in the repository for working examples