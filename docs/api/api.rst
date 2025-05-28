API Module
==========

The API module handles communication with OpenAI's Vision API, including batch processing and error handling.

.. automodule:: icvision.api
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

classify_components_batch
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: icvision.api.classify_components_batch

Manages parallel classification of multiple ICA components using OpenAI's Vision API. This function:

1. Splits components into batches for parallel processing
2. Handles concurrent API requests with rate limiting
3. Provides comprehensive error handling and fallbacks
4. Returns detailed classification results

**Example Usage:**

.. code-block:: python

   from icvision.api import classify_components_batch
   
   # Classify components from image files
   image_paths = ["comp_0.png", "comp_1.png", "comp_2.png"]
   component_names = ["IC0", "IC1", "IC2"]
   
   results_df = classify_components_batch(
       image_paths=image_paths,
       component_names=component_names,
       api_key="your_api_key",
       model_name="gpt-4.1",
       batch_size=5,
       max_concurrency=3
   )

classify_component_image_openai
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: icvision.api.classify_component_image_openai

Classifies a single ICA component using OpenAI's Vision API. This is the core classification function that:

1. Encodes the component image for API transmission
2. Sends the classification request with the specified prompt
3. Parses the API response to extract label, confidence, and reasoning
4. Handles API errors and provides fallback classifications

**Example Usage:**

.. code-block:: python

   from icvision.api import classify_component_image_openai
   
   # Classify a single component
   result = classify_component_image_openai(
       image_path="component_IC0.png",
       component_name="IC0", 
       api_key="your_api_key",
       model_name="gpt-4.1",
       prompt="Custom classification prompt..."
   )
   
   print(f"Label: {result['label']}")
   print(f"Confidence: {result['confidence']}")
   print(f"Reason: {result['reason']}")

Internal Helper Functions
-------------------------

.. autofunction:: icvision.api._classify_single_component_wrapper

Error Handling
--------------

The API module includes robust error handling for common issues:

**Rate Limiting**
   Automatic retry with exponential backoff when hitting API rate limits.

**API Errors**
   Graceful handling of API errors with fallback to "unknown" classification.

**Network Issues**
   Retry logic for temporary network connectivity problems.

**Invalid Responses**
   Fallback parsing when API responses don't match expected format.

**Example Error Handling:**

.. code-block:: python

   from icvision.api import classify_component_image_openai
   
   try:
       result = classify_component_image_openai(
           image_path="component.png",
           component_name="IC0",
           api_key="your_api_key"
       )
   except Exception as e:
       print(f"Classification failed: {e}")
       # Fallback result is automatically provided
       result = {
           'label': 'unknown',
           'confidence': 0.0,
           'reason': f'Classification failed: {str(e)}'
       }

Configuration
-------------

API behavior can be configured through parameters:

**Concurrency Control**
   - ``max_concurrency``: Limits parallel requests to prevent rate limiting
   - ``batch_size``: Controls memory usage during processing

**Model Selection**
   - ``model_name``: Choose between available OpenAI vision models
   - ``prompt``: Customize the classification instructions

**Timeout and Retries**
   - Built-in timeout handling for slow API responses
   - Automatic retry logic for transient failures