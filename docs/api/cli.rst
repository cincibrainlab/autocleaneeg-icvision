CLI Module
==========

The CLI module provides the command-line interface for ICVision, making it easy to process EEG data from the terminal.

.. automodule:: icvision.cli
   :members:
   :undoc-members:
   :show-inheritance:

Command-Line Interface
----------------------

The main entry point is the ``icvision`` command, which is automatically installed when you install the package.

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   # Basic usage
   icvision /path/to/raw_data.set /path/to/ica_data.fif

   # With output directory
   icvision /path/to/raw_data.set /path/to/ica_data.fif --output-dir results/

   # With custom settings
   icvision /path/to/raw_data.set /path/to/ica_data.fif \
       --model gpt-4.1 \
       --confidence-threshold 0.8 \
       --batch-size 10 \
       --verbose

Available Options
~~~~~~~~~~~~~~~~~

**Input/Output Options:**

``--output-dir, -o``
   Directory to save results (default: ``./icvision_results/``)

``--api-key``
   OpenAI API key (alternatively set ``OPENAI_API_KEY`` environment variable)

**Model Configuration:**

``--model``
   OpenAI model name (default: ``gpt-4.1``)

``--prompt-file``
   Path to file containing custom classification prompt

**Classification Settings:**

``--confidence-threshold``
   Minimum confidence for auto-exclusion (default: 0.8, range: 0.0-1.0)

``--labels-to-exclude``
   Space-separated list of labels to exclude (default: eye muscle heart line_noise channel_noise)

``--no-auto-exclude``
   Disable automatic component exclusion (components will be labeled but not excluded)

**Processing Options:**

``--batch-size``
   Number of components to process in each batch (default: 10)

``--max-concurrency``
   Maximum number of parallel API requests (default: 4)

**Output Options:**

``--no-report``
   Skip PDF report generation

``--report-filename-prefix``
   Prefix for generated report filename (default: icvision_report)

**Other Options:**

``--verbose, -v``
   Enable verbose logging output

``--version``
   Show ICVision version and exit

``--help, -h``
   Show help message and exit

Examples
~~~~~~~~

**Conservative Classification:**

.. code-block:: bash

   icvision data.set ica.fif \
       --confidence-threshold 0.9 \
       --labels-to-exclude eye muscle

**High-Throughput Processing:**

.. code-block:: bash

   icvision data.set ica.fif \
       --batch-size 20 \
       --max-concurrency 8 \
       --no-report

**Custom Prompt:**

.. code-block:: bash

   icvision data.set ica.fif \
       --prompt-file custom_prompt.txt \
       --output-dir custom_results/

**Label Only (No Exclusion):**

.. code-block:: bash

   icvision data.set ica.fif \
       --no-auto-exclude \
       --verbose

Functions
---------

main
~~~~

.. autofunction:: icvision.cli.main

The main entry point for the command-line interface. This function:

1. Parses command-line arguments
2. Validates input files and parameters  
3. Calls the core ``label_components`` function
4. Handles errors and provides user feedback

**Example programmatic usage:**

.. code-block:: python

   import sys
   from icvision.cli import main
   
   # Simulate command-line arguments
   sys.argv = [
       'icvision',
       'data/raw.fif', 
       'data/ica.fif',
       '--output-dir', 'results/',
       '--verbose'
   ]
   
   main()

setup_cli_logging
~~~~~~~~~~~~~~~~~

.. autofunction:: icvision.cli.setup_cli_logging

Configures logging output based on verbosity settings.

Error Handling
--------------

The CLI provides user-friendly error messages for common issues:

**File Not Found:**

.. code-block:: text

   Error: Raw data file not found: /path/to/missing_file.set
   Please check the file path and try again.

**Invalid API Key:**

.. code-block:: text

   Error: OpenAI API key not provided.
   Set OPENAI_API_KEY environment variable or use --api-key option.

**Invalid Parameters:**

.. code-block:: text

   Error: Confidence threshold must be between 0.0 and 1.0, got: 1.5

**Processing Errors:**

.. code-block:: text

   Error: Failed to process components: [error details]
   Check your input files and API connectivity.

Exit Codes
----------

The CLI uses standard exit codes:

- **0**: Success
- **1**: General error (file not found, invalid parameters, etc.)
- **2**: API/network error  
- **3**: Processing error

This allows for easy integration into shell scripts and automated workflows.

Integration with Shell Scripts
------------------------------

**Batch Processing Script:**

.. code-block:: bash

   #!/bin/bash
   
   # Process multiple subjects
   for subject in sub-001 sub-002 sub-003; do
       echo "Processing $subject..."
       
       if icvision "data/${subject}_raw.fif" "data/${subject}_ica.fif" \
           --output-dir "results/$subject/" \
           --verbose; then
           echo "✓ $subject completed successfully"
       else
           echo "✗ $subject failed"
       fi
   done

**Error Handling in Scripts:**

.. code-block:: bash

   #!/bin/bash
   
   # Run ICVision with error handling
   if icvision data.set ica.fif --output-dir results/ 2>error.log; then
       echo "Processing completed successfully"
       echo "Results saved in: results/"
   else
       echo "Processing failed. Check error.log for details."
       exit 1
   fi