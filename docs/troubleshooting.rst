Troubleshooting
===============

This guide helps you solve common issues when using ICVision. If you're new to API keys and environment variables, check our `Installation Guide <installation.html>`_ first.

API Key Issues
--------------

"No API key found" Error
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: ICVision says it can't find your OpenAI API key.

**Error Message Examples**:

.. code-block:: text

   Error: No OpenAI API key found. Please set OPENAI_API_KEY environment variable.
   ValueError: OpenAI API key is required but not found.

**Solutions**:

1. **Check the variable name** - It must be exactly ``OPENAI_API_KEY`` (all uppercase):
   
   .. code-block:: bash
   
      # Check if it's set correctly
      echo $OPENAI_API_KEY        # macOS/Linux
      echo %OPENAI_API_KEY%       # Windows Command Prompt
      echo $env:OPENAI_API_KEY    # Windows PowerShell

2. **Restart your application** after setting the environment variable:
   
   - Close your terminal/command prompt completely
   - Open a new terminal window
   - Try running ICVision again

3. **Verify you set it in the right place**:
   
   **Windows**:
   - System Environment Variables (permanent)
   - User Environment Variables (permanent)
   - Command prompt session (temporary)
   
   **macOS/Linux**:
   - ``~/.zshrc`` file (newer Macs)
   - ``~/.bashrc`` file (older Macs/Linux)
   - Current terminal session (temporary)

4. **Try the direct method** as a test:
   
   .. code-block:: bash
   
      icvision data.set ica.fif --api-key sk-proj-your_actual_key_here

**Still not working?** See our detailed `API Setup Guide <../API_SETUP_GUIDE.md>`_.

"Invalid API key" Error
~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: OpenAI rejects your API key.

**Error Message Examples**:

.. code-block:: text

   Error 401: Incorrect API key provided
   Authentication failed: Invalid API key

**Solutions**:

1. **Check for extra characters**:
   
   - No spaces before or after the key
   - No quotes in the environment variable value
   - Copy the full key (they're 51+ characters long)

2. **Regenerate your API key**:
   
   - Go to `OpenAI Platform <https://platform.openai.com/api-keys>`_
   - Delete the old key
   - Create a new one
   - Update your environment variable

3. **Verify your OpenAI account**:
   
   - Account is active and verified
   - Billing information is set up
   - You have available credits/usage allowance

4. **Test the key directly**:
   
   Create a test file ``test_key.py``:
   
   .. code-block:: python
   
      import openai
      import os
      
      # Get your API key
      api_key = os.environ.get('OPENAI_API_KEY')
      print(f"Key found: {bool(api_key)}")
      
      if api_key:
          print(f"Key starts with: {api_key[:15]}...")
          
          # Try to use it
          try:
              client = openai.OpenAI(api_key=api_key)
              # This will fail if the key is invalid
              models = client.models.list()
              print("✅ API key is valid!")
          except Exception as e:
              print(f"❌ API key test failed: {e}")
      else:
          print("❌ No API key found in environment")

"Rate limit exceeded" Error
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Too many requests to OpenAI API.

**Error Message Examples**:

.. code-block:: text

   Error 429: Rate limit exceeded
   You have exceeded your rate limit

**Solutions**:

1. **Wait a few minutes** and try again
2. **Reduce processing speed**:
   
   .. code-block:: bash
   
      icvision data.set ica.fif \
          --batch-size 5 \
          --max-concurrency 2

3. **Check your OpenAI plan limits**:
   
   - Free tier: Very limited requests per minute
   - Paid tiers: Higher limits based on usage history

4. **Process in smaller chunks**:
   
   .. code-block:: bash
   
      # Process fewer components at a time
      icvision data.set ica.fif --batch-size 3

"Insufficient credits" Error
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Not enough credits in your OpenAI account.

**Solutions**:

1. **Check your billing dashboard**:
   
   - Go to https://platform.openai.com/usage
   - View current usage and limits
   - Add credits if needed

2. **Set up auto-recharge** (recommended for ongoing research)
3. **Monitor usage** to avoid surprises

Installation Issues
-------------------

"pip install autoclean-icvision" Fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Installation command doesn't work.

**Error Types and Solutions**:

**Permission Denied Error**:

.. code-block:: bash

   # Use virtual environment (recommended)
   python -m venv icvision_env
   source icvision_env/bin/activate  # macOS/Linux
   icvision_env\Scripts\activate     # Windows
   pip install autoclean-icvision

   # OR use user installation
   pip install --user icvision

**Python Version Error**:

.. code-block:: bash

   # Check your Python version
   python --version

   # Need Python 3.8 or higher
   # Update Python if necessary

**Network/Firewall Error**:

.. code-block:: bash

   # Try with trusted hosts
   pip install --trusted-host pypi.org --trusted-host pypi.python.org icvision

   # Or check your network/firewall settings

"icvision command not found"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: After installation, ``icvision`` command doesn't work.

**Solutions**:

1. **Check if it's in your PATH**:
   
   .. code-block:: bash
   
      # Find where it was installed
      pip show icvision
      
      # Try running with full path
      python -m icvision.cli --help

2. **Reinstall with user flag**:
   
   .. code-block:: bash
   
      pip uninstall icvision
      pip install --user icvision

3. **Use Python module syntax**:
   
   .. code-block:: bash
   
      # Instead of: icvision data.set ica.fif
      # Use: python -m icvision.cli data.set ica.fif

Data Processing Issues
----------------------

"File not found" Errors
~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: ICVision can't find your data files.

**Solutions**:

1. **Use absolute paths**:
   
   .. code-block:: bash
   
      # Instead of: icvision data.set ica.fif
      # Use full paths:
      icvision /full/path/to/data.set /full/path/to/ica.fif

2. **Check file permissions**:
   
   .. code-block:: bash
   
      # Make sure you can read the files
      ls -la /path/to/your/files/

3. **Verify file formats**:
   
   - Raw data: ``.fif``, ``.set``, ``.edf``, etc.
   - ICA data: ``.fif`` only

"MNE loading errors"
~~~~~~~~~~~~~~~~~~~

**Symptoms**: Problems loading EEG data files.

**Common Solutions**:

1. **Install additional MNE dependencies**:
   
   .. code-block:: bash
   
      pip install mne[complete]

2. **Check file integrity**:
   
   .. code-block:: python
   
      import mne
      
      # Test loading your files
      raw = mne.io.read_raw_fif("your_file.fif", preload=False)
      ica = mne.preprocessing.read_ica("your_ica.fif")

3. **Convert file formats** if needed:
   
   .. code-block:: python
   
      # Convert .set to .fif
      raw = mne.io.read_raw_eeglab("data.set", preload=True)
      raw.save("data.fif", overwrite=True)

"Memory errors"
~~~~~~~~~~~~~~

**Symptoms**: Running out of memory during processing.

**Solutions**:

1. **Reduce batch sizes**:
   
   .. code-block:: bash
   
      icvision data.set ica.fif --batch-size 3

2. **Reduce concurrency**:
   
   .. code-block:: bash
   
      icvision data.set ica.fif --max-concurrency 1

3. **Process components in chunks** (for very large datasets):
   
   .. code-block:: python
   
      # Process subsets of components manually
      from icvision.core import label_components
      
      # Process first 20 components
      results = label_components(
          raw_data="data.fif",
          ica_data="ica.fif",
          component_indices=range(20),
          batch_size=5
      )

Network and Connectivity Issues
-------------------------------

"Connection timeout" Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Requests to OpenAI API time out.

**Solutions**:

1. **Check your internet connection**
2. **Try again later** (OpenAI servers might be busy)
3. **Increase timeout settings** (if available in your version)
4. **Check firewall settings** - make sure HTTPS traffic is allowed

"SSL certificate" Errors
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: SSL/TLS certificate verification fails.

**Solutions**:

1. **Update certificates**:
   
   .. code-block:: bash
   
      # Update pip and certificates
      pip install --upgrade pip certifi

2. **Check corporate firewall** - some organizations block external API calls

Environment and Configuration Issues
------------------------------------

"Wrong Python version" Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: ICVision doesn't work with your Python.

**Solutions**:

1. **Check your Python version**:
   
   .. code-block:: bash
   
      python --version
      # Need 3.8 or higher

2. **Use conda to manage Python versions**:
   
   .. code-block:: bash
   
      conda create -n icvision python=3.11
      conda activate icvision
      pip install autoclean-icvision

3. **Use pyenv** (macOS/Linux):
   
   .. code-block:: bash
   
      pyenv install 3.11.0
      pyenv virtualenv 3.11.0 icvision
      pyenv activate icvision

"Conflicting package versions"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Dependency conflicts during installation or runtime.

**Solutions**:

1. **Use a fresh virtual environment**:
   
   .. code-block:: bash
   
      python -m venv fresh_env
      source fresh_env/bin/activate  # macOS/Linux
      pip install autoclean-icvision

2. **Update all packages**:
   
   .. code-block:: bash
   
      pip install --upgrade pip setuptools wheel
      pip install --upgrade icvision

Getting Help
------------

Self-Diagnosis Checklist
~~~~~~~~~~~~~~~~~~~~~~~~

Before asking for help, try these steps:

.. code-block:: bash

   # 1. Check ICVision installation
   icvision --version
   
   # 2. Check Python version
   python --version
   
   # 3. Check API key
   echo $OPENAI_API_KEY  # Should show your key
   
   # 4. Test API key
   icvision --help  # Should work without API errors
   
   # 5. Check file paths
   ls -la /path/to/your/files/
   
   # 6. Try with minimal settings
   icvision data.set ica.fif --batch-size 1 --max-concurrency 1

Creating a Bug Report
~~~~~~~~~~~~~~~~~~~~~

If you need to report a bug, please include:

1. **Your system information**:
   
   .. code-block:: bash
   
      python --version
      icvision --version
      pip list | grep -E "(icvision|mne|openai)"

2. **Complete error message** (copy and paste the full output)

3. **Command you ran** that caused the error

4. **Sample files** (if possible, create minimal test files that reproduce the issue)

Where to Get Help
~~~~~~~~~~~~~~~~

**For ICVision Issues**:
- GitHub Issues: https://github.com/cincibrainlab/ICVision/issues
- Documentation: https://cincibrainlab.github.io/ICVision/

**For OpenAI API Issues**:
- OpenAI Help: https://help.openai.com/
- OpenAI Status: https://status.openai.com/

**For MNE-Python Issues**:
- MNE Documentation: https://mne.tools/
- MNE Discourse: https://mne.discourse.group/

**For Environment/Python Issues**:
- Python Documentation: https://docs.python.org/
- Your institution's IT support

Quick Reference
---------------

Common Commands for Diagnosis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Check installations
   python --version
   pip --version
   icvision --version
   
   # Check API key
   echo $OPENAI_API_KEY
   
   # Test minimal ICVision run
   icvision --help
   
   # Test with small batch
   icvision data.set ica.fif --batch-size 1 --verbose
   
   # Check file accessibility
   python -c "import mne; print(mne.io.read_raw_fif('your_file.fif', preload=False))"

Typical Cost Estimates
~~~~~~~~~~~~~~~~~~~~~~

To help with budgeting:

- **Small dataset** (20 components): $0.10 - $0.40
- **Medium dataset** (50 components): $0.25 - $1.00  
- **Large dataset** (100+ components): $0.50 - $2.00+

Costs depend on:
- Model used (gpt-4.1 vs gpt-4.1-mini)
- Image complexity
- OpenAI pricing (changes over time)

Remember: You only pay for what you actually process! 💰