Installation
============

This guide will walk you through installing ICVision and setting up everything you need to start analyzing your EEG data.

Requirements
------------

**System Requirements:**
- Python 3.8 or higher
- Internet connection (for OpenAI API)
- 4GB+ RAM recommended
- Windows, macOS, or Linux

**You'll Also Need:**
- OpenAI API key with Vision model access
- Credit card for OpenAI billing (typical cost: $0.50-$2.00 per 100 components)

Installing ICVision
-------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Install ICVision**

Open your terminal or command prompt and run:

.. code-block:: bash

   pip install autocleaneeg-icvision

**Step 2: Verify Installation**

Test that ICVision is installed correctly:

.. code-block:: bash

   icvision --version

You should see the version number displayed.

For Development
~~~~~~~~~~~~~~~

If you want to contribute to ICVision or use the latest development version:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/cincibrainlab/ICVision.git
      cd ICVision

2. Create and activate a virtual environment:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install in editable mode with development dependencies:

   .. code-block:: bash

      pip install -e ".[dev,test,docs]"

4. Install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Setting up OpenAI API
---------------------

⚠️ **Important**: ICVision requires an OpenAI API key to classify EEG components. This section will guide you through the complete setup process.

What is an API Key?
~~~~~~~~~~~~~~~~~~

An API key is like a password that allows ICVision to communicate with OpenAI's AI models. Think of it as:

- **Your digital signature** that identifies your OpenAI account
- **A secure way** for ICVision to send your EEG images to OpenAI for classification
- **Required for billing** - OpenAI needs to know who to charge for the service

Getting Your OpenAI API Key
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Create an OpenAI Account**

1. Go to `OpenAI's platform <https://platform.openai.com/>`_
2. Click "Sign up" (or "Log in" if you have an account)
3. Complete the registration with your email and password
4. Verify your email when prompted

**Step 2: Set Up Billing**

.. note::
   OpenAI requires a payment method, but you only pay for what you use. Research usage is typically very affordable ($0.50-$2.00 per 100 EEG components).

1. In your OpenAI dashboard, go to "Billing"
2. Add a payment method (credit card)
3. Set up usage limits (recommended: start with $5-10/month)
4. Consider setting up usage alerts

**Step 3: Generate Your API Key**

1. Navigate to "API Keys" in the left sidebar
2. Click "Create new secret key"
3. Give it a descriptive name (e.g., "ICVision Research Key")
4. **Copy the key immediately** - you won't be able to see it again!

Your API key will look like this:

.. code-block:: text

   sk-proj-abcd1234efgh5678ijkl9012mnop3456qrst7890uvwx1234yz567890ab

Configuring Your API Key
~~~~~~~~~~~~~~~~~~~~~~~~

You have several options for providing your API key to ICVision. Choose the method that works best for you:

Option 1: Environment Variable (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the **safest and most convenient** method.

**On Windows:**

*Method A: Command Prompt (Temporary)*

.. code-block:: cmd

   set OPENAI_API_KEY=sk-proj-your_actual_key_here

*Method B: PowerShell (Temporary)*

.. code-block:: powershell

   $env:OPENAI_API_KEY="sk-proj-your_actual_key_here"

*Method C: Permanent Setup*

1. Press ``Win + X`` → "System"
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "User variables", click "New"
5. Variable name: ``OPENAI_API_KEY``
6. Variable value: ``sk-proj-your_actual_key_here``
7. Click OK and restart your applications

**On macOS/Linux:**

*Temporary (current session only):*

.. code-block:: bash

   export OPENAI_API_KEY="sk-proj-your_actual_key_here"

*Permanent (recommended):*

.. code-block:: bash

   # For newer Macs (zsh):
   echo 'export OPENAI_API_KEY="sk-proj-your_actual_key_here"' >> ~/.zshrc
   source ~/.zshrc

   # For older Macs/Linux (bash):
   echo 'export OPENAI_API_KEY="sk-proj-your_actual_key_here"' >> ~/.bashrc
   source ~/.bashrc

Option 2: Environment File (.env)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Good for keeping keys organized by project:

1. Navigate to your project directory
2. Create a file named ``.env`` (note the dot at the beginning)
3. Add this line to the file:

   .. code-block:: text

      OPENAI_API_KEY=sk-proj-your_actual_key_here

4. Save the file

.. warning::
   Never share or commit ``.env`` files to version control systems like Git!

Option 3: Command Line (Quick Testing)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For quick testing, you can provide the key directly:

.. code-block:: bash

   icvision data.set ica.fif --api-key sk-proj-your_actual_key_here

.. note::
   This method is less secure as the key appears in your command history.

Testing Your Setup
~~~~~~~~~~~~~~~~~~

**Test 1: Check Environment Variable**

.. code-block:: bash

   # Windows (Command Prompt):
   echo %OPENAI_API_KEY%

   # Windows (PowerShell):
   echo $env:OPENAI_API_KEY

   # macOS/Linux:
   echo $OPENAI_API_KEY

**Expected Result**: You should see your API key printed.

**Test 2: Test ICVision**

.. code-block:: bash

   icvision --help

**Expected Result**: You should see the help message without any API key errors.

**Test 3: Quick Python Test**

Create a file called ``test_api.py``:

.. code-block:: python

   import os

   api_key = os.environ.get('OPENAI_API_KEY')
   if api_key:
       print("✅ API key found!")
       print(f"Key starts with: {api_key[:15]}...")
   else:
       print("❌ No API key found!")
       print("Please set the OPENAI_API_KEY environment variable.")

Run it:

.. code-block:: bash

   python test_api.py

Verifying Installation
---------------------

Complete Installation Test
~~~~~~~~~~~~~~~~~~~~~~~~~

Run this comprehensive test to make sure everything is working:

.. code-block:: bash

   # 1. Check ICVision is installed
   icvision --version

   # 2. Check API key is configured
   icvision --help

   # 3. Check Python environment
   python -c "import icvision; print('ICVision imported successfully!')"

**Expected Results:**
- Version number is displayed
- Help text appears without API errors
- Python import succeeds

Testing with Sample Data
~~~~~~~~~~~~~~~~~~~~~~~

If you have EEG data ready, try a quick test:

.. code-block:: bash

   # Replace with your actual file paths
   icvision /path/to/your_raw_data.set /path/to/your_ica_data.fif --output-dir test_results/

This will:
- Create a ``test_results/`` directory
- Process your data and save results
- Generate a PDF report
- Confirm everything is working end-to-end

Common Issues and Solutions
---------------------------

"No API key found" Error
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: ICVision can't find your API key.

**Solutions**:

1. **Check the variable name**: Must be exactly ``OPENAI_API_KEY`` (all capitals)
2. **Restart your terminal/application** after setting the environment variable
3. **Verify it's set correctly**:
   
   .. code-block:: bash
   
      # Check if the key is there
      echo $OPENAI_API_KEY  # macOS/Linux
      echo %OPENAI_API_KEY%  # Windows

4. **Try the direct method** as a test:
   
   .. code-block:: bash
   
      icvision data.set ica.fif --api-key YOUR_ACTUAL_KEY

"Invalid API key" Error
~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: OpenAI rejects your API key.

**Solutions**:

1. **Check for extra spaces** before/after your key
2. **Regenerate the key** in your OpenAI dashboard
3. **Verify your OpenAI account** is active and has billing set up
4. **Make sure you copied the full key** (they're quite long!)

"Permission denied" or "pip install fails"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Installation fails due to permissions.

**Solutions**:

1. **Use a virtual environment** (recommended):
   
   .. code-block:: bash
   
      python -m venv icvision_env
      source icvision_env/bin/activate  # macOS/Linux
      icvision_env\Scripts\activate     # Windows
      pip install autocleaneeg-icvision

2. **Use user installation**:
   
   .. code-block:: bash
   
      pip install --user icvision

3. **Update pip first**:
   
   .. code-block:: bash
   
      pip install --upgrade pip
      pip install autocleaneeg-icvision

Python Version Issues
~~~~~~~~~~~~~~~~~~~~

**Problem**: ICVision doesn't work with your Python version.

**Solutions**:

1. **Check your Python version**:
   
   .. code-block:: bash
   
      python --version

2. **Use Python 3.8 or higher**. If you have an older version:
   
   - **Install a newer Python** from `python.org <https://python.org>`_
   - **Use conda** if you're in a scientific environment:
     
     .. code-block:: bash
     
        conda create -n icvision python=3.11
        conda activate icvision
        pip install autocleaneeg-icvision

Cost and Usage Management
-------------------------

Understanding Costs
~~~~~~~~~~~~~~~~~~

**Typical Costs**:
- Processing 100 EEG components: $0.50-$2.00
- Depends on model used and image complexity
- You only pay for what you use

**Cost-Saving Tips**:
- Start with smaller datasets to test
- Use ``gpt-4.1-mini`` for testing (cheaper than ``gpt-4.1``)
- Set usage limits in your OpenAI account

**Monitoring Usage**:
1. Check your OpenAI dashboard regularly
2. Set up usage alerts
3. Review monthly bills

Recommended Settings for Researchers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**For Initial Testing**:

.. code-block:: bash

   icvision data.set ica.fif \
       --model gpt-4-vision-preview \
       --batch-size 5 \
       --confidence-threshold 0.9

**For Production Analysis**:

.. code-block:: bash

   icvision data.set ica.fif \
       --model gpt-4.1 \
       --batch-size 10 \
       --confidence-threshold 0.8 \
       --generate-report

Need More Help?
--------------

**For detailed API setup**: See the `API Setup Guide <../API_SETUP_GUIDE.md>`_ for step-by-step instructions with troubleshooting.

**For ICVision issues**: 
- Check the `GitHub Issues <https://github.com/cincibrainlab/ICVision/issues>`_
- Read the documentation at `cincibrainlab.github.io/ICVision <https://cincibrainlab.github.io/ICVision/>`_

**For OpenAI issues**:
- OpenAI Help Center: https://help.openai.com/
- OpenAI Status Page: https://status.openai.com/

**Remember**: The setup might seem complex at first, but once configured, ICVision will work seamlessly for all your EEG analyses! 🚀