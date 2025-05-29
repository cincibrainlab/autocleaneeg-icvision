ICVision Documentation
======================

.. image:: https://badge.fury.io/py/autoclean-icvision.svg
   :target: https://badge.fury.io/py/autoclean-icvision
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/autoclean-icvision.svg
   :target: https://pypi.org/project/autoclean-icvision/
   :alt: Python versions

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

Automated ICA component classification for EEG data using OpenAI's Vision API.

Overview
--------

ICVision automates the tedious process of classifying ICA components from EEG data by generating component visualizations and sending them to OpenAI's Vision API for intelligent artifact identification.

**Workflow**: Raw EEG + ICA → Generate component plots → OpenAI Vision classification → Automated artifact removal → Clean EEG data

Key Features
------------

- Automated classification of 7 component types (brain, eye, muscle, heart, line noise, channel noise, other)
- Multi-panel component plots (topography, time series, PSD, ERP-image)
- MNE-Python integration with ``.fif`` and ``.set`` file support
- Parallel processing with configurable batch sizes
- Command-line and Python API interfaces
- Comprehensive PDF reports and CSV results

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install autoclean-icvision

Set your OpenAI API key:

.. code-block:: bash

   export OPENAI_API_KEY='your_api_key_here'

Basic Usage
~~~~~~~~~~~

Command-line interface:

.. code-block:: bash

   icvision /path/to/your_raw_data.set /path/to/your_ica_decomposition.fif

Python API:

.. code-block:: python

   from icvision.core import label_components

   raw_cleaned, ica_updated, results_df = label_components(
       raw_data="path/to/raw_data.set",
       ica_data="path/to/ica_data.fif",
       output_dir="icvision_output"
   )

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   configuration
   examples
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/api
   api/cli
   api/plotting
   api/utils
   api/config
   api/reports

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`