icvision.compat module
======================

.. automodule:: icvision.compat
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``icvision.compat`` module provides drop-in replacement functions for MNE-ICALabel, allowing ICVision to be used with identical API and output format. This enables seamless migration from ICLabel to ICVision without changing existing code.

Key Functions
-------------

.. autofunction:: icvision.compat.label_components

.. autofunction:: icvision.compat.create_probability_matrix

.. autofunction:: icvision.compat.update_ica_with_icalabel_format

.. autofunction:: icvision.compat.validate_icalabel_compatibility

Utility Functions
-----------------

.. autofunction:: icvision.compat.get_icalabel_class_mapping

.. autofunction:: icvision.compat.get_mne_icalabel_key_mapping

Constants
---------

.. autodata:: icvision.compat.ICALABEL_CLASSES

.. autodata:: icvision.compat.ICVISION_TO_ICALABEL_DISPLAY

.. autodata:: icvision.compat.ICVISION_TO_MNE_ICALABEL