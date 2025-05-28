Plotting Module
===============

The plotting module generates comprehensive visualizations of ICA components for classification by OpenAI's Vision API.

.. automodule:: icvision.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

plot_components_batch
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: icvision.plotting.plot_components_batch

Generates visualization images for multiple ICA components in batch. This function:

1. Creates detailed multi-panel plots for each component
2. Saves images in a format suitable for API transmission
3. Handles memory management and matplotlib threading issues
4. Provides progress tracking and error handling

**Example Usage:**

.. code-block:: python

   from icvision.plotting import plot_components_batch
   import mne
   
   # Load your data
   raw = mne.io.read_raw_fif("data.fif", preload=True)
   ica = mne.preprocessing.read_ica("ica.fif")
   
   # Generate component plots
   image_paths = plot_components_batch(
       raw=raw,
       ica=ica,
       component_indices=[0, 1, 2, 3, 4],  # First 5 components
       output_dir="plots/",
       image_format="webp"
   )
   
   print(f"Generated {len(image_paths)} component plots")

plot_component_for_classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: icvision.plotting.plot_component_for_classification

Creates a detailed visualization of a single ICA component with multiple analysis panels:

- **Topography**: Spatial distribution of the component
- **Time Series**: Component activation over time  
- **Power Spectral Density (PSD)**: Frequency content analysis
- **ERP Image**: Trial-by-trial component activity

**Example Usage:**

.. code-block:: python

   from icvision.plotting import plot_component_for_classification
   import mne
   
   # Load data
   raw = mne.io.read_raw_fif("data.fif", preload=True) 
   ica = mne.preprocessing.read_ica("ica.fif")
   
   # Plot single component
   image_path = plot_component_for_classification(
       raw=raw,
       ica=ica,
       component_idx=0,
       output_path="component_IC0.webp",
       title="Component IC0 Analysis"
   )

**Plot Layout:**

The generated plot contains four panels arranged in a 2x2 grid:

.. code-block:: text

   ┌─────────────────┬─────────────────┐
   │   Topography    │   Time Series   │
   │                 │                 │
   ├─────────────────┼─────────────────┤
   │      PSD        │   ERP Image     │
   │                 │                 │
   └─────────────────┴─────────────────┘

**Panel Details:**

1. **Topography (top-left)**: Shows the spatial pattern of the component across electrode locations
2. **Time Series (top-right)**: Displays component activation over a representative time window
3. **PSD (bottom-left)**: Power spectral density showing frequency characteristics
4. **ERP Image (bottom-right)**: Time-frequency representation of component activity

Utility Functions
-----------------

.. autofunction:: icvision.plotting.save_ica_data

.. autofunction:: icvision.plotting.plot_ica_topographies_overview

Configuration Options
--------------------

Image Format Options
~~~~~~~~~~~~~~~~~~~

The plotting functions support multiple image formats:

- **WebP** (recommended): Best compression with good quality
- **PNG**: Lossless but larger files
- **JPEG**: Smaller files but lossy compression

**Example:**

.. code-block:: python

   # High quality for detailed analysis
   plot_component_for_classification(
       raw, ica, 0, "comp.png", image_format="png"
   )
   
   # Optimized for API transmission
   plot_component_for_classification(
       raw, ica, 0, "comp.webp", image_format="webp"
   )

Plot Customization
~~~~~~~~~~~~~~~~~

**Figure Size and DPI:**

.. code-block:: python

   # High resolution plots
   plot_component_for_classification(
       raw, ica, 0, "comp.png",
       figsize=(12, 10),  # Larger figure
       dpi=300           # High DPI
   )

**Time Window Selection:**

.. code-block:: python

   # Custom time window for time series
   plot_component_for_classification(
       raw, ica, 0, "comp.png",
       tmin=0.0,    # Start time
       tmax=10.0    # End time (seconds)
   )

**Frequency Range:**

.. code-block:: python

   # Focus on specific frequency range
   plot_component_for_classification(
       raw, ica, 0, "comp.png", 
       fmin=1.0,    # Minimum frequency (Hz)
       fmax=40.0    # Maximum frequency (Hz)
   )

Performance Considerations
-------------------------

Memory Management
~~~~~~~~~~~~~~~~

The plotting module includes several optimizations for handling large datasets:

- **Sequential Processing**: Components are plotted one at a time to minimize memory usage
- **Resource Cleanup**: Matplotlib figures and resources are properly cleaned up
- **Backend Management**: Ensures appropriate matplotlib backend for headless operation

**Memory-Efficient Batch Processing:**

.. code-block:: python

   # Process large numbers of components efficiently
   import numpy as np
   
   all_components = list(range(ica.n_components_))
   batch_size = 10
   
   for i in range(0, len(all_components), batch_size):
       batch = all_components[i:i + batch_size]
       
       image_paths = plot_components_batch(
           raw=raw,
           ica=ica, 
           component_indices=batch,
           output_dir=f"plots/batch_{i//batch_size}/",
           cleanup_after_batch=True  # Clean up after each batch
       )

Threading Considerations
~~~~~~~~~~~~~~~~~~~~~~~

Matplotlib can have issues with threading, especially in Jupyter notebooks or multi-threaded applications. The plotting module handles this by:

- Setting appropriate matplotlib backends
- Using thread-safe operations
- Proper resource cleanup

**Safe Usage in Threads:**

.. code-block:: python

   import threading
   from icvision.plotting import plot_component_for_classification
   
   def plot_component_thread(component_idx):
       # This is safe due to internal thread handling
       plot_component_for_classification(
           raw, ica, component_idx, f"comp_{component_idx}.webp"
       )
   
   # Create multiple threads (use with caution)
   threads = []
   for i in range(5):
       t = threading.Thread(target=plot_component_thread, args=(i,))
       threads.append(t)
       t.start()
   
   for t in threads:
       t.join()

Quality Settings
---------------

For optimal classification results, the plotting module uses settings optimized for OpenAI's Vision API:

**Default Settings:**
- Figure size: 10x8 inches
- DPI: 100 (good balance of quality and file size)
- Image format: WebP (best compression)
- Color scheme: High contrast for better visibility

**Custom Quality Settings:**

.. code-block:: python

   # High quality for detailed inspection
   plot_component_for_classification(
       raw, ica, 0, "high_quality.png",
       figsize=(16, 12),
       dpi=200,
       image_format="png"
   )
   
   # Optimized for API speed
   plot_component_for_classification(
       raw, ica, 0, "api_optimized.webp", 
       figsize=(8, 6),
       dpi=80,
       image_format="webp"
   )