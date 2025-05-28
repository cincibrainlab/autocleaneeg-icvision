Reports Module
==============

The reports module generates comprehensive PDF reports summarizing component classification results and visualizations.

.. automodule:: icvision.reports
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

generate_classification_report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: icvision.reports.generate_classification_report

Creates a comprehensive PDF report containing classification results, summary statistics, and component visualizations.

**Report Contents:**

1. **Title Page**: Project information and processing timestamp
2. **Summary Statistics**: Overall classification breakdown  
3. **Classification Results Table**: Detailed component-by-component results
4. **Component Visualizations**: Individual plots for each component

**Example Usage:**

.. code-block:: python

   from icvision.reports import generate_classification_report
   import pandas as pd
   import mne
   
   # Load your data
   raw = mne.io.read_raw_fif("data.fif", preload=True)
   ica = mne.preprocessing.read_ica("ica.fif")
   
   # Classification results
   results_df = pd.DataFrame({
       'component_name': ['IC0', 'IC1', 'IC2'],
       'label': ['brain', 'eye', 'muscle'],
       'confidence': [0.95, 0.87, 0.92],
       'reason': ['Clear brain pattern', 'Frontal blink artifact', 'Temporal muscle activity'],
       'exclude_vision': [False, True, True]
   })
   
   # Generate report
   report_path = generate_classification_report(
       ica_obj=ica,
       raw_obj=raw,
       results_df=results_df,
       output_dir="reports/",
       report_filename_prefix="subject01_report"
   )
   
   print(f"Report saved: {report_path}")

Internal Helper Functions
------------------------

_create_summary_table_page
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: icvision.reports._create_summary_table_page

Internal helper function that creates summary table pages for the PDF report.

Configuration and Customization
------------------------------

The reports module provides several ways to customize the generated reports:

**Report Content Control:**

.. code-block:: python

   # Generate report with specific components
   generate_classification_report(
       ica_obj=ica,
       raw_obj=raw,
       results_df=results_df,
       output_dir="reports/",
       components_to_detail="excluded_only"  # Only show excluded components
   )
   
   # Generate report with all components
   generate_classification_report(
       ica_obj=ica,
       raw_obj=raw,
       results_df=results_df,
       output_dir="reports/",
       components_to_detail="all"  # Show all components (default)
   )

**Custom Report Titles:**

.. code-block:: python

   # Custom report naming
   generate_classification_report(
       ica_obj=ica,
       raw_obj=raw,
       results_df=results_df,
       output_dir="reports/",
       report_filename_prefix="custom_analysis"
   )

Report Structure
---------------

The generated PDF report follows this structure:

1. **Cover Page**
   - Report title and generation timestamp
   - Basic processing information

2. **Summary Statistics Page**
   - Total components processed
   - Breakdown by classification label
   - Number of excluded components
   - Average confidence scores

3. **Classification Results Table**
   - Detailed table with all component classifications
   - Component names, labels, confidence scores, and reasoning

4. **Component Visualization Pages**
   - Individual component plots (based on `components_to_detail` setting)
   - Multi-panel visualization for each component
   - Topography, time series, PSD, and ERP-image views

Quality and Performance
----------------------

**Memory Management:**

The reports module is designed to handle large numbers of components efficiently:

- **Sequential Processing**: Components are processed one at a time to minimize memory usage
- **Temporary File Cleanup**: Intermediate files are automatically cleaned up
- **Progress Tracking**: Built-in progress indication for large reports

**Output Quality:**

- **High Resolution**: Components plots are rendered at high DPI for clarity
- **Professional Formatting**: Clean, publication-ready layout
- **Consistent Styling**: Standardized fonts, colors, and layout

Usage Examples
--------------

**Basic Report Generation:**

.. code-block:: python

   from icvision.reports import generate_classification_report
   from icvision.core import label_components
   
   # Run ICVision analysis
   raw_cleaned, ica_updated, results_df = label_components(
       raw_data="data.fif",
       ica_data="ica.fif", 
       output_dir="results/",
       generate_report=False  # We'll generate custom report
   )
   
   # Generate custom report
   report_path = generate_classification_report(
       ica_obj=ica_updated,
       raw_obj=raw_cleaned,
       results_df=results_df,
       output_dir="results/",
       report_filename_prefix="detailed_analysis"
   )

**Batch Report Generation:**

.. code-block:: python

   subjects = ["sub-001", "sub-002", "sub-003"]
   
   for subject_id in subjects:
       # Load results for this subject
       results_df = pd.read_csv(f"results/{subject_id}/icvision_results.csv")
       ica = mne.preprocessing.read_ica(f"results/{subject_id}/icvision_classified_ica.fif")
       raw = mne.io.read_raw_fif(f"data/{subject_id}_raw.fif")
       
       # Generate individual report
       generate_classification_report(
           ica_obj=ica,
           raw_obj=raw,
           results_df=results_df,
           output_dir=f"reports/{subject_id}/",
           report_filename_prefix=f"{subject_id}_icvision_report"
       )
       
       print(f"✓ Report generated for {subject_id}")

Integration with ICVision Workflow
----------------------------------

The reports module integrates seamlessly with the main ICVision workflow:

.. code-block:: python

   from icvision.core import label_components
   
   # ICVision automatically generates reports by default
   raw_cleaned, ica_updated, results_df = label_components(
       raw_data="data.fif",
       ica_data="ica.fif",
       output_dir="results/",
       generate_report=True,  # Default: automatic report generation
       report_filename_prefix="icvision_report"
   )
   
   # Report is automatically saved as:
   # results/icvision_report_all_comps.pdf

Troubleshooting
--------------

**Common Issues and Solutions:**

1. **Large File Sizes**: For reports with many components, consider using `components_to_detail="excluded_only"`
2. **Memory Errors**: Reduce batch sizes or process components in smaller groups
3. **Missing Visualizations**: Ensure component plot images exist in the output directory
4. **PDF Generation Errors**: Check that all required dependencies (matplotlib, reportlab) are installed

**Error Handling:**

.. code-block:: python

   try:
       report_path = generate_classification_report(
           ica_obj=ica,
           raw_obj=raw,
           results_df=results_df,
           output_dir="reports/"
       )
       print(f"✓ Report generated successfully: {report_path}")
   except Exception as e:
       print(f"✗ Report generation failed: {e}")
       # Fallback to basic text summary
       summary = results_df.describe()
       print(summary)