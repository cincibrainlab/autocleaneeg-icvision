# Mouse Validation Findings

This file records the current corpus-backed findings for mouse prompt work.

It combines:

- reviewer notes from the three decision CSVs
- PDF archive membership and per-component report text from the generated ICA reports
- current ICLabel summary labels extracted from the PDFs with `pdftotext`

## What Was Verified Locally

1. The seed review set exists on disk in [mouse-validation-review-set.md](/Users/sueo8x/Documents/Github/autocleaneeg-icvision/plan/mouse-validation-review-set.md).
2. All selected examples map cleanly from CSV entries to matching PDFs by replacing `_data_comp_epo` with `_data_ica_components_all.pdf`.
3. The PDFs are text-extractable and contain per-component pages with `Power Spectrum (1-100Hz)` sections plus component-level classification headers.
4. Across `chirp`, `rest`, and `rest2`, reviewer notes repeatedly identify mouse heart-rate examples that the current ICLabel-style reports classify as `Brain` or `Muscle`, which supports the need for a separate mouse prompt path.

## Comparison Examples

| dataset | entry | IC | reviewer expectation | current PDF label | confidence | interpretation |
|---|---|---|---|---|---:|---|
| chirp | `allego_13__uid1030-15-47-37_data_comp_epo` | `IC3` | clear HR | `Brain` | 0.78 | Human / ICLabel-style behavior misses reviewer-identified HR case. |
| chirp | `allego_13__uid1030-15-47-37_data_comp_epo` | `IC11` | clear HR mixed with muscle | `Brain` | 0.79 | Mixed HR case is not represented by current label. |
| rest | `allego_12__uid0204-14-35-29_data_comp_epo` | `IC2` | perfect heart rate example | `Brain` | 0.75 | Strong mouse HR-positive example mislabeled by current report. |
| rest | `allego_12__uid0204-14-35-29_data_comp_epo` | `IC12` | additional HR-positive support | `Muscle` | 0.44 | Secondary HR example is currently treated as muscle. |
| rest | `allego_12__uid0209-14-00-51_data_comp_epo` | `IC2` | perfect heart rate example | `Brain` | 0.99 | Clear disagreement between reviewer note and current classifier output. |
| rest | `allego_12__uid0209-14-00-51_data_comp_epo` | `IC4` | HR-positive | `Brain` | 0.89 | Another HR example currently labeled as brain. |
| rest | `allego_12__uid0209-14-00-51_data_comp_epo` | `IC6` | HR mixed with muscle | `Muscle` | 0.66 | Mixed case aligns partly with muscle but loses HR signal. |
| rest | `allego_12__uid0209-14-00-51_data_comp_epo` | `IC8` | HR mixed with muscle, very clear PSD | `Brain` | 0.82 | Reviewer specifically calls out PSD evidence; current label does not capture it. |
| rest2 | `allego_12__uid1218-16-44-29_data_comp_epo` | `IC5` | clear heart rate | `Muscle` | 0.48 | HR-positive example currently classified as muscle. |
| rest2 | `allego_12__uid1218-16-44-29_data_comp_epo` | `IC9` | clear heart rate, PSD makes it obvious | `Muscle` | 0.38 | Reviewer note explicitly anchors this to PSD evidence. |
| rest2 | `allego_8__uid1218-15-44-10_data_comp_epo` | `IC0` | debated brain / HR mix with HR more visible at `65-100Hz` | `Brain` | 0.63 | Supports the prompt rule that some mouse HR shows up mainly in high gamma. |
| rest2 | `allego_8__uid1218-15-44-10_data_comp_epo` | `IC1` | debated brain / HR mix with HR more visible at `65-100Hz` | `Brain` | 0.89 | Same pattern as above. |
| rest2 | `allego_8__uid1218-15-44-10_data_comp_epo` | `IC3` | debated brain / HR mix with HR more visible at `65-100Hz` | `Brain` | 0.90 | Supports keeping the `65-100Hz` rule in the mouse prompt. |
| rest2 | `allego_8__uid1218-15-44-10_data_comp_epo` | `IC5` | clear heart rate | `Muscle` | 0.96 | Strong HR-positive example currently called muscle. |
| rest2 | `allego_8__uid1218-15-44-10_data_comp_epo` | `IC18` | clear heart rate mixed with muscle | `Muscle` | 0.79 | Mixed case again shows HR being absorbed into muscle. |

## Rule Status

### Supported By Corpus Notes And Report Structure

- Mouse HR examples exist in all three datasets: `chirp`, `rest`, and `rest2`.
- Split / mixed HR cases exist and should be represented in the prompt.
- Rest2 reviewer notes explicitly support checking `65-100Hz` for mouse HR.
- Current human / ICLabel-style report behavior often collapses reviewer-identified HR examples into `Brain` or `Muscle`, which justifies the mouse-specific prompt policy.

### Still Provisional

- Direct visual confirmation that each selected PSD shows regularly spaced peaks separated by about `7.5-12.5 Hz`.
- Direct visual confirmation that the selected non-HR controls lack those regularly spaced peaks.
- Direct model-vs-model benchmark results comparing `classification_mode="human"` against `classification_mode="mouse"` on the same corpus.

## Current Conclusion

The repo now has enough corpus-backed evidence to justify the mouse prompt rules and the separate `classification_mode="mouse"` path. The remaining open work is empirical benchmark validation with live model inference and deeper visual confirmation of PSD peak spacing across a broader set.
