# Mouse Validation Review Set

This file records the initial mouse-mode verification set required by [mouse-mode-plan.md](/Users/sueo8x/Documents/Github/autocleaneeg-icvision/plan/mouse-mode-plan.md).

It is intended to make the prompt rules traceable to concrete repo assets instead of leaving them as tribal knowledge. The rows below are the current seed set for mouse prompt review across the `chirp`, `rest`, and `rest2` datasets.

## Review Set

| dataset | entry | IC / expected behavior | category | source CSV | matching PDF archive | reason included |
|---|---|---|---|---|---|---|
| chirp | `allego_13__uid1030-15-47-37_data_comp_epo` | `IC3` should be `heart` | HR-positive | `autoclean_exclusion_decisions.csv` | `sentinel_chirp_ICA_PDF_reports.zip` | Reviewer note: `IC3 is a near perfect example of clear HR.` |
| chirp | `allego_13__uid1030-15-47-37_data_comp_epo` | `IC11` should be reviewed as HR mixed with muscle | split / mixed HR | `autoclean_exclusion_decisions.csv` | `sentinel_chirp_ICA_PDF_reports.zip` | Reviewer note: `IC11 clear HR mixed with muscle.` |
| chirp | `allego_13__uid0309-14-33-41_data_comp_epo` | no `heart` call expected from notes alone | non-HR control | `autoclean_exclusion_decisions.csv` | `sentinel_chirp_ICA_PDF_reports.zip` | PASS row with empty reviewer notes; use as negative control. |
| rest | `allego_12__uid0204-14-35-29_data_comp_epo` | `IC2` should be `heart`; `IC12` is additional HR-positive support | HR-positive | `autoclean_exclusion_decisions_sentinel_Rest.csv` | `sentinel_rest_ICA_PDF_reports.zip` | Reviewer note: `IC2 is the perfect example of heart rate component in the mouse, IC12 also HR.` |
| rest | `allego_12__uid0209-14-00-51_data_comp_epo` | `IC6` and `IC8` should be reviewed as HR mixed with muscle | split / mixed HR | `autoclean_exclusion_decisions_sentinel_Rest.csv` | `sentinel_rest_ICA_PDF_reports.zip` | Reviewer note explicitly marks mixed HR and calls out a clear PSD. |
| rest | `allego_0__uid0114-11-29-19_data_comp_epo` | no `heart` call expected from notes alone | non-HR control | `autoclean_exclusion_decisions_sentinel_Rest.csv` | `sentinel_rest_ICA_PDF_reports.zip` | PASS row with empty reviewer notes; use as negative control. |
| rest2 | `allego_12__uid1218-16-44-29_data_comp_epo` | `IC5` should be `heart`; `IC9` is supporting HR-positive evidence | HR-positive | `autoclean_exclusion_decisions_sentinel_Rest2.csv` | `sentinel_rest2_ICA_PDF_reports.zip` | Reviewer note: `5 and 9 clear heart rate ... 9's PSD makes it obvious.` |
| rest2 | `allego_8__uid1218-15-44-10_data_comp_epo` | `IC5` should be `heart`; `IC18` is HR mixed with muscle; `IC0/1/3` are high-band HR debate cases | split / mixed HR | `autoclean_exclusion_decisions_sentinel_Rest2.csv` | `sentinel_rest2_ICA_PDF_reports.zip` | Reviewer note explicitly mentions HR appearing in `65-100Hz`, which supports the mouse prompt rule. |
| rest2 | `allego_0__uid1128-16-01-43_data_comp_epo` | no `heart` call expected from notes alone | non-HR control | `autoclean_exclusion_decisions_sentinel_Rest2.csv` | `sentinel_rest2_ICA_PDF_reports.zip` | PASS row with empty reviewer notes; use as negative control. |

## PDF Mapping Rule

Each review-set row maps back to a PDF inside the matching archive by replacing the suffix:

- `_data_comp_epo` -> `_data_ica_components_all.pdf`

Examples:

- `allego_13__uid1030-15-47-37_data_comp_epo` -> `allego_13__uid1030-15-47-37_data_ica_components_all.pdf`
- `allego_12__uid0204-14-35-29_data_comp_epo` -> `allego_12__uid0204-14-35-29_data_ica_components_all.pdf`
- `allego_8__uid1218-15-44-10_data_comp_epo` -> `allego_8__uid1218-15-44-10_data_ica_components_all.pdf`

## Prompt Rules Covered By This Set

- Mouse HR-positive examples with explicit reviewer confirmation
- Mixed / split HR cases where `heart` is distributed across multiple ICs or mixed with muscle
- Rest2 examples where reviewer notes explicitly call out HR evidence in `65-100Hz`
- Non-HR controls from the same datasets to check that `heart` is not over-called

## Current Limitation

This seed set is assembled from the repo CSV reviewer notes and archive contents and is ready for PDF-by-PDF inspection. It does not by itself prove that every row has already been visually re-reviewed in the PDFs.
