# ICVision baseline status — 2026-07-24

## Autoclean / ICLabel baseline

Status: complete for the Grace-reviewed historical image set.

- Source folder: `/cblstore/srv/Analysis/Nate_Projects/Projects/IC_Visual_AI`
- Grace labels: `updated_master_file.csv`
- EEG/ICA inputs: `SavedFiles/*.set` + matching `*.fdt`
- Rows evaluated: 679 / 679
- Unique saved EEGLAB ICA files: 12
- Baseline method: `mne-icalabel` ICLabel run against saved EEGLAB ICA decompositions loaded through MNE.
- Accuracy against Grace labels: `0.6597938144329897`

Important method note: MNE/ICLabel emitted warnings that the loaded raw objects did not appear to be CAR-referenced, were not detected as filtered 1–100 Hz, and the ICA objects were loaded as imported EEGLAB decompositions rather than freshly fitted extended-infomax MNE ICA objects. Treat this as the “ICLabel from saved historical EEGLAB ICA” baseline, not as a newly standardized Autoclean rerun baseline.

Artifacts:

- `iclabel_full_baseline.csv`
- `iclabel_full_baseline.summary.json`
- `icvision_baseline_file_map.csv`
- `tools/extract_iclabel_baseline_from_eeglab.py`

Top confusion pairs:

- `brain -> brain`: 153
- `muscle -> muscle`: 135
- `other -> other`: 94
- `brain -> other`: 71
- `muscle -> other`: 41
- `eye -> eye`: 39
- `eye -> other`: 35
- `channel -> channel`: 16
- `muscle -> channel`: 14
- `other -> brain`: 13

## Reconstructed old-prompt GPT vision baseline

Status: runner ready, full live run not complete.

Target model: `gpt-4-turbo-2024-04-09`

Do not use `gpt-4.1-mini` for this baseline.

Current blocker: the approved 1Password `Open Ai Key / credential` value was rejected by direct OpenAI, and the tested ClinCog proxy endpoints returned HTTP 403 / 1010 before reaching the model. The historical server scripts contain a direct OpenAI key that previously worked for fine-tuning-job status checks, but using a hardcoded source credential for new live model calls requires explicit approval after disclosure.

Canonical runner candidates:

- `tools/reconstruct_old_pipeline_baseline.py`: reads historical `training_set.jsonl` / `testing_set.jsonl`, strips Grace answer before model call, writes prediction metrics.
- `plans/baseline_current_pipeline_20260724/reconstructed_baseline_runner.py`: reads `updated_master_file.csv`, uses old prompt and S3 URL pattern, writes reconstructed-baseline CSV/summary.

Next safe step: after one approved valid route exists, run a one-row smoke with `gpt-4-turbo-2024-04-09`, then the full 679-row reconstructed baseline.


## Speed and token metrics added for PR slice

- Accuracy remains the primary metric: ICLabel from saved historical EEGLAB ICA reached 448/679 = 65.98% agreement with Grace. The reconstructed old-prompt GPT-4 Turbo run reached 64/307 = 20.85% on successful rows and 64/679 = 9.43% if transport/quota failures are counted as end-to-end failures.
- ICLabel speed: 134.7 seconds wall time for 679 rows, about 0.198 seconds/component, with zero LLM token usage.
- GPT-4 Turbo successful-row speed: mean 8.675 seconds/component, median 7.272 seconds, p95 17.452 seconds.
- GPT-4 Turbo successful-row token usage: 364,102 prompt + 36,350 completion = 400,452 total tokens across 307 successes; mean 1,304.4 total tokens/success.
- Public PR custody: row-level Grace labels, row-level GPT predictions, raw model text, server inventories, and historical-key helper scripts are intentionally withheld from the public repository. The committed artifacts are aggregate findings and reusable scripts only.
