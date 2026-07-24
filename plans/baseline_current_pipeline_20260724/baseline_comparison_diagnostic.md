# Baseline comparison diagnostic — Grace-reviewed ICA component labels

Date: 2026-07-24

Reference standard: Grace expert-reviewed labels from `updated_master_file.csv` in `/cblstore/srv/Analysis/Nate_Projects/Projects/IC_Visual_AI`.

File custody check:

- Grace rows: 679
- Unique Grace image filenames: 679
- `training_set.jsonl` + `testing_set.jsonl`: 679 unique filenames
- Physical images in `SavedImages101723/webp_dataset`: 679 `.webp` files
- Missing/extra/duplicate filenames across those sets: 0

## Baseline 1 — Autoclean / ICLabel from saved historical EEGLAB ICA

Artifact: `iclabel_full_baseline.csv`

Method:

- Loaded the 12 saved `.set/.fdt` pairs in `SavedFiles/`.
- Loaded saved EEGLAB ICA decompositions via MNE.
- Ran `mne-icalabel` ICLabel.
- Compared ICLabel prediction for each component number against Grace's label.

Coverage and accuracy:

- Rows attempted: 679
- Rows classified: 679
- Rows failed: 0
- Accuracy vs Grace: 448 / 679 = 65.98%

Speed and token usage:

- Recorded wall time for the full ICLabel extraction/classification pass: 134.7 seconds.
- Mean speed: 0.198 seconds per reviewed component row.
- Token usage: 0. ICLabel is a local/classical classifier in this run, not an LLM call.

Grace label distribution:

- brain: 224
- muscle: 200
- other: 123
- eye: 86
- channel: 29
- heart: 17

ICLabel prediction distribution:

- other: 249
- brain: 191
- muscle: 139
- eye: 47
- channel: 41
- heart: 11
- line_noise: 1

Top confusion/error patterns:

- brain -> other: 71
- muscle -> other: 41
- eye -> other: 35
- muscle -> channel: 14
- other -> brain: 13
- eye -> brain: 9

Method caveat:

MNE/ICLabel warned that the loaded raw data did not appear CAR-referenced, were not detected as filtered 1–100 Hz, and the ICA was imported from EEGLAB rather than fitted as a fresh MNE extended-infomax ICA object. This should be labeled as an ICLabel baseline from saved historical EEGLAB ICA, not a fully standardized fresh Autoclean rerun.

## Baseline 2 — Reconstructed old-prompt GPT vision baseline, `gpt-4-turbo-2024-04-09`

Artifact: `reconstructed_gpt4turbo_baseline.csv`

Method:

- Used the same 679 Grace-reviewed image filenames.
- Used the old `IC_Visual_AI` prompt wording.
- Used the old S3 image URL pattern.
- Used `gpt-4-turbo-2024-04-09` as the older vision-capable model.
- Used the historical key in-process on `cblprod` after explicit approval.

Coverage and accuracy:

- Rows attempted: 679
- Rows successfully classified: 307
- Rows failed: 372
- Successful-subset accuracy vs Grace: 64 / 307 = 20.85%
- If failures are counted as pipeline failures, end-to-end completed-run accuracy is 64 / 679 = 9.43%, but that number mixes model classification with quota/transport failure and should not be used as pure model accuracy.

Failure breakdown:

- 365 rows failed with `http_429` current quota exceeded.
- 7 rows failed with provider `http_500` server errors.

Speed and token usage, successful rows only:

- Successful call latency sum, if run serially: 2,663.342 seconds = 44.39 minutes.
- Mean latency per successful classification: 8.675 seconds.
- Median latency: 7.272 seconds.
- 95th percentile latency: 17.452 seconds.
- Max successful latency: 31.300 seconds.
- Total successful-row token usage: 364,102 prompt tokens + 36,350 completion tokens = 400,452 total tokens.
- Mean token usage per successful classification: 1,186.0 prompt tokens + 118.4 completion tokens = 1,304.4 total tokens.
- Cost was not calculated in this artifact because live model pricing can change and should be pulled from the current official pricing source before using it in planning.

Successful-subset Grace distribution:

- muscle: 87
- other: 76
- brain: 67
- eye: 47
- channel: 20
- heart: 10

Successful-subset GPT prediction distribution:

- brain: 235
- eye: 65
- muscle: 6
- channel: 1
- heart/other: 0

Top GPT confusion patterns on successful rows:

- other -> brain: 60
- muscle -> brain: 53
- eye -> brain: 44
- muscle -> eye: 33
- other -> eye: 14
- heart -> brain: 10
- channel -> eye: 9
- channel -> brain: 8

Interpretation:

The reconstructed old-prompt GPT baseline is currently much worse than ICLabel on the successful subset. It strongly overpredicts `brain`, almost never emits `other` or `heart`, and substantially misses muscle/eye/channel/other distinctions. This is consistent with the old prompt being too underspecified for diagnostic ICA images: it asks for broad labels but does not teach the model visual rules for topography, time course, spectrum, and segment-panel cues.

## Baseline comparison

| Baseline | Coverage | Accuracy vs Grace | Speed | Token usage | Main issue |
|---|---:|---:|---:|---:|---|
| ICLabel from saved EEGLAB ICA | 679/679 | 65.98% | 134.7 sec total; 0.198 sec/component | 0 | Overuses `other`; misses many brain/muscle/eye distinctions |
| GPT-4 Turbo old prompt, successful subset only | 307/679 | 20.85% | 8.675 sec mean successful call; 17.452 sec p95 | 1,304.4 mean total tokens/success | Severe `brain` overprediction; incomplete due quota |

Current conclusion:

The current/original old-prompt GPT vision approach is not competitive with ICLabel as-is. It is less accurate, slower per component, consumes LLM tokens, and is operationally less reliable under the available historical-key route. The biggest immediate opportunity is prompt/image-format improvement plus structured label definitions, not merely swapping models. However, the GPT baseline is incomplete until quota is available to classify the remaining 372 rows.

Next actions:

1. Preserve current partial GPT baseline as incomplete, not final.
2. When quota is available, rerun only failed GPT rows and regenerate the full 679-row GPT accuracy.
3. Use the ICLabel 65.98% result as the complete classical baseline.
4. Use GPT failure modes to guide prompt improvements: reduce brain overprediction, explicitly teach artifact visual signatures, require calibrated confidence, and include an `other/uncertain` policy.
