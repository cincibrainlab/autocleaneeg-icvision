# Terra updated-prompt baseline vs ICLabel baseline — Grace-reviewed ICA components

Date: 2026-07-24

Reference standard: Grace expert-reviewed labels for the same 679 historical IC_Visual_AI component images used in the prior baseline slice.

Scope:

- Baseline EEG/classical comparator: ICLabel run from saved historical EEGLAB ICA files.
- New model comparator: `gpt-5.6-terra` through the governed ClinCog Responses route using the updated diagnostic prompt from `.gate3-integration/prompts/default.txt`.
- No EEG or ICA data were modified. Terra classified already-rendered component images only.
- Row-level labels/predictions remain private because this repository is public. This artifact reports aggregate metrics only.

## Summary

| Metric | ICLabel baseline | Terra updated prompt | Difference, Terra - ICLabel |
|---|---:|---:|---:|
| Components attempted | 679 | 679 | 0 |
| Components completed | 679 | 679 | 0 |
| Failed classifications | 0 | 0 | 0 |
| Correct vs Grace | 448 | 335 | -113 |
| Accuracy vs Grace | 65.98% | 49.34% | -16.64 percentage points |
| Observed wall time | 134.7 sec | 502.2 sec with 8 workers | +367.5 sec |
| Mean per-component call/classification time | 0.198 sec/component | 5.852 sec/model call | +5.654 sec |
| Total LLM tokens | 0 | 963,077 | +963,077 |
| Mean LLM tokens/component | 0 | 1,418.4 | +1,418.4 |

Bottom line: Terra with the updated prompt is much better than the reconstructed old-prompt GPT baseline, but it does not beat the ICLabel baseline on this Grace-reviewed dataset. Accuracy is the priority metric, so ICLabel remains the stronger baseline today.

## Terra operational result

The Terra route itself worked cleanly once pointed at the governed ClinCog host:

- Endpoint profile: `https://ai.clincognition.com/v1/responses`
- Model: `gpt-5.6-terra`
- Prompt: updated diagnostic ICA prompt
- Rows completed: 679/679
- Failures: 0
- Observed concurrent wall time: 502.2 seconds using 8 workers
- Sum of successful call latencies: 3,973.596 seconds
- Mean latency: 5.852 seconds/component
- Median latency: 5.495 seconds/component
- 95th percentile latency: 8.969 seconds/component
- Max latency: 24.289 seconds/component
- Total tokens: 838,565 input + 124,512 output = 963,077 total

This is a major operational improvement over the reconstructed old-prompt GPT-4 Turbo baseline, which only completed 307/679 rows under the historical-key route and failed 372 rows, mostly due quota.

## Per-label comparison

| Label | Support | ICLabel precision | ICLabel recall | ICLabel F1 | Terra precision | Terra recall | Terra F1 | Terra read |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| brain | 224 | 80.10% | 68.30% | 73.73% | 67.34% | 59.82% | 63.36% | Worse than ICLabel on all three metrics. |
| channel_noise | 29 | 39.02% | 55.17% | 45.71% | 13.64% | 82.76% | 23.41% | Better recall, much worse precision; Terra over-calls channel noise. |
| eye | 86 | 82.98% | 45.35% | 58.65% | 83.33% | 17.44% | 28.85% | Similar precision, much worse recall. |
| heart | 17 | 100.00% | 64.71% | 78.57% | n/a | 0.00% | 0.00% | Terra did not emit heart at all. |
| muscle | 200 | 97.12% | 67.50% | 79.65% | 72.73% | 64.00% | 68.09% | Slightly lower recall and substantially lower precision. |
| other_artifact | 123 | 37.75% | 76.42% | 50.54% | 37.78% | 27.64% | 31.92% | Similar precision, much worse recall. |

## Where Terra excels

Against ICLabel, Terra's only clear metric win is `channel_noise` recall:

- ICLabel channel-noise recall: 16/29 = 55.17%.
- Terra channel-noise recall: 24/29 = 82.76%.

That may matter if the near-term goal is to avoid missing bad-channel-like components. But the tradeoff is severe: Terra predicted `channel_noise` 176 times for only 29 true channel-noise cases, so precision fell to 13.64%. In practical review terms, Terra is acting like an aggressive channel-noise screener, not an accurate final classifier.

Terra also excelled operationally compared with the old GPT baseline, not compared with ICLabel:

- Old GPT-4 Turbo reconstructed baseline completed only 307/679 rows.
- Terra completed 679/679 rows with zero failures.
- Old GPT overpredicted `brain`; Terra reduced that specific failure but introduced a different bias toward `channel_noise`.

## Main Terra failure modes

Top Terra confusions:

- muscle -> channel_noise: 54
- other_artifact -> channel_noise: 45
- brain -> channel_noise: 32
- eye -> brain: 29
- brain -> other_artifact: 28
- brain -> muscle: 21
- other_artifact -> muscle: 21
- eye -> channel_noise: 21
- other_artifact -> brain: 18
- eye -> other_artifact: 17
- heart -> brain: 14

Interpretation:

1. The updated prompt made Terra less brain-happy than the old GPT baseline, but it overcorrected toward `channel_noise`.
2. Terra is still poor at eye recall: it recognized only 15/86 true eye components.
3. Terra completely missed the heart class: 0/17 true heart components were labeled heart.
4. Terra's `other_artifact` behavior is too conservative: many true other artifacts are being forced into channel_noise, brain, or muscle.

## Recommendation

Do not treat the current Terra prompt as an accuracy improvement over the EEG/ICLabel baseline. It is a useful governed runtime and a much more reliable LLM route than the old GPT baseline, but its current classification behavior needs another prompt/image-evidence iteration before it can support the publication-oriented accuracy goal.

Minimal next prompt changes to test:

1. Tighten channel-noise criteria: require a truly isolated single-electrode scalp map before using `channel_noise`; otherwise prefer muscle or other_artifact.
2. Strengthen eye criteria: frontal/periocular topography plus slow blink/saccade activity should beat generic brain unless other panels contradict it.
3. Strengthen heart criteria: regular ~1 Hz QRS-like time-series deflections should be decisive even if the scalp map is broad or non-brain-like.
4. Add an explicit uncertainty policy: when cues are mixed, prefer `other_artifact` over forcing channel_noise.
5. Keep this dataset as the fixed validation set; do not tune against the final publication test set without a held-out split.

## Public custody boundary

The row-level Terra CSV, row-level ICLabel CSV, Grace label table, and remote inventories are intentionally not committed because this repository is public and those artifacts include real research filenames/labels. Aggregate metrics and reusable scripts are committed.
