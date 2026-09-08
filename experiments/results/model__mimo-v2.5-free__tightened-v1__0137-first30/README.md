# Run report — `model__mimo-v2.5-free__tightened-v1__0137-first30`

**Variable tested:** model: mimo-v2.5-free (OpenCode Go via CLI transport; same prompt, same 31 components)

Generated 2026-09-08 10:02 by `run_report.py`. All paths relative to repo root. This directory is self-contained: everything needed to audit or re-analyze this run lives here.


## 1. Recording(s) used

| File | Components in manifest | Data sha256[:16] |
|------|------------------------|------------------|
| `SavedFiles/0137_VDAudio_ICA.set` | 31 | `82644e29268a7e5e` |

## 2. Component scope

- `0137_VDAudio_ICA`: 31 components — IC0-IC30

Sampling rule for prelim runs: **contiguous first-30** (IC0–IC30 per recording, the high-variance ICA components). Some recordings decompose into fewer ICs; the manifest records exactly which exist.


## 3. Prompt used

- Prompt: `prompts/tightened_v1_strip.txt`
- sha256: `1cf59bda458303e2d56b10e5a3e70dd05c4aa9e55fdad5d7e73f7b42c344e384`
- Source file: `prompts/tightened_v1_strip.txt`

**Full prompt text:**

```text
Classify each of the {n} ICA components shown in this grid (labeled {labels}).

Each component shows:
- Topography map (scalp distribution)
- Time series (first 2.5 seconds)
- ERP-style image (continuous data segments)
- Power spectrum (1-55Hz)

Valid labels and distinguishing cues:
- "brain": Scalp map is roughly dipolar and not eye/edge focused, often central/parietal/temporal.
  Spectrum usually falls with frequency (1/f-like) and may show peaks around 5-30 Hz, especially
  alpha near 10 Hz. Time series is usually smoother/rhythmic rather than blink-like, step-like,
  or high-frequency bursting. A visible ERP/consistent segment pattern can support brain but is
  not sufficient if the scalp map is non-dipolar or artifact-like.
- "eye": Scalp map is frontal/periocular. Blinks often show strong bilateral frontal activity,
  large slow spikes/deflections, and power concentrated below about 5 Hz. Horizontal eye movement
  often shows left-right frontal polarity and step-like activity with stable intervals separated
  by fast saccade transitions. When the scalp map is frontal/periocular AND the time series or
  spectrum shows low-frequency-dominant, blink- or saccade-like structure, prefer eye over a
  generic dipolar brain read or a single-electrode channel_noise read: frontal focality alone is
  weak evidence for channel_noise, and a plausible-looking anterior dipole is not enough to call
  brain if the activity pattern is blink/saccade-like. Only let brain or channel_noise win this
  tie-break when another panel clearly contradicts eye (e.g., a sharp 50/60 Hz peak, a rhythmic
  8-13 Hz alpha spectrum with smooth non-blink activity, or an unambiguous isolated-electrode map).
- "muscle": Power spectrum is the leading clue: broad high-frequency power, often above 20 Hz,
  with a flat or rising high-frequency profile rather than clean 1/f decay. Scalp map may look
  shallow, edge-focused, temporal/jaw/frontal, or locally dipolar near muscle areas. Activity may
  appear spiky, erratic, or bursty.
- "heart": Look for regular ECG/QRS-like deflections about once per second in the time series or
  repeated bands in the segment image. Scalp map is often a broad near-linear gradient from a far
  source. Spectrum is usually not the main evidence. A clear, regular ~1 Hz QRS-like rhythm is
  decisive for heart on its own: do not talk yourself out of heart just because the scalp map is
  broad, diffuse, non-dipolar, or otherwise does not look like a typical brain or artifact
  topology. That kind of broad/diffuse map is itself consistent with a distant cardiac source, not
  evidence against it. When you see the rhythmic deflection pattern, label heart even if you would
  otherwise default to brain, muscle, or other_artifact from the scalp map alone.
- "line_noise": A sharp narrow peak at 50 or 60 Hz is decisive when that frequency is visible.
  Do not call line_noise for a notch/dip at 50/60 Hz or for a component that merely has mild line
  noise mixed into another clearer source.
- "channel_noise": This is a narrow, high-bar category, not a default guess for a noisy- or
  spiky-looking component. Use it ONLY when the scalp map is unambiguously dominated by a single
  isolated electrode: a sharp one-channel "island" with essentially no smooth spatial falloff to
  neighboring sites and no opposite pole. If the map instead shows broad high-frequency spectral
  power, jaw/temporal/frontal focality, or a spiky/erratic time series without a truly isolated
  single-electrode map, prefer muscle instead of channel_noise. If the evidence is ambiguous
  between an isolated bad electrode and a diffuse or unclear source, prefer muscle or
  other_artifact over channel_noise. Reserve channel_noise for cases where the single bad
  electrode is the obvious, unmistakable read.
- "other_artifact": Use when evidence is mixed, non-dipolar, splotchy/multipolar, contradictory,
  very late/low-variance-looking, or does not clearly fit the categories above. Weak alpha or an
  ERP-like pattern alone should not force a brain label if topology is not plausible. When panels
  genuinely conflict, the visual evidence is weak, or no category fits cleanly, other_artifact is
  the correct fallback: do not force channel_noise, muscle, or eye onto an ambiguous component
  just to avoid using other_artifact.

Confidence guidance:
- Use high confidence only when multiple panels agree or one decisive cue is present
  (clear blink/saccade, clear QRS rhythm, sharp 50/60 Hz peak, or isolated bad channel).
- Use lower confidence when cues conflict, the source looks mixed, or the component could plausibly
  be confused with another class.

Respond with JSON array (one object per component, {n} objects total):
{json_example}
```


## 4. Class distribution of the batch (skew report)

| True class | Count | Share |
|------------|-------|-------|
| brain | 10 | 32.3% |
| eye | 2 | 6.5% |
| muscle | 17 | 54.8% |
| heart | 2 | 6.5% |
| channel_noise | 0 | 0.0% |
| other_artifact | 0 | 0.0% |

> Raw accuracy on a skewed batch is dominated by the majority classes. See section 12 for the balanced metric.


## 5. Number of runs

- Models run: 1 (mimo-v2.5-free)
- API calls per model: one per strip
- Total classifications in this run: 31 components × 1 model(s)

## 6. Strip layout

- Components per strip: **9** (fixed by the strip protocol; final strip of a file may be smaller)
- Strips per recording: 4 for 31 components

## 7. Cost

| Model | API calls | Cost | Basis |
|-------|-----------|------|-------|
| `mimo-v2.5-free` | 4 | $0.0052 | **actual** (summed from CLI step_finish cost events; Go-subscription models are covered by quota, Zen models are out-of-pocket) |

## 8. Time

| Model | Strips | Total time | Median/strip | Min | Max |
|-------|--------|------------|--------------|-----|-----|
| `mimo-v2.5-free` | 4 | 131.6 s | 41.0 s | 13.5 s | 42.0 s |

## 9. Results breakdown


### mimo-v2.5-free

- Raw accuracy: **10/31 = 32.3%**
- Balanced (skew-normalized) accuracy: **20.9%**

| True class | Correct/Total | Recall |
|------------|---------------|--------|
| brain | 6/10 | 60% |
| eye | 0/2 | 0% |
| muscle | 4/17 | 24% |
| heart | 0/2 | 0% |

Predicted-label distribution: brain×15, eye×9, muscle×6, other_artifact×1

Top confusions (truth → prediction):
- muscle → brain: 7
- muscle → eye: 5
- brain → eye: 3
- eye → brain: 2
- heart → eye: 1
- heart → muscle: 1

Dominant failure mode: **muscle read as brain** (7 cases). Model language across these errors leans on topography/spectrum cues that fit the predicted class template; see per-component reasoning below and annotated renders for the visual evidence.

## 10. Most prevalent error modes


### mimo-v2.5-free

- Errors: **21/31**; high-confidence errors (confidence ≥0.80): **10**
- Dominant confusion pairs:
  - `muscle` → `brain`: 7
  - `muscle` → `eye`: 5
  - `brain` → `eye`: 3
  - `eye` → `brain`: 2
  - `heart` → `eye`: 1
- Largest prediction-count biases (predicted minus true):
  - `muscle`: -11
  - `eye`: +7
  - `brain`: +5
  - `heart`: -2
  - `other_artifact`: +1
- Weakest class recall: `eye` at 0/2 (0%)

## 11. Model justification per component


### mimo-v2.5-free

| IC | Truth | Predicted | Conf | Verdict | Model's stated reasoning |
|----|-------|-----------|------|---------|--------------------------|
| 0 | eye | brain | 0.75 | WRONG | Dipolar scalp map, smooth time series, organized segment banding, 1/f spectrum with subtle structure |
| 1 | heart | eye | 0.92 | WRONG | Frontal/periocular dipolar map, sharp rhythmic blink-like spikes in time series, low-frequency dominant spectrum |
| 2 | eye | brain | 0.7 | WRONG | Dipolar right temporal-parietal topography, slow rolling oscillations, organized segments, 1/f spectrum |
| 3 | brain | brain | 0.65 | OK | Central dipolar map, rhythmic activity, clean segments, spectrum shows 1/f with possible alpha |
| 4 | brain | brain | 0.7 | OK | Clear dipolar map, ~10 Hz rhythmic activity visible in time series, organized segments, alpha in spectrum |
| 5 | heart | muscle | 0.75 | WRONG | Bursty/erratic time series, elevated high-frequency power (>20 Hz), shallow edge-focused scalp map |
| 6 | muscle | muscle | 0.8 | OK | Highly localized right temporal hot spot, bursty high-frequency activity, elevated broadband high-freq power |
| 7 | brain | brain | 0.72 | OK | Dipolar frontocentral-posterior map, ~10 Hz alpha rhythm, clean segments, 1/f spectrum with alpha peak |
| 8 | brain | eye | 0.9 | WRONG | Strong bilateral frontal positive with posterior negative, large slow blink-like deflections, low-freq dominant spectrum |
| 9 | brain | eye | 0.7 | WRONG | Bilateral frontal scalp map with low-frequency spectral peak (~2 Hz) consistent with blink artifact; time series shows large slow fluctuations though not classic sharp blink deflections |
| 10 | muscle | muscle | 0.65 | OK | Left temporal/dipolar scalp focus with bursty high-frequency time series and relatively flat spectrum consistent with temporal muscle artifact |
| 11 | muscle | brain | 0.85 | WRONG | Right posterior dipolar scalp map with clear alpha-range spectral peak (~10 Hz) and smooth rhythmic time series consistent with neural source |
| 12 | muscle | eye | 0.7 | WRONG | Right frontal/periocular scalp focus with low-frequency spectral dominance and saccade-like transitions in time series suggesting horizontal eye movement |
| 13 | muscle | brain | 0.85 | WRONG | Left posterior dipolar scalp map with alpha-range spectral peak and smooth time series consistent with neural source |
| 14 | brain | eye | 0.9 | WRONG | Classic bilateral frontal blink pattern with large slow deflections, strong low-frequency spectral dominance, and scattered high-amplitude events in ERP image |
| 15 | muscle | brain | 0.75 | WRONG | Left posterior scalp map with alpha-range spectral peak (~10 Hz) and smooth time series consistent with neural source |
| 16 | brain | brain | 0.7 | OK | Broad central scalp distribution with clear alpha peak (~10 Hz) and smooth rhythmic time series suggesting medial brain source |
| 17 | muscle | muscle | 0.55 | OK | Left temporal/posterior focus with bursty noisy time series and broad spectral power; somewhat ambiguous between muscle artifact and noisy brain component |
| 18 | muscle | brain | 0.75 | WRONG | Right frontal/temporal focal scalp map with plausible dipole; time series shows continuous oscillatory activity (no blinks/saccades); power spectrum shows 1/f decay with moderate high-frequency content and no strong low-frequency dominance. |
| 19 | muscle | brain | 0.75 | WRONG | Right temporal focal scalp map with dipolar structure; time series shows relatively smooth rhythmic activity; power spectrum follows 1/f decay without low-frequency eye-like dominance or broad high-frequency muscle power. |
| 20 | brain | brain | 0.9 | OK | Left temporal/parietal broad dipolar scalp map; smooth rhythmic time series; clear alpha peak near 10 Hz in power spectrum with classic 1/f decay; ERP image shows structured segment patterns. |
| 21 | muscle | eye | 0.6 | WRONG | Right frontal/periocular blue scalp focus (consistent with HEOG); time series shows sharp spikes alternating with slower deflections (saccade-like transitions); elevated high-frequency content is present but frontal location and spike/slow structure favor eye over muscle. |
| 22 | muscle | eye | 0.9 | WRONG | Left frontal/periocular scalp focus; large periodic deflections in time series (likely saccades); power spectrum strongly concentrated below 5 Hz; clear horizontal banding in ERP image at deflection times. |
| 23 | muscle | brain | 0.85 | WRONG | Central-parietal dipolar scalp map; smooth oscillatory time series; power spectrum shows 1/f decay with possible alpha-range peak; ERP image shows structured, consistent segment patterns. |
| 24 | brain | muscle | 0.75 | WRONG | Broad, non-dipolar scalp map spanning frontal/bilateral regions; time series shows high-amplitude erratic fluctuations; power spectrum has elevated high-frequency power extending above 20 Hz (beyond 1/f decay), consistent with muscle contamination. |
| 25 | muscle | eye | 0.85 | WRONG | Right frontal/periocular scalp focus; large periodic deflections in time series (~1-2 per second); power spectrum strongly concentrated in low frequencies (<5 Hz); clear horizontal banding in ERP image at deflection times. |
| 26 | muscle | brain | 0.85 | WRONG | Right posterior temporal/occipital focal scalp map; smooth oscillatory time series; clear alpha peak near 10 Hz with 1/f spectral decay; structured ERP image patterns. |
| 27 | brain | brain | 0.9 | OK | Central/parietal dipolar scalp map, smooth oscillatory time series, and clear alpha peak near 10 Hz in power spectrum — classic brain signature. |
| 28 | muscle | eye | 0.9 | WRONG | Strong bilateral frontal scalp map, large slow deflections in time series consistent with blink artifacts, and power spectrum dominated by low frequencies below ~5 Hz. |
| 29 | muscle | other_artifact | 0.6 | WRONG | Left frontal/temporal focal scalp map with mixed evidence — no clear blink/saccade pattern, no alpha peak, no sharp line noise, no isolated electrode. Evidence is ambiguous between brain and artifact. |
| 30 | muscle | muscle | 0.8 | OK | Broad high-frequency power in spectrum (rising/flat above 20 Hz), erratic/bursty time series, and focal left temporal scalp map — characteristic of muscle artifact. |

## 12. Skew-normalized accuracy

- **mimo-v2.5-free**: raw 32.3% → balanced **20.9%** (unweighted mean of per-class recalls; every class counts equally regardless of frequency)

Balanced accuracy is the headline metric while the first-30 batch remains imbalanced (muscle-heavy). A stratified manifest would remove the need for normalization; both are supported by the skeleton.


## Provenance

- Manifest: `experiments/manifests/stage0_true_first30.csv` (sha256[:16] `8da094ff3a5ffd46`)
- Model registry: `experiments/models_registry.yaml` (sha256[:16] `a89f17b039b05c72`)
- Call audit logs: `logs/model__mimo-v2.5-free__tightened-v1__0137-first30_<model>.jsonl` (prompt sha, strip sha, raw responses, latency)
- Annotated renders: `annotated/<model>/IC*.png`; strips: `strips/`
