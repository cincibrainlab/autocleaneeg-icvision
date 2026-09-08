# Run report — `model__grok-build-0.1__tightened-v1__0137-first30`

**Variable tested:** model: grok-build-0.1 (OpenCode Zen via CLI transport; same prompt, same 31 components)

Generated 2026-09-08 09:19 by `run_report.py`. All paths relative to repo root. This directory is self-contained: everything needed to audit or re-analyze this run lives here.


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

> Raw accuracy on a skewed batch is dominated by the majority classes. See section 11 for the balanced metric.


## 5. Number of runs

- Models run: 1 (grok-build-0.1)
- API calls per model: one per strip
- Total classifications in this run: 31 components × 1 model(s)

## 6. Strip layout

- Components per strip: **9** (fixed by the strip protocol; final strip of a file may be smaller)
- Strips per recording: 4 for 31 components

## 7. Cost

| Model | API calls | Cost | Basis |
|-------|-----------|------|-------|
| `grok-build-0.1` | 4 | $0.0840 | **actual** (summed from CLI step_finish cost events; Go-subscription models are covered by quota, Zen models are out-of-pocket) |

## 8. Results breakdown


### grok-build-0.1

- Raw accuracy: **9/31 = 29.0%**
- Balanced (skew-normalized) accuracy: **19.4%**

| True class | Correct/Total | Recall |
|------------|---------------|--------|
| brain | 6/10 | 60% |
| eye | 0/2 | 0% |
| muscle | 3/17 | 18% |
| heart | 0/2 | 0% |

Predicted-label distribution: brain×10, eye×10, muscle×5, other_artifact×4, channel_noise×2

Top confusions (truth → prediction):
- muscle → eye: 7
- muscle → other_artifact: 4
- eye → brain: 2
- brain → muscle: 2
- brain → eye: 2
- muscle → channel_noise: 2

Dominant failure mode: **muscle read as eye** (7 cases). Model language across these errors leans on topography/spectrum cues that fit the predicted class template; see per-component reasoning below and annotated renders for the visual evidence.

## 9. Most prevalent error modes


### grok-build-0.1

- Errors: **22/31**; high-confidence errors (confidence ≥0.80): **8**
- Dominant confusion pairs:
  - `muscle` → `eye`: 7
  - `muscle` → `other_artifact`: 4
  - `eye` → `brain`: 2
  - `brain` → `muscle`: 2
  - `brain` → `eye`: 2
- Largest prediction-count biases (predicted minus true):
  - `muscle`: -12
  - `eye`: +8
  - `other_artifact`: +4
  - `heart`: -2
  - `channel_noise`: +2
- Weakest class recall: `eye` at 0/2 (0%)

## 10. Model justification per component


### grok-build-0.1

| IC | Truth | Predicted | Conf | Verdict | Model's stated reasoning |
|----|-------|-----------|------|---------|--------------------------|
| 0 | eye | brain | 0.75 | WRONG | roughly dipolar scalp, clean 1/f spectrum, structured horizontal bands in segment image |
| 1 | heart | eye | 0.95 | WRONG | left-right frontal polarity, step-like time series typical of horizontal eye movement |
| 2 | eye | brain | 0.8 | WRONG | dipolar scalp map, clean 1/f spectrum, consistent segment structure |
| 3 | brain | muscle | 0.7 | WRONG | spiky erratic high-frequency time series, frontal scalp, noisy unstructured segment |
| 4 | brain | muscle | 0.85 | WRONG | highly spiky erratic time series, noisy speckled segment, non-dipolar scalp |
| 5 | heart | brain | 0.6 | WRONG | clean 1/f spectrum, time series lacks clear blink/saccade or high-freq burst pattern |
| 6 | muscle | other_artifact | 0.7 | WRONG | focal non-dipolar scalp spot, single large spike, high-freq power in spectrum, mixed evidence |
| 7 | brain | brain | 0.6 | OK | roughly dipolar scalp map, clean 1/f spectrum |
| 8 | brain | brain | 0.6 | OK | dipolar scalp map, clean 1/f spectrum |
| 9 | brain | brain | 0.9 | OK | Central/vertical dipolar map, rhythmic time series, clean 1/f spectrum decline |
| 10 | muscle | eye | 0.75 | WRONG | Left frontal red focus, initial large deflection, low-freq dominant with non-1/f spectrum |
| 11 | muscle | muscle | 0.85 | OK | Spiky high-frequency time series, spectrum rises/flat at high freq, non-dipolar map |
| 12 | muscle | eye | 0.85 | WRONG | Frontal red/blue focus, slow large deflection at start, strong low-freq power |
| 13 | muscle | muscle | 0.9 | OK | Highly erratic bursty/spiky time series across entire segment, high-frequency power |
| 14 | brain | brain | 0.85 | OK | Roughly dipolar (top yellow, bottom blue), rhythmic waves, clean 1/f spectrum |
| 15 | muscle | brain | 0.7 | WRONG | Rhythmic time series, declining spectrum, map somewhat lateralized but not clearly artifactual |
| 16 | brain | brain | 0.8 | OK | Plausible dipolar-ish map, oscillatory activity, 1/f-like spectrum |
| 17 | muscle | eye | 0.8 | WRONG | Left red frontal focus, initial large deflection, low-frequency heavy spectrum |
| 18 | muscle | other_artifact | 0.65 | WRONG | right frontal focal red without clear dipole or isolated electrode; frontal but no strong blink/saccade structure in time series or segments |
| 19 | muscle | eye | 0.7 | WRONG | frontal/periocular map with left-right polarity; low-freq dominant time series consistent with eye activity |
| 20 | brain | eye | 0.85 | WRONG | strong central-frontal red focus (classic vertical EOG); scattered high-amplitude events in segment image; frontal map + low-freq pattern |
| 21 | muscle | other_artifact | 0.6 | WRONG | edge-focused blue spot on right; initial large deflection but no clear category match; not isolated channel, not dipolar brain, not frontal eye |
| 22 | muscle | eye | 0.75 | WRONG | left frontal red focus (periocular); frontal map with some low-freq content |
| 23 | muscle | eye | 0.75 | WRONG | left frontal red with right blue polarity; periocular location and low-freq spectrum |
| 24 | brain | eye | 0.8 | WRONG | strong left-right polarity across frontal area consistent with horizontal eye movement; dipolar but anterior/frontal rather than central brain |
| 25 | muscle | channel_noise | 0.7 | WRONG | small isolated blue spot on right with minimal spatial spread and no opposite pole; spiky time series; fits single bad electrode better than muscle or other |
| 26 | muscle | channel_noise | 0.65 | WRONG | focal red spot lower right, quite localized with little falloff; late component with no clear dipole or eye pattern |
| 27 | brain | brain | 0.6 | OK | central scalp map with blue focus, clean 1/f spectrum decay, time series relatively smooth without strong artifact patterns |
| 28 | muscle | eye | 0.8 | WRONG | frontal topography (red at top of map), large deflection in time series, low-frequency dominant spectrum |
| 29 | muscle | muscle | 0.75 | OK | spiky/erratic time series, focal edge topography, relatively higher high-frequency power in spectrum |
| 30 | muscle | other_artifact | 0.6 | WRONG | somewhat dipolar map but contradicted by spiky high-frequency bursting time series; spectrum 1/f but does not cleanly fit brain or muscle |

## 11. Skew-normalized accuracy

- **grok-build-0.1**: raw 29.0% → balanced **19.4%** (unweighted mean of per-class recalls; every class counts equally regardless of frequency)

Balanced accuracy is the headline metric while the first-30 batch remains imbalanced (muscle-heavy). A stratified manifest would remove the need for normalization; both are supported by the skeleton.


## Provenance

- Manifest: `experiments/manifests/stage0_true_first30.csv` (sha256[:16] `8da094ff3a5ffd46`)
- Model registry: `experiments/models_registry.yaml` (sha256[:16] `d1580df675d545e9`)
- Call audit logs: `logs/model__grok-build-0.1__tightened-v1__0137-first30_<model>.jsonl` (prompt sha, strip sha, raw responses, latency)
- Annotated renders: `annotated/<model>/IC*.png`; strips: `strips/`
