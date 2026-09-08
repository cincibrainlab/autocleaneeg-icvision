# Run report — `model__deepseek-v4-flash-vision-exp__tightened-v1__0137-first30`

**Variable tested:** model: deepseek-v4-flash-vision-exp (OpenCode Go via CLI transport; same prompt, same 31 components)

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

- Models run: 1 (deepseek-v4-flash-vision-exp)
- API calls per model: one per strip
- Total classifications in this run: 31 components × 1 model(s)

## 6. Strip layout

- Components per strip: **9** (fixed by the strip protocol; final strip of a file may be smaller)
- Strips per recording: 4 for 31 components

## 7. Cost

| Model | API calls | Cost | Basis |
|-------|-----------|------|-------|
| `deepseek-v4-flash-vision-exp` | 4 | $0.0122 | **actual** (summed from CLI step_finish cost events; Go-subscription models are covered by quota, Zen models are out-of-pocket) |

## 8. Results breakdown


### deepseek-v4-flash-vision-exp

- Raw accuracy: **17/31 = 54.8%**
- Balanced (skew-normalized) accuracy: **50.1%**

| True class | Correct/Total | Recall |
|------------|---------------|--------|
| brain | 3/10 | 30% |
| eye | 2/2 | 100% |
| muscle | 12/17 | 71% |
| heart | 0/2 | 0% |

Predicted-label distribution: muscle×12, other_artifact×10, eye×5, brain×4

Top confusions (truth → prediction):
- brain → other_artifact: 5
- muscle → other_artifact: 4
- brain → eye: 2
- heart → eye: 1
- heart → other_artifact: 1
- muscle → brain: 1

Dominant failure mode: **brain read as other_artifact** (5 cases). Model language across these errors leans on topography/spectrum cues that fit the predicted class template; see per-component reasoning below and annotated renders for the visual evidence.

## 9. Most prevalent error modes


### deepseek-v4-flash-vision-exp

- Errors: **14/31**; high-confidence errors (confidence ≥0.80): **3**
- Dominant confusion pairs:
  - `brain` → `other_artifact`: 5
  - `muscle` → `other_artifact`: 4
  - `brain` → `eye`: 2
  - `heart` → `eye`: 1
  - `heart` → `other_artifact`: 1
- Largest prediction-count biases (predicted minus true):
  - `other_artifact`: +10
  - `brain`: -6
  - `muscle`: -5
  - `eye`: +3
  - `heart`: -2
- Weakest class recall: `heart` at 0/2 (0%)

## 10. Model justification per component


### deepseek-v4-flash-vision-exp

| IC | Truth | Predicted | Conf | Verdict | Model's stated reasoning |
|----|-------|-----------|------|---------|--------------------------|
| 0 | eye | eye | 0.7 | OK | Frontal-dominant scalp map with large slow spike deflections and low-frequency-dominant spectrum; blink-like. |
| 1 | heart | eye | 0.85 | WRONG | Strong lateralized frontal dipole with clear step-like plateaus separated by fast transitions; decisive horizontal saccade pattern. |
| 2 | eye | eye | 0.65 | OK | Frontal/periocular map with large slow blip deflections and low-frequency-dominant spectrum; blink-like. |
| 3 | brain | other_artifact | 0.55 | WRONG | Splotchy non-dipolar central map with noisy, spiky time series and broad declining spectrum; no clear category. |
| 4 | brain | other_artifact | 0.5 | WRONG | Right temporal/lateral dipole with erratic spiky activity and broad spectrum; no decisive muscle or brain cue. |
| 5 | heart | other_artifact | 0.55 | WRONG | Frontal-pole map with rhythmic high-frequency bursts; neither clear blink/saccade, muscle, nor brain pattern. |
| 6 | muscle | other_artifact | 0.55 | WRONG | Weak/flat non-dipolar map with large slow rhythmic oscillation and low-frequency spectrum; low-variance-looking. |
| 7 | brain | other_artifact | 0.5 | WRONG | Central-looking map but slow irregular deflections and steep low-frequency spectrum without clear alpha peak; mixed evidence. |
| 8 | brain | other_artifact | 0.5 | WRONG | Small central/vertex focus with noisy time series and grainy ERP image; ambiguous, no clear category. |
| 9 | brain | eye | 0.95 | WRONG | Bilateral frontal/orbital negative scalp map with slow large blink deflections in time series and low-frequency-dominated spectrum declining with frequency. |
| 10 | muscle | muscle | 0.65 | OK | Left temporal/frontal focality, spiky erratic time series, and flat/rising high-frequency spectrum consistent with jaw/temporal muscle. |
| 11 | muscle | brain | 0.5 | WRONG | Temporal dipolar topography with smoother, more rhythmic time series and a mid-frequency spectral peak rather than flat high-frequency rise. |
| 12 | muscle | muscle | 0.6 | OK | Left frontal/temporal focus, large spiky bursts, and broad flat high-frequency spectrum. |
| 13 | muscle | muscle | 0.6 | OK | Left frontal focus, spiky bursty activity, and flat/rising high-frequency spectrum. |
| 14 | brain | brain | 0.5 | OK | Central-frontal dipolar map with slow large-amplitude waves and 1/f-declining spectrum; some spikiness creates uncertainty. |
| 15 | muscle | muscle | 0.6 | OK | Left frontal focus, bursty erratic time series, and broad high-frequency spectral power. |
| 16 | brain | eye | 0.95 | WRONG | Bilateral frontal negative topography, slow blink-like deflections, and low-frequency-dominant spectrum. |
| 17 | muscle | muscle | 0.55 | OK | Left frontal focality with moderate spiky activity and broad spectrum; not an isolated single-electrode island. |
| 18 | muscle | other_artifact | 0.4 | WRONG | Diffuse, non-dipolar map; noisy time series with a few sharp spikes; spectrum shows a broad low-mid bump with no clean 1/f or rhythmic alpha. Mixed/unclear source. |
| 19 | muscle | muscle | 0.5 | OK | Erratic, spiky high-frequency time series with a flat/low spectrum and a focal central blob; consistent with bursty muscle activity rather than a clean dipolar brain source. |
| 20 | brain | other_artifact | 0.4 | WRONG | Broad frontal map with a 1/f-like falling spectrum but a highly speckled high-variance segment image; conflicting brain vs muscle evidence, so fallback. |
| 21 | muscle | muscle | 0.6 | OK | Horizontal banding in the segment image (classic sustained muscle signature) with noisy spiky time series and a low flat spectrum. |
| 22 | muscle | muscle | 0.5 | OK | Diffuse low-amplitude map with a spectrum that rises toward high frequencies and an erratic time series; high-frequency broadband pattern. |
| 23 | muscle | muscle | 0.5 | OK | Rising high-frequency spectrum with an erratic time series and a dipolar-but-frontal map near muscle/jaw areas; high-frequency broadband favors muscle over brain. |
| 24 | brain | brain | 0.5 | OK | Clean 1/f falling spectrum, no blink/saccade pattern in the time series, and a plausible frontal map; argues against muscle despite a somewhat broad topology. |
| 25 | muscle | other_artifact | 0.5 | WRONG | Very low variance, nearly uniform quiet segments; diffuse low-amplitude map and low-power spectrum; looks like a late/low-variance residual rather than a clear source. |
| 26 | muscle | other_artifact | 0.4 | WRONG | Small focal red spot on an otherwise diffuse map, quiet segments, and slightly rising high-frequency spectrum; ambiguous and not an unmistakable single-electrode island. |
| 27 | brain | brain | 0.7 | OK | Clear frontal-posterior dipole topography with clean 1/f-like falling spectrum; noisy time series but no blink/saccade structure, consistent with a frontal brain component. |
| 28 | muscle | muscle | 0.55 | OK | Frontal/frontal-lateral scalp flocality with broadband spectrum that rises and stays high rather than 1/f decay; not low-frequency-dominant, so eye is weak; spiky/erratic activity favors muscle. |
| 29 | muscle | muscle | 0.6 | OK | Brain-like central dipole map is contradicted by a rising broadband high-frequency spectrum and spiky/erratic time series, the leading muscle cue. |
| 30 | muscle | muscle | 0.6 | OK | Central dipolar map but spectrum rises to a broadband hump around 20-30 Hz rather than 1/f decay, with noisy/spiky activity; muscle spectral pattern dominates. |

## 11. Skew-normalized accuracy

- **deepseek-v4-flash-vision-exp**: raw 54.8% → balanced **50.1%** (unweighted mean of per-class recalls; every class counts equally regardless of frequency)

Balanced accuracy is the headline metric while the first-30 batch remains imbalanced (muscle-heavy). A stratified manifest would remove the need for normalization; both are supported by the skeleton.


## Provenance

- Manifest: `experiments/manifests/stage0_true_first30.csv` (sha256[:16] `8da094ff3a5ffd46`)
- Model registry: `experiments/models_registry.yaml` (sha256[:16] `d1580df675d545e9`)
- Call audit logs: `logs/model__deepseek-v4-flash-vision-exp__tightened-v1__0137-first30_<model>.jsonl` (prompt sha, strip sha, raw responses, latency)
- Annotated renders: `annotated/<model>/IC*.png`; strips: `strips/`
