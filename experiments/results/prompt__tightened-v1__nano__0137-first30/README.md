# Run report — `prompt__tightened-v1__nano__0137-first30`

**Variable tested:** prompt: tightened_v1 with gpt-5.4-nano (same 31 components)

Generated 2026-09-06 13:30 by `run_report.py`. All paths relative to repo root. This directory is self-contained: everything needed to audit or re-analyze this run lives here.


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

> Raw accuracy on a skewed batch is dominated by the majority classes. See section 10 for the balanced metric.


## 5. Number of runs

- Models run: 1 (gpt-5.4-nano)
- API calls per model: one per strip
- Total classifications in this run: 31 components × 1 model(s)

## 6. Strip layout

- Components per strip: **9** (fixed by the strip protocol; final strip of a file may be smaller)
- Strips per recording: 4 for 31 components

## 7. Results breakdown


### gpt-5.4-nano

- Raw accuracy: **9/31 = 29.0%**
- Balanced (skew-normalized) accuracy: **15.3%**

| True class | Correct/Total | Recall |
|------------|---------------|--------|
| brain | 2/10 | 20% |
| eye | 0/2 | 0% |
| muscle | 7/17 | 41% |
| heart | 0/2 | 0% |

Predicted-label distribution: brain×12, muscle×12, eye×3, other_artifact×3, heart×1

Top confusions (truth → prediction):
- muscle → brain: 7
- brain → muscle: 5
- eye → brain: 2
- brain → other_artifact: 2
- muscle → eye: 2
- heart → eye: 1

Dominant failure mode: **muscle read as brain** (7 cases). Model language across these errors leans on topography/spectrum cues that fit the predicted class template; see per-component reasoning below and annotated renders for the visual evidence.

## 8. Most prevalent error modes


### gpt-5.4-nano

- Errors: **22/31**; high-confidence errors (confidence ≥0.80): **3**
- Dominant confusion pairs:
  - `muscle` → `brain`: 7
  - `brain` → `muscle`: 5
  - `eye` → `brain`: 2
  - `brain` → `other_artifact`: 2
  - `muscle` → `eye`: 2
- Largest prediction-count biases (predicted minus true):
  - `muscle`: -5
  - `other_artifact`: +3
  - `brain`: +2
  - `eye`: +1
  - `heart`: -1
- Weakest class recall: `eye` at 0/2 (0%)

## 9. Model justification per component


### gpt-5.4-nano

| IC | Truth | Predicted | Conf | Verdict | Model's stated reasoning |
|----|-------|-----------|------|---------|--------------------------|
| 0 | eye | brain | 0.72 | WRONG | Scalp map is broadly dipolar/central-parietal; time series is relatively smooth (not blink/saccade-like); spectrum shows typical 1/f-like decay with some mid-frequency structure. |
| 1 | heart | eye | 0.9 | WRONG | Strong frontal/periocular topography; time series shows blink-like large deflections; spectrum is dominated by low frequencies (<5 Hz). |
| 2 | eye | brain | 0.6 | WRONG | More dipolar/temporal-parietal distribution with no clear ECG/QRS or 50/60 Hz line; time series looks rhythmic/smooth; spectrum shows 1/f-like falloff with mid-band activity. |
| 3 | brain | muscle | 0.78 | WRONG | Topography is broad but with strong high-frequency/bursty appearance in the segment image; time series is jagged/spiky; spectrum has relatively elevated high-frequency power rather than clean 1/f-only decay. |
| 4 | brain | brain | 0.55 | OK | Central/temporal dipolar-ish map and no decisive blink/ECG/line-noise; time series is moderately smooth; spectrum shows mostly 1/f-like behavior with limited low-frequency dominance. |
| 5 | heart | brain | 0.52 | WRONG | Midline/central distribution with dipolar character; time series lacks clear blink/saccade steps and no QRS-like regularity; spectrum decreases with frequency though not strongly peaked. |
| 6 | muscle | muscle | 0.7 | OK | Topography is focal near the right temporal/temporal-jaw region; time series is relatively spiky; spectrum shows comparatively high-frequency content (less clean alpha/1/f-only). |
| 7 | brain | other_artifact | 0.45 | WRONG | Scalp map is not clearly dipolar and looks more diffuse/edge-like; time series is somewhat irregular; spectrum is not cleanly characteristic of eye/heart/line-noise, suggesting mixed artifact. |
| 8 | brain | heart | 0.8 | WRONG | Time series shows repeated, fairly regular deflection pattern consistent with ~1 Hz cardiac activity; scalp map is broad/diffuse as expected for a distant cardiac source. |
| 9 | brain | muscle | 0.78 | WRONG | Time series is highly jagged/bursty and the spectrum shows strong high-frequency content rather than clean 1/f decay; scalp map is not clearly eye/ERP-like. |
| 10 | muscle | muscle | 0.74 | OK | Broad, non-dipolar scalp distribution with a spiky/erratic time series and power that remains strong at higher frequencies. |
| 11 | muscle | brain | 0.62 | WRONG | Scalp map is roughly dipolar/central-parietal and the time series looks relatively smooth; spectrum shows more typical decreasing trend with some mid-band activity. |
| 12 | muscle | brain | 0.66 | WRONG | More dipolar/temporal-central topography and a less blink-like waveform; spectrum is not dominated by very low-frequency power or a sharp line peak. |
| 13 | muscle | brain | 0.58 | WRONG | Anterior-to-central distribution with comparatively smooth oscillatory-looking waveform; spectrum shows no decisive blink/muscle/line-noise signature. |
| 14 | brain | muscle | 0.8 | WRONG | Very high-frequency-dominated spectrum (broadband) with a noisy, rapidly varying time series; scalp map is not eye-like. |
| 15 | muscle | brain | 0.55 | WRONG | Topography has a plausible dipolar pattern and the waveform is not strongly blink/saccade-like; spectrum lacks a decisive 50/60 Hz peak and is not purely high-frequency muscle. |
| 16 | brain | muscle | 0.77 | WRONG | Time series shows strong fast fluctuations and the spectrum is broadband with elevated high-frequency power; scalp map is not clearly frontal eye. |
| 17 | muscle | muscle | 0.73 | OK | No clear ECG-like regularity or blink-like low-frequency dominance; waveform is noisy and spectrum shows substantial high-frequency power consistent with muscle/EMG. |
| 18 | muscle | muscle | 0.72 | OK | Topography is not clearly dipolar/brain-like and the spectrum/time-frequency show broad high-frequency activity with no clear alpha/ERP structure. |
| 19 | muscle | muscle | 0.7 | OK | More diffuse but with strong high-frequency content in the spectrum/time-frequency and a relatively jagged time series. |
| 20 | brain | brain | 0.55 | OK | Scalp map is roughly dipolar/central-parietal and the spectrum shows a typical 1/f-like falloff with some mid-frequency structure; time series is not blink-like. |
| 21 | muscle | brain | 0.6 | WRONG | Dipolar-ish posterior/temporal distribution with a smoother waveform and a spectrum that is not dominated by very low-frequency eye activity or sharp line noise. |
| 22 | muscle | eye | 0.78 | WRONG | Frontal/periorbital topography with strong low-frequency dominance in the spectrum and a blink-like slow deflection pattern in the time series. |
| 23 | muscle | muscle | 0.74 | OK | Spectrum/time-frequency show broad high-frequency power and the time series is spiky/erratic; scalp map is not a clean brain dipole. |
| 24 | brain | other_artifact | 0.45 | WRONG | Scalp map is broad and non-dipolar with mixed-looking spectral/time-frequency content; no decisive heart/line-noise/eye/muscle signature. |
| 25 | muscle | brain | 0.52 | WRONG | Posterior/central distribution with a relatively smooth time series and a spectrum consistent with neural activity (not strongly low-frequency eye or high-frequency muscle-dominated). |
| 26 | muscle | muscle | 0.66 | OK | Topography is not clearly dipolar and the spectrum/time-frequency show elevated broadband high-frequency power with a more noisy waveform. |
| 27 | brain | muscle | 0.78 | WRONG | Time series is highly jagged/spiky and the spectrum shows relatively strong high-frequency content rather than a clean 1/f falloff; scalp map is not clearly eye- or heart-like. |
| 28 | muscle | eye | 0.7 | WRONG | Scalp topography is strongly frontal/periocular, and the time series/segment image show blink-like large deflections with low-frequency dominance (spectrum concentrated at low frequencies). |
| 29 | muscle | brain | 0.62 | WRONG | More dipolar/central-parietal distribution with a smoother, less blink-like time course; spectrum shows typical EEG-like 1/f decay with some mid-band activity. |
| 30 | muscle | other_artifact | 0.45 | WRONG | Scalp map is focal near left temporal/frontal but not an unmistakable single-electrode island; time series and spectrum are mixed (not clearly line-noise/heart/eye, and not cleanly muscle-dominant). |

## 10. Skew-normalized accuracy

- **gpt-5.4-nano**: raw 29.0% → balanced **15.3%** (unweighted mean of per-class recalls; every class counts equally regardless of frequency)

Balanced accuracy is the headline metric while the first-30 batch remains imbalanced (muscle-heavy). A stratified manifest would remove the need for normalization; both are supported by the skeleton.


## Provenance

- Manifest: `experiments/manifests/stage0_true_first30.csv` (sha256[:16] `8da094ff3a5ffd46`)
- Model registry: `experiments/models_registry.yaml` (sha256[:16] `463a1800a9e69076`)
- Call audit logs: `logs/prompt__tightened-v1__nano__0137-first30_<model>.jsonl` (prompt sha, strip sha, raw responses, latency)
- Annotated renders: `annotated/<model>/IC*.png`; strips: `strips/`
