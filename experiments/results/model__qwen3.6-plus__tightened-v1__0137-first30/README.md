# Run report — `model__qwen3.6-plus__tightened-v1__0137-first30`

**Variable tested:** model: qwen3.6-plus (OpenCode Go via CLI transport; same prompt, same 31 components)

Generated 2026-09-08 08:44 by `run_report.py`. All paths relative to repo root. This directory is self-contained: everything needed to audit or re-analyze this run lives here.


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

- Models run: 1 (qwen3.6-plus)
- API calls per model: one per strip
- Total classifications in this run: 31 components × 1 model(s)

## 6. Strip layout

- Components per strip: **9** (fixed by the strip protocol; final strip of a file may be smaller)
- Strips per recording: 4 for 31 components

## 7. Results breakdown


### qwen3.6-plus

- Raw accuracy: **14/31 = 45.2%**
- Balanced (skew-normalized) accuracy: **28.8%**

| True class | Correct/Total | Recall |
|------------|---------------|--------|
| brain | 8/10 | 80% |
| eye | 0/2 | 0% |
| muscle | 6/17 | 35% |
| heart | 0/2 | 0% |

Predicted-label distribution: brain×20, muscle×8, eye×2, channel_noise×1

Top confusions (truth → prediction):
- muscle → brain: 9
- eye → brain: 2
- brain → muscle: 2
- heart → eye: 1
- heart → brain: 1
- muscle → channel_noise: 1

Dominant failure mode: **muscle read as brain** (9 cases). Model language across these errors leans on topography/spectrum cues that fit the predicted class template; see per-component reasoning below and annotated renders for the visual evidence.

## 8. Most prevalent error modes


### qwen3.6-plus

- Errors: **17/31**; high-confidence errors (confidence ≥0.80): **16**
- Dominant confusion pairs:
  - `muscle` → `brain`: 9
  - `eye` → `brain`: 2
  - `brain` → `muscle`: 2
  - `heart` → `eye`: 1
  - `heart` → `brain`: 1
- Largest prediction-count biases (predicted minus true):
  - `brain`: +10
  - `muscle`: -9
  - `heart`: -2
  - `channel_noise`: +1
- Weakest class recall: `eye` at 0/2 (0%)

## 9. Model justification per component


### qwen3.6-plus

| IC | Truth | Predicted | Conf | Verdict | Model's stated reasoning |
|----|-------|-----------|------|---------|--------------------------|
| 0 | eye | brain | 0.85 | WRONG | Dipolar topography with central distribution, 1/f spectrum, and rhythmic time series consistent with cortical activity. |
| 1 | heart | eye | 0.95 | WRONG | Strong frontal topography, low-frequency dominant spectrum, and time series with large blink-like deflections. |
| 2 | eye | brain | 0.8 | WRONG | Left temporal/frontal dipolar map, 1/f spectrum, and mixed frequency activity typical of brain signals. |
| 3 | brain | brain | 0.85 | OK | Frontal-central dipole, 1/f spectrum, and continuous activity consistent with brain sources. |
| 4 | brain | muscle | 0.75 | WRONG | Focal temporal topography, bursty time series, and spectrum with significant high-frequency power suggest muscle artifact. |
| 5 | heart | brain | 0.8 | WRONG | Frontal dipolar map, spectrum with possible alpha bump, and rhythmic activity pattern. |
| 6 | muscle | channel_noise | 0.85 | WRONG | Highly focal topography resembling a single electrode, erratic high-frequency time series, and broadband spectrum indicate a bad channel. |
| 7 | brain | muscle | 0.9 | WRONG | Broad gradient topography, high-frequency continuous activity, and broadband spectrum are characteristic of muscle noise. |
| 8 | brain | brain | 0.85 | OK | Occipital-parietal dipolar map, 1/f spectrum, and structured activity consistent with posterior brain sources. |
| 9 | brain | brain | 0.8 | OK | Dipolar scalp map with 1/f spectrum and slow wave activity, consistent with a cortical source. |
| 10 | muscle | muscle | 0.9 | OK | Bursty high-frequency activity in time series, broadband spectrum, and temporal scalp focus. |
| 11 | muscle | muscle | 0.9 | OK | Continuous high-frequency noise in time series and ERP image with flat broadband spectrum. |
| 12 | muscle | muscle | 0.9 | OK | Bursty activity with broadband spectrum and focal temporal topography. |
| 13 | muscle | muscle | 0.9 | OK | Bursty activity with broadband spectrum and focal temporal topography. |
| 14 | brain | brain | 0.6 | OK | Large dipolar topography with 1/f spectrum; lacks blink-like deflections or low-frequency dominance typical of eye artifacts. |
| 15 | muscle | brain | 0.9 | WRONG | Occipital-positive dipolar map consistent with alpha rhythm, supported by 1/f spectrum. |
| 16 | brain | brain | 0.8 | OK | Posterior-positive dipolar map with 1/f spectrum and slow wave activity. |
| 17 | muscle | muscle | 0.9 | OK | Bursty high-frequency activity with broadband spectrum and focal temporal map. |
| 18 | muscle | brain | 0.9 | WRONG | Dipolar right frontal topography, 1/f spectrum, rhythmic time series typical of neural activity |
| 19 | muscle | brain | 0.9 | WRONG | Dipolar frontal topography, 1/f spectrum, rhythmic time series typical of neural activity |
| 20 | brain | brain | 0.9 | OK | Dipolar central-parietal topography, 1/f spectrum, rhythmic time series typical of neural activity |
| 21 | muscle | brain | 0.9 | WRONG | Dipolar right frontal topography, 1/f spectrum, rhythmic time series typical of neural activity |
| 22 | muscle | brain | 0.9 | WRONG | Dipolar right temporal topography, 1/f spectrum, rhythmic time series typical of neural activity |
| 23 | muscle | brain | 0.9 | WRONG | Dipolar frontal topography, 1/f spectrum, rhythmic time series typical of neural activity |
| 24 | brain | brain | 0.9 | OK | Large dipolar distribution, 1/f spectrum, rhythmic time series typical of neural activity |
| 25 | muscle | brain | 0.9 | WRONG | Dipolar frontal topography, 1/f spectrum, rhythmic time series typical of neural activity |
| 26 | muscle | brain | 0.9 | WRONG | Dipolar right temporal topography, 1/f spectrum, rhythmic time series typical of neural activity |
| 27 | brain | brain | 0.95 | OK | Dipolar topography, clear 1/f spectrum with alpha peak, and rhythmic time series. |
| 28 | muscle | eye | 0.95 | WRONG | Frontal topography, power concentrated in low frequencies, and blink/saccade-like time series. |
| 29 | muscle | muscle | 0.8 | OK | Spectrum shows broad high-frequency power, time series is spiky/erratic, and topography is localized to muscle-prone areas. |
| 30 | muscle | brain | 0.85 | WRONG | Dipolar topography and 1/f spectral decay, despite some high-frequency noise. |

## 10. Skew-normalized accuracy

- **qwen3.6-plus**: raw 45.2% → balanced **28.8%** (unweighted mean of per-class recalls; every class counts equally regardless of frequency)

Balanced accuracy is the headline metric while the first-30 batch remains imbalanced (muscle-heavy). A stratified manifest would remove the need for normalization; both are supported by the skeleton.


## Provenance

- Manifest: `experiments/manifests/stage0_true_first30.csv` (sha256[:16] `8da094ff3a5ffd46`)
- Model registry: `experiments/models_registry.yaml` (sha256[:16] `5386429e979175ea`)
- Call audit logs: `logs/model__qwen3.6-plus__tightened-v1__0137-first30_<model>.jsonl` (prompt sha, strip sha, raw responses, latency)
- Annotated renders: `annotated/<model>/IC*.png`; strips: `strips/`
