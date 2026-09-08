# Run report — `prompt__tightened-v1-vs-strip-default__nano__0137-first30`

**Variable tested:** prompt: tightened_v1 vs strip_default (same nano model, same 31 components)

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

- Models run: 1 (gpt-5.4-nano)
- API calls per model: one per strip
- Total classifications in this run: 31 components × 1 model(s)

## 6. Strip layout

- Components per strip: **9** (fixed by the strip protocol; final strip of a file may be smaller)
- Strips per recording: 4 for 31 components

## 7. Cost

| Model | API calls | Cost | Basis |
|-------|-----------|------|-------|
| `gpt-5.4-nano` | 4 | ~$0.0078 | **estimated** (Zen pricing $0.2/$1M in, $1.25/$1M out × ~6000in/600out tokens per strip; images dominate input; actual cost not metered by this API path) |

## 8. Time

| Model | Strips | Total time | Median/strip | Min | Max |
|-------|--------|------------|--------------|-----|-----|
| `gpt-5.4-nano` | 4 | 18.4 s | 4.6 s | 3.6 s | 5.7 s |

## 9. Results breakdown


### gpt-5.4-nano

- Raw accuracy: **7/31 = 22.6%**
- Balanced (skew-normalized) accuracy: **12.4%**

| True class | Correct/Total | Recall |
|------------|---------------|--------|
| brain | 2/10 | 20% |
| eye | 0/2 | 0% |
| muscle | 5/17 | 29% |
| heart | 0/2 | 0% |

Predicted-label distribution: muscle×12, brain×11, eye×6, other_artifact×2

Top confusions (truth → prediction):
- brain → muscle: 7
- muscle → brain: 6
- muscle → eye: 4
- eye → brain: 2
- muscle → other_artifact: 2
- heart → eye: 1

Dominant failure mode: **brain read as muscle** (7 cases). Model language across these errors leans on topography/spectrum cues that fit the predicted class template; see per-component reasoning below and annotated renders for the visual evidence.

## 10. Most prevalent error modes


### gpt-5.4-nano

- Errors: **24/31**; high-confidence errors (confidence ≥0.80): **2**
- Dominant confusion pairs:
  - `brain` → `muscle`: 7
  - `muscle` → `brain`: 6
  - `muscle` → `eye`: 4
  - `eye` → `brain`: 2
  - `muscle` → `other_artifact`: 2
- Largest prediction-count biases (predicted minus true):
  - `muscle`: -5
  - `eye`: +4
  - `other_artifact`: +2
  - `heart`: -2
  - `brain`: +1
- Weakest class recall: `eye` at 0/2 (0%)

## 11. Model justification per component


### gpt-5.4-nano

| IC | Truth | Predicted | Conf | Verdict | Model's stated reasoning |
|----|-------|-----------|------|---------|--------------------------|
| 0 | eye | brain | 0.62 | WRONG | Scalp map shows a broad, non-eye dipolar-like pattern (central/parietal). Time series is relatively smooth without blink-like steps; spectrum shows typical 1/f-like decay with no decisive line-noise peak. |
| 1 | heart | eye | 0.78 | WRONG | Topography is strongly frontal/peri-ocular. Time series shows blink/saccade-like sharp deflections and the spectrum is dominated by low frequencies (no clear high-frequency muscle profile). |
| 2 | eye | brain | 0.6 | WRONG | More central/temporal dipolar distribution rather than peri-ocular. Time series is not dominated by blink-like spikes; spectrum shows broadband 1/f-like behavior with no sharp 50/60 Hz peak. |
| 3 | brain | muscle | 0.7 | WRONG | Scalp map is broad but the time series is highly jagged/spiky and the spectrum is relatively high-frequency-heavy (less clean 1/f falloff), consistent with muscle activity. |
| 4 | brain | muscle | 0.66 | WRONG | Temporal/frontal-leaning distribution with a noisy, irregular time course. Power spectrum does not show a clean alpha-like peak and appears more high-frequency/broadband. |
| 5 | heart | brain | 0.55 | WRONG | Topography resembles a central dipolar pattern. Time series is moderately rhythmic/smooth compared with muscle; spectrum shows mostly decreasing power with frequency (no decisive blink or line-noise cue). |
| 6 | muscle | other_artifact | 0.45 | WRONG | Topography is localized but not clearly an isolated single-electrode island; time series and spectrum look mixed (not a clean eye blink, heart rhythm, or clear muscle dominance). |
| 7 | brain | eye | 0.74 | WRONG | Frontal/peri-ocular topography with low-frequency-dominant spectrum and step-like/deflection structure in the time series consistent with eye movements/blinks. |
| 8 | brain | brain | 0.6 | OK | Scalp map shows a dipolar-like central source rather than peri-ocular. Time series is not dominated by blink/saccade spikes; spectrum shows typical broadband decay without a sharp 50/60 Hz line-noise peak. |
| 9 | brain | muscle | 0.78 | WRONG | Time series is highly jagged/bursty and the spectrum shows strong high-frequency content rather than clean 1/f/alpha dominance; scalp map is not clearly eye/ECG-like. |
| 10 | muscle | muscle | 0.74 | OK | No clear blink/ECG rhythm; waveform is spiky and spectrum appears relatively high-frequency-heavy with no sharp 50/60 Hz line. |
| 11 | muscle | muscle | 0.7 | OK | Scalp topography is focal (right/temporal) and the time series is irregular with elevated high-frequency power in the spectrum. |
| 12 | muscle | muscle | 0.76 | OK | Temporal/jaw-like focality in the scalp map plus bursty, high-frequency-dominant waveform and spectrum. |
| 13 | muscle | brain | 0.62 | WRONG | More dipolar/central-parietal-like scalp distribution and a smoother, sustained time course; spectrum shows more typical decreasing trend with some mid-band activity rather than broad high-frequency dominance. |
| 14 | brain | muscle | 0.8 | WRONG | Strong high-frequency/broadband character in the spectrum and a noisy, non-ERP-like time series; scalp map is not eye/ECG. |
| 15 | muscle | brain | 0.55 | WRONG | Scalp map resembles a plausible dipolar brain pattern and the time series is comparatively less blink/saccade-like; spectrum is not dominated by a single line or clear blink-band (<5 Hz) pattern. |
| 16 | brain | muscle | 0.73 | WRONG | Focal temporal/central activity with a jagged waveform and spectrum consistent with broadband high-frequency power. |
| 17 | muscle | eye | 0.66 | WRONG | Scalp topography is strongly anterior/peri-ocular and the time series shows step-like/deflection structure consistent with eye movements; spectrum emphasizes low-frequency content rather than clean alpha/brain rhythm. |
| 18 | muscle | muscle | 0.78 | OK | Time series is high-frequency/erratic with bursts; spectrum shows relatively elevated broadband high-frequency power rather than clean 1/f alpha; scalp map is not a clear dipole. |
| 19 | muscle | muscle | 0.74 | OK | No clear blink/QRS pattern; spectrum/time series look broadband and spiky with strong high-frequency content; scalp topography is not strongly dipolar. |
| 20 | brain | brain | 0.55 | OK | More dipolar/central-parietal-like scalp distribution and smoother rhythmic-ish activity; spectrum shows stronger low-to-mid frequencies with some alpha-range structure (not dominated by high-frequency muscle). |
| 21 | muscle | eye | 0.62 | WRONG | Scalp map is anterior/frontal with strong frontal weighting; time series shows slow deflection structure consistent with ocular artifacts; spectrum is dominated by low frequencies (reduced high-frequency dominance). |
| 22 | muscle | brain | 0.66 | WRONG | Scalp map resembles a plausible dipole/temporo-parietal distribution; time series is relatively smooth; spectrum shows typical decreasing trend with frequency and some mid-band structure (not broadband high-frequency). |
| 23 | muscle | eye | 0.6 | WRONG | Frontal/anterior topography with low-frequency-dominant spectrum and a step/slow-deflection-like time course suggest ocular contribution. |
| 24 | brain | muscle | 0.8 | WRONG | Very broadband, high-frequency-dominant spectrum and visibly noisy/bursty time series; scalp map is not clean brain dipole and looks more edge/jaw/temporal-like. |
| 25 | muscle | brain | 0.58 | WRONG | Topography is broadly distributed but not strongly ocular; time series is not blink-like and is comparatively smoother; spectrum shows more typical 1/f-like decay with some alpha/mid-band presence. |
| 26 | muscle | brain | 0.52 | WRONG | Scalp map has a localized posterior/temporal focus but not a single-electrode island; time series is not clearly blink/QRS; spectrum is not dominated by a sharp 50/60 Hz peak or strong broadband muscle dominance—closest match is neural/brain-like. |
| 27 | brain | muscle | 0.78 | WRONG | Time series is highly jagged/spiky and the spectrum shows relatively strong high-frequency content rather than a clean 1/f drop; scalp map is not clearly eye-frontal and looks more like a diffuse/edge-leaning artifact consistent with EMG. |
| 28 | muscle | eye | 0.74 | WRONG | Scalp topography is strongly frontal/periocular with a clear anterior focus; time series shows blink/saccade-like deflections and the spectrum is dominated by low frequencies (little clean high-frequency EMG profile). |
| 29 | muscle | brain | 0.62 | WRONG | Topography is more dipolar/brain-like (central/anterior-posterior pattern rather than isolated frontal eye focus); time series is comparatively smoother than muscle and the spectrum shows a more typical decreasing profile with some mid-band structure. |
| 30 | muscle | other_artifact | 0.55 | WRONG | Scalp map is not clearly dipolar brain and the time series/spectra look mixed (not a decisive QRS heart rhythm, not a sharp 50/60 Hz line peak, and not clearly eye- or muscle-dominant). |

## 12. Skew-normalized accuracy

- **gpt-5.4-nano**: raw 22.6% → balanced **12.4%** (unweighted mean of per-class recalls; every class counts equally regardless of frequency)

Balanced accuracy is the headline metric while the first-30 batch remains imbalanced (muscle-heavy). A stratified manifest would remove the need for normalization; both are supported by the skeleton.


## Provenance

- Manifest: `experiments/manifests/stage0_true_first30.csv` (sha256[:16] `8da094ff3a5ffd46`)
- Model registry: `experiments/models_registry.yaml` (sha256[:16] `a89f17b039b05c72`)
- Call audit logs: `logs/prompt__tightened-v1-vs-strip-default__nano__0137-first30_<model>.jsonl` (prompt sha, strip sha, raw responses, latency)
- Annotated renders: `annotated/<model>/IC*.png`; strips: `strips/`
