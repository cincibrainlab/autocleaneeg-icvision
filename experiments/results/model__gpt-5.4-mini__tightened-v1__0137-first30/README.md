# Run report — `model__gpt-5.4-mini__tightened-v1__0137-first30`

**Variable tested:** model: gpt-5.4-mini vs gpt-5.4-nano (same tightened_v1 prompt, same 31 components)

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

- Models run: 1 (gpt-5.4-mini)
- API calls per model: one per strip
- Total classifications in this run: 31 components × 1 model(s)

## 6. Strip layout

- Components per strip: **9** (fixed by the strip protocol; final strip of a file may be smaller)
- Strips per recording: 4 for 31 components

## 7. Cost

| Model | API calls | Cost | Basis |
|-------|-----------|------|-------|
| `gpt-5.4-mini` | 4 | ~$0.0288 | **estimated** (Zen pricing $0.75/$1M in, $4.5/$1M out × ~6000in/600out tokens per strip; images dominate input; actual cost not metered by this API path) |

## 8. Time

| Model | Strips | Total time | Median/strip | Min | Max |
|-------|--------|------------|--------------|-----|-----|
| `gpt-5.4-mini` | 4 | 16.9 s | 4.0 s | 2.5 s | 6.5 s |

## 9. Results breakdown


### gpt-5.4-mini

- Raw accuracy: **9/31 = 29.0%**
- Balanced (skew-normalized) accuracy: **51.5%**

| True class | Correct/Total | Recall |
|------------|---------------|--------|
| brain | 5/10 | 50% |
| eye | 2/2 | 100% |
| muscle | 1/17 | 6% |
| heart | 1/2 | 50% |

Predicted-label distribution: brain×12, other_artifact×10, eye×3, muscle×3, channel_noise×2, heart×1

Top confusions (truth → prediction):
- muscle → brain: 7
- muscle → other_artifact: 6
- brain → other_artifact: 3
- brain → muscle: 2
- muscle → channel_noise: 2
- heart → other_artifact: 1

Dominant failure mode: **muscle read as brain** (7 cases). Model language across these errors leans on topography/spectrum cues that fit the predicted class template; see per-component reasoning below and annotated renders for the visual evidence.

## 10. Most prevalent error modes


### gpt-5.4-mini

- Errors: **22/31**; high-confidence errors (confidence ≥0.80): **6**
- Dominant confusion pairs:
  - `muscle` → `brain`: 7
  - `muscle` → `other_artifact`: 6
  - `brain` → `other_artifact`: 3
  - `brain` → `muscle`: 2
  - `muscle` → `channel_noise`: 2
- Largest prediction-count biases (predicted minus true):
  - `muscle`: -14
  - `other_artifact`: +10
  - `brain`: +2
  - `channel_noise`: +2
  - `heart`: -1
- Weakest class recall: `muscle` at 1/17 (6%)

## 11. Model justification per component


### gpt-5.4-mini

| IC | Truth | Predicted | Conf | Verdict | Model's stated reasoning |
|----|-------|-----------|------|---------|--------------------------|
| 0 | eye | eye | 0.98 | OK | Strong frontal/periocular topography with large slow blink-like deflections and low-frequency-dominant spectrum. |
| 1 | heart | heart | 0.96 | OK | Regular repeating sharp deflections about once per second in the time series, consistent with ECG/QRS artifact. |
| 2 | eye | eye | 0.93 | OK | Frontal/asymmetric topography and slow, large-amplitude eye-movement-like activity with mostly low-frequency power. |
| 3 | brain | other_artifact | 0.72 | WRONG | No clear dipolar brain pattern; time series is noisy/irregular and spectrum lacks a clean brain-like or specific artifact signature. |
| 4 | brain | muscle | 0.88 | WRONG | Irregular spiky activity with relatively elevated high-frequency content and a non-brain, edge-weighted scalp pattern. |
| 5 | heart | other_artifact | 0.66 | WRONG | Broad non-dipolar topography with noisy time series and no decisive blink, heart, line-noise, or isolated-channel signature. |
| 6 | muscle | channel_noise | 0.97 | WRONG | Scalp map is dominated by a single isolated electrode island near the frontal area, which is the key criterion for channel_noise. |
| 7 | brain | other_artifact | 0.74 | WRONG | Diffuse, non-dipolar topography and noisy broadband activity without a clear specific artifact pattern. |
| 8 | brain | brain | 0.84 | OK | Reasonably dipolar scalp map with posterior/central focus and a smoother 1/f-like spectrum, more consistent with neural activity than artifact. |
| 9 | brain | brain | 0.86 | OK | Roughly dipolar scalp map and a 1/f-like spectrum with no strong blink or muscle signature; time series looks irregular but not clearly artifact-like. |
| 10 | muscle | other_artifact | 0.72 | WRONG | Very focal left-sided scalp hotspot with an odd spectrum that rises toward higher frequencies, but not a clean isolated bad channel and not a classic muscle or eye pattern. |
| 11 | muscle | other_artifact | 0.78 | WRONG | Strong edge-focused, non-brain-like topography with a localized posterior-lateral pattern; spectrum and time series do not cleanly match eye, heart, or brain. |
| 12 | muscle | other_artifact | 0.66 | WRONG | Frontal-lateral focal map and noisy time course, but no decisive blink rhythm, QRS pattern, or isolated bad electrode; mixed artifact appearance. |
| 13 | muscle | other_artifact | 0.74 | WRONG | Left temporal focal map with opposite polarity nearby, but the spectrum is broad and not clearly brain-like or a classic muscle/eye signature. |
| 14 | brain | brain | 0.9 | OK | Broad dipolar-looking topography and a clear falling 1/f spectrum; time series is irregular but compatible with neural activity rather than a specific artifact. |
| 15 | muscle | other_artifact | 0.7 | WRONG | Left temporal focality with non-dipolar structure and a fairly flat/noisy spectrum; does not meet the threshold for channel_noise or a cleaner artifact class. |
| 16 | brain | brain | 0.84 | OK | Reasonably dipolar scalp map and a spectrum that decays with frequency, without strong low-frequency blink structure or high-frequency muscle dominance. |
| 17 | muscle | other_artifact | 0.68 | WRONG | Frontal-lateral asymmetric map with noisy activity and no decisive blink, heart, line-noise, or isolated-electrode evidence; best treated as mixed artifact. |
| 18 | muscle | brain | 0.72 | WRONG | Focal but plausible dipolar scalp map with non-artifactual spectrum; time series is not blink-like or high-frequency dominated. |
| 19 | muscle | brain | 0.68 | WRONG | Clear dipolar left-right pattern and broadly 1/f-like spectrum without strong low-frequency blink or high-frequency muscle signature. |
| 20 | brain | muscle | 0.91 | WRONG | Irregular spiky activity with broad high-frequency power and a shallow/flat spectrum at higher frequencies; scalp map is edge/temporal-like. |
| 21 | muscle | brain | 0.63 | WRONG | Somewhat dipolar map and spectrum that still falls with frequency; no decisive blink, heart, or line-noise cue. |
| 22 | muscle | brain | 0.74 | WRONG | Small focal dipolar source near frontal region but spectrum is not muscle-like and time series is relatively non-blink-like. |
| 23 | muscle | brain | 0.66 | WRONG | Anterior focal dipole with a mostly decaying spectrum; no clear QRS/blink/50-60 Hz pattern, so brain is the best fit. |
| 24 | brain | other_artifact | 0.86 | WRONG | Broad non-dipolar, highly diffuse topography with mixed polarity and no clean brain-like source pattern; spectrum is not decisive for a specific artifact class. |
| 25 | muscle | brain | 0.71 | WRONG | Localized frontal/central dipolar map with no strong artifact spectral signature; time series is noisy but not clearly muscle or eye-like. |
| 26 | muscle | brain | 0.69 | WRONG | Focal lateral dipolar map and a broadly decaying spectrum without a sharp line-noise peak or obvious blink/QRS rhythm. |
| 27 | brain | brain | 0.86 | OK | Scalp map is broad and roughly dipolar/central rather than frontal or isolated; spectrum shows a typical 1/f falloff without a sharp line-noise peak; time series is irregular but not blink-like. |
| 28 | muscle | eye | 0.97 | WRONG | Strong frontal/periocular topography with a large slow deflection in the time series and power concentrated at very low frequencies, consistent with an eye blink component. |
| 29 | muscle | channel_noise | 0.91 | WRONG | Topography is dominated by a very small, isolated single-electrode-like island near the frontal edge with little smooth spatial spread, which is the key cue for bad channel noise. |
| 30 | muscle | muscle | 0.88 | OK | Left temporal/edge-focused map plus spiky irregular activity and a spectrum that is relatively elevated at higher frequencies compared with a clean brain-like 1/f shape, fitting muscle artifact. |

## 12. Skew-normalized accuracy

- **gpt-5.4-mini**: raw 29.0% → balanced **51.5%** (unweighted mean of per-class recalls; every class counts equally regardless of frequency)

Balanced accuracy is the headline metric while the first-30 batch remains imbalanced (muscle-heavy). A stratified manifest would remove the need for normalization; both are supported by the skeleton.


## Provenance

- Manifest: `experiments/manifests/stage0_true_first30.csv` (sha256[:16] `8da094ff3a5ffd46`)
- Model registry: `experiments/models_registry.yaml` (sha256[:16] `a89f17b039b05c72`)
- Call audit logs: `logs/model__gpt-5.4-mini__tightened-v1__0137-first30_<model>.jsonl` (prompt sha, strip sha, raw responses, latency)
- Annotated renders: `annotated/<model>/IC*.png`; strips: `strips/`
