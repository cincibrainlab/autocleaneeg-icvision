# Run report — `model__grok-4.6__tightened-v1__0137-first30`

**Variable tested:** model: grok-4.6 (OpenCode Go via CLI transport; same prompt, same 31 components)

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

- Models run: 1 (grok-4.6)
- API calls per model: one per strip
- Total classifications in this run: 31 components × 1 model(s)

## 6. Strip layout

- Components per strip: **9** (fixed by the strip protocol; final strip of a file may be smaller)
- Strips per recording: 4 for 31 components

## 7. Cost

| Model | API calls | Cost | Basis |
|-------|-----------|------|-------|
| `grok-4.6` | 4 | $0.1616 | **actual** (summed from CLI step_finish cost events; Go-subscription models are covered by quota, Zen models are out-of-pocket) |

## 8. Time

| Model | Strips | Total time | Median/strip | Min | Max |
|-------|--------|------------|--------------|-----|-----|
| `grok-4.6` | 4 | 378.2 s | 99.7 s | 58.8 s | 121.1 s |

## 9. Results breakdown


### grok-4.6

- Raw accuracy: **26/31 = 83.9%**
- Balanced (skew-normalized) accuracy: **68.5%**

| True class | Correct/Total | Recall |
|------------|---------------|--------|
| brain | 8/10 | 80% |
| eye | 2/2 | 100% |
| muscle | 16/17 | 94% |
| heart | 0/2 | 0% |

Predicted-label distribution: muscle×16, brain×10, other_artifact×3, eye×2

Top confusions (truth → prediction):
- heart → brain: 2
- brain → other_artifact: 2
- muscle → other_artifact: 1

Dominant failure mode: **heart read as brain** (2 cases). Model language across these errors leans on topography/spectrum cues that fit the predicted class template; see per-component reasoning below and annotated renders for the visual evidence.

## 10. Most prevalent error modes


### grok-4.6

- Errors: **5/31**; high-confidence errors (confidence ≥0.80): **1**
- Dominant confusion pairs:
  - `heart` → `brain`: 2
  - `brain` → `other_artifact`: 2
  - `muscle` → `other_artifact`: 1
- Largest prediction-count biases (predicted minus true):
  - `other_artifact`: +3
  - `heart`: -2
  - `muscle`: -1
- Weakest class recall: `heart` at 0/2 (0%)

## 11. Model justification per component


### grok-4.6

| IC | Truth | Predicted | Conf | Verdict | Model's stated reasoning |
|----|-------|-----------|------|---------|--------------------------|
| 0 | eye | eye | 0.88 | OK | Large slow blink-like deflections, steep 1/f low-frequency spectrum, frontal-weighted map |
| 1 | heart | brain | 0.62 | WRONG | Dipolar left-right topography with mid-band power then decay; not clearly saccadic or HF-muscle |
| 2 | eye | eye | 0.78 | OK | Right-frontal/periocular focus, slow deflections, power concentrated at low frequencies |
| 3 | brain | brain | 0.72 | OK | Smooth anterior-posterior dipole, 1/f spectrum, no blink/QRS/HF-muscle pattern |
| 4 | brain | brain | 0.68 | OK | Reasonably dipolar lateral map with falling spectrum; not isolated-channel or rising-HF muscle |
| 5 | heart | brain | 0.7 | WRONG | Posterior-anterior dipolar map and 1/f-like decay after a mid-band plateau |
| 6 | muscle | muscle | 0.92 | OK | Spectrum rises with frequency; focal temporal/frontal edge map and bursty high-frequency activity |
| 7 | brain | brain | 0.7 | OK | Central dipolar/sombrero map with clean 1/f decay and no QRS or line peak |
| 8 | brain | brain | 0.75 | OK | Posterior/occipital dipolar focus with 1/f spectrum, not an isolated electrode or muscle HF profile |
| 9 | brain | brain | 0.88 | OK | Dipolar central/parietal map, 1/f-falling spectrum, relatively smooth rhythmic time series; not eye/edge/muscle-like. |
| 10 | muscle | muscle | 0.82 | OK | Focal near-edge hotspot without a true single-electrode island; flat/broadband high-frequency spectrum and spiky activity point to EMG, not channel_noise. |
| 11 | muscle | muscle | 0.9 | OK | Right temporal/edge focus, dense high-frequency time series/ERP streaks, and a flat elevated high-frequency spectrum. |
| 12 | muscle | muscle | 0.72 | OK | Left frontal-temporal edge map; spectrum lacks clean 1/f and is not <5 Hz blink-like; activity is noisy rather than saccade/blink, so temporal EMG over eye/brain. |
| 13 | muscle | muscle | 0.9 | OK | Left posterior-lateral edge topography with bursty high-frequency activity and a flat/rising high-frequency spectrum. |
| 14 | brain | brain | 0.8 | OK | Broad dipolar anterior-posterior scalp map with clear 1/f spectral decay and smoother low-frequency time series. |
| 15 | muscle | muscle | 0.86 | OK | Left inferior/temporal edge focus, spiky high-frequency traces, and a flat high-frequency spectrum typical of EMG. |
| 16 | brain | brain | 0.86 | OK | Dipolar centro-parietal map, 1/f-like falling spectrum, and relatively smooth non-artifact time series. |
| 17 | muscle | muscle | 0.74 | OK | Left temporal/edge dipole-like map but spectrum recovers at high frequencies and traces are irregular/high-frequency, favoring temporal muscle over brain. |
| 18 | muscle | muscle | 0.72 | OK | Focal right temporal hotspot; spectrum dips then stays elevated at high frequencies; noisy/spiky time series. |
| 19 | muscle | muscle | 0.85 | OK | Right temporal focus with spatial spread; rising high-frequency spectrum; erratic high-frequency activity. |
| 20 | brain | brain | 0.78 | OK | Posterior/occipital dipolar-ish map; 1/f spectrum with mid-frequency power then decay; relatively rhythmic time series. |
| 21 | muscle | other_artifact | 0.58 | WRONG | Compact right-temporal focus but mixed panels: not a clean dipole, not a classic muscle HF rise, odd horizontal ERP banding; not an unambiguous single-electrode island. |
| 22 | muscle | muscle | 0.88 | OK | Left frontal/frontalis focus; classic muscle spectrum (dip then rise/plateau above ~20 Hz); high-frequency ERP streaking. |
| 23 | muscle | muscle | 0.68 | OK | Left frontotemporal edge-focused map; bursty time series; residual high-frequency power rather than clean brain 1/f. |
| 24 | brain | other_artifact | 0.82 | WRONG | Non-dipolar center-surround/rim topography; 1/f spectrum is not enough for brain; no QRS rhythm for heart. |
| 25 | muscle | muscle | 0.8 | OK | Focal left map; spectrum dips then rises and stays high; high-frequency ERP texture; not an isolated-electrode island. |
| 26 | muscle | muscle | 0.75 | OK | Right posterior-temporal focal map; broadband/flat high-frequency spectrum rather than 1/f; noisy activity. |
| 27 | brain | other_artifact | 0.58 | WRONG | Rim-heavy posterior/lateral map is not a clean dipole; 1/f spectrum but no alpha, erratic TS, no eye/QRS/line/single-channel cue. |
| 28 | muscle | muscle | 0.88 | OK | Spectrum drops then rises toward 55 Hz; edge-frontal map; bursty TS and streaky segments. |
| 29 | muscle | muscle | 0.86 | OK | Rising high-frequency spectrum; focal left frontal/temporal map with spiky erratic activity, not an isolated electrode. |
| 30 | muscle | muscle | 0.82 | OK | Left temporal focus with rising HF power and spiky TS; spectrum outweighs a plausible temporal dipole. |

## 12. Skew-normalized accuracy

- **grok-4.6**: raw 83.9% → balanced **68.5%** (unweighted mean of per-class recalls; every class counts equally regardless of frequency)

Balanced accuracy is the headline metric while the first-30 batch remains imbalanced (muscle-heavy). A stratified manifest would remove the need for normalization; both are supported by the skeleton.


## Provenance

- Manifest: `experiments/manifests/stage0_true_first30.csv` (sha256[:16] `8da094ff3a5ffd46`)
- Model registry: `experiments/models_registry.yaml` (sha256[:16] `a89f17b039b05c72`)
- Call audit logs: `logs/model__grok-4.6__tightened-v1__0137-first30_<model>.jsonl` (prompt sha, strip sha, raw responses, latency)
- Annotated renders: `annotated/<model>/IC*.png`; strips: `strips/`
