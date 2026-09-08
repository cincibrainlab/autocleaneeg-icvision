# Run report — `model__minimax-m3__tightened-v1__0137-first30`

**Variable tested:** model: minimax-m3 (OpenCode Go via CLI transport; same prompt, same 31 components)

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

- Models run: 1 (minimax-m3)
- API calls per model: one per strip
- Total classifications in this run: 31 components × 1 model(s)

## 6. Strip layout

- Components per strip: **9** (fixed by the strip protocol; final strip of a file may be smaller)
- Strips per recording: 4 for 31 components

## 7. Cost

| Model | API calls | Cost | Basis |
|-------|-----------|------|-------|
| `minimax-m3` | 4 | $0.0244 | **actual** (summed from CLI step_finish cost events; Go-subscription models are covered by quota, Zen models are out-of-pocket) |

## 8. Results breakdown


### minimax-m3

- Raw accuracy: **14/31 = 45.2%**
- Balanced (skew-normalized) accuracy: **45.7%**

| True class | Correct/Total | Recall |
|------------|---------------|--------|
| brain | 3/10 | 30% |
| eye | 2/2 | 100% |
| muscle | 9/17 | 53% |
| heart | 0/2 | 0% |

Predicted-label distribution: muscle×11, brain×9, eye×8, other_artifact×2, heart×1

Top confusions (truth → prediction):
- muscle → brain: 5
- brain → eye: 4
- brain → muscle: 2
- heart → eye: 1
- heart → brain: 1
- muscle → other_artifact: 1

Dominant failure mode: **muscle read as brain** (5 cases). Model language across these errors leans on topography/spectrum cues that fit the predicted class template; see per-component reasoning below and annotated renders for the visual evidence.

## 9. Most prevalent error modes


### minimax-m3

- Errors: **17/31**; high-confidence errors (confidence ≥0.80): **7**
- Dominant confusion pairs:
  - `muscle` → `brain`: 5
  - `brain` → `eye`: 4
  - `brain` → `muscle`: 2
  - `heart` → `eye`: 1
  - `heart` → `brain`: 1
- Largest prediction-count biases (predicted minus true):
  - `eye`: +6
  - `muscle`: -6
  - `other_artifact`: +2
  - `heart`: -1
  - `brain`: -1
- Weakest class recall: `heart` at 0/2 (0%)

## 10. Model justification per component


### minimax-m3

| IC | Truth | Predicted | Conf | Verdict | Model's stated reasoning |
|----|-------|-----------|------|---------|--------------------------|
| 0 | eye | eye | 0.78 | OK | Strong anterior (frontal) dipolar pattern with focal blue pole at top; time series shows large slow deflections consistent with vertical eye movement/blinks. |
| 1 | heart | eye | 0.97 | WRONG | Classic horizontal EOG signature: left-right frontal scalp gradient with step-like stable intervals and rapid saccade transitions in the time series. |
| 2 | eye | eye | 0.95 | OK | Strong focal red spot on frontal/vertex region with frontal topography; time series shows clear blink-like slow deflections. |
| 3 | brain | brain | 0.72 | OK | Posterior dipolar pattern (blue occipital, yellow anterior), smooth rhythmic time series, 1/f spectrum - consistent with posterior brain source. |
| 4 | brain | muscle | 0.85 | WRONG | Left temporal focal scalp map, spiky/erratic time series, and broad high-frequency power in spectrum rising above 20 Hz - clear muscle signature. |
| 5 | heart | brain | 0.72 | WRONG | Occipital-anterior dipolar pattern, relatively smooth time series, 1/f spectrum with apparent alpha-band emphasis - consistent with occipital brain activity. |
| 6 | muscle | muscle | 0.65 | OK | Very focal anterior scalp map but with visible surrounding spatial structure (not strict single-electrode isolation), highly spiky time series, and rising high-frequency spectrum - muscle activity with focal source. |
| 7 | brain | brain | 0.78 | OK | Central dipolar pattern with blue central and red peripheral sites, smooth rhythmic time series, clean 1/f spectrum - consistent with central/parietal brain activity. |
| 8 | brain | muscle | 0.7 | WRONG | Strong focal central vertex topography with opposite blue pole, spiky/bursty time series, and broad high-frequency spectral content - muscle with focal vertex/temporal source. |
| 9 | brain | eye | 0.92 | WRONG | Strong bilateral frontal dipolar map with characteristic blink topography; time series shows large slow deflections; spectrum dominated by low frequencies - classic blink component. |
| 10 | muscle | muscle | 0.78 | OK | Small left frontal/temporal focal map; spiky/bursty time series; spectrum shows broad sustained high-frequency power rather than clean 1/f - consistent with frontalis/temporal muscle. |
| 11 | muscle | brain | 0.7 | WRONG | Posterior/temporal dipolar map; time series shows rhythmic activity; spectrum falls with frequency in 1/f-like manner - plausible brain source, though posterior focality is somewhat unusual. |
| 12 | muscle | muscle | 0.7 | OK | Left frontal focal dipolar map with somewhat rhythmic time series; spectrum shows broad power; frontal focality without clean blink pattern suggests frontalis muscle over eye artifact. |
| 13 | muscle | muscle | 0.88 | OK | Left temporal/frontal focal dipolar; time series is spiky and erratic; spectrum shows broad high-frequency power with flat/rising high-freq profile - clear muscle signature (temporalis). |
| 14 | brain | brain | 0.95 | OK | Central sagittal midline dipolar map (top red, bottom blue) classic for central/parietal source; smooth rhythmic time series; spectrum shows clear peak ~8-13 Hz over 1/f - textbook alpha rhythm. |
| 15 | muscle | muscle | 0.78 | OK | Frontal focal dipolar; spiky time series with bursts; spectrum shows broad high-frequency content consistent with frontalis muscle rather than blink. |
| 16 | brain | eye | 0.72 | WRONG | Bilateral frontal dipolar map; time series shows slow rhythmic oscillations consistent with blink/eye activity; spectrum dominated by low frequencies - eye artifact most likely. |
| 17 | muscle | muscle | 0.75 | OK | Small left frontal focal map; spiky/bursty time series; spectrum shows broad sustained high-frequency power - consistent with frontalis muscle, not blink (no clear low-freq dominance). |
| 18 | muscle | other_artifact | 0.55 | WRONG | Focal scalp spot in upper right with spiky time series and roughly 1/f spectrum; not a clean isolated single electrode, not clearly dipolar brain, lacks clear blink/horizontal-EOG structure |
| 19 | muscle | brain | 0.95 | WRONG | Clear dipolar central/right topography, rhythmic time series with regular striping in ERP image, prominent alpha peak near 10 Hz with 1/f decay |
| 20 | brain | eye | 0.95 | WRONG | Bilateral frontal scalp map with opposing polarity, step-like low-frequency-dominated time series, strong low-frequency power with steep falloff - classic horizontal eye movement |
| 21 | muscle | muscle | 0.65 | OK | Focal non-dipolar spot at upper right edge with spiky/erratic time series; scalp topology and time-course pattern consistent with focal muscle rather than brain |
| 22 | muscle | brain | 0.85 | WRONG | Dipolar frontal-central topography, rhythmic alpha-band time series with regular segment striping, clear peak around 8-10 Hz |
| 23 | muscle | eye | 0.95 | WRONG | Bilateral frontal scalp map with large step-like blink deflections in time series, very strong low-frequency power below ~5 Hz - classic blink artifact |
| 24 | brain | other_artifact | 0.6 | WRONG | Unusual edge/ring topography with strong peripheral red and central blue; spectrum is steep 1/f-like (not the flat high-frequency profile typical of muscle), topology is not dipolar brain-like |
| 25 | muscle | brain | 0.7 | WRONG | Small focal central scalp map but clear alpha peak near 10 Hz with rhythmic time series and striping in ERP image - focal alpha source is plausible |
| 26 | muscle | muscle | 0.6 | OK | Focal spot at lower right with spiky time series and somewhat flat high-frequency spectrum component; lacks dipolar brain topology, consistent with focal muscle artifact |
| 27 | brain | eye | 0.6 | WRONG | Scalp map shows strong frontal-positive / central-negative polarity reversal across the whole head, characteristic of vertical EOG. Time series and 1/f spectrum have low-frequency-dominant structure consistent with eye movement/blink. Not classic blink deflections but frontal focality + low-freq dominance tips toward eye over generic brain. |
| 28 | muscle | heart | 0.6 | WRONG | ERP-style image shows very striking, regular horizontal bands across all segments at consistent latency — the signature of QRS complexes lining up across trials. Time series shows rhythmic deflections. Per the rules, a clear ~1 Hz QRS-like rhythm is decisive for heart on its own; the focal (rather than broad) scalp map does not override that. |
| 29 | muscle | brain | 0.55 | WRONG | ERP image has clear vertical (inter-trial consistent) structure and the time series/spectrum look like clean oscillatory neural activity (1/f decay). The scalp map is unusually focal at the vertex — could be a focal source or partial bad-channel contamination — but the activity panels are not noise-like, so brain is preferred over channel_noise/other_artifact. |
| 30 | muscle | muscle | 0.6 | OK | Spiky, erratic time series and focal temporal/lateral scalp map (blue/red cluster on the side, rest of map near neutral) are consistent with temporalis/jaw muscle artifact. Per the rules, when there is spiky/erratic activity with temporal focality, prefer muscle over channel_noise even if the map looks somewhat isolated. |

## 11. Skew-normalized accuracy

- **minimax-m3**: raw 45.2% → balanced **45.7%** (unweighted mean of per-class recalls; every class counts equally regardless of frequency)

Balanced accuracy is the headline metric while the first-30 batch remains imbalanced (muscle-heavy). A stratified manifest would remove the need for normalization; both are supported by the skeleton.


## Provenance

- Manifest: `experiments/manifests/stage0_true_first30.csv` (sha256[:16] `8da094ff3a5ffd46`)
- Model registry: `experiments/models_registry.yaml` (sha256[:16] `d1580df675d545e9`)
- Call audit logs: `logs/model__minimax-m3__tightened-v1__0137-first30_<model>.jsonl` (prompt sha, strip sha, raw responses, latency)
- Annotated renders: `annotated/<model>/IC*.png`; strips: `strips/`
