# Run report — `model__gemini-3.5-flash__tightened-v1__0137-first30-cli`

**Variable tested:** transport: OpenCode CLI Google route for gemini-3.5-flash (same prompt, same 31 components)

Generated 2026-09-06 13:24 by `run_report.py`. All paths relative to repo root. This directory is self-contained: everything needed to audit or re-analyze this run lives here.

## Transport

- OpenCode CLI: `1.18.29`
- Model selector: `opencode/gemini-3.5-flash`
- Route: OpenCode Zen Google provider through `opencode run --format json`
- The CLI used a no-tools `classifier` agent (`steps=1`) so the model could not invoke project commands or alter files.
- Raw CLI NDJSON, including the final model response, is stored in `logs/`.


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

> Raw accuracy on a skewed batch is dominated by the majority classes. See section 9 for the balanced metric.


## 5. Number of runs

- Models run: 1 (gemini-3.5-flash)
- API calls per model: one per strip
- Total classifications in this run: 31 components × 1 model(s)

## 6. Strip layout

- Components per strip: **9** (fixed by the strip protocol; final strip of a file may be smaller)
- Strips per recording: 4 for 31 components

## 7. Results breakdown


### gemini-3.5-flash

- Raw accuracy: **10/31 = 32.3%**
- Balanced (skew-normalized) accuracy: **42.9%**

| True class | Correct/Total | Recall |
|------------|---------------|--------|
| brain | 6/10 | 60% |
| eye | 1/2 | 50% |
| muscle | 2/17 | 12% |
| heart | 1/2 | 50% |

Predicted-label distribution: channel_noise×14, brain×8, muscle×3, other_artifact×3, eye×2, heart×1

Top confusions (truth → prediction):
- muscle → channel_noise: 14
- brain → other_artifact: 3
- eye → brain: 1
- brain → muscle: 1
- heart → brain: 1
- muscle → eye: 1

Dominant failure mode: **muscle read as channel_noise** (14 cases). Model language across these errors leans on topography/spectrum cues that fit the predicted class template; see per-component reasoning below and annotated renders for the visual evidence.

## 8. Model justification per component


### gemini-3.5-flash

| IC | Truth | Predicted | Conf | Verdict | Model's stated reasoning |
|----|-------|-----------|------|---------|--------------------------|
| 0 | eye | eye | 0.95 | OK | Frontal scalp distribution with classic blink-like slow deflection in the time series and low-frequency dominated power spectrum. |
| 1 | heart | heart | 0.95 | OK | Extremely regular, periodic QRS-like cardiac deflections in the time series (~1 Hz rhythm) and broad lateralized scalp map consistent with a distant source. |
| 2 | eye | brain | 0.85 | WRONG | Frontal-central dipole scalp map with a smooth, rhythmic time series and a clear alpha band peak (around 10 Hz) in the power spectrum. |
| 3 | brain | brain | 0.9 | OK | Symmetrical central-parietal dipolar scalp map, rhythmic time series, and 1/f spectral decay with a clear alpha peak. |
| 4 | brain | muscle | 0.85 | WRONG | Left temporal scalp focus near the muscle insertion, spiky/erratic time series, and a broad spectrum with significant high-frequency power. |
| 5 | heart | brain | 0.85 | WRONG | Clear sagittal dipolar scalp map with 1/f spectral decay and a prominent alpha peak. |
| 6 | muscle | channel_noise | 0.95 | WRONG | Scalp map is unambiguously dominated by a single isolated-electrode 'island' with no smooth spatial falloff to neighboring channels. |
| 7 | brain | other_artifact | 0.8 | WRONG | Non-dipolar, multipolar scalp map (central blue with bilateral temporal red regions) coupled with noisy, spiky time series activity and mixed spectral features. |
| 8 | brain | brain | 0.98 | OK | Classic occipital-parietal dipolar scalp map with a highly prominent alpha peak in the power spectrum and smooth, rhythmic time series. |
| 9 | brain | brain | 0.95 | OK | Clear dipolar central-parietal topography, smooth rhythmic time series, and 1/f spectrum with a distinct peak in the alpha band (~10 Hz). |
| 10 | muscle | channel_noise | 0.9 | WRONG | Topography shows a highly isolated, sharp single-electrode 'island' on the left side with no smooth spatial falloff, accompanied by a flat high-frequency spectrum. |
| 11 | muscle | channel_noise | 0.9 | WRONG | Topography displays a single isolated electrode 'island' on the right posterior/temporal edge, with a noisy time series and flat high-frequency spectrum. |
| 12 | muscle | channel_noise | 0.9 | WRONG | Topography displays a sharp, single-electrode focus in the left frontal region with no spatial falloff, and a flat power spectrum. |
| 13 | muscle | channel_noise | 0.9 | WRONG | Topography is dominated by an isolated single electrode on the left posterior edge, exhibiting a flat high-frequency spectrum. |
| 14 | brain | brain | 0.95 | OK | Topography shows a smooth, broad central-parietal distribution, with a classic 1/f power spectrum and rhythmic time-series activity. |
| 15 | muscle | channel_noise | 0.9 | WRONG | Topography shows a sharp, isolated single-electrode 'island' on the left posterior temporal edge, with a noisy time-series and non-1/f flat spectrum. |
| 16 | brain | brain | 0.98 | OK | Beautifully clear, smooth central-parietal dipole topography, rhythmic time-series, and a classic 1/f spectrum with a prominent alpha-band peak. |
| 17 | muscle | channel_noise | 0.9 | WRONG | Topography is dominated by a sharp, single-electrode focus on the left frontocentral region, with a flat power spectrum and noisy time-series. |
| 18 | muscle | channel_noise | 0.95 | WRONG | Scalp map is dominated by a single isolated electrode island on the right frontal area with no smooth spatial falloff to neighboring sites. |
| 19 | muscle | channel_noise | 0.95 | WRONG | Scalp map is dominated by a single isolated electrode on the right fronto-temporal region with no spatial falloff or opposite pole. |
| 20 | brain | brain | 0.98 | OK | Perfect dipolar scalp map (central-parietal), clear 1/f power spectrum with a typical alpha peak (~10 Hz), and smooth rhythmic time series. |
| 21 | muscle | channel_noise | 0.95 | WRONG | Scalp map shows a single isolated blue electrode island on the right frontal-temporal region with no spatial falloff. |
| 22 | muscle | channel_noise | 0.95 | WRONG | Scalp map is dominated by a single isolated electrode on the front-left area with no spatial falloff. |
| 23 | muscle | channel_noise | 0.95 | WRONG | Scalp map is dominated by a single isolated electrode at the top/frontal region with no spatial falloff. |
| 24 | brain | other_artifact | 0.9 | WRONG | Scalp map is splotchy and non-dipolar (blue center, red peripheral ring). Power spectrum shows 1/f decay, which is inconsistent with muscle, fitting the other_artifact fallback category perfectly. |
| 25 | muscle | channel_noise | 0.92 | WRONG | Scalp map shows a single isolated electrode island near the top-middle-left with no spatial falloff. |
| 26 | muscle | channel_noise | 0.95 | WRONG | Scalp map is dominated by a single isolated electrode on the right temporal region with no smooth spatial falloff. |
| 27 | brain | other_artifact | 0.8 | WRONG | The scalp map is splotchy, non-dipolar, and edge-focused (red at posterior and lateral edges, blue in center), and the time series is noisy without a clear classic artifact or brain rhythm, making other_artifact the most appropriate category. |
| 28 | muscle | eye | 0.95 | WRONG | The scalp map shows a strong frontal/periocular concentration, the time series exhibits a large, slow deflection characteristic of an eye blink/movement, and the power spectrum is heavily concentrated below 5 Hz. |
| 29 | muscle | muscle | 0.9 | OK | The scalp map is focused on the fronto-temporal edge, the time series is highly spiky and erratic, and the power spectrum shows broad, elevated high-frequency activity above 20 Hz. |
| 30 | muscle | muscle | 0.9 | OK | The scalp map shows a highly localized lateral dipole, the time series is spiky and erratic, and the power spectrum has a flat or rising high-frequency profile above 20 Hz. |

## 9. Skew-normalized accuracy

- **gemini-3.5-flash**: raw 32.3% → balanced **42.9%** (unweighted mean of per-class recalls; every class counts equally regardless of frequency)

Balanced accuracy is the headline metric while the first-30 batch remains imbalanced (muscle-heavy). A stratified manifest would remove the need for normalization; both are supported by the skeleton.


## Provenance

- Manifest: `experiments/manifests/stage0_true_first30.csv` (sha256[:16] `8da094ff3a5ffd46`)
- Model registry: `experiments/models_registry.yaml` (sha256[:16] `463a1800a9e69076`)
- Call audit logs: `logs/model__gemini-3.5-flash__tightened-v1__0137-first30-cli_<model>.jsonl` (prompt sha, strip sha, raw responses, latency)
- Annotated renders: `annotated/<model>/IC*.png`; strips: `strips/`
