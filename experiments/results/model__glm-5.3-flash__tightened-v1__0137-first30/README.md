# Run report — `model__glm-5.3-flash__tightened-v1__0137-first30`

**Variable tested:** model: glm-5.3-flash (OpenCode Go via CLI transport; same prompt, same 31 components)

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

- Models run: 1 (glm-5.3-flash)
- API calls per model: one per strip
- Total classifications in this run: 31 components × 1 model(s)

## 6. Strip layout

- Components per strip: **9** (fixed by the strip protocol; final strip of a file may be smaller)
- Strips per recording: 4 for 31 components

## 7. Cost

| Model | API calls | Cost | Basis |
|-------|-----------|------|-------|
| `glm-5.3-flash` | 4 | $0.0059 | **actual** (summed from CLI step_finish cost events; Go-subscription models are covered by quota, Zen models are out-of-pocket) |

## 8. Time

| Model | Strips | Total time | Median/strip | Min | Max |
|-------|--------|------------|--------------|-----|-----|
| `glm-5.3-flash` | 4 | 404.8 s | 115.8 s | 81.9 s | 118.1 s |

## 9. Results breakdown


### glm-5.3-flash

- Raw accuracy: **17/31 = 54.8%**
- Balanced (skew-normalized) accuracy: **51.2%**

| True class | Correct/Total | Recall |
|------------|---------------|--------|
| brain | 4/10 | 40% |
| eye | 1/2 | 50% |
| muscle | 11/17 | 65% |
| heart | 1/2 | 50% |

Predicted-label distribution: muscle×14, brain×7, eye×5, heart×3, channel_noise×1, other_artifact×1

Top confusions (truth → prediction):
- brain → muscle: 3
- brain → heart: 2
- muscle → brain: 2
- muscle → eye: 2
- heart → eye: 1
- eye → brain: 1

Dominant failure mode: **brain read as muscle** (3 cases). Model language across these errors leans on topography/spectrum cues that fit the predicted class template; see per-component reasoning below and annotated renders for the visual evidence.

## 10. Most prevalent error modes


### glm-5.3-flash

- Errors: **14/31**; high-confidence errors (confidence ≥0.80): **2**
- Dominant confusion pairs:
  - `brain` → `muscle`: 3
  - `brain` → `heart`: 2
  - `muscle` → `brain`: 2
  - `muscle` → `eye`: 2
  - `heart` → `eye`: 1
- Largest prediction-count biases (predicted minus true):
  - `eye`: +3
  - `muscle`: -3
  - `brain`: -3
  - `other_artifact`: +1
  - `channel_noise`: +1
- Weakest class recall: `brain` at 4/10 (40%)

## 11. Model justification per component


### glm-5.3-flash

| IC | Truth | Predicted | Conf | Verdict | Model's stated reasoning |
|----|-------|-----------|------|---------|--------------------------|
| 0 | eye | eye | 0.95 | OK | Frontal midline topography with large slow blink-like deflections in time series and low-frequency-dominant spectrum. |
| 1 | heart | eye | 0.9 | WRONG | Left-right frontal polarity map with step-like saccade activity (stable intervals separated by fast transitions). |
| 2 | eye | brain | 0.55 | WRONG | Roughly dipolar map with opposite pole, 1/f-like decaying spectrum, and some ERP banding; spiky time series lowers confidence. |
| 3 | brain | heart | 0.6 | WRONG | Broad near-linear gradient scalp map (distant source) with repeated QRS-like sharp deflections about once per second. |
| 4 | brain | muscle | 0.7 | WRONG | Left temporal edge-focused map, spiky/erratic bursty activity, and broad sustained high-frequency spectral power. |
| 5 | heart | heart | 0.5 | OK | Broad anterior-posterior gradient map with recurring sharp spiky deflections consistent with a distant cardiac source; less regular than D. |
| 6 | muscle | channel_noise | 0.6 | WRONG | Tiny isolated single-electrode island with no opposite pole, spectrum rising toward high frequencies, and dense high-frequency striping in segment image. |
| 7 | brain | muscle | 0.55 | WRONG | Left temporal edge-focused map with dense spiky bursty activity and sustained high-frequency spectral shoulder; not step-like like eye. |
| 8 | brain | brain | 0.6 | OK | Dipolar map with occipital focus and broad spectral hump in alpha range; bursty time series consistent with occipital alpha. |
| 9 | brain | brain | 0.6 | OK | Broadly dipolar frontal-vs-posterior-temporal map with 1/f-decaying spectrum and low-frequency hump; time series irregular but no blink/saccade or high-frequency profile. |
| 10 | muscle | muscle | 0.6 | OK | Spectrum clearly rises above ~20 Hz to a high plateau (broad high-frequency power); small focal island at left temporal edge is not an unmistakable isolated-electrode map, so muscle wins. |
| 11 | muscle | brain | 0.5 | WRONG | Lateral right-temporal dipolar map with smoothly decaying 1/f spectrum; spiky series is a bit noisy but no flat/rising high-frequency profile to indicate muscle. |
| 12 | muscle | muscle | 0.65 | OK | Spectrum dips then rises with substantial power above ~30 Hz; focal edge map near left frontal/temporal rim and bursty spiky series fit temporalis/frontalis muscle. |
| 13 | muscle | brain | 0.5 | WRONG | Left-temporal edge dipole with smooth 1/f-decaying spectrum; dense spiky series raises some doubt, but no high-frequency spectral prominence for muscle. |
| 14 | brain | eye | 0.75 | WRONG | Frontal-central positive map with large slow biphasic deflections, low-frequency-dominant spectrum, and coarse blobby ERP structure consistent with blinks. |
| 15 | muscle | muscle | 0.55 | OK | Left-temporal edge-focused map with spiky activity and spectrum that flattens (fails to decay) above ~20 Hz, fitting shallow temporal muscle. |
| 16 | brain | brain | 0.55 | OK | Dipolar map (left central blue vs frontal yellow) with decaying spectrum and low-frequency hump; large deflections are slow but not clearly blink/step-like and map is not frontal-bilateral. |
| 17 | muscle | muscle | 0.6 | OK | Left-temporal edge dipole with bursty spiky series and spectrum that rises again above ~35 Hz, indicating broad high-frequency muscle power. |
| 18 | muscle | muscle | 0.7 | OK | Spiky bursty time series with spectrum that stays elevated/flat into high frequencies rather than 1/f decay; focal near-edge central map. |
| 19 | muscle | muscle | 0.6 | OK | Map is dipolar but spectrum is flat with sustained HF power and series is erratic/spiky, unlike clean 1/f brain decay. |
| 20 | brain | muscle | 0.55 | WRONG | Very focal left posterior/edge map with continuous erratic high-frequency-looking activity and blotchy segment image; spectrum decay is the one conflicting cue. |
| 21 | muscle | other_artifact | 0.5 | WRONG | Small edge-focal map, intermittent sharp spikes, low overall power, and banded ERP image; cues don't cleanly fit muscle, eye, or channel_noise. |
| 22 | muscle | muscle | 0.75 | OK | Spectrum clearly rises from ~10 Hz into the 20-40 Hz range with spiky bursty series; frontal focality is not low-frequency blink-like. |
| 23 | muscle | eye | 0.7 | WRONG | Frontal left-right dipolar map with power concentrated at low frequencies and large slow deflections, consistent with blink/saccade activity. |
| 24 | brain | heart | 0.75 | WRONG | Broad near-linear peripheral-to-central gradient map with regularly repeating QRS-like spiky deflections in the time series. |
| 25 | muscle | muscle | 0.6 | OK | Spectrum shows a rising HF plateau above ~15 Hz with spiky bursts despite a small focal central map. |
| 26 | muscle | muscle | 0.7 | OK | Dense spiky continuous activity with flat sustained spectrum into 40+ Hz and blotchy segment image; dipolar posterior map is atypical but spectrum/activity dominate. |
| 27 | brain | brain | 0.55 | OK | Clean central radial dipolar topography and 1/f-like decaying spectrum support brain; spiky time series and noisy ERP image lower confidence. |
| 28 | muscle | eye | 0.85 | WRONG | Frontal/periocular dipolar map (eye-to-vertex polarity), low-frequency-dominant spectrum, and a large blink-like transient in the time series. |
| 29 | muscle | muscle | 0.7 | OK | Spectrum rises steadily toward 55 Hz (high-frequency dominant), spiky erratic time series; frontal focal spot has smooth spread, not an isolated single-electrode island. |
| 30 | muscle | muscle | 0.75 | OK | Rising high-frequency spectral profile, spiky bursting time series, and shallow temporal/edge dipolar map consistent with temporal/jaw muscle. |

## 12. Skew-normalized accuracy

- **glm-5.3-flash**: raw 54.8% → balanced **51.2%** (unweighted mean of per-class recalls; every class counts equally regardless of frequency)

Balanced accuracy is the headline metric while the first-30 batch remains imbalanced (muscle-heavy). A stratified manifest would remove the need for normalization; both are supported by the skeleton.


## Provenance

- Manifest: `experiments/manifests/stage0_true_first30.csv` (sha256[:16] `8da094ff3a5ffd46`)
- Model registry: `experiments/models_registry.yaml` (sha256[:16] `a89f17b039b05c72`)
- Call audit logs: `logs/model__glm-5.3-flash__tightened-v1__0137-first30_<model>.jsonl` (prompt sha, strip sha, raw responses, latency)
- Annotated renders: `annotated/<model>/IC*.png`; strips: `strips/`
