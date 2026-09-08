# Run report — `model__muse-spark-1.3__tightened-v1__0137-first30`

**Variable tested:** model: muse-spark-1.3 (OpenCode Go via CLI transport; same prompt, same 31 components)

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

- Models run: 1 (muse-spark-1.3)
- API calls per model: one per strip
- Total classifications in this run: 31 components × 1 model(s)

## 6. Strip layout

- Components per strip: **9** (fixed by the strip protocol; final strip of a file may be smaller)
- Strips per recording: 4 for 31 components

## 7. Cost

| Model | API calls | Cost | Basis |
|-------|-----------|------|-------|
| `muse-spark-1.3` | 4 | $0.0061 | **actual** (summed from CLI step_finish cost events; Go-subscription models are covered by quota, Zen models are out-of-pocket) |

## 8. Time

| Model | Strips | Total time | Median/strip | Min | Max |
|-------|--------|------------|--------------|-----|-----|
| `muse-spark-1.3` | 4 | 357.9 s | 103.6 s | 53.1 s | 147.4 s |

## 9. Results breakdown


### muse-spark-1.3

- Raw accuracy: **23/31 = 74.2%**
- Balanced (skew-normalized) accuracy: **75.1%**

| True class | Correct/Total | Recall |
|------------|---------------|--------|
| brain | 8/10 | 80% |
| eye | 2/2 | 100% |
| muscle | 12/17 | 71% |
| heart | 1/2 | 50% |

Predicted-label distribution: muscle×13, brain×8, eye×5, other_artifact×2, channel_noise×2, heart×1

Top confusions (truth → prediction):
- muscle → eye: 3
- muscle → channel_noise: 2
- brain → muscle: 1
- heart → other_artifact: 1
- brain → other_artifact: 1

Dominant failure mode: **muscle read as eye** (3 cases). Model language across these errors leans on topography/spectrum cues that fit the predicted class template; see per-component reasoning below and annotated renders for the visual evidence.

## 10. Most prevalent error modes


### muse-spark-1.3

- Errors: **8/31**; high-confidence errors (confidence ≥0.80): **4**
- Dominant confusion pairs:
  - `muscle` → `eye`: 3
  - `muscle` → `channel_noise`: 2
  - `brain` → `muscle`: 1
  - `heart` → `other_artifact`: 1
  - `brain` → `other_artifact`: 1
- Largest prediction-count biases (predicted minus true):
  - `muscle`: -4
  - `eye`: +3
  - `other_artifact`: +2
  - `channel_noise`: +2
  - `brain`: -2
- Weakest class recall: `heart` at 1/2 (50%)

## 11. Model justification per component


### muse-spark-1.3

| IC | Truth | Predicted | Conf | Verdict | Model's stated reasoning |
|----|-------|-----------|------|---------|--------------------------|
| 0 | eye | eye | 0.92 | OK | Frontal bilateral topography with large slow biphasic blink-like deflections and spectrum concentrated below ~5Hz with 1/f decay. |
| 1 | heart | heart | 0.9 | OK | Regular sharp narrow downward QRS-like spikes about 4 in 2.5s on flat baseline with broad left-right gradient consistent with distant cardiac source. |
| 2 | eye | eye | 0.68 | OK | Frontal/periocular involvement with large slow irregular deflections and low-frequency dominant 1/f spectrum, blink/saccade-like. |
| 3 | brain | brain | 0.65 | OK | Roughly dipolar central topography, smoother rhythmic continuous time series, 1/f-like spectrum with mid-frequency shoulder, no clear artifact cue. |
| 4 | brain | muscle | 0.78 | WRONG | Temporal/jaw edge-focused map, spiky erratic bursty time series, broad elevated high-frequency power above 20Hz without clean 1/f decay. |
| 5 | heart | other_artifact | 0.55 | WRONG | Mixed cues: broad frontal-occipital gradient, irregular sharp spikes plus high-frequency burst, plateau spectrum neither low-frequency eye nor clean brain nor broad muscle. |
| 6 | muscle | channel_noise | 0.97 | WRONG | Unmistakable single isolated right-frontal electrode island with no spatial falloff, decisive for bad channel. |
| 7 | brain | brain | 0.62 | OK | Central dipolar-like topography, continuous rhythmic activity, 1/f falling spectrum, no blink, QRS, high-frequency muscle or line-peak evidence. |
| 8 | brain | brain | 0.68 | OK | Posterior occipital focal with opposite frontal pole, continuous activity, 1/f-like spectrum with 5-30Hz shoulder, plausible visual brain source. |
| 9 | brain | brain | 0.8 | OK | Broad central dipolar map, smoother rhythmic time series, blotchy low-frequency ERP image, 1/f spectrum decaying from low frequencies without high-frequency rise. |
| 10 | muscle | muscle | 0.82 | OK | Left-frontal edge focal map, spiky transient time series, U-shaped spectrum with trough ~10Hz then flat high high-frequency plateau. |
| 11 | muscle | muscle | 0.88 | OK | Right-temporal edge bipolar focal map, continuous spiky high-frequency time series, flat high-frequency spectrum. |
| 12 | muscle | muscle | 0.82 | OK | Left-frontal focal map, bursty time series, spectrum dips then stays high with broad >20Hz power. |
| 13 | muscle | muscle | 0.88 | OK | Left-temporal edge focal map, continuous noisy bursting time series, flat high high-frequency spectrum. |
| 14 | brain | brain | 0.85 | OK | Central-anterior vs posterior dipolar-like map, smoother rhythmic time series, blotchy ERP image, clean 1/f decaying spectrum. |
| 15 | muscle | muscle | 0.86 | OK | Left-posterior edge focal bipolar map, spiky noisy time series, flat high-frequency spectrum. |
| 16 | brain | brain | 0.85 | OK | Central posterior vs anterior dipolar pattern, smoother time series, blotchy low-frequency ERP image, low-frequency dominant decaying spectrum. |
| 17 | muscle | muscle | 0.84 | OK | Left-lateral edge bipolar focal map, spiky high-frequency time series, U-shaped spectrum with high flat high-frequency plateau. |
| 18 | muscle | muscle | 0.85 | OK | Edge-focused right-temporal focal map with U-shaped spectrum showing flat/rising broad high-frequency power above ~20Hz and spiky early time series. |
| 19 | muscle | muscle | 0.85 | OK | Right-temporal edge focal map with continuous noisy time series and U-shaped spectrum rising to high high-frequency power, typical muscle. |
| 20 | brain | brain | 0.6 | OK | Posterior roughly dipolar red-blue map with 1/f spectrum showing 5-15Hz hump near alpha and decaying high frequencies, no muscle HF rise. |
| 21 | muscle | eye | 0.6 | WRONG | Right-frontal focal map with transient large spikes then quiet and spectrum dominated below ~5Hz with low high frequencies, saccade/blink-like. |
| 22 | muscle | muscle | 0.8 | OK | Left-frontal edge focal map with noisy bursty time series and spectrum dipping near 10Hz then jagged rise to high high-frequency power. |
| 23 | muscle | eye | 0.9 | WRONG | Frontal/periocular red-blue map with isolated large slow spike in time series and sharp low-frequency-dominant spectrum, classic blink. |
| 24 | brain | other_artifact | 0.7 | WRONG | Highly multipolar/splotchy peripheral-red central-blue map that is non-dipolar and implausible for brain despite 1/f-like decaying spectrum. |
| 25 | muscle | muscle | 0.65 | OK | Frontal small focal map with early burst then quiet and U-shaped spectrum rising to high high-frequency plateau, frontalis muscle. |
| 26 | muscle | muscle | 0.8 | OK | Right-temporal edge red-blue focal map with continuous high-frequency noisy time series and sustained flat high-frequency spectrum. |
| 27 | brain | brain | 0.7 | OK | Central blue with surrounding positivity, roughly dipolar not eye/edge focused; 1/f falling spectrum; continuous noisy but non-blink time series. |
| 28 | muscle | eye | 0.9 | WRONG | Frontal/periocular map with strong anterior focus; large slow blink-like deflection in time series; spectrum dominated below ~5Hz with dip then flat. |
| 29 | muscle | channel_noise | 0.85 | WRONG | Scalp map shows sharp single-electrode red island left-frontal with no smooth falloff on flat green background; unambiguous isolated channel. |
| 30 | muscle | muscle | 0.75 | OK | Edge-focused left temporal splotchy map; spiky erratic time series; spectrum dips low then rises to 55Hz with broad high-frequency power. |

## 12. Skew-normalized accuracy

- **muse-spark-1.3**: raw 74.2% → balanced **75.1%** (unweighted mean of per-class recalls; every class counts equally regardless of frequency)

Balanced accuracy is the headline metric while the first-30 batch remains imbalanced (muscle-heavy). A stratified manifest would remove the need for normalization; both are supported by the skeleton.


## Provenance

- Manifest: `experiments/manifests/stage0_true_first30.csv` (sha256[:16] `8da094ff3a5ffd46`)
- Model registry: `experiments/models_registry.yaml` (sha256[:16] `a89f17b039b05c72`)
- Call audit logs: `logs/model__muse-spark-1.3__tightened-v1__0137-first30_<model>.jsonl` (prompt sha, strip sha, raw responses, latency)
- Annotated renders: `annotated/<model>/IC*.png`; strips: `strips/`
