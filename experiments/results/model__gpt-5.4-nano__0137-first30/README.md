# Run report — `model__gpt-5.4-nano__0137-first30`

**Variable tested:** model: gpt-5.4-nano (baseline measurement, no comparison run yet)

Generated 2026-09-08 09:19 by `run_report.py`. All paths relative to repo root. This directory is self-contained: everything needed to audit or re-analyze this run lives here.


## 1. Recording(s) used

| File | Components in manifest | Data sha256[:16] |
|------|------------------------|------------------|
| `SavedFiles/0137_VDAudio_ICA.set` | 31 | `82644e29268a7e5e` |

## 2. Component scope

- `0137_VDAudio_ICA`: 31 components — IC0-IC30

Sampling rule for prelim runs: **contiguous first-30** (IC0–IC30 per recording, the high-variance ICA components). Some recordings decompose into fewer ICs; the manifest records exactly which exist.


## 3. Prompt used

- Prompt: `strip_default (icvision built-in)`
- sha256: `d4c3f0e120964c9875bf7b80dc5b037cd41f698b3111dcc48b85268a0fd9e1e2`
- Source file: `prompts/strip_default.txt`

**Full prompt text:**

```text
Classify each of the {n} ICA components shown in this grid (labeled {labels}).

Each component shows:
- Topography map (scalp distribution)
- Time series (first 2.5 seconds)
- ERP-style image (continuous data segments)
- Power spectrum (1-55Hz)

Categories:
- "brain": Dipolar pattern (can be central, parietal, OR lateral/temporal), 1/f spectrum with alpha (8-12Hz) or beta (13-30Hz) peaks. NOTE: Lateral/edge topography with alpha peak = brain, not muscle
- "eye": Frontal/periocular focus with low-frequency dominated spectrum (<4Hz) AND large slow deflections in time series. Frontal focal + slow deflections = eye, even if topography looks focal
- "muscle": Edge-focused topography AND flat/rising high-frequency spectrum (no alpha peak). Must have BOTH features
- "heart": ~1Hz rhythmic deflections in time series, broad scalp distribution
- "line_noise": Sharp narrow peak at 50/60Hz
- "channel_noise": Single isolated focal spot (one sensor) with flat/noisy spectrum AND erratic/random time series. NOT eye if spectrum is low-frequency dominated with slow deflections
- "other_artifact": Doesn't fit above categories

Respond with JSON array (one object per component):
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

## 8. Results breakdown


### gpt-5.4-nano

- Raw accuracy: **6/31 = 19.4%**
- Balanced (skew-normalized) accuracy: **14.0%**

| True class | Correct/Total | Recall |
|------------|---------------|--------|
| brain | 5/10 | 50% |
| eye | 0/2 | 0% |
| muscle | 1/17 | 6% |
| heart | 0/2 | 0% |

Predicted-label distribution: brain×19, muscle×5, other_artifact×3, eye×2, channel_noise×2

Top confusions (truth → prediction):
- muscle → brain: 11
- brain → muscle: 3
- eye → brain: 2
- muscle → other_artifact: 2
- muscle → channel_noise: 2
- heart → muscle: 1

Dominant failure mode: **muscle read as brain** (11 cases). Model language across these errors leans on topography/spectrum cues that fit the predicted class template; see per-component reasoning below and annotated renders for the visual evidence.

## 9. Most prevalent error modes


### gpt-5.4-nano

- Errors: **25/31**; high-confidence errors (confidence ≥0.80): **2**
- Dominant confusion pairs:
  - `muscle` → `brain`: 11
  - `brain` → `muscle`: 3
  - `eye` → `brain`: 2
  - `muscle` → `other_artifact`: 2
  - `muscle` → `channel_noise`: 2
- Largest prediction-count biases (predicted minus true):
  - `muscle`: -12
  - `brain`: +9
  - `other_artifact`: +3
  - `channel_noise`: +2
  - `heart`: -2
- Weakest class recall: `eye` at 0/2 (0%)

## 10. Model justification per component


### gpt-5.4-nano

| IC | Truth | Predicted | Conf | Verdict | Model's stated reasoning |
|----|-------|-----------|------|---------|--------------------------|
| 0 | eye | brain | 0.62 | WRONG | Central/parietal dipolar-like topography with clear alpha/beta activity; spectrum shows 8–12 Hz/13–30 Hz peaks rather than flat high-frequency or strong <4 Hz dominance. |
| 1 | heart | muscle | 0.78 | WRONG | Edge/lateral topography with high-frequency dominated spectrum (no clear alpha peak) and time series with relatively fast, irregular fluctuations. |
| 2 | eye | brain | 0.66 | WRONG | Parietal/central dipolar pattern and 1/f spectrum with an alpha/beta peak; not strongly frontal/slow (<4 Hz) and not edge-only high-frequency. |
| 3 | brain | brain | 0.7 | OK | Midline/central topography with alpha/beta-like spectral structure and non-flat spectrum; time series looks like ongoing neural activity rather than rhythmic heart/slow eye. |
| 4 | brain | brain | 0.6 | OK | Lateral/temporal-ish dipolar topography with alpha/beta peak in the spectrum; not frontal slow-deflection eye and not edge-only muscle. |
| 5 | heart | brain | 0.58 | WRONG | Central topography with 1/f spectrum and visible alpha/beta components; time series not dominated by <4 Hz slow deflections and not edge-focused high-frequency. |
| 6 | muscle | other_artifact | 0.45 | WRONG | Topography is focal but spectrum/time series do not match clean eye (<4 Hz slow) or muscle (edge + high-frequency/no alpha) or heart (~1 Hz rhythmic). |
| 7 | brain | eye | 0.74 | WRONG | Strong frontal/periocular distribution with very slow, large deflections in the time series and low-frequency dominated spectrum (no alpha/beta peak dominance). |
| 8 | brain | other_artifact | 0.42 | WRONG | Focal frontal-like topography but spectrum/time series do not clearly show the required eye signature (<4 Hz dominance with slow deflections) nor muscle (edge + high-frequency/no alpha) or heart (~1 Hz rhythmic). |
| 9 | brain | muscle | 0.78 | WRONG | Edge/lateral scalp topography with no clear alpha peak; spectrum dominated by higher frequencies and time series looks jagged/fast. |
| 10 | muscle | channel_noise | 0.7 | WRONG | Topography shows a strong focal spot (single-sensor-like) with flat/noisy spectrum and irregular time series. |
| 11 | muscle | brain | 0.62 | WRONG | Dipolar/central-parietal pattern with 1/f spectrum plus a visible alpha/beta-band prominence; time series shows structured oscillatory activity. |
| 12 | muscle | brain | 0.6 | WRONG | Parietal/central dipolar topography and non-flat spectrum with alpha/beta-range activity rather than a narrow line or pure high-frequency noise. |
| 13 | muscle | brain | 0.66 | WRONG | Lateral/temporal dipolar pattern with clear alpha/beta-band structure in the spectrum and relatively smooth oscillatory time series. |
| 14 | brain | muscle | 0.8 | WRONG | Edge-focused topography and spectrum lacking alpha peak with strong high-frequency content; time series shows high-frequency bursts. |
| 15 | muscle | brain | 0.58 | WRONG | Broad dipolar/temporal-parietal topography with non-flat spectrum and some alpha/beta prominence; not consistent with pure muscle/eye. |
| 16 | brain | brain | 0.55 | OK | Central/parietal dipolar topography and 1/f-like spectrum with alpha/beta-range activity; time series shows rhythmic structure. |
| 17 | muscle | eye | 0.52 | WRONG | Frontal/periocular-like contribution with strong slow deflections in the time series and low-frequency dominance in the spectrum (<4 Hz). |
| 18 | muscle | brain | 0.78 | WRONG | Parietal/central dipolar topography with clear alpha/beta-like structure; spectrum shows 1/f with an alpha-range peak rather than flat high-frequency. |
| 19 | muscle | brain | 0.74 | WRONG | Central/parietal dipolar pattern; spectrum not dominated by <4 Hz and not flat high-frequency; time series looks neurogenic rather than slow ocular. |
| 20 | brain | brain | 0.7 | OK | Midline/central dipolar topography with alpha-range activity; ERP-like segments show structured fluctuations rather than eye-like slow drifts. |
| 21 | muscle | brain | 0.66 | WRONG | Broad central/parietal distribution with 1/f spectrum and no strong line-noise/flat muscle signature; time series lacks ~1 Hz heart rhythm and lacks strong slow frontal deflections. |
| 22 | muscle | brain | 0.62 | WRONG | Frontal-central topography but not periocular; spectrum shows typical brain 1/f with alpha/beta content; time series not dominated by large slow <4 Hz deflections. |
| 23 | muscle | muscle | 0.8 | OK | Edge/lateral topography with high-frequency-dominated spectrum (no clear alpha peak) and relatively noisy/fast time series consistent with muscle artifact. |
| 24 | brain | muscle | 0.83 | WRONG | Strong edge/lateral topography and spectrum dominated by higher frequencies without alpha peak; time series appears irregular/noisy rather than slow ocular or rhythmic heart. |
| 25 | muscle | brain | 0.71 | WRONG | Midline/central topography with 1/f spectrum and alpha-range peak; time series shows structured oscillatory activity rather than slow frontal eye deflections. |
| 26 | muscle | brain | 0.6 | WRONG | Right-lateral frontal/temporal dipolar pattern; spectrum shows typical brain-like 1/f with some alpha/beta structure; not a sharp 50/60 Hz peak, not flat high-frequency muscle, and not <4 Hz slow eye. |
| 27 | brain | brain | 0.62 | OK | Central/parietal dipolar-like topography with a clear 8–12 Hz alpha peak in the spectrum; time series shows oscillatory activity rather than slow frontal drift. |
| 28 | muscle | other_artifact | 0.45 | WRONG | Topography is more focal/edge-like than typical brain, but the spectrum/time course do not show a clear line-noise peak or classic muscle (high-frequency only) or eye (frontal slow <4 Hz) pattern. |
| 29 | muscle | brain | 0.7 | WRONG | Dipolar central topography with a 1/f-like spectrum and visible alpha/beta-range activity; not dominated by <4 Hz slow deflections or a sharp 50/60 Hz peak. |
| 30 | muscle | channel_noise | 0.6 | WRONG | Spectrum is relatively flat/noisy without a distinct alpha/beta peak and the time series looks irregular; topography is strongly focal (single-sensor-like) rather than a distributed dipole. |

## 11. Skew-normalized accuracy

- **gpt-5.4-nano**: raw 19.4% → balanced **14.0%** (unweighted mean of per-class recalls; every class counts equally regardless of frequency)

Balanced accuracy is the headline metric while the first-30 batch remains imbalanced (muscle-heavy). A stratified manifest would remove the need for normalization; both are supported by the skeleton.


## Provenance

- Manifest: `experiments/manifests/stage0_true_first30.csv` (sha256[:16] `8da094ff3a5ffd46`)
- Model registry: `experiments/models_registry.yaml` (sha256[:16] `d1580df675d545e9`)
- Call audit logs: `logs/model__gpt-5.4-nano__0137-first30_<model>.jsonl` (prompt sha, strip sha, raw responses, latency)
- Annotated renders: `annotated/<model>/IC*.png`; strips: `strips/`
