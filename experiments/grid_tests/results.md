# Grid Classification Experiments

## Test Data
- **Raw file**: D0079_rest_postica_raw.fif
- **ICA file**: D0079_rest_postica-ica.fif (127 components)
- **API**: vision.autocleaneeg.org/v1

---

## Experiment 1: 4 Components (2x2 Grid)
**Image**: `grid_2x2_4comp.webp`
**Model**: gpt-5.2
**Time**: ~8 seconds

| Component | IC Index | Label | Confidence | Reason |
|-----------|----------|-------|------------|--------|
| A | IC0 | eye | 0.93 | Strong frontal/periocular topography and large slow deflections |
| B | IC1 | eye | 0.90 | Clear frontal maximum with slow drifting waveform |
| C | IC2 | brain | 0.78 | More dipolar/central pattern, 1/f spectrum with mid-frequency peaks |
| D | IC3 | brain | 0.74 | Dipolar centro-parietal distribution, 1/f spectrum |

---

## Experiment 2: 6 Components (2x3 Grid)
**Image**: `grid_2x3_6comp.webp`
**Model**: gpt-5.2
**Time**: ~11 seconds

| Component | IC Index | Label | Confidence | Reason |
|-----------|----------|-------|------------|--------|
| A | IC0 | channel_noise | 0.78 | Very focal hot spot near edge, no clear dipole |
| B | IC1 | eye | 0.88 | Strong frontal/periocular dominance, low-frequency power |
| C | IC2 | brain | 0.60 | Broad central distribution, 1/f spectrum with mid-frequency bumps |
| D | IC3 | brain | 0.67 | Posterior/central dipolar topography |
| E | IC4 | brain | 0.55 | Moderately dipolar lateral-central topography |
| F | IC5 | brain | 0.52 | Focal central-parietal hotspot, 1/f spectrum |

---

## Experiment 3: 9 Components (3x3 Grid) - Model Comparison
**Image**: `grid_3x3_9comp_instant.webp`

### GPT-5.2 (~18 seconds)

| Component | IC Index | Label | Confidence | Reason |
|-----------|----------|-------|------------|--------|
| A | IC0 | channel_noise | 0.90 | Very focal hotspot at extreme edge |
| B | IC1 | eye | 0.95 | Strong frontal/periocular dominance |
| C | IC2 | brain | 0.75 | Near-dipolar/central pattern, 1/f with mid-frequency peaks |
| D | IC3 | brain | 0.70 | Posterior/central dipolar topography |
| E | IC4 | brain | 0.70 | Compact dipolar source-like map |
| F | IC5 | brain | 0.65 | Midline/central focal source |
| G | IC6 | muscle | 0.60 | Edge/lateralized topography, elevated high-frequency |
| H | IC7 | brain | 0.60 | Broad dipolar field (left-right) |
| I | IC8 | channel_noise | 0.85 | Highly focal small patch near inferior edge |

### GPT-5.1 (~7 seconds)

| Component | IC Index | Label | Confidence | Reason |
|-----------|----------|-------|------------|--------|
| A | IC0 | eye | 0.86 | Strong frontal/periocular focus |
| B | IC1 | eye | 0.86 | Symmetric frontal dipole |
| C | IC2 | brain | 0.74 | Central midline dipolar field |
| D | IC3 | brain | 0.78 | Parietal/central dipolar pattern |
| E | IC4 | brain | 0.80 | Posterior-parietal dipole |
| F | IC5 | brain | 0.76 | Focal central-parietal dipolar topography |
| G | IC6 | muscle | 0.80 | Edge-focused lateral topography |
| H | IC7 | brain | 0.72 | Lateral central dipolar field |
| I | IC8 | channel_noise | 0.83 | Highly focal isolated spot |

### GPT-5.2(low) (~26 seconds)

| Component | IC Index | Label | Confidence | Reason |
|-----------|----------|-------|------------|--------|
| A | IC0 | channel_noise | 0.70 | Very focal hotspot on extreme frontal edge |
| B | IC1 | eye | 0.95 | Strong frontal/periocular field, low-frequency |
| C | IC2 | brain | 0.85 | Broad symmetric central pattern, 1/f spectrum |
| D | IC3 | brain | 0.85 | Clear dipolar anterior-posterior pattern |
| E | IC4 | brain | 0.80 | Left-right dipolar (temporal-lateralized) |
| F | IC5 | brain | 0.70 | Centrally distributed, 1/f with peaks |
| G | IC6 | muscle | 0.65 | Edge-dominant field, EMG-like spectrum |
| H | IC7 | brain | 0.80 | Clean lateral dipole, 1/f decay |
| I | IC8 | channel_noise | 0.75 | Very focal small hotspot |

### Model Agreement Summary (9 components)

| Component | GPT-5.2 | GPT-5.1 | GPT-5.2(low) | Consensus |
|-----------|---------|---------|--------------|-----------|
| A (IC0) | channel_noise | eye | channel_noise | **SPLIT** (2 ch_noise, 1 eye) |
| B (IC1) | eye | eye | eye | YES |
| C (IC2) | brain | brain | brain | YES |
| D (IC3) | brain | brain | brain | YES |
| E (IC4) | brain | brain | brain | YES |
| F (IC5) | brain | brain | brain | YES |
| G (IC6) | muscle | muscle | muscle | YES |
| H (IC7) | brain | brain | brain | YES |
| I (IC8) | channel_noise | channel_noise | channel_noise | YES |

**Agreement Rate**: 8/9 (89%) - All models agree except IC0

---

## Experiment 4: 4 Components - Thinking Model Comparison
**Image**: `grid_2x2_4comp_thinking.webp`

### GPT-5.2 (instant, ~8 seconds)

| Component | IC Index | Label | Confidence |
|-----------|----------|-------|------------|
| A | IC0 | eye | 0.93 |
| B | IC1 | eye | 0.90 |
| C | IC2 | brain | 0.78 |
| D | IC3 | brain | 0.74 |

### GPT-5.2(high) (thinking, ~23 seconds)

| Component | IC Index | Label | Confidence |
|-----------|----------|-------|------------|
| A | IC0 | eye | 0.78 |
| B | IC1 | eye | 0.90 |
| C | IC2 | brain | 0.87 |
| D | IC3 | brain | 0.82 |

### Comparison

| Component | Instant | Thinking | Agreement |
|-----------|---------|----------|-----------|
| A (IC0) | eye (0.93) | eye (0.78) | YES |
| B (IC1) | eye (0.90) | eye (0.90) | YES |
| C (IC2) | brain (0.78) | brain (0.87) | YES |
| D (IC3) | brain (0.74) | brain (0.82) | YES |

**Notes**:
- Thinking model has slightly higher confidence on brain components
- Instant model has higher confidence on eye components
- Both agree on all labels
- Thinking takes ~3x longer (23s vs 8s)

---

## Performance Summary

| Grid Size | Components | Model | Time (s) | API Calls |
|-----------|------------|-------|----------|-----------|
| 2x2 | 4 | gpt-5.2 | ~8 | 1 |
| 2x2 | 4 | gpt-5.2(high) | ~23 | 1 |
| 2x3 | 6 | gpt-5.2 | ~11 | 1 |
| 3x3 | 9 | gpt-5.2 | ~18 | 1 |
| 3x3 | 9 | gpt-5.1 | ~7 | 1 |
| 3x3 | 9 | gpt-5.2(low) | ~26 | 1 |
| 3x3 | 9 | gpt-5.2(high) | TIMEOUT | - |
| 3x4 | 12 | gpt-5.2 | ~21 | 1 |
| 3x4 | 12 | gpt-5.2(low) | ~90* | 1 |

*with retries due to Cloudflare timeout

**Comparison to single-component approach:**
- 128 components at 1 per request = 128 API calls
- 128 components at 9 per request = ~15 API calls (8.5x reduction)

---

## Experiment 5: Layout Comparison (9 Components, gpt-5.2)

Three layouts tested with identical components (IC0-8):

### Layout 1: Minimal (--compact) - No labels
**Image**: `layout_minimal_9comp.webp` (649KB)
**Time**: ~13 seconds

| Component | IC Index | Label | Confidence |
|-----------|----------|-------|------------|
| A | IC0 | eye | 0.90 |
| B | IC1 | eye | 0.90 |
| C | IC2 | brain | 0.80 |
| D | IC3 | brain | 0.75 |
| E | IC4 | brain | 0.70 |
| F | IC5 | brain | 0.65 |
| G | IC6 | brain | 0.60 |
| H | IC7 | brain | 0.70 |
| I | IC8 | other_artifact | 0.55 |

### Layout 2: Label Only (--layout label_only) - Corner labels
**Image**: `layout_label_only_9comp.webp` (651KB)
**Time**: ~15 seconds

| Component | IC Index | Label | Confidence |
|-----------|----------|-------|------------|
| A | IC0 | eye | 0.90 |
| B | IC1 | eye | 0.85 |
| C | IC2 | muscle | 0.75 |
| D | IC3 | brain | 0.80 |
| E | IC4 | brain | 0.70 |
| F | IC5 | brain | 0.65 |
| G | IC6 | other_artifact | 0.60 |
| H | IC7 | muscle | 0.70 |
| I | IC8 | channel_noise | 0.85 |

### Layout 3: Strip (--layout strip) - Horizontal rows
**Image**: `layout_strip_9comp.webp` (672KB)
**Time**: ~16 seconds

| Component | IC Index | Label | Confidence |
|-----------|----------|-------|------------|
| A | IC0 | channel_noise | 0.86 |
| B | IC1 | eye | 0.94 |
| C | IC2 | brain | 0.86 |
| D | IC3 | brain | 0.84 |
| E | IC4 | brain | 0.78 |
| F | IC5 | brain | 0.72 |
| G | IC6 | muscle | 0.70 |
| H | IC7 | muscle | 0.74 |
| I | IC8 | channel_noise | 0.92 |

### Layout Comparison Summary

| IC | Minimal | Label Only | Strip | Consensus |
|----|---------|------------|-------|-----------|
| IC0 | eye | eye | channel_noise | SPLIT (2 eye, 1 ch_noise) |
| IC1 | eye | eye | eye | YES |
| IC2 | brain | muscle | brain | SPLIT (2 brain, 1 muscle) |
| IC3 | brain | brain | brain | YES |
| IC4 | brain | brain | brain | YES |
| IC5 | brain | brain | brain | YES |
| IC6 | brain | other_artifact | muscle | NO CONSENSUS |
| IC7 | brain | muscle | muscle | SPLIT (2 muscle, 1 brain) |
| IC8 | other_artifact | channel_noise | channel_noise | SPLIT (2 ch_noise, 1 other) |

**Agreement Rate**: 4/9 (44%) unanimous

### Key Observations

1. **Strip Layout** shows higher confidence scores overall (avg 0.82 vs 0.72 for minimal)
2. **IC0** remains controversial - eye in grid layouts, channel_noise in strip
3. **IC6, IC7** show layout-dependent variation between brain/muscle/other
4. **IC8** consistently classified as artifact type (channel_noise or other)
5. Strip layout matches original 9-comp results better for artifact detection

### Recommendations

- **Strip layout** provides clearer visual separation and higher confidence
- **Label only** adds component identification without reducing readability
- **Minimal** may lose some visual context, reducing classification accuracy
- Consider using strip for ambiguous cases or when higher confidence is needed

---

## Experiment 6: Revised Prompt - Alpha Peak Emphasis

The original prompt caused misclassification of lateral brain components as muscle. Key revision:

**Old prompt:**
```
- "muscle": Positive spectral slope at high frequencies, edge-focused topography
```

**Revised prompt:**
```
- "brain": Dipolar pattern (can be central, parietal, OR lateral/temporal), 1/f spectrum with alpha (8-12Hz) or beta (13-30Hz) peaks. NOTE: Lateral/edge topography with alpha peak = brain, not muscle
- "muscle": Edge-focused topography AND flat/rising high-frequency spectrum (no alpha peak). Must have BOTH features
```

### Results with Revised Prompt (9 Components, gpt-5.2)

#### Layout 1: Minimal (--compact)
**Image**: `layout_minimal_9comp_v2.webp`

| Component | IC Index | Label | Confidence |
|-----------|----------|-------|------------|
| A | IC0 | other_artifact | 0.70 |
| B | IC1 | eye | 0.95 |
| C | IC2 | brain | 0.75 |
| D | IC3 | brain | 0.90 |
| E | IC4 | brain | 0.85 |
| F | IC5 | brain | 0.80 |
| G | IC6 | brain | 0.60 |
| H | IC7 | brain | 0.75 |
| I | IC8 | channel_noise | 0.90 |

#### Layout 2: Label Only (--layout label_only)
**Image**: `layout_label_only_9comp_v2.webp`

| Component | IC Index | Label | Confidence |
|-----------|----------|-------|------------|
| A | IC0 | eye | 0.88 |
| B | IC1 | eye | 0.90 |
| C | IC2 | brain | 0.65 |
| D | IC3 | brain | 0.78 |
| E | IC4 | brain | 0.85 |
| F | IC5 | brain | 0.80 |
| G | IC6 | brain | 0.67 |
| H | IC7 | brain | 0.86 |
| I | IC8 | channel_noise | 0.92 |

#### Layout 3: Strip (--layout strip)
**Image**: `layout_strip_9comp_v2.webp`

| Component | IC Index | Label | Confidence |
|-----------|----------|-------|------------|
| A | IC0 | channel_noise | 0.78 |
| B | IC1 | eye | 0.92 |
| C | IC2 | brain | 0.88 |
| D | IC3 | brain | 0.84 |
| E | IC4 | brain | 0.87 |
| F | IC5 | brain | 0.81 |
| G | IC6 | muscle | 0.74 |
| H | IC7 | brain | 0.80 |
| I | IC8 | channel_noise | 0.86 |

### Revised Prompt Comparison Summary

| IC | Minimal v2 | Label Only v2 | Strip v2 | Consensus |
|----|------------|---------------|----------|-----------|
| IC0 | other_artifact | eye | channel_noise | NO (artifact types vary) |
| IC1 | eye | eye | eye | **YES** |
| IC2 | brain | brain | brain | **YES** |
| IC3 | brain | brain | brain | **YES** |
| IC4 | brain | brain | brain | **YES** |
| IC5 | brain | brain | brain | **YES** |
| IC6 | brain | brain | muscle | SPLIT (2 brain, 1 muscle) |
| IC7 | brain | brain | brain | **YES** |
| IC8 | channel_noise | channel_noise | channel_noise | **YES** |

**Agreement Rate**: 7/9 (78%) unanimous - improved from 44%

### Key Improvements from Revised Prompt

1. **IC7 now correctly classified as brain** across all layouts (was muscle in strip v1)
2. **IC6** remains ambiguous - minimal/label_only say brain, strip says muscle (expert review confirms muscle is correct)
3. **IC0** still varies (eye vs channel_noise vs other_artifact) - this component is genuinely ambiguous
4. **Higher overall agreement** - 78% vs 44% with old prompt

### Expert Validation

After visual inspection of IC6 and IC7 spectra:
- **IC7**: Has clear alpha peak → correctly brain (revised prompt fixed this)
- **IC6**: Lacks clear alpha peak, has edge topography → correctly muscle (strip layout got this right)

### Final Recommendations

1. Use **revised prompt** with alpha peak emphasis for better lateral brain classification
2. **Strip layout** appears most accurate for artifact detection (IC6, IC8)
3. Components with <0.70 confidence should be flagged for expert review
4. IC0-type ambiguity (frontal focal) may require additional features to resolve
