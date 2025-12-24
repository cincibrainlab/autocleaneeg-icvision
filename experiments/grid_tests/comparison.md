# Cross-Model Component Classification Comparison

## Components 0-3 (Present in all tests)

| IC | 4-comp 5.2 | 4-comp 5.2(high) | 6-comp 5.2 | 9-comp 5.2 | 9-comp 5.1 | 9-comp 5.2(low) | 12-comp 5.2 | 12-comp 5.2(low) |
|----|------------|------------------|------------|------------|------------|-----------------|-------------|------------------|
| 0 | eye (0.93) | eye (0.78) | ch_noise (0.78) | ch_noise (0.90) | eye (0.86) | ch_noise (0.70) | eye (0.90) | eye (0.62) |
| 1 | eye (0.90) | eye (0.90) | eye (0.88) | eye (0.95) | eye (0.86) | eye (0.95) | eye (0.90) | eye (0.90) |
| 2 | brain (0.78) | brain (0.87) | brain (0.60) | brain (0.75) | brain (0.74) | brain (0.85) | brain (0.70) | brain (0.78) |
| 3 | brain (0.74) | brain (0.82) | brain (0.67) | brain (0.70) | brain (0.78) | brain (0.85) | brain (0.75) | brain (0.73) |

### IC0 Classification Variability
- **eye**: 4-comp 5.2, 4-comp 5.2(high), 9-comp 5.1, 12-comp 5.2, 12-comp 5.2(low)
- **channel_noise**: 6-comp 5.2, 9-comp 5.2, 9-comp 5.2(low)

**Observation**: IC0 is the most inconsistent - classified as "eye" in smaller grids (4-comp) and larger grids (12-comp), but "channel_noise" in medium grids (6-comp, 9-comp). This may indicate IC0 is ambiguous or that context from surrounding components affects classification.

### IC1-3 Stability
- **IC1**: Consistently "eye" across ALL tests (100% agreement)
- **IC2**: Consistently "brain" across ALL tests (100% agreement)
- **IC3**: Consistently "brain" across ALL tests (100% agreement)

---

## Components 4-5 (Present in 6, 9, 12-comp tests)

| IC | 6-comp 5.2 | 9-comp 5.2 | 9-comp 5.1 | 9-comp 5.2(low) | 12-comp 5.2 | 12-comp 5.2(low) |
|----|------------|------------|------------|-----------------|-------------|------------------|
| 4 | brain (0.55) | brain (0.70) | brain (0.80) | brain (0.80) | muscle (0.65) | ch_noise (0.66) |
| 5 | brain (0.52) | brain (0.65) | brain (0.76) | brain (0.70) | brain (0.70) | brain (0.55) |

### IC4 Classification Variability
- **brain**: 6-comp 5.2, 9-comp (all models)
- **muscle**: 12-comp 5.2
- **channel_noise**: 12-comp 5.2(low)

**Observation**: IC4 shows instability in 12-component grids - different models classify it differently when more components are present.

### IC5 Stability
- Consistently "brain" across ALL tests (100% agreement)

---

## Components 6-8 (Present in 9 and 12-comp tests)

| IC | 9-comp 5.2 | 9-comp 5.1 | 9-comp 5.2(low) | 12-comp 5.2 | 12-comp 5.2(low) |
|----|------------|------------|-----------------|-------------|------------------|
| 6 | muscle (0.60) | muscle (0.80) | muscle (0.65) | brain (0.80) | brain (0.74) |
| 7 | brain (0.60) | brain (0.72) | brain (0.80) | brain (0.75) | eye (0.70) |
| 8 | ch_noise (0.85) | ch_noise (0.83) | ch_noise (0.75) | ch_noise (0.85) | ch_noise (0.63) |

### IC6 Classification Variability
- **muscle**: 9-comp (all models)
- **brain**: 12-comp (both models)

**Observation**: IC6 flips from "muscle" to "brain" when going from 9 to 12 components.

### IC7 Classification Variability
- **brain**: 9-comp (all), 12-comp 5.2
- **eye**: 12-comp 5.2(low)

### IC8 Stability
- Consistently "channel_noise" across ALL tests (100% agreement)

---

## Components 9-11 (Present only in 12-comp tests)

| IC | 12-comp 5.2 | 12-comp 5.2(low) | Agreement |
|----|-------------|------------------|-----------|
| 9 | muscle (0.80) | muscle (0.72) | YES |
| 10 | brain (0.70) | brain (0.69) | YES |
| 11 | eye (0.65) | eye (0.67) | YES |

All three new components show agreement between models.

---

## Summary Statistics

### Stability by Component (across all tests where present)

| Component | Tests | Agreement Rate | Most Common Label |
|-----------|-------|----------------|-------------------|
| IC0 | 8 | 62.5% | eye (5/8) |
| IC1 | 8 | 100% | eye |
| IC2 | 8 | 100% | brain |
| IC3 | 8 | 100% | brain |
| IC4 | 6 | 66.7% | brain (4/6) |
| IC5 | 6 | 100% | brain |
| IC6 | 5 | 60% | muscle (3/5) |
| IC7 | 5 | 80% | brain (4/5) |
| IC8 | 5 | 100% | channel_noise |
| IC9 | 2 | 100% | muscle |
| IC10 | 2 | 100% | brain |
| IC11 | 2 | 100% | eye |

### Overall Findings

1. **Highly Stable Components** (100% agreement):
   - IC1 (eye), IC2 (brain), IC3 (brain), IC5 (brain), IC8 (channel_noise)

2. **Unstable Components**:
   - IC0: Varies between eye/channel_noise depending on grid size
   - IC4: brain in smaller grids, muscle/ch_noise in 12-comp
   - IC6: muscle in 9-comp, brain in 12-comp

3. **Model Consistency**:
   - gpt-5.2 instant is most consistent across grid sizes
   - gpt-5.1 tends to classify ambiguous components as "eye" more often
   - gpt-5.2(low) reasoning sometimes differs from instant

4. **Grid Size Effect**:
   - Smaller grids (4-comp) may have higher per-component resolution
   - Larger grids (12-comp) show more classification variability
   - Sweet spot appears to be 6-9 components for balance of speed and accuracy

5. **Recommendations**:
   - Use 9-component grids for production (good balance)
   - Use gpt-5.2 instant (fastest, most reliable)
   - Flag components with <0.70 confidence for manual review
   - Consider ensemble voting across multiple grid configurations for ambiguous cases
