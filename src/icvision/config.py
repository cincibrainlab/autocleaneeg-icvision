"""
Configuration constants and settings for ICVision.

This module contains the OpenAI prompt, label mappings, and other constants
used throughout the ICVision package.
"""

# Default OpenAI model for vision classification
DEFAULT_MODEL = "gpt-4.1"

# Component label definitions in priority order
COMPONENT_LABELS = [
    "brain",
    "eye",
    "muscle",
    "heart",
    "line_noise",
    "channel_noise",
    "other_artifact",
]

# Mapping from ICVision labels to MNE-compatible labels
ICVISION_TO_MNE_LABEL_MAP = {
    "brain": "brain",
    "eye": "eog",
    "muscle": "muscle",
    "heart": "ecg",
    "line_noise": "line_noise",
    "channel_noise": "ch_noise",
    "other_artifact": "other",
}

# Default labels to exclude (all except brain)
DEFAULT_EXCLUDE_LABELS = [
    "eye",
    "muscle",
    "heart",
    "line_noise",
    "channel_noise",
    "other_artifact",
]

# OpenAI prompt for ICA component classification
OPENAI_ICA_PROMPT_PROSE = """Analyze this EEG ICA component image and classify into ONE category:

**UNDERSTANDING THE PLOTS:**
- **Topography** (top left): Spatial distribution of component activity across the scalp
- **Time Series** (top right): Component activation over time - look for rhythmic patterns, spikes, or characteristic temporal signatures
- **Power Spectrum** (bottom right): Frequency content - shows how power varies across frequencies
- **Continuous Data Segments** (bottom left): Trial-by-trial visualization showing component activity across multiple time segments. This ERP-image style plot reveals:
  * **Temporal consistency**: Consistent patterns across trials indicate reliable sources (brain/artifacts)
  * **Artifact signatures**: Sporadic, random patterns often indicate noise; regular artifact patterns (eye, heart, muscle) show characteristic timing
  * **Neural oscillations**: Brain components often show consistent oscillatory patterns across trials
  * **Artifact detection**: Muscle artifacts show sporadic high-frequency bursts; eye artifacts show movement-related patterns; heart artifacts show regular ~1Hz rhythmic activity

**COMPONENT CLASSIFICATION:**

- "brain": Dipolar pattern in CENTRAL, PARIETAL, or TEMPORAL regions (NOT PERIOCULAR FOCUSED or EDGE-FOCUSED). 1/f-like spectrum with possible peaks at 6-35Hz. Rhythmic, wave-like time series WITHOUT abrupt level shifts. MUST show decreasing power with increasing frequency (1/f pattern).

- "eye":
  * **PRIMARY PATTERN**: DIPOLAR pattern (two opposite poles) ALWAYS tightly focused ABOVE/BESIDE the PERIOCULAR regions (Fp1, Fp2, F7, F8)
  * **TOPOGRAPHY**: Look for bilateral activity in the ANTERIOR-MOST electrodes (Fp1, Fp2, F7, F8) - can be left-right OR up-down oriented. Key is having CLEAR DIPOLAR STRUCTURE with both positive and negative poles CONFINED TO EYE-ADJACENT ELECTRODES.
  * **SPECTRUM**: Usually dominated by low frequencies (1-5Hz) with 1/f-like decrease
  * **TIME SERIES**: Step-like, slow waves, or blink-related patterns
  * **CRITICAL DISTINCTION**: Eye has DIPOLAR pattern ONLY ANTERIOR to the PERIOCULAR electrodes (Fp1, Fp2, F7, F8) AND ABSOLUTELY NO dipolar pattern extension to mid-frontal (F3, F4, Fz), CENTRAL, PARIETAL, TEMPORAL, or OCCIPITAL regions, while channel_noise has only ONE focal spot without opposite pole and brain can have dipolar pattern extension to the mid-frontal (F3, F4, Fz), CENTRAL, PARIETAL, TEMPORAL, or OCCIPITAL regions
  * **ANATOMICAL BOUNDARY**: Must be strongest at electrodes DIRECTLY ABOVE/BESIDE the PERIOCULAR regions (Fp1, Fp2, F7, F8) with no dipolar pattern extension to mid-frontal (F3, F4, Fz), CENTRAL, PARIETAL, TEMPORAL, or OCCIPITAL regions
  * **KEY RULE**: If you see dipolar pattern ONLY ABOVE/BESIDE the eyes (Fp1, Fp2, F7, F8) with 1/f spectrum → "eye". If dipolar pattern includes ANY electrodes over CENTRAL, PARIETAL, TEMPORAL, or OCCIPITAL regions → NOT eye.
  * **PRIORITY**: Periocular dipolar patterns = eye, frontal cortical patterns that extend to mid-frontal (F3, F4, Fz) or central regions = brain, single spots = muscle/noise

- "muscle": (SPECTRAL SIGNATURE IS THE MOST DOMINANT INDICATOR)
  * DECISIVE SPECTRAL FEATURE (Primary and Often Conclusive Muscle Indicator): The power spectrum exhibits a CLEAR and SUSTAINED POSITIVE SLOPE, meaning power consistently INCREASES with increasing frequency, starting from the available frequency range upwards (typically from around 20-30Hz if present, or from the lowest available frequency if data is high-pass filtered above 30Hz). This often looks like the spectrum is 'curving upwards' or 'scooping upwards' at higher frequencies. IF THIS DISTINCT SPECTRAL SIGNATURE IS OBSERVED, THE COMPONENT IS TO BE CLASSIFIED AS 'muscle', EVEN IF other features might seem ambiguous or resemble other categories. This spectral cue is the strongest determinant for muscle.
  * IMPORTANT: Muscle artifacts often appear as SINGLE FOCAL SPOTS that look like channel noise. The distinction is BIOLOGICAL PLAUSIBILITY - focal spots near muscle areas (temporal, frontal, jaw) are usually muscle, not channel noise.
  * **CONTINUOUS DATA SEGMENTS**: Muscle artifacts often show sporadic, high-frequency bursts or irregular patterns across trials, unlike the consistent patterns of brain activity
  * OTHER SUPPORTING MUSCLE CHARACTERISTICS (Use if spectral cue is present, or with caution if spectral cue is less definitive but clearly NOT 1/f):
    *   Topography: Common patterns include (a) very localized 'bowtie' or 'shallow dipole' patterns (two small, adjacent areas of opposite polarity, often taking up <25% of the scalp map, can appear anywhere but frequently temporal/posterior) OR (b) more diffuse activity, typically along the EDGE of the scalp (temporal, occipital, neck regions) OR (c) single focal spots near muscle locations.
    *   Time Series: Often shows spiky, high-frequency, and somewhat erratic activity.

- "heart": (CRITICAL - EXAMINE TIME SERIES AND CONTINUOUS DATA SEGMENTS!)
  * **PRIMARY INDICATOR**: TIME SERIES must show ANY KIND of rhythmic deflections (upward spikes, downward dips, or step changes) that repeat approximately every 1 second (0.8-1.5 seconds apart). These can be subtle - look for ANY regular pattern that occurs roughly once per second.
  * **CONTINUOUS DATA SEGMENTS**: Look for regular vertical bands or stripes occurring ~once per second across trials - this shows consistent cardiac activity across time segments
  * **DECISIVE RULE**: If you see ANY rhythmic pattern repeating every ~1000ms in the time series OR regular temporal patterns in continuous data segments, classify as "heart" IMMEDIATELY regardless of all other features.
  * **COMMON PATTERNS**: Heart artifacts often appear as regular downward deflections, upward spikes, or step-like changes occurring ~once per second
  * **TOPOGRAPHY**: IGNORE topography for heart detection - can be dipolar, broad, focal, or any pattern
  * **SPECTRUM**: IGNORE spectrum for heart detection - often looks brain-like with 1/f pattern
  * **CRITICAL**: The ONLY diagnostic feature for heart is rhythmic ~1Hz activity in time series AND/OR consistent temporal patterns in continuous data segments

- "line_noise":
  * MUST show SHARP PEAK at 50/60Hz in spectrum - NOT a notch/dip (notches are filters, not line noise).
  * NOTE: Almost all components show a notch at 60Hz from filtering - this is NOT line noise!
  * Line noise requires a POSITIVE PEAK at 50/60Hz, not a negative dip.
  * IMPORTANT: If data is high-pass filtered above 50/60Hz, line noise detection is not applicable.

- "channel_noise":
  * **PRIMARY CRITERION**: SINGLE ELECTRODE "hot/cold spot" - tiny, isolated circular area WITHOUT an opposite pole (not dipolar).
  * **DECISIVE SPATIAL RULE**: If topography shows single tiny focal spot with NO clear opposite pole, this is likely channel_noise regardless of spectrum appearance (bad electrodes can have noisy/artifactual spectra).
  * **LOCATION GUIDE**: More common in central/interior regions, but can occur anywhere due to electrode problems.
  * **KEY DISTINCTION FROM MUSCLE**: Muscle can also appear as focal spots but usually has some spatial extent or is part of broader patterns. True channel noise is extremely localized to essentially one electrode.
  * **CONTINUOUS DATA SEGMENTS**: True channel noise shows extremely localized, often random or flat patterns across trials, different from the structured patterns of other artifacts
  * **Compare with eye**: Channel noise has only ONE focal point, while eye has TWO opposite poles (dipole). Eye dipoles are also typically larger and more structured.
  * **Compare with muscle**: If uncertain between channel_noise and muscle for focal spots, prioritize spatial characteristics over spectral ones
  * TRUE channel noise is rare but when present, spatial localization is the most reliable indicator.

- "other_artifact": Components that don't clearly fit other categories, INCLUDING:
  * Complex multi-polar topographies (multiple hotspots, irregular patterns)
  * Dipolar patterns that are noisy, irregular, or poorly formed
  * Components with mixed or contradictory features
  * Any pattern that looks "brain-like" but is complex, noisy, or atypical
  * When in doubt between brain and other patterns → choose "other_artifact"

CLASSIFICATION PRIORITY (IMPORTANT: Evaluate in this order. Later rules apply only if earlier conditions are not met or are ambiguous):
1.  FIRST: EXAMINE TIME SERIES CAREFULLY for ANY rhythmic deflections (up/down spikes, dips, steps) repeating every ~1 second → "heart" (OVERRIDES all other features)
2.  ELSE IF SINGLE TINY FOCAL SPOT without clear opposite pole (non-dipolar) → "channel_noise" (spatial characteristics are decisive for bad electrodes, regardless of spectrum)
3.  ELSE IF power spectrum dominated by low frequencies (1-5Hz) and clear dipolar pattern STRICTLY ONLY ANTERIOR to the PERIOCULAR electrodes (Fp1, Fp2, F7, F8) AND ABSOLUTELY NO dipolar pattern extension to mid-frontal (F3, F4, Fz), CENTRAL, PARIETAL, TEMPORAL, or OCCIPITAL regions → "eye"
4.  ELSE IF Spectrum shows SHARP PEAK (not notch) at 50/60Hz → "line_noise"
5.  ELSE IF Power spectrum exhibits a CLEAR and SUSTAINED increase in power starting from the available frequency range upwards (typically from ~20-30 Hz if present, or from the lowest available frequency if data is high-pass filtered above 30Hz). The slope must be consistently upward, not flat or decreasing, and the trend should be visually obvious. Disregard any dips or notches at 50 or 60 Hz, as these result from filtering and do not reflect the true slope. → "muscle". (THIS IS A DECISIVE RULE FOR MUSCLE. If this spectral pattern is present, classify as 'muscle' even if the topography isn't a perfect 'bowtie' or edge artifact, and before considering 'brain').
6.  ELSE IF Single focal spot near muscle areas (temporal, frontal, jaw) → "muscle" (many muscle artifacts look like single spots)
7.  ELSE IF (Topography is a clear 'bowtie'/'shallow dipole' OR distinct EDGE activity) AND (Time series is spiky/high-frequency OR spectrum is generally high-frequency without being clearly 1/f and also not clearly a positive slope) → "muscle" (Secondary muscle check, for cases where the positive slope is less perfect but other muscle signs are strong and it's definitely not brain).
8.  ELSE IF Dipolar pattern in mid-frontal (F3, F4, Fz), CENTRAL, PARIETAL, or TEMPORAL regions (AND NOT already definitively classified as 'muscle' by its spectral signature under rule 5) AND spectrum shows a clear general 1/f pattern (overall DECREASING power with increasing frequency, AND ABSOLUTELY NO sustained positive slope at high frequencies) → "brain"
9.  ELSE → "other_artifact"

IMPORTANT: A 60Hz NOTCH (negative dip) in spectrum is normal filtering, seen in most components, and should NOT be used for classification! Do not include this in your reasoning.

RETURN FORMAT: You must respond with ONLY a valid JSON object in the following exact format:
{
    "label": "one_of_the_valid_labels",
    "confidence": 0.95,
    "reason": "detailed_reasoning_for_the_classification"
}

The "label" must be exactly one of: brain, eye, muscle, heart, line_noise, channel_noise, other_artifact
The "confidence" must be a number between 0.0 and 1.0
The "reason" should provide detailed reasoning for your classification decision.

Example JSON response:
{"label": "eye", "confidence": 0.95, "reason": "Strong frontal topography with left-right dipolar pattern (horizontal eye movement) or frontal positivity with spike-like patterns (vertical eye movement/blinks). Low-frequency dominated spectrum and characteristic time series confirm eye activity."}
"""

OPENAI_ICA_PROMPT_JSON = """
{
  "task": "Analyze an EEG ICA component image and classify into ONE category",
  "plot_descriptions": {
    "topography": {
      "location": "top left",
      "description": "Spatial distribution of component activity across the scalp"
    },
    "time_series": {
      "location": "top right",
      "description": "Component activation over time - look for rhythmic patterns, spikes, or characteristic temporal signatures"
    },
    "power_spectrum": {
      "location": "bottom right",
      "description": "Frequency content - shows how power varies across frequencies"
    },
    "continuous_data_segments": {
      "location": "bottom left",
      "description": "Trial-by-trial visualization showing component activity across multiple time segments. This ERP-image style plot reveals: Temporal consistency (Consistent patterns across trials indicate reliable sources - brain/artifacts), Artifact signatures (Sporadic, random patterns often indicate noise; regular artifact patterns - eye, heart, muscle - show characteristic timing), Neural oscillations (Brain components often show consistent oscillatory patterns across trials), Artifact detection (Muscle artifacts show sporadic high-frequency bursts; eye artifacts show movement-related patterns; heart artifacts show regular ~1Hz rhythmic activity)"
    }
  },
  "classification_categories": {
    "brain": {
      "topography": "Dipolar pattern in CENTRAL, PARIETAL, or TEMPORAL regions (NOT PERIOCULAR FOCUSED or EDGE-FOCUSED)",
      "spectrum": "1/f-like spectrum with possible peaks at 6-35Hz",
      "time_series": "Rhythmic, wave-like time series WITHOUT abrupt level shifts",
      "additional": "MUST show decreasing power with increasing frequency (1/f pattern)"
    },
    "eye": {
      "primary_pattern": "DIPOLAR pattern (two opposite poles) ALWAYS tightly focused ABOVE/BESIDE the PERIOCULAR regions (Fp1, Fp2, F7, F8)",
      "topography": "Bilateral activity in the ANTERIOR-MOST electrodes (Fp1, Fp2, F7, F8) - can be left-right OR up-down oriented. Key is having CLEAR DIPOLAR STRUCTURE with both positive and negative poles CONFINED TO EYE-ADJACENT ELECTRODES",
      "spectrum": "Usually dominated by low frequencies (1-5Hz) with 1/f-like decrease",
      "time_series": "Step-like, slow waves, or blink-related patterns",
      "critical_distinction": "Eye has DIPOLAR pattern ONLY ANTERIOR to the PERIOCULAR electrodes (Fp1, Fp2, F7, F8) AND ABSOLUTELY NO dipolar pattern extension to mid-frontal (F3, F4, Fz), CENTRAL, PARIETAL, TEMPORAL, or OCCIPITAL regions, while channel_noise has only ONE focalteller focal spot without opposite pole and brain can have dipolar pattern extension to the mid-frontal (F3, F4, Fz), CENTRAL, PARIETAL, TEMPORAL, or OCCIPITAL regions",
      "anatomical_boundary": "Must be strongest at electrodes DIRECTLY ABOVE/BESIDE the PERIOCULAR regions (Fp1, Fp2, F7, F8) with no dipolar pattern extension to mid-frontal (F3, F4, Fz), CENTRAL, PARIETAL, TEMPORAL, or OCCIPITAL regions",
      "key_rule": "If you see dipolar pattern ONLY ABOVE/BESIDE the eyes (Fp1, Fp2, F7, F8) with 1/f spectrum → 'eye'. If dipolar pattern includes ANY electrodes over CENTRAL, PARIETAL, TEMPORAL, or OCCIPITAL regions → NOT eye",
      "priority": "Periocular dipolar patterns = eye, frontal cortical patterns that extend to mid-frontal (F3, F4, Fz) or central regions = brain, single spots = muscle/noise"
    },
    "muscle": {
      "decisive_spectral_feature": "The power spectrum exhibits a CLEAR and SUSTAINED POSITIVE SLOPE, meaning power consistently INCREASES with increasing frequency, starting from the available frequency range upwards (typically from around 20-30Hz if present, or from the lowest available frequency if data is high-pass filtered above 30Hz). This often looks like the spectrum is 'curving upwards' or 'scooping upwards' at higher frequencies. IF THIS DISTINCT SPECTRAL SIGNATURE IS OBSERVED, THE COMPONENT IS TO BE CLASSIFIED AS 'muscle', EVEN IF other features might seem ambiguous or resemble other categories",
      "important_note": "Muscle artifacts often appear as SINGLE FOCAL SPOTS that look like channel noise. The distinction is BIOLOGICAL PLAUSIBILITY - focal spots near muscle areas (temporal, frontal, jaw) are usually muscle, not channel noise",
      "continuous_data_segments": "Muscle artifacts often show sporadic, high-frequency bursts or irregular patterns across trials, unlike the consistent patterns of brain activity",
      "other_characteristics": {
        "topography": "Common patterns include (a) very localized 'bowtie' or 'shallow dipole' patterns (two small, adjacent areas of opposite polarity, often taking up <25% of the scalp map, can appear anywhere but frequently temporal/posterior) OR (b) more diffuse activity, typically along the EDGE of the scalp (temporal, occipital, neck regions) OR (c) single focal spots near muscle locations",
        "time_series": "Often shows spiky, high-frequency, and somewhat erratic activity"
      }
    },
    "heart": {
      "primary_indicator": "TIME SERIES must show ANY KIND of rhythmic deflections (upward spikes, downward dips, or step changes) that repeat approximately every 1 second (0.8-1.5 seconds apart). These can be subtle - look for ANY regular pattern that occurs roughly once per second",
      "continuous_data_segments": "Regular vertical bands or stripes occurring ~once per second across trials - this shows consistent cardiac activity across time segments",
      "decisive_rule": "If you see ANY rhythmic pattern repeating every ~1000ms in the time series OR regular temporal patterns in continuous data segments, classify as 'heart' IMMEDIATELY regardless of all other features",
      "common_patterns": "Heart artifacts often appear as regular downward deflections, upward spikes, or step-like changes occurring ~once per second",
      "topography": "IGNORE topography for heart detection - can be dipolar, broad, focal, or any pattern",
      "spectrum": "IGNORE spectrum for heart detection - often looks brain-like with 1/f pattern",
      "critical": "The ONLY diagnostic feature for heart is rhythmic ~1Hz activity in time series AND/OR consistent temporal patterns in continuous data segments"
    },
    "line_noise": {
      "criterion": "MUST show SHARP PEAK at 50/60Hz in spectrum - NOT a notch/dip (notches are filters, not line noise)",
      "note": "Almost all components show a notch at 60Hz from filtering - this is NOT line noise! Line noise requires a POSITIVE PEAK at 50/60Hz, not a negative dip",
      "important": "If data is high-pass filtered above 50/60Hz, line noise detection is not applicable"
    },
    "channel_noise": {
      "primary_criterion": "SINGLE ELECTRODE 'hot/cold spot' - tiny, isolated circular area WITHOUT an opposite pole (not dipolar)",
      "decisive_spatial_rule": "If topography shows single tiny focal spot with NO clear opposite pole, this is likely channel_noise regardless of spectrum appearance (bad electrodes can have noisy/artifactual spectra)",
      "location_guide": "More common in central/interior regions, but can occur anywhere due to electrode problems",
      "key_distinction_from_muscle": "Muscle can also appear as focal spots but usually has some spatial extent or is part of broader patterns. True channel noise is extremely localized to essentially one electrode",
      "continuous_data_segments": "True channel noise shows extremely localized, often random or flat patterns across trials, different from the structured patterns of other artifacts",
      "compare_with_eye": "Channel noise has only ONE focal point, while eye has TWO opposite poles (dipole). Eye dipoles are also typically larger and more structured",
      "compare_with_muscle": "If uncertain between channel_noise and muscle for focal spots, prioritize spatial characteristics over spectral ones",
      "note": "True channel noise is rare but when present, spatial localization is the most reliable indicator"
    },
    "other_artifact": {
      "description": "Components that don't clearly fit other categories, INCLUDING: Complex multi-polar topographies (multiple hotspots, irregular patterns), Dipolar patterns that are noisy, irregular, or poorly formed, Components with mixed or contradictory features, Any pattern that looks 'brain-like' but is complex, noisy, or atypical, When in doubt between brain and other patterns → choose 'other_artifact'"
    }
  },
  "classification_priority": [
    {
      "step": 1,
      "rule": "EXAMINE TIME SERIES CAREFULLY for ANY rhythmic deflections (up/down spikes, dips, steps) repeating every ~1 second → 'heart' (OVERRIDES all other features)"
    },
    {
      "step": 2,
      "rule": "ELSE IF SINGLE TINY FOCAL SPOT without clear opposite pole (non-dipolar) → 'channel_noise' (spatial characteristics are decisive for bad electrodes, regardless of spectrum)"
    },
    {
      "step": 3,
      "rule": "ELSE IF power spectrum dominated by low frequencies (1-5Hz) and clear dipolar pattern STRICTLY ONLY ANTERIOR to the PERIOCULAR electrodes (Fp1, Fp2, F7, F8) AND ABSOLUTELY NO dipolar pattern extension to mid-frontal (F3, F4, Fz), CENTRAL, PARIETAL, TEMPORAL, or OCCIPITAL regions → 'eye'"
    },
    {
      "step": 4,
      "rule": "ELSE IF Spectrum shows SHARP PEAK (not notch) at 50/60Hz → 'line_noise'"
    },
    {
      "step": 5,
      "rule": "ELSE IF Power spectrum exhibits a CLEAR and SUSTAINED increase in power starting from the available frequency range upwards (typically from ~20-30 Hz if present, or from the lowest available frequency if data is high-pass filtered above 30Hz). The slope must be consistently upward, not flat or decreasing, and the trend should be visually obvious. Disregard any dips or notches at 50 or 60 Hz, as these result from filtering and do not reflect the true slope. → 'muscle'. (THIS IS A DECISIVE RULE FOR MUSCLE. If this spectral pattern is present, classify as 'muscle' even if the topography isn't a perfect 'bowtie' or edge artifact, and before considering 'brain')"
    },
    {
      "step": 6,
      "rule": "ELSE IF Single focal spot near muscle areas (temporal, frontal, jaw) → 'muscle' (many muscle artifacts look like single spots)"
    },
    {
      "step": 7,
      "rule": "ELSE IF (Topography is a clear 'bowtie'/'shallow dipole' OR distinct EDGE activity) AND (Time series is spiky/high-frequency OR spectrum is generally high-frequency without being clearly 1/f and also not clearly a positive slope) → 'muscle' (Secondary muscle check, for cases where the positive slope is less perfect but other muscle signs are strong and it's definitely not brain)"
    },
    {
      "step": 8,
      "rule": "ELSE IF Dipolar pattern in mid-frontal (F3, F4, Fz), CENTRAL, PARIETAL, or TEMPORAL regions (AND NOT already definitively classified as 'muscle' by its spectral signature under rule 5) AND spectrum shows a clear general 1/f pattern (overall DECREASING power with increasing frequency, AND ABSOLUTELY NO sustained positive slope at high frequencies) → 'brain'"
    },
    {
      "step": 9,
      "rule": "ELSE → 'other_artifact'"
    }
  ],
  "important_note": "A 50HZ or 60Hz NOTCH (negative dip) in spectrum is normal filtering, seen in most components, and should NOT be used for classification! Do not include this in your reasoning",
}
RETURN FORMAT: You must respond with ONLY a valid JSON object in the following exact format:
{
    "label": "one_of_the_valid_labels",
    "confidence": 0.95,
    "reason": "detailed_reasoning_for_the_classification"
}

The "label" must be exactly one of: brain, eye, muscle, heart, line_noise, channel_noise, other_artifact
The "confidence" must be a number between 0.0 and 1.0
The "reason" should provide detailed reasoning for your classification decision.

Example JSON response:
{"label": "eye", "confidence": 0.95, "reason": "Strong frontal topography with left-right dipolar pattern (horizontal eye movement) or frontal positivity with spike-like patterns (vertical eye movement/blinks). Low-frequency dominated spectrum and characteristic time series confirm eye activity."}

"""

OPENAI_ICA_PROMPT = """
{
  "scoring_system": {
    "description": "A scoring system for classifying EEG ICA components based on topography, time series, power spectrum, and continuous data segments. Scores are assigned with weights reflecting diagnostic importance. Decisive features override others to prevent misclassifications, with confidence adjusted for strong evidence. Detailed scoring ensures transparency.",
    "features": {
      "topography": {
        "description": "Spatial distribution of component activity across the scalp",
        "weight": {
          "brain": 0.3,
          "eye": 0.4,
          "muscle": 0.2,
          "heart": 0.0,
          "line_noise": 0.0,
          "channel_noise": 0.5,
          "other_artifact": 0.2
        },
        "scoring_criteria": {
          "brain": {"description": "Dipolar pattern in CENTRAL, PARIETAL, or TEMPORAL regions (NOT PERIOCULAR or EDGE-FOCUSED)", "score": {"dipolar_central_parietal_temporal": 0.9, "dipolar_mid_frontal": 0.7, "non_dipolar_or_periocular": 0.1, "edge_focused": 0.1}},
          "eye": {"description": "Clear dipolar pattern ONLY ANTERIOR to PERIOCULAR electrodes (Fp1, Fp2, F7, F8) with NO extension", "score": {"dipolar_periocular_only": 0.95, "bilateral_anterior_electrodes": 0.8, "dipolar_with_extension": 0.1, "non_dipolar": 0.05}},
          "muscle": {"description": "Localized 'bowtie'/'shallow dipole', diffuse EDGE activity, or focal spots near muscle areas (temporal, frontal, jaw)", "score": {"bowtie_or_shallow_dipole": 0.7, "edge_activity": 0.6, "single_focal_spot_muscle_area": 0.5, "other_patterns": 0.2}},
          "heart": {"description": "IGNORE topography", "score": {"any_pattern": 0.0}},
          "line_noise": {"description": "IGNORE topography", "score": {"any_pattern": 0.0}},
          "channel_noise": {"description": "Single tiny focal spot with NO opposite pole, isolated to one electrode", "score": {"single_focal_spot_no_opposite_pole": 0.95, "other_patterns": 0.05}},
          "other_artifact": {"description": "Complex multi-polar or irregular patterns", "score": {"complex_multipolar": 0.7, "noisy_dipolar": 0.5, "other_patterns": 0.3}}
        }
      },
      "time_series": {
        "description": "Component activation over time",
        "weight": {
          "brain": 0.3,
          "eye": 0.3,
          "muscle": 0.2,
          "heart": 0.8,
          "line_noise": 0.0,
          "channel_noise": 0.2,
          "other_artifact": 0.2
        },
        "scoring_criteria": {
          "brain": {"description": "Rhythmic, wave-like WITHOUT abrupt shifts", "score": {"rhythmic_wave_like": 0.9, "smooth_no_spikes": 0.7, "abrupt_shifts_or_spikes": 0.1}},
          "eye": {"description": "Step-like, slow waves, or blink-related", "score": {"step_like_or_blink": 0.9, "slow_waves": 0.7, "other_patterns": 0.1}},
          "muscle": {"description": "Spiky, high-frequency, erratic", "score": {"spiky_high_frequency": 0.8, "erratic": 0.6, "other_patterns": 0.2}},
          "heart": {"description": "Rhythmic deflections ~1 second apart", "score": {"rhythmic_1hz": 0.95, "other_patterns": 0.05}},
          "line_noise": {"description": "IGNORE time series", "score": {"any_pattern": 0.0}},
          "channel_noise": {"description": "Random or flat", "score": {"random_or_flat": 0.8, "other_patterns": 0.2}},
          "other_artifact": {"description": "Mixed or contradictory", "score": {"mixed_contradictory": 0.6, "other_patterns": 0.3}}
        }
      },
      "power_spectrum": {
        "description": "Frequency content showing power variation",
        "weight": {
          "brain": 0.3,
          "eye": 0.2,
          "muscle": 0.5,
          "heart": 0.0,
          "line_noise": 0.9,
          "channel_noise": 0.2,
          "other_artifact": 0.2
        },
        "scoring_criteria": {
          "brain": {"description": "1/f-like with peaks at 6-35Hz", "score": {"1f_with_peaks_6_35hz": 0.9, "1f_no_peaks": 0.7, "positive_slope_or_flat": 0.1}},
          "eye": {"description": "Low frequencies (1-5Hz) with 1/f-like decrease", "score": {"low_freq_1_5hz_1f": 0.8, "other_patterns": 0.2}},
          "muscle": {"description": "CLEAR POSITIVE SLOPE starting from the available frequency range upwards (typically from ~20-30Hz if present, or from the lowest available frequency if data is high-pass filtered above 30Hz) (decisive)", "score": {"positive_slope_available_freq": 0.95, "high_freq_no_clear_slope": 0.4, "1f_like": 0.05}},
          "heart": {"description": "IGNORE spectrum", "score": {"any_pattern": 0.0}},
          "line_noise": {"description": "SHARP PEAK at 50/60Hz (if these frequencies are present in the data; if high-pass filtered above 50/60Hz, line noise detection is not applicable)", "score": {"sharp_peak_50_60hz": 0.95, "notch_or_other": 0.05}},
          "channel_noise": {"description": "Noisy or artifactual", "score": {"noisy_artifactual": 0.7, "other_patterns": 0.3}},
          "other_artifact": {"description": "Mixed or contradictory", "score": {"mixed_contradictory": 0.6, "other_patterns": 0.3}}
        }
      },
      "continuous_data_segments": {
        "description": "Trial-by-trial visualization",
        "weight": {
          "brain": 0.1,
          "eye": 0.1,
          "muscle": 0.1,
          "heart": 0.2,
          "line_noise": 0.1,
          "channel_noise": 0.1,
          "other_artifact": 0.4
        },
        "scoring_criteria": {
          "brain": {"description": "Consistent oscillatory", "score": {"consistent_oscillatory": 0.9, "inconsistent_patterns": 0.2}},
          "eye": {"description": "Movement-related", "score": {"movement_related": 0.8, "other_patterns": 0.2}},
          "muscle": {"description": "Sporadic high-frequency bursts", "score": {"sporadic_high_freq_bursts": 0.8, "other_patterns": 0.2}},
          "heart": {"description": "Regular ~1 second bands", "score": {"regular_1hz_bands": 0.95, "other_patterns": 0.05}},
          "line_noise": {"description": "No specific pattern", "score": {"any_pattern": 0.3}},
          "channel_noise": {"description": "Localized, random, or flat", "score": {"random_flat": 0.8, "other_patterns": 0.2}},
          "other_artifact": {"description": "Sporadic or complex", "score": {"sporadic_complex": 0.7, "other_patterns": 0.3}}
        }
      }
    },
    "calculation_method": {
      "description": "Calculate final scores by summing weighted feature scores. Apply priority rules to override non-decisive features. Normalize scores and adjust confidence for decisive features.",
      "steps": [
        {
          "step": 1,
          "description": "Assign scores to each feature for all component types based on observed patterns, ensuring detailed justification."
        },
        {
          "step": 2,
          "description": "Multiply each feature score by its weight to get weighted scores."
        },
        {
          "step": 3,
          "description": "Sum weighted scores for each type to get raw final scores."
        },
        {
          "step": 4,
          "description": "Apply priority rules: If heart's time series or segments score ≥0.95, set heart to 1.0, others to 0.0. If muscle's power spectrum scores ≥0.95 for positive_slope_available_freq, downweight channel_noise topography to 0.1 unless isolated to one electrode. If line_noise's spectrum scores ≥0.95, set line_noise to 1.0, others to 0.0."
        },
        {
          "step": 5,
          "description": "Normalize raw scores across all types to sum to 1.0, unless a priority rule applies."
        },
        {
          "step": 6,
          "description": "Select the type with the highest normalized score. Set confidence to the normalized score, but adjust to max(0.9, normalized_score * 1.5) if a decisive feature (heart’s ~1Hz, muscle’s positive slope, line_noise’s 50/60Hz) scores ≥0.95 with weight ≥0.5. Provide detailed reasoning with all feature scores and comparisons."
        }
      ]
    }
  }
  RETURN FORMAT: You must respond with ONLY a valid JSON object in the following exact format:
{
    "label": "one_of_the_valid_labels",
    "confidence": 0.95,
    "reason": "detailed_reasoning_for_the_classification"
}

The "label" must be exactly one of: brain, eye, muscle, heart, line_noise, channel_noise, other_artifact
The "confidence" must be a number between 0.0 and 1.0
The "reason" should provide detailed reasoning for your classification decision.

Example JSON response:
{"label": "eye", "confidence": 0.95, "reason": "Strong frontal topography with left-right dipolar pattern (horizontal eye movement) or frontal positivity with spike-like patterns (vertical eye movement/blinks). Low-frequency dominated spectrum and characteristic time series confirm eye activity."}
}
"""

# Default configuration parameters
DEFAULT_CONFIG = {
    "confidence_threshold": 0.8,
    "auto_exclude": True,
    "labels_to_exclude": DEFAULT_EXCLUDE_LABELS,
    "batch_size": 10,
    "max_concurrency": 5,
    "model_name": DEFAULT_MODEL,
    "generate_report": True,
}

# Color mapping for visualization
COLOR_MAP = {
    "brain": "#d4edda",  # Light green
    "eye": "#f9e79f",  # Light yellow
    "muscle": "#f5b7b1",  # Light red
    "heart": "#d7bde2",  # Light purple
    "line_noise": "#add8e6",  # Light blue
    "channel_noise": "#ffd700",  # Gold/Orange
    "other_artifact": "#e9ecef",  # Light grey
}

# OpenAI Pricing (as of 2025-05-29)
OPENAI_PRICING = {
    "gpt-4.1": {
        "input": 2.00,  # $2.00 per 1M tokens
        "cached_input": 0.50,  # $0.50 per 1M tokens (cached)
        "output": 8.00,  # $8.00 per 1M tokens
    },
    "gpt-4.1-mini": {
        "input": 0.40,  # $0.40 per 1M tokens
        "cached_input": 0.10,  # $0.10 per 1M tokens (cached)
        "output": 1.60,  # $1.60 per 1M tokens
    },
    "gpt-4.1-nano": {
        "input": 0.10,  # $0.10 per 1M tokens
        "cached_input": 0.025,  # $0.025 per 1M tokens (cached)
        "output": 0.40,  # $0.40 per 1M tokens
    },
}
