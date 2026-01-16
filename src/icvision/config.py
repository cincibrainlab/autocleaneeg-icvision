"""
Configuration constants and settings for ICVision.

This module contains the OpenAI prompt, label mappings, and other constants
used throughout the ICVision package.
"""

import os
from pathlib import Path

# Default OpenAI model for vision classification
DEFAULT_MODEL = "gpt-4.1"

# Prompt file locations (searched in order)
PROMPT_SEARCH_PATHS = [
    Path.cwd() / "prompts",  # Current working directory
    Path.home() / ".icvision" / "prompts",  # User home
    Path(__file__).parent.parent.parent.parent / "prompts",  # Package prompts dir
]


def load_prompt(prompt_name: str = "default") -> str:
    """Load a prompt from external file.

    Searches for {prompt_name}.txt in:
    1. ./prompts/ (current working directory)
    2. ~/.icvision/prompts/ (user home)
    3. Package prompts/ directory

    Args:
        prompt_name: Name of prompt file (without .txt extension)

    Returns:
        Prompt text content
    """
    filename = f"{prompt_name}.txt"

    for search_path in PROMPT_SEARCH_PATHS:
        prompt_file = search_path / filename
        if prompt_file.exists():
            return prompt_file.read_text().strip()

    # Fallback to built-in short prompt
    return DEFAULT_SHORT_PROMPT


# Short, efficient prompt that works within proxy timeout limits
DEFAULT_SHORT_PROMPT = """Classify this EEG ICA component into ONE category and provide your answer in JSON format:

Categories:
- "brain": Dipolar central/parietal pattern, 1/f spectrum with possible 6-35Hz peaks
- "eye": Frontal/periocular dipolar pattern, low-frequency dominated
- "muscle": Positive spectral slope at high frequencies, edge-focused topography
- "heart": ~1Hz rhythmic deflections in time series
- "line_noise": Sharp peak at 50/60Hz
- "channel_noise": Single isolated focal spot, no dipole
- "other_artifact": Doesn't fit above categories

Respond with JSON: {"label": "category_name", "confidence": 0.0-1.0, "reason": "brief explanation"}"""

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


# Load prompt from external file or use built-in default
OPENAI_ICA_PROMPT = load_prompt("default")


# Strip prompt template for batch classification
# Supports variable number of components (1-9) with letter labels A-I
STRIP_PROMPT_TEMPLATE = """Classify each of the {n} ICA components shown in this grid (labeled {labels}).

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
{json_example}"""


def get_strip_prompt(n_components: int) -> str:
    """Generate classification prompt for N components in a strip image.

    Args:
        n_components: Number of components in the strip (1-52)

    Returns:
        Formatted prompt string
    """
    # Generate labels: A-Z, then AA-AZ for up to 52 components
    single_labels = [chr(ord("A") + i) for i in range(26)]
    double_labels = ["A" + chr(ord("A") + i) for i in range(26)]
    all_labels = single_labels + double_labels
    labels = all_labels[:n_components]
    labels_str = ", ".join(labels)

    # Build JSON example
    json_lines = [
        f'  {{"component": "{lbl}", "label": "category", "confidence": 0.0-1.0, "reason": "brief explanation"}}'
        for lbl in labels
    ]
    json_example = "[\n" + ",\n".join(json_lines) + "\n]"

    return STRIP_PROMPT_TEMPLATE.format(n=n_components, labels=labels_str, json_example=json_example)


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
