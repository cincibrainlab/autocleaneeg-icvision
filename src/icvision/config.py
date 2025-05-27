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
OPENAI_ICA_PROMPT = """Analyze this EEG ICA component image and classify into ONE category:

- "brain": Dipolar pattern in CENTRAL, PARIETAL, or TEMPORAL regions (NOT FRONTAL or EDGE-FOCUSED). 1/f-like spectrum with possible peaks at 8-12Hz. Rhythmic, wave-like time series WITHOUT abrupt level shifts. MUST show decreasing power with increasing frequency (1/f pattern) - a flat or random fluctuating spectrum is NOT brain activity.

- "eye":
  * Two main types of eye components:
    1. HORIZONTAL eye movements: Characterized by a TIGHTLY FOCUSED dipolar pattern, CONFINED PRIMARILY to the LEFT-RIGHT FRONTAL regions (e.g., distinct red on one far-frontal side, blue on the opposite far-frontal side). The active areas should be relatively compact and clearly located frontally. Time series typically shows step-like or square-wave patterns. This pattern is eye UNLESS the time series shows the prominent, sharp, repetitive QRS-like spikes characteristic of "heart".
    2. VERTICAL eye movements/blinks: FRONTAL midline or bilateral positivity/negativity. Time series shows distinctive spikes or slow waves.
  * Both types show power concentrated in lower frequencies (<5Hz).
  * DO NOT be misled by 60Hz notches in the spectrum - these are normal filtering artifacts, NOT line noise.
  * Key distinction: Eye components have activity TIGHTLY FOCUSED in frontal regions. Eye component dipoles are much more FOCUSED and less widespread than the broad gradients seen in "heart" components.
  * CRITICAL: NEVER classify a component with clear FOCUSED LEFT-RIGHT FRONTAL dipole as muscle. This pattern is eye, BUT ALWAYS CHECK TIME SERIES FOR QRS COMPLEXES TO RULE OUT "heart" if the 'dipole' appears very broad or global.
  * RULE: If you see TIGHTLY FOCUSED LEFT-RIGHT FRONTAL dipole pattern or STRONG FRONTAL activation with spike patterns, AND NO QRS in time series, classify as "eye".

- "muscle": (SPECTRAL SIGNATURE IS THE MOST DOMINANT INDICATOR)
  * DECISIVE SPECTRAL FEATURE (Primary and Often Conclusive Muscle Indicator): The power spectrum exhibits a CLEAR and SUSTAINED POSITIVE SLOPE, meaning power consistently INCREASES with increasing frequency, typically starting from around 20-30Hz and continuing upwards. This often looks like the spectrum is 'curving upwards' or 'scooping upwards' at higher frequencies. IF THIS DISTINCT SPECTRAL SIGNATURE IS OBSERVED, THE COMPONENT IS TO BE CLASSIFIED AS 'muscle', EVEN IF other features might seem ambiguous or resemble other categories. This spectral cue is the strongest determinant for muscle.
  * OTHER SUPPORTING MUSCLE CHARACTERISTICS (Use if spectral cue is present, or with caution if spectral cue is less definitive but clearly NOT 1/f):
    *   Topography: Common patterns include (a) very localized 'bowtie' or 'shallow dipole' patterns (two small, adjacent areas of opposite polarity, often taking up <25% of the scalp map, can appear anywhere but frequently temporal/posterior) OR (b) more diffuse activity, typically along the EDGE of the scalp (temporal, occipital, neck regions).
    *   Time Series: Often shows spiky, high-frequency, and somewhat erratic activity.

- "heart":
  * TOPOGRAPHY: Characterized by a VERY BROAD, diffuse electrical field gradient across a large area of the scalp. This often manifests as large positive (red) and negative (blue) regions on somewhat opposite sides of the head, but these regions are WIDESPREAD and NOT TIGHTLY FOCUSED like an eye dipole.
  * TIME SERIES (CRITICAL & DECISIVE IDENTIFIER): Look for PROMINENT, SHARP, REPETITIVE SPIKES in the 'Scrolling IC Activity' plot that stand out significantly from the background rhythm. These are QRS-like complexes (heartbeats). They are typically large in amplitude, can be positive-going or negative-going sharp deflections, and repeat at roughly 0.8-1.5 Hz (around once per second, though ICA can make the rhythm appear less than perfectly regular). THE PRESENCE OF THESE DISTINCTIVE, RECURRING, SHARP SPIKES IS THE STRONGEST AND MOST DEFINITIVE INDICATOR FOR "heart".
  * IF QRS IS PRESENT: If these clear, sharp, repetitive QRS-like spikes are visible in the time series, the component should be classified as "heart". This QRS signature, when combined with a BROAD topography, takes precedence over superficial resemblances to other patterns.
  * SPECTRUM: Often noisy or may not show a clear 1/f pattern. May show harmonics of the heart rate.

- "line_noise":
  * MUST show SHARP PEAK at 50/60Hz in spectrum - NOT a notch/dip (notches are filters, not line noise).
  * NOTE: Almost all components show a notch at 60Hz from filtering - this is NOT line noise!
  * Line noise requires a POSITIVE PEAK at 50/60Hz, not a negative dip.

- "channel_noise":
  * SINGLE ELECTRODE "hot/cold spot" - tiny, isolated circular area typically without an opposite pole.
  * Compare with eye: Channel noise has only ONE focal point, while eye has TWO opposite poles (dipole). Eye dipoles are also typically larger and more structured.
  * Example: A tiny isolated red or blue spot on one electrode, not a dipolar pattern.
  * Time series may show any pattern; the focal topography is decisive.

- "other_artifact": Components not fitting above categories.

CLASSIFICATION PRIORITY (IMPORTANT: Evaluate in this order. Later rules apply only if earlier conditions are not met or are ambiguous):
1.  IF 'Scrolling IC Activity' shows PROMINENT, SHARP, REPETITIVE SPIKES (QRS-like complexes...) AND topography is VERY BROAD... → "heart".
2.  ELSE IF TIGHTLY FOCUSED LEFT-RIGHT FRONTAL dipole... (and NO QRS) → "eye"
3.  ELSE IF SINGLE ELECTRODE isolated focality → "channel_noise"
4.  ELSE IF Spectrum shows SHARP PEAK (not notch) at 50/60Hz → "line_noise"
5.  ELSE IF Power spectrum exhibits a CLEAR and SUSTAINED POSITIVE SLOPE (power INCREASES with increasing frequency from ~20-30Hz upwards, often 'curving' or 'scooping' upwards) → "muscle". (THIS IS A DECISIVE RULE FOR MUSCLE. If this spectral pattern is present, classify as 'muscle' even if the topography isn't a perfect 'bowtie' or edge artifact, and before considering 'brain').
6.  ELSE IF (Topography is a clear 'bowtie'/'shallow dipole' OR distinct EDGE activity) AND (Time series is spiky/high-frequency OR spectrum is generally high-frequency without being clearly 1/f and also not clearly a positive slope) → "muscle" (Secondary muscle check, for cases where the positive slope is less perfect but other muscle signs are strong and it's definitely not brain).
7.  ELSE IF Dipolar pattern in CENTRAL, PARIETAL, or TEMPORAL regions (AND NOT already definitively classified as 'muscle' by its spectral signature under rule 5) AND spectrum shows a clear general 1/f pattern (overall DECREASING power with increasing frequency, AND ABSOLUTELY NO sustained positive slope at high frequencies) → "brain"
8.  ELSE → "other_artifact"

IMPORTANT: A 60Hz NOTCH (negative dip) in spectrum is normal filtering, seen in most components, and should NOT be used for classification! Do not include this in your reasoning.

Return: ("label", confidence_score, "detailed_reasoning")

Example: ("eye", 0.95, "Strong frontal topography with left-right dipolar pattern (horizontal eye movement) or frontal positivity with spike-like patterns (vertical eye movement/blinks). Low-frequency dominated spectrum and characteristic time series confirm eye activity.")
"""

# Default configuration parameters
DEFAULT_CONFIG = {
    "confidence_threshold": 0.8,
    "auto_exclude": True,
    "labels_to_exclude": DEFAULT_EXCLUDE_LABELS,
    "batch_size": 10,
    "max_concurrency": 5,
    "model_name": DEFAULT_MODEL,
    "generate_report": False,
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
