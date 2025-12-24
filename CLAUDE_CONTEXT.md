# Claude Context Dump - ICVision Project

## Current State (2025-12-23)

### What We Built Today

1. **Grid Classification System** (`test_grid_classify.py`)
   - Strip layout for efficient multi-component classification
   - Supports 9-24 components per image (24 is practical limit before timeout)
   - Category-based prompt with alpha peak emphasis
   - Labels show both letter and IC index (e.g., "A:IC47")

2. **Prompt Engineering Results**
   - Category-based prompts (89% accuracy) > Decision trees (67-78%)
   - Key insight: Alpha peak (8-12Hz) = brain, regardless of topography
   - Muscle requires BOTH edge topography AND no alpha peak

3. **Manuscript Figures** (`experiments/grid_tests/`)
   - 6 publication-ready figures with captions
   - fig1: Accuracy-efficiency tradeoff
   - fig2: Prompt engineering comparison
   - fig3: Classification distribution
   - fig4: Confidence score distribution
   - fig5: API efficiency (21x reduction with batch=24)
   - fig6: Model agreement heatmap

### Next Task: Human Rater Web App

**Goal**: Distributed scoring platform for experts across the country

**Design decisions made**:
- React + Vite frontend (responsive, keyboard-driven)
- FastAPI backend (integrates with existing Python code)
- Single-component strips for rapid-fire rating
- Frictionless start - no signup screen required

**Input scheme** (number OR letter):
| Category | Number | Letter |
|----------|--------|--------|
| Brain | 1 | B |
| Eye | 2 | E |
| Muscle | 3 | M |
| Heart | 4 | H |
| Channel Noise | 5 | C |
| Line Noise | 6 | L |
| Other | 7 | O |
| Accept | 0 | SPACE |
| Flag | 9 | ? |
| Back | - | ← |

**Experience badge** ("The Polite Nag"):
- Pulsing badge in corner until user sets level
- Levels: Novice / Trained / Expert
- Can rate without setting, but can't submit at end
- Once set, collapses to quiet header badge

**Data to capture**:
- IC index, model label, model confidence
- Human label, response time (ms), flagged status
- Rater experience level
- Timestamp

**Proposed structure**:
```
/rater/
├── frontend/          # React + Vite
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── ComponentStrip.jsx
│   │   │   ├── RatingButtons.jsx
│   │   │   ├── ExperienceBadge.jsx
│   │   │   └── ProgressBar.jsx
│   │   └── hooks/
│   │       └── useKeyboard.js
│   └── package.json
├── backend/
│   ├── main.py        # FastAPI app
│   ├── models.py      # SQLAlchemy models
│   └── database.py    # DB connection
└── README.md
```

### Key Files

- `test_grid_classify.py` - Main classification script
- `experiments/grid_tests/results.md` - All experiment documentation
- `experiments/grid_tests/plot_figures.py` - Figure generation
- `experiments/grid_tests/fig*.png` - Manuscript figures

### API Configuration

```bash
OPENAI_API_KEY=dev-local-key
OPENAI_BASE_URL=https://vision.autocleaneeg.org/v1
```

Models tested: gpt-5.2, gpt-5.2(low), gpt-5.2(medium), gpt-5.2(high)

### Test Data Location

```
/Users/ernie/Documents/GitHub/Tasks/test-pre-ica-python/
├── D0079_rest_postica_raw.fif
└── D0079_rest_postica-ica.fif
```

### Categories (7 total)

1. **brain** - Dipolar pattern, 1/f spectrum with alpha (8-12Hz) or beta peaks
2. **eye** - Frontal focus, low-freq dominated (<4Hz), large slow deflections
3. **muscle** - Edge topography AND flat/rising high-freq (no alpha)
4. **heart** - ~1Hz rhythmic deflections, broad scalp distribution
5. **line_noise** - Sharp narrow peak at 50/60Hz
6. **channel_noise** - Single focal spot, flat spectrum, erratic time series
7. **other_artifact** - Doesn't fit above

### Git Status

6 commits pushed to origin/main. Clean working tree.

Repository: https://github.com/cincibrainlab/autocleaneeg-icvision.git
