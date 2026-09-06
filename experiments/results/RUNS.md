# Run index — one line per run; details live inside each run directory

Naming: `<variable>__<value>__<context>` — exactly one variable tested per run, everything else frozen as control.

| Run | When | Variable tested | Models | Manifest | Accuracy |
|-----|------|-----------------|--------|----------|----------|
| `plumbing__cached-strips__0137-screen10__gpt-5.4-nano` | 2026-09-06 10:35 | (none — plumbing shakedown on cached strips; superseded) | gpt-5.4-nano | stage0_0137_screen10.csv (10 comps) | 20.0% |
| `model__gpt-5.4-nano__0137-first30` | 2026-09-06 11:56 | model: gpt-5.4-nano (baseline measurement) | gpt-5.4-nano | stage0_true_first30.csv (31 comps, IC0-30) | 19.4% raw / 14.0% balanced |
