# Run index — one line per run; details live inside each run directory

Naming: `<variable>__<value>__<context>` — exactly one variable tested per run, everything else frozen as control.

| Run | When | Variable tested | Models | Manifest | Accuracy |
|-----|------|-----------------|--------|----------|----------|
| `plumbing__cached-strips__0137-screen10__gpt-5.4-nano` | 2026-09-06 10:35 | (none — plumbing shakedown on cached strips; superseded) | gpt-5.4-nano | stage0_0137_screen10.csv (10 comps) | 20.0% |
| `model__gpt-5.4-nano__0137-first30` | 2026-09-06 11:56 | model: gpt-5.4-nano (baseline measurement) | gpt-5.4-nano | stage0_true_first30.csv (31 comps, IC0-30) | 19.4% raw / 14.0% balanced |
| `prompt__tightened-v1-vs-strip-default__nano__0137-first30` | 2026-09-06 12:43 | prompt: tightened_v1_strip.txt vs strip_default.txt (same model, same components, same data) | gpt-5.4-nano | `experiments/manifests/stage0_true_first30.csv` (31 comps) | 22.6% |
| `prompt__tightened-v1__nano__0137-first30` | 2026-09-06 12:48 | prompt: tightened_v1_strip.txt vs strip_default.txt (same model, same 31 components, same renders) | gpt-5.4-nano | `experiments/manifests/stage0_true_first30.csv` (31 comps) | 29.0% |
| `model__gpt-5.4-mini__tightened-v1__0137-first30` | 2026-09-06 12:50 | model: gpt-5.4-mini vs gpt-5.4-nano (same prompt: tightened_v1, same 31 components, same renders) | gpt-5.4-mini | `experiments/manifests/stage0_true_first30.csv` (31 comps) | 29.0% |
