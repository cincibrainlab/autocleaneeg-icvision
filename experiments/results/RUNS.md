# Run index — one line per run; details live inside each run directory

Naming: `<variable>__<value>__<context>` — exactly one variable tested per run, everything else frozen as control.

| Run | When | Variable tested | Models | Manifest | Accuracy |
|-----|------|-----------------|--------|----------|----------|
| `plumbing__cached-strips__0137-screen10__gpt-5.4-nano` | 2026-09-06 10:35 | (none — plumbing shakedown on cached strips; superseded) | gpt-5.4-nano | stage0_0137_screen10.csv (10 comps) | 20.0% |
| `model__gpt-5.4-nano__0137-first30` | 2026-09-06 11:56 | model: gpt-5.4-nano (baseline measurement) | gpt-5.4-nano | stage0_true_first30.csv (31 comps, IC0-30) | 19.4% raw / 14.0% balanced |
| `model__gemini-3.5-flash__tightened-v1__0137-first30-cli` | 2026-09-06 13:24 | transport: OpenCode CLI Google route for gemini-3.5-flash (same prompt, same 31 components) | gemini-3.5-flash | `experiments/manifests/stage0_true_first30.csv` (31 comps) | 32.3% |
| `model__grok-4.6__tightened-v1__0137-first30` | 2026-09-06 13:49 | model: grok-4.6 (OpenCode Go via CLI transport; same prompt, same 31 components) | grok-4.6 | `experiments/manifests/stage0_true_first30.csv` (31 comps) | 83.9% |
| `model__deepseek-v4-flash-vision-exp__tightened-v1__0137-first30` | 2026-09-08 08:18 | model: deepseek-v4-flash-vision-exp (OpenCode Go via CLI transport; same prompt, same 31 components) | deepseek-v4-flash-vision-exp | `experiments/manifests/stage0_true_first30.csv` (31 comps) | 54.8% |
| `model__minimax-m3__tightened-v1__0137-first30` | 2026-09-08 08:38 | model: minimax-m3 (OpenCode Go via CLI transport; same prompt, same 31 components) | minimax-m3 | `experiments/manifests/stage0_true_first30.csv` (31 comps) | 45.2% |
| `model__qwen3.6-plus__tightened-v1__0137-first30` | 2026-09-08 08:44 | model: qwen3.6-plus (OpenCode Go via CLI transport; same prompt, same 31 components) | qwen3.6-plus | `experiments/manifests/stage0_true_first30.csv` (31 comps) | 45.2% |
| `model__grok-build-0.1__tightened-v1__0137-first30` | 2026-09-08 08:57 | model: grok-build-0.1 (OpenCode Zen via CLI transport; same prompt, same 31 components) | grok-build-0.1 | `experiments/manifests/stage0_true_first30.csv` (31 comps) | 29.0% |
