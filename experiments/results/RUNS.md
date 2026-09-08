# Run index — one line per run; details live inside each run directory

Naming: `<variable>__<value>__<context>` — exactly one variable tested per run, everything else frozen as control.

| Run | When | Variable tested | Models | Manifest | Accuracy |
|-----|------|-----------------|--------|----------|----------|
| `model__grok-4.6__tightened-v1__0137-first30` | 2026-09-06 13:49 | model: grok-4.6 (OpenCode Go via CLI transport; same prompt, same 31 components) | grok-4.6 | `experiments/manifests/stage0_true_first30.csv` (31 comps) | 83.9% |
| `model__deepseek-v4-flash-vision-exp__tightened-v1__0137-first30` | 2026-09-08 08:18 | model: deepseek-v4-flash-vision-exp (OpenCode Go via CLI transport; same prompt, same 31 components) | deepseek-v4-flash-vision-exp | `experiments/manifests/stage0_true_first30.csv` (31 comps) | 54.8% |
| `model__muse-spark-1.3__tightened-v1__0137-first30` | 2026-09-08 09:53 | model: muse-spark-1.3 (OpenCode Go via CLI transport; same prompt, same 31 components) | muse-spark-1.3 | `experiments/manifests/stage0_true_first30.csv` (31 comps) | 74.2% |
| `model__glm-5.3-flash__tightened-v1__0137-first30` | 2026-09-08 09:54 | model: glm-5.3-flash (OpenCode Go via CLI transport; same prompt, same 31 components) | glm-5.3-flash | `experiments/manifests/stage0_true_first30.csv` (31 comps) | 54.8% |

## Below-threshold / diagnostic runs

These runs are intentionally not listed as headline result rows because accuracy was ≤50% or the run was only a plumbing shakedown. Their result folders remain committed when they contain diagnostic value: each run report records the raw CSV, call log, dominant confusion, prediction bias, high-confidence wrong count, weakest class recall, and per-component model reasoning.

| Run | Accuracy | Diagnostic summary |
|-----|----------|--------------------|
| `plumbing__cached-strips__0137-screen10__gpt-5.4-nano` | 20.0% | plumbing shakedown only; superseded by full first-30 runs |
| `model__gpt-5.4-nano__0137-first30` | 19.4% raw / 14.0% balanced | dominant error: muscle→brain; weak eye recall |
| `model__gemini-3.5-flash__tightened-v1__0137-first30-cli` | 32.3% | dominant error: muscle→channel_noise; many high-confidence wrong calls |
| `model__gpt-5.4-mini__tightened-v1__0137-first30` | 29.0% | dominant error: muscle→brain; weak muscle recall |
| `model__minimax-m3__tightened-v1__0137-first30` | 45.2% | dominant error: muscle→brain; weak heart recall |
| `model__qwen3.6-plus__tightened-v1__0137-first30` | 45.2% | dominant error: muscle→brain; weak eye recall |
| `model__grok-build-0.1__tightened-v1__0137-first30` | 29.0% | dominant error: muscle→eye; weak eye recall |
| `model__mimo-v2.5-free__tightened-v1__0137-first30` | 32.3% | dominant error: muscle→brain; weak eye recall |
