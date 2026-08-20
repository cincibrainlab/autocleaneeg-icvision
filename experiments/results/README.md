# Raw results

Raw, unmodified output CSVs for every result reported as a finding in
`plan/plan-log.md`. Each row is one classified component: `set_path`
(subject/source file, the clustering unit), `component_index`, true label,
predicted label(s), confidence, and the model's stated reason.

Deliberately **not** included: smoke tests, screening runs, or anything that
didn't itself become a reported number in `plan-log.md` — those are cheap,
disposable sanity checks, not evidentiary artifacts. Committing every one of
those would bury the real results in noise.

Score any of these directly against `experiments/scoring/subject_clustered_scoring.py`
to independently reproduce the pooled accuracy and subject-clustered
confidence interval reported for it — that's the point of committing them.

| File | Reported as | Pooled accuracy |
|---|---|---|
| `2026-08-19_gpt4.1_strip_production_baseline_full679.csv` | `gpt-4.1`, strip mode, unmodified `STRIP_PROMPT_TEMPLATE`, full 679-set — the real production strip baseline | 40.5% |
| `2026-08-19_gpt4.1_single_vs_strip_controlled_78sample.csv` | `gpt-4.1`, both single mode (`prompts/default.txt`) and strip mode, same 78-component sample — the controlled check for whether layout alone explained the low baseline number. Has both `single_predicted_label` and `strip_predicted_label` columns; score with `--pred-col` to select which | 33.3% (both) |
| `2026-08-19_gpt4.1_tightened_78sample.csv` | `gpt-4.1` + tightened prompt (`prompts/tightened_v1.txt`), 78-sample | 41.0% |
| `2026-08-19_gpt4.1_combined_78sample.csv` | `gpt-4.1` + combined prompt (`prompts/combined_v1.txt`, stashed), 78-sample | 35.9% |
| `2026-08-19_terra_tightened_78sample.csv` | `gpt-5.6-terra` + tightened prompt, 78-sample — **superseded**, sample was stratified toward hard categories | 62.8% |
| `2026-08-20_terra_tightened_full679.csv` | `gpt-5.6-terra` + tightened prompt, full 679-set — the current reference number, subject-clustered 95% CI [52.6%, 68.3%] | 57.14% |

All verified against the reported numbers in `plan-log.md` at the time this directory was created (2026-08-20) by running `subject_clustered_scoring.py` on each file directly.
