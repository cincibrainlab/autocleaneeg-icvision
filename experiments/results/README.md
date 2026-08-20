# Raw results

Raw, unmodified output CSVs for every result reported as a finding in
`plan/plan-log.md`. Each row is one classified component: `set_path`
(subject/source file, the clustering unit), `component_index`, true label,
predicted label(s), confidence, and the model's stated reason.

Deliberately **not** included: smoke tests, or anything that didn't itself
become a reported number in `plan-log.md` — those are cheap, disposable
sanity checks, not evidentiary artifacts, and committing every one of them
would bury the real results in noise. Screening-pass runs (the 78-sample
scale) *are* included once they produce a number that gets cited as a
finding, even a negative one (a prompt underperforming) — the bar is "was
this cited as evidence," not "did it win."

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
| `2026-08-20_gpt4.1_detailed_original_strip_78sample.csv` | `gpt-4.1` + `prompts/detailed_original_strip.txt` (weighted decisive-feature scoring system), same 78-sample as the tightened/combined rows above | 34.6% (below tightened's 41.0% on the identical sample; CIs overlap) |
| `2026-08-20_terra_detailed_original_strip_78sample.csv` | `gpt-5.6-terra` + `prompts/detailed_original_strip.txt`, same 78-sample as the tightened row above | 55.1% (below tightened's 62.8% on the identical sample; CIs overlap) |
| `2026-08-20_gpt4.1_tightened_v2_78sample.csv` | `gpt-4.1` + `prompts/tightened_v2_strip.txt`, same 78-sample | 33.3% (below `tightened_v1_strip.txt`'s 41.0% on the identical sample — the channel_noise fix overcorrected) |
| `2026-08-20_terra_tightened_v2_78sample.csv` | `gpt-5.6-terra` + `prompts/tightened_v2_strip.txt`, same 78-sample | 64.1% (marginally above `tightened_v1_strip.txt`'s 62.8%; CIs overlap — mixed/inconclusive result across the two models) |
| `2026-08-20_terra_production_default_78sample.csv` | `gpt-5.6-terra` + unmodified `prompts/strip_default.txt`, same 78-sample — the clean model-only comparison point never actually run for `terra` before | 53.8% |
| `2026-08-20_sol_production_default_78sample.csv` | `gpt-5.6-sol` + unmodified `prompts/strip_default.txt`, same 78-sample | 60.3% |
| `2026-08-20_sol_tightened_v1_78sample.csv` | `gpt-5.6-sol` + `prompts/tightened_v1_strip.txt`, 78-sample screen | 70.5% — the screening pass that graduated to the full-679 run below |
| `2026-08-20_sol_tightened_v1_full679.csv` | `gpt-5.6-sol` + `prompts/tightened_v1_strip.txt`, **full 679-set** — graduated from the screening pass above | **69.07% (469/679), subject-clustered 95% CI [64.9%, 79.2%] — highest and tightest result of any configuration tested this session.** Lower CI bound sits almost exactly at ICLabel's 65.98% point estimate; most of the CI's mass is above it. Still overlaps `terra`'s CI (barely). See caveat below on ICLabel's own CI |

All verified against the reported numbers in `plan-log.md` at the time this directory was created (2026-08-20) by running `subject_clustered_scoring.py` on each file directly.

**Prompt length is not the driver of the pattern above** — on this same 78-sample, apples-to-apples (`gpt-4.1`): `strip_default.txt` (168 words, the shipped production prompt) scores worst at 33.3%, `tightened_v1.txt` (704 words) scores best at 41.0%, and `detailed_original_strip.txt` (815 words) and `combined_v1.txt` (949 words) fall in between at 34.6%/35.9%. The shortest prompt is not the best, and the two worst-performing long prompts share a specific structural feature the winner lacks: an explicit numeric weighted-scoring system with "decisive feature" override rules (e.g. "if X's power spectrum scores ≥0.95, set X to 1.0, others to 0.0"). The working hypothesis is that this rigid algorithmic framing, not verbosity itself, causes systematic over-collapse into specific categories (`detailed_original_strip.txt` drove `gpt-4.1`'s `other_artifact` to 0/12 correct; drove `gpt-5.6-terra`'s predictions to lean heavily on `channel_noise`/`brain`). See `experiments/runners/run_screen.py` for the run mechanism.

**Clean model-only comparison** (unmodified `strip_default.txt`, identical 78-sample, zero prompt engineering — the fairest three-way model comparison available): `gpt-4.1` 33.3% < `gpt-5.6-terra` 53.8% < `gpt-5.6-sol` 60.3%.

**Caveat on the ICLabel 65.98% reference used throughout this document**: it is a fixed point estimate, not a subject-clustered interval — nobody has computed one, because ICLabel's own raw per-component predictions on Grace's 679-set were never committed to this directory. Every "beats/loses to ICLabel" claim in `plan-log.md` is therefore comparing a real interval (this project's own models) against a number with no interval of its own. Worth closing as a real gap before any external use of these numbers.
