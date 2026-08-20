# Plan Log

## 2026-08-19: Production Accuracy Baseline — icvision vs ICLabel

**Context**: PI requested a solid, defensible production accuracy baseline before considering any migration to `gpt-5.6-terra`. This entry documents that investigation end to end, in the order the work should be read (not strictly the order it happened): the infrastructure bug found and fixed first, then the true unmodified-production baseline, then a pre-existing prompt candidate, then a new prompt candidate, then a separate binary-task pipeline-reproduction thread, then honest methodology limitations.

**Ground truth used**: Grace's independently, blindly labeled 679-component set (`updated_master_file.csv`, 12 source `.set` files, 7-category labels), and its locked-132 hard-case subset. All numbers below are scored against this same ground truth unless noted otherwise. Reference point: **MNE-ICLabel scores 65.98% on the full 679-set, 58.3% on the locked-132 subset** — both real, local, zero-cost, already the default classifier inside `autocleaneeg_pipeline`.

### 1. Infrastructure bug found and fixed first

While running the real production strip-batch code path, discovered that `classify_strip_image()`'s markdown-fence JSON stripping only worked when the closing fence was the literal last line of the response — any trailing text after it (which `gpt-4.1` adds fairly often) caused a silent parse failure, with the whole affected strip batch falling back to a fake `other_artifact` result with no indication anything went wrong. `classify_component_image_openai()` (single-image path) had no fence-handling at all. Roughly half of all strip batches failed this way on first measurement.

- **Issue**: [#13](https://github.com/cincibrainlab/autocleaneeg-icvision/issues/13)
- **Fix (PR)**: [#14](https://github.com/cincibrainlab/autocleaneeg-icvision/pull/14) — `_extract_json_payload()`, regex-based fence extraction with bracket-matching fallback, validated against the real captured failing response plus 4 other cases (clean fence, no fence, object vs array, nested braces in string values)
- **Every result below was measured with this fix applied locally**, ahead of the PR merging — numbers are not reproducible from `main` as it stands until #14 lands.

### 2. Real production baseline — unmodified code, unmodified prompts

Two distinct production configurations exist and use two *different* prompts (this was initially mischaracterized mid-investigation and corrected):

| Config | Model | Layout | Prompt (unmodified, shipped) | Sample | Accuracy | Raw results |
|---|---|---|---|---|---|---|
| Single-mode default | `gpt-4.1` | single | `prompts/default.txt` (commit `c506659`, 2025-12-23, Ernie Pedapati) | 78-component stratified sample | **33.3%** | [`experiments/results/2026-08-19_gpt4.1_single_vs_strip_controlled_78sample.csv`](../experiments/results/2026-08-19_gpt4.1_single_vs_strip_controlled_78sample.csv) (`--pred-col single_predicted_label`) |
| Strip-mode default (what the pipeline actually forces) | `gpt-4.1` | strip | `STRIP_PROMPT_TEMPLATE` in `config.py` (commit `ea5f683`, 2026-01-15, Ernie Pedapati) | full 679-set | **40.5%** | [`experiments/results/2026-08-19_gpt4.1_strip_production_baseline_full679.csv`](../experiments/results/2026-08-19_gpt4.1_strip_production_baseline_full679.csv) |

Note: the single-mode row's 33.3% and strip mode's 33.3% on the same 78-sample were both drawn from the one controlled comparison file, [`2026-08-19_gpt4.1_single_vs_strip_controlled_78sample.csv`](../experiments/results/2026-08-19_gpt4.1_single_vs_strip_controlled_78sample.csv) (both `single_predicted_label` and `strip_predicted_label` columns present; select via `--pred-col` when scoring).

Both endpoints were `gpt-4.1` via a freshly-provisioned Azure APIM gateway (`api-key` header + `api-version` query param, not the SDK's default `Authorization: Bearer` — required a client-construction patch at the call site, not a change to icvision's own source).

### 3. Tightened prompt — pre-existing artifact, not authored during this investigation

Written earlier on the unmerged `terra/integration` branch (commit `c3a804a`, "docs: tighten channel_noise, eye, heart, and fallback cues in ICA prompt"), never merged to `main`, never previously accuracy-tested. Strip mode has no `custom_prompt` hook in production code, so this was tested by substituting the prompt's category-guidance text into strip mode's required response-format wrapper (the framing/JSON-array-format portions copied verbatim from the real `STRIP_PROMPT_TEMPLATE`; only category definitions swapped).

| Model | Sample | Accuracy | Note | Raw results |
|---|---|---|---|---|
| `gpt-4.1` | 78-sample | 41.0% | | [`experiments/results/2026-08-19_gpt4.1_tightened_78sample.csv`](../experiments/results/2026-08-19_gpt4.1_tightened_78sample.csv) |
| `gpt-5.6-terra` | 78-sample | 62.8% | **Superseded by full-set result below — sample was stratified toward hard categories, inflating this number** | [`experiments/results/2026-08-19_terra_tightened_78sample.csv`](../experiments/results/2026-08-19_terra_tightened_78sample.csv) |
| `gpt-5.6-terra` | **full 679-set** | **57.14%** | Real, unbiased result. Per-category: channel_noise 93%, eye 69%, heart 65%, brain 61%, muscle 54%, other_artifact 38%. Still **8.8 points below ICLabel's 65.98%**. Dominant remaining error: `muscle→channel_noise` (65 cases) | [`experiments/results/2026-08-20_terra_tightened_full679.csv`](../experiments/results/2026-08-20_terra_tightened_full679.csv) |

### 4. Combined prompt — new work, authored during this investigation

`prompts/combined_v1.txt`, written today, synthesizing elements from the tightened prompt and an older archived prompt (`prompts/detailed_original.txt`, retired 2025-12-23), informed by error-pattern analysis on the locked-132 subset.

| Model | Sample | Accuracy | Raw results |
|---|---|---|---|
| `gpt-4.1` | 78-sample | 35.9% (underperformed the tightened prompt on the same sample/model) | [`experiments/results/2026-08-19_gpt4.1_combined_78sample.csv`](../experiments/results/2026-08-19_gpt4.1_combined_78sample.csv) |

**Not yet run**: combined prompt + `gpt-5.6-terra`, at any scale.

### 5. Separate thread: real E2E pipeline reproduction + ICLabel binary accuracy (RESTORE-RCT)

Distinct task (binary reject/keep, not 7-category) and distinct methodology (full `autocleaneeg_pipeline` v3.0.0 run from raw `.bdf` end to end, not classifier-only comparison against a fixed decomposition) — kept separate from the icvision numbers above, not directly comparable to them.

- Recovered the exact custom task config (`RESTORE_RCT_Biosemi64.py`), SHA-256-verified identical to what the original clinical ground truth was built against; pipeline's fixed `ICA(random_state=97)` default made the fresh run reproduce human-corrected ground truth exactly on the first subject tested (57/57 components).
- **First scoring pass showed misleading ~100% agreement across 9 subjects** — root cause: 7 of 9 subjects had `status=auto` in the QC control sheet, meaning no independent human correction was ever applied; comparing a deterministic rerun against its own prior unreviewed output is close to circular.
- **Corrected**: rescoped to only the 15 subjects across the full cohort with a genuine, independent `manual_fixes/*.json` human correction on record. Result: **898/901 components correct = 99.67%**, 0 false positives, 3 false negatives, $0 cost (ICLabel runs locally, no API involved).

### 6. Methodology limitations — not yet addressed, should be disclosed with any external use of these numbers

1. **Train/test contamination risk**: both the tightened and combined prompts were evaluated on data whose error patterns had already been inspected (the tightened prompt indirectly, the combined prompt directly, during its own authoring today). No held-out set has been used that was never inspected during prompt design.
2. **No repeated trials / variance estimates anywhere** — every number above is a single run. `temperature=0.2` reduces but does not eliminate run-to-run variance; we cannot currently distinguish real effect sizes from noise.
3. **Single-rater ground truth** — Grace's 679-set has no second reviewer and no inter-rater agreement statistic.
4. **Strip-batched components are not i.i.d.** — components sharing a batch/API call are correlated; naive per-component accuracy does not account for this clustering.
5. **Endpoint provenance is not fully confirmed** — neither the Azure `gpt-4.1` gateway nor the CLIProxy `gpt-5.6-terra` gateway used here has been confirmed as the actual endpoint real production traffic uses (`vision.autocleaneeg.org`, referenced elsewhere in this project's history, was never reached with a working credential during this investigation).
6. **Multiple-comparisons**: roughly 8 configurations were tried in an exploratory, iterative fashion without correction for multiple comparisons.

**Status**: `gpt-5.6-terra`, even with the best prompt tested so far, does not yet beat ICLabel on the real 7-category task at true scale (57.14% vs. 65.98%). Planned next steps (not yet run): evidence injection (numeric channel-loading features, targets the dominant `muscle→channel_noise` confusion specifically), reasoning-effort tuning, smaller strip sizes, combined prompt on `gpt-5.6-terra`, and closing the methodology gaps above before treating any of these numbers as final.

### Addendum (same day): closing documentation gaps identified in review

1. **RESTORE-RCT ground truth scope**: RESTORE-RCT is *not* being treated as primary ground truth for the icvision accuracy question — Grace's 679-set is, and remains the anchor for every 7-category number above. The RESTORE-RCT thread (Section 5) stays scoped to what it actually demonstrates (E2E pipeline reproducibility + ICLabel binary accuracy on a real clinical dataset), not extended into a second icvision ground truth. Separately, `ica_control_sheet.csv` was found to be stale for at least 2 of the first 9 subjects checked (RT2-014, RT2-021 — corrections existed in `manual_fixes/*.json` but were never back-propagated to the control sheet); the 15-subject 99.67% result already only uses `manual_fixes/*.json` as the authoritative source when both exist, so this staleness does not affect that number, but it's a known data-quality issue in that ground truth source worth flagging for anyone using it independently.

2. **Label-normalization bug, called out explicitly**: Grace's ground truth (`updated_master_file.csv`) uses shorthand category names (`"other"`, `"channel"`) while icvision predicts full names (`"other_artifact"`, `"channel_noise"`). An initial scoring pass compared these directly as strings, undercounting genuine matches. All accuracy numbers in this document are the corrected values (true labels normalized to icvision's naming before comparison); this note exists so the bug and its fix are traceable, not just silently baked into the final numbers.

3. **The production baseline is the fixed anchor for all future model comparisons**: `gpt-4.1` (resolved snapshot `gpt-4.1-2025-04-14` at time of testing, via Azure), `STRIP_PROMPT_TEMPLATE` (strip layout, the pipeline's forced default) or `prompts/default.txt` (single layout, the package's own default), `strip_size=9`, `temperature=0.2` — **40.5%** (strip, full 679-set) / **33.3%** (single, 78-sample) is the baseline. Every subsequent result in this document (tightened prompt, combined prompt, any future model) is explicitly an *iteration measured against this fixed anchor*, not a new baseline of its own. Any future "does model/prompt X help" claim should state its delta against these two numbers, not against another iteration's result.

4. **Production's `detail` parameter, verified empirically (not assumed)**: production code never sets an explicit `detail` value on the `input_image` block (confirmed by reading `src/icvision/api.py` directly). Rather than assume this defaults to `"auto"`, this was tested directly: sent the same real strip image three ways (`detail="low"`, `detail="high"`, `detail` omitted) to the same Azure `gpt-4.1` endpoint and compared `usage.input_tokens` in the response. Result: `low` → 102 tokens, `high` → 782 tokens, **omitted → 782 tokens** — identical to explicit `high`, not `low`. So production is already effectively running high-detail images despite never saying so explicitly; this was not a hidden cost/accuracy tradeoff being silently applied.

5. **Test scripts — deferred, not fundamental to the baseline claim**: the baseline and every iteration's result is already fully specified in prose + tables above (model, prompt, layout, sample, exact parameters, result) — sufficient to state and defend what the baseline *is*. The runner scripts themselves (currently only on `/tmp` on `cblprod`, not version-controlled) matter for *re-running or extending* this work, not for the baseline claim itself. Deferred; should be revisited before any external/published use of these numbers, since right now none of this is independently reproducible by anyone without those scripts.

6. **Tightened prompt made identifiable**: committed as `prompts/tightened_v1.txt` (exact byte-for-byte copy from `terra/integration` branch commit `c3a804a`, SHA-256 `c4420d34...`), alongside `prompts/README.md` documenting the provenance, SHA-256, and live/retired/experimental status of every file in that directory. Neither `tightened_v1.txt` nor `combined_v1.txt` is wired into any code path — both were tested via the strip-mode prompt adapter described in Section 3.

### Second addendum (same day): deeper audit — findings that materially change how confident to be in the numbers above

These were found by going back and interrogating the ground truth itself, not the code. Unlike the first addendum, at least one of these (subject clustering) changes the actual statistical confidence that should be placed in every comparative claim made in this document, not just how it's presented.

1. **True independent sample size is ~12, not 679.** All 679 components come from only 12 unique recordings. Computed real per-subject accuracy for the two full-679 results already reported above:

   | Config | Pooled accuracy (n=679) | Per-subject range (n=12) | Per-subject mean | Subject-clustered 95% CI | Raw results |
   |---|---|---|---|---|---|
   | `gpt-5.6-terra` + tightened | 57.14% | 45.6% – 78.3% | 60.5% | **[52.6%, 68.3%]** | [`experiments/results/2026-08-20_terra_tightened_full679.csv`](../experiments/results/2026-08-20_terra_tightened_full679.csv) |
   | `gpt-4.1` production baseline | 40.5% | 23.1% – 84.8% | 44.0% | **[32.0%, 56.0%]** | [`experiments/results/2026-08-19_gpt4.1_strip_production_baseline_full679.csv`](../experiments/results/2026-08-19_gpt4.1_strip_production_baseline_full679.csv) |

   Reproduce either CI directly: `python experiments/scoring/subject_clustered_scoring.py experiments/results/<file>.csv`.

   Two consequences that overturn how this document's headline claims should be read: (a) the two configs' subject-clustered intervals **overlap** (52.6%–56.0%) — under proper uncertainty, "terra beats the gpt-4.1 baseline" is no longer a safely settled claim, only a naive per-component calculation made it look that way; (b) terra's upper bound (68.3%) **exceeds ICLabel's 65.98%** — "terra loses to ICLabel by 8.8 points" is similarly not a settled fact once subject clustering is accounted for. Caveat on the caveat: a t-based interval with only 12 clusters is a rough approximation, not precision (normal-distribution assumptions get shaky at this n, especially given how skewed the gpt-4.1 per-subject values are) — but the qualitative direction (real uncertainty is much larger than the pooled numbers implied) is robust.

2. **Grace's 679-set is not exhaustive per file, and the selection process is undocumented.** Every source file except the smallest has real gaps in its component-number sequence (e.g. `0604_vdaudio`: components numbered 1-115, only 103 labeled, 12 missing). We do not know whether omitted components were skipped for being too ambiguous to call, technically excluded, or some other reason. If ambiguous cases were systematically omitted, every accuracy number in this document — including ICLabel's 65.98% reference — is measured on an easier-than-representative sample. This needs to be resolved by asking whoever built `updated_master_file.csv` how components were selected, not something closeable from the data alone.

3. **Zero `line_noise` examples anywhere in the ground truth.** Label distribution across all 679 rows: `{eye: 86, brain: 224, muscle: 200, heart: 17, channel: 29, other: 123}` — no `line_noise` at all. Every number in this document is silently untested on one of the 7 categories every classifier discussed here is actually asked to predict.

4. **Selection effect in choosing the tightened prompt as "the sound candidate."** Its *text* was written independently, before this investigation and not informed by this eval data — that part genuinely escapes the train/test contamination concern in item 1 of the original entry. But *choosing to report it* as the leading candidate happened only after observing it outperform every other tested configuration on this same data, out of roughly 8 tried. That is a real, if smaller, selection effect (closer to "the best of several tried horses" than a blind confirmatory result) and should be described that way, not as an unqualified win.

5. **A candidate fix for the subject-count problem exists but is deliberately deferred, not started.** `IC_Visual_AI_new_APDinASDfiles` (a separate, previously unused dataset — 33 independent subjects vs. this document's 12, ~3,558 labeled components across `chirp` and `rest` tasks) was evaluated as a possible held-out validation set. It would meaningfully shrink the subject-clustered intervals above (~1.7x tighter, from cluster count alone) and is completely unbiased by every test in this document, which would make it a genuine confirmatory check rather than another iteration on already-inspected data. It does **not** fix items 2 or 3 above (same undocumented-selection-gaps pattern found in 28 of its 32 source files; also zero `line_noise` examples) and requires real unbuilt infrastructure (its `.set` files are epoched, not continuous, which the current plotting/classification code path does not support). Decision: **kept in reserve for the next phase of work, not started now**, so it stays available as a genuinely blind test rather than getting used up prematurely.

**Status**: this closes out the baseline-establishment phase of this project's history — every claim above is either backed by a real, validated measurement or explicitly flagged with the uncertainty that remains. Everything from this point forward (prompt/model iteration, evidence injection, the `IC_Visual_AI_new_APDinASDfiles` validation pass, repeated-trials confidence intervals) is modernization built on top of this documented baseline, not part of establishing it.

## 2026-08-20: Results directory established; `detailed_original_strip.txt` smoke test

Committed `experiments/results/` — one CSV per number reported above, verified to reproduce the exact accuracy already stated in this document by scoring each file directly with `experiments/scoring/subject_clustered_scoring.py`. See `experiments/results/README.md` for the full file-by-file index; every table above now links its row to the underlying CSV.

Ran a connectivity/format smoke test of `prompts/detailed_original_strip.txt` (the strip-mode adaptation of the archived weighted-scoring prompt, see `prompts/README.md`) against the `gpt-5.6-terra` / ClinCog endpoint: 8/9 correct on a single 9-component batch. **Not included in `experiments/results/`** and not a reported finding — per the smoke-test convention established this session, a single small batch with no repeated trials, no subject clustering, and picked from whatever was on hand isn't evidence, just a check that the prompt renders correctly and the endpoint round-trips a real response. The planned real test for this prompt is the 78-sample screening pass (both `gpt-4.1` and `gpt-5.6-terra`) described as next steps below — that result, once run, will get its own row and CSV the same way every other result in this document does.

## 2026-08-20: `detailed_original_strip.txt` 78-sample screening — finding: more detail hurts, not helps

Ran the planned screening pass: `prompts/detailed_original_strip.txt` against the same 78-component stratified sample used for every other prompt-variant screen, via the real PR #15 `custom_prompt` mechanism (`experiments/runners/run_detailed_original_strip_screen.py`, not the pre-PR#15 monkey-patch scripts used earlier this session).

| Model | Prompt | Accuracy | Subject-clustered 95% CI (n=12) | Raw results |
|---|---|---|---|---|
| `gpt-4.1` | `detailed_original_strip.txt` | 34.6% (27/78) | [15.4%, 45.0%] | [`experiments/results/2026-08-20_gpt4.1_detailed_original_strip_78sample.csv`](../experiments/results/2026-08-20_gpt4.1_detailed_original_strip_78sample.csv) |
| `gpt-4.1` | `tightened_v1.txt` (for comparison, same sample) | 41.0% (32/78) | [32.2%, 68.3%] | [`experiments/results/2026-08-19_gpt4.1_tightened_78sample.csv`](../experiments/results/2026-08-19_gpt4.1_tightened_78sample.csv) |
| `gpt-5.6-terra` | `detailed_original_strip.txt` | 55.1% (43/78) | [28.9%, 65.2%] | [`experiments/results/2026-08-20_terra_detailed_original_strip_78sample.csv`](../experiments/results/2026-08-20_terra_detailed_original_strip_78sample.csv) |
| `gpt-5.6-terra` | `tightened_v1.txt` (for comparison, same sample) | 62.8% (49/78) | [34.4%, 76.2%] | [`experiments/results/2026-08-19_terra_tightened_78sample.csv`](../experiments/results/2026-08-19_terra_tightened_78sample.csv) |

Both configs underperform the tightened prompt on the identical sample; CIs overlap in both cases (not statistically settled at n=12), but the same-direction result across two different models is a real signal, not just noise. Per-category breakdown shows *why*: `gpt-4.1`'s `other_artifact` collapsed to 0/12 correct, and `gpt-5.6-terra` badly underperformed on `eye` (25%) and `heart` (33%) while over-committing to `channel_noise` (15/15, but at the cost of everything else). **Decision: does not clear the bar for a full-679 run.** Stage 2 of the originally planned two-stage screening process is skipped for this prompt.

**Finding, stated explicitly (confirmed independently by the PI's mentor, who had flagged this same pattern previously and it had been forgotten): more prompt detail does not mean better classification, and can actively hurt it.** But the mechanism is not simply "shorter is better" — on this same 78-sample, apples-to-apples with `gpt-4.1`: the shortest prompt tested, `prompts/strip_default.txt` (168 words, the actual shipped production prompt), scores *worst* of all four at 33.3%; `tightened_v1.txt` (704 words) scores *best* at 41.0%; `detailed_original_strip.txt` (815 words) and `combined_v1.txt` (949 words) fall in between at 34.6%/35.9%. So prompt length alone does not predict accuracy — the two worst-performing *long* prompts share a specific structural feature the best-performing one lacks: an explicit numeric weighted-scoring system with hard "decisive feature" override rules (e.g. "if X's power spectrum scores ≥0.95, set X to 1.0, all others to 0.0"). Working hypothesis: it's this rigid algorithmic framing — which pushes the model toward mechanically applying override rules rather than holistically weighing evidence — that drives the category-collapse behavior observed above, not verbosity itself. `tightened_v1.txt` is longer than the shipped default but contains only descriptive per-category guidance, no scoring algorithm, and is the best performer found so far across every prompt tested this session.

**Next step under consideration, not yet run**: a further-tightened prompt variant that keeps `tightened_v1.txt`'s specific per-category clarifying cues (its apparent strength) while testing whether trimming it further helps or hurts — distinct from the failed hypothesis of *adding* a formal scoring system, which this entry now closes out as tested and rejected.

## 2026-08-20: `tightened_v2_strip.txt` drafted — not yet tested

New candidate: `prompts/tightened_v2_strip.txt`, a strip-native template (unlike `tightened_v1.txt`, which is single-mode text adapted to strip via a wrapper) built through the real PR #15 `custom_prompt` mechanism directly, matching `detailed_original_strip.txt`'s approach. Validated to render correctly via `config.get_strip_prompt(n, template=...)` for n=3, 9, 22.

**Transparency note, same caveat class as `tightened_v1.txt`'s own selection effect (second addendum, item 4) but stronger**: this variant was not written blind. It directly targets `tightened_v1.txt`'s own two weakest points as already measured and reported on Grace's 679-set in Section 3 above — `other_artifact` at 38% (worst category) and the dominant `muscle→channel_noise` confusion (65 cases). Both fixes are informed by having seen those specific numbers: the `channel_noise` criteria now require an explicit AND on topology *and* a non-elevated spectrum (spectral evidence overrides a focal-looking topography), and `other_artifact` guidance now cites its true ~18% base rate (123/679) from the same ground truth, to counter under-use. This is real train/test contamination, more direct than anything flagged so far this session — any accuracy gain on Grace's data should be read as partly "fit to this specific dataset's error pattern," not purely a generalizable prompt improvement, until confirmed on a genuinely unseen set (e.g. the reserved `IC_Visual_AI_new_APDinASDfiles`, per the earlier addendum, item 5).

At 631 words it also happens to land between `tightened_v1.txt` (704) and `strip_default.txt` (168) — not by design, but worth noting given the length-vs-structure finding immediately above.

**Not yet run against any model.**

## 2026-08-20: `gpt-5.6-sol` — connectivity and vision-path verified, not yet screened

Before adding a third model to the test matrix, checked what's actually reachable from cblprod rather than assuming `sol` shares `terra`'s setup:

- `GET /v1/models` on the ClinCog gateway (`https://openai.cincibrainlab.com/v1`, same bearer auth as `terra`) lists `gpt-5.6-sol` alongside `gpt-5.6-terra` and a third, previously-unseen sibling `gpt-5.6-luna`. Same endpoint, same auth — no new client-construction path needed, unlike the Azure gateway.
- Sent a real strip image (the same 9-component grid used for the `detailed_original_strip.txt` smoke test) through `gpt-5.6-sol`'s vision path directly: it correctly described all four panel types (topography, time series, ERP-style segments, power spectrum) — confirms genuine image understanding, not a canned/text-only fallback. Worth checking explicitly given this project's own prior history with vision-routing bugs on other endpoints (see 2026-01-17, "Local Endpoint Test (Vision Routing Fix)").
- Sent the actual strip-mode classification prompt (9 components, same categories as `strip_default.txt`): returned a clean, valid JSON array with 9 well-formed, plausibly-varied objects (label/confidence/reason per component) — response-format compliance confirmed.
- **Infrastructure note, not a `sol`-specific issue**: raw `curl`/`urllib` requests without a browser/SDK-style `User-Agent` get blocked by this gateway's Cloudflare WAF (`error code: 1010`). Reproduced identically against known-working `terra` with the same bare `urllib` request, confirming it's a client-header artifact of ad-hoc debugging, not an access restriction on `sol`. Not a concern for any real test script, since the `openai` Python SDK (what `classify_components_strip_batch` actually uses) sets its own `User-Agent`.

**Smoke test only — not included in `experiments/results/`, no accuracy claim.** Planned next step: same two-stage protocol as every other candidate — 78-sample screen against (a) unmodified `strip_default.txt`, the one model-isolated comparison point that was never actually run for `terra` either, and (b) `tightened_v1.txt`, directly comparable to `terra`'s existing 62.8%/41.0% numbers on the same sample. Also planned: generalizing the current prompt-specific runner (`experiments/runners/run_detailed_original_strip_screen.py`) into one reusable `--model`/`--prompt-file`-parametrized script before adding this third model's runs, rather than writing another near-duplicate one-off script.

## 2026-08-20: Runner generalized; `tightened_v2_strip.txt` and `gpt-5.6-sol` screened — new leading candidate

Replaced `run_detailed_original_strip_screen.py` with `experiments/runners/run_screen.py`, parametrized by `--model`/`--prompt-file` (model→endpoint mapping fixed internally, since it's one canonical endpoint per model, verified empirically above — no separate `--endpoint` flag to get out of sync). Credentials read from environment variables, never hardcoded.

Ran 4 screening passes (same 78-sample stratified set used throughout):

| Model | Prompt | Accuracy | Subject-clustered 95% CI (n=12) | Raw results |
|---|---|---|---|---|
| `gpt-4.1` | `tightened_v2_strip.txt` | 33.3% (26/78) | [15.1%, 43.1%] | [`experiments/results/2026-08-20_gpt4.1_tightened_v2_78sample.csv`](../experiments/results/2026-08-20_gpt4.1_tightened_v2_78sample.csv) |
| `gpt-5.6-terra` | `tightened_v2_strip.txt` | 64.1% (50/78) | [58.3%, 84.9%]\* | [`experiments/results/2026-08-20_terra_tightened_v2_78sample.csv`](../experiments/results/2026-08-20_terra_tightened_v2_78sample.csv) |
| `gpt-5.6-sol` | unmodified `strip_default.txt` | 60.3% (47/78) | [50.7%, 86.4%] | [`experiments/results/2026-08-20_sol_production_default_78sample.csv`](../experiments/results/2026-08-20_sol_production_default_78sample.csv) |
| `gpt-5.6-sol` | `tightened_v1_strip.txt` | **70.5% (55/78)** | [62.8%, 88.4%] | [`experiments/results/2026-08-20_sol_tightened_v1_78sample.csv`](../experiments/results/2026-08-20_sol_tightened_v1_78sample.csv) |

\* This CI is oddly narrow relative to its own pooled accuracy because per-subject accuracy on this run was unusually consistent except for two 0%-scoring outlier files — a reminder that a t-based interval with 12 clusters can look deceptively tight; treat as approximate.

**Bug found and fixed en route**: the first attempt used `prompt_tightened.txt` (single-mode text, ending in a literal `{"label": ...}` JSON object) directly as a strip-mode `custom_prompt` — crashed immediately (`KeyError: '"label"'`), because it was never actually a strip-native template. Every prior `tightened_v1.txt` strip-mode result (including the ones already reported above) was produced by the old monkey-patch scripts, which spliced its category text into a hardcoded wrapper — a real strip template never existed until now. Fixed by authoring `prompts/tightened_v1_strip.txt`: the same `tightened_v1.txt` category guidance, verbatim and unedited, wrapped with strip's real `{n}`/`{labels}`/`{json_example}` placeholders via the actual `custom_prompt` mechanism — reproducing the same tested content through the correct mechanism, not a new variant. Verified to render for n=3, 9, 22 before use.

**`tightened_v2_strip.txt`: mixed, inconclusive result, not a win.** Worse than `tightened_v1_strip.txt` on `gpt-4.1` (33.3% vs 41.0%) and marginally better on `gpt-5.6-terra` (64.1% vs 62.8%); CIs overlap on both. The channel_noise fix (targeting the previously-dominant muscle→channel_noise over-triggering) overcorrected on `gpt-4.1`, collapsing channel_noise recall to 20% (3/15, only 4 total predictions across 78 components) — the opposite failure mode. Does not replace `tightened_v1_strip.txt` as the leading candidate based on this evidence.

**`gpt-5.6-sol` is the new leading candidate.** Both its screening numbers are the highest measured this session — including 60.3% with *zero prompt engineering* (unmodified production prompt), the clean model-only comparison point `terra` never got. `sol` + `tightened_v1_strip.txt` at 70.5% pooled is the first configuration in this project's history to exceed ICLabel's 65.98% full-679 reference at the pooled level. Every CI still overlaps every other config's at n=12 — this is not yet a statistically settled claim, and it is one more entry in an already-large multiple-comparisons pile (now well past the "roughly 8" configurations noted in Section 6 of the original baseline entry; that count should be treated as stale and due for a real tally before any external use of these numbers).

**Recommended next step, not yet run**: graduate `gpt-5.6-sol` + `tightened_v1_strip.txt` to the full 679-set, mirroring exactly how `terra` + `tightened_v1` was promoted from its own 78-sample screen. This is the same incumbent prompt, not a newly-fit one, and fills a real gap (no full-679 result exists yet for any model besides `gpt-4.1` and `terra`).

## 2026-08-20: `gpt-5.6-terra`'s own model-only baseline point, filled in

`terra` never actually had a clean, prompt-unmodified accuracy number — every `terra` result reported so far used the tightened prompt. Ran `gpt-5.6-terra` + unmodified `strip_default.txt` on the same 78-sample: **53.8% (42/78)**, subject-clustered 95% CI [26.1%, 71.0%] ([`experiments/results/2026-08-20_terra_production_default_78sample.csv`](../experiments/results/2026-08-20_terra_production_default_78sample.csv)).

This completes the three-way clean model comparison — same unmodified `strip_default.txt`, same 78-sample, zero prompt engineering on any of them:

| Model | Unmodified prompt, 78-sample | Raw results |
|---|---|---|
| `gpt-4.1` | 33.3% | [`experiments/results/2026-08-19_gpt4.1_single_vs_strip_controlled_78sample.csv`](../experiments/results/2026-08-19_gpt4.1_single_vs_strip_controlled_78sample.csv) (`--pred-col strip_predicted_label`) |
| `gpt-5.6-terra` | 53.8% | [`experiments/results/2026-08-20_terra_production_default_78sample.csv`](../experiments/results/2026-08-20_terra_production_default_78sample.csv) |
| `gpt-5.6-sol` | 60.3% | [`experiments/results/2026-08-20_sol_production_default_78sample.csv`](../experiments/results/2026-08-20_sol_production_default_78sample.csv) |

This is the fairest apples-to-apples model ranking available in this document — no prompt-engineering confound. Directionally consistent with every prompt-engineered comparison run so far (`sol` ≥ `terra` > `gpt-4.1`), though CIs have not been checked pairwise here and should be before treating the ordering as statistically settled.

**In progress, not yet complete**: `gpt-5.6-sol` + `tightened_v1_strip.txt` on the full 679-set (mirrors `terra` + `tightened_v1`'s existing full-679 promotion), using a newly built normalized manifest (`/tmp/grace_679_manifest_normalized.csv` on cblprod, converted from `icvision_baseline_file_map.csv`'s 1-based `component`/raw `grace_label` columns to the 0-based `component_index`/normalized `true_label_norm` schema `run_screen.py` expects — verified against the known label distribution, `{eye: 86, brain: 224, muscle: 200, heart: 17, channel_noise: 29, other_artifact: 123}`, before use).

## 2026-08-20: `gpt-5.6-sol` + `tightened_v1_strip.txt`, full 679-set — new leading result

The full-679 graduation run above completed: **469/679 = 69.07%**, subject-clustered 95% CI **[64.9%, 79.2%]** ([`experiments/results/2026-08-20_sol_tightened_v1_full679.csv`](../experiments/results/2026-08-20_sol_tightened_v1_full679.csv)). Per-category: heart 94%, muscle 77%, brain 74%, eye 66%, other_artifact 51%, channel_noise 48%.

This is the highest pooled accuracy and the tightest, highest-positioned confidence interval of any configuration measured across this entire session — for comparison, `terra` + `tightened_v1` on the same full-679 set: 57.14%, CI [52.6%, 68.3%] (Section 3 above). The 78-sample screening estimate (70.5%) held up closely at full scale (69.07%), which is itself a useful data point: the screening protocol has now correctly predicted a full-679 result twice (`terra` earlier, `sol` now).

**On the ICLabel comparison, stated precisely rather than declared a clean win**: `sol`'s CI lower bound (64.9%) sits almost exactly at ICLabel's 65.98% point estimate — most of the CI's probability mass is above it, but the bottom edge dips just under. This is the closest any configuration has come to a defensible "beats ICLabel" claim, but two things hold it back from being a clean, certain claim: (1) ICLabel's own 65.98% has never had a subject-clustered interval computed — its raw per-component predictions were never committed to `experiments/results/`, so it's being compared as a fixed number against a real interval, not interval-to-interval; (2) `sol`'s CI still overlaps `terra`'s CI (barely — terra's ceiling 68.3% vs sol's floor 64.9%), so `sol` is not yet statistically distinguishable from `terra` either, only directionally ahead with a much tighter interval.

**Honest framing**: this is real, substantial progress — the tightest and highest interval yet, closing most of the gap to ICLabel that the original baseline entry described as an 8.8-point deficit — but "terra loses to ICLabel" from the original baseline entry should now read as "the newest configuration tested, sol + tightened_v1, plausibly matches or exceeds ICLabel, pending an ICLabel confidence interval that doesn't exist yet." Recommended next step: commit ICLabel's raw per-component 679-set predictions to `experiments/results/` so its own subject-clustered CI can finally be computed — this has been a real gap in every ICLabel comparison made in this document, not just this one.

---

## 2026-01-15: RFC-001 Strip Layout Integration

**Document**: `multi-tracing-production.qmd`

**Summary**: Created RFC proposing integration of 9-component strip layout as drop-in replacement for single-image classification. The strip layout reduces API calls from N to N/9 by batching multiple ICA components into a single image.

**Key deliverables**:
- Feature comparison table (27 items) mapping current single-image implementation to strip layout equivalents
- 4-phase implementation plan: Core Integration, Output Compatibility, CLI/API Surface, Batch Windowing
- Mermaid flowcharts documenting both architectures
- Callouts identifying 5 pending design decisions requiring stakeholder input

**Pending decisions**:
1. Strip size configuration (fixed vs configurable)
2. PDF report format in strip mode
3. Default layout behavior
4. Error handling strategy for partial batch failures
5. Validation dataset availability for accuracy comparison

**Status**: Awaiting design decision responses before implementation.

**GitHub Issue**: [#9](https://github.com/cincibrainlab/autocleaneeg-icvision/issues/9) - Feature: Strip Layout Integration for Batch ICA Classification

---

## 2026-01-15: Phase 1 Detailed Execution Plan

**Document**: `multi-tracing-production.qmd` (updated)

**Decision resolved**: Strip size fixed at 9 components per image.

**Remainder handling**: Final batch contains 1-8 components when total is not divisible by 9. Same 4-column layout with fewer rows; prompt specifies exact component count.

**Phase 1 steps defined**:
1. Step 1.1: Extract `plot_single_component_subplot()` to `plotting.py`
2. Step 1.2: Extract `create_strip_image()` to `plotting.py`
3. Step 1.3: Add `classify_strip_image()` to `api.py`
4. Step 1.4: Create batch orchestration with windowing logic
5. Step 1.5: Integration dispatch in `classify_components_batch()`

**Pitfalls identified**: Precomputed sources optimization, axis indexing, figure memory management, JSON parsing edge cases, component count validation, DataFrame schema parity.

**Status**: Ready to begin Phase 1 implementation.

---

## 2026-01-15: Phase 1 Implementation Complete

**Summary**: Implemented all Phase 1 steps for strip layout integration.

**Changes made**:

1. **`plotting.py`** - Added strip layout functions:
   - `plot_single_component_subplot()`: Plots a single ICA component into provided axes dict (topo, ts, erp, psd)
   - `create_strip_image()`: Creates a strip image with N components in 4-column layout (topo | ts | erp | psd per row)

2. **`config.py`** - Added strip prompt:
   - `STRIP_PROMPT_TEMPLATE`: Multi-component classification prompt supporting 1-52 components
   - `get_strip_prompt(n)`: Generates formatted prompt with letter labels (A-Z, AA-AZ)

3. **`api.py`** - Added strip classification functions:
   - `classify_strip_image()`: Sends strip image to API, parses JSON array response, maps letter labels to component indices
   - `classify_components_strip_batch()`: Batch orchestration with windowing (processes N components in batches of `strip_size`)
   - Updated `classify_components_batch()` with `layout` and `strip_size` parameters for dispatch

**Key implementation details**:
- Precomputed ICA sources optimization implemented
- Remainder handling: Final batch contains 1-8 components with same layout
- Error handling: Failed batches fall back to "other_artifact" classification
- DataFrame schema parity: Strip results produce identical columns as single-image

**API changes**:
```python
classify_components_batch(
    ...,
    layout="strip",      # NEW: "single" or "strip"
    strip_size=9,        # NEW: components per strip (default: 9)
)
```

**Status**: Phase 1 complete. Ready for testing and Phase 2 (output compatibility).

---

## 2026-01-15: Phase 1 Documented & Phase 2 Preflight

**Document**: `multi-tracing-production.qmd` (updated)

**Summary**: Added Phase 1 completion table and Phase 2 preflight assessment to RFC documentation. Updated model references to gpt-5.2 for custom endpoint compatibility.

**Phase 1 completion table**: Documents all 5 implementation steps with file locations and status:
- `plot_single_component_subplot()` at `plotting.py:530`
- `create_strip_image()` at `plotting.py:702`
- `classify_strip_image()` at `api.py:289`
- `classify_components_strip_batch()` at `api.py:461`
- Integration dispatch with `layout` parameter

**Phase 2 preflight findings**:
- **Ready (no changes needed)**: `save_results()`, `_update_ica_with_classifications()`, cleaned Raw export
- **Requires work**: PDF report generation (strip vs individual layout decision), custom prompt file support

**Model update**: All references updated from gpt-4.1 to gpt-5.2 for custom OpenAI endpoint.

**Commit**: `9a97e1c`

**Status**: Phase 2 ready to proceed pending PDF report format decision.

---

## 2026-01-15: Phase 2 Implementation Complete (TDD)

**Summary**: Fixed strip DataFrame schema for output compatibility using TDD approach.

**Problem identified**: Strip layout produced DataFrame with incompatible column names:
- `component` instead of `component_index`
- `ic_type` instead of `label`
- `exclude` instead of `exclude_vision`
- Missing `component_name`

**TDD process**:
1. Wrote 10 tests in `tests/test_strip_compatibility.py` (Red phase)
2. All tests failed initially, confirming schema mismatch
3. Fixed `classify_components_strip_batch()` in `api.py:600-613` (Green phase)
4. All 10 tests passing

**Test categories**:
- `TestDataFrameSchemaParity` (4 tests): Column names, types, format, index
- `TestSaveResultsIntegration` (1 test): CSV export compatibility
- `TestUpdateICAIntegration` (2 tests): ICA object updates, exclusion handling
- `TestRemainderHandling` (1 test): Partial batch validation
- `TestMNELabelMapping` (2 tests): Label mapping verification

**Code fix** (`api.py`):
```python
# Before (incompatible)
{"component": idx, "ic_type": label, "exclude": should_exclude}

# After (compatible)
{"component_index": idx, "component_name": f"IC{idx}", "label": label, "exclude_vision": should_exclude}
```

**Commit**: `60f17dd`

**Status**: Phase 2 complete. Strip layout now produces drop-in compatible output.

---

## 2026-01-15: PDF Report Option A Verified

**Decision**: Option A selected — generate individual images for PDF reports in strip mode.

**Finding**: No code changes required. The existing `generate_classification_report()` function already generates individual component images fresh from the ICA object, independent of how classification was performed. The Phase 2 DataFrame schema fix ensures compatibility.

**Tests added** (`test_strip_compatibility.py`):
- `TestPDFReportIntegration::test_generate_report_accepts_strip_dataframe`
- `TestPDFReportIntegration::test_generate_report_artifacts_only_with_strip_dataframe`

**Test count**: 12/12 passing

**Commit**: `4a9aa97`

**Status**: PDF report generation works with strip mode. 2 of 5 open questions now resolved.

---

## 2026-01-15: Phase 3 & Phase 4 Complete (TDD)

**Summary**: Completed CLI/API surface and error handling using TDD approach.

### Phase 3: CLI and API Surface

**Changes**:
- Added `--layout` flag to CLI (`single`/`strip`, default: `single`)
- Added `--strip-size` flag (default: 9)
- Added `layout` and `strip_size` parameters to:
  - `core.label_components()`
  - `compat.label_components()`
- Parameters flow through to `classify_components_batch()`

**Test suite**: `tests/test_phase3_cli_api.py` (10 tests)

### Phase 4: Error Handling

**Decision**: Retry with exponential backoff selected

**Changes**:
- Added `max_retries` parameter to `classify_strip_image()` (default: 3)
- Extracted `_call_openai_api()` helper for testability
- Implemented exponential backoff: 1s → 2s → 4s
- Exhausted retries fall back to `other_artifact` label

**Test suite**: `tests/test_phase4_retry.py` (7 tests)

### Summary

| Phase | Tests | Status |
|-------|-------|--------|
| Phase 1 (Core) | - | ✅ Complete |
| Phase 2 (Output) | 12 | ✅ Complete |
| Phase 3 (CLI/API) | 10 | ✅ Complete |
| Phase 4 (Retry) | 7 | ✅ Complete |
| **Total** | **29** | **All passing** |

**Commit**: `7266f0b`

**Status**: All 4 phases complete. 4 of 5 open questions resolved. Ready for production testing.

---

## 2026-01-15: Pipeline Integration Documentation

**Document**: `multi-tracing-production.qmd` (updated)

**Summary**: Added comprehensive section documenting how ICVision is integrated in `autocleaneeg_pipeline` and proposed optimizations with strip layout.

**Original implementation documented**:
- Integration point: `ica_processing.py` uses `icvision.compat.label_components()`
- Three classification modes: `iclabel`, `icvision`, `hybrid`
- Hybrid mode: ICLabel on all components, then ICVision reclassifies first N (default: 20)
- DataFrame schema with source metadata: `iclabel_ic_type`, `icvision_ic_type`, etc.
- Fallback behavior: ICVision failure falls back to ICLabel results

**Proposed optimizations**:
- Hybrid mode API call reduction: 20 calls → 3 batches (85% reduction)
- Full ICVision mode: 127 calls → 15 batches (88% reduction)
- Latency savings: ~70s → ~11s for hybrid, ~445s → ~53s for full

**Implementation plan for pipeline**:
| Step | Task | Status |
|------|------|--------|
| 1-3 | ICVision `layout` parameter support | ✅ Complete |
| 4 | Update pipeline kwargs to pass `layout='strip'` | TODO |
| 5 | Integration test hybrid mode + strip | TODO |
| 6 | Accuracy validation study | TODO |

**Backward compatibility**: Default `layout="single"` ensures zero-disruption adoption.

**Commit**: `8d5c9f9`

**Status**: Documentation complete. Ready for pipeline integration when accuracy validated.

---

## 2026-01-15: PDF Report Fix - Preserve Original Raw Data

**Issue**: When using strip mode, PDF report showed incomplete panels for excluded components. Topography rendered correctly, but time series showed scale "1e-15", ERP images were uniform green, and PSD showed flat lines at -200 dB.

**Root cause investigation**:
1. Strip images (`.webp`) rendered all 4 panels correctly — problem was specific to PDF generation
2. Initial fix: Changed `generate_classification_report()` to receive `raw` instead of `raw_cleaned`
3. Problem persisted because `_apply_artifact_rejection()` modified `raw` **in-place** via `ica.apply(raw)`
4. By the time PDF report was generated, `raw` had already been modified

**Solution**: Modified `_apply_artifact_rejection()` to work on a copy:

```python
# Before (in-place modification)
def _apply_artifact_rejection(raw, ica):
    if ica.exclude:
        ica.apply(raw)  # Modifies raw in-place!
    return raw

# After (preserves original)
def _apply_artifact_rejection(raw, ica):
    raw_cleaned = raw.copy()  # Make copy first
    if ica.exclude:
        ica.apply(raw_cleaned)  # Apply to copy
    return raw_cleaned
```

**Files changed**:
- `src/icvision/core.py:393-414` — `_apply_artifact_rejection()` now returns copy
- `tests/test_core.py:374-408` — Updated test to verify copy behavior

**Test results**: 1 related test updated and passing. 57/61 tests passing overall (4 pre-existing failures unrelated to this fix).

**Status**: PDF report fix complete. Original raw data preserved for showing full component visualizations including excluded components.

---

## 2026-01-15: PSD Frequency Limit Change (45Hz Default)

**Issue**: PSD plots showed notch filter artifacts in the 50-60Hz range, making the spectrum appear distorted.

**Solution**: Changed default PSD frequency limit from 80Hz to 45Hz to avoid displaying the notch filter dip region.

**Files changed**:
- `src/icvision/plotting.py`:
  - `plot_component_for_classification()`: 80Hz → 45Hz default
  - `plot_single_component_subplot()`: 55Hz → 45Hz default
  - Updated docstrings in both functions and `create_strip_image()`
- `src/icvision/api.py`: Updated docstring
- `src/icvision/core.py`: Updated docstring
- `src/icvision/cli.py`: Updated help text

**Rationale**: Line noise is typically at 50Hz (Europe/Asia) or 60Hz (Americas). Notch filters create dips in this region that distort the PSD appearance. By capping at 45Hz, we show clean spectral content up to the alpha/beta range without notch filter artifacts.

**Backward compatibility**: Users can still specify higher frequencies via `--psd-fmax` CLI flag or `psd_fmax` parameter.

**Status**: Complete. PSD plots now avoid notch filter artifacts by default.

---

## 2026-01-16: Visual Examples Added to RFC Documentation

**Document**: `multi-tracing-production.qmd` (updated)

**Summary**: Added visual examples section showing strip layout with 45Hz PSD cutoff.

**Changes made**:

1. **`plan/images/strip_example_45hz.png`** - Added example strip image demonstrating:
   - 9 ICA components in 4-column layout (topo, time series, ERP, PSD)
   - PSD plots showing 1-45Hz range (avoiding notch filter region)

2. **`multi-tracing-production.qmd`** - Added "Visual Examples" section:
   - Embedded strip image with caption
   - Key observations callout explaining each column
   - Before vs After callout tip explaining the 45Hz change rationale

3. **`.gitignore`** - Added exception for `plan/**/*.png` to allow documentation images

**Commit**: `2540a16`

**Status**: RFC documentation now includes visual examples of the PSD frequency change.

---

## 2026-01-16: Pipeline Integration Step 4 Complete

**Document**: `multi-tracing-production.qmd` (updated)

**Summary**: Updated `autocleaneeg_pipeline` to use strip layout by default for ICVision classification.

**Changes made**:

1. **`autocleaneeg_pipeline/src/autoclean/functions/ica/ica_processing.py`**:
   - `icvision` method (line 212): Added `icvision_kwargs = {"layout": "strip", **kwargs}`
   - `hybrid` method (line 275): Added `icvision_kwargs = {"layout": "strip", **kwargs}`
   - Updated docstring to document `layout` parameter (default: 'strip')

2. **`multi-tracing-production.qmd`**:
   - Marked step 4 as ✅ Complete in implementation table
   - Added callout documenting the pipeline changes

**Benefits**:
- ~88% reduction in API calls (9 components per call instead of 1)
- Backward compatible (users can override with `layout="single"`)

**Commits**:
- Pipeline: `05b9c37` (autocleaneeg_pipeline)
- RFC: `1c6575e` (autocleaneeg-icvision)

**Status**: Pipeline integration step 4 complete. Steps 5-6 (integration testing, validation study) remain TODO.

---

## 2026-01-16: Pipeline Execution Documentation Added

**Document**: `multi-tracing-production.qmd` (updated)

**Summary**: Added documentation for running and testing the pipeline with strip mode.

**New section**: "Running the Pipeline with Strip Mode"

**Contents**:
1. **Prerequisites**: Workspace setup, `.env` configuration, task file requirements
2. **CLI Commands**: Basic usage, explicit parameters, dry-run, non-interactive mode
3. **Example**: Using BiotrialResting1020 task with test EEG file
4. **Verification**: How to confirm strip mode is active in logs
5. **Override Callout**: How to revert to single-image mode if needed

**Test data paths documented**:
- Task file: `~/sandbox/Autoclean-EEG/tasks/BiotrialResting1020.py`
- Input file: `~/Downloads/qEEG/201001_D1BL_EC.set`
- Output: `~/sandbox/Autoclean-EEG/output`

**Commit**: `ecb83e0`

**Status**: Documentation complete. Ready for integration testing (step 5).

---

## 2026-01-16: Benchmark Comparison - Single vs Strip Mode

**Document**: `multi-tracing-production.qmd` (updated)

**Summary**: Ran real benchmark tests comparing single and strip classification modes.

**Test Configuration**:
- Data: `201001_D1BL_EC_pre_ica_raw.set` (24 ICA components)
- Model: `gpt-5.2`
- Endpoint: `https://openai.cincibrainlab.com/v1`
- PSD cutoff: 45Hz

**Performance Results**:

| Metric | Single Mode | Strip Mode | Improvement |
|--------|-------------|------------|-------------|
| Total Time | 66.50s | 51.24s | 23% faster |
| API Calls | 24 | 3 | 87.5% reduction |
| Est. Cost | $0.29 | ~$0.04 | ~86% savings |

**Classification Agreement**: 19/24 (79.2%)

5 disagreements on ambiguous components with lower confidence scores. Both modes correctly identified clear artifacts and brain components.

**Commit**: `13b8679`

**Status**: Benchmark complete. Strip mode validated with significant cost/time savings.

---

## 2026-01-16: Strip Batch Images Added to RFC

**Document**: `multi-tracing-production.qmd` (updated)

**Summary**: Added actual strip batch images for manual visual review.

**Images added**:
- `strip_batch_0.png`: Components IC0-IC8 (9 components)
- `strip_batch_1.png`: Components IC9-IC17 (9 components)
- `strip_batch_2.png`: Components IC18-IC23 (6 components)

**Documentation updates**:
- New section "Strip Mode Classification Images" with all 3 batches
- Visual review guide callout explaining artifact patterns to look for

**Commit**: `7f7f964`

**Status**: RFC now contains actual classification input images for manual review.

---

## 2026-01-16: API Call Details and Reasoning Mode Documented

**Document**: `multi-tracing-production.qmd` (updated)

**Summary**: Added detailed documentation of the actual API call structure and reasoning mode.

**New section**: "API Call Details"

**Contents**:
1. **API structure**: `client.responses.create()` endpoint with full parameter example
2. **Input format**: User message + image in base64 webp format
3. **Classification prompt**: Full 9-component prompt with category definitions
4. **Reasoning mode callout**: Documents GPT-5.2 behavior:
   - Uses OpenAI Responses API (not Chat Completions)
   - No explicit `reasoning_effort` parameter
   - Temperature 0.2 for consistent outputs
   - Returns JSON with classification + confidence + reasoning

**Commit**: `6d64397`

**Status**: API internals fully documented for transparency.

---

## 2026-01-16: Low Reasoning Mode Comparison

**Document**: `multi-tracing-production.qmd` (updated)

**Summary**: Tested `reasoning_effort='low'` parameter to evaluate potential speed/accuracy tradeoffs.

**Results** (updated with second test run):

| Metric | Default | Low Reasoning |
|--------|---------|---------------|
| Time | 51.24s | 81.85s (+60%) |
| Artifacts | 5 | 7 |

**Implementation**: Added `reasoning_effort` parameter threading through:
- `api.py`: `_call_openai_api`, `classify_strip_image`, `classify_components_strip_batch`, `classify_components_batch`
- `core.py`: `label_components`
- `compat.py`: `label_components`
- `cli.py`: `--reasoning-effort` argument

**Unexpected finding**: Low reasoning was **slower** than default (81.85s vs 51.24s). This may be endpoint-specific behavior with CLIProxy.

**Recommendation**: Use default reasoning (no explicit `reasoning_effort` parameter) for best performance with CLIProxy endpoint.

**Status**: Documented in RFC. Full parameter threading implemented.

---

## 2026-01-17: Local Endpoint Test (Vision Routing Fix)

**Document**: `multi-tracing-production.qmd` (updated)

**Summary**: Tested local endpoint (`http://localhost:28080/v1`) with vision routing fix that prevents image requests from being routed to wrong model.

**Results**:

| Endpoint | Reasoning | Time | Artifacts |
|----------|-----------|------|-----------|
| Production | Default | 51.24s | 5 |
| Production | Low | 81.85s | 7 |
| Production | None | 82.18s | 8 |
| Local (fix) | Default | 68.54s | 7 |
| Local (fix) | Low | 75.41s | 6 |

**Key findings**:
- Local endpoint slower than production default but faster than production low/none
- Local default artifacts (7) = production low artifacts (7)
- Vision routing fix may route to different model configuration
- Notable classification differences: IC6, IC8 as `channel_noise` on local

**Recommendation**: Production with default reasoning for speed; local or production low/none for more aggressive artifact detection.

**Status**: Both endpoints documented and compared in RFC.

---

## 2026-01-17: Final Reasoning Effort Analysis (Production Patched)

**Document**: `multi-tracing-production.qmd` (updated)

**Summary**: Completed comprehensive testing of reasoning effort parameter after CLIProxy production patch.

**Final Results (Production, Patched)**:

| Reasoning | Time | Artifacts | Notes |
|-----------|------|-----------|-------|
| `none` | **50.60s** | 7 | Fastest |
| Default (→medium) | 56.11s | 5 | Proxy default |
| `low` | 81.36s | 4 | Slowest |
| `minimal` | ERROR | - | Not supported |

**Key findings**:
1. Supported values: `none`, `low`, `medium`, `high`, `xhigh`
2. `minimal` is NOT supported by gpt-5.2
3. `low` is paradoxically slower than `medium` (OpenAI API behavior)
4. CLIProxy defaults to `medium` when no parameter provided
5. `none` is fastest option (50.6s vs 56.1s default)

**Root cause** (from CLIProxy team): Proxy translator always sets `reasoning.effort` - either from request or defaulting to `"medium"`. OpenAI appears to have optimized the `"medium"` path for vision tasks.

**Recommendations**:
- Fastest: `--reasoning-effort none`
- Balanced: No parameter (proxy sends `medium`)
- Avoid: `--reasoning-effort low`

**Status**: Complete. RFC updated with final recommendations.
