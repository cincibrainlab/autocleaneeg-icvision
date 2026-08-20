# Prompt provenance

Each file here is loaded verbatim (`Path.read_text().strip()`) and sent as the
model prompt — no comments or metadata inside the `.txt` files themselves, so
what's tested is exactly what's committed. Provenance lives here instead.

| File | SHA-256 | Origin | Status |
|---|---|---|---|
| `default.txt` | `962f1516a24f7c8793ecc81b7c43d89861baf83344315905590d003d105da931` | Commit `c506659`, 2025-12-23, Ernie Pedapati | **Live production prompt** — loaded by `OPENAI_ICA_PROMPT = load_prompt("default")`, used by single-mode classification |
| `detailed_original.txt` | — | Same commit `c506659`; archived when `default.txt` replaced it | Retired, unused by any code path. Single-object output format only (see `detailed_original_strip.txt` for the strip-mode adaptation) |
| `tightened_v1.txt` | `c4420d3435629873b9111cc90fa0f64f5d5323fba73356f9f051db382eb81bdc` | Copied verbatim from `prompts/default.txt` on the **unmerged** `terra/integration` branch, commit `c3a804a` ("docs: tighten channel_noise, eye, heart, and fallback cues in ICA prompt") | Not wired into any code path on `main` — tested against strip mode via the `custom_prompt` mechanism added in PR #15 |
| `combined_v1.txt` | (see file) | Authored 2026-08-19, synthesizing `tightened_v1.txt` and `detailed_original.txt`, informed by error-pattern analysis on the locked-132 hard-case subset | Stashed — not currently the active candidate (train/test contamination risk since it was tuned against eval data; see `plan/plan-log.md`) |
| `strip_default.txt` | — | Extracted verbatim from `STRIP_PROMPT_TEMPLATE` (commit `ea5f683`, 2026-01-15, Ernie Pedapati) in PR #15 | **Pending PR #15** — once merged, this becomes the live production strip-mode prompt, loaded the same way `default.txt` is. Template with `{n}`/`{labels}`/`{json_example}` placeholders, formatted per-batch by `get_strip_prompt()` |
| `detailed_original_strip.txt` | — | Authored 2026-08-20: `detailed_original.txt`'s scoring-system content (weights, decisive-feature priority rules), re-wrapped with strip mode's grid framing and JSON-array response format instead of its original single-object format. All literal JSON braces in the scoring block are escaped (`{{`/`}}`) so `str.format()` doesn't misparse them as placeholders — verified to render correctly and produce valid embedded JSON for varying batch sizes (n=3, n=9) | Not yet run against real ground truth — built via the real `custom_prompt` template mechanism from PR #15, not a monkey-patch |

**On `custom_prompt` for strip mode** (PR #15): it's a *template*, not a
pre-rendered string — must contain the same `{n}`/`{labels}`/`{json_example}`
placeholders `strip_default.txt` does, since `classify_components_strip_batch`
reformats it fresh per batch. This matters because the final batch in a run is
often smaller than `strip_size`, and a pre-rendered prompt would have the
wrong label count/JSON schema for it.
