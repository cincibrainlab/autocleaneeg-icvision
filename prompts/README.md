# Prompt provenance

Each file here is loaded verbatim (`Path.read_text().strip()`) and sent as the
model prompt — no comments or metadata inside the `.txt` files themselves, so
what's tested is exactly what's committed. Provenance lives here instead.

| File | SHA-256 | Origin | Status |
|---|---|---|---|
| `default.txt` | `962f1516a24f7c8793ecc81b7c43d89861baf83344315905590d003d105da931` | Commit `c506659`, 2025-12-23, Ernie Pedapati | **Live production prompt** — loaded by `OPENAI_ICA_PROMPT = load_prompt("default")`, used by single-mode classification |
| `detailed_original.txt` | — | Same commit `c506659`; archived when `default.txt` replaced it | Retired, unused by any code path |
| `tightened_v1.txt` | `c4420d3435629873b9111cc90fa0f64f5d5323fba73356f9f051db382eb81bdc` | Copied verbatim from `prompts/default.txt` on the **unmerged** `terra/integration` branch, commit `c3a804a` ("docs: tighten channel_noise, eye, heart, and fallback cues in ICA prompt") | Not wired into any code path in this repo — tested by substituting its category-guidance text into strip mode's response-format wrapper (see `plan/plan-log.md`, 2026-08-19 entry) |
| `combined_v1.txt` | (see file) | Authored 2026-08-19, synthesizing `tightened_v1.txt` and `detailed_original.txt`, informed by error-pattern analysis on the locked-132 hard-case subset | Same as above — not wired into any code path, tested via the same adapter |

Note: production's real strip-mode prompt is `STRIP_PROMPT_TEMPLATE` in
`src/icvision/config.py` (commit `ea5f683`, 2026-01-15, Ernie Pedapati) — it
does **not** live in this directory as a `.txt` file, since it's built
programmatically (component labels + JSON-array format are interpolated in).
