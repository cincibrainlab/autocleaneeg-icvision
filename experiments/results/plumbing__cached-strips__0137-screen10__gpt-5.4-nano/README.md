# Stage 0 run — provenance and storage layout

Run date: 2026-09-06. Purpose: plumbing shakedown of the sweep skeleton, not a scientific result.

## What ran
- Model: `gpt-5.4-nano` (registry tier=cheap, gateway=opencode-go, base_url https://opencode.ai/zen/v1)
- Scope: pilot manifest `experiments/manifests/stage0_0137_first30.csv` = 0137_VDAudio_ICA.set components 0,6,7,19,20,23,25 (the first-30 subset available as cached images)
- Images: cached strip renders from the prior screen-120 run (`~/Desktop/accuracy_sweep/.work/gpt-5.5_medium_0137_VDAudio_ICA/strip_batch_{0,1}.webp`). Strips are model-independent (identical renders across models), 9 components per strip, letter labels A–I burned into the panels.
- NOTE: the initial run also scored 3 out-of-scope late components (IC34, IC46, IC59) because the cached strips carried them. They are excluded from pilot scope; rows remain in the CSV flagged here for transparency.

## Prompt provenance (scientist requirement: prompt on slides + iterations compared)
- Prompt: `prompts/strip_default.txt` (the icvision built-in `strip_default`, unmodified)
- sha256 (first 16): d4c3f0e120964c98
- This is the weakest prompt from the prior iteration testing (33% in prompt tests); it was NOT one of the tightened variants. Prompt-iteration experiments later use `--prompt-file prompts/tightened_v1_strip.txt` etc., and every call logs its prompt hash in the JSONL.

## Storage layout (everything needed for reporting lives in this repo)
- `experiments/results/stage0/stage0_gpt-5.4-nano.csv` — one row per component: set_path, component_index, true_label_norm, predicted_label, confidence, reason
- `experiments/results/stage0/logs/stage0_gpt-5.4-nano.jsonl` — one record per API call: strip file, component indices, prompt name + sha256, strip sha256, raw parsed response, latency, errors
- `experiments/results/stage0/overlays/*.png` — strip render + reasoning panel per component row (predicted label, confidence, truth, PASS/FAIL, model's stated reason)
- `experiments/FINDINGS.md` — append-only findings journal, one block per run
- `experiments/models_registry.yaml` — model definitions (sha256 at run time recorded in JSONL via prompt/gateway provenance)

## Known blocker: true first-30 renders
Fresh strip renders require the raw EEGLAB .set files (ICA decomposition + raw EEG), which live on the lab server at `/cblstore/srv/Analysis/Nate_Projects/Projects/IC_Visual_AI/SavedFiles/` (hosts `cbl` 10.152.2.244 and `cblprod` 10.154.3.172 — both unreachable without VPN as of 2026-09-06). No local copy exists on this machine (searched: no .set files for the 12 study recordings). When on the lab network:

    rsync -av --progress cbl:/cblstore/srv/Analysis/Nate_Projects/Projects/IC_Visual_AI/SavedFiles/ ~/data/IC_Visual_AI/SavedFiles/

(≈ size TBD; 12 files). Then the runner's render mode can produce contiguous first-30 strips for all 12 files and stage 0 re-runs on the true manifest.

## Post-run scope correction (kept for audit honesty)

- The initial run also scored 3 out-of-scope late components (IC34, IC46, IC59) inherited from the cached screen-120 strips; scope was corrected afterwards to the 7 first-30 comps (`stage0_0137_first30.csv`). Pilot-scope accuracy: 1/7. This run is superseded by `../model__gpt-5.4-nano__0137-first30/` (true contiguous first-30, fresh renders) and is retained only as the plumbing-validation artifact.
