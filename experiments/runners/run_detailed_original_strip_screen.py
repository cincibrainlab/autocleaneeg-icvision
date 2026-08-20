#!/usr/bin/env python3
"""78-sample screening run for prompts/detailed_original_strip.txt in strip
mode, via the real PR #15 custom_prompt mechanism (classify_components_strip_batch(...,
custom_prompt=template)) -- not the pre-PR#15 monkey-patch approach the older
run_prompt_variant_strip*.py scripts used. detailed_original_strip.txt is a
complete template (its own framing + JSON schema + scoring-system block, with
{n}/{labels}/{json_example} placeholders), not bare category-guidance text to
be spliced into a hardcoded wrapper, so it goes through custom_prompt as-is.

Endpoint (Azure gpt-4.1 vs ClinCog gpt-5.6-terra) is selected via --endpoint.
Credentials are read from the environment (AZURE_ICVISION_API_KEY,
CLINCOG_API_KEY) -- never hardcode them here.
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/tmp/icvision_prod_test_cwd")
import matplotlib

matplotlib.use("Agg")
import mne

mne.set_log_level("ERROR")
import openai
from icvision import api

# Note: deliberately NOT applying /tmp/fixed_strip_parser.py's monkey-patch here.
# That patch forked classify_strip_image() to work around the markdown-fence
# parsing bug (issue #13) before it was fixed properly in PR #14, and predates
# PR #15's custom_prompt parameter. PR #14 and #15 are both merged and synced
# into /tmp/icvision_prod_test_cwd/icvision/ already, so the native
# classify_strip_image() has both the real fix and custom_prompt support --
# applying the old fork here would silently override it with a stale copy
# that doesn't know about custom_prompt.

BASE_DIR = Path("/cblstore/srv/Analysis/Nate_Projects/Projects/IC_Visual_AI")
SAMPLE_CSV = "/tmp/single_mode_sample.csv"
PROMPT_FILE = "/tmp/icvision_prod_test_cwd/prompts/detailed_original_strip.txt"

ENDPOINTS = {
    "azure": {
        "api_key": os.environ.get("AZURE_ICVISION_API_KEY"),
        "api_version": "2025-03-01-preview",
        "base_url": "https://ext-team-ai-gateway.azure-api.net/external-teams-foundry/openai",
        "model": "gpt-4.1",
        "auth": "azure",
    },
    "clincog": {
        "api_key": os.environ.get("CLINCOG_API_KEY"),
        "api_version": None,
        "base_url": "https://openai.cincibrainlab.com/v1",
        "model": "gpt-5.6-terra",
        "auth": "bearer",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", choices=["azure", "clincog"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ep = ENDPOINTS[args.endpoint]
    if not ep["api_key"]:
        env_var = "AZURE_ICVISION_API_KEY" if args.endpoint == "azure" else "CLINCOG_API_KEY"
        sys.exit(f"Missing credential: set {env_var} in the environment before running.")

    if ep["auth"] == "azure":
        _OrigOpenAI = openai.OpenAI

        def _PatchedOpenAI(*a, **kw):
            kw["default_headers"] = {**kw.get("default_headers", {}), "api-key": ep["api_key"]}
            kw["default_query"] = {**kw.get("default_query", {}), "api-version": ep["api_version"]}
            return _OrigOpenAI(*a, **kw)

        api.openai.OpenAI = _PatchedOpenAI

    custom_prompt = Path(PROMPT_FILE).read_text()
    variant_name = f"detailed_original_strip_{args.endpoint}"

    sample_rows = list(csv.DictReader(open(SAMPLE_CSV)))
    by_file = defaultdict(list)
    for r in sample_rows:
        by_file[r["set_path"]].append(r)
    print(f"[{variant_name}] Sample: {len(sample_rows)} components across {len(by_file)} files")

    all_results = []
    t_start = time.time()

    for set_path, rows in sorted(by_file.items()):
        full_path = BASE_DIR / set_path
        print(f"\n=== {set_path} ({len(rows)} components) ===")
        raw = mne.io.read_raw_eeglab(str(full_path), preload=True)
        ica = mne.preprocessing.read_ica_eeglab(str(full_path))
        zero_based = [int(r["component_index"]) for r in rows]
        truth_by_idx = {int(r["component_index"]): r for r in rows}

        t0 = time.time()
        results_df, meta = api.classify_components_strip_batch(
            ica,
            raw,
            ep["api_key"],
            component_indices=zero_based,
            model_name=ep["model"],
            strip_size=9,
            base_url=ep["base_url"],
            auto_exclude=False,
            custom_prompt=custom_prompt,
            output_dir=Path(f"/tmp/{variant_name}_output/{set_path.replace('/', '_')}"),
        )
        elapsed = time.time() - t0
        print(f"  {set_path}: {elapsed:.1f}s, {meta.get('n_batches')} batches")

        for _, row in results_df.iterrows():
            idx = row["component_index"]
            truth = truth_by_idx.get(idx, {})
            all_results.append(
                {
                    "set_path": set_path,
                    "component_index": idx,
                    "true_label_norm": truth.get("true_label_norm", "UNKNOWN"),
                    "predicted_label": row["label"],
                    "confidence": row["confidence"],
                    "reason": row["reason"],
                }
            )
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["set_path", "component_index", "true_label_norm", "predicted_label", "confidence", "reason"])
            w.writeheader()
            w.writerows(all_results)

    total_elapsed = time.time() - t_start
    print(f"\n[{variant_name}] Total elapsed: {total_elapsed:.1f}s for {len(all_results)} components")

    correct = sum(1 for r in all_results if r["predicted_label"] == r["true_label_norm"])
    n = len(all_results)
    print(f"\n[{variant_name}] ACCURACY: {correct}/{n} = {correct/n:.4f}")

    by_cat = defaultdict(lambda: [0, 0])
    for r in all_results:
        cat = r["true_label_norm"]
        by_cat[cat][1] += 1
        if r["predicted_label"] == cat:
            by_cat[cat][0] += 1
    print(f"\n[{variant_name}] Per-category:")
    for cat, (c, n_) in sorted(by_cat.items()):
        print(f"  {cat:15s} {c}/{n_} = {c/n_:.2f}")

    pred_dist = defaultdict(int)
    for r in all_results:
        pred_dist[r["predicted_label"]] += 1
    print(f"\n[{variant_name}] Predicted distribution: {dict(pred_dist)}")


if __name__ == "__main__":
    main()
