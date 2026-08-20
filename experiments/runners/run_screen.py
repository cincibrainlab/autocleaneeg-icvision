#!/usr/bin/env python3
"""Generalized 78-sample screening runner: any model x any strip-mode prompt.

Replaces run_detailed_original_strip_screen.py (and the older, retired
run_prompt_variant_strip*.py monkey-patch scripts) with one script
parametrized by --model and --prompt-file, via the real PR #15 custom_prompt
mechanism (classify_components_strip_batch(..., custom_prompt=template)).

Model -> endpoint mapping is fixed here (one canonical endpoint per model,
verified empirically -- see plan/plan-log.md, 2026-08-20 entries for
gpt-5.6-sol's connectivity/vision-path verification), so --model is the only
selector needed; there is no separate --endpoint flag to get out of sync
with it.

Credentials are read from the environment (AZURE_ICVISION_API_KEY,
CLINCOG_API_KEY) -- never hardcode them here.

Usage:
    python run_screen.py --model gpt-4.1 --output out.csv
        (omit --prompt-file to use the unmodified production STRIP_PROMPT_TEMPLATE)
    python run_screen.py --model gpt-5.6-sol --prompt-file /path/to/tightened_v1.txt --output out.csv
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

BASE_DIR = Path("/cblstore/srv/Analysis/Nate_Projects/Projects/IC_Visual_AI")
SAMPLE_CSV = "/tmp/single_mode_sample.csv"

MODELS = {
    "gpt-4.1": {
        "api_key_env": "AZURE_ICVISION_API_KEY",
        "api_version": "2025-03-01-preview",
        "base_url": "https://ext-team-ai-gateway.azure-api.net/external-teams-foundry/openai",
        "auth": "azure",
    },
    "gpt-5.6-terra": {
        "api_key_env": "CLINCOG_API_KEY",
        "api_version": None,
        "base_url": "https://openai.cincibrainlab.com/v1",
        "auth": "bearer",
    },
    "gpt-5.6-sol": {
        "api_key_env": "CLINCOG_API_KEY",
        "api_version": None,
        "base_url": "https://openai.cincibrainlab.com/v1",
        "auth": "bearer",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--prompt-file", default=None, help="Strip-mode custom_prompt template; omit to use the unmodified production STRIP_PROMPT_TEMPLATE")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-csv", default=SAMPLE_CSV)
    parser.add_argument("--strip-size", type=int, default=9)
    args = parser.parse_args()

    mc = MODELS[args.model]
    api_key = os.environ.get(mc["api_key_env"])
    if not api_key:
        sys.exit(f"Missing credential: set {mc['api_key_env']} in the environment before running.")

    if mc["auth"] == "azure":
        _OrigOpenAI = openai.OpenAI

        def _PatchedOpenAI(*a, **kw):
            kw["default_headers"] = {**kw.get("default_headers", {}), "api-key": api_key}
            kw["default_query"] = {**kw.get("default_query", {}), "api-version": mc["api_version"]}
            return _OrigOpenAI(*a, **kw)

        api.openai.OpenAI = _PatchedOpenAI

    custom_prompt = Path(args.prompt_file).read_text() if args.prompt_file else None
    prompt_name = Path(args.prompt_file).stem if args.prompt_file else "production_default"
    variant_name = f"{args.model}_{prompt_name}"

    sample_rows = list(csv.DictReader(open(args.sample_csv)))
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
            api_key,
            component_indices=zero_based,
            model_name=args.model,
            strip_size=args.strip_size,
            base_url=mc["base_url"],
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
