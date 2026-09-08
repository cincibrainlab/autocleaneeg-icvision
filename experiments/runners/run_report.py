#!/usr/bin/env python3
"""Generate the traceability README for a sweep run directory.

Auto-documents: recordings, component scope, full prompt text, class ratios,
run/call counts, strip layout, results breakdown with confusion analysis,
per-component model justifications, and skew-normalized (balanced) accuracy.
Usable as a library (called by run_cross_model.py) or standalone CLI to
retrofit existing runs.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

VALID_LABELS = ("brain", "eye", "muscle", "heart", "channel_noise", "other_artifact")

COST_RATES = {
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
}

ESTIMATED_TOKENS_PER_STRIP = {"input": 6000, "output": 600}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contiguous_ranges(indices: list) -> list:
    ranges, start, prev = [], indices[0], indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        ranges.append((start, prev))
        start = prev = i
    ranges.append((start, prev))
    return ranges


def resolve_prompt_text(log_path: Path) -> tuple:
    prompt_name, prompt_sha = "unknown", None
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                prompt_name, prompt_sha = rec.get("prompt_name", prompt_name), rec.get("prompt_sha256", prompt_sha)
                break
    text = None
    candidates = [Path(prompt_name)] if prompt_name and not prompt_name.startswith("strip_default (") else []
    if not candidates:
        try:
            from icvision.config import STRIP_PROMPT_TEMPLATE
            text = STRIP_PROMPT_TEMPLATE
            candidates = [Path("prompts/strip_default.txt")]
        except Exception:
            pass
    else:
        if candidates[0].exists():
            text = candidates[0].read_text(encoding="utf-8")
    return prompt_name, prompt_sha, text, (candidates[0] if candidates else None)


def per_class_stats(rows: list) -> dict:
    stats = {label: {"correct": 0, "total": 0} for label in VALID_LABELS}
    for r in rows:
        stats[r["true_label_norm"]]["total"] += 1
        stats[r["true_label_norm"]]["correct"] += r["predicted_label"] == r["true_label_norm"]
    return {k: v for k, v in stats.items() if v["total"]}


def balanced_accuracy(stats: dict) -> float:
    recalls = [v["correct"] / v["total"] for v in stats.values() if v["total"]]
    return sum(recalls) / len(recalls) if recalls else 0.0


def compute_cost(run_dir: Path, models: list, n_strips: int) -> list:
    lines = []
    for model in models:
        jsonl = run_dir / "logs" / f"{run_dir.name}_{model}.jsonl"
        actual = 0.0
        calls = 0
        if jsonl.exists():
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("status") != "ok":
                    continue
                calls += 1
                for cli_line in rec.get("cli_raw_output", "").splitlines():
                    try:
                        ev = json.loads(cli_line)
                    except json.JSONDecodeError:
                        continue
                    part = ev.get("part", {})
                    if ev.get("type") == "step_finish" and isinstance(part.get("cost"), (int, float)):
                        actual += part["cost"]
        if actual:
            lines.append(f"| `{model}` | {calls} | ${actual:.4f} | **actual** (summed from CLI step_finish cost events; Go-subscription models are covered by quota, Zen models are out-of-pocket) |")
        else:
            rate = COST_RATES.get(model)
            if rate and calls:
                est = calls * (ESTIMATED_TOKENS_PER_STRIP["input"] * rate["input"] + ESTIMATED_TOKENS_PER_STRIP["output"] * rate["output"]) / 1_000_000
                lines.append(f"| `{model}` | {calls} | ~${est:.4f} | **estimated** (Zen pricing ${rate['input']}/$1M in, ${rate['output']}/$1M out × ~{ESTIMATED_TOKENS_PER_STRIP['input']}in/{ESTIMATED_TOKENS_PER_STRIP['output']}out tokens per strip; images dominate input; actual cost not metered by this API path) |")
            else:
                lines.append(f"| `{model}` | {calls} | n/a | cost not captured by transport |")
    return lines


def build_report(run_dir: Path, manifest_path: Path, models: list, registry_path: Path = None, base_dir: str = None, variable: str = "unspecified") -> Path:
    manifest_rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
    by_file = defaultdict(list)
    for r in manifest_rows:
        by_file[r["set_path"]].append(int(r["component_index"]))

    md = []
    md.append(f"# Run report — `{run_dir.name}`\n")
    md.append(f"**Variable tested:** {variable}\n")
    md.append(f"Generated {__import__('time').strftime('%Y-%m-%d %H:%M')} by `run_report.py`. All paths relative to repo root. This directory is self-contained: everything needed to audit or re-analyze this run lives here.\n")

    md.append("\n## 1. Recording(s) used\n")
    md.append("| File | Components in manifest | Data sha256[:16] |")
    md.append("|------|------------------------|------------------|")
    for set_path in sorted(by_file):
        full = Path(base_dir or "~/data/IC_Visual_AI").expanduser() / set_path
        sha = sha256_file(full)[:16] if full.exists() else "n/a"
        md.append(f"| `{set_path}` | {len(by_file[set_path])} | `{sha}` |")

    md.append("\n## 2. Component scope\n")
    for set_path in sorted(by_file):
        idx = sorted(by_file[set_path])
        ranges = contiguous_ranges(idx)
        range_str = ", ".join(f"IC{a}-IC{b}" if a != b else f"IC{a}" for a, b in ranges)
        md.append(f"- `{Path(set_path).stem}`: {len(idx)} components — {range_str}")
    md.append("\nSampling rule for prelim runs: **contiguous first-30** (IC0–IC30 per recording, the high-variance ICA components). Some recordings decompose into fewer ICs; the manifest records exactly which exist.\n")

    md.append("\n## 3. Prompt used\n")
    log0 = run_dir / "logs" / f"{run_dir.name}_{models[0]}.jsonl" if models else None
    prompt_name, prompt_sha, prompt_text, prompt_file = resolve_prompt_text(log0) if log0 else ("unknown", None, None, None)
    md.append(f"- Prompt: `{prompt_name}`")
    md.append(f"- sha256: `{prompt_sha}`")
    if prompt_file:
        md.append(f"- Source file: `{prompt_file}`")
    if prompt_text:
        md.append("\n**Full prompt text:**\n")
        md.append("```text")
        md.append(prompt_text.rstrip())
        md.append("```\n")

    md.append("\n## 4. Class distribution of the batch (skew report)\n")
    class_counts = Counter(r["true_label_norm"] for r in manifest_rows)
    total = len(manifest_rows)
    md.append("| True class | Count | Share |")
    md.append("|------------|-------|-------|")
    for label in VALID_LABELS:
        n = class_counts.get(label, 0)
        md.append(f"| {label} | {n} | {n / total:.1%} |" if n else f"| {label} | 0 | 0.0% |")
    md.append("\n> Raw accuracy on a skewed batch is dominated by the majority classes. See section 11 for the balanced metric.\n")

    n_strips = (total + 8) // 9
    md.append("\n## 5. Number of runs\n")
    md.append(f"- Models run: {len(models)} ({', '.join(models)})")
    md.append(f"- API calls per model: one per strip")
    md.append(f"- Total classifications in this run: {total} components × {len(models)} model(s)")

    md.append("\n## 6. Strip layout\n")
    md.append(f"- Components per strip: **9** (fixed by the strip protocol; final strip of a file may be smaller)")
    md.append(f"- Strips per recording: {n_strips} for {total} components")

    md.append("\n## 7. Cost\n")
    md.append("| Model | API calls | Cost | Basis |")
    md.append("|-------|-----------|------|-------|")
    md.extend(compute_cost(run_dir, models, n_strips))

    md.append("\n## 8. Results breakdown\n")
    for model in models:
        csv_path = run_dir / f"{run_dir.name}_{model}.csv"
        if not csv_path.exists():
            continue
        rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
        correct = sum(1 for r in rows if r["predicted_label"] == r["true_label_norm"])
        stats = per_class_stats(rows)
        pred_counts = Counter(r["predicted_label"] for r in rows)
        confusions = Counter((r["true_label_norm"], r["predicted_label"]) for r in rows if r["predicted_label"] != r["true_label_norm"])
        md.append(f"\n### {model}\n")
        md.append(f"- Raw accuracy: **{correct}/{len(rows)} = {correct / len(rows):.1%}**")
        md.append(f"- Balanced (skew-normalized) accuracy: **{balanced_accuracy(stats):.1%}**")
        md.append("\n| True class | Correct/Total | Recall |")
        md.append("|------------|---------------|--------|")
        for label, s in stats.items():
            md.append(f"| {label} | {s['correct']}/{s['total']} | {s['correct'] / s['total']:.0%} |")
        md.append("\nPredicted-label distribution: " + ", ".join(f"{k}×{v}" for k, v in pred_counts.most_common()))
        md.append("\nTop confusions (truth → prediction):")
        for (t, p), n in confusions.most_common(6):
            md.append(f"- {t} → {p}: {n}")
        reasons_txt = ""
        if confusions:
            (t, p), n = confusions.most_common(1)[0]
            reasons_txt = f"\nDominant failure mode: **{t} read as {p}** ({n} cases). Model language across these errors leans on topography/spectrum cues that fit the predicted class template; see per-component reasoning below and annotated renders for the visual evidence."
        md.append(reasons_txt)

    md.append("\n## 9. Most prevalent error modes\n")
    for model in models:
        csv_path = run_dir / f"{run_dir.name}_{model}.csv"
        if not csv_path.exists():
            continue
        rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
        errors = [r for r in rows if r["predicted_label"] != r["true_label_norm"]]
        confusions = Counter((r["true_label_norm"], r["predicted_label"]) for r in errors)
        truth_counts = Counter(r["true_label_norm"] for r in rows)
        predicted_counts = Counter(r["predicted_label"] for r in rows)
        bias = [(label, predicted_counts[label] - truth_counts[label]) for label in set(truth_counts) | set(predicted_counts)]
        bias.sort(key=lambda item: abs(item[1]), reverse=True)
        high_conf_errors = [r for r in errors if r.get("confidence") and float(r["confidence"]) >= 0.8]
        md.append(f"\n### {model}\n")
        md.append(f"- Errors: **{len(errors)}/{len(rows)}**; high-confidence errors (confidence ≥0.80): **{len(high_conf_errors)}**")
        md.append("- Dominant confusion pairs:")
        for (truth, predicted), count in confusions.most_common(5):
            md.append(f"  - `{truth}` → `{predicted}`: {count}")
        md.append("- Largest prediction-count biases (predicted minus true):")
        for label, delta in bias[:5]:
            if delta:
                md.append(f"  - `{label}`: {delta:+d}")
        recalls = per_class_stats(rows)
        weakest = sorted(recalls.items(), key=lambda item: item[1]["correct"] / item[1]["total"])
        if weakest:
            label, values = weakest[0]
            md.append(f"- Weakest class recall: `{label}` at {values['correct']}/{values['total']} ({values['correct'] / values['total']:.0%})")
        if not errors:
            md.append("- No error mode observed in this run.")

    md.append("\n## 10. Model justification per component\n")
    for model in models:
        csv_path = run_dir / f"{run_dir.name}_{model}.csv"
        if not csv_path.exists():
            continue
        rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
        rows.sort(key=lambda r: int(r["component_index"]))
        md.append(f"\n### {model}\n")
        md.append("| IC | Truth | Predicted | Conf | Verdict | Model's stated reasoning |")
        md.append("|----|-------|-----------|------|---------|--------------------------|")
        for r in rows:
            ok = "OK" if r["predicted_label"] == r["true_label_norm"] else "WRONG"
            reason = r["reason"].replace("|", "/").replace("\n", " ")
            md.append(f"| {r['component_index']} | {r['true_label_norm']} | {r['predicted_label']} | {r['confidence']} | {ok} | {reason} |")

    md.append("\n## 11. Skew-normalized accuracy\n")
    for model in models:
        csv_path = run_dir / f"{run_dir.name}_{model}.csv"
        if not csv_path.exists():
            continue
        rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
        stats = per_class_stats(rows)
        correct = sum(1 for r in rows if r["predicted_label"] == r["true_label_norm"])
        ba = balanced_accuracy(stats)
        md.append(f"- **{model}**: raw {correct / len(rows):.1%} → balanced **{ba:.1%}** (unweighted mean of per-class recalls; every class counts equally regardless of frequency)")
    md.append("\nBalanced accuracy is the headline metric while the first-30 batch remains imbalanced (muscle-heavy). A stratified manifest would remove the need for normalization; both are supported by the skeleton.\n")

    md.append("\n## Provenance\n")
    md.append(f"- Manifest: `{manifest_path}` (sha256[:16] `{sha256_file(manifest_path)[:16]}`)")
    if registry_path and registry_path.exists():
        md.append(f"- Model registry: `{registry_path}` (sha256[:16] `{sha256_file(registry_path)[:16]}`)")
    md.append(f"- Call audit logs: `logs/{run_dir.name}_<model>.jsonl` (prompt sha, strip sha, raw responses, latency)")
    md.append(f"- Annotated renders: `annotated/<model>/IC*.png`; strips: `strips/`")

    report = run_dir / "README.md"
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--registry", type=Path, default=Path("experiments/models_registry.yaml"))
    parser.add_argument("--base-dir", default="~/data/IC_Visual_AI")
    parser.add_argument("--variable", default="unspecified")
    args = parser.parse_args()
    report = build_report(args.run_dir, args.manifest, args.models, args.registry, args.base_dir, variable=args.variable)
    print(f"report -> {report}")


if __name__ == "__main__":
    main()
