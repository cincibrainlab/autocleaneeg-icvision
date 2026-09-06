#!/usr/bin/env python3
"""Registry-driven cross-model classification runner (the sweep skeleton).

Any model in models_registry.yaml, any manifest CSV, zero code changes.

Modes:
  --images-root DIR   use cached strip images rendered by prior runs
                      (looks for {images-root}/*_{stem}/strip_batch_{n}.webp)
                      component->strip mapping is deterministic: per-file
                      sorted indices chunked into strips of --strip-size.

Per model produces:
  results/{tag}_{model}.csv            same schema as prior sweeps
  logs/{tag}_{model}.jsonl             one record per API call (raw + parsed)
  overlays/{tag}_{model}_{stem}_b{n}.png  strip image + reasoning panel
  appends a run block to experiments/FINDINGS.md
"""
import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image, ImageDraw, ImageFont

from icvision.api import StrictClassificationError, classify_strip_image
from cli_transport import classify_strip_cli

FIELDS = ("set_path", "component_index", "true_label_norm", "predicted_label", "confidence", "reason")
VALID_LABELS = {"brain", "eye", "muscle", "heart", "channel_noise", "other_artifact"}
AUTH_JSON = Path.home() / ".local/share/opencode/auth.json"


def load_registry(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_api_key(registry: dict, gateway: str, auth_entry: Optional[str]) -> str:
    spec = registry["api_keys"][gateway]
    key = os.environ.get(spec.get("env", ""))
    if key:
        return key
    if auth_entry and AUTH_JSON.exists():
        data = json.loads(AUTH_JSON.read_text(encoding="utf-8"))
        entry = data.get(auth_entry) or {}
        if entry.get("key"):
            return entry["key"]
    raise SystemExit(f"no API key for gateway '{gateway}' (env {spec.get('env')} or auth.json:{auth_entry})")


def select_models(registry: dict, requested: list, allow_premium: bool) -> list:
    models = registry["models"]
    chosen = []
    for name in requested:
        spec = models.get(name)
        if spec is None:
            raise SystemExit(f"model '{name}' not in registry")
        if not spec.get("enabled", True):
            print(f"skip {name}: disabled in registry")
            continue
        if not spec.get("vision", False):
            print(f"skip {name}: not vision-capable")
            continue
        if spec.get("tier") == "premium" and not allow_premium:
            print(f"skip {name}: premium (needs --allow-premium)")
            continue
        chosen.append((name, spec))
    return chosen


def group_strips(manifest_rows: list, strip_size: int) -> list:
    by_file = defaultdict(list)
    for row in manifest_rows:
        by_file[row["set_path"]].append(row)
    strips = []
    for set_path in sorted(by_file):
        rows = sorted(by_file[set_path], key=lambda r: int(r["component_index"]))
        for i in range(0, len(rows), strip_size):
            strips.append(rows[i : i + strip_size])
    return strips


def find_cached_strip(images_root: Path, set_path: str, batch_idx: int) -> Optional[Path]:
    stem = Path(set_path).stem
    for candidate in sorted(images_root.glob(f"*_{stem}")):
        path = candidate / f"strip_batch_{batch_idx}.webp"
        if path.exists():
            return path
    return None


def render_strip(base_dir: Path, set_path: str, batch_rows: list, out_path: Path, strip_size: int, batch_idx: int, render_cache: dict) -> None:
    import mne

    mne.set_log_level("ERROR")
    from icvision.plotting import create_strip_image

    if set_path not in render_cache:
        full = base_dir / set_path
        raw = mne.io.read_raw_eeglab(str(full), preload=True)
        ica = mne.preprocessing.read_ica_eeglab(str(full))
        render_cache[set_path] = (ica, raw)
    ica, raw = render_cache[set_path]
    indices = [int(r["component_index"]) for r in batch_rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    create_strip_image(ica, raw, indices, out_path)


def verify_strip_layout(path: Path, n_rows: int) -> None:
    width, height = Image.open(path).size
    per_row = height / n_rows
    if not 150 <= per_row <= 450:
        raise ValueError(f"{path} layout mismatch: {height}px / {n_rows} rows = {per_row:.0f}px per row")


def letter(i: int) -> str:
    if i < 26:
        return chr(ord("A") + i)
    return "A" + chr(ord("A") + i - 26)


def wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = f"{current} {word}".strip()
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def make_overlay(strip_path: Path, batch_rows: list[dict], results: dict, out_path: Path, tag: str, model: str) -> None:
    strip = Image.open(strip_path).convert("RGB")
    n = len(batch_rows)
    row_h = strip.height / n
    panel_w, pad = 760, 12
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        bold = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        font = bold = ImageFont.load_default()
    canvas = Image.new("RGB", (strip.width + panel_w, strip.height), "white")
    canvas.paste(strip, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([strip.width, 0, canvas.width, strip.height], fill="#f7f7f7")
    draw.text((strip.width + pad, 8), f"{tag} | {model}", fill="#333", font=bold)
    y = 44
    for i, row in enumerate(batch_rows):
        idx = int(row["component_index"])
        res = results.get(idx)
        pred = res["label"] if res else "MISSING"
        conf = f"{res['confidence']:.2f}" if res else "-"
        reason = res["reason"] if res else "no response for this component"
        truth = row["true_label_norm"]
        ok = pred == truth
        mark, color = ("PASS", "#0a7d32") if ok else ("FAIL", "#b3261e")
        draw.text((strip.width + pad, y), f"{letter(i)} = IC{idx}  pred={pred} ({conf})  truth={truth}  {mark}", fill=color, font=bold)
        y += 26
        for line in wrap(reason, 62)[:3]:
            draw.text((strip.width + pad + 8, y), line, fill="#444", font=font)
            y += 22
        y = max(y, int((i + 1) * row_h)) + 6
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")


def run_model(model: str, spec: dict, registry: dict, strips: list, images_root: Optional[Path], out_dir: Path, tag: str, strip_size: int, prompt_file: Optional[Path] = None) -> dict:
    model_dir = out_dir
    jsonl_path = model_dir / "logs" / f"{tag}_{model}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["strip_key"])
    api_key = resolve_api_key(registry, spec["gateway"], spec.get("auth_entry") or {"opencode-go": "opencode-go", "clincog": None}.get(spec["gateway"]))
    base_url = registry.get("gateway_urls", {}).get(spec.get("gateway"))
    prompt_name = "strip_default (icvision built-in)"
    custom_prompt = None
    if prompt_file is not None:
        custom_prompt = prompt_file.read_text(encoding="utf-8")
        prompt_name = str(prompt_file)
        prompt_sha = hashlib.sha256(custom_prompt.encode()).hexdigest()
    else:
        from icvision.config import STRIP_PROMPT_TEMPLATE
        prompt_sha = hashlib.sha256(STRIP_PROMPT_TEMPLATE.encode()).hexdigest()
    rows_out, stats = [], {"calls": 0, "retries": 0, "components": 0, "parse_ok": 0, "latency": []}
    csv_path = model_dir / f"{tag}_{model}.csv"
    if csv_path.exists():
        rows_out = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    known = {(r["set_path"], int(r["component_index"])) for r in rows_out}

    with jsonl_path.open("a", encoding="utf-8") as log:
        for batch_idx, batch_rows in enumerate(strips):
            set_path = batch_rows[0]["set_path"]
            strip_key = f"{set_path}#b{batch_idx}"
            indices = [int(r["component_index"]) for r in batch_rows]
            if all((set_path, i) in known for i in indices):
                continue
            if images_root is not None:
                strip_path = find_cached_strip(images_root, set_path, batch_idx)
                if strip_path is None:
                    raise SystemExit(f"no cached strip for {set_path} batch {batch_idx} under {images_root}")
            else:
                raise SystemExit("render-from-data mode not enabled yet (needs cblstore mount)")
            verify_strip_layout(strip_path, len(batch_rows))
            record = {"model": model, "strip_key": strip_key, "strip": str(strip_path), "indices": indices, "prompt_name": prompt_name, "prompt_sha256": prompt_sha, "strip_sha256": hashlib.sha256(open(strip_path, "rb").read()).hexdigest()[:16]}
            start = time.time()
            try:
                raw_cli = None
                if spec.get("protocol") == "opencode-cli":
                    from icvision.config import get_strip_prompt
                    cli_prompt = get_strip_prompt(len(indices), template=custom_prompt)
                    parsed, raw_cli = classify_strip_cli(strip_path, cli_prompt, spec["cli_model"], indices)
                else:
                    parsed = classify_strip_image(strip_path, indices, api_key, model_name=model, base_url=base_url, strict_mode=True, custom_prompt=custom_prompt)
            except (StrictClassificationError, Exception) as exc:
                record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "latency_s": round(time.time() - start, 2)})
                log.write(json.dumps(record) + "\n")
                raise
            latency = round(time.time() - start, 2)
            results = {int(r["component_idx"]): r for r in parsed}
            record.update({"status": "ok", "latency_s": latency, "response": parsed})
            if raw_cli is not None:
                record["cli_raw_output"] = raw_cli
            log.write(json.dumps(record) + "\n")
            stats["calls"] += 1
            stats["latency"].append(latency)
            stats["components"] += len(indices)
            stats["parse_ok"] += sum(1 for r in parsed if r.get("label") in VALID_LABELS)
            for row in batch_rows:
                idx = int(row["component_index"])
                res = results.get(idx, {})
                rows_out.append({"set_path": set_path, "component_index": idx, "true_label_norm": row["true_label_norm"], "predicted_label": str(res.get("label", "MISSING")).lower(), "confidence": res.get("confidence", ""), "reason": res.get("reason", "")})
            overlay = model_dir / "overlays" / f"{tag}_{model}_{Path(set_path).stem}_b{batch_idx}.png"
            make_overlay(strip_path, batch_rows, results, overlay, tag, model)
            print(f"  {model} {Path(set_path).stem} b{batch_idx}: {len(indices)} comps in {latency}s -> {overlay.name}")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    stats["median_latency"] = sorted(stats["latency"])[len(stats["latency"]) // 2] if stats["latency"] else 0
    return {"rows": rows_out, "stats": stats}


def score(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(1 for r in rows if r["predicted_label"] == r["true_label_norm"])
    by_class = defaultdict(lambda: [0, 0])
    for r in rows:
        by_class[r["true_label_norm"]][1] += 1
        by_class[r["true_label_norm"]][0] += r["predicted_label"] == r["true_label_norm"]
    return {"n": total, "accuracy": correct / total if total else 0, "by_class": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_class.items())}}


def append_findings(findings: Path, tag: str, model: str, stats: dict, scores: dict) -> None:
    findings.parent.mkdir(parents=True, exist_ok=True)
    if not findings.exists():
        findings.write_text("# Sweep findings journal\n\nOne block per run, newest at the bottom.\n\n---\n\n", encoding="utf-8")
    with findings.open("a", encoding="utf-8") as fh:
        fh.write(f"## {tag} — {model} ({time.strftime('%Y-%m-%d %H:%M')})\n\n")
        fh.write(f"- API calls: {stats['calls']} | components: {stats['components']} | median latency: {stats['median_latency']}s\n")
        fh.write(f"- Parse validity: {stats['parse_ok']}/{stats['components']} responses returned a valid label\n")
        fh.write(f"- Accuracy: {scores['accuracy']:.1%} of {scores['n']} components\n")
        fh.write(f"- Per-class (correct/total): {scores['by_class']}\n")
        fh.write(f"- Overlays + raw call log under the run directory for audit\n\n---\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=Path("experiments/models_registry.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--tag", required=True, help="run tag, e.g. stage0")
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    parser.add_argument("--images-root", type=Path, default=None, help="cached strip images root")
    parser.add_argument("--base-dir", type=Path, default=None, help="root containing SavedFiles/... .set; renders fresh strips (overrides --images-root)")
    parser.add_argument("--strip-size", type=int, default=9)
    parser.add_argument("--allow-premium", action="store_true")
    parser.add_argument("--prompt-file", type=Path, default=None, help="alternate prompt template (default: icvision strip_default)")
    parser.add_argument("--variable", required=True, help="what this run tests, e.g. 'model: gpt-5.4-nano' or 'prompt: tightened-v1 vs strip-default'")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    manifest_rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    if not manifest_rows:
        raise SystemExit("manifest is empty")
    for row in manifest_rows:
        if row["true_label_norm"] not in VALID_LABELS:
            raise SystemExit(f"bad true_label_norm {row['true_label_norm']!r}")
    strips = group_strips(manifest_rows, args.strip_size)
    models = select_models(registry, args.models, args.allow_premium)
    if not models:
        raise SystemExit("no models selected")
    run_dir = args.out / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    n_strips = len(strips)
    print(f"manifest {args.manifest}: {len(manifest_rows)} comps -> {n_strips} strips of <= {args.strip_size}")

    if args.base_dir is not None:
        strips_root = run_dir / "strips"
        render_cache = {}
        by_file = defaultdict(list)
        for batch_idx, batch_rows in enumerate(strips):
            by_file[batch_rows[0]["set_path"]].append((batch_idx, batch_rows))
        for set_path in sorted(by_file):
            for batch_idx, batch_rows in by_file[set_path]:
                out_path = strips_root / f"renders_{Path(set_path).stem}" / f"strip_batch_{batch_idx}.webp"
                if out_path.exists():
                    continue
                render_strip(args.base_dir, set_path, batch_rows, out_path, args.strip_size, batch_idx, render_cache)
                print(f"  rendered {Path(set_path).stem} b{batch_idx} ({len(batch_rows)} comps)")
        args.images_root = strips_root

    for model, spec in models:
        print(f"=== {model} ({spec['family']}, {spec['tier']}, via {spec['gateway']}) ===")
        result = run_model(model, spec, registry, strips, args.images_root, run_dir, args.tag, args.strip_size, args.prompt_file)
        scores = score(result["rows"])
        print(f"  accuracy {scores['accuracy']:.1%} ({scores['n']} comps)")

    from run_report import build_report
    report = build_report(run_dir, args.manifest, [m for m, _ in models], args.registry, str(args.base_dir or "~/data/IC_Visual_AI"), variable=args.variable)
    index = Path("experiments/results/RUNS.md")
    line = f"| `{args.tag}` | {time.strftime('%Y-%m-%d %H:%M')} | {args.variable} | {', '.join(m for m, _ in models)} | `{args.manifest}` ({len(manifest_rows)} comps) | " + ", ".join(f"{score(list(csv.DictReader((run_dir / f'{args.tag}_{m}.csv').open(newline='', encoding='utf-8'))))['accuracy']:.1%}" for m, _ in models if (run_dir / f"{args.tag}_{m}.csv").exists()) + " |\n"
    if not index.exists():
        index.write_text("# Run index — one line per run; details live inside each run directory\n\nNaming: `<variable>__<value>__<context>` — exactly one variable tested per run, everything else frozen as control.\n\n| Run | When | Variable tested | Models | Manifest | Accuracy |\n|-----|------|-----------------|--------|----------|----------|\n", encoding="utf-8")
    with index.open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(f"report -> {report} | index row added -> {index}")


if __name__ == "__main__":
    main()
