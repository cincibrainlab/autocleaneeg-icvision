#!/usr/bin/env python3
"""Re-render each classified component with the model's reasoning drawn onto the image.

Reads a run's results CSV (predicted_label, confidence, reason) and produces one
annotated PNG per component: the standard 4-panel render (topography, time series,
ERP image, spectrum) plus a banner (truth vs prediction) and the full reasoning text.
"""
import argparse
import csv
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

mne.set_log_level("ERROR")
from icvision.plotting import plot_single_component_subplot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="e.g. experiments/results/stage0_true")
    parser.add_argument("--model", required=True, help="model id as used in the results CSV filename")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="default: <run-dir>/annotated/<model>/")
    args = parser.parse_args()

    csv_path = args.run_dir / f"{args.run_dir.name}_{args.model}.csv"
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    by_file = defaultdict(list)
    for row in rows:
        by_file[row["set_path"]].append(row)

    out_dir = args.out or (args.run_dir / "annotated" / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    for set_path in sorted(by_file):
        full = args.base_dir / set_path
        raw = mne.io.read_raw_eeglab(str(full), preload=True)
        ica = mne.preprocessing.read_ica_eeglab(str(full))
        for row in sorted(by_file[set_path], key=lambda r: int(r["component_index"])):
            idx = int(row["component_index"])
            truth, pred = row["true_label_norm"], row["predicted_label"]
            conf = float(row["confidence"]) if row["confidence"] else 0.0
            ok = truth == pred
            banner_color = "#0a7d32" if ok else "#b3261e"
            banner = f"IC{idx}   TRUTH: {truth}   PREDICTED: {pred} ({conf:.2f})   {'CORRECT' if ok else 'WRONG'}"

            fig = plt.figure(figsize=(16, 3.2), dpi=150)
            gs = fig.add_gridspec(2, 4, height_ratios=[3.0, 1.05], hspace=0.12, wspace=0.08, left=0.03, right=0.99, top=0.86, bottom=0.02)
            axes = {"topo": fig.add_subplot(gs[0, 0]), "ts": fig.add_subplot(gs[0, 1]), "erp": fig.add_subplot(gs[0, 2]), "psd": fig.add_subplot(gs[0, 3])}
            plot_single_component_subplot(ica, raw, idx, axes, f"IC{idx}", precomputed_sources=None)
            fig.text(0.5, 0.935, banner, ha="center", va="center", fontsize=15, fontweight="bold", color="white", bbox={"boxstyle": "round,pad=0.45", "facecolor": banner_color, "edgecolor": "none"})
            text_ax = fig.add_subplot(gs[1, :])
            text_ax.axis("off")
            reason = row["reason"].strip() or "(no reason returned)"
            wrapped = "\n".join(textwrap.wrap(reason, width=150))
            text_ax.text(0.0, 0.92, "MODEL REASONING:", fontsize=10, fontweight="bold", va="top", transform=text_ax.transAxes, color="#333333")
            text_ax.text(0.0, 0.62, wrapped, fontsize=9.5, va="top", transform=text_ax.transAxes, color="#111111", wrap=True)

            fname = f"IC{idx:03d}_{truth}_pred-{pred}_{'OK' if ok else 'WRONG'}.png"
            fig.savefig(out_dir / fname, format="png")
            plt.close(fig)
            print(f"  {fname}")
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
