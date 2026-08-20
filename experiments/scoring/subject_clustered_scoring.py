#!/usr/bin/env python3
"""Standard scoring for icvision accuracy results: pooled AND subject-clustered.

Background (see plan/plan-log.md, 2026-08-19/20 entries): every accuracy
number this project has measured against Grace's labeled dataset draws from
only ~12 independent recordings, no matter how many components are pooled
together (679, or any subset of it). A naive per-component accuracy and
confidence interval treats those components as independent draws, which
understates uncertainty and can make two configurations look clearly
different when they aren't (or look settled-worse-than-ICLabel when the gap
is actually within noise). This was found retroactively, by hand, for one
pair of results -- this script makes computing it the default going forward,
for every future run, not a manual afterthought.

Usage:
    python subject_clustered_scoring.py results.csv
    python subject_clustered_scoring.py results.csv --pred-col strip_predicted_label
    python subject_clustered_scoring.py results.csv --subject-col set_path

Accepts any of this project's standard results CSV shapes:
    - true_label_norm (already normalized) or true_label (Grace's raw shorthand,
      normalized here the same way every prior scoring pass has: "other" ->
      "other_artifact", "channel" -> "channel_noise")
    - predicted_label / strip_predicted_label / single_predicted_label (pick
      via --pred-col; defaults to trying predicted_label first)
    - set_path as the clustering unit (the source .set file / subject a
      component came from -- consistent across every run script used in this
      project's baseline investigation)
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LABEL_NORMALIZATION = {"other": "other_artifact", "channel": "channel_noise"}

# t-distribution critical values for a 95% CI, indexed by degrees of freedom
# (n_subjects - 1). Covers the range this project's datasets actually produce;
# extend if a future dataset has a different subject count.
T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    20: 2.086, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}


def _t_critical(df: int) -> float:
    if df in T_CRITICAL_95:
        return T_CRITICAL_95[df]
    # fall back to nearest available df, conservatively rounding down
    available = sorted(k for k in T_CRITICAL_95 if k <= df)
    if available:
        return T_CRITICAL_95[available[-1]]
    return 1.960  # z-critical, large-sample fallback


def _normalize_label(label: str) -> str:
    return LABEL_NORMALIZATION.get(label, label)


def _pick_column(fieldnames: List[str], candidates: List[str], purpose: str) -> str:
    for c in candidates:
        if c in fieldnames:
            return c
    raise ValueError(
        f"Could not find a {purpose} column. Tried {candidates}, found columns: {fieldnames}"
    )


def score(
    csv_path: str,
    pred_col: Optional[str] = None,
    true_col: Optional[str] = None,
    subject_col: str = "set_path",
) -> Dict:
    """Score a results CSV both ways: pooled and subject-clustered.

    Returns a dict with pooled accuracy, per-subject accuracy, and a
    subject-clustered confidence interval -- see module docstring for why
    the subject-clustered numbers are the ones that should be trusted for
    any comparative claim ("does X beat Y").
    """
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    fieldnames = list(rows[0].keys())

    if true_col is None:
        true_col = _pick_column(fieldnames, ["true_label_norm", "true_label"], "true-label")
    if pred_col is None:
        pred_col = _pick_column(
            fieldnames,
            ["predicted_label", "strip_predicted_label", "single_predicted_label"],
            "predicted-label",
        )

    by_subject: Dict[str, List[int]] = defaultdict(list)  # 1=correct, 0=wrong
    for r in rows:
        true_label = _normalize_label(r[true_col])
        pred_label = r[pred_col]
        by_subject[r[subject_col]].append(1 if pred_label == true_label else 0)

    n_total = sum(len(v) for v in by_subject.values())
    n_correct = sum(sum(v) for v in by_subject.values())
    pooled_accuracy = n_correct / n_total

    per_subject_acc = {s: sum(v) / len(v) for s, v in by_subject.items()}
    subj_vals = list(per_subject_acc.values())
    n_subjects = len(subj_vals)

    result = {
        "csv_path": csv_path,
        "true_col": true_col,
        "pred_col": pred_col,
        "n_components": n_total,
        "n_subjects": n_subjects,
        "pooled_accuracy": pooled_accuracy,
        "per_subject_accuracy": per_subject_acc,
    }

    if n_subjects >= 2:
        subj_mean = statistics.mean(subj_vals)
        subj_stdev = statistics.stdev(subj_vals)
        se = subj_stdev / (n_subjects ** 0.5)
        t_crit = _t_critical(n_subjects - 1)
        ci_lo, ci_hi = subj_mean - t_crit * se, subj_mean + t_crit * se
        result.update(
            {
                "per_subject_mean": subj_mean,
                "per_subject_stdev": subj_stdev,
                "per_subject_min": min(subj_vals),
                "per_subject_max": max(subj_vals),
                "subject_clustered_se": se,
                "subject_clustered_ci_95": (ci_lo, ci_hi),
            }
        )
    else:
        result["subject_clustered_ci_95"] = None
        result["_warning"] = "Fewer than 2 subjects -- cannot compute a subject-clustered CI at all."

    return result


def print_report(result: Dict) -> None:
    print(f"=== {Path(result['csv_path']).name} ===")
    print(f"columns used: true={result['true_col']!r}, predicted={result['pred_col']!r}, subject=set_path")
    print(f"n_components={result['n_components']}, n_subjects={result['n_subjects']}")
    print()
    print(f"POOLED accuracy (naive, treats all {result['n_components']} components as independent): "
          f"{result['n_components']} rows -> {result['pooled_accuracy']:.4f}")
    print()

    if "_warning" in result:
        print(f"WARNING: {result['_warning']}")
        return

    print("Per-subject accuracy:")
    for subj, acc in sorted(result["per_subject_accuracy"].items(), key=lambda x: -x[1]):
        print(f"  {subj:40s} {acc:.3f}")
    print()
    print(f"Per-subject mean={result['per_subject_mean']:.4f}, "
          f"stdev={result['per_subject_stdev']:.4f}, "
          f"range=[{result['per_subject_min']:.3f}, {result['per_subject_max']:.3f}]")
    lo, hi = result["subject_clustered_ci_95"]
    print(f"SUBJECT-CLUSTERED 95% CI (n={result['n_subjects']}, use this for any comparative claim): "
          f"[{lo:.4f}, {hi:.4f}]")
    if result["n_subjects"] < 15:
        print(f"  (caveat: t-based CI with only {result['n_subjects']} clusters is a rough approximation, "
              "not precision -- normal-distribution assumptions get shaky at this n. Treat this as a "
              "correction from false confidence to honest uncertainty, not as a precise interval.)")


def compare(result_a: Dict, result_b: Dict, label_a: str = "A", label_b: str = "B") -> None:
    """Print whether two results' subject-clustered CIs overlap -- the honest
    way to ask "does X beat Y" given this project's subject-clustering issue."""
    print(f"\n=== Comparing {label_a} vs {label_b} ===")
    print(f"{label_a}: pooled={result_a['pooled_accuracy']:.4f}", end="")
    if result_a.get("subject_clustered_ci_95"):
        lo, hi = result_a["subject_clustered_ci_95"]
        print(f", subject-clustered 95% CI=[{lo:.4f}, {hi:.4f}]")
    else:
        print()
    print(f"{label_b}: pooled={result_b['pooled_accuracy']:.4f}", end="")
    if result_b.get("subject_clustered_ci_95"):
        lo, hi = result_b["subject_clustered_ci_95"]
        print(f", subject-clustered 95% CI=[{lo:.4f}, {hi:.4f}]")
    else:
        print()

    if result_a.get("subject_clustered_ci_95") and result_b.get("subject_clustered_ci_95"):
        a_lo, a_hi = result_a["subject_clustered_ci_95"]
        b_lo, b_hi = result_b["subject_clustered_ci_95"]
        overlap = max(a_lo, b_lo) <= min(a_hi, b_hi)
        if overlap:
            print("Intervals OVERLAP -- do not claim one config beats the other from this alone.")
        else:
            print("Intervals do NOT overlap -- a real, defensible difference (still only a rough "
                  "approximation at this subject count, but not just noise).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to a results CSV")
    parser.add_argument("--pred-col", default=None, help="Predicted-label column name (auto-detected if omitted)")
    parser.add_argument("--true-col", default=None, help="True-label column name (auto-detected if omitted)")
    parser.add_argument("--subject-col", default="set_path", help="Column identifying the subject/source file (default: set_path)")
    parser.add_argument("--compare-to", default=None, help="Optional second CSV to compare against (same column-detection rules apply)")
    args = parser.parse_args()

    result = score(args.csv_path, pred_col=args.pred_col, true_col=args.true_col, subject_col=args.subject_col)
    print_report(result)

    if args.compare_to:
        result2 = score(args.compare_to, subject_col=args.subject_col)
        print()
        print_report(result2)
        compare(result, result2, label_a=Path(args.csv_path).stem, label_b=Path(args.compare_to).stem)


if __name__ == "__main__":
    main()
