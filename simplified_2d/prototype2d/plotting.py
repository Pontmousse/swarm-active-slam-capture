"""
Plotting utilities for saved prototype runs.

Usage:
  python -m simplified_2d.prototype2d.plotting --results path/to/results --metric hull
  python -m simplified_2d.prototype2d.plotting --compare convex_hull_area r1 r2 \\
      --save-dir path/to/out
  python -m simplified_2d.prototype2d.plotting --batch-summary path/to/summary.csv \\
      --group-key decision_backend --value-key integrated_control_effort
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt


def _load_metrics(results_dir: str) -> List[Dict]:
    path = os.path.join(results_dir, "metrics_history.pkl")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _finalize_figure(save_dir: str | None, filename_stem: str) -> None:
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"{filename_stem}.png")
        plt.savefig(out, dpi=140, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_metric(
    results_dir: str,
    metric_key: str,
    title: str,
    label: str | None = None,
) -> None:
    metrics = _load_metrics(results_dir)
    times = [row["time"] for row in metrics]
    values = [row[metric_key] for row in metrics]
    lbl = label or os.path.basename(results_dir.rstrip(os.sep))
    plt.plot(times, values, label=lbl)
    plt.xlabel("time (s)")
    plt.ylabel(metric_key)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()


def plot_mode_counts(results_dir: str) -> None:
    metrics = _load_metrics(results_dir)
    times = [row["time"] for row in metrics]
    modes = {"s": [], "e": [], "c": [], "d": [], "p": []}
    for row in metrics:
        counts = row.get("mode_counts", {})
        for key in modes:
            modes[key].append(counts.get(key, 0))
    for key, values in modes.items():
        if any(values):
            plt.plot(times, values, label=f"mode {key}")
    plt.xlabel("time (s)")
    plt.ylabel("count")
    plt.title("Mode counts")
    plt.grid(True, alpha=0.3)
    plt.legend()


def plot_behavior_stacked(results_dir: str) -> None:
    metrics = _load_metrics(results_dir)
    times = [row["time"] for row in metrics]
    keys_set = set()
    for row in metrics:
        bc = row.get("behavior_counts") or {}
        keys_set.update(bc.keys())
    keys = sorted(keys_set)
    if not keys:
        plt.text(
            0.5,
            0.5,
            "no behavior_counts recorded",
            transform=plt.gca().transAxes,
            ha="center",
        )
        return

    stacks: Dict[str, List[float]] = {k: [] for k in keys}
    for row in metrics:
        bc = row.get("behavior_counts") or {}
        for k in keys:
            stacks[k].append(float(bc.get(k, 0)))

    plt.stackplot(times, *[stacks[k] for k in keys], labels=[f"beh:{k}" for k in keys], alpha=0.85)
    plt.xlabel("time (s)")
    plt.ylabel("agent count")
    plt.title(f"behavior_counts (stacked) — {os.path.basename(results_dir)}")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper left", ncol=2, fontsize=8)


def compare_metric(metric_key: str, results_dirs: List[str]) -> None:
    plt.figure(figsize=(8, 4))
    for results_dir in results_dirs:
        plot_metric(results_dir, metric_key, metric_key)
    plt.tight_layout()


def plot_batch_summary_bars(
    summary_csv_path: str,
    value_key: str,
    group_key: str,
) -> None:
    with open(summary_csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    groups: Dict[str, List[float]] = {}
    for row in rows:
        try:
            v = float(row[value_key])
        except (KeyError, TypeError, ValueError):
            continue
        g = str(row.get(group_key, "unknown"))
        groups.setdefault(g, []).append(v)

    ordered = sorted(groups.keys())
    means = []
    stderr = []
    for g in ordered:
        vals = groups[g]
        n = len(vals)
        mu = sum(vals) / max(n, 1)
        if n > 1:
            variance = sum((x - mu) ** 2 for x in vals) / (n - 1)
            import math

            s = math.sqrt(variance)
            se = s / math.sqrt(n)
        else:
            se = 0.0
        means.append(mu)
        stderr.append(se)

    x = range(len(ordered))
    plt.figure(figsize=(7, 4))
    plt.bar(list(x), means, yerr=stderr, capsize=4)
    plt.xticks(list(x), ordered, rotation=20, ha="right")
    plt.ylabel(value_key)
    plt.title(os.path.basename(summary_csv_path))
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot prototype runs or batch summaries")
    parser.add_argument("--results", help="Results directory to plot")
    parser.add_argument("--metric", default="convex_hull_area", help="Metric key")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--save-dir", default=None, help="If set, write PNGs instead of plt.show")
    parser.add_argument(
        "--batch-summary",
        default=None,
        help="CSV from evaluation batch (mean±stderr bar chart per group)",
    )
    parser.add_argument(
        "--group-key",
        default="decision_backend",
        help="Column to group batches (CSV header)",
    )
    parser.add_argument(
        "--value-key",
        default="integrated_control_effort",
        help="Numeric column from summary.csv",
    )
    parser.add_argument("dirs", nargs="*", help="Multiple result dirs for --compare")

    args = parser.parse_args()

    if args.batch_summary:
        plot_batch_summary_bars(args.batch_summary, args.value_key, args.group_key)
        stem = "batch_" + args.value_key.replace(".", "_")
        _finalize_figure(args.save_dir, stem)
        return

    if args.compare:
        if not args.dirs:
            raise SystemExit("Provide result directories after --compare.")
        compare_metric(args.metric, args.dirs)
        _finalize_figure(args.save_dir, f"compare_{args.metric}")
    else:
        if not args.results:
            raise SystemExit("Provide --results or --batch-summary.")
        plt.figure(figsize=(8, 4))
        if args.metric == "mode_counts":
            plot_mode_counts(args.results)
            stem = "mode_counts"
        elif args.metric == "behavior_stacked":
            plot_behavior_stacked(args.results)
            stem = "behavior_stacked"
        else:
            plot_metric(args.results, args.metric, args.metric)
            stem = args.metric
        plt.tight_layout()
        _finalize_figure(args.save_dir, stem)


if __name__ == "__main__":
    main()
