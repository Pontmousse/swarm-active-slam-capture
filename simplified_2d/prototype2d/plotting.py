"""
Plotting utilities for saved prototype runs.

Usage:
  python -m simplified_2d.prototype2d.plotting --results path/to/results --metric hull
  python -m simplified_2d.prototype2d.plotting --compare hull results/run1 results/run2
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt


def _load_metrics(results_dir: str) -> List[Dict]:
    path = os.path.join(results_dir, "metrics_history.pkl")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def plot_metric(results_dir: str, metric_key: str, title: str) -> None:
    metrics = _load_metrics(results_dir)
    times = [row["time"] for row in metrics]
    values = [row[metric_key] for row in metrics]
    plt.plot(times, values, label=os.path.basename(results_dir))
    plt.xlabel("time (s)")
    plt.ylabel(metric_key)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()


def plot_mode_counts(results_dir: str) -> None:
    metrics = _load_metrics(results_dir)
    times = [row["time"] for row in metrics]
    modes = {"s": [], "e": [], "c": [], "d": []}
    for row in metrics:
        counts = row.get("mode_counts", {})
        for key in modes:
            modes[key].append(counts.get(key, 0))
    for key, values in modes.items():
        plt.plot(times, values, label=f"mode {key}")
    plt.xlabel("time (s)")
    plt.ylabel("count")
    plt.title("Mode counts")
    plt.grid(True, alpha=0.3)
    plt.legend()


def compare_metric(metric_key: str, results_dirs: List[str]) -> None:
    for results_dir in results_dirs:
        plot_metric(results_dir, metric_key, metric_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot prototype results")
    parser.add_argument("--results", help="Results directory to plot")
    parser.add_argument("--metric", default="convex_hull_area", help="Metric key to plot")
    parser.add_argument("--compare", action="store_true", help="Compare multiple runs")
    parser.add_argument("dirs", nargs="*", help="Result directories for compare")
    args = parser.parse_args()

    if args.compare:
        if not args.dirs:
            raise SystemExit("Provide results directories for comparison.")
        compare_metric(args.metric, args.dirs)
    else:
        if not args.results:
            raise SystemExit("Provide --results to plot.")
        if args.metric == "mode_counts":
            plot_mode_counts(args.results)
        else:
            plot_metric(args.results, args.metric, args.metric)

    plt.show()


if __name__ == "__main__":
    main()

