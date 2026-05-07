"""Batch Phase 4 experiments over a Cartesian product matrix (YAML/JSON scenarios)."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from ..io import save_json
from ..simulator import run_simulation


def _repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "simplified_2d").is_dir():
            return parent
    return start.resolve()


def _resolve_path(repo: Path, maybe_rel: str) -> Path:
    path = Path(maybe_rel)
    if path.is_absolute():
        return path
    return (repo / path).resolve()


def _iter_cartesian(matrix: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    keys = sorted(matrix.keys())
    values = [matrix[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _flatten_perf(perf_path: Path) -> Dict[str, Any]:
    with open(perf_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    out: Dict[str, Any] = {}
    skip = {"time_in_mode_per_agent_sec", "time_in_behavior_per_agent_sec"}
    for k, v in data.items():
        if k in skip:
            continue
        out[k if k != "experiment" else "run_experiment"] = v
    return out


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def run_batch_from_scenario(scenario_path: str) -> Tuple[Path, List[Dict[str, Any]]]:
    repo = _repo_root(Path(scenario_path).parent)
    scen = Path(scenario_path).resolve()
    with open(scen, "r", encoding="utf-8") as handle:
        spec = json.load(handle)

    base_cfg_path = _resolve_path(repo, spec["base_config"])
    batch_out = _resolve_path(repo, spec.get("batch_output_dir", "results/phase4_batch"))
    matrix = spec["matrix"]

    batch_out.mkdir(parents=True, exist_ok=True)
    with open(base_cfg_path, "r", encoding="utf-8") as handle:
        base = json.load(handle)

    manifest_runs: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for overlay in _iter_cartesian(matrix):
        merged = dict(base)
        merged.update(overlay)
        batch_slug = "__".join(f"{k}_{overlay[k]}" for k in sorted(overlay.keys()))
        batch_name = spec.get("batch_name", batch_out.name)
        merged["name"] = f"{batch_name}_{batch_slug}"

        merged_root = overlay.get("output_root")
        if merged_root is None:
            merged["output_root"] = str(batch_out)
        else:
            merged["output_root"] = str(_resolve_path(repo, str(merged_root)))

        cfg_fd, cfg_path_str = tempfile.mkstemp(prefix="proto2d_cfg_", suffix=".json")
        os.close(cfg_fd)
        cfg_path = Path(cfg_path_str)
        try:
            save_json(str(cfg_path), merged)
            results_dir = run_simulation(str(cfg_path))
        finally:
            if cfg_path.exists():
                cfg_path.unlink()

        perf_path = Path(results_dir) / "performance.json"
        row = dict(overlay)
        row.update(_flatten_perf(perf_path))
        row["results_dir"] = results_dir
        manifest_runs.append({"overlay": overlay, "results_dir": results_dir})
        rows.append(row)

    summary_csv = batch_out / "summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())

        def _sort_key(fn: str) -> Tuple[int, str]:
            priority = {"decision_backend": -2, "rng_seed": -1}
            return (priority.get(fn, 0), fn)

        fieldnames = sorted(fieldnames, key=_sort_key)
        with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow(row)

    aggregates: Dict[str, Any] = {}

    stratify_key = None
    for cand in sorted(matrix.keys()):
        if cand != "rng_seed":
            stratify_key = cand
            break

    numeric_metrics = [
        "time_to_all_docked",
        "integrated_control_effort",
        "total_fuel_consumed",
        "final_map_coverage_ratio",
        "mean_decision_latency_sec",
        "llm_calls_total",
    ]

    if stratify_key is not None and "rng_seed" in matrix:
        for metric in numeric_metrics:
            bucket: Dict[str, List[float]] = {}
            for row in rows:
                pol = str(row.get(stratify_key, "unknown"))
                raw = row.get(metric)
                if raw is None or raw == "":
                    continue
                if metric == "time_to_all_docked" and raw is None:
                    continue
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    continue
                bucket.setdefault(pol, []).append(val)

            aggregates[metric] = {
                policy: {"mean": _mean_std(vals)[0], "std": _mean_std(vals)[1]}
                for policy, vals in sorted(bucket.items())
                if vals
            }

        policies = sorted({str(row.get(stratify_key)) for row in rows})
        dock_frac: Dict[str, float] = {}
        for policy in policies:
            n = sum(1 for row in rows if str(row.get(stratify_key)) == policy)
            docks = sum(
                1
                for row in rows
                if str(row.get(stratify_key)) == policy
                and row.get("time_to_all_docked") is not None
                and row.get("time_to_all_docked") != ""
                and row.get("time_to_all_docked") != "None"
            )
            dock_frac[str(policy)] = float(docks / max(n, 1))

        aggregates["fraction_all_docked"] = dock_frac
        aggregates["stratify_key"] = stratify_key

    summary_json = batch_out / "summary.json"
    summary_payload = {
        "scenario": str(scen),
        "base_config": str(base_cfg_path),
        "runs": rows,
        "aggregates_across_rng_seed": aggregates,
    }
    save_json(str(summary_json), summary_payload)

    manifest_path = batch_out / "batch_manifest.json"
    save_json(
        str(manifest_path),
        {
            "scenario": str(scen),
            "runs": manifest_runs,
        },
    )

    return batch_out, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 batch evaluations from scenario JSON.")
    parser.add_argument("--scenario", required=True, help="path to scenarios/*.json manifest")
    args = parser.parse_args()
    batch_dir, rows = run_batch_from_scenario(args.scenario)
    print(f"[batch] wrote {len(rows)} rows under {batch_dir}", flush=True)


if __name__ == "__main__":
    main()
