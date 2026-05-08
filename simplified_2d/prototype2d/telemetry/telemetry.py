"""
Load pickles and derive time series (metrics + per-agent fields).
Comparable role to DDFGO++/Plot_Telemetry_Func.py for the 2D prototype.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def load_pickled_list(path: str) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


@dataclass
class ResultsBundle:
    results_dir: str
    metrics: List[Dict[str, Any]]
    agents_history: List[List[Dict[str, Any]]]
    dt_hint: Optional[float]

    @property
    def times(self) -> List[float]:
        return [float(row["time"]) for row in self.metrics]


def load_results_bundle(results_dir: str) -> ResultsBundle:
    results_dir = os.path.abspath(results_dir)
    mpath = os.path.join(results_dir, "metrics_history.pkl")
    apath = os.path.join(results_dir, "agents_history.pkl")
    if not os.path.isfile(mpath):
        raise FileNotFoundError(f"missing metrics file: {mpath}")
    metrics = load_pickled_list(mpath)
    agents_history = load_pickled_list(apath) if os.path.isfile(apath) else []
    dt_hint: Optional[float] = None
    if len(metrics) >= 2:
        dt_hint = float(metrics[1]["time"]) - float(metrics[0]["time"])
    return ResultsBundle(
        results_dir=results_dir,
        metrics=metrics,
        agents_history=agents_history,
        dt_hint=dt_hint,
    )


def apply_time_range(
    times: Sequence[float],
    values_list: Sequence[Sequence[float]],
    time_range: Optional[Sequence[float]],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    t = np.asarray(times, dtype=float)
    ys = [np.asarray(v, dtype=float) for v in values_list]
    if not time_range or len(time_range) != 2:
        return t, ys
    t0, t1 = float(time_range[0]), float(time_range[1])
    mask = (t >= t0) & (t <= t1)
    return t[mask], [y[mask] for y in ys]


def scalar_from_agent_snapshot(agent: Dict[str, Any], field: str) -> float:
    """Scalar from one agents_history row (list fields → length)."""
    if field == "map_frontier_length":
        mf = agent.get("map_frontier") or []
        return float(len(mf)) if isinstance(mf, list) else 0.0
    if field in ("map_size", "map_length"):
        m = agent.get("map") or {}
        return float(len(m)) if isinstance(m, dict) else 0.0
    if field == "land_set_length":
        ls = agent.get("land_set") or []
        return float(len(ls)) if isinstance(ls, list) else 0.0
    if field == "comm_set_length":
        cs = agent.get("comm_set") or []
        return float(len(cs)) if isinstance(cs, list) else 0.0

    raw = agent.get(field)
    if isinstance(raw, (list, tuple, dict)):
        return float(len(raw))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


def agent_series_for_ids(
    agents_history: List[List[Dict[str, Any]]],
    agent_ids: Iterable[int],
    field: str,
) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {int(aid): [] for aid in agent_ids}
    aid_set = set(out.keys())
    for step_agents in agents_history:
        by_id = {int(a["id"]): a for a in step_agents if "id" in a}
        for aid in aid_set:
            snap = by_id.get(aid)
            if snap is None:
                out[aid].append(float("nan"))
            else:
                out[aid].append(scalar_from_agent_snapshot(snap, field))
    return out


def metric_series(metrics: List[Dict[str, Any]], key: str) -> List[float]:
    row: List[float] = []
    for m in metrics:
        if key not in m:
            row.append(float("nan"))
            continue
        try:
            row.append(float(m[key]))
        except (TypeError, ValueError):
            row.append(float("nan"))
    return row


def discover_metric_keys(metrics: List[Dict[str, Any]], max_samples: int = 8) -> List[str]:
    if not metrics:
        return []
    keys: set[str] = set()
    for row in metrics[:max_samples]:
        for k, v in row.items():
            if isinstance(v, dict):
                continue
            try:
                float(v)
                keys.add(str(k))
            except (TypeError, ValueError):
                continue
    return sorted(keys)
