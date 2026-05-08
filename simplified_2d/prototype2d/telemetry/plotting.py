"""
Plot telemetry from saved prototype2d runs.

Config-driven:
  python -m simplified_2d.prototype2d.telemetry --config path/to/telemetry.json

Legacy (backward compatible):
  python -m simplified_2d.prototype2d.plotting --results DIR --metric convex_hull_area
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from .telemetry import (
    ResultsBundle,
    agent_series_for_ids,
    apply_time_range,
    load_results_bundle,
    metric_series,
)

JsonDict = Dict[str, Any]


def _want_interactive_window(cfg_or_args: Mapping[str, Any], save_dir_here: Optional[str]) -> bool:
    """
    Decide whether to call plt.show() after drawing.
    - If config sets interactive/show to a boolean, use it.
    - Otherwise: show window when not saving this figure anywhere (save_dir_here is None).
    """
    for key in ("interactive", "show"):
        if key not in cfg_or_args:
            continue
        val = cfg_or_args[key]
        if val is None:
            continue
        return bool(val)
    return save_dir_here is None


def _finalize_figure(
    save_dir: Optional[str],
    filename_stem: str,
    *,
    interactive: bool,
) -> None:
    """
    Optionally write PNG then optionally open an interactive matplotlib window.
    Use --interactive / --no-interactive or telemetry.json interactive/show keys.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.abspath(os.path.join(save_dir, f"{filename_stem}.png"))
        plt.savefig(out, dpi=140, bbox_inches="tight")
        print(f"[telemetry] saved PNG → {out}", flush=True)

    if interactive:
        plt.show(block=True)
    plt.close()


def _resolve_path(base_dir: Optional[str], p: Optional[str]) -> Optional[str]:
    if p is None or p == "":
        return None
    if os.path.isabs(p):
        return os.path.abspath(p)
    if base_dir:
        return os.path.abspath(os.path.join(base_dir, p))
    return os.path.abspath(p)


def _print_highlight_agent_summary(results_dir: str, highlight_one_based: int) -> None:
    if highlight_one_based < 1:
        print("[telemetry] warning: --highlight-agent must be >= 1; ignoring.")
        return
    perf_path = os.path.join(results_dir, "performance.json")
    if not os.path.isfile(perf_path):
        print(
            "[telemetry] no performance.json in results dir "
            "(per-agent dwell is recorded there after simulation).",
        )
        return
    with open(perf_path, encoding="utf-8") as handle:
        perf = json.load(handle)
    aid = str(highlight_one_based - 1)
    modes = (perf.get("time_in_mode_per_agent_sec") or {}).get(aid)
    behaves = (perf.get("time_in_behavior_per_agent_sec") or {}).get(aid)
    print(
        f"[telemetry] agent list index {highlight_one_based} "
        f"(simulator id {aid}); config panels can plot per-agent series.",
    )
    if modes:
        print(f"[telemetry]   time in mode (s): {modes}")
    if behaves:
        print(f"[telemetry]   time in behavior (s): {behaves}")
    if not modes and not behaves:
        print(f"[telemetry]   no dwell entry for simulator id {aid}")


def _load_metrics_legacy(results_dir: str) -> List[JsonDict]:
    path = os.path.join(results_dir, "metrics_history.pkl")
    import pickle

    with open(path, "rb") as handle:
        return pickle.load(handle)


def plot_metric_rows(
    times: Sequence[float],
    values: Sequence[float],
    *,
    title: str,
    y_label: str,
    label: str,
    x_label: str = "time (s)",
) -> None:
    lbl = label
    plt.plot(times, values, label=lbl)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)


def plot_metric(results_dir: str, metric_key: str, title: str, label: Optional[str]) -> None:
    metrics = _load_metrics_legacy(results_dir)
    times = [float(row["time"]) for row in metrics]
    values = metric_series(metrics, metric_key)
    lbl = label or os.path.basename(results_dir.rstrip(os.sep))
    plot_metric_rows(times, values, title=title, y_label=metric_key, label=lbl)


def plot_mode_counts(results_dir: str) -> None:
    metrics = _load_metrics_legacy(results_dir)
    times = [float(row["time"]) for row in metrics]
    modes = {"s": [], "e": [], "c": [], "d": [], "p": []}
    for row in metrics:
        counts = row.get("mode_counts") or {}
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
    metrics = _load_metrics_legacy(results_dir)
    times = [float(row["time"]) for row in metrics]
    keys_set = set()
    for row in metrics:
        bc = row.get("behavior_counts") or {}
        keys_set.update(bc.keys())
    keys = sorted(keys_set)
    if not keys:
        plt.text(0.5, 0.5, "no behavior_counts recorded", transform=plt.gca().transAxes, ha="center")
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
    for results_dir in results_dirs:
        plot_metric(results_dir, metric_key, metric_key, None)


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
    means: List[float] = []
    stderr: List[float] = []
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


def _expand_series(bundle: ResultsBundle, spec: JsonDict) -> List[Tuple[str, List[float]]]:
    """Return one or more (label, y) aligned with bundle.metrics length."""
    src = str(spec.get("source", "metrics")).lower()
    lab = spec.get("label")
    if src == "metrics":
        key = str(spec["key"])
        y = metric_series(bundle.metrics, key)
        lbl = str(lab) if lab else key
        return [(lbl, y)]

    if src == "agents":
        if not bundle.agents_history:
            raise ValueError("agents series requested but agents_history.pkl missing or empty")
        field = str(spec["field"])
        raw_ids = spec.get("agents", "all")
        if raw_ids == "all":
            ids = sorted({int(a["id"]) for a in bundle.agents_history[0]})
        elif isinstance(raw_ids, list):
            ids = [int(x) for x in raw_ids]
        else:
            ids = [int(raw_ids)]
        if not ids:
            raise ValueError("agents series requires a non-empty agent id list")

        overrides = spec.get("labels") or {}
        if not isinstance(overrides, Mapping):
            overrides = {}

        agg = str(spec.get("aggregate", "")).lower()
        agent_map = agent_series_for_ids(bundle.agents_history, ids, field)
        ys_list = [agent_map[a] for a in ids]

        if agg == "mean":
            n = len(bundle.metrics)
            arr: List[float] = []
            for i in range(n):
                vals = [ys[i] for ys in ys_list if i < len(ys)]
                vv = [v for v in vals if v == v]
                arr.append(sum(vv) / len(vv) if vv else float("nan"))
            lbl = str(lab) if lab else f"{field} mean(ids={ids})"
            return [(lbl, arr)]

        out: List[Tuple[str, List[float]]] = []
        default_lbl = str(lab) if lab else ""
        for aid in ids:
            lbl_raw = overrides.get(str(aid))
            curve_lbl = str(lbl_raw) if lbl_raw else (default_lbl if default_lbl else f"a{aid} {field}")
            out.append((curve_lbl, agent_map.get(aid, [float("nan")] * len(bundle.metrics))))
        return out

    raise ValueError(f"unknown series source: {spec!r}")


def _align_bundle_timelines(bundle: ResultsBundle) -> ResultsBundle:
    """Trim metrics vs agents_history to matching length."""
    nm = len(bundle.metrics)
    na = len(bundle.agents_history)
    if not bundle.agents_history:
        return bundle
    if nm == na:
        return bundle
    n_min = min(nm, na)
    print(
        f"[telemetry] warning: trimming metrics ({nm}) vs agents_history ({na}) to length {n_min}",
        flush=True,
    )
    return ResultsBundle(
        results_dir=bundle.results_dir,
        metrics=bundle.metrics[:n_min],
        agents_history=bundle.agents_history[:n_min],
        dt_hint=bundle.dt_hint,
    )


def _panels_from_quick_args(
    metrics_key: Optional[str],
    agents: Optional[str],
    agent_field: Optional[str],
) -> List[JsonDict]:
    if metrics_key:
        return [
            {
                "title": metrics_key,
                "series": [{"source": "metrics", "key": metrics_key, "label": metrics_key}],
            }
        ]
    if agents is not None and agent_field:
        raw = [int(x.strip()) for x in agents.split(",") if x.strip()]
        return [
            {
                "title": agent_field,
                "series": [
                    {
                        "source": "agents",
                        "field": agent_field,
                        "agents": raw,
                        "labels": {},
                    }
                ]
            }
        ]
    return []


def run_config(cfg: JsonDict, config_path_for_resolve: Optional[str]) -> None:
    base = os.path.dirname(config_path_for_resolve) if config_path_for_resolve else None
    raw_results = cfg.get("results_dir")
    if not raw_results:
        raise SystemExit("telemetry.json missing results_dir")

    rd = _resolve_path(base, str(raw_results))
    if rd is None:
        raise SystemExit("results_dir could not be resolved")
    bundle = load_results_bundle(rd)
    bundle = _align_bundle_timelines(bundle)

    raw_trange = cfg.get("time_range")
    time_range: Optional[Tuple[float, float]]
    if (
        isinstance(raw_trange, list)
        and len(raw_trange) == 2
        and all(isinstance(x, (int, float)) for x in raw_trange)
    ):
        time_range = (float(raw_trange[0]), float(raw_trange[1]))
    else:
        time_range = None

    save_dir_cfg = cfg.get("save_dir")
    save_root = _resolve_path(base, save_dir_cfg) if save_dir_cfg else None

    legacy_panels_cfg = cfg.get("panels_legacy")
    if legacy_panels_cfg is not None:
        figures_spec: Sequence[Dict[str, Any]] = [
            {
                "title": cfg.get("title", "telemetry"),
                "figsize": cfg.get("figsize", [9, 4]),
                "filename": cfg.get("filename", "telemetry"),
                "panels": legacy_panels_cfg,
            }
        ]
    else:
        figures_raw = cfg.get("figures")
        if isinstance(figures_raw, list):
            figures_spec = figures_raw
        else:
            mq = cfg.get("metrics_quick")
            aq = cfg.get("agents_quick")
            af = cfg.get("agents_field_quick")
            quick_panels = _panels_from_quick_args(
                str(mq) if mq else None,
                str(aq) if aq else None,
                str(af) if af else None,
            )
            if quick_panels:
                figures_spec = [
                    {
                        "title": "quick plot",
                        "figsize": [9, 4],
                        "filename": cfg.get("filename", "telemetry_quick"),
                        "panels": quick_panels,
                    }
                ]
            else:
                raise SystemExit("telemetry.json must contain 'figures' or quick keys.")

    highlight = cfg.get("highlight_agent_one_based")
    if highlight is not None:
        _print_highlight_agent_summary(bundle.results_dir, int(highlight))

    times_full = bundle.times

    for fig_ix, fig_spec in enumerate(figures_spec):
        title_main = fig_spec.get("title", f"figure_{fig_ix}")
        figsize = fig_spec.get("figsize", [10, 4])
        if isinstance(figsize, list) and len(figsize) == 2:
            w, h = float(figsize[0]), float(figsize[1])
        else:
            w, h = 10.0, 4.0
        filename_stem = str(fig_spec.get("filename") or fig_spec.get("id") or f"fig{fig_ix}")

        panels = fig_spec.get("panels") or []
        if not panels:
            continue

        n = len(panels)
        layout_local = fig_spec.get("subplot_layout", cfg.get("subplot_layout", "vertical"))

        save_dir_here = fig_spec.get("save_dir")
        save_dir_eff = (
            _resolve_path(base, str(save_dir_here)) if save_dir_here else save_root
        )

        if layout_local == "horizontal" and n > 1:
            fig_shape = (w * float(n), h)
            figure, axes = plt.subplots(
                1,
                n,
                figsize=fig_shape,
                squeeze=False,
            )
            axes_flat = list(axes[0])
        else:
            fig_shape = (w, h * float(n))
            figure, axes = plt.subplots(n, 1, figsize=fig_shape, squeeze=False)
            axes_flat = list(axes[:, 0])

        for pix, panel in enumerate(panels):
            ax = axes_flat[pix]
            panel_title = panel.get("title", f"panel{pix}")

            raw_series_list = panel.get("series") or []
            legends: List[str] = []
            entries: List[List[float]] = []
            times_for_panel = bundle.times

            for s in raw_series_list:
                for lbl, ys in _expand_series(bundle, s):
                    legends.append(str(lbl))
                    entries.append(ys)

            if not entries:
                continue

            t_arr, y_arrs = apply_time_range(times_for_panel, entries, time_range)
            plot_kwargs = dict(panel.get("plot") or {})
            plot_kwargs.pop("label", None)

            for yi, ys in enumerate(y_arrs):
                ax.plot(t_arr, ys, label=legends[yi], **plot_kwargs)

            if n > 1:
                ax.set_title(panel_title)
            else:
                ax.set_title(f"{title_main} — {panel_title}")
            ax.set_xlabel(str(panel.get("x_label", "time (s)")))
            ax.set_ylabel(str(panel.get("y_label", panel_title)))
            ylim = panel.get("ylim")
            if isinstance(ylim, list) and len(ylim) == 2:
                ax.set_ylim(float(ylim[0]), float(ylim[1]))
            ax.grid(True, alpha=0.3)

            legend_flag = panel.get("legend")
            if legend_flag if legend_flag is not None else True:
                ax.legend(fontsize=8)

        if n > 1:
            figure.suptitle(title_main, y=1.02)
        plt.tight_layout()
        stem_out = cfg.get("output_per_figure_suffix")
        fname = f"{filename_stem}_{stem_out}" if stem_out else filename_stem
        _finalize_figure(
            save_dir_eff,
            fname,
            interactive=_want_interactive_window(cfg, save_dir_eff),
        )


def _load_json_config(path: str) -> JsonDict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run_argparse_legacy(args: argparse.Namespace) -> None:
    def _finalize_inter(args_ns: argparse.Namespace, stem: str) -> None:
        save_here = (
            os.path.abspath(args_ns.save_dir) if getattr(args_ns, "save_dir", None) else None
        )
        if args_ns.interactive:
            iw = True
        elif getattr(args_ns, "no_interactive", False):
            iw = False
        else:
            iw = save_here is None
        _finalize_figure(save_here, stem, interactive=iw)

    if args.batch_summary:
        plot_batch_summary_bars(args.batch_summary, args.value_key, args.group_key)
        stem = "batch_" + args.value_key.replace(".", "_")
        _finalize_inter(args, stem)
        return

    if args.compare:
        if not args.dirs:
            raise SystemExit("Provide result directories after --compare.")
        plt.figure(figsize=(8, 4))
        compare_metric(args.metric, args.dirs)
        plt.tight_layout()
        _finalize_inter(args, f"compare_{args.metric}")
    else:
        if not args.results:
            raise SystemExit("Provide --results, --batch-summary, or --config.")
        if args.highlight_agent is not None:
            _print_highlight_agent_summary(args.results, args.highlight_agent)
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
        _finalize_inter(args, stem)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot prototype telemetry (config JSON or legacy flags)."
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to telemetry.json (figures, agents, time_range, save_dir)",
    )
    parser.add_argument("--results", help="Legacy: results directory")
    parser.add_argument(
        "--highlight-agent",
        type=int,
        default=None,
        metavar="N",
        help="Print dwell stats from performance.json for agent index N (1-based).",
    )
    parser.add_argument("--metric", default="convex_hull_area")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Show matplotlib window (can combine with --save-dir)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Do not pop up a matplotlib window",
    )
    parser.add_argument("--batch-summary", default=None)
    parser.add_argument("--group-key", default="decision_backend")
    parser.add_argument("--value-key", default="integrated_control_effort")
    parser.add_argument(
        "--metrics-key",
        default=None,
        help="Shortcut: plot one swarm metric (requires --results).",
    )
    parser.add_argument(
        "--agents",
        default=None,
        help="Comma agent ids with --agents-field (requires --results).",
    )
    parser.add_argument(
        "--agents-field",
        default=None,
        dest="agents_field",
        help="Per-agent scalar field or alias (map_frontier_length, map_size, fuel_consumed, ...)",
    )
    parser.add_argument("dirs", nargs="*", help="Result dirs after --compare")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.interactive and args.no_interactive:
        parser.error("Use only one of --interactive/-i or --no-interactive.")

    if args.config:
        cfg = _load_json_config(os.path.abspath(args.config))
        run_config(cfg, os.path.abspath(args.config))
        return

    if (
        args.results
        and (args.metrics_key or (args.agents and args.agents_field))
        and not args.compare
        and not args.batch_summary
    ):
        quick_cfg = {
            "results_dir": args.results,
            "save_dir": args.save_dir,
            "figures": [
                {
                    "title": "CLI quick plots",
                    "figsize": [9, 4],
                    "filename": "telemetry_cli_quick",
                    "panels": [
                        *_panels_from_quick_args(args.metrics_key, None, None),
                        *_panels_from_quick_args(
                            None,
                            args.agents,
                            args.agents_field if args.agents else None,
                        ),
                    ],
                }
            ],
        }
        if args.highlight_agent is not None:
            quick_cfg["highlight_agent_one_based"] = args.highlight_agent
        if args.interactive:
            quick_cfg["interactive"] = True
        elif args.no_interactive:
            quick_cfg["interactive"] = False

        run_config(quick_cfg, None)
        return

    run_argparse_legacy(args)


if __name__ == "__main__":
    main()
