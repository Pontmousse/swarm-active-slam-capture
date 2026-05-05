# SwarmActiveCapture — CLI cheat sheet

Run commands from the **repository root** (`SwarmActiveCapture/`) unless noted. Use `python3` (or `python` on your system).

---

## 2D prototype (`simplified_2d/prototype2d/`)

Default experiment layout (from `[simplified_2d/prototype2d/config.json](simplified_2d/prototype2d/config.json)`):


| Role                                   | Path                                                                                                      |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Experiment config                      | `simplified_2d/prototype2d/config.json`                                                                   |
| Target geometry (simulator input)      | `simplified_2d/prototype2d/sketch.json` (field `target_json_path`; swap for e.g. `rectangle_target.json`) |
| Results root                           | `simplified_2d/prototype2d/results/`                                                                      |
| Run folder for config `"name": "v1.0"` | `simplified_2d/prototype2d/results/v1.0/`                                                                 |


After a full simulation, a run directory typically contains pickles such as `agents_history.pkl`, `target_history.pkl`, `attachment_points.pkl`, `metrics_history.pkl`, `messages_history.pkl`, plus `performance.json`.

---

### Run the 2D simulator

The module entry point always loads `**config.json` next to `simulator.py`** (same directory as the default config path in code):

```bash
cd /path/to/SwarmActiveCapture
python3 -m simplified_2d.prototype2d.simulator
```

To use another JSON config **without editing the file**, call Python once:

```bash
python3 -c "from simplified_2d.prototype2d.simulator import run_simulation; run_simulation('simplified_2d/prototype2d/config.json')"
```

```bash
python3 -c "from simplified_2d.prototype2d.simulator import run_simulation; run_simulation('path/to/my_experiment.json')"
```

Edit in `[config.json](simplified_2d/prototype2d/config.json)`: `target_json_path`, `output_root`, `name`, durations, agents, delays, etc.

---

### Animate a saved run

Requires at least `agents_history.pkl` and `attachment_points.pkl` under `--results`. Optional `--target` improves dense/hull drawing; `--highlight-agent` marks one agent (1-based index) and its landmark map.

```bash
python3 -m simplified_2d.prototype2d.animation \
  --results simplified_2d/prototype2d/results/v1.0 \
  --target simplified_2d/prototype2d/sketch.json
```

```bash
python3 -m simplified_2d.prototype2d.animation \
  --results simplified_2d/prototype2d/results/v1.0 \
  --target simplified_2d/prototype2d/rectangle_target.json \
  --interval 80 \
  --margin 0.15
```

```bash
python3 -m simplified_2d.prototype2d.animation \
  --results simplified_2d/prototype2d/results/v1.0 \
  --target simplified_2d/prototype2d/sketch.json \
  --highlight-agent 1
```


| Option                | Meaning                                                                        |
| --------------------- | ------------------------------------------------------------------------------ |
| `--results`           | **Required.** Directory containing `*_history.pkl` files.                      |
| `--target`            | Optional target JSON (matches `target_json_path` used for the run).            |
| `--interval`          | Frame interval in ms (default `100`).                                          |
| `--margin`            | Extra axis padding as a fraction of span (default `0.12`).                     |
| `--highlight-agent N` | `N ≥ 1`: highlight *N*th agent (square) + darker cyan map overlay; omit = off. |


---

### Plot metrics from a run

```bash
python3 -m simplified_2d.prototype2d.plotting \
  --results simplified_2d/prototype2d/results/v1.0 \
  --metric convex_hull_area
```

```bash
python3 -m simplified_2d.prototype2d.plotting \
  --results simplified_2d/prototype2d/results/v1.0 \
  --metric map_coverage_ratio
```

```bash
python3 -m simplified_2d.prototype2d.plotting \
  --results simplified_2d/prototype2d/results/v1.0 \
  --metric mode_counts
```

Compare the same metric across several result directories:

```bash
python3 -m simplified_2d.prototype2d.plotting --compare convex_hull_area \
  simplified_2d/prototype2d/results/v1.0 \
  simplified_2d/prototype2d/results/other_run
```

`**--metric` keys** (from saved `metrics_history.pkl` rows):  
`time`, `convex_hull_area`, `min_distance`, `mean_distance`, `max_distance`, `control_effort`, `fuel_consumed_total`, `message_age_mean`, `message_age_max`, `map_size_mean`, `map_coverage_ratio`, `ap_conflicts`, `ap_coverage_ratio`, `capture_error_mean`.  
Special: `--metric mode_counts` plots mode population curves instead of a single scalar.

---

### Tkinter sketch tool (freehand → `dense_points` + attachments)

```bash
python3 -m simplified_2d.prototype2d.target_sketch_tk \
  -o simplified_2d/prototype2d/sketch.json \
  --attachment-edge-stride 10
```

```bash
python3 -m simplified_2d.prototype2d.target_sketch_tk \
  -o simplified_2d/prototype2d/sketch.json \
  --load simplified_2d/prototype2d/rectangle_target.json
```

```bash
python3 -m simplified_2d.prototype2d.target_sketch_tk \
  -o simplified_2d/prototype2d/sketch.json \
  --lines-draft simplified_2d/prototype2d/my_sketch_lines.json \
  --dense-spacing 0.0 \
  --simplify-epsilon 0.0 \
  --export-line -1 \
  --attachment-edge-stride 10 \
  --xmin -1 --xmax 1 --ymin -1 --ymax 1 \
  --canvas-width 720 --canvas-height 720 \
  --min-step 0.02 --snap-tol 0.08
```


| Option                               | Meaning                                                                                  |
| ------------------------------------ | ---------------------------------------------------------------------------------------- |
| `-o` / `--output`                    | **Required.** Simulator target JSON path (Export button).                                |
| `--load`                             | Lines draft (`*_lines.json`) or existing target JSON.                                    |
| `--lines-draft`                      | Where “Save lines draft” writes (default `<output_stem>_lines.json`).                    |
| `--dense-spacing`                    | Extra thinning between exported landmarks along each stroke (`0` = keep sketch samples). |
| `--simplify-epsilon`                 | RDP simplification per line (`0` = off).                                                 |
| `--export-line N`                    | Export only line index `N`; `-1` = all lines.                                            |
| `--attachment-edge-stride K`         | Attachment every *K* polyline edges (`1` = densest).                                     |
| `--xmin` … `--ymax`                  | World bounds mapped to the canvas.                                                       |
| `--canvas-width` / `--canvas-height` | Initial canvas size.                                                                     |
| `--min-step`                         | Minimum world distance between samples while dragging.                                   |
| `--snap-tol`                         | Release within this distance of start → closed stroke.                                   |


---

## Other runnable modules (repo root)

These are separate stacks (3D / hardware); flags differ per file.

```bash
python3 run_active_slam.py
```

```bash
python3 SwarmCapture+/Swarm_Target_Capture+.py
```

```bash
python3 SwarmCapture+/Load_Target.py
```

```bash
python3 DDFGO++/SwarmDDFGO++.py
```

```bash
python3 DDFGO++/Recording.py
```

```bash
python3 simplified_2d/simplified_swarm.py
```

*(The last one expects RoboMaster / `swarm_control` dependencies per that script.)*

---

## Quick copy-paste: default 2D loop

```bash
cd /path/to/SwarmActiveCapture

# 1) (Optional) Edit targets or config
#    simplified_2d/prototype2d/config.json
#    simplified_2d/prototype2d/sketch.json

python3 -m simplified_2d.prototype2d.simulator

python3 -m simplified_2d.prototype2d.animation \
  --results simplified_2d/prototype2d/results/v1.0 \
  --target simplified_2d/prototype2d/sketch.json

python3 -m simplified_2d.prototype2d.plotting \
  --results simplified_2d/prototype2d/results/v1.0 \
  --metric convex_hull_area
```

Replace `/path/to/SwarmActiveCapture` with your clone path.