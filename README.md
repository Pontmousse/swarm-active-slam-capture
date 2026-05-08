# SwarmActiveCapture

Run commands from the **repository root** with `python3` (adjust if your interpreter is `python`).

Paths below use this clone as `~/projects/swarm-active-slam-capture` — replace with your own.

---

## Main path: 2D prototype simulator + telemetry plot

Runs are configured in `simplified_2d/prototype2d/config.json` (`name`, `output_root`, durations, backends, …). Outputs land in `{output_root}/{name}/` (often `simplified_2d/prototype2d/results/<name>/`). After a run you get pickles such as `metrics_history.pkl` and `agents_history.pkl`.

One runnable example from the repo root (`v4.0` should match `name` in your `config.json`):

```bash
cd ~/projects/swarm-active-slam-capture

python3 -m simplified_2d.prototype2d.simulator

python3 -m simplified_2d.prototype2d.telemetry \
  --results simplified_2d/prototype2d/results/v4.0 \
  --metrics-key map_coverage_ratio \
  --interactive

python3 -m simplified_2d.prototype2d.telemetry \
  --results simplified_2d/prototype2d/results/v4.0 \
  --agents 0,1,2,3 \
  --agents-field map_size \
  --interactive

```

PNG only (prints an absolute path — use `$HOME` or `$PWD`/`$(pwd)` so you know exactly where files go):

`--save-dir "$HOME/Desktop/telemetry_out"` and `--no-interactive`. Same command with **`--interactive`** as well saves and opens a matplotlib window.

Config-driven runs: copy `simplified_2d/prototype2d/telemetry/telemetry.example.json` → edit `results_dir`, then `-c telemetry.json`. Optional **`interactive`** / legacy **`show`**: booleans override window behaviour; omit them to default to window only when that figure isn’t saving to disk.

Older entry point: `python3 -m simplified_2d.prototype2d.plotting` (same flags).

Animate a run:

```bash
python3 -m simplified_2d.prototype2d.animation \
  --results simplified_2d/prototype2d/results/v4.0 \
  --target simplified_2d/prototype2d/sketch.json
```

Draw targets / export `sketch.json`: see `python3 -m simplified_2d.prototype2d.target_sketch_tk --help`.

---

## Other runnable modules

```bash
python3 run_active_slam.py
python3 SwarmCapture+/Swarm_Target_Capture+.py
python3 DDFGO++/SwarmDDFGO++.py
python3 simplified_2d/simplified_swarm.py
```

*(Separate stacks — each script documents its own flags and dependencies.)*
