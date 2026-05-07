# Phase 4 — Evaluation quick start

Runs under this folder align with **`simplified_2d/plan.md`** (results layout + metrics emphasis).

## Optional: `.env` and OpenAI policy

Install extras (repository root):

```bash
pip install -r simplified_2d/prototype2d/requirements-optional.txt
This optional requirements file now includes `openai-agents` for tool-driven behavior decisions.

```

Put **`OPENAI_API_KEY=...`** in a **`.env`** file at the repository root *or* in **`simplified_2d/prototype2d/`** (nearest `.env` found walking upward from the package wins). **`run_simulation`** loads it automatically via **`python-dotenv`** (no overwrite of already-exported env vars).

Cheap first test: **`config.openai_smoke.json`** — **`gpt-4o-mini`**, **`decision_period` 12 s**, **24 s** horizon (~6 LLM rounds for three agents).

```bash
python3 -c "from simplified_2d.prototype2d.simulator import run_simulation; run_simulation('simplified_2d/prototype2d/config.openai_smoke.json')"
```

## Single simulation

From the repository root:

```bash
python3 -c "from simplified_2d.prototype2d.simulator import run_simulation; run_simulation('simplified_2d/prototype2d/config.phase4.json')"
```

Artifacts land in **`simplified_2d/prototype2d/results/<experiment_name>/`**, including:

- **`config.json`**, **`target.json`** — exact provenance snapshot
- **`metrics_history.pkl`**, **`agents_history.pkl`**, … — raw timelines
- **`performance.json`** — compact KPIs (`time_to_all_docked`, **`integrated_control_effort`**, map/convex-hull/frontier summaries, dwell times, optional LLM token totals)

## Batch comparison (two policies)

The scenario manifest **`scenarios/phase4_compare.json`** fixes “observation parity” (same sensing, delays, seeds, durations) except the swept keys (e.g. **`decision_backend`** + **`rng_seed`**).

```bash
python3 -m simplified_2d.prototype2d.evaluation.batch \
  --scenario simplified_2d/prototype2d/scenarios/phase4_compare.json
```

Outputs (`batch_output_dir` in the scenario):

- **`summary.csv`** — one row per Cartesian product assignment
- **`summary.json`** — rows plus **`aggregates_across_rng_seed`** (means/std per policy when **`rng_seed` is part of the matrix)
- **`batch_manifest.json`** — run labels → **`results_dir`**

Naming convention:** each run **`name`** becomes **`${batch_name}_${key1}_${value1}__...`** nested under **`batch_output_dir`**.

## Plots without a display

Plot a telemetry series:

```bash
python3 -m simplified_2d.prototype2d.plotting \
  --results simplified_2d/prototype2d/results/phase4_mission_baseline \
  --metric map_coverage_ratio --save-dir /tmp/proto2d_figs
```

Stacked **`behavior_counts`**:

```bash
python3 -m simplified_2d.prototype2d.plotting \
  --results simplified_2d/prototype2d/results/phase4_mission_baseline \
  --metric behavior_stacked --save-dir /tmp/proto2d_figs
```

Bar chart from **`summary.csv`** (mean ± stderr bars within each **`decision_backend`** group):

```bash
python3 -m simplified_2d.prototype2d.plotting \
  --batch-summary simplified_2d/prototype2d/results/phase4_batch_compare/summary.csv \
  --group-key decision_backend --value-key integrated_control_effort \
  --save-dir /tmp/proto2d_figs
```

## Interpretation cues

- **`time_to_all_docked`** is **`null`** if not all agents reach mode **`d`** by horizon end (`performance.json`).
- **`frontier_coverage_ratio`** compares **`|⋃ agent frontiers|`** to a fixed oracle denominator (dense topology with one synthetic “missing landmark” snapshot); interpret as **relative** frontier pressure, not a physical unit.
- **`RandomBackend`** is intentionally chaotic (stress baseline); deterministic comparisons should prefer **`FSM`** vs **`openai`** with fixed **`rng_seed`** and awareness that cloud LLMs remain weakly stochastic and may omit token usage payloads on some transports.
- Structured operator hints outperform pure NL parsing in earlier phases; richer NL grounding is deliberately out-of-scope until later research.

## Quick validation checklist (agentic upgrade)

- Verify decision traces include tool-driven outputs by checking **`prompt_traces.pkl`** for backend `openai_agents_sdk`.
- Confirm invalid or missing tool calls degrade to `hold` (look for decision-invalid counters in `metrics_history.pkl` / `performance.json`).
- Check search stability and pointing:
  - reduce `search_bounce_gain` / `search_lateral_noise`,
  - tune `pointing_kp`, `pointing_kd`, `pointing_tau_cap`.
- Confirm follow/search guidance in sparse-visibility intervals by inspecting `messages_history.pkl` payload `navigation_hint` and behavior transitions.
