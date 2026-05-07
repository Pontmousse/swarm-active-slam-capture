# Implementation Phases (Prototype 2D)

This document splits the hybrid LLM + classical swarm prototype into **four phases of roughly equal scope**. Each phase ends with something runnable and measurable. Later phases assume earlier artifacts exist but avoid big-bang integration.

---

## Phase 1 — Physics, safety shell, and motion primitives

**Goal:** A stepping simulator with double-integrator agents, a rigid target, delayed channels stubbed or minimal, and **always-on** 2D analogues of flocking + antiflocking (from `SwarmCapture+/Controllers.py` spirit: attraction to landmarks / target geometry + repulsion from peers), plus velocity damping. No LLM yet.

**Deliverables**

- Shared physics clock; propagation with `[Fx, Fy, tau]` and mass/inertia.
- Per-agent **behavior slot** (`hold`, `search`, …) driven by a **placeholder** policy (fixed schedule or simple FSM) so control wiring exists before intelligence.
- Implement **hold** (brake / damp) and **search** (bounded roam, e.g. bounce-in-box style adapted from the commented 3D search pattern—no landmark alignment here).
- **Flocking + antiflocking** blended every step: configurable gains and radii; documented mapping from 3D landmark + neighbor sets to 2D (e.g. dense points / CoM vs other agents).
- Optional: minimal target hull standoff so agents do not sit inside the contour.
- Experiment config + results folder layout aligned with `plan.md`; save trajectories and basic metrics (positions, fuel proxy, mode or behavior id).

**Exit criteria**

- N agents run for full horizon without NaNs; separation forces bounded; search visibly explores a region.
- Swapping the placeholder policy does not require touching flocking/antiflocking math.

### Phase 1 implementation notes (as built)

- `**phase1_behaviors_only`** in `[ExperimentConfig](simplified_2d/prototype2d/model.py)`: when `true`, agents stay in mode `**p**`, AP assignment / docking are skipped, and control is `**compose_phase1_control**` or, if `**phase2_behaviors_enabled**`, `**compose_behavior_control**` in `[swarm_forces.py](simplified_2d/prototype2d/swarm_forces.py)` (`hold` / bounce `**search**` / `**characterize**` / `**follow**` + flocking + antiflocking + swarm damping + optional standoff, then clamp by `**shell_force_cap**`).
- **Legacy path** (`phase1_behaviors_only: false`): modes `**s` / `e` / `c` / `d`** unchanged; PD controllers in `[controllers.py](simplified_2d/prototype2d/controllers.py)` get an additive `**swarm_shell_for_legacy**` (flocking + antiflocking + standoff, no extra swarm velocity damping to avoid doubling with PD damping).
- **Placeholder policy**: `[placeholder_policy.py](simplified_2d/prototype2d/placeholder_policy.py)` — switches `**search` → `hold`** at `**phase1_hold_switch_time**` (only when `phase1_behaviors_only` is true).
- **Example config**: `[config.phase1.json](simplified_2d/prototype2d/config.phase1.json)`. Run from repo root:  
`python3 -c "from simplified_2d.prototype2d.simulator import run_simulation; run_simulation('simplified_2d/prototype2d/config.phase1.json')"`
- **Telemetry**: `[MetricsSnapshot](simplified_2d/prototype2d/model.py)` adds `**behavior_counts`** and `**min_inter_agent_distance**`; agent broadcast payload includes `**behavior**`.

---

## Phase 2 — Map, frontiers, communicate, follow & characterize

**Goal:** Dense-target map memory, **frontier extraction**, delayed perception and broadcast communication as in `plan.md`, plus behaviors that depend on shared information.

**Deliverables**

- Per-agent map (`Map[point_id] → observations`) updated from **delayed** perception pipeline (constant delay OK initially).
- **Frontier algorithm:** given observed vs unobserved dense boundary points (adjacency from target topology), compute frontier ids or positions; store live field `**map_frontier`** on each agent and persist in **agent history** snapshots (`map_frontier` column/list).
- **characterize:** controller drives motion toward frontier targets (e.g. nearest frontier point in world frame, with standoff) while flocking/antiflocking remain on.
- Structured **broadcast** comm + inbox + `last_messages_by_sender`; timestamps for delay per recipient.
- **follow:** interpret peer intent as a **typed hint** (target CoM, world point, or point id list) derived from messages—start with a minimal structured payload inside the simulator message schema so behavior is testable; NL text can remain the human-readable field for later LLM rounds.
- Dock/capture **not** required yet beyond “move toward point” if needed for smoke tests.

**Exit criteria**

- Agents with different visibility under delay build different maps; frontiers update as coverage grows.
- At least one agent can **follow** another agent’s hint and reduce distance to the indicated region.
- **characterize** measurably increases map coverage vs search-only runs on the same seed.

### Phase 2 implementation notes (as built)

- **Dense topology:** `[load_target_definition](simplified_2d/prototype2d/io.py)` fills `**dense_point_ids_ordered`** (JSON file order of `dense_points`) and `**dense_adjacency**` via `**build_dense_boundary_topology**`; `**dense_boundary_closed**` closes last↔first neighbors when `true`.
- **Frontiers:** `[frontiers.compute_map_frontier](simplified_2d/prototype2d/frontiers.py)` — observed dense ids with at least one adjacent id not yet in the agent map. Recomputed each step after perception updates in `[simulator.py](simplified_2d/prototype2d/simulator.py)`; stored on `**Agent.map_frontier`** (included in `**agents_history**`).
- **Map entries:** perception updates may set `**last_world_position`** per point id when positions are known from current `**dense_world**`.
- **Behaviors:** `[compose_behavior_control](simplified_2d/prototype2d/swarm_forces.py)` when `**phase1_behaviors_only`** and `**phase2_behaviors_enabled**` — `**characterize**` (PD toward nearest frontier + standoff; empty frontier → CoM or hold via `**characterize_fallback**`), `**follow**` (`**resolve_follow_goal**` from inbox `**navigation_hint**`; tie-break: latest `**sent_time**`, then lowest sender id). Otherwise `**compose_phase1_control**` unchanged.
- **Broadcast payload:** `**navigation_hint`**: `{ kind: none | target_com | world_point | point_ids, world_xy, point_ids }`, `**intent_text**` (optional demo tag). Deterministic demo: agent `**hint_demo_broadcast_agent_id**` emits `**target_com**` (or `**point_ids**` if `**hint_demo_point_ids**` non-empty) once `**len(map) >= hint_demo_min_map_points**`.
- **Placeholder:** `[placeholder_policy.py](simplified_2d/prototype2d/placeholder_policy.py)` — if `**phase2_behaviors_enabled`**: `**search**` until `**phase2_search_until**`, then `**characterize**` until `**phase2_characterize_until**`, then `**follow**` for non-scout agents and `**characterize**` for the scout.
- **Metrics:** `**frontier_size_mean`**, `**global_frontier_union_count**` on `[MetricsSnapshot](simplified_2d/prototype2d/model.py)`.
- **Example config:** `[config.phase2.json](simplified_2d/prototype2d/config.phase2.json)`. Run:  
`python3 -c "from simplified_2d.prototype2d.simulator import run_simulation; run_simulation('simplified_2d/prototype2d/config.phase2.json')"`

---

## Phase 3 — Capture, dock gates, LLM decision loop (hybrid)

**Goal:** Full tactical stack: **capture** (PD toward assigned AP, 2D analogue of `Capture_PID`), **dock** with simulator-enforced gates (distance, speed, alignment, AP conflict rules). **LLM** chooses behavior + bounded parameters on a **decision period**, not every physics step; deterministic validator clamps outputs.

**Deliverables**

- **capture:** PD to assigned AP; optional normal-alignment term in 2D.
- **dock:** request from policy; commit only if gates pass; update agent mode/pose per `plan.md` (e.g. rigid attachment record).
- **BehaviorCommand** schema: `behavior`, numeric params within global limits (e.g. aggressiveness, search box scale), optional `outbound_message`, optional `target_ap_id`.
- **Decision backends:** `FSM` / `random` baselines + **OpenAI Agents SDK** (or equivalent) backend behind a single interface `decide(snapshot) → BehaviorCommand`.
- Logging: LLM call count, wall-clock latency per call, parse failures, behavior transitions.

**Exit criteria**

- Same scenario runs with FSM-only vs LLM-backend without code changes to flocking or primitives.
- Dock only occurs when gates satisfied; invalid LLM outputs fall back to safe default (e.g. hold) with logged failure.

### Phase 3 implementation notes (as built)

- **[`BehaviorCommand`](simplified_2d/prototype2d/behavior_command.py):** `behavior` ∈ `search|hold|characterize|follow|capture|dock`, `params` (e.g. `aggressiveness` ∈ [0,1]), optional `target_ap_id`, optional `outbound_message`. **`validate_and_clamp`** returns `None` on illegal payloads → simulator applies **`hold`**.
- **Capture PID:** [`capture_pid_controller`](simplified_2d/prototype2d/controllers.py) implements **F_p + F_v + optional F_n** (same geometry as 3D `Capture_PID`); **`capture_alignment_gain`** scales **F_n** (default **0** keeps PD-like capture).
- **Dock gates:** [`docking.can_dock`](simplified_2d/prototype2d/docking.py) — distance ≤ **`dock_distance`**, relative speed ≤ **`dock_max_rel_speed`**, velocity-based alignment with inward normal **`dock_heading_dot_threshold`** (low-speed branch uses geometric approach ray).
- **`phase3_mission_enabled`:** enables AP bidding while **`phase1_behaviors_only`** ( **`run_ap_logic`** ); **`compose_behavior_control`** gains **`capture`** / **`dock`** and receives **`attachment_world`** for PID. Dock commitment (`mode == d`) is evaluated in [`simulator.py`](simplified_2d/prototype2d/simulator.py) before forces when **`behavior == dock`** and gates pass.
- **Decision loop:** if **`phase3_mission_enabled`** OR **`decision_backend` ≠ `fsm`**, periodic **`decide`** ([`decision/`](simplified_2d/prototype2d/decision/) — **`FSMBackend`**, **`RandomBackend`**, **`OpenAIBackend`**) on **`decision_period`**; else **`placeholder_policy.update_behaviors`** (Phase 1/2 schedule unchanged).
- **Telemetry:** **`MetricsSnapshot`** adds **`decision_*`** / **`llm_calls_step`**; **`performance.json`** adds **`decision_calls_total`**, **`decision_invalid_total`**, **`decision_latency_sum_sec`**, **`llm_calls_total`**.
- **Optional OpenAI:** install **`openai`** + **`python-dotenv`** (see **`requirements-optional.txt`**); put **`OPENAI_API_KEY`** in repo or package **`.env`**, or export it manually; **`decision_backend: openai`**. **`run_simulation`** calls **`maybe_load_dotenv()`** once (does not override existing env). Cheap smoke config: **`config.openai_smoke.json`**. Failures → **`hold`**.
- **Example config:** [`config.phase3.json`](simplified_2d/prototype2d/config.phase3.json).

---

## Phase 4 — Experiments, metrics, and evaluation harness

**Goal:** Repeatable scenarios, configs, and reporting so you can compare policies and delay settings against the metrics already planned (and LLM-specific ones).

**Deliverables**

- **Scenario library:** at least one full mission script (release → search → characterize/follow → capture → dock) with seeds and JSON configs.
- **Metric suite:** convex hull / distances / map coverage / frontier coverage / AP coverage / control effort / fuel / message age / time-to-dock / **LLM overhead** (calls, tokens optional, latency) / behavior histograms.
- **Batch runner:** sweep seeds and configs; write `performance.json` + plots consistent with `plan.md` output layout.
- Short **readme section** in this folder or cross-link: how to run Phase 4 evaluation and interpret comparisons (hybrid vs classical baseline).

**Exit criteria**

- One command (or documented sequence) reproduces a paper-ready comparison table for at least two policies on identical observation parity rules.
- Documentation lists known limitations (e.g. NL richness, frontier definition, delay models).

### Phase 4 implementation notes (as built)

- **Reproducibility:** **`rng_seed`** in [`ExperimentConfig`](simplified_2d/prototype2d/model.py) seeds [`DelayModel`](simplified_2d/prototype2d/delays.py) RNGs (`np.random.default_rng` streams), [`search_bounce_controller`](simplified_2d/prototype2d/swarm_forces.py) lateral noise (`random.Random`), and [`RandomBackend`](simplified_2d/prototype2d/decision/backends.py).
- **Provenance:** each run writes **`config.json`** and **`target.json`** under **`results/<experiment_name>/`** (plus existing pickles / `performance.json`), matching [`plan.md` (simplified_2d)](simplified_2d/plan.md).
- **Metrics additions:** [`metrics.py`](simplified_2d/prototype2d/metrics.py) — convex hull perimeter, target-in-agent-hull ray test, minimum agent→dense-boundary clearance, oracle frontier denominator + **`compute_frontier_coverage_ratio`**. **[`MetricsSnapshot`](simplified_2d/prototype2d/model.py)** adds **`convex_hull_perimeter`**, **`target_center_inside_hull`**, **`min_agent_boundary_distance`**, **`frontier_coverage_ratio`**, **`llm_*_tokens_step`**. **`performance.json`** adds integrated control effort (\(\sum_t \|F\|_1 \Delta t\) over agents), **`mean_decision_latency_sec`** (all periodic `decide` calls), **`time_in_mode_per_agent_sec`**, **`time_in_behavior_per_agent_sec`**, LLM token totals (OpenAI backend when exposed), **`final_*`** rollups from the last timestep.
- **OpenAI:** [`OpenAIBackend`](simplified_2d/prototype2d/decision/backends.py) sets **`last_usage`** from API usage when present; simulator aggregates **`llm_prompt_tokens_total`** / **`llm_completion_tokens_total`** (models without usage stay at **0**).
- **Canonical config:** [`config.phase4.json`](simplified_2d/prototype2d/config.phase4.json) (`rng_seed`, longer horizon for docking experiments).
- **Batch runner:** **`python -m simplified_2d.prototype2d.evaluation.batch --scenario simplified_2d/prototype2d/scenarios/phase4_compare.json`** writes **`summary.csv`**, **`summary.json`** (aggregates keyed by **`decision_backend`** when **`rng_seed` is swept), and **`batch_manifest.json`** under the scenario’s **`batch_output_dir`**.
- **Plots:** **`python -m simplified_2d.prototype2d.plotting --save-dir ./figs`** (non-interactive); **`--metric behavior_stacked`**; **`--batch-summary summary.csv --group-key decision_backend --value-key integrated_control_effort`**.
- **Documentation:** **[`EVALUATION.md`](simplified_2d/prototype2d/EVALUATION.md)** — commands, interpreting summaries, limits (NL grounding, stochastic random policy, API nondeterminism).

---

## Dependency sketch

```text
Phase 1 (physics + safety + hold/search)
    → Phase 2 (map + frontier + comm + characterize + follow)
        → Phase 3 (capture + dock + LLM hybrid)
            → Phase 4 (experiments + metrics + harness)
```

---

## Notes

- **Equal load** here means comparable breadth (systems touched + integration risk), not identical line counts.
- If scope slips, **drop** narrative NL parsing first; keep typed hints for follow until Phase 4 polish.
- Align always-on flocking/antiflocking tuning with metrics that matter in Phase 4 (e.g. min inter-agent distance, hull clearance).

