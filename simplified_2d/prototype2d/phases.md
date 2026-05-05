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

