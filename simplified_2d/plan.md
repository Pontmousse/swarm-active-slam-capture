# 2D Swarm Prototype Simulator Plan

## Purpose

This module will provide a lightweight 2D research simulator for testing new swarm controllers before porting them to the full `SwarmCapture+` simulator.

The goal is not to reproduce PyBullet physics or the full 3D spacecraft stack. Instead, the prototype should preserve the architectural concepts that matter for controller development:

- shared physics time
- per-agent modes
- target landmarks and attachment points
- delayed perception, communication, and actuation
- simple SLAM/map emulation
- reusable experiment configuration
- telemetry and performance outputs for comparing controller variants

The existing `simplified_2d/simplified_swarm.py` is only a reference sketch. This new prototype should be a cleaner small module.

## Proposed Module Layout

```text
simplified_2d/prototype2d/
  __init__.py
  model.py              # dataclasses, state definitions, target/AP schemas
  simulator.py          # shared physics clock and step loop
  modes.py              # s/e/c/d mode switching
  controllers.py        # prototype controller implementations
  delays.py             # asynchronous delay queues and delay models
  perception.py         # 2D field-of-view sensing
  mapping.py            # simple observed-point-id map
  metrics.py            # performance metrics
  plotting.py           # telemetry plots
  target_sketch_tk.py   # Tk sketch → dense landmarks + edge-midpoint APs (stride-controlled) + bbox contour
  io.py                 # JSON/pickle save/load utilities
  plan.md               # this design document
```

The implementation should stay modular enough to support clean experiments, but the first version can keep the algorithms simple.

## State Model

The simulator remains 2D, with double-integrator dynamics close to free-floating spacecraft motion.

Agent state:

```text
[x, y, theta, vx, vy, omega]
```

Target state:

```text
[x, y, theta, vx, vy, omega]
```

Controller output per free agent:

```text
u_i = [Fx, Fy, tau]
```

Nominal continuous-time dynamics:

```text
x_dot     = vx
y_dot     = vy
theta_dot = omega

vx_dot    = Fx / mass
vy_dot    = Fy / mass
omega_dot = tau / inertia
```

The target may have prescribed velocity/angular velocity or a simple target propagation model. Docking should not change target mass, inertia, or impulse in the first version.

## Agent Data Model

Agent dictionaries or dataclasses should preserve names similar to the full simulator where useful:

```text
ID
TimeStep
Iteration
State
Mode
Mass
Inertia

CommSet

LandSet
Map

LC
LCD

APs
APs_Bids
Target
ActionTime

Control_Force
Control_Torque
Fuel_Consumed

DockPose
DockTime
```

This intentionally keeps the full simulator vocabulary only where it is useful for the 2D prototype. Extra fields such as smoothed states, collision sets, feature-index sets, and dock contact points can be added later if an experiment needs them.

## Modes

The initial prototype should support exactly four modes:

- `s`: search / exploration / initial target discovery
- `e`: encapsulation around the target/observed points
- `c`: capture / approach assigned attachment point
- `d`: docked / rigidly attached to target body

The mode system should be easy to extend later. A dispatch-table style is preferred:

```text
mode_controllers = {
  "s": search_controller,
  "e": encapsulate_controller,
  "c": capture_controller,
  "d": docked_controller,
}
```

Mode switching should live in `modes.py`, separate from controller force/torque generation.

## Simulation Step Phases

The prototype should mirror the full simulator's conceptual step loop:

```text
1. begin simulation step
2. perception phase
3. communication phase
4. decision / mode-switching phase
5. control phase
6. actuation-delay phase
7. physics propagation
8. telemetry / metric recording
```

Physics has one common simulation time for all agents and the target. Delay/asynchrony affects what each agent perceives, receives, and applies, not the underlying truth clock.

## Delay and Asynchrony Model

Delay modeling is a primary objective of this prototype.

The simulator should support three independent delay channels:

1. Perception delay
  - A target point is visible at physics time `t`.
  - It is delivered to the agent's perceptual/mapping pipeline at `t + delay_perception_i(t)`.
2. Communication delay
  - Agent `i` broadcasts a message at physics time `t`.
  - Agent `j` receives it at `t + delay_comm_ij(t)`.
3. Actuation delay
  - Agent `i` computes command `u_i(t)`.
  - The physics engine applies it at `t + delay_actuation_i(t)`.

Each agent can have its own delay evolution. Communication can also be per-link and asymmetric.

Delay models should support:

- zero delay / fully synchronized baseline
- constant delay
- constant delay plus jitter
- random-walk delay
- per-agent schedules for true asynchrony

The zero-delay configuration must cancel all asynchrony cleanly for comparison experiments.

The implementation should use timestamped queues/messages rather than only indexing into old history arrays. This makes different agent clocks, delay histories, and delayed actuation easier to test without introducing packet-loss/network models outside the current scope.

Packet loss, bounded stale-message rules, and other network reliability models are out of scope for the first version.

## Communication and Message Schema

The first communication model should be structured, simple, and broadcast-first.

Communication is part of the simulator model, not an external networking stack. It defines what each agent makes available to other agents, who receives it, and when delayed information becomes available for decisions.

### Topology

Initial topology:

- Agents broadcast to neighbors inside `communication_radius`.
- A neighbor is any other agent whose true position is within communication radius at send time.
- Broadcast is the default communication mode.

The schema should also leave room for later direct and operator messages:

- `broadcast`: agent to all current communication neighbors.
- `direct`: agent to one selected recipient, reserved for later.
- `operator_broadcast`: external operator/ground-station style instruction to multiple agents, reserved for later.
- `operator_direct`: external operator/ground-station style instruction to one agent, reserved for later.

Only `broadcast` needs to be active in the first implementation.

### Delivery

Communication delay is applied per recipient, not only per sent message.

If agent `i` broadcasts at time `t`, each recipient `j` gets its own delivery event:

```text
deliver_time_ij = t + delay_comm_ij(t)
```

Implementation preference:

- Create one queued message copy per recipient.
- Store `sent_time`, `deliver_time`, `sender_id`, and `recipient_id` on each queued copy.
- Move messages from the pending queue to the recipient inbox when `deliver_time <= current_time`.

This keeps asynchronous delivery simple: the same broadcast can arrive at different agents at different times.

### Minimal Message Fields

Each queued message should have a stable structured schema:

```json
{
  "message_id": "msg_000123",
  "sent_time": 12.35,
  "deliver_time": 12.55,
  "sender_id": 2,
  "recipient_id": 0,
  "channel": "broadcast",
  "message_type": "agent_status",
  "payload": {
    "state": [1.2, 0.4, 0.1, 0.02, 0.0, 0.01],
    "mode": "e",
    "target_ap": 2003,
    "action_time": 14.0,
    "visible_point_ids": [1001, 1002, 1008],
    "known_point_ids": [1001, 1002, 1008, 1010]
  },
  "metadata": {}
}
```

Required top-level fields:

- `message_id`
- `sent_time`
- `deliver_time`
- `sender_id`
- `recipient_id`
- `channel`
- `message_type`
- `payload`
- `metadata`

Initial `payload` fields:

- `state`: sender state `[x, y, theta, vx, vy, omega]`
- `mode`: sender mode (`s`, `e`, `c`, or `d`)
- `target_ap`: sender's selected attachment point, or `null`
- `action_time`: sender readiness/action-time variable
- `visible_point_ids`: point IDs observed in the current/recent perception update
- `known_point_ids`: point IDs currently stored in sender map memory

This message content is intentionally compact. It is enough for delayed neighbor state, delayed mode awareness, simple AP conflict handling, and shared map-coverage experiments.

### Agent Inbox

Each agent should maintain:

```text
Inbox
LastMessageBySender
```

`Inbox` stores newly delivered messages for the current decision cycle.

`LastMessageBySender` stores the most recent delivered message from each sender. Controllers and mode-switch logic can use this without scanning full message history.

Message history can still be saved globally for telemetry/debugging:

```text
messages_history.pkl
```

### Future Agentic Communication Placeholder

The long-term goal includes agents reasoning and communicating with natural language using LLMs, plus possible external operator/ground-station instructions.

The first version should not model LLM behavior, but the schema should leave placeholders for it. Future messages may use:

```json
{
  "message_type": "natural_language",
  "payload": {
    "text": "I see the right edge and will take AP 2003.",
    "intent": "claim_attachment_point",
    "references": {
      "target_ap": 2003,
      "known_point_ids": [1008, 1010]
    }
  }
}
```

Operator messages may later use:

```json
{
  "sender_id": "operator",
  "recipient_id": 0,
  "channel": "operator_direct",
  "message_type": "instruction",
  "payload": {
    "text": "Prioritize enclosing the target before capture.",
    "priority": "high"
  }
}
```

These are reserved for later. The first implementation should only require structured `agent_status` broadcast messages.

## True State Access

For now, controllers may receive true global state. This is intentional for controller prototyping and controlled comparisons.

However, perception, map memory, and communication messages should still be simulated and recorded. Later experiments can switch controllers from truth-based inputs to delayed/perceived/map-based inputs without redesigning the simulator.

## Target Model

The target remains simple but reusable.

Target geometry should be represented in the target body frame:

- contour points
- dense boundary points
- selected attachment/key points
- point IDs
- normals

World-frame point positions, velocities, and normals are computed from the target state at each simulation step.

The first implementation should support boundary points only, but the GUI should not make the data format impossible to extend to interior points later.

## Target Designer GUI

Author targets with `target_sketch_tk.py` (Tkinter): **Export simulator target** packs traced samples into `dense_points`; attachment points lie on polyline edge midpoints (nearest landmark), subsampled via `--attachment-edge-stride`; `contour_points` is a bbox placeholder; optional `*_lines.json` draft. Optional helper `build_target_definition_from_polygon` in the same module can resample a closed polygon with spacing + normals (no separate Matplotlib app).

Desired workflow:

1. Open target designer.
2. Draw or click a target contour.
3. Generate or accept dense target points along the boundary.
4. Select specified keypoints/attachment points.
5. Assign stable IDs to all target points.
6. Estimate or assign outward normals.
7. Save target geometry to JSON.
8. Reload the same target JSON for repeated experiments.

Example JSON structure:

```json
{
  "name": "prototype_target_v1",
  "units": "m",
  "body_frame": true,
  "contour_points": [
    {"id": 0, "x": -0.2, "y": -0.2},
    {"id": 1, "x": 0.2, "y": -0.2}
  ],
  "dense_points": [
    {"id": 1000, "x": -0.18, "y": -0.2, "normal": [0.0, -1.0]}
  ],
  "attachment_points": [
    {
      "id": 2000,
      "point_id": 1000,
      "x": -0.18,
      "y": -0.2,
      "normal": [0.0, -1.0],
      "label": "bottom_edge_1"
    }
  ]
}
```

All IDs should be stable across runs so telemetry, maps, and attachment assignments are comparable.

## Attachment Points

Attachment points should be analogous to the full simulator's `Attachment_Point` objects, but in 2D.

Each attachment point should expose:

```text
idx or id
point_id
position[t]
velocity[t]
normal[t]
label
```

Agents should maintain:

```text
APs
APs_Bids
Target
```

Initial AP assignment logic can copy the full simulator concept:

- detect visible/reachable APs
- compute simple distance-only bids
- resolve conflicts using local or delayed neighbor information

Velocity and normal-alignment bid terms may be added later, but the first version should stay deterministic and easy to inspect.

## Simple SLAM / Map Emulation

The target is modeled as a rigid set of identifiable 2D points.

Each point has a unique stable ID. When an agent observes a point, it observes that point's ID and optionally a relative measurement.

Per-agent map memory should track observed point IDs:

```text
Map[point_id] = {
  first_seen,
  last_seen,
  num_observations,
  last_relative_position,
  last_world_position_if_using_truth,
  last_observation_age
}
```

This is not full SLAM. It is an intentionally simple map/visibility memory that lets us test:

- how much of the target each agent has seen
- how communication delay affects shared map knowledge
- how stale observations affect decisions
- how controllers behave under partial target knowledge

## Perception Model

Each agent has a field of view:

```text
fov_radius
fov_angle
heading theta
```

A target point is visible if:

1. It is within `fov_radius`.
2. It lies inside the angular field of view around the agent heading.
3. Its normal is compatible with visibility, so points on the back side of the target are not observed.

For the normal check, use a simple facing condition. Let:

```text
point_to_agent = normalize(agent_position - point_position)
normal_world   = outward normal of target point
```

The point is visible only if:

```text
dot(normal_world, point_to_agent) > normal_visibility_threshold
```

The default threshold can be `0.0`, meaning the point must face at least partly toward the agent.

This is the first occlusion/back-face model. It does not need full geometric ray occlusion initially.

## Docking Model

When an agent reaches docking conditions in capture mode, it becomes part of the target body.

The first version should not change target mass, inertia, or impulse.

Instead:

1. Store the agent's fixed relative pose in the target body frame:

```text
DockPose = [dx, dy, dtheta]
```

1. Set:

```text
Mode = "d"
DockTime = current_iteration or current_time
```

1. During propagation, the docked agent follows the target rigid transform.

Docking should still be recorded in telemetry so capture time and final assignments can be analyzed.

## Experiment Configuration

Use JSON for experiment configs.

The config should include:

- simulation duration
- `dt`
- number of agents
- initial agent states
- target JSON path
- initial target state
- target motion parameters
- masses/inertias
- controller gains
- mode thresholds
- sensing radius/angle
- normal visibility threshold
- communication radius
- delay model definitions
- random seed
- output directory/name
- plotting/save intervals

Every experiment output folder should save a copy of the config used for the run.

## Outputs and Telemetry

The simulator should save reusable raw data and compact performance summaries.

Suggested output layout:

```text
simplified_2d/prototype2d/results/<experiment_name>/
  config.json
  target.json
  agents_history.pkl
  target_history.pkl
  attachment_points.pkl
  metrics_history.pkl
  messages_history.pkl
  performance.json
```

Plotting utilities should follow the spirit of `DDFGO++/Plot_Telemetry_Func.py` and `SwarmCapture+/Plot_Telemetry_Func.py`: reusable functions that extract quantities by case/name and can be reused across experiments.

## Metrics

Record metrics over time, not only at the end.

Important metrics:

- convex hull area of all agents
- convex hull perimeter
- target center inside agent convex hull
- minimum/mean/maximum agent-target distance
- minimum inter-agent distance
- mode counts over time
- time spent in each mode per agent
- time to all docked
- AP assignment conflicts
- AP coverage ratio
- capture error to assigned AP
- total control effort
- fuel proxy
- mean/max message age
- per-agent map size
- shared/global observed target coverage ratio

The convex hull area metric is especially important for comparing encapsulation behavior across controllers and delay settings.

## Initial Implementation Priorities

1. Define dataclasses/schemas for agents, target points, attachment points, configs, messages, and metrics.
2. Implement target JSON load/save and a basic Matplotlib target designer.
3. Implement 2D target transform utilities for point positions, velocities, and normals.
4. Implement perception with radius, angle, and normal-facing checks.
5. Implement per-agent map memory of observed point IDs.
6. Implement delay queues for perception, communication, and actuation.
7. Implement basic `s/e/c/d` mode switching.
8. Implement simple baseline controllers for each mode.
9. Implement propagation and docking-as-rigid-attachment.
10. Implement metrics and output saving.
11. Implement plotting/animation helpers.

## Open Questions Before Coding

Most initial design choices are now settled:

1. Mode switching should use simple conditioned thresholds for now, but the code should leave a clean opening for more complex agentic decisions later. The long-term direction is swarm intelligence: agents may eventually reason, communicate in natural language through LLMs, and receive external instructions from an operator or ground station.
2. Communication messages should start simple and include compact state/mode/target-related data. Leave placeholders for richer message content later, including text or structured outputs from future LLM agents.
3. AP bids should be distance-only in the first version.
4. Controllers should be deterministic and simple in the first version.
5. The default target fixture should be a rectangle.
6. Outputs should live under `simplified_2d/prototype2d/results/`.

Remaining implementation detail to choose during coding:

- exact default numeric thresholds for `s -> e`, `e -> c`, and `c -> d`
- exact JSON schema names for experiment configs

