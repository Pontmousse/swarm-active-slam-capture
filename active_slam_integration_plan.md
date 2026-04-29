# Active SLAM Integration Plan

## Objective

Integrate the current PyBullet swarm-capture simulator and the DDFGO++ decentralized SLAM pipeline into a single active SLAM simulation workflow.

The long-term goal is to allow SLAM outputs, such as agent pose estimates, sparse/dense maps, target kinematic estimates, uncertainty, and map quality, to influence swarm controller decisions or behavior selection during the simulation, instead of running control first and SLAM afterward as a separate replay stage.

The first implementation step is not to combine the loops immediately. The first step is to refactor the two current main scripts into self-contained, import-safe modules:

- `SwarmCapture+/Swarm_Target_Capture+.py`
- `DDFGO++/SwarmDDFGO++.py`

Each file should still be runnable as a standalone script through an `if __name__ == "__main__":` entry point, but should also expose functions/classes that a new root-level active SLAM runner can call.

## Current Architecture

### SwarmCapture+

`Swarm_Target_Capture+.py` is the PyBullet-based swarm simulation. It currently performs these responsibilities in one top-level script:

- Connects to PyBullet.
- Configures simulation, control, lidar, target, and visualization parameters.
- Loads the target model and target point cloud.
- Creates attachment points.
- Creates agents and their PyBullet bodies.
- Runs the full physics/control loop.
- Performs lidar ray casting and local target observation.
- Computes neighborhood sets and communication sets.
- Runs behavior selection and controller logic.
- Applies forces/torques or docked pose updates.
- Advances PyBullet.
- Records agent, target, target point cloud, and attachment-point history.
- Periodically writes Excel and pickle outputs.
- Computes final performance metrics.
- Disconnects PyBullet.

This is currently a batch simulator.

### DDFGO++

`SwarmDDFGO++.py` is the offline SLAM/factor-graph pipeline. It currently performs these responsibilities in one top-level script:

- Loads pickled simulation histories from disk.
- Downsamples histories.
- Initializes GTSAM noise models, symbols, factors, and graph structures.
- Adds SLAM-specific fields into every agent history entry.
- Initializes per-agent factor graphs.
- Replays the full simulation history.
- Adds noisy GPS, odometry, bearing/range, kinematic, and optional target-parameter factors.
- Performs decentralized neighbor feature sharing.
- Updates sparse maps, dense merged maps, target estimates, and error metrics.
- Periodically saves output pickles.

This is currently an offline replay pipeline, not an online SLAM engine.

## Desired Architecture

The integrated architecture should eventually look like this:

1. A root active SLAM runner initializes the physics simulator.
2. The runner initializes the SLAM engine from the initial simulator state.
3. The runner advances physics/control at high frequency, for example 240 Hz.
4. The runner collects sensor observations and history frames from the simulator.
5. At a lower SLAM frequency, for example 1-10 Hz, the runner calls the SLAM engine with the latest observation frame.
6. The SLAM engine updates agent pose estimates, maps, target estimates, and quality metrics.
7. The controller or behavior-selection layer consumes selected SLAM outputs.
8. The simulation continues with closed-loop map-aware behavior.
9. Saving, plotting, and evaluation remain optional outputs controlled by the runner.

## Phase 1: Make Both Main Scripts Import-Safe

### Goal

Both main files should be safe to import without running a simulation, opening PyBullet, loading pickles, or writing files.

### Required Refactor

- Move all top-level execution into callable functions.
- Add `if __name__ == "__main__": main()` to each script.
- Preserve existing standalone behavior through `main()`.
- Avoid changing numerical behavior during this phase unless a bug prevents modularization.

### Proposed Standalone Entrypoints

For `Swarm_Target_Capture+.py`:

- `main()`
- `run_simulation(config=None)`

For `SwarmDDFGO++.py`:

- `main()`
- `run_batch_from_config(config_module=config)`
- `run_batch_from_histories(agents_history, target_history, target_point_cloud_history, config_obj)`

## Phase 2: Extract Explicit Runtime State

### Swarm Simulation State

The PyBullet script should expose a runtime object or dictionary that owns:

- PyBullet client ID.
- Target body ID.
- Agent body IDs.
- Target point cloud.
- Target state.
- Attachment points.
- Current agent dictionaries.
- Agent, target, and point-cloud histories.
- Simulation parameters.
- Sensor parameters.
- Controller parameters.
- Save/output configuration.
- Current iteration and simulation time.

### SLAM Runtime State

The DDFGO++ script should expose a runtime object or dictionary that owns:

- Per-agent factor graphs.
- Per-agent GTSAM values.
- GTSAM symbols.
- Noise models.
- Landmark registries.
- Keyframe histories.
- Current SLAM iteration.
- Current map estimates.
- Current target kinematic estimates.
- Configuration values derived from `config.py`.
- Output/save settings.

## Phase 3: Split PyBullet Into Setup, Step, Finalize

### Proposed Units

- `create_simulation_config()`
- `connect_physics(config)`
- `load_target(config, physics_client)`
- `create_attachment_points(target_state, target_pcd, config)`
- `create_agents(config, physics_client)`
- `initialize_histories(sim_state)`
- `sense_agent(sim_state, agent_id)`
- `update_agent_neighborhoods(sim_state, agent_id)`
- `update_agent_behavior(sim_state, agent_id)`
- `compute_agent_control(sim_state, agent_id)`
- `apply_agent_control(sim_state, agent_id)`
- `step_physics(sim_state)`
- `record_simulation_frame(sim_state)`
- `save_simulation_outputs(sim_state)`
- `compute_performance_metrics(sim_state)`
- `teardown_simulation(sim_state)`

### Important Design Point

The future active SLAM runner needs a function that advances exactly one physics step, not only a function that runs the whole simulation.

The eventual interface can be conceptually:

```text
sim_state = initialize_simulation(...)
frame = step_simulation(sim_state, controller_feedback=None)
```

The `frame` should include the data needed by SLAM:

- Agent true states.
- Target true state.
- Agent observations, especially `LandSet`.
- Communication sets.
- Target point cloud snapshot if needed for evaluation.
- Time and iteration metadata.

## Phase 4: Split DDFGO++ Into Batch and Online Paths

### Batch Path

The batch path should preserve the current behavior:

```text
histories = load_histories(...)
slam_state = initialize_slam_from_histories(histories, config)
results = run_slam_batch(slam_state, histories)
save_results(results)
```

This keeps existing plotting and offline experiments usable.

### Online Path

The online path should support:

```text
slam_state = initialize_slam_online(initial_frame, config)
slam_output = step_slam(slam_state, frame)
```

The online `step_slam()` should:

- Accept one new frame or a small buffer of frames.
- Add pose, odometry, landmark, communication, and kinematic factors for that frame.
- Optimize/update the current per-agent estimates.
- Update maps and target kinematics.
- Return controller-facing outputs.

### Key Challenge

`SwarmDDFGO++.py` currently assumes the full history is available before SLAM starts. The online path must remove that assumption. It should initialize only the first frame, then append/process new frames as they arrive.

## Phase 5: Root-Level Active SLAM Runner

After both modules are callable, create a new root script. Possible name:

- `run_active_slam.py`

Its responsibilities:

- Load unified config.
- Initialize PyBullet simulation.
- Initialize SLAM.
- Maintain timing/scheduling between physics and SLAM.
- Run physics at high frequency.
- Run SLAM at lower frequency.
- Pass SLAM feedback into behavior selection or controllers.
- Save final histories and results.
- Handle teardown reliably.

Conceptual scheduling:

```text
for physics_step in range(total_steps):
    controller_feedback = latest_slam_output
    frame = step_simulation(sim_state, controller_feedback)

    if should_run_slam(physics_step):
        slam_output = step_slam(slam_state, frame)
        latest_slam_output = slam_output
```

## Controller Feedback Interface

The integration should not directly expose all SLAM internals to the controller. Instead, define a small feedback object that can evolve over time.

Potential feedback fields:

- `agent_pose_estimates`
- `agent_pose_uncertainty`
- `target_com_estimate`
- `target_velocity_estimate`
- `target_angular_velocity_estimate`
- `sparse_maps`
- `dense_maps`
- `map_quality`
- `coverage_metrics`
- `frontier_or_next_best_view_candidates`
- `recommended_behavior_mode`
- `recommended_control_frame`
- `confidence`

Early integration can start with read-only feedback, for example target COM/velocity estimates and map quality, before allowing the map to modify control decisions.

## Major Risks And Mitigations

### Risk 1: Top-Level Side Effects Break Imports

Both main scripts currently execute immediately on import.

Mitigation:

- First refactor only enough to make imports safe.
- Preserve standalone execution through `main()`.
- Add a simple smoke check that importing each module does not start PyBullet, load pickles, or write files.

### Risk 2: Non-Importable File And Folder Names

The `+` characters in `SwarmCapture+`, `DDFGO++`, `Swarm_Target_Capture+.py`, and `SwarmDDFGO++.py` make normal Python imports awkward.

Mitigation:

- Use wrapper modules with safe names.
- Or create package-safe aliases later.
- Avoid renaming folders as the first step unless the wider codebase paths are updated carefully.

### Risk 3: DDFGO++ Is Offline, Not Online

The SLAM pipeline currently replays existing histories from disk. Active SLAM needs incremental updates during simulation.

Mitigation:

- Keep the existing batch path intact.
- Add an online API beside it.
- Refactor shared logic into functions used by both batch and online modes.
- Start with one SLAM update per saved simulator frame, then introduce lower-frequency scheduling.

### Risk 4: History Schema Mismatch

The simulator uses fields such as `Control_Frame`, while DDFGO++ still references older fields such as `LCD_Frame`, `DockSet`, and `Odometry` in cleanup logic.

Mitigation:

- Define one agent-frame schema.
- Make cleanup tolerant with optional-key removal.
- Add schema normalization at the boundary between simulation and SLAM.
- Avoid silent field renames without documenting them.

### Risk 5: Timing And Frequency Coupling

Physics runs at high frequency, while SLAM should run slower. Incorrect scheduling can make odometry, target propagation, and kinematic factors inconsistent.

Mitigation:

- Track physical time explicitly.
- Pass actual `dt` and elapsed time into SLAM.
- Avoid hard-coded `1/240` inside SLAM utilities where possible.
- Test with simple frequency ratios first, such as SLAM every 240 physics steps.

### Risk 6: Hidden Dependence On Ground Truth

DDFGO++ currently uses true pose and target point-cloud history in several places for simulated measurements and evaluation. Active SLAM should carefully separate truth, measurements, estimates, and metrics.

Mitigation:

- Define separate objects for truth, observations, estimates, and evaluation-only data.
- Keep target point cloud truth available for metrics, but do not feed it into controller decisions unless explicitly intended.
- Label all truth-based fields clearly.

### Risk 7: Memory Growth

Both systems store large histories. Active SLAM with dense maps can grow quickly.

Mitigation:

- Keep full-history saving optional.
- Use sliding windows for online SLAM.
- Store dense scans downsampled.
- Save checkpoints periodically but allow lightweight in-memory operation.
- Define a retention policy for per-frame debug fields.

### Risk 8: I/O Inside Compute Loops

Both current scripts save files during the main computation. This makes integrated experiments slower and harder to control.

Mitigation:

- Move all saving behind explicit functions.
- Make periodic saves configurable.
- In integrated mode, let the root runner control output cadence.

### Risk 9: PyBullet Lifecycle Management

PyBullet connection, reset, and disconnect are currently top-level side effects.

Mitigation:

- Store the physics client in simulation state.
- Provide explicit setup and teardown functions.
- Use `try/finally` in standalone and integrated runners.
- Avoid disconnecting inside a step function.

### Risk 10: Randomness And Reproducibility

Both scripts use random sampling for agent initialization, communication, feature selection, and noise.

Mitigation:

- Add a seed to the unified config.
- Initialize Python `random` and NumPy RNG deterministically when requested.
- Keep stochastic behavior enabled by default if that matches existing experiments.

### Risk 11: Controller Feedback Instability

Early SLAM estimates may be noisy or delayed. If control immediately trusts them, the swarm behavior may become unstable.

Mitigation:

- Start with passive SLAM feedback logging.
- Add confidence thresholds before feedback affects control.
- Low-pass filter SLAM-derived target estimates.
- Keep fallback behavior based on raw perception.
- Gate behavior switching using map quality or estimate confidence.

### Risk 12: Duplicated Or Conflicting Target Kinematics

The simulator has true target state; DDFGO++ estimates target COM, velocity, and angular velocity. Active control must decide which one to use.

Mitigation:

- Use true target state only for simulation physics and evaluation.
- Use SLAM estimates for active behavior only after a controlled switch.
- Record both true and estimated target kinematics for debugging.

### Risk 13: Existing Results/Plotting Scripts May Break

Changing history structure can break animation and plotting scripts.

Mitigation:

- Preserve output pickle schema where practical.
- Add compatibility fields during transition.
- Keep batch output paths and tags stable.
- Update plotting scripts only after core modularization is stable.

### Risk 14: Large Refactor Could Change Numerical Behavior

Moving code into functions can accidentally change ordering, copying, mutation, or random calls.

Mitigation:

- Refactor in small phases.
- Compare standalone outputs before and after modularization.
- Start with smoke runs.
- Add simple regression checks for history length, number of agents, final modes, and basic map growth.

## Proposed Implementation Sequence

### Step 1: Import-Safe Wrapping

- Add `main()` to both main scripts.
- Move top-level executable code under `main()` or helper functions.
- Verify importing each file does not run the simulation.

### Step 2: Configuration Objects

- Create or expose config objects for both systems.
- Keep defaults equal to current hard-coded values.
- Move hard-coded paths toward module-relative paths.

### Step 3: PyBullet Engine Extraction

- Extract setup logic.
- Extract one-step simulation logic.
- Extract save/performance/teardown logic.
- Preserve current standalone run behavior.

### Step 4: DDFGO Batch Refactor

- Extract history loading.
- Extract SLAM initialization.
- Extract per-timestep/per-agent SLAM update logic.
- Preserve current batch replay behavior.

### Step 5: DDFGO Online API

- Add initialization from one simulator frame.
- Add `step_slam()` that consumes new frames incrementally.
- Return map and estimate feedback.

### Step 6: Root Active SLAM Script

- Create the root runner.
- Add timing scheduler for physics and SLAM.
- Initially run SLAM passively without affecting control.
- Save integrated outputs.

### Step 7: Close The Active Loop

- Feed selected SLAM outputs into controller or behavior selection.
- Start with conservative feedback channels.
- Add confidence gates and fallback behavior.

## Initial Success Criteria

The first refactor is successful when:

- Both main scripts can still run standalone.
- Both main scripts can be imported without side effects.
- The PyBullet script exposes a callable full simulation function.
- The DDFGO++ script exposes a callable batch SLAM function.
- Existing output pickle names and contents remain compatible, except for clearly documented intentional changes.

The first integration is successful when:

- A root runner can initialize both systems.
- Physics advances at high frequency.
- SLAM updates at lower frequency.
- SLAM output is produced during the simulation.
- The controller can receive SLAM feedback, even if it initially ignores it.

## Near-Term Recommendation

Start with the safest possible refactor:

1. Add import-safe `main()` boundaries.
2. Extract full-run functions without changing internal logic.
3. Then split the large loops into setup/step/save functions.

This keeps the current experiments working while creating the minimum structure needed for an active SLAM runner.