# Swarm Active SLAM Capture Documentation

## Scientific Objective

This project studies active multi-agent capture of a moving spacecraft-like target while simultaneously building and sharing a map of that target. The root workflow combines:

- a 3D PyBullet spacecraft swarm simulator in `SwarmCapture+/`;
- a DDFGO++ factor-graph SLAM and map-merging stack in `DDFGO++/`;
- an online orchestration layer in `run_active_slam.py`;
- a lighter 2D prototype in `simplified_2d/prototype2d/` for fast experiments with modes, delays, communication, mapping, and capture logic.

The main scientific questions represented in the code are how agents search, encapsulate, select capture points, dock, communicate observations, and reconstruct target state/maps under decentralized or delayed information.

## Repository Map

| Path | Role |
| --- | --- |
| `run_active_slam.py` | Root online runner that steps the 3D simulator and periodically calls online DDFGO++ SLAM. |
| `shared_config.py` | Shared identity, timing, checkpoint, visualization, and data path defaults used by simulation and SLAM. |
| `SwarmCapture+/` | 3D PyBullet target-capture simulator, control logic, perception, target loading, telemetry, and animation scripts. |
| `DDFGO++/` | Decentralized factor graph optimization, feature processing, custom factors, map merging, target-state estimation, and mapping visualizations. |
| `simplified_2d/prototype2d/` | Standalone 2D prototype simulator with JSON config, delayed perception/communication/actuation, simple controllers, metrics, plotting, and animation. |
| `utilities/` | Independent geometry, coverage, ellipsoid, RANSAC plane, contact-point, mock-data, and gossip demos. |
| `requirements.txt` | Python dependencies for the common numerical, plotting, simulation, and utility stack. `gtsam` is commented out but is required by `DDFGO++/SwarmDDFGO++.py`. |

## Main 3D Online Workflow

Run from the repository root:

```bash
python run_active_slam.py
```

`run_active_slam.py` dynamically loads:

- `SwarmCapture+/Swarm_Target_Capture+.py` as the simulator;
- `DDFGO++/SwarmDDFGO++.py` as the SLAM module.

The runner:

1. Builds default `ActiveSlamRunnerConfig`, whose `slam_period_seconds` defaults to `shared_config.stride`.
2. Sets `DDFGO++/config.py::save_every_slam_updates` so online SLAM checkpoints align with `shared_config.CHECKPOINT_INTERVAL_SECONDS`.
3. Initializes the simulator.
4. Steps simulation until completion or `max_steps`.
5. Buffers simulator frames.
6. Initializes online SLAM from the first frame.
7. Calls `step_slam()` whenever the elapsed simulated time reaches `slam_period_seconds`.
8. Saves final simulator outputs and returns performance plus the latest SLAM feedback history.

### Current Integration Boundary

The active runner passes `agents_commands` and `slam_feedback` into the simulator API. DDFGO++ returns feedback containing pose estimates, target estimates, map summaries, dense map point arrays, and map-quality fields.

When `DDFGO++/config.py::oracle_growing_map` is `True`, each SLAM update skips iSAM and voxel-accumulates simulator `LandSet` observations into `MergedMapSet` using truth poses and target kinematics (fast controller prototyping).

`build_slam_feedback()` exposes per-agent `merged_map_sets` and `merged_map_shared_sets` (numpy `N×3`). `run_decision_phase()` copies these into each agent as `MergedMapSet` / `MergedMapSharedSet`, runs ellipsoid coverage (`MapCoverageRatio`, `MapExploreDirection`, …), then control runs.

In encapsulation mode (`Mode == 'e'`), `Controllers.Encapsulate()` adds a gated explore attraction force when `MapCoverageRatio` is below `explore_coverage_threshold` (see `SimulationConfig`): a small push along `MapExploreDirection`, scaled by `(threshold - ratio) / threshold`. Set `explore_attraction_gain = 0` to disable. Flocking/antiflocking and LCD pointing are unchanged.

Remaining gaps:

- `build_agents_commands()` currently returns `None`.
- `run_control_phase()` accepts `agents_commands` but discards it.
- Attitude still uses `LCD` via `Control_Frame`; explore direction affects translation only.

So the present online workflow is simulation plus SLAM feedback, oracle map growth, coverage-based explore biasing in encapsulation, and checkpointing.

## Shared Configuration

`shared_config.py` defines the project-level defaults:

- `DT = 240`, used by the 3D simulator as `dt = 1 / DT`;
- `N = 2`, default number of 3D agents;
- `D = 15`, default 3D simulation duration in seconds;
- `object_name = "Orion_Capsule"`;
- `stride = 0.1`, the default online SLAM period in simulated seconds;
- `online_decentralized_comm_period_seconds = 0.05`;
- `CHECKPOINT_INTERVAL_SECONDS = 0.5`;
- `VIS_CUBESAT_SIZE_M = 0.3`;
- shared data paths under `SwarmCapture+/Data` and `Data/Dynamic_Target`.

The shared tag format is:

```text
N<N>_D<D>_dt<DT>_<object_name>
```

For the default config, this is `N2_D15_dt240_Orion_Capsule`.

## 3D Simulator: `SwarmCapture+/`

The primary simulator is `SwarmCapture+/Swarm_Target_Capture+.py`.

### Core Model

The simulator represents a swarm of CubeSat-like agents capturing a loaded target object. It uses:

- PyBullet for physics, rigid bodies, target loading, force/torque application, contact, and docking constraints;
- Open3D point clouds for target geometry and observations;
- ray-cast lidar/cone perception for target landmarks;
- attachment-point detection and bidding for capture-point selection;
- neighborhood sets for communication, collision, flocking, and anti-flocking interactions;
- PID/potential-field style control for pointing, encapsulation, capture, and Clohessy-Wiltshire-Hill force compensation.

Supported target names in the loader include `Turksat`, `Nonconvex`, `Orion_Capsule`, `Motor`, and `Separation_Stage`, each with object-specific URDF/PLY scale settings.

### Simulation Phases

Each simulation step is decomposed into explicit phases:

1. `begin_simulation_step()`
2. per-agent `begin_agent_step()`
3. `run_perception_phase()`
4. `run_communication_phase()`
5. `run_decision_phase()`
6. `run_control_phase()`
7. `run_physics_phase()`
8. `record_simulation_frame()`

The frame returned to online SLAM contains iteration, simulated time, agent true states, target true state, per-agent landmark observations, communication sets, and the target point cloud.

### Simulator Defaults

The `SimulationConfig` dataclass controls gains, number of agents, duration, detection/communication radii, lidar rays, observation noise, force/torque limits, visualization toggles, output paths, and performance weights. Defaults are pulled partly from `shared_config.py`.

Important defaults include:

- `num_agents = shared_config.N`;
- `duration_seconds = shared_config.D`;
- `dt = 1 / shared_config.DT`;
- `feature_observation_std = 0`;
- `num_features = 10`;
- `communication_radius = 300`;
- `detection_radius = 100`;
- `target_velocity = (0.0, 0.0, 0.0)`;
- `target_angular_velocity = 0*(0.005, -0.01, 0.01)`, which evaluates to an empty tuple in Python and should be treated carefully if target rotation is changed.
- `explore_attraction_gain = 0.1` — encapsulation-mode force along `MapExploreDirection` when coverage is low;
- `explore_coverage_threshold = 0.90` — disable explore attraction above this `MapCoverageRatio`.

### 3D Outputs

Final simulator outputs are written under `SwarmCapture+/Data/` by default:

- `simulation_data_<tag>.xlsx`
- `Agents_History_<tag>.pkl`
- `Target_History_<tag>.pkl`
- `Target_PointCloud_<tag>.pkl`
- `Attachment_Points_<tag>.pkl`

The scalar performance score is written to `SwarmCapture+/performance.json` by default. It combines dock reward, inverse fuel, inverse force-spike penalty, contact velocity reward, and pointing reward using `SimulationConfig.performance_weights`.

## DDFGO++ SLAM: `DDFGO++/`

The main SLAM file is `DDFGO++/SwarmDDFGO++.py`, configured by `DDFGO++/config.py`.

### Core Model

DDFGO++ builds and updates factor graphs using GTSAM. It estimates agent poses, sparse landmarks, target center of mass, target velocity, and target angular velocity. It also maintains dense or hybrid merged maps through Open3D point-cloud utilities.

Major components:

- `Custom_Factors.py`: custom landmark, kinematic, center-of-mass, target velocity, and angular velocity factor errors.
- `Feature_Processing.py`: FPFH-like feature descriptors and feature noise helpers.
- `LandmarkRegistry.py`: registry-based landmark identity tracking.
- `map_merging.py`: local scan storage, scan placement, voxel merging, dense map construction, and map-vs-target error metrics.
- `helper.py`: pose noise, measurement conversion, target parameter measurement, covariance, kinematics, angular velocity estimators, graph keys, and smoothing.

### Modes

DDFGO++ supports:

- batch mode, loading saved simulator histories via `config.get_data_paths()`;
- online mode, initialized from live simulator frames and updated by frame buffers from `run_active_slam.py`.

The online mode converts simulator frames into the history shape expected by the batch body, then processes one selected SLAM update frame at a time.

### Notable SLAM Defaults

From `DDFGO++/config.py`:

- `sw = 30` sliding window length;
- `Max_Land = 20`;
- `map_mode = "hybrid"` with sparse landmarks and dense scans;
- `oracle_growing_map = False` — when `True`, skip iSAM and accumulate `LandSet` into `MergedMapSet` with truth poses (controller prototyping);
- `Decentralized = True`;
- `Qn = 2`, `Ql = 5`;
- `Kinem = "n_step_Kinem"`;
- `calculate_w_method = "w4"`;
- `feature_id_namespace_policy = "registry"`;
- `enable_outlier_rejection = True`;
- `use_window_anchor_prior = False`;
- `use_sparse_gps_prior = False`;
- `unc = 0.000001`, which scales most noise settings.

`DDFGO++/SwarmDDFGO++.py` imports `gtsam` directly, so a working GTSAM Python installation is required even though `requirements.txt` leaves it commented.

### SLAM Outputs

DDFGO++ saves result pickles under `DDFGO++/Results/`:

- `Agents_History_<results_tag>.pkl`
- `Target_History_<results_tag>.pkl`

The result tag includes the shared simulation tag, kinematic mode, and decentralized mode, for example:

```text
N2_D15_dt240_Orion_Capsule_n_step_Kinem_Multi_Qn2_Ql5
```

Online checkpoints use `save_every_slam_updates`, aligned by `run_active_slam.py` to the shared checkpoint interval.

## 2D Prototype: `simplified_2d/prototype2d/`

The 2D prototype is a smaller, faster simulator for testing architecture and metrics without PyBullet or the full 3D stack.

Run the default config:

```bash
python -m simplified_2d.prototype2d.simulator
```

The default config is `simplified_2d/prototype2d/config.json`. It currently defines:

- 3 agents;
- `dt = 0.05`;
- 12 seconds duration;
- a moving 2D target;
- a sketch-based target geometry in `sketch.json`;
- search, encapsulate, capture, and dock behavior;
- perception, communication, and actuation delays.

Agent modes are:

- `s`: search;
- `e`: encapsulate;
- `c`: capture;
- `d`: docked.

The prototype records per-step histories and metrics under `simplified_2d/prototype2d/results/<experiment_name>/`, including:

- `agents_history.pkl`
- `target_history.pkl`
- `attachment_points.pkl`
- `metrics_history.pkl`
- `messages_history.pkl`
- `performance.json`

Metrics include convex hull area, distance statistics, total control effort, fuel proxy, mode counts, message age, map size, map coverage ratio, attachment-point conflicts, attachment-point coverage ratio, and capture error.

Animation and plotting entry points:

```bash
python -m simplified_2d.prototype2d.animation --results simplified_2d/prototype2d/results/v1.0 --target simplified_2d/prototype2d/sketch.json
python -m simplified_2d.prototype2d.plotting --results simplified_2d/prototype2d/results/v1.0 --metric convex_hull_area
```

The 2D target sketch tool is:

```bash
python -m simplified_2d.prototype2d.target_sketch_tk -o simplified_2d/prototype2d/sketch.json
```

## Utilities

The `utilities/` folder contains standalone research helpers:

- `plane_ransac.py`: RANSAC plane segmentation and plotting.
- `contact_points.py`: support polygon extraction and contact-point sampling on plane segments.
- `candidate_gossip.py` and `demo_candidate_gossip.py`: decentralized sharing and merging of contact candidates.
- `coverage.py`: ellipsoid patch coverage modeling and visualization.
- `ellipsoid.py`: PCA ellipsoid fitting and projection helpers.
- `mock_data.py`: synthetic satellite, agent, ellipsoid, observation, frontier, and contact-point scenes.

These utilities are not automatically invoked by `run_active_slam.py`.

## Visualization

The [`visualization/`](visualization/) folder contains a lightweight matplotlib animator for DDFGO++ dense maps (no Open3D / PyBullet).

Run from the repository root after a SLAM run has written `DDFGO++/Results/Agents_History_<results_tag>.pkl` (online checkpoints or batch SLAM). Expects `MergedMapSet` on each agent (oracle `oracle_growing_map` or hybrid/dense `map_mode`).

```bash
python visualization/animate_merged_map.py
python visualization/animate_merged_map.py --highlight-agent 1 --agents-pickle DDFGO++/Results/Agents_History_<tag>.pkl
python visualization/animate_merged_map.py --no-save
```

Defaults:

- resolves the agents pickle via `DDFGO++/config.py::get_results_paths()`;
- highlights agent 1 (1-based), draws all agents as oriented wireframe cubes with pointing arrows and motion trails;
- animates the highlighted agent's growing `MergedMapSet` (falls back to `MergedMapSharedSet` with a console warning if local map is empty);
- saves `visualization/output/merged_map_agent<id>.gif` (`--save` on by default; use `--no-save` to preview only).

`animate_coverage.py` reads **SwarmCapture+** sim pickles (`SwarmCapture+/Data/Agents_History*.pkl`) and animates `MapEllipsoid`, patch coverage, `MapExploreDirection` (cyan), and controller pointing via `LCD` (orange) for one highlighted agent.

```bash
python visualization/animate_coverage.py
python visualization/animate_coverage.py --highlight-agent 1 --downsample 5
python visualization/animate_coverage.py --agents-pickle SwarmCapture+/Data/Agents_HistoryN2_D5_dt240_Orion_Capsule.pkl --no-save
```

Defaults:

- resolves the sim agents pickle from `shared_config.get_sim_data_paths()` (also tries `Agents_History<tag>.pkl` without underscore);
- reuses `utilities/coverage/coverage.py` (`add_patch_collection`, `compute_coverage_ratio`, `ellipsoid_point_from_angles`, etc.);
- saves `visualization/output/coverage_agent<id>.gif` (`--save` on by default; use `--no-save` to preview only).

## Known Assumptions and Limitations

- The active scientific target is a moving spacecraft-like object, defaulting to `Orion_Capsule`.
- The 3D simulator's default target motion is effectively static unless target velocity/angular velocity are changed.
- Attachment-point velocity bidding treats velocity alignment as neutral when either velocity is effectively zero and bounds the inverse relative-velocity term away from division by zero.
- The online SLAM integration currently produces feedback but does not yet drive simulator decisions or control.
- The 3D simulator and DDFGO++ depend on path and filename agreement through `shared_config.py`; keep `N`, `D`, `DT`, and `object_name` synchronized when reusing saved data.
- `DDFGO++/config.py::feature_id_namespace_policy` is currently `"registry"`. The code also supports `"source_feature_idx"` and `"auto"`, but changing this affects graph IDs and decentralized map sharing assumptions.
- Noise levels in DDFGO++ are mostly scaled by `unc`; near-zero uncertainty also triggers special behavior such as pose low-pass bypass and optional target-kinematics truth use.
- Several files are script-style research code with module-level global state. Prefer small, verified changes and avoid assuming thread safety or reentrancy.
- Some plotting and animation scripts expect outputs to already exist with the current shared tag.

## Maintenance Rule

Any change to the scientific objective, simulation/SLAM method, experimental assumptions, target model, timing model, noise model, communication model, controller behavior, metrics, or output schema should be followed by an update to this `documentation.md`.
