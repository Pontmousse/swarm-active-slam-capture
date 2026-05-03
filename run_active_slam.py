import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
import shared_config


PROJECT_ROOT = Path(__file__).resolve().parent
SIM_MODULE_PATH = PROJECT_ROOT / "SwarmCapture+" / "Swarm_Target_Capture+.py"
DDFGO_MODULE_PATH = PROJECT_ROOT / "DDFGO++" / "SwarmDDFGO++.py"


@dataclass
class ActiveSlamSchedule:
    perception_delay_steps: int = 0
    communication_delay_steps: int = 0
    decision_delay_steps: int = 0
    control_delay_steps: int = 0
    physics_delay_steps: int = 0
    slam_delay_steps: int = 0


@dataclass
class ActiveSlamRunnerConfig:
    slam_period_seconds: float = shared_config.stride
    max_steps: int | None = None
    schedule: ActiveSlamSchedule = field(default_factory=ActiveSlamSchedule)


def load_module_from_path(module_name, module_path):
    module_path = Path(module_path).resolve()
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def should_run_slam(simulation_frame, runner_config, last_slam_time):
    period = float(runner_config.slam_period_seconds)
    if period <= 0.0:
        raise ValueError("slam_period_seconds must be > 0.")
    elapsed = simulation_frame["sim_time"] - last_slam_time
    return elapsed >= period


def build_agents_commands(schedule, simulation_frame=None, slam_feedback=None):
    _ = schedule, simulation_frame, slam_feedback
    return None


def run_simulation_phase(sim, sim_state, agents_commands=None, slam_feedback=None):
    return sim.step_simulation(
        sim_state,
        agents_commands=agents_commands,
        slam_feedback=slam_feedback,
    )


def initialize_slam_phase(ddfgo, simulation_frame, config_module):
    return ddfgo.initialize_slam_online(
        simulation_frame,
        config_module=config_module,
    )


def run_slam_phase(ddfgo, slam_state, frame_buffer):
    return ddfgo.step_slam(slam_state, frame_buffer)


def record_slam_feedback(slam_feedback_history, slam_feedback):
    if slam_feedback is not None:
        slam_feedback_history.append(slam_feedback)


def run_active_slam(
    sim_config=None,
    ddfgo_config_module=None,
    runner_config=None,
):
    runner_config = runner_config or ActiveSlamRunnerConfig()
    sim = load_module_from_path("swarm_active_capture_sim", SIM_MODULE_PATH)
    ddfgo = load_module_from_path("swarm_active_capture_ddfgo", DDFGO_MODULE_PATH)

    if ddfgo_config_module is None:
        ddfgo_config_module = ddfgo.config

    # Match SLAM checkpoint spacing (in simulated time) to shared CHECKPOINT_INTERVAL_SECONDS.
    ddfgo_config_module.save_every_slam_updates = shared_config.slam_checkpoint_every_updates(
        float(runner_config.slam_period_seconds)
    )

    sim_state = sim.initialize_simulation(config=sim_config)
    frame_buffer = []
    slam_feedback_history = []
    latest_slam_feedback = None
    slam_state = None
    last_slam_time = 0.0

    try:
        max_steps = runner_config.max_steps
        while sim_state["current_iteration"] < sim_state["num_iter"]:
            if max_steps is not None and sim_state["current_iteration"] >= max_steps:
                break

            agents_commands = build_agents_commands(
                runner_config.schedule,
                slam_feedback=latest_slam_feedback,
            )
            simulation_frame = run_simulation_phase(
                sim,
                sim_state,
                agents_commands=agents_commands,
                slam_feedback=latest_slam_feedback,
            )
            frame_buffer.append(simulation_frame)

            if slam_state is None:
                slam_state = initialize_slam_phase(ddfgo, simulation_frame, ddfgo_config_module)
                latest_slam_feedback = slam_state.get("latest_slam_feedback")
                record_slam_feedback(slam_feedback_history, latest_slam_feedback)
                frame_buffer.clear()
                last_slam_time = simulation_frame["sim_time"]
                continue

            if should_run_slam(simulation_frame, runner_config, last_slam_time):
                latest_slam_feedback = run_slam_phase(ddfgo, slam_state, frame_buffer)

                record_slam_feedback(slam_feedback_history, latest_slam_feedback)
                frame_buffer.clear()
                last_slam_time = simulation_frame["sim_time"]

            if simulation_frame.get("done"):
                break

        sim.save_simulation_outputs(sim_state)
        performance = sim.compute_performance_metrics(sim_state)
        return {
            "performance": performance,
            "sim_tag": sim_state["tag"],
            "sim_data_dir": sim_state["paths"]["data_dir"],
            "latest_slam_feedback": latest_slam_feedback,
            "slam_feedback_history": slam_feedback_history,
            "slam_updates": len(slam_feedback_history),
        }
    finally:
        sim.teardown_simulation(sim_state)


def main():
    return run_active_slam()


if __name__ == "__main__":
    main()
