import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SIM_MODULE_PATH = PROJECT_ROOT / "SwarmCapture+" / "Swarm_Target_Capture+.py"
DDFGO_MODULE_PATH = PROJECT_ROOT / "DDFGO++" / "SwarmDDFGO++.py"


@dataclass
class ActiveSlamRunnerConfig:
    slam_stride_steps: int = 240
    slam_period_seconds: float | None = None
    max_steps: int | None = None


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
    if runner_config.slam_period_seconds is not None:
        elapsed = simulation_frame["sim_time"] - last_slam_time
        return elapsed >= runner_config.slam_period_seconds

    stride = max(1, int(runner_config.slam_stride_steps))
    return simulation_frame["iteration"] % stride == 0


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

            simulation_frame = sim.step_simulation(sim_state, agents_commands=None)
            frame_buffer.append(simulation_frame)

            if slam_state is None:
                slam_state = ddfgo.initialize_slam_online(
                    simulation_frame,
                    config_module=ddfgo_config_module,
                )
                latest_slam_feedback = slam_state.get("latest_slam_feedback")
                if latest_slam_feedback is not None:
                    slam_feedback_history.append(latest_slam_feedback)
                frame_buffer.clear()
                last_slam_time = simulation_frame["sim_time"]
                continue

            if should_run_slam(simulation_frame, runner_config, last_slam_time):
                latest_slam_feedback = ddfgo.step_slam(slam_state, frame_buffer)
                slam_feedback_history.append(latest_slam_feedback)
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
