"""
Shared project-level configuration for values used by both simulation and SLAM.
Keep this file focused on cross-cutting identity/runtime defaults.
"""

from pathlib import Path

# Common simulation identity
DT = 240
N = 2
D = 15
object_name = "Orion_Capsule"

stride = 0.1  # seconds per simulation step; controls SLAM frequency and sim time scaling

# Online DDFGO++ only: target simulated seconds between decentralized comm attempts.
# Converted to SLAM-update stride as every = max(1, floor(period / stride)).
# Lower = more frequent (e.g. 0.05 with stride 0.1 => every SLAM update for debugging).
online_decentralized_comm_period_seconds = 0.05

# Checkpoint cadence (simulated seconds)
# - SwarmCapture+ writes Excel + pickle histories when sim time crosses this interval.
# - Online DDFGO++ checkpoint pickles every N SLAM updates, with N chosen so that
#   N * slam_period_seconds ≈ CHECKPOINT_INTERVAL_SECONDS (see slam_checkpoint_every_updates).
CHECKPOINT_INTERVAL_SECONDS = 0.5


def slam_checkpoint_every_updates(slam_period_seconds: float) -> int:
    """How many SLAM periods fit in CHECKPOINT_INTERVAL_SECONDS (>= 1 update)."""
    if slam_period_seconds <= 0.0:
        raise ValueError("slam_period_seconds must be > 0")
    return max(1, int(round(CHECKPOINT_INTERVAL_SECONDS / slam_period_seconds)))

# Visualization sizing (meters)
# The Cube.obj asset is authored as a 0.2 m cube; this value controls
# rendered animation size consistently across mapping/simulation scripts.
VIS_CUBESAT_SIZE_M = 0.3

# Offscreen mapping animations (Open3D look_at(center, eye, up)): shared fixed eye in world frame.
# Both Animate_Single_Agent_Mapping_Offscreen.py and Animate_All_Agents_Mapping_Offscreen.py use this.
animation_camera_eye_xyz = (-12.0, 3.0, 3.0)

PROJECT_ROOT = Path(__file__).resolve().parent
SWARMCAPTURE_DIR = PROJECT_ROOT / "SwarmCapture+"
SWARMCAPTURE_DATA_DIR = SWARMCAPTURE_DIR / "Data"
DDFGO_DATA_DIR = PROJECT_ROOT / "Data" / "Dynamic_Target"


def get_tag(n=N, d=D, dt=DT, name=object_name):
    return f"N{int(n)}_D{int(d)}_dt{int(dt)}_{name}"


def get_sim_data_paths(n=N, d=D, dt=DT, name=object_name):
    tag = get_tag(n=n, d=d, dt=dt, name=name)
    return {
        "tag": tag,
        "agents": str(SWARMCAPTURE_DATA_DIR / f"Agents_History_{tag}.pkl"),
        "target": str(SWARMCAPTURE_DATA_DIR / f"Target_History_{tag}.pkl"),
        "target_pcd": str(SWARMCAPTURE_DATA_DIR / f"Target_PointCloud_{tag}.pkl"),
        "attachment_points": str(SWARMCAPTURE_DATA_DIR / f"Attachment_Points_{tag}.pkl"),
        "excel": str(SWARMCAPTURE_DATA_DIR / f"simulation_data_{tag}.xlsx"),
    }
