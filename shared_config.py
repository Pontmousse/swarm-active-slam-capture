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

# Visualization sizing (meters)
# The Cube.obj asset is authored as a 0.2 m cube; this value controls
# rendered animation size consistently across mapping/simulation scripts.
VIS_CUBESAT_SIZE_M = 0.3

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
