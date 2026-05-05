from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class ExperimentConfig:
    name: str
    dt: float
    duration: float
    num_agents: int
    agent_mass: float
    agent_inertia: float
    target_mass: float
    target_inertia: float
    initial_agent_states: List[List[float]]
    initial_target_state: List[float]
    target_json_path: str
    output_root: str
    search_duration: float
    encapsulate_duration: float
    dock_distance: float
    capture_gain: float
    encapsulate_gain: float
    search_gain: float
    damping_gain: float
    fuel_coeff: float
    fov_radius: float = 2.5
    fov_angle: float = 2.0
    normal_visibility_threshold: float = 0.0
    communication_radius: float = 3.0
    ap_detection_radius: float = 1.5
    perception_delay: Dict[str, Any] = field(default_factory=lambda: {"type": "zero", "value": 0.0})
    communication_delay: Dict[str, Any] = field(default_factory=lambda: {"type": "zero", "value": 0.0})
    actuation_delay: Dict[str, Any] = field(default_factory=lambda: {"type": "zero", "value": 0.0})

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Agent:
    id: int
    state: List[float]
    mode: str
    mass: float
    inertia: float
    time_step: float = 0.0
    iteration: int = 0
    comm_set: List[int] = field(default_factory=list)
    inbox: List[Dict[str, Any]] = field(default_factory=list)
    last_messages_by_sender: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    land_set: List[int] = field(default_factory=list)
    map: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    lc: Optional[List[float]] = None
    lcd: Optional[List[float]] = None
    aps: List[int] = field(default_factory=list)
    aps_bids: List[float] = field(default_factory=list)
    target_ap: Optional[int] = None
    action_time: float = 0.0
    control_force: List[float] = field(default_factory=lambda: [0.0, 0.0])
    control_torque: float = 0.0
    fuel_consumed: float = 0.0
    dock_pose: Optional[List[float]] = None
    dock_time: Optional[float] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["state"] = list(self.state)
        data.pop("inbox", None)
        data.pop("last_messages_by_sender", None)
        return data


@dataclass
class TargetState:
    state: List[float]
    mass: float
    inertia: float

    def to_dict(self) -> Dict:
        return {"state": list(self.state), "mass": self.mass, "inertia": self.inertia}


@dataclass
class TargetPoint:
    id: int
    x: float
    y: float
    normal: Optional[List[float]] = None


@dataclass
class AttachmentPoint:
    id: int
    point_id: int
    x: float
    y: float
    normal: Optional[List[float]] = None
    label: Optional[str] = None


@dataclass
class TargetDefinition:
    name: str
    contour_points: List[TargetPoint]
    dense_points: List[TargetPoint]
    attachment_points: List[AttachmentPoint]


@dataclass
class MetricsSnapshot:
    time: float
    convex_hull_area: float
    min_distance: float
    mean_distance: float
    max_distance: float
    control_effort: float
    fuel_consumed_total: float
    mode_counts: Dict[str, int]
    message_age_mean: float = 0.0
    message_age_max: float = 0.0
    map_size_mean: float = 0.0
    map_coverage_ratio: float = 0.0
    ap_conflicts: int = 0
    ap_coverage_ratio: float = 0.0
    capture_error_mean: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

