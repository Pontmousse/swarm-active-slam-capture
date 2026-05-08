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
    dock_distance: float
    capture_gain: float
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
    # Always-on swarm shell
    flk_potential: float = 1.0
    flk_radius: float = 2.0
    antflk_potential: float = 2.0
    antflk_radius: float = 0.45
    swarm_damping_gain: float = 0.35
    search_box: List[float] = field(default_factory=lambda: [-3.0, 3.0, -3.0, 3.0])
    search_bounce_gain: float = 120.0
    search_lateral_noise: float = 0.25
    hold_velocity_gain: float = 1.5
    hold_torque_gain: float = 0.5
    target_standoff_distance: float = 0.15
    target_standoff_gain: float = 40.0
    shell_force_cap: float = 250.0
    # Mapping / coordination
    dense_boundary_closed: bool = True
    characterize_gain: float = 1.0
    characterize_standoff: float = 0.12
    follow_gain: float = 1.1
    characterize_fallback: str = "com"
    pointing_kp: float = 1.5
    pointing_kd: float = 0.6
    pointing_tau_cap: float = 0.9
    behavior_search_until: float = 4.0
    behavior_characterize_until: float = 12.0
    behavior_capture_until: float = 22.0
    hint_demo_broadcast_agent_id: int = 0
    hint_demo_min_map_points: int = 8
    hint_demo_point_ids: List[int] = field(default_factory=list)
    # Capture / dock / decision
    capture_alignment_gain: float = 0.0
    dock_max_rel_speed: float = 1.5
    dock_heading_dot_threshold: float = 0.85
    dock_fallback_behavior: str = "capture"
    # Preferred config key for backend selection; aliases to decision_backend.
    llm: Optional[str] = None
    decision_backend: str = "fsm"
    decision_period: float = 1.0
    openai_model: str = "gpt-4o-mini"
    message_summary_period_steps: int = 1
    # Phase 4: reproducibility + evaluation harness
    rng_seed: int = 0

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
    behavior: str = "search"
    map_frontier: List[int] = field(default_factory=list)
    last_decision_time: float = -1.0
    behavior_aggressiveness: float = 0.5
    behavior_params: Dict[str, Any] = field(default_factory=dict)
    decision_backend_last: str = ""
    llm_call_count: int = 0
    decision_target_ap_id: Optional[int] = None
    outbound_message: Optional[str] = None
    outbound_channel: str = "broadcast"
    outbound_recipient_id: Optional[int] = None
    outbound_reason: Optional[str] = None
    #: Recent decisions: {time, previous_behavior, behavior, target_ap_id, rationale_preview}
    behavior_history: List[Dict[str, Any]] = field(default_factory=list)

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
    dense_point_ids_ordered: List[int] = field(default_factory=list)
    dense_adjacency: Dict[int, List[int]] = field(default_factory=dict)


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
    behavior_counts: Dict[str, int] = field(default_factory=dict)
    min_inter_agent_distance: float = 0.0
    frontier_size_mean: float = 0.0
    global_frontier_union_count: int = 0
    decision_calls_step: int = 0
    decision_invalid_step: int = 0
    decision_latency_sum_sec_step: float = 0.0
    llm_calls_step: int = 0
    convex_hull_perimeter: float = 0.0
    target_center_inside_hull: float = 0.0
    min_agent_boundary_distance: float = 0.0
    frontier_coverage_ratio: float = 0.0
    llm_prompt_tokens_step: int = 0
    llm_completion_tokens_step: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

