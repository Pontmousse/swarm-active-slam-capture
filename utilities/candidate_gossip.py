"""
candidate_gossip.py

Simple decentralized gossip-style consensus and temporal memory for contact
point candidates.

Assumption:
    All positions, normals, and parent plane equations are already expressed in
    the same target/map frame. Frame alignment is intentionally out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CandidateMessage:
    agent_id: int
    local_cp_id: int
    position: np.ndarray
    normal: np.ndarray
    confidence: float
    area_support: float
    parent_segment_id: int
    parent_plane_equation: np.ndarray
    step: int


@dataclass
class SharedContactCandidate:
    shared_cp_id: int
    position: np.ndarray
    normal: np.ndarray
    confidence: float
    area_support: float
    parent_plane_equation: np.ndarray
    supporting_agents: set[int] = field(default_factory=set)
    source_ids: list[tuple[int, int]] = field(default_factory=list)
    first_seen_step: int = 0
    last_seen_step: int = 0
    observation_count: int = 0


@dataclass
class CandidateMatchThresholds:
    position_threshold: float = 0.35
    normal_angle_threshold_deg: float = 20.0
    parent_plane_angle_threshold_deg: float = 20.0
    parent_plane_distance_threshold: float = 0.08


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)

    if norm < 1e-12:
        return v

    return v / norm


def angle_between_normals_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    n1 = normalize(n1)
    n2 = normalize(n2)

    dot = float(np.clip(abs(np.dot(n1, n2)), -1.0, 1.0))

    return float(np.degrees(np.arccos(dot)))


def plane_distance_to_point(plane_equation: np.ndarray, point: np.ndarray) -> float:
    plane_equation = np.asarray(plane_equation, dtype=float)
    point = np.asarray(point, dtype=float)

    normal = plane_equation[:3]
    d = float(plane_equation[3])
    normal_norm = np.linalg.norm(normal)

    if normal_norm < 1e-12:
        return float("inf")

    return float(abs(np.dot(normal, point) + d) / normal_norm)


def orient_same_direction(
    reference_normal: np.ndarray,
    new_normal: np.ndarray,
) -> np.ndarray:
    reference_normal = normalize(reference_normal)
    new_normal = normalize(new_normal)

    if np.dot(reference_normal, new_normal) < 0:
        return -new_normal

    return new_normal


def normalize_plane_equation(plane_equation: np.ndarray) -> np.ndarray:
    plane_equation = np.asarray(plane_equation, dtype=float).copy()
    normal_norm = np.linalg.norm(plane_equation[:3])

    if normal_norm < 1e-12:
        return plane_equation

    return plane_equation / normal_norm


def candidate_matches_shared(
    message: CandidateMessage,
    shared: SharedContactCandidate,
    thresholds: CandidateMatchThresholds,
) -> tuple[bool, dict]:
    message_plane = normalize_plane_equation(message.parent_plane_equation)
    shared_plane = normalize_plane_equation(shared.parent_plane_equation)

    position_distance = float(np.linalg.norm(message.position - shared.position))
    normal_angle_deg = angle_between_normals_deg(message.normal, shared.normal)
    parent_plane_angle_deg = angle_between_normals_deg(
        message_plane[:3],
        shared_plane[:3],
    )
    message_to_shared_plane_distance = plane_distance_to_point(
        shared_plane,
        message.position,
    )
    shared_to_message_plane_distance = plane_distance_to_point(
        message_plane,
        shared.position,
    )

    diagnostics = {
        "position_distance": position_distance,
        "normal_angle_deg": normal_angle_deg,
        "parent_plane_angle_deg": parent_plane_angle_deg,
        "message_to_shared_plane_distance": message_to_shared_plane_distance,
        "shared_to_message_plane_distance": shared_to_message_plane_distance,
    }

    is_match = (
        position_distance <= thresholds.position_threshold and
        normal_angle_deg <= thresholds.normal_angle_threshold_deg and
        parent_plane_angle_deg <= thresholds.parent_plane_angle_threshold_deg and
        message_to_shared_plane_distance <= thresholds.parent_plane_distance_threshold and
        shared_to_message_plane_distance <= thresholds.parent_plane_distance_threshold
    )

    return is_match, diagnostics


def merge_message_into_shared(
    shared: SharedContactCandidate,
    message: CandidateMessage,
) -> SharedContactCandidate:
    message_confidence = float(np.clip(message.confidence, 1e-6, 1.0))
    shared_weight = float(max(shared.observation_count, 1)) * max(shared.confidence, 1e-6)
    message_weight = message_confidence
    total_weight = shared_weight + message_weight

    shared.position = (
        shared_weight * shared.position +
        message_weight * message.position
    ) / total_weight

    aligned_normal = orient_same_direction(shared.normal, message.normal)
    shared.normal = normalize(
        shared_weight * shared.normal +
        message_weight * aligned_normal
    )

    shared_plane = normalize_plane_equation(shared.parent_plane_equation)
    message_plane = normalize_plane_equation(message.parent_plane_equation)

    if np.dot(shared_plane[:3], message_plane[:3]) < 0:
        message_plane = -message_plane

    shared.parent_plane_equation = normalize_plane_equation(
        shared_weight * shared_plane +
        message_weight * message_plane
    )

    shared.confidence = min(
        1.0,
        0.5 * shared.confidence + 0.5 * message_confidence + 0.03,
    )
    shared.area_support = max(shared.area_support, float(message.area_support))
    shared.supporting_agents.add(message.agent_id)

    source_id = (message.agent_id, message.local_cp_id)

    if source_id not in shared.source_ids:
        shared.source_ids.append(source_id)

    shared.last_seen_step = max(shared.last_seen_step, int(message.step))
    shared.observation_count += 1

    return shared


class CandidateGossipMap:
    def __init__(self) -> None:
        self.candidates: list[SharedContactCandidate] = []
        self.next_shared_cp_id: int = 0

    def update_with_messages(
        self,
        messages: list[CandidateMessage],
        thresholds: CandidateMatchThresholds,
        current_step: int,
    ) -> None:
        for message in messages:
            self.add_or_merge_message(message, thresholds)

        for candidate in self.candidates:
            candidate.last_seen_step = max(
                candidate.last_seen_step,
                min(candidate.last_seen_step, current_step),
            )

    def add_or_merge_message(
        self,
        message: CandidateMessage,
        thresholds: CandidateMatchThresholds,
    ) -> SharedContactCandidate:
        best_candidate = None
        best_score = float("inf")

        for candidate in self.candidates:
            is_match, diagnostics = candidate_matches_shared(
                message,
                candidate,
                thresholds,
            )

            if not is_match:
                continue

            score = (
                diagnostics["position_distance"] / thresholds.position_threshold +
                diagnostics["normal_angle_deg"] / thresholds.normal_angle_threshold_deg +
                diagnostics["parent_plane_angle_deg"] /
                thresholds.parent_plane_angle_threshold_deg
            )

            if score < best_score:
                best_candidate = candidate
                best_score = score

        if best_candidate is not None:
            return merge_message_into_shared(best_candidate, message)

        new_candidate = SharedContactCandidate(
            shared_cp_id=self.next_shared_cp_id,
            position=np.asarray(message.position, dtype=float).copy(),
            normal=normalize(message.normal),
            confidence=float(np.clip(message.confidence, 0.0, 1.0)),
            area_support=float(message.area_support),
            parent_plane_equation=normalize_plane_equation(
                message.parent_plane_equation,
            ),
            supporting_agents={message.agent_id},
            source_ids=[(message.agent_id, message.local_cp_id)],
            first_seen_step=int(message.step),
            last_seen_step=int(message.step),
            observation_count=1,
        )

        self.next_shared_cp_id += 1
        self.candidates.append(new_candidate)

        return new_candidate

    def decay_unseen(
        self,
        current_step: int,
        max_missed_steps: int = 10,
        decay_rate: float = 0.95,
        min_confidence: float = 0.1,
    ) -> None:
        kept_candidates = []

        for candidate in self.candidates:
            missed_steps = current_step - candidate.last_seen_step

            if missed_steps > 0:
                candidate.confidence *= decay_rate ** missed_steps

            if missed_steps <= max_missed_steps or candidate.confidence >= min_confidence:
                kept_candidates.append(candidate)

        self.candidates = kept_candidates

    def export_messages(self, agent_id: int, step: int) -> list[CandidateMessage]:
        messages = []

        for candidate in self.candidates:
            messages.append(
                CandidateMessage(
                    agent_id=agent_id,
                    local_cp_id=candidate.shared_cp_id,
                    position=candidate.position.copy(),
                    normal=candidate.normal.copy(),
                    confidence=float(candidate.confidence),
                    area_support=float(candidate.area_support),
                    parent_segment_id=-1,
                    parent_plane_equation=candidate.parent_plane_equation.copy(),
                    step=step,
                )
            )

        return messages

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.candidates) == 0:
            return (
                np.empty((0, 3)),
                np.empty((0, 3)),
                np.empty((0,), dtype=int),
            )

        positions = np.array([candidate.position for candidate in self.candidates])
        normals = np.array([candidate.normal for candidate in self.candidates])
        ids = np.array([candidate.shared_cp_id for candidate in self.candidates])

        return positions, normals, ids


def make_candidate_messages(
    agent_id: int,
    contact_points,
    plane_segments,
    step: int,
) -> list[CandidateMessage]:
    segments_by_id = {
        segment.segment_id: segment
        for segment in plane_segments
    }
    messages = []

    for contact_point in contact_points:
        segment = segments_by_id[contact_point.parent_segment_id]

        messages.append(
            CandidateMessage(
                agent_id=agent_id,
                local_cp_id=contact_point.cp_id,
                position=np.asarray(contact_point.position, dtype=float).copy(),
                normal=normalize(contact_point.normal),
                confidence=float(contact_point.confidence),
                area_support=float(contact_point.area_support),
                parent_segment_id=contact_point.parent_segment_id,
                parent_plane_equation=normalize_plane_equation(
                    segment.plane_equation,
                ),
                step=step,
            )
        )

    return messages


if __name__ == "__main__":
    thresholds = CandidateMatchThresholds()
    gossip_map = CandidateGossipMap()

    plane = np.array([0.0, 0.0, 1.0, 0.0])

    messages = [
        CandidateMessage(
            agent_id=0,
            local_cp_id=0,
            position=np.array([1.0, 1.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            confidence=0.7,
            area_support=0.25,
            parent_segment_id=0,
            parent_plane_equation=plane,
            step=1,
        ),
        CandidateMessage(
            agent_id=0,
            local_cp_id=1,
            position=np.array([1.08, 0.95, 0.01]),
            normal=np.array([0.0, 0.02, 1.0]),
            confidence=0.75,
            area_support=0.30,
            parent_segment_id=0,
            parent_plane_equation=plane,
            step=2,
        ),
        CandidateMessage(
            agent_id=1,
            local_cp_id=0,
            position=np.array([0.94, 1.04, -0.01]),
            normal=np.array([0.01, 0.0, 1.0]),
            confidence=0.8,
            area_support=0.28,
            parent_segment_id=2,
            parent_plane_equation=plane,
            step=2,
        ),
        CandidateMessage(
            agent_id=1,
            local_cp_id=1,
            position=np.array([3.0, 3.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            confidence=0.65,
            area_support=0.25,
            parent_segment_id=3,
            parent_plane_equation=plane,
            step=2,
        ),
    ]

    gossip_map.update_with_messages(
        messages=messages,
        thresholds=thresholds,
        current_step=2,
    )

    print("Shared candidates:")

    for candidate in gossip_map.candidates:
        print(
            f"  shared_cp_id={candidate.shared_cp_id} "
            f"position={np.round(candidate.position, 3)} "
            f"confidence={candidate.confidence:.3f} "
            f"agents={sorted(candidate.supporting_agents)} "
            f"sources={candidate.source_ids} "
            f"observations={candidate.observation_count}"
        )
