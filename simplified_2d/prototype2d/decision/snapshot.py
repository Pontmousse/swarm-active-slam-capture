from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..model import Agent, ExperimentConfig

MAX_SHARED_POINT_IDS = 10
# Cap how many prior decisions appear in LLM payloads (full list trimmed in simulator).
MAX_BEHAVIOR_HISTORY_IN_SNAPSHOT = 16


def _subset_ids(values: List[int], limit: int = MAX_SHARED_POINT_IDS) -> List[int]:
    uniq = {int(v) for v in values}
    return sorted(uniq)[:limit]


def build_decision_snapshot(
    agent: Agent,
    agents: List[Agent],
    config: ExperimentConfig,
    current_time: float,
    attachment_visible_ids: List[int],
    target_center_xy: np.ndarray,
) -> Dict[str, Any]:
    """Compact local observation bundle for decision backends."""
    id_to_pos = {int(pt): np.asarray(meta.get("last_world_position"), dtype=float)
                 for pt, meta in agent.map.items()
                 if isinstance(meta, dict) and meta.get("last_world_position") is not None}
    visible_known_ids = [pid for pid in agent.land_set if int(pid) in id_to_pos]
    visible_centroid = None
    if visible_known_ids:
        pts = [id_to_pos[int(pid)].reshape(2) for pid in visible_known_ids]
        c = np.mean(np.stack(pts, axis=0), axis=0)
        visible_centroid = [float(c[0]), float(c[1])]

    neighbor_heading_mean_deg = None
    if agent.comm_set:
        headings = [float(agents[nid].state[2]) for nid in agent.comm_set if 0 <= int(nid) < len(agents)]
        if headings:
            s = float(np.sum(np.sin(headings)))
            c = float(np.sum(np.cos(headings)))
            neighbor_heading_mean_deg = float(np.degrees(np.arctan2(s, c)))

    n_inbox = len(agent.inbox)
    latest_peers = []
    inbox_messages = []
    for msg in agent.inbox[-5:]:
        payload = msg.get("payload") or {}
        latest_peers.append(
            {
                "sender_id": msg.get("sender_id"),
                "behavior": payload.get("behavior"),
                "target_ap": payload.get("target_ap"),
                "summary": payload.get("summary"),
            }
        )
    for msg in agent.inbox[-10:]:
        payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else {}
        inbox_messages.append(
            {
                "sender_id": msg.get("sender_id"),
                "channel": msg.get("channel"),
                "sent_time": msg.get("sent_time"),
                "content": payload.get("content"),
            }
        )

    return {
        "agent_id": agent.id,
        "time": float(current_time),
        "state": list(agent.state),
        "mode": agent.mode,
        "behavior": agent.behavior,
        "behavior_history": list(
            getattr(agent, "behavior_history", [])[-MAX_BEHAVIOR_HISTORY_IN_SNAPSHOT:]
        ),
        "map_size": len(agent.map),
        "frontier_size": len(agent.map_frontier),
        "known_point_ids": _subset_ids(list(agent.map.keys())),
        "target_ap": agent.target_ap,
        "decision_target_ap_id": getattr(agent, "decision_target_ap_id", None),
        "visible_ap_ids": list(attachment_visible_ids),
        "neighbors": list(agent.comm_set),
        "neighbor_heading_mean_deg": neighbor_heading_mean_deg,
        "visible_known_point_ids": _subset_ids([int(v) for v in visible_known_ids]),
        "visible_points_centroid_xy": visible_centroid,
        "inbox_short": latest_peers,
        "inbox_messages": inbox_messages,
        "inbox_count": n_inbox,
        "target_center_xy": [float(target_center_xy[0]), float(target_center_xy[1])],
        "limits": {
            "decision_period": float(getattr(config, "decision_period", 1.0)),
            "ap_detection_radius": float(config.ap_detection_radius),
            "dock_distance": float(config.dock_distance),
        },
    }
