"""Validated high-level behavior commands for Phase 3 decision backends."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Union

from .model import ExperimentConfig

ALLOWED_BEHAVIORS = frozenset(
    {"search", "hold", "explore", "characterize", "follow", "capture"}
)


@dataclass
class BehaviorCommand:
    behavior: str = "hold"
    params: Dict[str, Any] = field(default_factory=dict)
    target_ap_id: Optional[int] = None
    outbound_message: Optional[str] = None
    message_channel: str = "broadcast"
    message_recipient_id: Optional[int] = None
    ap_reason: Optional[str] = None


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, v))


def _clamp_float(
    x: Any,
    lo: float,
    hi: float,
    default: Optional[float] = None,
) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _clamp_behavior_params(behavior: str, raw_params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(raw_params)
    params["aggressiveness"] = _clamp01(params.get("aggressiveness", 0.5))

    # Shared optional orientation guidance (degrees in [-180, 180]).
    if "target_orientation_deg" in params:
        v = _clamp_float(params.get("target_orientation_deg"), -180.0, 180.0, None)
        if v is None:
            params.pop("target_orientation_deg", None)
        else:
            params["target_orientation_deg"] = v

    if "look_world_xy" in params:
        lw = params.get("look_world_xy")
        if isinstance(lw, (list, tuple)) and len(lw) >= 2:
            x = _clamp_float(lw[0], -1e6, 1e6, None)
            y = _clamp_float(lw[1], -1e6, 1e6, None)
            if x is not None and y is not None:
                params["look_world_xy"] = [x, y]
            else:
                params.pop("look_world_xy", None)
        else:
            params.pop("look_world_xy", None)

    if "search_box_scale" in params:
        s = _clamp_float(params["search_box_scale"], 0.25, 3.0, None)
        if s is None:
            params.pop("search_box_scale", None)
        else:
            params["search_box_scale"] = s

    if behavior == "capture" and "alignment_weight" in params:
        w = _clamp_float(params["alignment_weight"], 0.0, 2.0, None)
        if w is None:
            params.pop("alignment_weight", None)
        else:
            params["alignment_weight"] = w

    return params


def validate_and_clamp(
    raw: Union[BehaviorCommand, Dict[str, Any], None],
    config: ExperimentConfig,
) -> Optional[BehaviorCommand]:
    """
    Validate and clamp fields. Returns None if the payload cannot be interpreted;
    callers should fall back to a safe default (typically hold).
    """
    if raw is None:
        return None

    if isinstance(raw, BehaviorCommand):
        cmd = raw
    elif isinstance(raw, dict):
        beh = raw.get("behavior")
        if beh is not None and not isinstance(beh, str):
            return None
        params = raw.get("params")
        if params is not None and not isinstance(params, dict):
            return None
        tap = raw.get("target_ap_id")
        if tap is not None:
            try:
                tap = int(tap)
            except (TypeError, ValueError):
                return None
        out_msg = raw.get("outbound_message")
        if out_msg is not None and not isinstance(out_msg, str):
            return None
        msg_channel = str(raw.get("message_channel", "broadcast")).lower()
        if msg_channel not in {"broadcast", "direct"}:
            return None
        msg_recipient = raw.get("message_recipient_id")
        if msg_recipient is not None:
            try:
                msg_recipient = int(msg_recipient)
            except (TypeError, ValueError):
                return None
        ap_reason = raw.get("ap_reason")
        if ap_reason is not None and not isinstance(ap_reason, str):
            return None
        cmd = BehaviorCommand(
            behavior=str(beh or "hold").lower(),
            params=dict(params or {}),
            target_ap_id=tap,
            outbound_message=out_msg,
            message_channel=msg_channel,
            message_recipient_id=msg_recipient,
            ap_reason=ap_reason,
        )
    else:
        return None

    if cmd.behavior not in ALLOWED_BEHAVIORS:
        return None

    params = _clamp_behavior_params(cmd.behavior, cmd.params)

    tap = cmd.target_ap_id
    if tap is not None:
        if tap < 0:
            return None

    channel = str(getattr(cmd, "message_channel", "broadcast")).lower()
    if channel not in {"broadcast", "direct"}:
        return None
    msg_recipient = cmd.message_recipient_id
    if msg_recipient is not None and msg_recipient < 0:
        return None
    if channel == "direct" and msg_recipient is None:
        return None
    ap_reason = cmd.ap_reason
    if ap_reason is not None and not isinstance(ap_reason, str):
        return None
    return replace(
        cmd,
        params=params,
        message_channel=channel,
        message_recipient_id=msg_recipient,
        ap_reason=ap_reason,
    )
