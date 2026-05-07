from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from ..behavior_command import BehaviorCommand
from ..model import ExperimentConfig

ALLOWED_BEHAVIORS = ("search", "hold", "explore", "characterize", "follow", "capture")


def _params_from_flat_args(args: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if args.get("aggressiveness") is not None:
        params["aggressiveness"] = args.get("aggressiveness")
    if args.get("target_orientation_deg") is not None:
        params["target_orientation_deg"] = args.get("target_orientation_deg")
    if args.get("look_world_x") is not None and args.get("look_world_y") is not None:
        params["look_world_xy"] = [args.get("look_world_x"), args.get("look_world_y")]
    if args.get("search_box_scale") is not None:
        params["search_box_scale"] = args.get("search_box_scale")
    if args.get("alignment_weight") is not None:
        params["alignment_weight"] = args.get("alignment_weight")
    return params


def _command_from_action(
    action_name: str,
    args: Dict[str, Any],
) -> Optional[BehaviorCommand]:
    if action_name == "set_behavior":
        behavior = str(args.get("behavior", "hold")).lower()
        if behavior not in ALLOWED_BEHAVIORS:
            return None
        params = _params_from_flat_args(args)
        tap = args.get("target_ap_id")
        out = args.get("outbound_message")
        decision_reason = args.get("decision_reason")
        msg_channel = str(args.get("message_channel", "broadcast")).lower()
        if msg_channel not in {"broadcast", "direct"}:
            msg_channel = "broadcast"
        msg_recipient = args.get("message_recipient_id")
        try:
            tap_i = int(tap) if tap is not None else None
        except (TypeError, ValueError):
            tap_i = None
        try:
            msg_recipient_i = int(msg_recipient) if msg_recipient is not None else None
        except (TypeError, ValueError):
            msg_recipient_i = None
        out_text = str(out).strip() if out is not None else ""
        reason_text = str(decision_reason).strip() if decision_reason is not None else ""
        chosen_message = out_text or reason_text
        if not chosen_message:
            return None
        return BehaviorCommand(
            behavior=behavior,
            params=dict(params),
            target_ap_id=tap_i,
            outbound_message=chosen_message,
            message_channel=msg_channel,
            message_recipient_id=msg_recipient_i,
            ap_reason=str(args.get("ap_reason")) if args.get("ap_reason") is not None else None,
        )
    return None


def _merge_optional_actions(
    base: BehaviorCommand,
    ap_cmd: Optional[Dict[str, Any]],
    msg_cmd: Optional[Dict[str, Any]],
) -> BehaviorCommand:
    if ap_cmd is not None:
        ap_id = ap_cmd.get("target_ap_id")
        try:
            base.target_ap_id = int(ap_id) if ap_id is not None else None
        except (TypeError, ValueError):
            base.target_ap_id = None
        reason = ap_cmd.get("ap_reason")
        base.ap_reason = str(reason) if reason is not None else None
    if msg_cmd is not None:
        channel = str(msg_cmd.get("message_channel", "broadcast")).lower()
        base.message_channel = "direct" if channel == "direct" else "broadcast"
        rid = msg_cmd.get("message_recipient_id")
        try:
            base.message_recipient_id = int(rid) if rid is not None else None
        except (TypeError, ValueError):
            base.message_recipient_id = None
        content = msg_cmd.get("outbound_message")
        base.outbound_message = str(content) if content is not None else None
    return base


class FSMBackend:
    """Deterministic behavior timeline for a full mission run."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    def decide(self, agent_id: int, snapshot: Dict[str, Any]) -> BehaviorCommand:
        cfg = self._config
        t = float(snapshot.get("time", 0.0))

        t_s = float(cfg.behavior_search_until)
        t_c = float(cfg.behavior_characterize_until)
        t_cap = float(cfg.behavior_capture_until)
        if t < t_s:
            beh = "search"
        elif t < t_c:
            beh = "explore"
        elif t < t_cap:
            beh = "capture"
        else:
            beh = "characterize"
        params: Dict[str, Any] = {"aggressiveness": 0.65}
        if beh == "search":
            params["target_orientation_deg"] = 0.0
        return BehaviorCommand(behavior=beh, params=params)


class RandomBackend:
    """Legal behavior stress test."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._rng = random.Random(int(getattr(config, "rng_seed", 0)))

    def decide(self, agent_id: int, snapshot: Dict[str, Any]) -> BehaviorCommand:
        cfg = self._config
        choices = ["search", "hold", "explore", "characterize", "follow", "capture"]
        beh = self._rng.choice(choices)
        tap: Optional[int] = None
        if beh == "capture":
            vis = snapshot.get("visible_ap_ids") or []
            if vis:
                tap = int(self._rng.choice(vis))
        return BehaviorCommand(
            behavior=beh,
            params={
                "aggressiveness": self._rng.uniform(0.3, 1.0),
                "target_orientation_deg": self._rng.uniform(-180.0, 180.0),
            },
            target_ap_id=tap,
            outbound_message=f"random:{beh}",
        )


class OpenAIBackend:
    """OpenAI Agents SDK tool-calling backend; falls back to hold on errors."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self.last_usage: Optional[Dict[str, int]] = None
        self.last_trace: Optional[Dict[str, Any]] = None

    def decide(self, agent_id: int, snapshot: Dict[str, Any]) -> BehaviorCommand:
        self.last_usage = None
        model = getattr(self._config, "openai_model", "gpt-4o-mini")
        system = ""
        system += "You are a spacecraft swarm policy. Use tools to inspect observations, then call \\n"
        system += "set_behavior exactly once and optionally assign_ap/send_message.\\n"
        system += "Tool budget: at most 3 observation calls total, at most 1 communication call \\n"
        system += "(send_message), and exactly 1 mandatory set_behavior call. \\n"
        system += "snapshot.behavior_history lists recent commits (newest last): time, previous_behavior, behavior, optional rationale_preview—use alongside snapshot.behavior. \\n"
        system += "It is acceptable to choose the current behavior from the history, if it is still valid and relevant to the snapshot. Actually, we want to be very sure before we trigger a new behavior, especially the 'capture' behavior.\\n"
        system += "Every set_behavior must include a non-empty rationale in outbound_message or decision_reason. \\n"
        system += "You must end with finish_decision. Prefer simple decisions and concise messages.\\n"

        system += "**Behavior guide (choose one that fits the snapshot):** \\n"
        system += "hold = damp motion and hold position when waiting or stabilizing; do not use hold as a substitute for refining known target geometry. To be used mainly at the beginning of the mission, or when the target is not in sight. and while waiting for instructions. \\n"
        system += "search = broad motion when the target or map is still completely unknown. To be used when the target is not in sight, and when there are no visible points in sight. \\n"
        system += "follow = coordinate with neighbors listed in the snapshot (relative motion / headings); you have no oracle about whether a peer 'knows' the target except via inbox content. To be used when you are following a neighbouring agent that knows where the target is. \\n"
        system += "explore = when frontier_size is substantial, bias motion to expand the known-map boundary. To be used when the frontier or when there are many neighbouring agents, and you need to explore unknown areas elsewhere. \\n"
        system += "characterize = when you already observe target structure (map / visible points) and the frontier is small or you need local refinement—prefer this over hold for 'nothing new on frontier' situations. To be used when you are near the target, and you need to refine the local geometry and details or to simply hover around the target and achieve a stable position or collective encapsulation.\\n"        
        system += "capture = approach and align toward docking; use assign_ap/tool args when committing to a visible attachment point. To be used when you are near the target, and you have assigned an attachment point and you need to capture the target now. We aim to synchronize all agents to capture the target at the same time and achieve good surrounding coverage. And we also want this behavior to be triggered only once for each capture attempt. \\n"

        user_payload: Dict[str, Any] = {"agent_id": agent_id, "snapshot": snapshot}
        self.last_trace = {
            "backend": "openai_agents_sdk",
            "agent_id": int(agent_id),
            "model": str(model),
            "system_prompt": system,
            "user_payload": user_payload,
            "response_json": {},
            "raw_content": None,
            "usage": {},
            "latency_sec": 0.0,
            "error": False,
        }
        t0 = time.perf_counter()
        try:
            from agents import Agent as SDKAgent, Runner, function_tool  # type: ignore
        except ImportError:
            return BehaviorCommand(behavior="hold", params={"aggressiveness": 0.5})

        action_log: List[Tuple[str, Dict[str, Any]]] = []
        obs_calls = 0

        def _record_action(name: str, payload: Dict[str, Any]) -> str:
            action_log.append((name, dict(payload)))
            return f"ok:{name}"

        @function_tool
        def observe_local_state() -> str:
            nonlocal obs_calls
            obs_calls += 1
            data = {
                "state": snapshot.get("state"),
                "behavior": snapshot.get("behavior"),
                "mode": snapshot.get("mode"),
                "time": snapshot.get("time"),
            }
            return _record_action("observe_local_state", data)

        @function_tool
        def observe_visibility() -> str:
            nonlocal obs_calls
            obs_calls += 1
            data = {
                "visible_ap_ids": snapshot.get("visible_ap_ids"),
                "visible_known_point_ids": snapshot.get("visible_known_point_ids"),
                "visible_points_centroid_xy": snapshot.get("visible_points_centroid_xy"),
            }
            return _record_action("observe_visibility", data)

        @function_tool
        def observe_frontier() -> str:
            nonlocal obs_calls
            obs_calls += 1
            data = {
                "frontier_size": snapshot.get("frontier_size"),
                "known_point_ids": snapshot.get("known_point_ids"),
            }
            return _record_action("observe_frontier", data)

        @function_tool
        def observe_attachment_points() -> str:
            nonlocal obs_calls
            obs_calls += 1
            data = {
                "visible_ap_ids": snapshot.get("visible_ap_ids"),
                "target_ap": snapshot.get("target_ap"),
            }
            return _record_action("observe_attachment_points", data)

        @function_tool
        def observe_inbox() -> str:
            nonlocal obs_calls
            obs_calls += 1
            data = {
                "inbox_count": snapshot.get("inbox_count"),
                "inbox_messages": snapshot.get("inbox_messages"),
            }
            return _record_action("observe_inbox", data)

        @function_tool
        def set_behavior(
            behavior: str = "hold",
            aggressiveness: float = 0.5,
            target_orientation_deg: Optional[float] = None,
            look_world_x: Optional[float] = None,
            look_world_y: Optional[float] = None,
            search_box_scale: Optional[float] = None,
            alignment_weight: Optional[float] = None,
            message_channel: str = "broadcast",
            message_recipient_id: Optional[int] = None,
            outbound_message: Optional[str] = None,
            decision_reason: Optional[str] = None,
            target_ap_id: Optional[int] = None,
            ap_reason: Optional[str] = None,
        ) -> str:
            return _record_action(
                "set_behavior",
                {
                    "behavior": behavior,
                    "aggressiveness": aggressiveness,
                    "target_orientation_deg": target_orientation_deg,
                    "look_world_x": look_world_x,
                    "look_world_y": look_world_y,
                    "search_box_scale": search_box_scale,
                    "alignment_weight": alignment_weight,
                    "target_ap_id": target_ap_id,
                    "ap_reason": ap_reason,
                    "message_channel": message_channel,
                    "message_recipient_id": message_recipient_id,
                    "outbound_message": outbound_message,
                    "decision_reason": decision_reason,
                },
            )

        @function_tool
        def assign_ap(
            target_ap_id: Optional[int] = None,
            ap_reason: Optional[str] = None,
        ) -> str:
            return _record_action(
                "assign_ap",
                {
                    "target_ap_id": target_ap_id,
                    "ap_reason": ap_reason,
                },
            )

        @function_tool
        def send_message(
            message_channel: str = "broadcast",
            outbound_message: Optional[str] = None,
            message_recipient_id: Optional[int] = None,
        ) -> str:
            return _record_action(
                "send_message",
                {
                    "message_channel": message_channel,
                    "message_recipient_id": message_recipient_id,
                    "outbound_message": outbound_message,
                },
            )

        @function_tool
        def finish_decision() -> str:
            return _record_action("finish_decision", {})

        sdk_agent = SDKAgent(
            name="SwarmPolicyAgent",
            model=str(model),
            instructions=system,
            tools=[
                observe_local_state,
                observe_visibility,
                observe_frontier,
                observe_attachment_points,
                observe_inbox,
                set_behavior,
                assign_ap,
                send_message,
                finish_decision,
            ],
        )

        try:
            result = Runner.run_sync(
                sdk_agent,
                json.dumps(user_payload),
                max_turns=6,
            )
            raw = getattr(result, "final_output", "")
            data = {"actions": action_log}
            if self.last_trace is not None:
                self.last_trace["raw_content"] = str(raw)
                self.last_trace["response_json"] = data if isinstance(data, dict) else {}
            usage = getattr(result, "usage", None)
            if usage is not None:
                pu = getattr(usage, "prompt_tokens", None)
                cu = getattr(usage, "completion_tokens", None)
                tu = getattr(usage, "total_tokens", None)
                self.last_usage = {
                    "prompt_tokens": int(pu or 0),
                    "completion_tokens": int(cu or 0),
                    "total_tokens": int(tu or 0),
                }
                if self.last_trace is not None:
                    self.last_trace["usage"] = dict(self.last_usage)
        except Exception:
            if self.last_trace is not None:
                self.last_trace["error"] = True
                self.last_trace["latency_sec"] = float(time.perf_counter() - t0)
            return BehaviorCommand(behavior="hold", params={"aggressiveness": 0.5})

        if self.last_trace is not None:
            self.last_trace["latency_sec"] = float(time.perf_counter() - t0)
        primary: Optional[BehaviorCommand] = None
        ap_cmd: Optional[Dict[str, Any]] = None
        msg_cmd: Optional[Dict[str, Any]] = None
        finished = False
        observation_calls = 0
        set_behavior_calls = 0
        send_message_calls = 0
        for name, args in action_log:
            if name.startswith("observe_"):
                observation_calls += 1
            if name == "set_behavior":
                set_behavior_calls += 1
                cmd = _command_from_action(name, args)
                if cmd is not None:
                    primary = cmd
            elif name == "assign_ap":
                ap_cmd = dict(args)
            elif name == "send_message":
                send_message_calls += 1
                msg_cmd = dict(args)
            elif name == "finish_decision":
                finished = True

        if self.last_trace is not None:
            self.last_trace["response_json"] = {
                "actions": action_log,
                "finished": finished,
                "observation_calls": observation_calls,
                "set_behavior_calls": set_behavior_calls,
                "send_message_calls": send_message_calls,
            }

        if (
            primary is None
            or not finished
            or set_behavior_calls != 1
            or send_message_calls > 1
            or len(action_log) > 8
        ):
            primary = BehaviorCommand(behavior="hold", params={"aggressiveness": 0.5})
            if self.last_trace is not None:
                self.last_trace["error"] = True
                self.last_trace["response_json"]["fallback_reason"] = (
                    "missing_set_behavior_or_finish_or_action_shape_invalid_or_reason_missing"
                )
        return _merge_optional_actions(primary, ap_cmd, msg_cmd)
