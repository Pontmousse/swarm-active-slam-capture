#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional


def _safe(value: Any) -> str:
    return "null" if value is None else str(value)


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def format_message(msg: Dict[str, Any], idx: int) -> str:
    sent = msg.get("sent_time")
    deliver = msg.get("deliver_time")
    sender = msg.get("sender_id")
    recipient = msg.get("recipient_id")
    channel = msg.get("channel")
    mtype = msg.get("message_type")
    mid = msg.get("message_id")
    payload = msg.get("payload") or {}
    metadata = msg.get("metadata") or {}

    lines = []
    lines.append(f"### {idx}. `{mid}`")
    lines.append("")
    lines.append(f"- from: `{_safe(sender)}`")
    lines.append(f"- to: `{_safe(recipient)}`")
    lines.append(f"- sent_time: `{_safe(sent)}`")
    lines.append(f"- deliver_time: `{_safe(deliver)}`")
    lines.append(f"- channel: `{_safe(channel)}`")
    lines.append(f"- type: `{_safe(mtype)}`")
    lines.append("")
    lines.append("**payload**")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(payload, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    if metadata:
        lines.append("**metadata**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(metadata, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def write_messages_markdown(
    messages: List[Dict[str, Any]],
    output_md_path: str,
    source_path: str = "messages_history.pkl",
) -> str:
    header = [
        "# Messages History",
        "",
        f"- source: `{source_path}`",
        f"- generated_at: `{datetime.utcnow().isoformat()}Z`",
        f"- message_count: `{len(messages)}`",
        "",
        "---",
        "",
    ]

    chunks = ["\n".join(header)]
    for i, msg in enumerate(messages, start=1):
        if not isinstance(msg, dict):
            continue
        chunks.append(format_message(msg, i))
        chunks.append("\n---\n")
    _write_text(output_md_path, "\n".join(chunks))
    return output_md_path


def _filter_inbox_for_agent(
    messages: List[Dict[str, Any]], agent_id: Optional[int]
) -> List[Dict[str, Any]]:
    if agent_id is None:
        return messages
    return [
        msg
        for msg in messages
        if isinstance(msg, dict) and int(msg.get("recipient_id", -1)) == int(agent_id)
    ]


def format_prompt_trace(trace: Dict[str, Any], idx: int) -> str:
    lines = []
    lines.append(f"### {idx}. agent={_safe(trace.get('agent_id'))} t={_safe(trace.get('time'))}")
    lines.append("")
    lines.append(f"- backend: `{_safe(trace.get('backend'))}`")
    lines.append(f"- model: `{_safe(trace.get('model'))}`")
    lines.append(f"- latency_sec: `{_safe(trace.get('latency_sec'))}`")
    lines.append(f"- error: `{_safe(trace.get('error'))}`")
    lines.append(f"- raw_content: `{_safe(trace.get('raw_content'))}`")
    lines.append("")
    lines.append("**system_prompt**")
    lines.append("")
    lines.append("```text")
    lines.append(str(trace.get("system_prompt", "")))
    lines.append("```")
    lines.append("")
    lines.append("**user_payload**")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(trace.get("user_payload", {}), indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("**response_json**")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(trace.get("response_json", {}), indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    usage = trace.get("usage") or {}
    if usage:
        lines.append("**usage**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(usage, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def write_prompt_traces_markdown(
    prompt_traces: List[Dict[str, Any]],
    output_md_path: str,
    source_path: str = "prompt_traces.pkl",
) -> str:
    header = [
        "# Prompt Traces",
        "",
        f"- source: `{source_path}`",
        f"- generated_at: `{datetime.utcnow().isoformat()}Z`",
        f"- trace_count: `{len(prompt_traces)}`",
        "",
        "---",
        "",
    ]
    chunks = ["\n".join(header)]
    for i, trace in enumerate(prompt_traces, start=1):
        if not isinstance(trace, dict):
            continue
        chunks.append(format_prompt_trace(trace, i))
        chunks.append("\n---\n")
    _write_text(output_md_path, "\n".join(chunks))
    return output_md_path


def export_from_results_dir(results_dir: str, agent_id: Optional[int] = None) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    msg_pkl = os.path.join(results_dir, "messages_history.pkl")
    if os.path.exists(msg_pkl):
        messages = _load_pickle(msg_pkl)
        if isinstance(messages, list):
            filtered_messages = _filter_inbox_for_agent(messages, agent_id)
            msg_name = (
                f"messages_inbox_agent_{int(agent_id)}.md"
                if agent_id is not None
                else "messages_history.md"
            )
            outputs["messages_md"] = write_messages_markdown(
                filtered_messages,
                os.path.join(results_dir, msg_name),
                source_path=msg_pkl,
            )

    traces_pkl = os.path.join(results_dir, "prompt_traces.pkl")
    if os.path.exists(traces_pkl):
        traces = _load_pickle(traces_pkl)
        if isinstance(traces, list):
            filtered_traces = traces
            if agent_id is not None:
                filtered_traces = [
                    tr
                    for tr in traces
                    if isinstance(tr, dict) and int(tr.get("agent_id", -1)) == int(agent_id)
                ]
            trace_name = (
                f"prompt_traces_agent_{int(agent_id)}.md"
                if agent_id is not None
                else "prompt_traces.md"
            )
            outputs["prompt_traces_md"] = write_prompt_traces_markdown(
                filtered_traces,
                os.path.join(results_dir, trace_name),
                source_path=traces_pkl,
            )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert simulator history pickles to Markdown."
    )
    parser.add_argument(
        "path",
        help="Path to results dir or to messages_history.pkl",
    )
    parser.add_argument(
        "--prompt-traces-pkl",
        default=None,
        help="Optional explicit path to prompt_traces.pkl (used with pkl path mode).",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=None,
        help="If provided, write inbox/prompt traces only for this recipient/agent id.",
    )
    args = parser.parse_args()

    path = args.path
    if os.path.isdir(path):
        outputs = export_from_results_dir(path, agent_id=args.agent_id)
        for _, out_path in outputs.items():
            print(f"Wrote: {out_path}")
        return

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing path: {path}")

    messages = _load_pickle(path)
    if not isinstance(messages, list):
        raise TypeError(f"Expected list in messages pickle, got {type(messages).__name__}")
    filtered_messages = _filter_inbox_for_agent(messages, args.agent_id)
    base_name = (
        f"messages_inbox_agent_{int(args.agent_id)}.md"
        if args.agent_id is not None
        else "messages_history.md"
    )
    out_path = os.path.join(os.path.dirname(path), base_name)
    print(f"Wrote: {write_messages_markdown(filtered_messages, out_path, source_path=path)}")

    traces_path = args.prompt_traces_pkl
    if traces_path:
        traces = _load_pickle(traces_path)
        if not isinstance(traces, list):
            raise TypeError(
                f"Expected list in prompt traces pickle, got {type(traces).__name__}"
            )
        filtered_traces = traces
        if args.agent_id is not None:
            filtered_traces = [
                tr
                for tr in traces
                if isinstance(tr, dict) and int(tr.get("agent_id", -1)) == int(args.agent_id)
            ]
        trace_name = (
            f"prompt_traces_agent_{int(args.agent_id)}.md"
            if args.agent_id is not None
            else "prompt_traces.md"
        )
        out_traces = os.path.join(os.path.dirname(traces_path), trace_name)
        print(
            f"Wrote: {write_prompt_traces_markdown(filtered_traces, out_traces, source_path=traces_path)}"
        )


if __name__ == "__main__":
    main()