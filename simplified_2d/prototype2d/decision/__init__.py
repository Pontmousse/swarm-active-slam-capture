"""Pluggable decision backends for Phase 3 (FSM, random, optional OpenAI)."""

from __future__ import annotations

from .backends import FSMBackend, OpenAIBackend, RandomBackend
from .protocol import DecisionBackend
from .snapshot import build_decision_snapshot

__all__ = [
    "DecisionBackend",
    "FSMBackend",
    "RandomBackend",
    "OpenAIBackend",
    "build_decision_snapshot",
    "resolve_backend_name",
    "make_backend",
]


def resolve_backend_name(config) -> str:
    legacy_name = getattr(config, "decision_backend", None)
    llm_name = getattr(config, "llm", None)
    chosen = llm_name if llm_name is not None else legacy_name
    return (chosen or "fsm").lower()


def make_backend(config) -> DecisionBackend:
    name = resolve_backend_name(config)
    if name == "random":
        return RandomBackend(config)
    if name == "openai":
        return OpenAIBackend(config)
    return FSMBackend(config)
