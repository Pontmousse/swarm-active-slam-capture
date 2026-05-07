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
    "make_backend",
]


def make_backend(config) -> DecisionBackend:
    name = (getattr(config, "decision_backend", "fsm") or "fsm").lower()
    if name == "random":
        return RandomBackend(config)
    if name == "openai":
        return OpenAIBackend(config)
    return FSMBackend(config)
