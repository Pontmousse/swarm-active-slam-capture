from __future__ import annotations

from typing import Any, Dict, Protocol

from ..behavior_command import BehaviorCommand


class DecisionBackend(Protocol):
    def decide(self, agent_id: int, snapshot: Dict[str, Any]) -> BehaviorCommand:
        ...
