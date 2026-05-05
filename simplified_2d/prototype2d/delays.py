from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class DelayState:
    value: float = 0.0


@dataclass
class DelayModel:
    config: Dict[str, Any]
    state_by_agent: Dict[int, DelayState] = field(default_factory=dict)

    def sample(self, agent_id: int) -> float:
        model_type = self.config.get("type", "zero")
        if model_type == "zero":
            return 0.0
        if model_type == "constant":
            return float(self.config.get("value", 0.0))
        if model_type == "constant_jitter":
            base = float(self.config.get("value", 0.0))
            jitter = float(self.config.get("jitter", 0.0))
            return max(0.0, base + np.random.uniform(-jitter, jitter))
        if model_type == "random_walk":
            sigma = float(self.config.get("sigma", 0.01))
            min_val = float(self.config.get("min", 0.0))
            max_val = float(self.config.get("max", 1.0))
            state = self.state_by_agent.setdefault(agent_id, DelayState(value=float(self.config.get("value", 0.0))))
            state.value = float(np.clip(state.value + np.random.normal(0.0, sigma), min_val, max_val))
            return state.value
        if model_type == "schedule":
            schedule: List[float] = self.config.get("per_agent", [])
            if agent_id < len(schedule):
                return float(schedule[agent_id])
            return float(self.config.get("fallback", 0.0))
        return 0.0

