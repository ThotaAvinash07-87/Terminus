"""Base classes and foundational structures for Dynamic System simulation blocks."""

from __future__ import annotations
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class Block:
    """Base class for all dynamic simulation blocks."""

    def __init__(self, name: str, num_inputs: int = 1, num_outputs: int = 1):
        self.name = name.strip()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.inputs: List[Any] = [0.0] * num_inputs
        self.outputs: List[Any] = [0.0] * num_outputs
        # State vector
        self.states = np.zeros(0, dtype=float)
        self.direct_feedthrough: bool = True  # True if output depends directly on input at time t
        self.sample_time: float = 0.0         # 0.0 for continuous, >0 for discrete
        self.parameters: Dict[str, Any] = {}

    @property
    def num_states(self) -> int:
        return len(self.states)

    def compute_output(self, t: float) -> List[Any]:
        """Calculates outputs given current states, inputs, and time t."""
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        """Calculates time derivative d(states)/dt for ODE integration."""
        return np.zeros(0, dtype=float)

    def discrete_update(self, t: float) -> None:
        """Updates internal discrete states at sample time intervals."""
        pass

    def reset_states(self) -> None:
        """Resets internal states to initial conditions."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self.name}' in={self.num_inputs} out={self.num_outputs} states={self.num_states}>"
