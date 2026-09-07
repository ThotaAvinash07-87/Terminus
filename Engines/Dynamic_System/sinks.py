"""Signal Sinks, Scopes, Loggers, and Terminators for Dynamic Systems."""

from __future__ import annotations
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from .base import Block


class ScopeSinkBlock(Block):
    """Records signal history for plotting, analysis, and metric extraction."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=0)
        self.time_history: List[float] = []
        self.signal_history: List[float] = []
        self.direct_feedthrough = False

    def record(self, t: float) -> None:
        self.time_history.append(float(t))
        val = self.inputs[0]
        self.signal_history.append(float(val) if isinstance(val, (int, float, np.number)) else val)

    def reset_states(self) -> None:
        self.time_history.clear()
        self.signal_history.clear()


ScopeBlock = ScopeSinkBlock
FloatingScopeBlock = ScopeSinkBlock
ScopeViewerBlock = ScopeSinkBlock


class DisplaySinkBlock(Block):
    """Show and track value of input signal."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.current_value: Any = 0.0
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        self.current_value = self.inputs[0]
        self.outputs[0] = self.current_value
        return self.outputs


DisplayBlock = DisplaySinkBlock


class TerminatorBlock(Block):
    """Terminate unconnected or ignored output lines."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=0)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[Any]:
        return []


class StopSimulationBlock(Block):
    """Stop simulation execution when input signal is non-zero."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=0)
        self.should_stop: bool = False
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        u = float(self.inputs[0])
        if abs(u) > 1e-6:
            self.should_stop = True
        return []

    def reset_states(self) -> None:
        self.should_stop = False


class ToWorkspaceBlock(Block):
    """Log time and signal history to workspace memory structure."""

    def __init__(self, name: str, variable_name: str = "simout"):
        super().__init__(name, num_inputs=1, num_outputs=0)
        self.var_name = variable_name
        self.time_history: List[float] = []
        self.signal_history: List[Any] = []
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[Any]:
        self.time_history.append(float(t))
        self.signal_history.append(self.inputs[0])
        return []

    def get_data(self) -> Dict[str, Any]:
        return {
            "time": np.array(self.time_history),
            "signals": {"values": np.array(self.signal_history), "name": self.var_name}
        }

    def reset_states(self) -> None:
        self.time_history.clear()
        self.signal_history.clear()


class ToFileBlock(Block):
    """Write logged simulation data to file (CSV or JSON)."""

    def __init__(self, name: str, filename: str = "sim_log.json"):
        super().__init__(name, num_inputs=1, num_outputs=0)
        self.filename = filename
        self.time_history: List[float] = []
        self.signal_history: List[Any] = []
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[Any]:
        self.time_history.append(float(t))
        self.signal_history.append(float(self.inputs[0]) if isinstance(self.inputs[0], (int, float, np.number)) else str(self.inputs[0]))
        return []

    def save_to_file(self) -> None:
        data = {"time": self.time_history, "data": self.signal_history}
        with open(self.filename, "w") as f:
            json.dump(data, f, indent=2)

    def reset_states(self) -> None:
        self.time_history.clear()
        self.signal_history.clear()


RecordSinkBlock = ToWorkspaceBlock
