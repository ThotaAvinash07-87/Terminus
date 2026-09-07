"""Ports, Hierarchy, Conditional Execution, and Subsystem blocks for Dynamic Systems."""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from .base import Block


class InportBlock(Block):
    """Create input port for subsystem or external input."""

    def __init__(self, name: str, port_number: int = 1):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.port_num = port_number
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = self.inputs[0]
        return self.outputs


Inport = InportBlock


class OutportBlock(Block):
    """Create output port for subsystem or external output."""

    def __init__(self, name: str, port_number: int = 1):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.port_num = port_number
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = self.inputs[0]
        return self.outputs


Outport = OutportBlock


class InBusElementBlock(InportBlock):
    """Select input element from external bus."""
    pass


class OutBusElementBlock(OutportBlock):
    """Specify output to external bus."""
    pass


class EnableBlock(Block):
    """Add enable port to subsystem."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        is_enabled = 1.0 if u > 0.0 else 0.0
        self.outputs[0] = is_enabled
        return self.outputs


class TriggerBlock(Block):
    """Add trigger port to subsystem (rising, falling, or either edge)."""

    def __init__(self, name: str, trigger_type: str = "rising"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.trig_type = trigger_type.lower()
        self.prev_u = 0.0
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        fired = False
        if self.trig_type == "rising" and self.prev_u <= 0.0 and u > 0.0:
            fired = True
        elif self.trig_type == "falling" and self.prev_u >= 0.0 and u < 0.0:
            fired = True
        elif self.trig_type == "either" and ((self.prev_u <= 0.0 and u > 0.0) or (self.prev_u >= 0.0 and u < 0.0)):
            fired = True
        self.prev_u = u
        self.outputs[0] = 1.0 if fired else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.prev_u = 0.0


class SubsystemBlock(Block):
    """Group blocks to create model hierarchy with an internal forward pass function."""

    def __init__(
        self,
        name: str,
        forward_fn: Optional[Callable[[List[Any], float], List[Any]]] = None,
        num_inputs: int = 1,
        num_outputs: int = 1
    ):
        super().__init__(name, num_inputs=num_inputs, num_outputs=num_outputs)
        self.fwd_fn = forward_fn if forward_fn is not None else (lambda u, t: u)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        res = self.fwd_fn(self.inputs, t)
        if isinstance(res, (list, tuple)):
            for i in range(min(len(res), self.num_outputs)):
                self.outputs[i] = res[i]
        else:
            self.outputs[0] = res
        return self.outputs


class EnabledSubsystemBlock(Block):
    """Subsystem whose execution is enabled when input 0 (enable port) > 0."""

    def __init__(
        self,
        name: str,
        forward_fn: Optional[Callable[[List[Any], float], List[Any]]] = None,
        num_inputs: int = 2,
        num_outputs: int = 1
    ):
        super().__init__(name, num_inputs=num_inputs, num_outputs=num_outputs)
        self.fwd_fn = forward_fn if forward_fn is not None else (lambda u, t: [u[1]])
        self.held_outputs = [0.0] * num_outputs
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        enable_sig = float(self.inputs[0])
        if enable_sig > 0.0:
            data_inputs = self.inputs[1:]
            res = self.fwd_fn(self.inputs, t)
            if isinstance(res, (list, tuple)):
                self.held_outputs = list(res)
            else:
                self.held_outputs[0] = res
        for i in range(self.num_outputs):
            self.outputs[i] = self.held_outputs[i]
        return self.outputs

    def reset_states(self) -> None:
        self.held_outputs = [0.0] * self.num_outputs


class TriggeredSubsystemBlock(Block):
    """Subsystem whose execution is triggered by rising edge on input 0."""

    def __init__(
        self,
        name: str,
        forward_fn: Optional[Callable[[List[Any], float], List[Any]]] = None,
        num_inputs: int = 2,
        num_outputs: int = 1
    ):
        super().__init__(name, num_inputs=num_inputs, num_outputs=num_outputs)
        self.fwd_fn = forward_fn if forward_fn is not None else (lambda u, t: [u[1]])
        self.prev_trig = 0.0
        self.held_outputs = [0.0] * num_outputs
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        trig = float(self.inputs[0])
        if self.prev_trig <= 0.0 and trig > 0.0:
            res = self.fwd_fn(self.inputs, t)
            if isinstance(res, (list, tuple)):
                self.held_outputs = list(res)
            else:
                self.held_outputs[0] = res
        self.prev_trig = trig
        for i in range(self.num_outputs):
            self.outputs[i] = self.held_outputs[i]
        return self.outputs

    def reset_states(self) -> None:
        self.prev_trig = 0.0
        self.held_outputs = [0.0] * self.num_outputs


class EnabledAndTriggeredSubsystemBlock(Block):
    """Subsystem whose execution is enabled by input 0 and triggered by input 1."""

    def __init__(
        self,
        name: str,
        forward_fn: Optional[Callable[[List[Any], float], List[Any]]] = None,
        num_inputs: int = 3,
        num_outputs: int = 1
    ):
        super().__init__(name, num_inputs=num_inputs, num_outputs=num_outputs)
        self.fwd_fn = forward_fn if forward_fn is not None else (lambda u, t: [u[2]])
        self.prev_trig = 0.0
        self.held_outputs = [0.0] * num_outputs
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        en = float(self.inputs[0])
        trig = float(self.inputs[1]) if len(self.inputs) > 1 else 0.0
        if en > 0.0 and self.prev_trig <= 0.0 and trig > 0.0:
            res = self.fwd_fn(self.inputs, t)
            if isinstance(res, (list, tuple)):
                self.held_outputs = list(res)
            else:
                self.held_outputs[0] = res
        self.prev_trig = trig
        for i in range(self.num_outputs):
            self.outputs[i] = self.held_outputs[i]
        return self.outputs

    def reset_states(self) -> None:
        self.prev_trig = 0.0
        self.held_outputs = [0.0] * self.num_outputs


class ResettableSubsystemBlock(SubsystemBlock):
    """Subsystem whose internal states reset with external trigger."""
    pass


class ForEachSubsystemBlock(Block):
    """Apply algorithm to each element of array."""

    def __init__(self, name: str, element_fn: Callable[[Any], Any]):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.fn = element_fn
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        inp = self.inputs[0]
        if isinstance(inp, (list, tuple, np.ndarray)):
            self.outputs[0] = [self.fn(x) for x in inp]
        else:
            self.outputs[0] = self.fn(inp)
        return self.outputs


class ForIteratorSubsystemBlock(Block):
    """Repeat subsystem execution during step for N iterations."""

    def __init__(self, name: str, num_iterations: int = 5, iter_fn: Optional[Callable[[int, Any], Any]] = None):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.n_iter = int(num_iterations)
        self.fn = iter_fn if iter_fn is not None else (lambda i, u: u + i)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        val = self.inputs[0]
        for i in range(self.n_iter):
            val = self.fn(i, val)
        self.outputs[0] = val
        return self.outputs


class WhileIteratorSubsystemBlock(Block):
    """Repeat subsystem execution while condition is true."""

    def __init__(self, name: str, max_iterations: int = 100, step_fn: Optional[Callable[[Any], Tuple[Any, bool]]] = None):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.max_iter = int(max_iterations)
        self.step_fn = step_fn if step_fn is not None else (lambda u: (u / 2.0, u > 1.0))
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        val = self.inputs[0]
        for _ in range(self.max_iter):
            val, cont = self.step_fn(val)
            if not cont:
                break
        self.outputs[0] = val
        return self.outputs


class IfBlock(Block):
    """Select subsystem execution using if/else logic.
    Outputs: [action_0, action_1, ...]
    """

    def __init__(self, name: str, condition_str: str = "u1 > 0"):
        super().__init__(name, num_inputs=1, num_outputs=2)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u1 = float(self.inputs[0])
        cond = (u1 > 0.0)
        self.outputs[0] = 1.0 if cond else 0.0
        self.outputs[1] = 0.0 if cond else 1.0
        return self.outputs


class IfActionSubsystemBlock(EnabledSubsystemBlock):
    """Subsystem enabled by an If block action port."""
    pass


class SwitchCaseBlock(Block):
    """Select subsystem execution using switch/case statement logic."""

    def __init__(self, name: str, cases: Sequence[int] = (1, 2, 3)):
        self.cases = list(cases)
        # Outputs: len(cases) + 1 (for default)
        super().__init__(name, num_inputs=1, num_outputs=len(self.cases) + 1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = int(round(float(self.inputs[0])))
        for i in range(self.num_outputs):
            self.outputs[i] = 0.0
        matched = False
        for i, c in enumerate(self.cases):
            if u == c:
                self.outputs[i] = 1.0
                matched = True
                break
        if not matched:
            self.outputs[-1] = 1.0  # Default case
        return self.outputs


class SwitchCaseActionSubsystemBlock(EnabledSubsystemBlock):
    """Subsystem enabled by Switch Case action port."""
    pass


class ModelBlock(SubsystemBlock):
    """Reference another model to create model hierarchy."""
    pass


ModelReferenceBlock = ModelBlock
VariantSubsystemBlock = SubsystemBlock


class UnitSystemConfigurationBlock(Block):
    """Restrict units to specified allowed unit system (e.g. 'SI', 'English')."""

    def __init__(self, name: str, unit_system: str = "SI"):
        super().__init__(name, num_inputs=0, num_outputs=0)
        self.unit_sys = unit_system


class FunctionCallGeneratorBlock(Block):
    """Provide function-call events at regular sampling rate."""

    def __init__(self, name: str, sample_time: float = 0.01):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.ts = max(1e-6, float(sample_time))
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = 1.0
        return self.outputs


class FunctionCallSubsystemBlock(EnabledSubsystemBlock):
    """Subsystem controlled by function-call input."""
    pass


class FunctionCallSplitBlock(Block):
    """Junction for splitting function-call lines."""

    def __init__(self, name: str, num_branches: int = 2):
        super().__init__(name, num_inputs=1, num_outputs=num_branches)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        for i in range(self.num_outputs):
            self.outputs[i] = self.inputs[0]
        return self.outputs


class FunctionCallFeedbackLatchBlock(Block):
    """Break feedback loop involving data signals between function-call blocks."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.latched_val = 0.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[Any]:
        out = self.latched_val
        self.latched_val = self.inputs[0]
        self.outputs[0] = out
        return self.outputs


class FunctionElementBlock(Block):
    """Specify function provided via exporting function port."""

    def __init__(self, name: str, function_name: str = "calc"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.func_name = function_name
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = self.inputs[0]
        return self.outputs


class FunctionElementCallBlock(Block):
    """Specify function call to issue via invoking function port."""

    def __init__(self, name: str, function_name: str = "calc"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.func_name = function_name
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = self.inputs[0]
        return self.outputs
