"""Model Verification, Assertion, and Dynamic Range Checking blocks for Dynamic Systems."""

from __future__ import annotations
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from .base import Block


class AssertionBlock(Block):
    """Check whether input signal is non-zero (True). Sets violation flag if zero."""

    def __init__(self, name: str, assertion_message: str = "Assertion failed!"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.msg = assertion_message
        self.is_valid = True
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        passed = abs(u) > 1e-6
        self.is_valid = passed
        self.outputs[0] = 1.0 if passed else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.is_valid = True


class CheckStaticRangeBlock(Block):
    """Check that signal falls inside fixed [min, max] range of amplitudes."""

    def __init__(self, name: str, min_val: float = -10.0, max_val: float = 10.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.min_v = float(min_val)
        self.max_v = float(max_val)
        self.is_valid = True
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        inside = (self.min_v <= u <= self.max_v)
        self.is_valid = inside
        self.outputs[0] = 1.0 if inside else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.is_valid = True


class CheckDynamicRangeBlock(Block):
    """Check that signal falls inside range [min, max] that varies from time step to time step:
    Input 0: u (signal)
    Input 1: max
    Input 2: min
    """

    def __init__(self, name: str):
        super().__init__(name, num_inputs=3, num_outputs=1)
        self.is_valid = True
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        max_v = float(self.inputs[1]) if len(self.inputs) > 1 else 1.0
        min_v = float(self.inputs[2]) if len(self.inputs) > 2 else -1.0
        if min_v > max_v:
            min_v, max_v = max_v, min_v
        inside = (min_v <= u <= max_v)
        self.is_valid = inside
        self.outputs[0] = 1.0 if inside else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.is_valid = True


class CheckStaticGapBlock(Block):
    """Check that a gap of forbidden range (min_gap, max_gap) occurs in signal."""

    def __init__(self, name: str, min_gap: float = -1.0, max_gap: float = 1.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.min_g = float(min_gap)
        self.max_g = float(max_gap)
        self.is_valid = True
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        outside_gap = not (self.min_g < u < self.max_g)
        self.is_valid = outside_gap
        self.outputs[0] = 1.0 if outside_gap else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.is_valid = True


class CheckDynamicGapBlock(Block):
    """Check dynamic gap:
    Input 0: u (signal)
    Input 1: max gap
    Input 2: min gap
    """

    def __init__(self, name: str):
        super().__init__(name, num_inputs=3, num_outputs=1)
        self.is_valid = True
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        max_g = float(self.inputs[1]) if len(self.inputs) > 1 else 1.0
        min_g = float(self.inputs[2]) if len(self.inputs) > 2 else -1.0
        outside_gap = not (min_g < u < max_g)
        self.is_valid = outside_gap
        self.outputs[0] = 1.0 if outside_gap else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.is_valid = True


class CheckStaticLowerBoundBlock(Block):
    """Check that signal is >= (or optionally >) static lower bound."""

    def __init__(self, name: str, lower_bound: float = 0.0, strict: bool = False):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.bound = float(lower_bound)
        self.strict = strict
        self.is_valid = True
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        ok = (u > self.bound) if self.strict else (u >= self.bound)
        self.is_valid = ok
        self.outputs[0] = 1.0 if ok else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.is_valid = True


class CheckStaticUpperBoundBlock(Block):
    """Check that signal is <= (or optionally <) static upper bound."""

    def __init__(self, name: str, upper_bound: float = 10.0, strict: bool = False):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.bound = float(upper_bound)
        self.strict = strict
        self.is_valid = True
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        ok = (u < self.bound) if self.strict else (u <= self.bound)
        self.is_valid = ok
        self.outputs[0] = 1.0 if ok else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.is_valid = True


class CheckDynamicLowerBoundBlock(Block):
    """Check that signal 0 is >= signal 1 (lower bound input)."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.is_valid = True
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        bound = float(self.inputs[1]) if len(self.inputs) > 1 else 0.0
        ok = (u >= bound)
        self.is_valid = ok
        self.outputs[0] = 1.0 if ok else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.is_valid = True


class CheckDynamicUpperBoundBlock(Block):
    """Check that signal 0 is <= signal 1 (upper bound input)."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.is_valid = True
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        bound = float(self.inputs[1]) if len(self.inputs) > 1 else 0.0
        ok = (u <= bound)
        self.is_valid = ok
        self.outputs[0] = 1.0 if ok else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.is_valid = True


class CheckDiscreteGradientBlock(Block):
    """Check that |u[k] - u[k-1]| <= max_gradient."""

    def __init__(self, name: str, max_gradient: float = 1.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.max_g = float(max_gradient)
        self.prev_u: Optional[float] = None
        self.is_valid = True
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        if self.prev_u is not None:
            grad = abs(u - self.prev_u)
            self.is_valid = (grad <= self.max_g + 1e-9)
        else:
            self.is_valid = True
        self.prev_u = u
        self.outputs[0] = 1.0 if self.is_valid else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.prev_u = None
        self.is_valid = True


class CheckInputResolutionBlock(Block):
    """Check that input signal has specified resolution q (i.e. u is integer multiple of q)."""

    def __init__(self, name: str, resolution: float = 0.1, tolerance: float = 1e-4):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.q = max(1e-9, float(resolution))
        self.tol = float(tolerance)
        self.is_valid = True
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        rem = abs(u % self.q)
        is_res = (rem < self.tol or abs(rem - self.q) < self.tol)
        self.is_valid = is_res
        self.outputs[0] = 1.0 if is_res else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.is_valid = True
