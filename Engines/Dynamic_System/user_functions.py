"""User-Defined Functions, Python/C Callers, and S-Function blocks for Dynamic Systems."""

from __future__ import annotations
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from .base import Block


class FcnBlock(Block):
    """Apply specified math expression to input: e.g. 'u[0]**2 + sin(u[0])' or 'u + 2*u'."""

    def __init__(self, name: str, expression: str = "u[0]**2", num_inputs: int = 1):
        super().__init__(name, num_inputs=num_inputs, num_outputs=1)
        self.expr = expression
        self.compiled = compile(expression, "<fcn_expr>", "eval")
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = [float(x) for x in self.inputs]
        local_dict = {
            "u": u,
            "t": t,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "exp": math.exp,
            "log": math.log,
            "sqrt": math.sqrt,
            "abs": abs,
            "pi": math.pi,
            "e": math.e,
            "np": np
        }
        if len(u) == 1:
            local_dict["u0"] = u[0]
        try:
            res = eval(self.compiled, {"__builtins__": {}}, local_dict)
            self.outputs[0] = float(res)
        except Exception:
            self.outputs[0] = 0.0
        return self.outputs


class MATLABFunctionBlock(Block):
    """Execute native custom Python/MATLAB callable function `def f(inputs, t): ...`."""

    def __init__(
        self,
        name: str,
        func: Optional[Callable[[List[Any], float], Any]] = None,
        num_inputs: int = 1,
        num_outputs: int = 1
    ):
        super().__init__(name, num_inputs=num_inputs, num_outputs=num_outputs)
        self.func = func if func is not None else (lambda u, t: u[0] * 2.0)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        res = self.func(self.inputs, t)
        if isinstance(res, (list, tuple)):
            for i in range(min(len(res), self.num_outputs)):
                self.outputs[i] = res[i]
        else:
            self.outputs[0] = res
        return self.outputs


PythonCodeBlock = MATLABFunctionBlock
InterpretedMATLABFunctionBlock = MATLABFunctionBlock


class SFunctionBlock(Block):
    """Include Level-2 S-Function simulation block with custom state updates and outputs."""

    def __init__(
        self,
        name: str,
        mdl_outputs: Callable[[Any, float], Any],
        mdl_derivatives: Optional[Callable[[Any, float], np.ndarray]] = None,
        num_inputs: int = 1,
        num_outputs: int = 1,
        num_states: int = 0
    ):
        super().__init__(name, num_inputs=num_inputs, num_outputs=num_outputs)
        self.mdl_outputs = mdl_outputs
        self.mdl_derivs = mdl_derivatives
        self.states = np.zeros(num_states, dtype=float)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        res = self.mdl_outputs(self, t)
        if isinstance(res, (list, tuple)):
            self.outputs = list(res)
        else:
            self.outputs[0] = res
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        if self.mdl_derivs is not None:
            return self.mdl_derivs(self, t)
        return np.zeros(self.num_states, dtype=float)


SFunctionBuilderBlock = SFunctionBlock


class CCallerBlock(Block):
    """Integrate and call C code function simulator."""

    def __init__(self, name: str, c_func: Callable[..., Any], num_inputs: int = 1, num_outputs: int = 1):
        super().__init__(name, num_inputs=num_inputs, num_outputs=num_outputs)
        self.c_func = c_func
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        res = self.c_func(*self.inputs)
        if isinstance(res, (list, tuple)):
            self.outputs = list(res)
        else:
            self.outputs[0] = res
        return self.outputs


CFunctionBlock = CCallerBlock
FunctionCallerBlock = CCallerBlock
SimulinkFunctionBlock = MATLABFunctionBlock
MATLABSystemBlock = MATLABFunctionBlock


class InitializeFunctionBlock(Block):
    """Execute on model initialize event."""

    def __init__(self, name: str, init_action: Callable[[], None]):
        super().__init__(name, num_inputs=0, num_outputs=0)
        self.action = init_action

    def initialize(self) -> None:
        self.action()


class ReinitializeFunctionBlock(InitializeFunctionBlock):
    """Execute on model reinitialize event."""
    pass


class ResetFunctionBlock(InitializeFunctionBlock):
    """Execute on model reset event."""
    pass


class TerminateFunctionBlock(InitializeFunctionBlock):
    """Execute on model terminate event."""
    pass
