"""Comprehensive Math, Arithmetic, and Complex Operation blocks for Dynamic Systems."""

from __future__ import annotations
import cmath
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from CORE.common_math import parse_eng_unit
from .base import Block


class GainBlock(Block):
    """Proportional Gain: y = K * u."""

    def __init__(self, name: str, gain: Union[str, float] = 1.0, k: Optional[Union[str, float]] = None):
        super().__init__(name, num_inputs=1, num_outputs=1)
        actual_gain = k if k is not None else gain
        self.gain = parse_eng_unit(actual_gain)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = self.gain * float(self.inputs[0])
        return self.outputs


class SliderGainBlock(GainBlock):
    """Gain with tunable slider range [low, high]."""

    def __init__(self, name: str, gain: float = 1.0, low: float = 0.0, high: float = 10.0):
        super().__init__(name, gain=gain)
        self.low = float(low)
        self.high = float(high)


class SumBlock(Block):
    """Sum / Difference Junction: y = signs[0]*u[0] + signs[1]*u[1] + ..."""

    def __init__(self, name: str, signs: str = "+-"):
        super().__init__(name, num_inputs=len(signs), num_outputs=1)
        self.signs = [1.0 if s == "+" else -1.0 for s in signs]
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        total = 0.0
        for i, s in enumerate(self.signs):
            if i < len(self.inputs):
                total += s * float(self.inputs[i])
        self.outputs[0] = total
        return self.outputs


class ProductBlock(Block):
    """Multiplication & Division: y = (u[0] * u[1] ...) / (u[k] ...) with zero-division safeguard."""

    def __init__(self, name: str, operations: str = "**"):
        super().__init__(name, num_inputs=len(operations), num_outputs=1)
        self.ops = operations
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        res = 1.0
        for i, op in enumerate(self.ops):
            if i < len(self.inputs):
                val = float(self.inputs[i])
                if op == "*":
                    res *= val
                elif op == "/":
                    denom = val if abs(val) > 1e-12 else (1e-12 if val >= 0 else -1e-12)
                    res /= denom
        self.outputs[0] = res
        return self.outputs


class DivideBlock(Block):
    """Divide one input by another: y = u1 / u2."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u1 = float(self.inputs[0])
        u2 = float(self.inputs[1]) if len(self.inputs) > 1 else 1.0
        denom = u2 if abs(u2) > 1e-12 else (1e-12 if u2 >= 0 else -1e-12)
        self.outputs[0] = u1 / denom
        return self.outputs


class DotProductBlock(Block):
    """Generate dot product of two vector inputs."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        v1 = np.asarray(self.inputs[0], dtype=float) if isinstance(self.inputs[0], (list, tuple, np.ndarray)) else np.array([float(self.inputs[0])])
        v2 = np.asarray(self.inputs[1], dtype=float) if len(self.inputs) > 1 and isinstance(self.inputs[1], (list, tuple, np.ndarray)) else np.array([float(self.inputs[1]) if len(self.inputs) > 1 else 0.0])
        self.outputs[0] = float(np.dot(v1, v2))
        return self.outputs


class ProductOfElementsBlock(Block):
    """Calculate product of elements in input vector or scalar."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = self.inputs[0]
        if isinstance(u, (list, tuple, np.ndarray)):
            self.outputs[0] = float(np.prod(u))
        else:
            self.outputs[0] = float(u)
        return self.outputs


class AbsBlock(Block):
    """Output absolute value of input: y = |u|."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = abs(float(self.inputs[0]))
        return self.outputs


class SignBlock(Block):
    """Indicate sign of input: 1 for u>0, -1 for u<0, 0 for u==0."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        self.outputs[0] = 1.0 if u > 0 else (-1.0 if u < 0 else 0.0)
        return self.outputs


class SqrtBlock(Block):
    """Calculate square root, signed square root, or reciprocal square root."""

    def __init__(self, name: str, function: str = "sqrt"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.fn = function.lower()
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        if self.fn == "signed_sqrt":
            self.outputs[0] = math.copysign(math.sqrt(abs(u)), u)
        elif self.fn == "rsqrt":
            self.outputs[0] = 1.0 / math.sqrt(max(1e-12, u))
        else:
            self.outputs[0] = math.sqrt(max(0.0, u))
        return self.outputs


class BiasBlock(Block):
    """Add bias to input: y = u + bias."""

    def __init__(self, name: str, bias: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.bias = float(bias)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = float(self.inputs[0]) + self.bias
        return self.outputs


class UnaryMinusBlock(Block):
    """Negate input: y = -u."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = -float(self.inputs[0])
        return self.outputs


class MathFunctionBlock(Block):
    """Applies unary/binary math functions:
    exp, log, 10^u, log10, square, sqrt, pow, reciprocal, hypot, rem, mod, transpose.
    """

    FUNC_MAP: Dict[str, Callable[[float], float]] = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": lambda x: math.exp(max(-100.0, min(100.0, x))),
        "log": lambda x: math.log(max(1e-12, x)),
        "log10": lambda x: math.log10(max(1e-12, x)),
        "10^u": lambda x: 10.0 ** max(-100.0, min(100.0, x)),
        "magnitude^2": lambda x: x * x,
        "square": lambda x: x * x,
        "sqrt": lambda x: math.sqrt(max(0.0, x)),
        "pow": lambda x: x * x,
        "reciprocal": lambda x: 1.0 / (x if abs(x) > 1e-12 else (1e-12 if x >= 0 else -1e-12)),
        "abs": abs,
        "sign": lambda x: float(np.sign(x)),
    }

    def __init__(self, name: str, function: str = "sin"):
        num_in = 2 if function.lower() in ("pow", "hypot", "rem", "mod") else 1
        super().__init__(name, num_inputs=num_in, num_outputs=1)
        self.func_name = function.lower()
        self.func = self.FUNC_MAP.get(self.func_name, math.sin)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u1 = float(self.inputs[0])
        if self.func_name == "pow" and len(self.inputs) > 1:
            u2 = float(self.inputs[1])
            self.outputs[0] = u1 ** u2
        elif self.func_name == "hypot" and len(self.inputs) > 1:
            u2 = float(self.inputs[1])
            self.outputs[0] = math.hypot(u1, u2)
        elif self.func_name == "rem" and len(self.inputs) > 1:
            u2 = float(self.inputs[1])
            denom = u2 if abs(u2) > 1e-12 else 1e-12
            self.outputs[0] = math.remainder(u1, denom)
        elif self.func_name == "mod" and len(self.inputs) > 1:
            u2 = float(self.inputs[1])
            denom = u2 if abs(u2) > 1e-12 else 1e-12
            self.outputs[0] = u1 % denom
        else:
            try:
                self.outputs[0] = self.func(u1)
            except Exception:
                self.outputs[0] = 0.0
        return self.outputs


class RoundingFunctionBlock(Block):
    """Apply rounding function: 'floor', 'ceil', 'round', 'fix' (round toward zero)."""

    def __init__(self, name: str, function: str = "round"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.fn = function.lower()
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        if self.fn == "floor":
            self.outputs[0] = float(math.floor(u))
        elif self.fn == "ceil":
            self.outputs[0] = float(math.ceil(u))
        elif self.fn == "fix":
            self.outputs[0] = float(math.trunc(u))
        else:
            self.outputs[0] = float(round(u))
        return self.outputs


class TrigonometricFunctionBlock(Block):
    """Specified trigonometric function on input:
    sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, sinc.
    """

    def __init__(self, name: str, function: str = "sin"):
        num_in = 2 if function.lower() == "atan2" else 1
        super().__init__(name, num_inputs=num_in, num_outputs=1)
        self.fn = function.lower()
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u1 = float(self.inputs[0])
        try:
            if self.fn == "sin":
                y = math.sin(u1)
            elif self.fn == "cos":
                y = math.cos(u1)
            elif self.fn == "tan":
                y = math.tan(u1)
            elif self.fn == "asin":
                y = math.asin(max(-1.0, min(1.0, u1)))
            elif self.fn == "acos":
                y = math.acos(max(-1.0, min(1.0, u1)))
            elif self.fn == "atan":
                y = math.atan(u1)
            elif self.fn == "atan2":
                u2 = float(self.inputs[1]) if len(self.inputs) > 1 else 0.0
                y = math.atan2(u1, u2)
            elif self.fn == "sinh":
                y = math.sinh(max(-50.0, min(50.0, u1)))
            elif self.fn == "cosh":
                y = math.cosh(max(-50.0, min(50.0, u1)))
            elif self.fn == "tanh":
                y = math.tanh(u1)
            elif self.fn == "sinc":
                y = math.sin(math.pi * u1) / (math.pi * u1) if abs(u1) > 1e-12 else 1.0
            else:
                y = math.sin(u1)
        except Exception:
            y = 0.0
        self.outputs[0] = y
        return self.outputs


class SineWaveFunctionBlock(Block):
    """Generate sine wave using external signal as time/phase source: y = amp*sin(freq*u + phase) + bias."""

    def __init__(self, name: str, amplitude: float = 1.0, frequency: float = 1.0, phase: float = 0.0, bias: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.amp = float(amplitude)
        self.freq = float(frequency)
        self.phase = float(phase)
        self.bias = float(bias)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        self.outputs[0] = self.bias + self.amp * math.sin(self.freq * u + self.phase)
        return self.outputs


class PolynomialBlock(Block):
    """Perform evaluation of polynomial coefficients on input: P(u) = a[0]*u^n + a[1]*u^(n-1) + ... + a[n]."""

    def __init__(self, name: str, coefficients: Sequence[float]):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.coeffs = np.asarray(coefficients, dtype=float)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        self.outputs[0] = float(np.polyval(self.coeffs, u))
        return self.outputs


class MinMaxBlock(Block):
    """Output minimum or maximum input value."""

    def __init__(self, name: str, function: str = "min", num_inputs: int = 2):
        super().__init__(name, num_inputs=num_inputs, num_outputs=1)
        self.fn = function.lower()
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        vals = [float(x) for x in self.inputs]
        self.outputs[0] = min(vals) if self.fn == "min" else max(vals)
        return self.outputs


class MinMaxRunningResettableBlock(Block):
    """Determine running minimum or maximum of signal over time with reset port:
    Input 0: u (signal)
    Input 1: reset trigger
    """

    def __init__(self, name: str, function: str = "min"):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.fn = function.lower()
        self.curr_val: Optional[float] = None
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        rst = float(self.inputs[1]) if len(self.inputs) > 1 else 0.0

        if abs(rst) > 1e-6 or self.curr_val is None:
            self.curr_val = u
        else:
            self.curr_val = min(self.curr_val, u) if self.fn == "min" else max(self.curr_val, u)

        self.outputs[0] = self.curr_val
        return self.outputs

    def reset_states(self) -> None:
        self.curr_val = None


class ComplexToMagnitudeAngleBlock(Block):
    """Compute magnitude and phase angle of complex input."""

    def __init__(self, name: str, output_in_degrees: bool = False):
        super().__init__(name, num_inputs=1, num_outputs=2)
        self.in_deg = output_in_degrees
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        val = complex(self.inputs[0])
        mag, phase = cmath.polar(val)
        if self.in_deg:
            phase = math.degrees(phase)
        self.outputs[0] = mag
        self.outputs[1] = phase
        return self.outputs


class MagnitudeAngleToComplexBlock(Block):
    """Convert magnitude and phase angle to complex signal."""

    def __init__(self, name: str, input_in_degrees: bool = False):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.in_deg = input_in_degrees
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        mag = float(self.inputs[0])
        angle = float(self.inputs[1]) if len(self.inputs) > 1 else 0.0
        if self.in_deg:
            angle = math.radians(angle)
        c = cmath.rect(mag, angle)
        self.outputs[0] = c
        return self.outputs


class ComplexToRealImagBlock(Block):
    """Output real and imaginary parts of complex input."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=2)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        val = complex(self.inputs[0])
        self.outputs[0] = val.real
        self.outputs[1] = val.imag
        return self.outputs


class RealImagToComplexBlock(Block):
    """Convert real and imaginary inputs to complex signal."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        re = float(self.inputs[0])
        im = float(self.inputs[1]) if len(self.inputs) > 1 else 0.0
        self.outputs[0] = complex(re, im)
        return self.outputs


class WeightedSampleTimeMathBlock(Block):
    """Support calculations involving sample time: y = u * K * Ts (or u / (K * Ts))."""

    def __init__(self, name: str, weight: float = 1.0, sample_time: float = 0.01, operation: str = "*"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.w = float(weight)
        self.ts = max(1e-6, float(sample_time))
        self.op = operation
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        factor = self.w * self.ts
        if self.op == "/":
            self.outputs[0] = u / (factor if abs(factor) > 1e-12 else 1e-12)
        else:
            self.outputs[0] = u * factor
        return self.outputs


class IncrementRealWorldBlock(Block):
    """Increase real-world value of signal by one: y = u + 1."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = float(self.inputs[0]) + 1.0
        return self.outputs


class DecrementRealWorldBlock(Block):
    """Decrease real-world value of signal by one: y = u - 1."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = float(self.inputs[0]) - 1.0
        return self.outputs


class IncrementStoredIntegerBlock(Block):
    """Increase stored integer value of signal by one: y = int(u) + 1."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = float(int(round(float(self.inputs[0]))) + 1)
        return self.outputs


class DecrementStoredIntegerBlock(Block):
    """Decrease stored integer value of signal by one: y = int(u) - 1."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = float(int(round(float(self.inputs[0]))) - 1)
        return self.outputs


class DecrementToZeroBlock(Block):
    """Decrease real-world value of signal by one, but only to zero: y = max(0, u - 1)."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = max(0.0, float(self.inputs[0]) - 1.0)
        return self.outputs


class DecrementTimeToZeroBlock(Block):
    """Decrease real-world value of signal by sample time, but only to zero: y = max(0, u - Ts)."""

    def __init__(self, name: str, sample_time: float = 0.01):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.ts = max(1e-6, float(sample_time))
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = max(0.0, float(self.inputs[0]) - self.ts)
        return self.outputs


class FindNonzeroElementsBlock(Block):
    """Find indices of non-zero elements in array."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = np.asarray(self.inputs[0], dtype=float)
        indices = np.flatnonzero(u)
        self.outputs[0] = indices.tolist()
        return self.outputs


class AlgebraicConstraintBlock(Block):
    """Constrain input signal: asserts residue z = 0 or outputs constrained value."""

    def __init__(self, name: str, initial_guess: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.z = float(initial_guess)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        # y is the resolved constraint variable
        self.outputs[0] = float(self.inputs[0])
        return self.outputs
