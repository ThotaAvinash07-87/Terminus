"""Discrete-time and digital control simulation blocks for Dynamic Systems."""

from __future__ import annotations
import math
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.signal as signal
from CORE.common_math import parse_eng_unit
from .base import Block


class ZeroOrderHoldBlock(Block):
    """Zero-Order Hold (ZOH): discrete sample-and-hold with sample time Ts."""

    def __init__(self, name: str, sample_time: float = 0.01):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.sample_time = max(1e-6, float(sample_time))
        self.held_value = 0.0
        self.last_sample_t = -1.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        if sample_instant > self.last_sample_t:
            self.held_value = float(self.inputs[0])
            self.last_sample_t = sample_instant

        self.outputs[0] = self.held_value
        return self.outputs

    def reset_states(self) -> None:
        self.held_value = 0.0
        self.last_sample_t = -1.0


class UnitDelayBlock(Block):
    """Discrete Unit Delay z^-1: y[k] = u[k-1]."""

    def __init__(self, name: str, sample_time: float = 0.01, initial_condition: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.sample_time = max(1e-6, float(sample_time))
        self.ic = float(initial_condition)
        self.prev_val = self.ic
        self.curr_val = self.ic
        self.last_sample_t = -1.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        if sample_instant > self.last_sample_t:
            self.prev_val = self.curr_val
            self.curr_val = float(self.inputs[0])
            self.last_sample_t = sample_instant

        self.outputs[0] = self.prev_val
        return self.outputs

    def reset_states(self) -> None:
        self.prev_val = self.ic
        self.curr_val = self.ic
        self.last_sample_t = -1.0


class MemoryBlock(Block):
    """Memory Block: outputs the input from previous major time step."""

    def __init__(self, name: str, initial_condition: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.ic = float(initial_condition)
        self.prev_val = self.ic
        self.curr_val = self.ic
        self.last_t = -1.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        if t > self.last_t:
            self.prev_val = self.curr_val
            self.curr_val = float(self.inputs[0])
            self.last_t = t
        self.outputs[0] = self.prev_val
        return self.outputs

    def reset_states(self) -> None:
        self.prev_val = self.ic
        self.curr_val = self.ic
        self.last_t = -1.0


class DelayBlock(Block):
    """Discrete Delay: delays input signal by N sample periods (z^-N)."""

    def __init__(self, name: str, delay_length: int = 1, sample_time: float = 0.01, initial_condition: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.delay_length = max(1, int(delay_length))
        self.sample_time = max(1e-6, float(sample_time))
        self.ic = float(initial_condition)
        self.buffer = deque([self.ic] * self.delay_length, maxlen=self.delay_length)
        self.last_sample_t = -1.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        if sample_instant > self.last_sample_t:
            self.buffer.append(float(self.inputs[0]))
            self.last_sample_t = sample_instant
        self.outputs[0] = self.buffer[0]
        return self.outputs

    def reset_states(self) -> None:
        self.buffer = deque([self.ic] * self.delay_length, maxlen=self.delay_length)
        self.last_sample_t = -1.0


class VariableIntegerDelayBlock(Block):
    """Delays input signal by variable integer number of sample periods:
    Input 0: u[k] (signal)
    Input 1: d[k] (integer delay count)
    """

    def __init__(self, name: str, max_delay: int = 100, sample_time: float = 0.01, initial_condition: float = 0.0):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.max_delay = max(1, int(max_delay))
        self.sample_time = max(1e-6, float(sample_time))
        self.ic = float(initial_condition)
        self.buffer = deque([self.ic] * (self.max_delay + 1), maxlen=self.max_delay + 1)
        self.last_sample_t = -1.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        if sample_instant > self.last_sample_t:
            self.buffer.append(float(self.inputs[0]))
            self.last_sample_t = sample_instant

        delay = int(round(float(self.inputs[1]))) if len(self.inputs) > 1 else 1
        delay = max(0, min(self.max_delay, delay))
        idx = len(self.buffer) - 1 - delay
        self.outputs[0] = self.buffer[max(0, idx)]
        return self.outputs

    def reset_states(self) -> None:
        self.buffer = deque([self.ic] * (self.max_delay + 1), maxlen=self.max_delay + 1)
        self.last_sample_t = -1.0


class ResettableDelayBlock(Block):
    """Delay input signal with external reset port:
    Input 0: u[k] (signal)
    Input 1: reset trigger
    """

    def __init__(self, name: str, delay_length: int = 1, sample_time: float = 0.01, initial_condition: float = 0.0):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.delay_length = max(1, int(delay_length))
        self.sample_time = max(1e-6, float(sample_time))
        self.ic = float(initial_condition)
        self.buffer = deque([self.ic] * self.delay_length, maxlen=self.delay_length)
        self.last_sample_t = -1.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        if sample_instant > self.last_sample_t:
            rst = float(self.inputs[1]) if len(self.inputs) > 1 else 0.0
            if abs(rst) > 1e-6:
                self.buffer = deque([self.ic] * self.delay_length, maxlen=self.delay_length)
            else:
                self.buffer.append(float(self.inputs[0]))
            self.last_sample_t = sample_instant
        self.outputs[0] = self.buffer[0]
        return self.outputs

    def reset_states(self) -> None:
        self.buffer = deque([self.ic] * self.delay_length, maxlen=self.delay_length)
        self.last_sample_t = -1.0


class TappedDelayBlock(Block):
    """Delay scalar signal multiple sample periods and output all delayed versions:
    Outputs: [u[k-1], u[k-2], ..., u[k-N]].
    """

    def __init__(self, name: str, num_taps: int = 4, sample_time: float = 0.01, initial_condition: float = 0.0):
        self.num_taps = max(1, int(num_taps))
        super().__init__(name, num_inputs=1, num_outputs=self.num_taps)
        self.sample_time = max(1e-6, float(sample_time))
        self.ic = float(initial_condition)
        self.buffer = deque([self.ic] * self.num_taps, maxlen=self.num_taps)
        self.last_sample_t = -1.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        if sample_instant > self.last_sample_t:
            self.buffer.appendleft(float(self.inputs[0]))
            self.last_sample_t = sample_instant
        for i in range(self.num_taps):
            self.outputs[i] = self.buffer[i]
        return self.outputs

    def reset_states(self) -> None:
        self.buffer = deque([self.ic] * self.num_taps, maxlen=self.num_taps)
        self.last_sample_t = -1.0


PropagationDelayBlock = DelayBlock


class DifferenceBlock(Block):
    """Calculates change in signal over one sample time: y[k] = u[k] - u[k-1]."""

    def __init__(self, name: str, sample_time: float = 0.01, initial_condition: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.sample_time = max(1e-6, float(sample_time))
        self.prev_u = float(initial_condition)
        self.last_sample_t = -1.0
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        diff = u - self.prev_u
        if sample_instant > self.last_sample_t:
            self.prev_u = u
            self.last_sample_t = sample_instant
        self.outputs[0] = diff
        return self.outputs

    def reset_states(self) -> None:
        self.prev_u = 0.0
        self.last_sample_t = -1.0


class DiscreteDerivativeBlock(Block):
    """Discrete-time Derivative: y[k] = (u[k] - u[k-1]) / Ts."""

    def __init__(self, name: str, sample_time: float = 0.01, initial_condition: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.sample_time = max(1e-6, float(sample_time))
        self.prev_u = float(initial_condition)
        self.last_sample_t = -1.0
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        deriv = (u - self.prev_u) / self.sample_time
        if sample_instant > self.last_sample_t:
            self.prev_u = u
            self.last_sample_t = sample_instant
        self.outputs[0] = deriv
        return self.outputs

    def reset_states(self) -> None:
        self.prev_u = 0.0
        self.last_sample_t = -1.0


class DiscreteFIRFilterBlock(Block):
    """Discrete FIR Filter: y[k] = sum_{i=0}^M b[i] * u[k-i]."""

    def __init__(self, name: str, coefficients: Sequence[float], sample_time: float = 0.01):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.b = np.asarray(coefficients, dtype=float)
        self.sample_time = max(1e-6, float(sample_time))
        self.buffer = deque([0.0] * len(self.b), maxlen=len(self.b))
        self.last_sample_t = -1.0
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        if sample_instant > self.last_sample_t:
            self.buffer.appendleft(u)
            self.last_sample_t = sample_instant
        else:
            self.buffer[0] = u

        y = float(np.dot(self.b, list(self.buffer)))
        self.outputs[0] = y
        return self.outputs

    def reset_states(self) -> None:
        self.buffer = deque([0.0] * len(self.b), maxlen=len(self.b))
        self.last_sample_t = -1.0


class DiscreteFilterBlock(Block):
    """Discrete IIR Filter:
    a[0]*y[k] = sum_{i=0}^M b[i]*u[k-i] - sum_{j=1}^N a[j]*y[k-j].
    """

    def __init__(
        self,
        name: str,
        num: Sequence[float],
        den: Sequence[float],
        sample_time: float = 0.01
    ):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.b = np.asarray(num, dtype=float)
        self.a = np.asarray(den, dtype=float)
        # Normalize by a[0]
        if abs(self.a[0]) < 1e-12:
            raise ValueError("Leading denominator coefficient a[0] cannot be zero.")
        self.b = self.b / self.a[0]
        self.a = self.a / self.a[0]

        self.sample_time = max(1e-6, float(sample_time))
        self.u_buf = deque([0.0] * len(self.b), maxlen=len(self.b))
        self.y_buf = deque([0.0] * max(1, len(self.a) - 1), maxlen=max(1, len(self.a) - 1))
        self.last_sample_t = -1.0
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time

        if sample_instant > self.last_sample_t:
            self.u_buf.appendleft(u)
            self.last_sample_t = sample_instant
        else:
            self.u_buf[0] = u

        b_term = float(np.dot(self.b, list(self.u_buf)))
        a_term = float(np.dot(self.a[1:], list(self.y_buf)[:len(self.a)-1])) if len(self.a) > 1 else 0.0
        y = b_term - a_term
        self.outputs[0] = y

        if sample_instant > self.last_sample_t:
            self.y_buf.appendleft(y)
        else:
            self.y_buf[0] = y

        return self.outputs

    def reset_states(self) -> None:
        self.u_buf = deque([0.0] * len(self.b), maxlen=len(self.b))
        self.y_buf = deque([0.0] * max(1, len(self.a) - 1), maxlen=max(1, len(self.a) - 1))
        self.last_sample_t = -1.0


DiscreteTransferFcnBlock = DiscreteFilterBlock


class TransferFcnFirstOrderBlock(Block):
    """Discrete First-Order Transfer Function H(z) = K*(1 - alpha) / (z - alpha)."""

    def __init__(self, name: str, gain: float = 1.0, pole: float = 0.9, sample_time: float = 0.01):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.k = float(gain)
        self.alpha = float(pole)
        self.sample_time = max(1e-6, float(sample_time))
        self.y_prev = 0.0
        self.last_sample_t = -1.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        if sample_instant > self.last_sample_t:
            u = float(self.inputs[0])
            self.y_prev = self.alpha * self.y_prev + self.k * (1.0 - self.alpha) * u
            self.last_sample_t = sample_instant
        self.outputs[0] = self.y_prev
        return self.outputs

    def reset_states(self) -> None:
        self.y_prev = 0.0
        self.last_sample_t = -1.0


class TransferFcnLeadOrLagBlock(DiscreteFilterBlock):
    """Discrete Lead or Lag Compensator: H(z) = ((1 - a)*z + (a - 1)*b) / (z - b)."""

    def __init__(self, name: str, a: float = 0.5, b: float = 0.8, sample_time: float = 0.01):
        num = [1.0 - a, (a - 1.0) * b]
        den = [1.0, -b]
        super().__init__(name, num=num, den=den, sample_time=sample_time)


class TransferFcnRealZeroBlock(DiscreteFilterBlock):
    """Discrete Transfer Function with real zero and no pole: H(z) = gain * (z - z0) / z."""

    def __init__(self, name: str, zero: float = 0.5, gain: float = 1.0, sample_time: float = 0.01):
        num = [gain, -gain * zero]
        den = [1.0, 0.0]
        super().__init__(name, num=num, den=den, sample_time=sample_time)


class TransferFcnDirectFormIIBlock(Block):
    """Discrete Transfer Function Direct Form II Realization."""

    def __init__(self, name: str, num: Sequence[float], den: Sequence[float], sample_time: float = 0.01):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.b = np.asarray(num, dtype=float)
        self.a = np.asarray(den, dtype=float)
        if abs(self.a[0]) < 1e-12:
            raise ValueError("Denominator leading coefficient must be non-zero.")
        self.b = self.b / self.a[0]
        self.a = self.a / self.a[0]

        self.order = max(len(self.b), len(self.a)) - 1
        # Pad b and a to order + 1
        b_pad = np.zeros(self.order + 1)
        b_pad[:len(self.b)] = self.b
        self.b = b_pad
        a_pad = np.zeros(self.order + 1)
        a_pad[:len(self.a)] = self.a
        self.a = a_pad

        self.sample_time = max(1e-6, float(sample_time))
        self.v_states = np.zeros(self.order, dtype=float)
        self.last_sample_t = -1.0
        self.direct_feedthrough = bool(abs(self.b[0]) > 1e-12)

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time

        # v[k] = u[k] - sum_{j=1}^N a[j]*v[k-j]
        v0 = u - float(np.dot(self.a[1:], self.v_states))
        # y[k] = b[0]*v[k] + sum_{j=1}^M b[j]*v[k-j]
        y = self.b[0] * v0 + float(np.dot(self.b[1:], self.v_states))

        if sample_instant > self.last_sample_t:
            if self.order > 0:
                self.v_states = np.roll(self.v_states, 1)
                self.v_states[0] = v0
            self.last_sample_t = sample_instant

        self.outputs[0] = y
        return self.outputs

    def reset_states(self) -> None:
        self.v_states = np.zeros(self.order, dtype=float)
        self.last_sample_t = -1.0


class TransferFcnDirectFormIITimeVaryingBlock(TransferFcnDirectFormIIBlock):
    """Discrete Direct Form II with dynamic/time-varying coefficient updates."""
    pass


class DiscreteZeroPoleBlock(DiscreteFilterBlock):
    """Discrete Zero-Pole-Gain transfer function: H(z) = K * prod(z - z_i) / prod(z - p_i)."""

    def __init__(self, name: str, zeros: Sequence[float], poles: Sequence[float], gain: float = 1.0, sample_time: float = 0.01):
        z = np.asarray(zeros, dtype=complex)
        p = np.asarray(poles, dtype=complex)
        k = float(gain)
        num, den = signal.zpk2tf(z, p, k)
        num = np.real(num)
        den = np.real(den)
        super().__init__(name, num=num, den=den, sample_time=sample_time)


class DiscreteStateSpaceBlock(Block):
    """Discrete State-Space System: x[k+1] = A*x[k] + B*u[k], y[k] = C*x[k] + D*u[k]."""

    def __init__(
        self,
        name: str,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        sample_time: float = 0.01,
        x0: Optional[Sequence[float]] = None
    ):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.D = np.asarray(D, dtype=float)

        if self.B.ndim == 1:
            self.B = self.B.reshape(-1, 1)
        if self.C.ndim == 1:
            self.C = self.C.reshape(1, -1)
        if self.D.ndim == 1:
            self.D = self.D.reshape(self.C.shape[0], self.B.shape[1])

        num_in = self.B.shape[1]
        num_out = self.C.shape[0]
        super().__init__(name, num_inputs=num_in, num_outputs=num_out)

        self.sample_time = max(1e-6, float(sample_time))
        self.x0 = np.asarray(x0, dtype=float) if x0 is not None else np.zeros(self.A.shape[0], dtype=float)
        self.disc_states = np.array(self.x0, dtype=float)
        self.last_sample_t = -1.0
        self.direct_feedthrough = bool(np.any(np.abs(self.D) > 1e-12))

    def compute_output(self, t: float) -> List[float]:
        u = np.array(self.inputs, dtype=float).reshape(-1, 1)
        x = self.disc_states.reshape(-1, 1)
        y = self.C @ x + self.D @ u

        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        if sample_instant > self.last_sample_t:
            x_next = self.A @ x + self.B @ u
            self.disc_states = x_next.flatten()
            self.last_sample_t = sample_instant

        self.outputs = [float(val) for val in y.flatten()]
        return self.outputs

    def reset_states(self) -> None:
        self.disc_states = np.array(self.x0, dtype=float)
        self.last_sample_t = -1.0


FixedPointStateSpaceBlock = DiscreteStateSpaceBlock


class DiscreteTimeIntegratorBlock(Block):
    """Discrete-Time Integrator / Accumulator.
    Methods: 'Forward Euler' (x[k+1] = x[k] + Ts*u[k]),
             'Backward Euler' (x[k] = x[k-1] + Ts*u[k]),
             'Trapezoidal' (x[k] = x[k-1] + Ts/2*(u[k] + u[k-1])).
    """

    def __init__(
        self,
        name: str,
        sample_time: float = 0.01,
        initial_condition: float = 0.0,
        integration_method: str = "Forward Euler",
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None
    ):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.sample_time = max(1e-6, float(sample_time))
        self.ic = float(initial_condition)
        self.method = integration_method.lower()
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

        self.state_x = self.ic
        self.prev_u = 0.0
        self.last_sample_t = -1.0
        self.direct_feedthrough = "backward" in self.method or "trapezoid" in self.method

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time

        out_val = self.state_x
        if "backward" in self.method:
            out_val = self.state_x + self.sample_time * u
        elif "trapezoid" in self.method:
            out_val = self.state_x + (self.sample_time / 2.0) * (u + self.prev_u)

        if self.lower_limit is not None:
            out_val = max(self.lower_limit, out_val)
        if self.upper_limit is not None:
            out_val = min(self.upper_limit, out_val)

        if sample_instant > self.last_sample_t:
            if "forward" in self.method:
                self.state_x += self.sample_time * u
            else:
                self.state_x = out_val
            if self.lower_limit is not None:
                self.state_x = max(self.lower_limit, self.state_x)
            if self.upper_limit is not None:
                self.state_x = min(self.upper_limit, self.state_x)
            self.prev_u = u
            self.last_sample_t = sample_instant

        self.outputs[0] = out_val
        return self.outputs

    def reset_states(self) -> None:
        self.state_x = self.ic
        self.prev_u = 0.0
        self.last_sample_t = -1.0


class DiscretePIDBlock(Block):
    """Discrete PID Controller."""

    def __init__(
        self,
        name: str,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        sample_time: float = 0.01,
        n_filter: float = 100.0,
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None
    ):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.sample_time = max(1e-6, float(sample_time))
        self.n_filter = float(n_filter)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

        self.i_state = 0.0
        self.d_state = 0.0
        self.prev_e = 0.0
        self.last_sample_t = -1.0
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        e = float(self.inputs[0])
        p_term = self.kp * e
        i_term = self.ki * self.i_state
        d_term = self.kd * (e - self.prev_e) / self.sample_time
        u_raw = p_term + i_term + d_term

        u_out = u_raw
        if self.lower_limit is not None:
            u_out = max(self.lower_limit, u_out)
        if self.upper_limit is not None:
            u_out = min(self.upper_limit, u_out)

        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        if sample_instant > self.last_sample_t:
            # Clamped integrator update
            if not (self.upper_limit is not None and u_out >= self.upper_limit and e > 0) and \
               not (self.lower_limit is not None and u_out <= self.lower_limit and e < 0):
                self.i_state += self.sample_time * e
            self.prev_e = e
            self.last_sample_t = sample_instant

        self.outputs[0] = u_out
        return self.outputs

    def reset_states(self) -> None:
        self.i_state = 0.0
        self.d_state = 0.0
        self.prev_e = 0.0
        self.last_sample_t = -1.0


DiscretePIDControllerBlock = DiscretePIDBlock


class DiscretePIDController2DOFBlock(Block):
    """Discrete Two-Degree-of-Freedom PID Controller:
    Inputs: [r, y].
    """

    def __init__(
        self,
        name: str,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        b: float = 1.0,
        c: float = 0.0,
        sample_time: float = 0.01,
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None
    ):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.b = float(b)
        self.c = float(c)
        self.sample_time = max(1e-6, float(sample_time))
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

        self.i_state = 0.0
        self.prev_d_in = 0.0
        self.last_sample_t = -1.0
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        r = float(self.inputs[0]) if len(self.inputs) > 0 else 0.0
        y = float(self.inputs[1]) if len(self.inputs) > 1 else 0.0
        e = r - y
        d_in = self.c * r - y

        p_term = self.kp * (self.b * r - y)
        i_term = self.ki * self.i_state
        d_term = self.kd * (d_in - self.prev_d_in) / self.sample_time
        u_raw = p_term + i_term + d_term

        u_out = u_raw
        if self.lower_limit is not None:
            u_out = max(self.lower_limit, u_out)
        if self.upper_limit is not None:
            u_out = min(self.upper_limit, u_out)

        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        if sample_instant > self.last_sample_t:
            if not (self.upper_limit is not None and u_out >= self.upper_limit and e > 0) and \
               not (self.lower_limit is not None and u_out <= self.lower_limit and e < 0):
                self.i_state += self.sample_time * e
            self.prev_d_in = d_in
            self.last_sample_t = sample_instant

        self.outputs[0] = u_out
        return self.outputs

    def reset_states(self) -> None:
        self.i_state = 0.0
        self.prev_d_in = 0.0
        self.last_sample_t = -1.0
