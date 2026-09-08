"""Continuous and state-space simulation blocks for Dynamic Systems."""

from __future__ import annotations
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.signal as signal
from CORE.common_math import parse_eng_unit, sanitize_array
from .base import Block


class IntegratorBlock(Block):
    """Continuous Integrator: x_dot = u, y = x.
    Features: initial conditions, saturation limits [lower_limit, upper_limit],
    anti-windup derivative freezing, and external reset triggers.
    """

    def __init__(
        self,
        name: str,
        initial_condition: float = 0.0,
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None,
        reset_type: str = "none",  # 'none', 'rising', 'falling', 'level'
        x0: Optional[float] = None
    ):
        num_inputs = 2 if reset_type != "none" else 1
        super().__init__(name, num_inputs=num_inputs, num_outputs=1)
        init_val = x0 if x0 is not None else initial_condition
        self.initial_condition = float(init_val)
        self.lower_limit = float(lower_limit) if lower_limit is not None else None
        self.upper_limit = float(upper_limit) if upper_limit is not None else None
        self.reset_type = reset_type.lower()
        self.prev_reset_in = 0.0

        self.states = np.array([self.initial_condition], dtype=float)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        val = float(self.states[0])
        if self.lower_limit is not None:
            val = max(self.lower_limit, val)
        if self.upper_limit is not None:
            val = min(self.upper_limit, val)
        self.outputs[0] = val
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        u = float(self.inputs[0]) if len(self.inputs) > 0 else 0.0

        # Handle external reset if configured
        if self.reset_type != "none" and len(self.inputs) > 1:
            rst = float(self.inputs[1])
            do_reset = False
            if self.reset_type == "level" and abs(rst) > 1e-6:
                do_reset = True
            elif self.reset_type == "rising" and self.prev_reset_in <= 0.0 and rst > 0.0:
                do_reset = True
            elif self.reset_type == "falling" and self.prev_reset_in >= 0.0 and rst < 0.0:
                do_reset = True
            self.prev_reset_in = rst

            if do_reset:
                self.states[0] = self.initial_condition
                return np.array([0.0], dtype=float)

        # Anti-windup clamping: freeze derivative if at limit and driving further into saturation
        curr_x = self.states[0]
        if self.upper_limit is not None and curr_x >= self.upper_limit and u > 0:
            return np.array([0.0], dtype=float)
        if self.lower_limit is not None and curr_x <= self.lower_limit and u < 0:
            return np.array([0.0], dtype=float)

        return np.array([u], dtype=float)

    def reset_states(self) -> None:
        self.states = np.array([self.initial_condition], dtype=float)
        self.prev_reset_in = 0.0


class IntegratorLimitedBlock(IntegratorBlock):
    """Integrate signal with strict upper and lower saturation bounds."""

    def __init__(
        self,
        name: str,
        initial_condition: float = 0.0,
        lower_limit: float = -1.0,
        upper_limit: float = 1.0,
        reset_type: str = "none"
    ):
        super().__init__(
            name=name,
            initial_condition=initial_condition,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            reset_type=reset_type
        )


class SecondOrderIntegratorBlock(Block):
    """Second-Order Integrator: d2x/dt2 = u.
    States: [x, dx/dt].
    Outputs: [x, dx/dt] or [x].
    """

    def __init__(
        self,
        name: str,
        ic_x: float = 0.0,
        ic_dx: float = 0.0,
        lower_limit_x: Optional[float] = None,
        upper_limit_x: Optional[float] = None,
        lower_limit_dx: Optional[float] = None,
        upper_limit_dx: Optional[float] = None,
        output_both: bool = True
    ):
        num_outputs = 2 if output_both else 1
        super().__init__(name, num_inputs=1, num_outputs=num_outputs)
        self.ic_x = float(ic_x)
        self.ic_dx = float(ic_dx)
        self.lower_limit_x = lower_limit_x
        self.upper_limit_x = upper_limit_x
        self.lower_limit_dx = lower_limit_dx
        self.upper_limit_dx = upper_limit_dx
        self.output_both = output_both

        self.states = np.array([self.ic_x, self.ic_dx], dtype=float)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        x = float(self.states[0])
        dx = float(self.states[1])
        if self.lower_limit_x is not None:
            x = max(self.lower_limit_x, x)
        if self.upper_limit_x is not None:
            x = min(self.upper_limit_x, x)
        if self.lower_limit_dx is not None:
            dx = max(self.lower_limit_dx, dx)
        if self.upper_limit_dx is not None:
            dx = min(self.upper_limit_dx, dx)

        self.outputs[0] = x
        if self.output_both and len(self.outputs) > 1:
            self.outputs[1] = dx
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        u = float(self.inputs[0])
        x = self.states[0]
        dx = self.states[1]

        # Velocity derivative
        ddx = u

        # Clamping
        if self.upper_limit_dx is not None and dx >= self.upper_limit_dx and ddx > 0:
            ddx = 0.0
        elif self.lower_limit_dx is not None and dx <= self.lower_limit_dx and ddx < 0:
            ddx = 0.0

        if self.upper_limit_x is not None and x >= self.upper_limit_x and dx > 0:
            dx = 0.0
        elif self.lower_limit_x is not None and x <= self.lower_limit_x and dx < 0:
            dx = 0.0

        return np.array([dx, ddx], dtype=float)

    def reset_states(self) -> None:
        self.states = np.array([self.ic_x, self.ic_dx], dtype=float)


class SecondOrderIntegratorLimitedBlock(SecondOrderIntegratorBlock):
    """Second-Order Integrator with required saturation bounds on position and velocity."""

    def __init__(
        self,
        name: str,
        ic_x: float = 0.0,
        ic_dx: float = 0.0,
        lower_limit_x: float = -10.0,
        upper_limit_x: float = 10.0,
        lower_limit_dx: float = -5.0,
        upper_limit_dx: float = 5.0,
        output_both: bool = True
    ):
        super().__init__(
            name=name,
            ic_x=ic_x,
            ic_dx=ic_dx,
            lower_limit_x=lower_limit_x,
            upper_limit_x=upper_limit_x,
            lower_limit_dx=lower_limit_dx,
            upper_limit_dx=upper_limit_dx,
            output_both=output_both
        )


class DerivativeBlock(Block):
    """Filtered Derivative Block: H(s) = s / (tau * s + 1).
    Avoids infinite high-frequency noise amplification of pure derivative s.
    """

    def __init__(self, name: str, tau: float = 0.01):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.tau = max(1e-6, float(tau))
        # State: x (filtered state)
        self.states = np.zeros(1, dtype=float)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        x = self.states[0]
        # y = (u - x) / tau
        self.outputs[0] = (u - x) / self.tau
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        u = float(self.inputs[0])
        x = self.states[0]
        # dx/dt = (u - x) / tau
        return np.array([(u - x) / self.tau], dtype=float)

    def reset_states(self) -> None:
        self.states = np.zeros(1, dtype=float)


class TransferFunctionBlock(Block):
    """Continuous LTI Transfer Function H(s) = Num(s) / Den(s) in Controllable Canonical State-Space."""

    def __init__(self, name: str, num: Sequence[float], den: Sequence[float]):
        super().__init__(name, num_inputs=1, num_outputs=1)
        num_arr = np.asarray(num, dtype=float)
        den_arr = np.asarray(den, dtype=float)

        tf_sys = signal.TransferFunction(num_arr, den_arr)
        ss_sys = tf_sys.to_ss()

        self.A = np.asarray(ss_sys.A, dtype=float)
        self.B = np.asarray(ss_sys.B, dtype=float)
        self.C = np.asarray(ss_sys.C, dtype=float)
        self.D = np.asarray(ss_sys.D, dtype=float)

        self.num = list(num_arr)
        self.den = list(den_arr)

        self.states = np.zeros(self.A.shape[0], dtype=float)
        self.direct_feedthrough = bool(np.abs(self.D[0, 0]) > 1e-12)

    def compute_output(self, t: float) -> List[float]:
        u = np.array([[float(self.inputs[0])]], dtype=float)
        x = self.states.reshape(-1, 1)
        y = self.C @ x + self.D @ u
        self.outputs[0] = float(y[0, 0])
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        u = np.array([[float(self.inputs[0])]], dtype=float)
        x = self.states.reshape(-1, 1)
        x_dot = self.A @ x + self.B @ u
        return x_dot.flatten()

    def reset_states(self) -> None:
        self.states = np.zeros(self.A.shape[0], dtype=float)


TransferFcnBlock = TransferFunctionBlock


class ZeroPoleBlock(Block):
    """Continuous Zero-Pole-Gain transfer function:
    H(s) = K * prod(s - z_i) / prod(s - p_i).
    """

    def __init__(self, name: str, zeros: Sequence[float], poles: Sequence[float], gain: float = 1.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        z = np.asarray(zeros, dtype=complex)
        p = np.asarray(poles, dtype=complex)
        k = float(gain)

        num, den = signal.zpk2tf(z, p, k)
        num = np.real(num)
        den = np.real(den)

        tf_sys = signal.TransferFunction(num, den)
        ss_sys = tf_sys.to_ss()

        self.A = np.asarray(ss_sys.A, dtype=float)
        self.B = np.asarray(ss_sys.B, dtype=float)
        self.C = np.asarray(ss_sys.C, dtype=float)
        self.D = np.asarray(ss_sys.D, dtype=float)

        self.states = np.zeros(self.A.shape[0], dtype=float)
        self.direct_feedthrough = bool(np.abs(self.D[0, 0]) > 1e-12)

    def compute_output(self, t: float) -> List[float]:
        u = np.array([[float(self.inputs[0])]], dtype=float)
        x = self.states.reshape(-1, 1)
        y = self.C @ x + self.D @ u
        self.outputs[0] = float(y[0, 0])
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        u = np.array([[float(self.inputs[0])]], dtype=float)
        x = self.states.reshape(-1, 1)
        x_dot = self.A @ x + self.B @ u
        return x_dot.flatten()

    def reset_states(self) -> None:
        self.states = np.zeros(self.A.shape[0], dtype=float)


class StateSpaceBlock(Block):
    """Linear State-Space representation: dx/dt = A*x + B*u, y = C*x + D*u."""

    def __init__(
        self,
        name: str,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
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

        self.x0 = np.asarray(x0, dtype=float) if x0 is not None else np.zeros(self.A.shape[0], dtype=float)
        self.states = np.array(self.x0, dtype=float)
        self.direct_feedthrough = bool(np.any(np.abs(self.D) > 1e-12))

    def compute_output(self, t: float) -> List[float]:
        u = np.array(self.inputs, dtype=float).reshape(-1, 1)
        x = self.states.reshape(-1, 1)
        y = self.C @ x + self.D @ u
        self.outputs = [float(val) for val in y.flatten()]
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        u = np.array(self.inputs, dtype=float).reshape(-1, 1)
        x = self.states.reshape(-1, 1)
        x_dot = self.A @ x + self.B @ u
        return x_dot.flatten()

    def reset_states(self) -> None:
        self.states = np.array(self.x0, dtype=float)


class DescriptorStateSpaceBlock(Block):
    """Descriptor State-Space representation: E*dx/dt = A*x + B*u, y = C*x + D*u.
    Solves for dx/dt = pinv(E) * (A*x + B*u).
    """

    def __init__(
        self,
        name: str,
        E: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        x0: Optional[Sequence[float]] = None
    ):
        self.E = np.asarray(E, dtype=float)
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

        # Compute pseudo-inverse or inverse of E
        try:
            self.E_inv = np.linalg.inv(self.E)
        except np.linalg.LinAlgError:
            self.E_inv = np.linalg.pinv(self.E)

        num_in = self.B.shape[1]
        num_out = self.C.shape[0]
        super().__init__(name, num_inputs=num_in, num_outputs=num_out)

        self.x0 = np.asarray(x0, dtype=float) if x0 is not None else np.zeros(self.A.shape[0], dtype=float)
        self.states = np.array(self.x0, dtype=float)
        self.direct_feedthrough = bool(np.any(np.abs(self.D) > 1e-12))

    def compute_output(self, t: float) -> List[float]:
        u = np.array(self.inputs, dtype=float).reshape(-1, 1)
        x = self.states.reshape(-1, 1)
        y = self.C @ x + self.D @ u
        self.outputs = [float(val) for val in y.flatten()]
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        u = np.array(self.inputs, dtype=float).reshape(-1, 1)
        x = self.states.reshape(-1, 1)
        rhs = self.A @ x + self.B @ u
        x_dot = self.E_inv @ rhs
        return x_dot.flatten()

    def reset_states(self) -> None:
        self.states = np.array(self.x0, dtype=float)


class TransportDelayBlock(Block):
    """Pure continuous time delay: y(t) = u(t - delay_time)."""

    def __init__(self, name: str, delay_time: float = 0.1, initial_output: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.delay_time = max(1e-6, float(delay_time))
        self.initial_output = float(initial_output)
        self.buffer_time: List[float] = []
        self.buffer_val: List[float] = []
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        # Record current input with timestamp
        self.buffer_time.append(t)
        self.buffer_val.append(float(self.inputs[0]))

        target_t = t - self.delay_time
        if target_t < 0 or len(self.buffer_time) < 2:
            self.outputs[0] = self.initial_output
        else:
            self.outputs[0] = float(np.interp(target_t, self.buffer_time, self.buffer_val))

        # Trim buffer to save memory
        if len(self.buffer_time) > 1000 and target_t > self.buffer_time[200]:
            self.buffer_time = self.buffer_time[200:]
            self.buffer_val = self.buffer_val[200:]

        return self.outputs

    def reset_states(self) -> None:
        self.buffer_time.clear()
        self.buffer_val.clear()


class VariableTimeDelayBlock(Block):
    """Continuous variable time delay: y(t) = u(t - tau(t)).
    Input 0: signal u(t)
    Input 1: variable delay tau(t) (or fixed initial parameter)
    """

    def __init__(self, name: str, max_delay: float = 10.0, initial_output: float = 0.0):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.max_delay = max(1e-6, float(max_delay))
        self.initial_output = float(initial_output)
        self.buffer_time: List[float] = []
        self.buffer_val: List[float] = []
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        self.buffer_time.append(t)
        self.buffer_val.append(float(self.inputs[0]))
        curr_delay = max(0.0, min(float(self.inputs[1]), self.max_delay)) if len(self.inputs) > 1 else 0.0

        target_t = t - curr_delay
        if target_t < 0 or len(self.buffer_time) < 2:
            self.outputs[0] = self.initial_output
        else:
            self.outputs[0] = float(np.interp(target_t, self.buffer_time, self.buffer_val))

        if len(self.buffer_time) > 1000 and (t - self.max_delay) > self.buffer_time[200]:
            self.buffer_time = self.buffer_time[200:]
            self.buffer_val = self.buffer_val[200:]

        return self.outputs

    def reset_states(self) -> None:
        self.buffer_time.clear()
        self.buffer_val.clear()


VariableTransportDelayBlock = VariableTimeDelayBlock
EntityTransportDelayBlock = VariableTimeDelayBlock


class FirstOrderHoldBlock(Block):
    """Linearly extrapolated First-Order Hold (FOH) on sampled signal:
    y(t) = u[k] + (u[k] - u[k-1])/Ts * (t - t_k).
    """

    def __init__(self, name: str, sample_time: float = 0.01):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.sample_time = max(1e-6, float(sample_time))
        self.u_curr = 0.0
        self.u_prev = 0.0
        self.t_last_sample = 0.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        if sample_instant > self.t_last_sample or t == 0.0:
            self.u_prev = self.u_curr
            self.u_curr = float(self.inputs[0])
            self.t_last_sample = sample_instant

        dt_since = max(0.0, t - self.t_last_sample)
        slope = (self.u_curr - self.u_prev) / self.sample_time
        self.outputs[0] = self.u_curr + slope * dt_since
        return self.outputs

    def reset_states(self) -> None:
        self.u_curr = 0.0
        self.u_prev = 0.0
        self.t_last_sample = 0.0


class PIDBlock(Block):
    """Continuous PID Controller with anti-windup clamping and derivative filtering:
    u = Kp*e + Ki*integral(e) + Kd*N*(e - x_f).
    """

    def __init__(
        self,
        name: str,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        n_filter: float = 100.0,
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None
    ):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.n_filter = float(n_filter)
        self.lower_limit = float(lower_limit) if lower_limit is not None else None
        self.upper_limit = float(upper_limit) if upper_limit is not None else None

        # States: [x_integral, x_derivative_filter]
        self.states = np.zeros(2, dtype=float)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        e = float(self.inputs[0])
        x_i, x_d = self.states[0], self.states[1]
        p_term = self.kp * e
        i_term = self.ki * x_i
        d_term = self.kd * self.n_filter * (e - x_d)
        u_raw = p_term + i_term + d_term

        # Output saturation
        u_out = u_raw
        if self.lower_limit is not None:
            u_out = max(self.lower_limit, u_out)
        if self.upper_limit is not None:
            u_out = min(self.upper_limit, u_out)

        self.outputs[0] = u_out
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        e = float(self.inputs[0])
        x_i, x_d = self.states[0], self.states[1]

        # Anti-windup clamping on integrator state
        dx_i = e
        if self.upper_limit is not None and self.outputs[0] >= self.upper_limit and e > 0:
            dx_i = 0.0
        elif self.lower_limit is not None and self.outputs[0] <= self.lower_limit and e < 0:
            dx_i = 0.0

        dx_d = self.n_filter * (e - x_d)
        return np.array([dx_i, dx_d], dtype=float)

    def reset_states(self) -> None:
        self.states = np.zeros(2, dtype=float)


PIDControllerBlock = PIDBlock


class PIDController2DOFBlock(Block):
    """Two-Degree-of-Freedom (2DOF) PID Controller:
    Inputs: [r (setpoint), y (measurement)].
    Output: u = P*(b*r - y) + I*integral(r - y) + D*N*(c*r - y - x_f).
    b: setpoint weight on proportional action
    c: setpoint weight on derivative action
    """

    def __init__(
        self,
        name: str,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        n_filter: float = 100.0,
        b: float = 1.0,
        c: float = 0.0,
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None
    ):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.n_filter = float(n_filter)
        self.b = float(b)
        self.c = float(c)
        self.lower_limit = float(lower_limit) if lower_limit is not None else None
        self.upper_limit = float(upper_limit) if upper_limit is not None else None

        # States: [x_integral, x_derivative_filter]
        self.states = np.zeros(2, dtype=float)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        r = float(self.inputs[0]) if len(self.inputs) > 0 else 0.0
        y = float(self.inputs[1]) if len(self.inputs) > 1 else 0.0
        e = r - y

        x_i, x_d = self.states[0], self.states[1]
        p_term = self.kp * (self.b * r - y)
        i_term = self.ki * x_i
        d_input = self.c * r - y
        d_term = self.kd * self.n_filter * (d_input - x_d)
        u_raw = p_term + i_term + d_term

        u_out = u_raw
        if self.lower_limit is not None:
            u_out = max(self.lower_limit, u_out)
        if self.upper_limit is not None:
            u_out = min(self.upper_limit, u_out)

        self.outputs[0] = u_out
        return self.outputs

    def compute_derivatives(self, t: float) -> np.ndarray:
        r = float(self.inputs[0]) if len(self.inputs) > 0 else 0.0
        y = float(self.inputs[1]) if len(self.inputs) > 1 else 0.0
        e = r - y
        x_i, x_d = self.states[0], self.states[1]

        dx_i = e
        if self.upper_limit is not None and self.outputs[0] >= self.upper_limit and e > 0:
            dx_i = 0.0
        elif self.lower_limit is not None and self.outputs[0] <= self.lower_limit and e < 0:
            dx_i = 0.0

        d_input = self.c * r - y
        dx_d = self.n_filter * (d_input - x_d)
        return np.array([dx_i, dx_d], dtype=float)

    def reset_states(self) -> None:
        self.states = np.zeros(2, dtype=float)
