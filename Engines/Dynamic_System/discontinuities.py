"""Discontinuities and nonlinear actuator simulation blocks for Dynamic Systems."""

from __future__ import annotations
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from .base import Block


class SaturationBlock(Block):
    """Saturation: clamps input signal to [lower_limit, upper_limit]."""

    def __init__(self, name: str, lower_limit: float = -1.0, upper_limit: float = 1.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.lower_limit = float(lower_limit)
        self.upper_limit = float(upper_limit)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        low = min(self.lower_limit, self.upper_limit)
        high = max(self.lower_limit, self.upper_limit)
        self.outputs[0] = max(low, min(u, high))
        return self.outputs


class SaturationDynamicBlock(Block):
    """Dynamic Saturation: clamps input signal u to dynamic [lower_limit, upper_limit] inputs.
    Input 0: u (signal)
    Input 1: upper limit
    Input 2: lower limit
    """

    def __init__(self, name: str):
        super().__init__(name, num_inputs=3, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        upper = float(self.inputs[1]) if len(self.inputs) > 1 else 1.0
        lower = float(self.inputs[2]) if len(self.inputs) > 2 else -1.0
        if lower > upper:
            lower, upper = upper, lower
        self.outputs[0] = max(lower, min(u, upper))
        return self.outputs


class RateLimiterBlock(Block):
    """Rate Limiter: limits the time rate of change of the signal:
    falling_slew_rate <= dy/dt <= rising_slew_rate.
    """

    def __init__(
        self,
        name: str,
        rising_slew_rate: float = 100.0,
        falling_slew_rate: float = -100.0,
        initial_output: Optional[float] = None
    ):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.rising_rate = float(rising_slew_rate)
        self.falling_rate = float(falling_slew_rate)
        self.initial_output = float(initial_output) if initial_output is not None else None
        self.prev_t: Optional[float] = None
        self.prev_y: Optional[float] = self.initial_output
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        if self.prev_t is None:
            self.prev_t = t
            if self.prev_y is None:
                self.prev_y = u
            self.outputs[0] = self.prev_y
            return self.outputs

        if t <= self.prev_t:
            self.outputs[0] = self.prev_y if self.prev_y is not None else u
            return self.outputs

        dt = t - self.prev_t
        rate = (u - self.prev_y) / dt
        limited_rate = max(self.falling_rate, min(rate, self.rising_rate))
        y = self.prev_y + limited_rate * dt

        self.prev_t = t
        self.prev_y = y
        self.outputs[0] = y
        return self.outputs

    def reset_states(self) -> None:
        self.prev_t = None
        self.prev_y = self.initial_output


class RateLimiterDynamicBlock(Block):
    """Dynamic Rate Limiter:
    Input 0: u (signal)
    Input 1: rising limit (R)
    Input 2: falling limit (F)
    """

    def __init__(self, name: str, initial_output: Optional[float] = None):
        super().__init__(name, num_inputs=3, num_outputs=1)
        self.initial_output = float(initial_output) if initial_output is not None else None
        self.prev_t: Optional[float] = None
        self.prev_y: Optional[float] = self.initial_output
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        rising = float(self.inputs[1]) if len(self.inputs) > 1 else 100.0
        falling = float(self.inputs[2]) if len(self.inputs) > 2 else -100.0

        if self.prev_t is None:
            self.prev_t = t
            if self.prev_y is None:
                self.prev_y = u
            self.outputs[0] = self.prev_y
            return self.outputs

        if t <= self.prev_t:
            self.outputs[0] = self.prev_y if self.prev_y is not None else u
            return self.outputs

        dt = t - self.prev_t
        rate = (u - self.prev_y) / dt
        limited_rate = max(falling, min(rate, rising))
        y = self.prev_y + limited_rate * dt

        self.prev_t = t
        self.prev_y = y
        self.outputs[0] = y
        return self.outputs

    def reset_states(self) -> None:
        self.prev_t = None
        self.prev_y = self.initial_output


class DeadZoneBlock(Block):
    """Dead Zone: outputs 0 when lower_limit <= u <= upper_limit, else offsets proportionally."""

    def __init__(self, name: str, start_zone: float = -0.5, end_zone: float = 0.5):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.start_zone = float(start_zone)
        self.end_zone = float(end_zone)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        if u > self.end_zone:
            self.outputs[0] = u - self.end_zone
        elif u < self.start_zone:
            self.outputs[0] = u - self.start_zone
        else:
            self.outputs[0] = 0.0
        return self.outputs


class DeadZoneDynamicBlock(Block):
    """Dynamic Dead Zone:
    Input 0: u (signal)
    Input 1: upper limit (end zone)
    Input 2: lower limit (start zone)
    """

    def __init__(self, name: str):
        super().__init__(name, num_inputs=3, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        end_z = float(self.inputs[1]) if len(self.inputs) > 1 else 0.5
        start_z = float(self.inputs[2]) if len(self.inputs) > 2 else -0.5
        if start_z > end_z:
            start_z, end_z = end_z, start_z

        if u > end_z:
            self.outputs[0] = u - end_z
        elif u < start_z:
            self.outputs[0] = u - start_z
        else:
            self.outputs[0] = 0.0
        return self.outputs


class BacklashBlock(Block):
    """Mechanical Backlash / Hysteresis with deadband width."""

    def __init__(self, name: str, deadband_width: float = 1.0, initial_output: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.deadband = float(deadband_width)
        self.prev_y = float(initial_output)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        half_band = self.deadband / 2.0
        if u > self.prev_y + half_band:
            self.prev_y = u - half_band
        elif u < self.prev_y - half_band:
            self.prev_y = u + half_band
        self.outputs[0] = self.prev_y
        return self.outputs

    def reset_states(self) -> None:
        self.prev_y = 0.0


class RelayBlock(Block):
    """Relay / Schmitt Trigger (Bang-Bang controller with turn-on/turn-off hysteresis thresholds)."""

    def __init__(
        self,
        name: str,
        switch_on_point: float = 0.5,
        switch_off_point: float = -0.5,
        output_on: float = 1.0,
        output_off: float = 0.0
    ):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.on_thresh = float(switch_on_point)
        self.off_thresh = float(switch_off_point)
        self.y_on = float(output_on)
        self.y_off = float(output_off)
        self.state_on = False
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        if u >= self.on_thresh:
            self.state_on = True
        elif u <= self.off_thresh:
            self.state_on = False
        self.outputs[0] = self.y_on if self.state_on else self.y_off
        return self.outputs

    def reset_states(self) -> None:
        self.state_on = False


class CoulombViscousFrictionBlock(Block):
    """Realistic Mechanical Friction: F = F_coulomb * sgn(v) + b_viscous * v + F_static * exp(-|v|/v_stribeck)."""

    def __init__(
        self,
        name: str,
        f_coulomb: float = 1.0,
        b_viscous: float = 0.1,
        f_static: float = 1.5,
        v_stribeck: float = 0.05
    ):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.fc = float(f_coulomb)
        self.b = float(b_viscous)
        self.fs = float(f_static)
        self.vs = max(1e-4, float(v_stribeck))
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        v = float(self.inputs[0])
        stribeck = (self.fs - self.fc) * math.exp(-abs(v) / self.vs)
        friction = (self.fc + stribeck) * np.tanh(v * 50.0) + self.b * v
        self.outputs[0] = float(friction)
        return self.outputs


class QuantizerBlock(Block):
    """Quantization of continuous signal to discrete steps of size q."""

    def __init__(self, name: str, quantization_interval: float = 0.1):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.q = max(1e-9, float(quantization_interval))
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        self.outputs[0] = round(u / self.q) * self.q
        return self.outputs


class HitCrossingBlock(Block):
    """Detects when input signal crosses an offset threshold value.
    Directions: 'rising', 'falling', 'either'.
    Outputs 1.0 at crossing instant, 0.0 otherwise.
    """

    def __init__(self, name: str, hit_crossing_offset: float = 0.0, direction: str = "either"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.offset = float(hit_crossing_offset)
        self.direction = direction.lower()
        self.prev_u: Optional[float] = None
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        hit = 0.0
        if self.prev_u is not None:
            if self.direction == "rising" and self.prev_u < self.offset and u >= self.offset:
                hit = 1.0
            elif self.direction == "falling" and self.prev_u > self.offset and u <= self.offset:
                hit = 1.0
            elif self.direction == "either" and ((self.prev_u < self.offset and u >= self.offset) or (self.prev_u > self.offset and u <= self.offset)):
                hit = 1.0

        self.prev_u = u
        self.outputs[0] = hit
        return self.outputs

    def reset_states(self) -> None:
        self.prev_u = None


class PWMBlock(Block):
    """Generate ideal Pulse-Width Modulated (PWM) signal corresponding to input duty cycle [0, 1]."""

    def __init__(self, name: str, frequency: float = 1000.0, amplitude: float = 1.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.freq = max(1e-3, float(frequency))
        self.period = 1.0 / self.freq
        self.amplitude = float(amplitude)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        duty = max(0.0, min(1.0, float(self.inputs[0])))
        t_phase = (t % self.period) / self.period
        self.outputs[0] = self.amplitude if t_phase < duty else 0.0
        return self.outputs


class VariablePulseGeneratorBlock(Block):
    """Generate ideal time-varying pulse signal:
    Input 0: period
    Input 1: duty cycle (0..1)
    Input 2: amplitude
    """

    def __init__(self, name: str, default_period: float = 1.0, default_duty: float = 0.5, default_amplitude: float = 1.0):
        super().__init__(name, num_inputs=3, num_outputs=1)
        self.def_period = float(default_period)
        self.def_duty = float(default_duty)
        self.def_amp = float(default_amplitude)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        period = max(1e-6, float(self.inputs[0])) if len(self.inputs) > 0 and self.inputs[0] > 0 else self.def_period
        duty = max(0.0, min(1.0, float(self.inputs[1]))) if len(self.inputs) > 1 else self.def_duty
        amp = float(self.inputs[2]) if len(self.inputs) > 2 else self.def_amp

        t_cycle = t % period
        self.outputs[0] = amp if t_cycle < (period * duty) else 0.0
        return self.outputs


class WrapToZeroBlock(Block):
    """Set output to zero if input exceeds threshold or falls below zero."""

    def __init__(self, name: str, threshold: float = 10.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.threshold = float(threshold)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        if u > self.threshold or u < 0.0:
            self.outputs[0] = 0.0
        else:
            self.outputs[0] = u
        return self.outputs
