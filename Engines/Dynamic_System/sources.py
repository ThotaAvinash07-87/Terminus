"""Signal Generators and Sources for Dynamic Systems."""

from __future__ import annotations
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from .base import Block


class ConstantBlock(Block):
    """Constant Value Source."""

    def __init__(self, name: str, value: float = 1.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.value = float(value)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = self.value
        return self.outputs


class EnumeratedConstantBlock(Block):
    """Generate enumerated constant value (integer or string token enum)."""

    def __init__(self, name: str, enum_value: Any = 0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.enum_val = enum_value
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = self.enum_val
        return self.outputs


class GroundBlock(Block):
    """Ground unconnected input port: always outputs 0.0."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = 0.0
        return self.outputs


class StepSourceBlock(Block):
    """Step Input Source."""

    def __init__(self, name: str, step_time: float = 0.0, amplitude: float = 1.0, initial_value: float = 0.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.step_time = float(step_time)
        self.amplitude = float(amplitude)
        self.initial_value = float(initial_value)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = self.amplitude if t >= self.step_time else self.initial_value
        return self.outputs


StepBlock = StepSourceBlock


class RampSourceBlock(Block):
    """Ramp Signal Source: y(t) = initial_value + slope * (t - start_time) for t >= start_time."""

    def __init__(self, name: str, slope: float = 1.0, start_time: float = 0.0, initial_value: float = 0.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.slope = float(slope)
        self.start_time = float(start_time)
        self.initial_value = float(initial_value)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        if t >= self.start_time:
            self.outputs[0] = self.initial_value + self.slope * (t - self.start_time)
        else:
            self.outputs[0] = self.initial_value
        return self.outputs


RampBlock = RampSourceBlock


class SineSourceBlock(Block):
    """Sine Wave Generator."""

    def __init__(self, name: str, freq: float = 1.0, amplitude: float = 1.0, offset: float = 0.0, phase: float = 0.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.freq = float(freq)
        self.amplitude = float(amplitude)
        self.offset = float(offset)
        self.phase = float(phase)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = self.offset + self.amplitude * math.sin(2.0 * math.pi * self.freq * t + math.radians(self.phase))
        return self.outputs


SineWaveBlock = SineSourceBlock


class ChirpSignalBlock(Block):
    """Generate sine wave with frequency swept from f0 to f1 at target time t1."""

    def __init__(self, name: str, f0: float = 0.1, f1: float = 10.0, t1: float = 10.0, amplitude: float = 1.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.f0 = float(f0)
        self.f1 = float(f1)
        self.t1 = max(1e-6, float(t1))
        self.amp = float(amplitude)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        # Instantaneous frequency: f(t) = f0 + (f1 - f0) * (t / t1)
        # Phase: phi(t) = 2*pi*(f0*t + 0.5*(f1 - f0)/t1 * t^2)
        phi = 2.0 * math.pi * (self.f0 * t + 0.5 * (self.f1 - self.f0) * (t ** 2) / self.t1)
        self.outputs[0] = self.amp * math.sin(phi)
        return self.outputs


class ClockBlock(Block):
    """Display and provide continuous simulation time t."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = float(t)
        return self.outputs


class DigitalClockBlock(Block):
    """Output simulation time sampled at fixed interval Ts."""

    def __init__(self, name: str, sample_time: float = 0.01):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.ts = max(1e-6, float(sample_time))
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.ts + 1e-9)
        self.outputs[0] = float(k * self.ts)
        return self.outputs


class PulseGeneratorBlock(Block):
    """Pulse Train Generator."""

    def __init__(self, name: str, period: float = 1.0, duty_cycle: float = 0.5, amplitude: float = 1.0, delay: float = 0.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.period = max(1e-6, float(period))
        self.duty = max(0.0, min(1.0, float(duty_cycle)))
        self.amplitude = float(amplitude)
        self.delay = float(delay)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        if t < self.delay:
            self.outputs[0] = 0.0
            return self.outputs
        t_cycle = (t - self.delay) % self.period
        self.outputs[0] = self.amplitude if t_cycle < (self.period * self.duty) else 0.0
        return self.outputs


class BandLimitedWhiteNoiseBlock(Block):
    """Band-Limited Gaussian Random Noise Generator."""

    def __init__(self, name: str, noise_power: float = 0.1, sample_time: float = 0.01, seed: int = 42):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.noise_power = float(noise_power)
        self.sample_time = max(1e-6, float(sample_time))
        self.rng = np.random.RandomState(seed)
        self.curr_val = 0.0
        self.last_sample_t = -1.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.sample_time + 1e-9)
        sample_instant = k * self.sample_time
        if sample_instant > self.last_sample_t:
            std = math.sqrt(self.noise_power / self.sample_time)
            self.curr_val = float(self.rng.normal(0.0, std))
            self.last_sample_t = sample_instant
        self.outputs[0] = self.curr_val
        return self.outputs

    def reset_states(self) -> None:
        self.curr_val = 0.0
        self.last_sample_t = -1.0


class RandomNumberBlock(Block):
    """Normally Distributed Random Number Generator with mean and variance."""

    def __init__(self, name: str, mean: float = 0.0, variance: float = 1.0, sample_time: float = 0.01, seed: int = 42):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.mean = float(mean)
        self.std = math.sqrt(max(0.0, float(variance)))
        self.ts = max(1e-6, float(sample_time))
        self.rng = np.random.RandomState(seed)
        self.curr_val = self.mean
        self.last_sample_t = -1.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.ts + 1e-9)
        sample_instant = k * self.ts
        if sample_instant > self.last_sample_t:
            self.curr_val = float(self.rng.normal(self.mean, self.std))
            self.last_sample_t = sample_instant
        self.outputs[0] = self.curr_val
        return self.outputs

    def reset_states(self) -> None:
        self.curr_val = self.mean
        self.last_sample_t = -1.0


class UniformRandomNumberBlock(Block):
    """Uniformly Distributed Random Number Generator in [min, max]."""

    def __init__(self, name: str, minimum: float = -1.0, maximum: float = 1.0, sample_time: float = 0.01, seed: int = 42):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.min_val = float(minimum)
        self.max_val = float(maximum)
        self.ts = max(1e-6, float(sample_time))
        self.rng = np.random.RandomState(seed)
        self.curr_val = (self.min_val + self.max_val) / 2.0
        self.last_sample_t = -1.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.ts + 1e-9)
        sample_instant = k * self.ts
        if sample_instant > self.last_sample_t:
            self.curr_val = float(self.rng.uniform(self.min_val, self.max_val))
            self.last_sample_t = sample_instant
        self.outputs[0] = self.curr_val
        return self.outputs

    def reset_states(self) -> None:
        self.curr_val = (self.min_val + self.max_val) / 2.0
        self.last_sample_t = -1.0


class RepeatingSequenceBlock(Block):
    """Generate arbitrarily shaped periodic signal defined by time and output values."""

    def __init__(self, name: str, time_values: Sequence[float], output_values: Sequence[float]):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.t_pts = np.asarray(time_values, dtype=float)
        self.y_pts = np.asarray(output_values, dtype=float)
        self.period = self.t_pts[-1] - self.t_pts[0] if len(self.t_pts) > 1 else 1.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        t_mod = self.t_pts[0] + (t % self.period)
        self.outputs[0] = float(np.interp(t_mod, self.t_pts, self.y_pts))
        return self.outputs


RepeatingSequenceInterpolatedBlock = RepeatingSequenceBlock


class RepeatingSequenceStairBlock(Block):
    """Output and repeat discrete-time sequence in stair-step fashion."""

    def __init__(self, name: str, output_values: Sequence[float], sample_time: float = 0.1):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.y_pts = list(output_values)
        self.ts = max(1e-6, float(sample_time))
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.ts + 1e-9)
        idx = k % len(self.y_pts)
        self.outputs[0] = float(self.y_pts[idx])
        return self.outputs


class CounterFreeRunningBlock(Block):
    """Count up 0, 1, 2, ..., 2^num_bits - 1 and overflow wrap back to zero."""

    def __init__(self, name: str, num_bits: int = 8, sample_time: float = 0.01):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.max_count = (1 << int(num_bits))
        self.ts = max(1e-6, float(sample_time))
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.ts + 1e-9)
        self.outputs[0] = float(k % self.max_count)
        return self.outputs


class CounterLimitedBlock(Block):
    """Count up 0, 1, 2, ..., upper_limit and wrap to zero."""

    def __init__(self, name: str, upper_limit: int = 10, sample_time: float = 0.01):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.limit = int(upper_limit) + 1
        self.ts = max(1e-6, float(sample_time))
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        k = int(t / self.ts + 1e-9)
        self.outputs[0] = float(k % self.limit)
        return self.outputs


class SignalGeneratorBlock(Block):
    """Generate various waveforms: 'sine', 'square', 'sawtooth', 'random'."""

    def __init__(self, name: str, wave_type: str = "sine", amplitude: float = 1.0, frequency: float = 1.0):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.wave_type = wave_type.lower()
        self.amp = float(amplitude)
        self.freq = float(frequency)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        tau = (t * self.freq) % 1.0
        if self.wave_type == "sine":
            val = self.amp * math.sin(2.0 * math.pi * self.freq * t)
        elif self.wave_type == "square":
            val = self.amp if tau < 0.5 else -self.amp
        elif self.wave_type == "sawtooth":
            val = self.amp * (2.0 * tau - 1.0)
        elif self.wave_type == "random":
            val = self.amp * (np.random.rand() * 2.0 - 1.0)
        else:
            val = self.amp * math.sin(2.0 * math.pi * self.freq * t)
        self.outputs[0] = float(val)
        return self.outputs


WaveformGeneratorBlock = SignalGeneratorBlock
SignalEditorBlock = RepeatingSequenceBlock


class FromWorkspaceBlock(Block):
    """Load signal data from time-series workspace arrays: time_data, signal_data."""

    def __init__(self, name: str, time_data: Sequence[float], signal_data: Sequence[float]):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.t_arr = np.asarray(time_data, dtype=float)
        self.y_arr = np.asarray(signal_data, dtype=float)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = float(np.interp(t, self.t_arr, self.y_arr))
        return self.outputs


FromFileBlock = FromWorkspaceBlock
FromSpreadsheetBlock = FromWorkspaceBlock
PlaybackBlock = FromWorkspaceBlock
