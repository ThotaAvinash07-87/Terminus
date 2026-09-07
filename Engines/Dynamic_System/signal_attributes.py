"""Signal Attributes, Data Types, Probes, and Conversion blocks for Dynamic Systems."""

from __future__ import annotations
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from .base import Block


class ICBlock(Block):
    """Initial Condition (IC) Block: outputs initial_value at t=0, then passes input u through."""

    def __init__(self, name: str, initial_value: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.ic = float(initial_value)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        if t <= 0.0:
            self.outputs[0] = self.ic
        else:
            self.outputs[0] = self.inputs[0]
        return self.outputs


class DataTypeConversionBlock(Block):
    """Convert input signal to specified data type: 'double', 'single', 'int32', 'int16', 'uint8', 'boolean'."""

    TYPE_MAP = {
        "double": float,
        "single": np.float32,
        "float": float,
        "int": int,
        "int32": np.int32,
        "int16": np.int16,
        "int8": np.int8,
        "uint32": np.uint32,
        "uint16": np.uint16,
        "uint8": np.uint8,
        "boolean": bool,
        "bool": bool,
    }

    def __init__(self, name: str, target_data_type: str = "double"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.target_type_str = target_data_type.lower()
        self.cast_fn = self.TYPE_MAP.get(self.target_type_str, float)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        val = self.inputs[0]
        if self.target_type_str in ("bool", "boolean"):
            self.outputs[0] = 1.0 if bool(abs(float(val)) > 1e-6) else 0.0
        else:
            self.outputs[0] = float(self.cast_fn(val))
        return self.outputs


class DataTypeConversionInheritedBlock(DataTypeConversionBlock):
    """Convert from one data type to another using inherited type."""
    pass


class DataTypeDuplicateBlock(Block):
    """Force all inputs to same data type."""

    def __init__(self, name: str, num_inputs: int = 2):
        super().__init__(name, num_inputs=num_inputs, num_outputs=0)
        self.direct_feedthrough = True


class DataTypePropagationBlock(Block):
    """Set data type based on reference signals."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = self.inputs[0]
        return self.outputs


class DataTypeScalingStripBlock(Block):
    """Remove scaling and map fixed-point to integer."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = float(int(round(float(self.inputs[0]))))
        return self.outputs


class ProbeBlock(Block):
    """Output signal attributes: width, sample time, complex flag.
    Outputs: [width (int), is_complex (0 or 1)].
    """

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=2)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        val = self.inputs[0]
        if isinstance(val, (list, tuple, np.ndarray)):
            w = len(val)
        else:
            w = 1
        is_c = 1.0 if isinstance(val, complex) or np.iscomplexobj(val) else 0.0
        self.outputs[0] = float(w)
        self.outputs[1] = is_c
        return self.outputs


class WidthBlock(Block):
    """Output width/dimension of input vector."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        val = self.inputs[0]
        if isinstance(val, (list, tuple, np.ndarray)):
            self.outputs[0] = float(len(val))
        else:
            self.outputs[0] = 1.0
        return self.outputs


class RateTransitionBlock(Block):
    """Handle data transfer between blocks operating at different sample rates."""

    def __init__(self, name: str, out_sample_time: float = 0.01):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.out_ts = max(1e-6, float(out_sample_time))
        self.held_val = 0.0
        self.last_sample_t = -1.0
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[Any]:
        k = int(t / self.out_ts + 1e-9)
        sample_instant = k * self.out_ts
        if sample_instant > self.last_sample_t:
            self.held_val = self.inputs[0]
            self.last_sample_t = sample_instant
        self.outputs[0] = self.held_val
        return self.outputs

    def reset_states(self) -> None:
        self.held_val = 0.0
        self.last_sample_t = -1.0


class SignalConversionBlock(Block):
    """Convert virtual signals or bus signals to nonvirtual continuous arrays."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = self.inputs[0]
        return self.outputs


class SignalSpecificationBlock(Block):
    """Specify desired dimensions and attributes of signal."""

    def __init__(self, name: str, dimensions: int = 1):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.dims = dimensions
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = self.inputs[0]
        return self.outputs


class UnitConversionBlock(Block):
    """Convert physical engineering units (e.g. deg to rad, C to K, km/h to m/s)."""

    UNIT_CONVERSIONS = {
        ("deg", "rad"): math.radians,
        ("rad", "deg"): math.degrees,
        ("c", "k"): lambda c: c + 273.15,
        ("k", "c"): lambda k: k - 273.15,
        ("f", "c"): lambda f: (f - 32.0) * 5.0 / 9.0,
        ("c", "f"): lambda c: (c * 9.0 / 5.0) + 32.0,
        ("km/h", "m/s"): lambda v: v / 3.6,
        ("m/s", "km/h"): lambda v: v * 3.6,
        ("rpm", "rad/s"): lambda r: r * 2.0 * math.pi / 60.0,
        ("rad/s", "rpm"): lambda w: w * 60.0 / (2.0 * math.pi),
    }

    def __init__(self, name: str, input_unit: str = "deg", output_unit: str = "rad"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.u_in = input_unit.lower()
        self.u_out = output_unit.lower()
        self.conv = self.UNIT_CONVERSIONS.get((self.u_in, self.u_out), lambda x: x)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        self.outputs[0] = float(self.conv(u))
        return self.outputs


class WeightedSampleTimeBlock(Block):
    """Scale signal by sample time."""

    def __init__(self, name: str, sample_time: float = 0.01):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.ts = max(1e-6, float(sample_time))
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        self.outputs[0] = float(self.inputs[0]) * self.ts
        return self.outputs


class BusToVectorBlock(Block):
    """Convert bus values into a flat numeric vector."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        inp = self.inputs[0]
        if isinstance(inp, dict):
            vec = [float(v) for v in inp.values()]
            self.outputs[0] = np.array(vec, dtype=float)
        else:
            self.outputs[0] = np.atleast_1d(inp)
        return self.outputs
