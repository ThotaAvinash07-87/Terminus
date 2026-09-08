"""Classified Block Library Catalog and Parameter Registry for Dynamic Systems.

Provides systematic indexing, metadata, port definitions, parameter specifications,
and robust factory instantiation for all 17 block library categories in TerminusECE.
"""

from __future__ import annotations
import ast
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np

from CORE.common_math import parse_eng_unit
from .base import Block
from . import blocks as B


@dataclass
class PortDescriptor:
    """Specification for an input or output port."""
    name: str
    port_index: int
    data_type: str = "float"  # float, int, bool, matrix, string, vector, bus, any
    description: str = ""
    dimension: Optional[Tuple[int, ...]] = None  # None = scalar or dynamic


@dataclass
class ParameterDescriptor:
    """Specification for a tunable block parameter."""
    name: str
    param_type: type
    default_value: Any
    description: str = ""
    unit: str = ""
    choices: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class BlockDescriptor:
    """Complete metadata descriptor for a classified block library entry."""
    category: str
    type_name: str
    display_name: str
    block_class: Type[Block]
    description: str
    input_ports: List[PortDescriptor] = field(default_factory=list)
    output_ports: List[PortDescriptor] = field(default_factory=list)
    parameters: Dict[str, ParameterDescriptor] = field(default_factory=dict)
    direct_feedthrough: bool = True
    has_continuous_states: bool = False
    has_discrete_states: bool = False
    aliases: List[str] = field(default_factory=list)


def parse_parameter_value(val_str: Any, target_type: type) -> Any:
    """Parses a string or raw input into the target parameter type."""
    if isinstance(val_str, (int, float, list, np.ndarray, bool)) and not isinstance(val_str, str):
        if target_type == float:
            return float(val_str)
        if target_type == int:
            return int(val_str)
        if target_type == bool:
            return bool(val_str)
        if target_type in (list, np.ndarray, tuple):
            return val_str
        return val_str

    s = str(val_str).strip()
    if s.lower() == "none" or s == "":
        return None
    if s.lower() in ("true", "yes", "on"):
        return True
    if s.lower() in ("false", "no", "off"):
        return False

    # Try list / array literal: [1, 2, 3] or [1 2 3]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        # Handle space or comma separated numbers
        if not inner:
            return []
        tokens = [x for x in re.split(r'[\s,]+', inner) if x]
        try:
            parsed = [parse_eng_unit(t) for t in tokens]
            if target_type == np.ndarray:
                return np.array(parsed, dtype=float)
            return parsed
        except Exception:
            return tokens

    # Try engineering unit / float
    if target_type == float:
        try:
            return parse_eng_unit(s)
        except Exception:
            return float(s)
    if target_type == int:
        try:
            return int(parse_eng_unit(s))
        except Exception:
            return int(s)
    if target_type == str:
        return s.strip("\"'")

    # Default fallback: try float -> int -> str
    try:
        return parse_eng_unit(s)
    except Exception:
        return s


class DynamicBlockCatalog:
    """Systematic catalog and library browser for all Dynamic System blocks."""

    # 17 Systematic Categories
    CATEGORIES = [
        "Continuous",
        "Discrete",
        "Math Operations",
        "Matrix Operations",
        "Discontinuities",
        "Sources",
        "Sinks",
        "Routing",
        "Logic & Bit Operations",
        "Lookup Tables",
        "Signal Attributes",
        "Ports & Subsystems",
        "Model Verification",
        "Strings",
        "User Functions",
        "Dashboard",
        "Messages & Events",
    ]

    _catalog: Dict[str, BlockDescriptor] = {}
    _alias_map: Dict[str, str] = {}

    @classmethod
    def register(cls, descriptor: BlockDescriptor) -> None:
        key = descriptor.type_name.lower()
        cls._catalog[key] = descriptor
        cls._alias_map[key] = key
        for alias in descriptor.aliases:
            cls._alias_map[alias.lower()] = key

    @classmethod
    def get_descriptor(cls, name_or_alias: str) -> Optional[BlockDescriptor]:
        cleaned = name_or_alias.lower().strip()
        # Handle category/type format e.g. continuous/integrator
        if "/" in cleaned:
            cleaned = cleaned.split("/")[-1]
        resolved = cls._alias_map.get(cleaned)
        if resolved:
            return cls._catalog.get(resolved)
        return None

    @classmethod
    def list_categories(cls) -> List[str]:
        return cls.CATEGORIES

    @classmethod
    def get_blocks_by_category(cls, category: str) -> List[BlockDescriptor]:
        cat_lower = category.lower().replace("&", "and").strip()
        results = []
        for desc in cls._catalog.values():
            if desc.category.lower().replace("&", "and").strip() == cat_lower:
                results.append(desc)
        return sorted(results, key=lambda d: d.display_name)

    @classmethod
    def search_blocks(cls, query: str) -> List[BlockDescriptor]:
        q = query.lower().strip()
        matches = []
        for key, desc in cls._catalog.items():
            if (q in key or q in desc.display_name.lower() or 
                q in desc.category.lower() or q in desc.description.lower() or
                any(q in a.lower() for a in desc.aliases)):
                matches.append(desc)
        return matches

    @classmethod
    def instantiate(cls, block_type: str, name: str, **params) -> Block:
        """Instantiates and configures a block with parameter type validation."""
        desc = cls.get_descriptor(block_type)
        if desc is None:
            raise KeyError(
                f"Unknown block type '{block_type}'. Use 'library' or 'library search <query>' to browse available blocks."
            )

        kwargs = {}
        for p_name, p_val in params.items():
            p_desc = desc.parameters.get(p_name)
            if p_desc:
                target_type = p_desc.param_type
                try:
                    kwargs[p_name] = parse_parameter_value(p_val, target_type)
                except Exception as e:
                    raise ValueError(f"Parameter '{p_name}' could not be parsed as {target_type.__name__}: {e}")
            else:
                # Direct pass-through
                kwargs[p_name] = p_val

        # Fill in defaults for parameters not provided
        for p_name, p_desc in desc.parameters.items():
            if p_name not in kwargs and p_desc.default_value is not None:
                kwargs[p_name] = p_desc.default_value

        try:
            block = desc.block_class(name=name, **kwargs)
        except TypeError as err:
            # Fallback if block constructor signatures differ slightly
            try:
                block = desc.block_class(name)
                for k, v in kwargs.items():
                    if hasattr(block, k):
                        setattr(block, k, v)
            except Exception as e:
                raise ValueError(f"Failed to instantiate {desc.display_name} '{name}': {e} (Original: {err})")

        # Save parameter descriptors on the block for inspection & tuning
        block.parameters = kwargs
        return block


# ---------------------------------------------------------------------------
# Systematic Block Catalog Initialization
# ---------------------------------------------------------------------------
def _init_catalog() -> None:
    R = DynamicBlockCatalog.register

    # ================= 1. CONTINUOUS =================
    R(BlockDescriptor(
        category="Continuous",
        type_name="integrator",
        display_name="Integrator (1/s)",
        block_class=B.IntegratorBlock,
        description="Continuous-time integrator with optional state saturation and external reset.",
        input_ports=[PortDescriptor("in", 0, "float", "Signal derivative to integrate")],
        output_ports=[PortDescriptor("out", 0, "float", "Integrated continuous state")],
        parameters={
            "initial_condition": ParameterDescriptor("initial_condition", float, 0.0, "Initial state value at t=0"),
            "lower_limit": ParameterDescriptor("lower_limit", float, None, "Lower saturation limit"),
            "upper_limit": ParameterDescriptor("upper_limit", float, None, "Upper saturation limit"),
        },
        direct_feedthrough=False,
        has_continuous_states=True,
        aliases=["int", "1/s", "continuous_integrator"]
    ))

    R(BlockDescriptor(
        category="Continuous",
        type_name="derivative",
        display_name="Derivative (s)",
        block_class=B.DerivativeBlock,
        description="Continuous-time filtered derivative s / (tau*s + 1).",
        input_ports=[PortDescriptor("in", 0, "float", "Input signal")],
        output_ports=[PortDescriptor("out", 0, "float", "Time derivative of input signal")],
        parameters={
            "tau": ParameterDescriptor("tau", float, 0.01, "First-order lowpass filter time constant", unit="s"),
        },
        direct_feedthrough=True,
        has_continuous_states=True,
        aliases=["deriv", "s"]
    ))

    R(BlockDescriptor(
        category="Continuous",
        type_name="transfer_fcn",
        display_name="Transfer Fcn",
        block_class=B.TransferFunctionBlock,
        description="Linear continuous-time transfer function H(s) = Num(s) / Den(s).",
        input_ports=[PortDescriptor("in", 0, "float", "Input signal u(t)")],
        output_ports=[PortDescriptor("out", 0, "float", "Output signal y(t)")],
        parameters={
            "num": ParameterDescriptor("num", list, [1.0], "Numerator polynomial coefficients in descending powers of s"),
            "den": ParameterDescriptor("den", list, [1.0, 1.0], "Denominator polynomial coefficients in descending powers of s"),
        },
        direct_feedthrough=True,
        has_continuous_states=True,
        aliases=["tf", "transferfunction", "transfer_function"]
    ))

    R(BlockDescriptor(
        category="Continuous",
        type_name="state_space",
        display_name="State-Space",
        block_class=B.StateSpaceBlock,
        description="Continuous linear state-space system dx/dt = Ax + Bu, y = Cx + Du.",
        input_ports=[PortDescriptor("in", 0, "float", "Input vector u(t)")],
        output_ports=[PortDescriptor("out", 0, "float", "Output vector y(t)")],
        parameters={
            "A": ParameterDescriptor("A", list, [[-1.0]], "System state matrix A"),
            "B": ParameterDescriptor("B", list, [[1.0]], "Input matrix B"),
            "C": ParameterDescriptor("C", list, [[1.0]], "Output matrix C"),
            "D": ParameterDescriptor("D", list, [[0.0]], "Feedthrough matrix D"),
            "x0": ParameterDescriptor("x0", list, None, "Initial state vector"),
        },
        direct_feedthrough=True,
        has_continuous_states=True,
        aliases=["ss", "statespace"]
    ))

    R(BlockDescriptor(
        category="Continuous",
        type_name="pid",
        display_name="PID Controller",
        block_class=B.PIDBlock,
        description="Continuous-time PID controller with filtered derivative and anti-windup clamping.",
        input_ports=[PortDescriptor("in", 0, "float", "Error signal e(t)")],
        output_ports=[PortDescriptor("out", 0, "float", "Control signal u(t)")],
        parameters={
            "kp": ParameterDescriptor("kp", float, 1.0, "Proportional gain Kp"),
            "ki": ParameterDescriptor("ki", float, 0.0, "Integral gain Ki"),
            "kd": ParameterDescriptor("kd", float, 0.0, "Derivative gain Kd"),
            "n_filter": ParameterDescriptor("n_filter", float, 100.0, "Derivative filter coefficient N"),
            "lower_limit": ParameterDescriptor("lower_limit", float, None, "Lower anti-windup saturation limit"),
            "upper_limit": ParameterDescriptor("upper_limit", float, None, "Upper anti-windup saturation limit"),
        },
        direct_feedthrough=True,
        has_continuous_states=True,
        aliases=["pid_controller", "pidcontroller"]
    ))

    R(BlockDescriptor(
        category="Continuous",
        type_name="pid_2dof",
        display_name="PID Controller (2-DOF)",
        block_class=B.PIDController2DOFBlock,
        description="Two-Degree-of-Freedom PID controller with setpoint weighting (b, c).",
        input_ports=[
            PortDescriptor("r", 0, "float", "Reference setpoint r(t)"),
            PortDescriptor("y", 1, "float", "Measured process output y(t)")
        ],
        output_ports=[PortDescriptor("u", 0, "float", "Control effort u(t)")],
        parameters={
            "kp": ParameterDescriptor("kp", float, 1.0, "Proportional gain"),
            "ki": ParameterDescriptor("ki", float, 1.0, "Integral gain"),
            "kd": ParameterDescriptor("kd", float, 0.0, "Derivative gain"),
            "b": ParameterDescriptor("b", float, 1.0, "Setpoint weight on proportional action"),
            "c": ParameterDescriptor("c", float, 0.0, "Setpoint weight on derivative action"),
            "n_filter": ParameterDescriptor("n_filter", float, 100.0, "Filter divisor N"),
        },
        direct_feedthrough=True,
        has_continuous_states=True,
        aliases=["pid2dof", "2dof_pid"]
    ))

    R(BlockDescriptor(
        category="Continuous",
        type_name="transport_delay",
        display_name="Transport Delay",
        block_class=B.TransportDelayBlock,
        description="Delays the input signal by a specified time delay u(t - Td).",
        input_ports=[PortDescriptor("in", 0, "float", "Input signal")],
        output_ports=[PortDescriptor("out", 0, "float", "Delayed signal")],
        parameters={
            "delay_time": ParameterDescriptor("delay_time", float, 0.1, "Time delay duration Td", unit="s"),
            "initial_output": ParameterDescriptor("initial_output", float, 0.0, "Initial output during delay interval"),
        },
        direct_feedthrough=False,
        has_continuous_states=False,
        aliases=["delay", "timedelay", "time_delay"]
    ))

    R(BlockDescriptor(
        category="Continuous",
        type_name="zero_pole",
        display_name="Zero-Pole",
        block_class=B.ZeroPoleBlock,
        description="Continuous Transfer function defined by zeros, poles, and gain K.",
        input_ports=[PortDescriptor("in", 0, "float", "Input signal")],
        output_ports=[PortDescriptor("out", 0, "float", "Filtered signal")],
        parameters={
            "zeros": ParameterDescriptor("zeros", list, [], "List of zeros"),
            "poles": ParameterDescriptor("poles", list, [-1.0], "List of poles"),
            "gain": ParameterDescriptor("gain", float, 1.0, "System scalar gain K"),
        },
        direct_feedthrough=True,
        has_continuous_states=True,
        aliases=["zpk"]
    ))

    # ================= 2. DISCRETE =================
    R(BlockDescriptor(
        category="Discrete",
        type_name="unit_delay",
        display_name="Unit Delay (z^-1)",
        block_class=B.UnitDelayBlock,
        description="Delays signal by 1 discrete sample period Ts.",
        input_ports=[PortDescriptor("in", 0, "float", "Input signal")],
        output_ports=[PortDescriptor("out", 0, "float", "Delayed discrete signal")],
        parameters={
            "initial_condition": ParameterDescriptor("initial_condition", float, 0.0, "Initial discrete state"),
            "sample_time": ParameterDescriptor("sample_time", float, 0.01, "Discrete sample period Ts", unit="s"),
        },
        direct_feedthrough=False,
        has_discrete_states=True,
        aliases=["delay1", "z^-1", "unitdelay"]
    ))

    R(BlockDescriptor(
        category="Discrete",
        type_name="zero_order_hold",
        display_name="Zero-Order Hold (ZOH)",
        block_class=B.ZeroOrderHoldBlock,
        description="Samples and holds continuous input at regular sample intervals Ts.",
        input_ports=[PortDescriptor("in", 0, "float", "Continuous input signal")],
        output_ports=[PortDescriptor("out", 0, "float", "Discrete staircase sampled output")],
        parameters={
            "sample_time": ParameterDescriptor("sample_time", float, 0.01, "Sampling period Ts", unit="s"),
        },
        direct_feedthrough=True,
        has_discrete_states=True,
        aliases=["zoh", "sampleandhold"]
    ))

    R(BlockDescriptor(
        category="Discrete",
        type_name="discrete_integrator",
        display_name="Discrete-Time Integrator",
        block_class=B.DiscreteTimeIntegratorBlock,
        description="Discrete-time numerical integrator (Forward Euler, Backward Euler, or Trapezoidal).",
        input_ports=[PortDescriptor("in", 0, "float", "Signal to integrate")],
        output_ports=[PortDescriptor("out", 0, "float", "Accumulated discrete state")],
        parameters={
            "gainval": ParameterDescriptor("gainval", float, 1.0, "Integration gain coefficient"),
            "initial_condition": ParameterDescriptor("initial_condition", float, 0.0, "Initial accumulator state"),
            "sample_time": ParameterDescriptor("sample_time", float, 0.01, "Sample period Ts", unit="s"),
            "method": ParameterDescriptor("method", str, "ForwardEuler", "Integration method (ForwardEuler, BackwardEuler, Trapezoidal)"),
        },
        direct_feedthrough=False,
        has_discrete_states=True,
        aliases=["dint", "discrete_int"]
    ))

    R(BlockDescriptor(
        category="Discrete",
        type_name="discrete_filter",
        display_name="Discrete Filter / IIR / FIR",
        block_class=B.DiscreteFilterBlock,
        description="Discrete Infinite or Finite Impulse Response digital filter H(z) = Num(z)/Den(z).",
        input_ports=[PortDescriptor("in", 0, "float", "Digital input stream")],
        output_ports=[PortDescriptor("out", 0, "float", "Filtered digital output")],
        parameters={
            "num": ParameterDescriptor("num", list, [1.0], "Numerator coefficients in z^-1"),
            "den": ParameterDescriptor("den", list, [1.0, -0.9], "Denominator coefficients in z^-1"),
            "sample_time": ParameterDescriptor("sample_time", float, 0.01, "Sample time Ts", unit="s"),
        },
        direct_feedthrough=True,
        has_discrete_states=True,
        aliases=["df", "discrete_tf", "iir", "fir"]
    ))

    R(BlockDescriptor(
        category="Discrete",
        type_name="discrete_state_space",
        display_name="Discrete State-Space",
        block_class=B.DiscreteStateSpaceBlock,
        description="Discrete linear state-space system x[k+1] = Ax[k] + Bu[k], y[k] = Cx[k] + Du[k].",
        input_ports=[PortDescriptor("in", 0, "float", "Input vector u[k]")],
        output_ports=[PortDescriptor("out", 0, "float", "Output vector y[k]")],
        parameters={
            "A": ParameterDescriptor("A", list, [[0.9]], "Discrete transition matrix A"),
            "B": ParameterDescriptor("B", list, [[0.1]], "Input matrix B"),
            "C": ParameterDescriptor("C", list, [[1.0]], "Output matrix C"),
            "D": ParameterDescriptor("D", list, [[0.0]], "Direct feedthrough matrix D"),
            "sample_time": ParameterDescriptor("sample_time", float, 0.01, "Sample time Ts", unit="s"),
        },
        direct_feedthrough=True,
        has_discrete_states=True,
        aliases=["dss"]
    ))

    # ================= 3. MATH OPERATIONS =================
    R(BlockDescriptor(
        category="Math Operations",
        type_name="gain",
        display_name="Gain",
        block_class=B.GainBlock,
        description="Multiplies the input signal by a constant scalar or matrix gain K.",
        input_ports=[PortDescriptor("in", 0, "float", "Input signal")],
        output_ports=[PortDescriptor("out", 0, "float", "Scaled signal K*u")],
        parameters={
            "gain": ParameterDescriptor("gain", float, 1.0, "Multiplication factor K"),
        },
        direct_feedthrough=True,
        aliases=["k", "scale"]
    ))

    R(BlockDescriptor(
        category="Math Operations",
        type_name="sum",
        display_name="Sum / Add / Subtract",
        block_class=B.SumBlock,
        description="Performs addition and subtraction of input signals according to sign pattern.",
        input_ports=[
            PortDescriptor("in1", 0, "float", "Input 1"),
            PortDescriptor("in2", 1, "float", "Input 2"),
        ],
        output_ports=[PortDescriptor("out", 0, "float", "Combined sum/diff")],
        parameters={
            "signs": ParameterDescriptor("signs", str, "+-", "Sign sequence (e.g. '++', '+-', '+++', '+-+')"),
        },
        direct_feedthrough=True,
        aliases=["add", "subtract", "plus", "minus"]
    ))

    R(BlockDescriptor(
        category="Math Operations",
        type_name="product",
        display_name="Product / Divide",
        block_class=B.ProductBlock,
        description="Multiplies or divides inputs according to operation characters ('*', '/').",
        input_ports=[
            PortDescriptor("in1", 0, "float", "Input 1"),
            PortDescriptor("in2", 1, "float", "Input 2"),
        ],
        output_ports=[PortDescriptor("out", 0, "float", "Resulting product or quotient")],
        parameters={
            "operations": ParameterDescriptor("operations", str, "**", "Operations string (e.g. '**', '*/', '***')"),
        },
        direct_feedthrough=True,
        aliases=["mult", "multiply", "div", "divide"]
    ))

    R(BlockDescriptor(
        category="Math Operations",
        type_name="math_function",
        display_name="Math Function",
        block_class=B.MathFunctionBlock,
        description="Applies mathematical functions (exp, log, log10, sqrt, square, pow, sin, cos, tan, etc.).",
        input_ports=[PortDescriptor("in", 0, "float", "Input signal u")],
        output_ports=[PortDescriptor("out", 0, "float", "f(u)")],
        parameters={
            "function": ParameterDescriptor(
                "function", str, "sin", "Mathematical function name",
                choices=["exp", "log", "log10", "sqrt", "square", "pow", "reciprocal", "sin", "cos", "tan", "abs", "sign"]
            ),
        },
        direct_feedthrough=True,
        aliases=["mathfunc", "func", "math"]
    ))

    R(BlockDescriptor(
        category="Math Operations",
        type_name="abs",
        display_name="Absolute Value",
        block_class=B.AbsBlock,
        description="Computes the absolute value |u| of the input.",
        input_ports=[PortDescriptor("in", 0, "float", "Input signal")],
        output_ports=[PortDescriptor("out", 0, "float", "|u|")],
        parameters={},
        direct_feedthrough=True,
        aliases=["absolute"]
    ))

    R(BlockDescriptor(
        category="Math Operations",
        type_name="minmax",
        display_name="Min / Max",
        block_class=B.MinMaxBlock,
        description="Outputs the minimum or maximum among all input signals.",
        input_ports=[
            PortDescriptor("in1", 0, "float", "Input 1"),
            PortDescriptor("in2", 1, "float", "Input 2"),
        ],
        output_ports=[PortDescriptor("out", 0, "float", "min(u) or max(u)")],
        parameters={
            "function": ParameterDescriptor("function", str, "min", "Choice of 'min' or 'max'", choices=["min", "max"]),
        },
        direct_feedthrough=True,
        aliases=["min", "max"]
    ))

    R(BlockDescriptor(
        category="Math Operations",
        type_name="polynomial",
        display_name="Polynomial Evaluation",
        block_class=B.PolynomialBlock,
        description="Evaluates polynomial P(u) = c0*u^N + c1*u^(N-1) + ... + cN.",
        input_ports=[PortDescriptor("in", 0, "float", "Input variable u")],
        output_ports=[PortDescriptor("out", 0, "float", "Evaluated polynomial P(u)")],
        parameters={
            "coefficients": ParameterDescriptor("coefficients", list, [1.0, 0.0], "Polynomial coefficients in descending order"),
        },
        direct_feedthrough=True,
        aliases=["poly"]
    ))

    # ================= 4. DISCONTINUITIES =================
    R(BlockDescriptor(
        category="Discontinuities",
        type_name="saturation",
        display_name="Saturation",
        block_class=B.SaturationBlock,
        description="Clamps the input signal within upper and lower bounds.",
        input_ports=[PortDescriptor("in", 0, "float", "Input signal")],
        output_ports=[PortDescriptor("out", 0, "float", "Clamped signal")],
        parameters={
            "lower_limit": ParameterDescriptor("lower_limit", float, -1.0, "Lower clamp limit"),
            "upper_limit": ParameterDescriptor("upper_limit", float, 1.0, "Upper clamp limit"),
        },
        direct_feedthrough=True,
        aliases=["sat", "clamp"]
    ))

    R(BlockDescriptor(
        category="Discontinuities",
        type_name="rate_limiter",
        display_name="Rate Limiter",
        block_class=B.RateLimiterBlock,
        description="Limits the first derivative du/dt of the signal to specified slew rates.",
        input_ports=[PortDescriptor("in", 0, "float", "Input signal")],
        output_ports=[PortDescriptor("out", 0, "float", "Rate-limited signal")],
        parameters={
            "rising_slew_rate": ParameterDescriptor("rising_slew_rate", float, 100.0, "Maximum positive rate of change", unit="units/s"),
            "falling_slew_rate": ParameterDescriptor("falling_slew_rate", float, -100.0, "Maximum negative rate of change", unit="units/s"),
        },
        direct_feedthrough=True,
        has_continuous_states=False,
        aliases=["ratelimit", "slew", "ratelimiter"]
    ))

    R(BlockDescriptor(
        category="Discontinuities",
        type_name="dead_zone",
        display_name="Dead Zone",
        block_class=B.DeadZoneBlock,
        description="Creates a zero-output band between start_zone and end_zone.",
        input_ports=[PortDescriptor("in", 0, "float", "Input signal")],
        output_ports=[PortDescriptor("out", 0, "float", "Dead-zone signal")],
        parameters={
            "start_zone": ParameterDescriptor("start_zone", float, -0.5, "Lower threshold of dead zone"),
            "end_zone": ParameterDescriptor("end_zone", float, 0.5, "Upper threshold of dead zone"),
        },
        direct_feedthrough=True,
        aliases=["deadzone", "deadband"]
    ))

    R(BlockDescriptor(
        category="Discontinuities",
        type_name="backlash",
        display_name="Backlash / Hysteresis",
        block_class=B.BacklashBlock,
        description="Models mechanical backlash and deadband hysteresis.",
        input_ports=[PortDescriptor("in", 0, "float", "Input displacement")],
        output_ports=[PortDescriptor("out", 0, "float", "Engaged output displacement")],
        parameters={
            "deadband_width": ParameterDescriptor("deadband_width", float, 1.0, "Total deadband gap width"),
            "initial_output": ParameterDescriptor("initial_output", float, 0.0, "Initial engaged position"),
        },
        direct_feedthrough=True,
        aliases=["hysteresis"]
    ))

    R(BlockDescriptor(
        category="Discontinuities",
        type_name="relay",
        display_name="Relay / Schmitt Trigger",
        block_class=B.RelayBlock,
        description="Hysteretic switch (bang-bang relay / Schmitt trigger).",
        input_ports=[PortDescriptor("in", 0, "float", "Input signal")],
        output_ports=[PortDescriptor("out", 0, "float", "Bistable state output")],
        parameters={
            "switch_on_point": ParameterDescriptor("switch_on_point", float, 0.5, "Turn-on threshold"),
            "switch_off_point": ParameterDescriptor("switch_off_point", float, -0.5, "Turn-off threshold"),
            "output_on": ParameterDescriptor("output_on", float, 1.0, "High output level"),
            "output_off": ParameterDescriptor("output_off", float, 0.0, "Low output level"),
        },
        direct_feedthrough=True,
        aliases=["schmitt", "bangbang"]
    ))

    R(BlockDescriptor(
        category="Discontinuities",
        type_name="coulomb_friction",
        display_name="Coulomb & Viscous Friction",
        block_class=B.CoulombViscousFrictionBlock,
        description="Simulates mechanical friction: Coulomb stiction + linear viscous damping.",
        input_ports=[PortDescriptor("v", 0, "float", "Relative velocity")],
        output_ports=[PortDescriptor("f", 0, "float", "Friction force/torque")],
        parameters={
            "f_coulomb": ParameterDescriptor("f_coulomb", float, 1.0, "Coulomb friction magnitude", unit="N"),
            "b_viscous": ParameterDescriptor("b_viscous", float, 0.1, "Viscous friction damping coefficient", unit="N*s/m"),
            "f_static": ParameterDescriptor("f_static", float, 1.5, "Static breakaway stiction force", unit="N"),
        },
        direct_feedthrough=True,
        aliases=["friction", "fric"]
    ))

    R(BlockDescriptor(
        category="Discontinuities",
        type_name="quantizer",
        display_name="Quantizer",
        block_class=B.QuantizerBlock,
        description="Discretizes continuous signal into integer steps of quantization interval Q.",
        input_ports=[PortDescriptor("in", 0, "float", "Input signal")],
        output_ports=[PortDescriptor("out", 0, "float", "Quantized staircase signal")],
        parameters={
            "quantization_interval": ParameterDescriptor("quantization_interval", float, 0.1, "Quantization step size Q"),
        },
        direct_feedthrough=True,
        aliases=["quant", "adc_dac"]
    ))

    # ================= 5. SOURCES =================
    R(BlockDescriptor(
        category="Sources",
        type_name="constant",
        display_name="Constant",
        block_class=B.ConstantBlock,
        description="Outputs a constant numeric value or array.",
        input_ports=[],
        output_ports=[PortDescriptor("out", 0, "float", "Constant value")],
        parameters={
            "value": ParameterDescriptor("value", float, 1.0, "Constant output value"),
        },
        direct_feedthrough=False,
        aliases=["const", "dc"]
    ))

    R(BlockDescriptor(
        category="Sources",
        type_name="step",
        display_name="Step",
        block_class=B.StepSourceBlock,
        description="Generates a step transition from initial_value to amplitude at step_time.",
        input_ports=[],
        output_ports=[PortDescriptor("out", 0, "float", "Step signal")],
        parameters={
            "step_time": ParameterDescriptor("step_time", float, 1.0, "Time at which step transition occurs", unit="s"),
            "initial_value": ParameterDescriptor("initial_value", float, 0.0, "Output value before step time"),
            "amplitude": ParameterDescriptor("amplitude", float, 1.0, "Final output value after step time"),
        },
        direct_feedthrough=False,
        aliases=["step_source", "heaviside"]
    ))

    R(BlockDescriptor(
        category="Sources",
        type_name="ramp",
        display_name="Ramp",
        block_class=B.RampSourceBlock,
        description="Generates a linearly increasing or decreasing ramp signal with given slope.",
        input_ports=[],
        output_ports=[PortDescriptor("out", 0, "float", "Ramp signal")],
        parameters={
            "slope": ParameterDescriptor("slope", float, 1.0, "Rate of change of ramp", unit="units/s"),
            "start_time": ParameterDescriptor("start_time", float, 0.0, "Time at which ramp starts", unit="s"),
            "initial_output": ParameterDescriptor("initial_output", float, 0.0, "Initial output value before start time"),
        },
        direct_feedthrough=False,
        aliases=["ramp_source", "slope"]
    ))

    R(BlockDescriptor(
        category="Sources",
        type_name="sine_wave",
        display_name="Sine Wave",
        block_class=B.SineSourceBlock,
        description="Generates a sinusoidal wave: y(t) = Amp * sin(2*pi*f*t + Phase) + Bias.",
        input_ports=[],
        output_ports=[PortDescriptor("out", 0, "float", "Sinusoidal signal")],
        parameters={
            "freq": ParameterDescriptor("freq", float, 1.0, "Frequency in Hertz", unit="Hz"),
            "amplitude": ParameterDescriptor("amplitude", float, 1.0, "Peak amplitude"),
            "phase": ParameterDescriptor("phase", float, 0.0, "Phase offset in radians", unit="rad"),
            "bias": ParameterDescriptor("bias", float, 0.0, "DC bias offset"),
        },
        direct_feedthrough=False,
        aliases=["sine", "sin", "sinewave"]
    ))

    R(BlockDescriptor(
        category="Sources",
        type_name="pulse_generator",
        display_name="Pulse Generator",
        block_class=B.PulseGeneratorBlock,
        description="Generates periodic pulse or square wave train with configurable duty cycle.",
        input_ports=[],
        output_ports=[PortDescriptor("out", 0, "float", "Pulse train")],
        parameters={
            "period": ParameterDescriptor("period", float, 1.0, "Pulse period", unit="s"),
            "duty_cycle": ParameterDescriptor("duty_cycle", float, 0.5, "Duty cycle (fraction 0.0 to 1.0)"),
            "amplitude": ParameterDescriptor("amplitude", float, 1.0, "Pulse high amplitude"),
            "phase_delay": ParameterDescriptor("phase_delay", float, 0.0, "Phase delay", unit="s"),
        },
        direct_feedthrough=False,
        aliases=["pulse", "square", "clock_gen"]
    ))

    R(BlockDescriptor(
        category="Sources",
        type_name="white_noise",
        display_name="Band-Limited White Noise",
        block_class=B.BandLimitedWhiteNoiseBlock,
        description="Generates band-limited Gaussian white noise with specified noise power.",
        input_ports=[],
        output_ports=[PortDescriptor("out", 0, "float", "Random noise signal")],
        parameters={
            "noise_power": ParameterDescriptor("noise_power", float, 0.1, "Noise power spectral density"),
            "sample_time": ParameterDescriptor("sample_time", float, 0.01, "Correlation sample time", unit="s"),
            "seed": ParameterDescriptor("seed", int, 42, "Pseudo-random generator seed"),
        },
        direct_feedthrough=False,
        aliases=["noise", "whitenoise", "gaussian_noise"]
    ))

    R(BlockDescriptor(
        category="Sources",
        type_name="clock",
        display_name="Clock",
        block_class=B.ClockBlock,
        description="Outputs the current continuous simulation time t.",
        input_ports=[],
        output_ports=[PortDescriptor("time", 0, "float", "Simulation time t (seconds)")],
        parameters={},
        direct_feedthrough=False,
        aliases=["time", "sim_time"]
    ))

    # ================= 6. SINKS =================
    R(BlockDescriptor(
        category="Sinks",
        type_name="scope",
        display_name="Scope",
        block_class=B.ScopeSinkBlock,
        description="Records and plots input signal history across simulation time.",
        input_ports=[PortDescriptor("in", 0, "float", "Signal to record and display")],
        output_ports=[],
        parameters={
            "buffer_size": ParameterDescriptor("buffer_size", int, 100000, "Maximum recorded sample count"),
        },
        direct_feedthrough=False,
        aliases=["scope_sink", "plotter"]
    ))

    R(BlockDescriptor(
        category="Sinks",
        type_name="display",
        display_name="Display",
        block_class=B.DisplaySinkBlock,
        description="Displays the current numerical value of the input signal.",
        input_ports=[PortDescriptor("in", 0, "float", "Signal to display")],
        output_ports=[],
        parameters={},
        direct_feedthrough=False,
        aliases=["disp", "gauge_sink"]
    ))

    R(BlockDescriptor(
        category="Sinks",
        type_name="terminator",
        display_name="Terminator",
        block_class=B.TerminatorBlock,
        description="Terminates an unused signal wire to suppress dangling port warnings.",
        input_ports=[PortDescriptor("in", 0, "any", "Unused signal")],
        output_ports=[],
        parameters={},
        direct_feedthrough=False,
        aliases=["term", "ground_sink"]
    ))

    R(BlockDescriptor(
        category="Sinks",
        type_name="to_workspace",
        display_name="To Workspace",
        block_class=B.ToWorkspaceBlock,
        description="Saves signal history to MATLAB/Numerical workspace variable array.",
        input_ports=[PortDescriptor("in", 0, "any", "Signal to export")],
        output_ports=[],
        parameters={
            "var_name": ParameterDescriptor("var_name", str, "simout", "Workspace variable destination name"),
        },
        direct_feedthrough=False,
        aliases=["toworkspace", "simout"]
    ))

    # ================= 7. ROUTING =================
    R(BlockDescriptor(
        category="Routing",
        type_name="mux",
        display_name="Mux",
        block_class=B.MuxBlock,
        description="Combines multiple scalar or vector inputs into a single composite vector signal.",
        input_ports=[
            PortDescriptor("in1", 0, "float", "Input 1"),
            PortDescriptor("in2", 1, "float", "Input 2"),
        ],
        output_ports=[PortDescriptor("out", 0, "vector", "Multiplexed vector signal")],
        parameters={
            "num_inputs": ParameterDescriptor("num_inputs", int, 2, "Number of input channels to multiplex"),
        },
        direct_feedthrough=True,
        aliases=["multiplexer"]
    ))

    R(BlockDescriptor(
        category="Routing",
        type_name="demux",
        display_name="Demux",
        block_class=B.DemuxBlock,
        description="Splits a vector signal into individual scalar or sub-vector components.",
        input_ports=[PortDescriptor("in", 0, "vector", "Vector input")],
        output_ports=[
            PortDescriptor("out1", 0, "float", "Output component 1"),
            PortDescriptor("out2", 1, "float", "Output component 2"),
        ],
        parameters={
            "num_outputs": ParameterDescriptor("num_outputs", int, 2, "Number of split output channels"),
        },
        direct_feedthrough=True,
        aliases=["demultiplexer"]
    ))

    R(BlockDescriptor(
        category="Routing",
        type_name="switch",
        display_name="Switch",
        block_class=B.SwitchBlock,
        description="Passes input 1 if control input >= threshold, otherwise passes input 3.",
        input_ports=[
            PortDescriptor("in1", 0, "float", "Signal input 1 (passed if true)"),
            PortDescriptor("ctrl", 1, "float", "Control signal (checked against threshold)"),
            PortDescriptor("in2", 2, "float", "Signal input 2 (passed if false)"),
        ],
        output_ports=[PortDescriptor("out", 0, "float", "Switched output")],
        parameters={
            "threshold": ParameterDescriptor("threshold", float, 0.0, "Switching threshold condition"),
        },
        direct_feedthrough=True,
        aliases=["mux2to1", "if_switch"]
    ))

    R(BlockDescriptor(
        category="Routing",
        type_name="goto",
        display_name="Goto",
        block_class=B.GotoBlock,
        description="Sends a signal to corresponding From blocks using a scoped or global tag.",
        input_ports=[PortDescriptor("in", 0, "any", "Signal to broadcast")],
        output_ports=[],
        parameters={
            "tag": ParameterDescriptor("tag", str, "A", "Signal identifier tag name"),
        },
        direct_feedthrough=True,
        aliases=["goto_tag"]
    ))

    R(BlockDescriptor(
        category="Routing",
        type_name="from",
        display_name="From",
        block_class=B.FromBlock,
        description="Receives signal from a matching Goto block tag without visual wire clutter.",
        input_ports=[],
        output_ports=[PortDescriptor("out", 0, "any", "Received signal")],
        parameters={
            "tag": ParameterDescriptor("tag", str, "A", "Signal identifier tag to receive"),
        },
        direct_feedthrough=True,
        aliases=["from_tag"]
    ))

    # ================= 8. LOGIC & BIT OPERATIONS =================
    R(BlockDescriptor(
        category="Logic & Bit Operations",
        type_name="logical_operator",
        display_name="Logical Operator",
        block_class=B.LogicalOperatorBlock,
        description="Performs Boolean logic (AND, OR, NAND, NOR, XOR, NOT) on inputs.",
        input_ports=[
            PortDescriptor("in1", 0, "bool", "Logic input 1"),
            PortDescriptor("in2", 1, "bool", "Logic input 2"),
        ],
        output_ports=[PortDescriptor("out", 0, "bool", "Boolean logic result")],
        parameters={
            "operator": ParameterDescriptor("operator", str, "AND", "Logic operation", choices=["AND", "OR", "NAND", "NOR", "XOR", "NOT"]),
        },
        direct_feedthrough=True,
        aliases=["logic", "and", "or", "xor", "not"]
    ))

    R(BlockDescriptor(
        category="Logic & Bit Operations",
        type_name="relational_operator",
        display_name="Relational Operator",
        block_class=B.RelationalOperatorBlock,
        description="Compares two inputs (==, ~=, <, <=, >, >=).",
        input_ports=[
            PortDescriptor("in1", 0, "float", "Left operand"),
            PortDescriptor("in2", 1, "float", "Right operand"),
        ],
        output_ports=[PortDescriptor("out", 0, "bool", "1.0 if condition holds, else 0.0")],
        parameters={
            "operator": ParameterDescriptor("operator", str, "<=", "Comparison operator", choices=["==", "~=", "<", "<=", ">", ">="]),
        },
        direct_feedthrough=True,
        aliases=["compare", "relational", "cmp"]
    ))

    # ================= 9. LOOKUP TABLES =================
    R(BlockDescriptor(
        category="Lookup Tables",
        type_name="lookup_table_1d",
        display_name="1-D Lookup Table",
        block_class=B.LookupTable1DBlock,
        description="Computes piecewise linear interpolation/extrapolation y = f(x) from data vectors.",
        input_ports=[PortDescriptor("x", 0, "float", "Table input coordinate x")],
        output_ports=[PortDescriptor("y", 0, "float", "Interpolated output y = f(x)")],
        parameters={
            "break_points": ParameterDescriptor("break_points", list, [0.0, 1.0, 2.0], "Monotonically increasing grid coordinates"),
            "table_data": ParameterDescriptor("table_data", list, [0.0, 1.0, 4.0], "Target table values at grid points"),
        },
        direct_feedthrough=True,
        aliases=["lut1d", "lookup1d", "table1d"]
    ))

    R(BlockDescriptor(
        category="Lookup Tables",
        type_name="lookup_table_2d",
        display_name="2-D Lookup Table",
        block_class=B.LookupTable2DBlock,
        description="Computes 2D bilinear interpolation z = f(x, y) over row and column breakpoints.",
        input_ports=[
            PortDescriptor("row_in", 0, "float", "Row coordinate x"),
            PortDescriptor("col_in", 1, "float", "Column coordinate y"),
        ],
        output_ports=[PortDescriptor("z", 0, "float", "Interpolated output z = f(x, y)")],
        parameters={
            "row_points": ParameterDescriptor("row_points", list, [0.0, 1.0], "Row break points"),
            "col_points": ParameterDescriptor("col_points", list, [0.0, 1.0], "Column break points"),
            "table_data": ParameterDescriptor("table_data", list, [[0.0, 1.0], [1.0, 2.0]], "2D matrix data"),
        },
        direct_feedthrough=True,
        aliases=["lut2d", "lookup2d", "table2d"]
    ))

    # ================= 10. MATRIX OPERATIONS =================
    R(BlockDescriptor(
        category="Matrix Operations",
        type_name="matrix_product",
        display_name="Matrix Multiply",
        block_class=B.ProductMatrixMultiplyBlock,
        description="Performs matrix multiplication A * B or element-wise multi-dimensional product.",
        input_ports=[
            PortDescriptor("A", 0, "matrix", "Matrix input A"),
            PortDescriptor("B", 1, "matrix", "Matrix input B"),
        ],
        output_ports=[PortDescriptor("C", 0, "matrix", "Product C = A @ B")],
        parameters={},
        direct_feedthrough=True,
        aliases=["matmul", "matrix_mult"]
    ))

    R(BlockDescriptor(
        category="Matrix Operations",
        type_name="transpose",
        display_name="Matrix Transpose",
        block_class=B.TransposeBlock,
        description="Transposes vector or 2D matrix input A^T.",
        input_ports=[PortDescriptor("in", 0, "matrix", "Input matrix")],
        output_ports=[PortDescriptor("out", 0, "matrix", "Transposed matrix A^T")],
        parameters={},
        direct_feedthrough=True,
        aliases=["matrix_transpose"]
    ))

    # ================= 11. SIGNAL ATTRIBUTES =================
    R(BlockDescriptor(
        category="Signal Attributes",
        type_name="rate_transition",
        display_name="Rate Transition",
        block_class=B.RateTransitionBlock,
        description="Safely transfers signals between different sample rates or continuous/discrete domains.",
        input_ports=[PortDescriptor("in", 0, "any", "Input signal at Source Rate")],
        output_ports=[PortDescriptor("out", 0, "any", "Resampled signal at Target Rate")],
        parameters={
            "sample_time": ParameterDescriptor("sample_time", float, 0.01, "Target sample period Ts", unit="s"),
        },
        direct_feedthrough=True,
        has_discrete_states=True,
        aliases=["ratetransition", "rate_conv"]
    ))

    R(BlockDescriptor(
        category="Signal Attributes",
        type_name="initial_condition",
        display_name="Initial Condition (IC)",
        block_class=B.ICBlock,
        description="Outputs initial value at t=0, and passes through input signal for t > 0.",
        input_ports=[PortDescriptor("in", 0, "any", "Normal input signal")],
        output_ports=[PortDescriptor("out", 0, "any", "IC output")],
        parameters={
            "value": ParameterDescriptor("value", float, 0.0, "Initial value at t=0"),
        },
        direct_feedthrough=True,
        aliases=["ic"]
    ))

    # ================= 12. MODEL VERIFICATION =================
    R(BlockDescriptor(
        category="Model Verification",
        type_name="assertion",
        display_name="Assertion",
        block_class=B.AssertionBlock,
        description="Verifies that input condition signal remains non-zero (True). Raises alert if False.",
        input_ports=[PortDescriptor("in", 0, "bool", "Boolean assertion condition")],
        output_ports=[],
        parameters={
            "stop_when_false": ParameterDescriptor("stop_when_false", bool, False, "Stop simulation if assertion fails"),
        },
        direct_feedthrough=True,
        aliases=["assert"]
    ))

    R(BlockDescriptor(
        category="Model Verification",
        type_name="check_static_range",
        display_name="Check Static Range",
        block_class=B.CheckStaticRangeBlock,
        description="Verifies that signal remains within static bounds [min, max].",
        input_ports=[PortDescriptor("in", 0, "float", "Signal to monitor")],
        output_ports=[],
        parameters={
            "min_val": ParameterDescriptor("min_val", float, -100.0, "Lower bound limit"),
            "max_val": ParameterDescriptor("max_val", float, 100.0, "Upper bound limit"),
        },
        direct_feedthrough=True,
        aliases=["check_range", "range_check"]
    ))

    # ================= 13. USER FUNCTIONS =================
    R(BlockDescriptor(
        category="User Functions",
        type_name="fcn",
        display_name="Math Expression (Fcn)",
        block_class=B.FcnBlock,
        description="Evaluates custom mathematical expression string f(u) (e.g. 'sin(u[0]) + 2*u[1]').",
        input_ports=[PortDescriptor("u", 0, "any", "Input argument vector u")],
        output_ports=[PortDescriptor("y", 0, "float", "Expression result")],
        parameters={
            "expression": ParameterDescriptor("expression", str, "u", "Math formula expression"),
        },
        direct_feedthrough=True,
        aliases=["mathexpr", "custom_func"]
    ))


_init_catalog()
