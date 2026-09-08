"""Digital Logic gates, flip-flops, and clock generator definitions."""

from __future__ import annotations
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union


class LogicValue(IntEnum):
    LOW = 0
    HIGH = 1
    UNKNOWN = -1
    HIGH_Z = -2

    def __str__(self) -> str:
        if self == LogicValue.LOW:
            return "0"
        elif self == LogicValue.HIGH:
            return "1"
        elif self == LogicValue.UNKNOWN:
            return "X"
        return "Z"


class LogicGate:
    """Base class for digital logic gates with propagation delay."""

    def __init__(self, name: str, input_wires: List[str], output_wire: str, delay_ns: float = 1.0):
        self.name = name.strip()
        self.input_wires = [w.strip() for w in input_wires]
        self.output_wire = output_wire.strip()
        self.delay_ns = float(delay_ns)

    @property
    def gate_type(self) -> str:
        return self.__class__.__name__.replace("Gate", "").upper()

    def evaluate(self, inputs: List[LogicValue]) -> LogicValue:
        return LogicValue.UNKNOWN


class AndGate(LogicGate):
    def evaluate(self, inputs: List[LogicValue]) -> LogicValue:
        if any(v == LogicValue.LOW for v in inputs):
            return LogicValue.LOW
        if any(v == LogicValue.UNKNOWN for v in inputs):
            return LogicValue.UNKNOWN
        return LogicValue.HIGH


class OrGate(LogicGate):
    def evaluate(self, inputs: List[LogicValue]) -> LogicValue:
        if any(v == LogicValue.HIGH for v in inputs):
            return LogicValue.HIGH
        if any(v == LogicValue.UNKNOWN for v in inputs):
            return LogicValue.UNKNOWN
        return LogicValue.LOW


class NotGate(LogicGate):
    def evaluate(self, inputs: List[LogicValue]) -> LogicValue:
        if not inputs:
            return LogicValue.UNKNOWN
        v = inputs[0]
        if v == LogicValue.HIGH:
            return LogicValue.LOW
        elif v == LogicValue.LOW:
            return LogicValue.HIGH
        return LogicValue.UNKNOWN


class NandGate(LogicGate):
    def evaluate(self, inputs: List[LogicValue]) -> LogicValue:
        if any(v == LogicValue.LOW for v in inputs):
            return LogicValue.HIGH
        if any(v == LogicValue.UNKNOWN for v in inputs):
            return LogicValue.UNKNOWN
        return LogicValue.LOW


class NorGate(LogicGate):
    def evaluate(self, inputs: List[LogicValue]) -> LogicValue:
        if any(v == LogicValue.HIGH for v in inputs):
            return LogicValue.LOW
        if any(v == LogicValue.UNKNOWN for v in inputs):
            return LogicValue.UNKNOWN
        return LogicValue.HIGH


class XorGate(LogicGate):
    def evaluate(self, inputs: List[LogicValue]) -> LogicValue:
        if any(v == LogicValue.UNKNOWN for v in inputs):
            return LogicValue.UNKNOWN
        high_count = sum(1 for v in inputs if v == LogicValue.HIGH)
        return LogicValue.HIGH if (high_count % 2 == 1) else LogicValue.LOW


class XnorGate(LogicGate):
    def evaluate(self, inputs: List[LogicValue]) -> LogicValue:
        if any(v == LogicValue.UNKNOWN for v in inputs):
            return LogicValue.UNKNOWN
        high_count = sum(1 for v in inputs if v == LogicValue.HIGH)
        return LogicValue.LOW if (high_count % 2 == 1) else LogicValue.HIGH


class BufferGate(LogicGate):
    def evaluate(self, inputs: List[LogicValue]) -> LogicValue:
        return inputs[0] if inputs else LogicValue.UNKNOWN


class Mux2to1(LogicGate):
    """2-to-1 Multiplexer: inputs = [in0, in1, sel]."""

    def __init__(self, name: str, in0: str, in1: str, sel: str, output_wire: str, delay_ns: float = 1.0):
        super().__init__(name, [in0, in1, sel], output_wire, delay_ns=delay_ns)

    def evaluate(self, inputs: List[LogicValue]) -> LogicValue:
        in0, in1, sel = inputs[0], inputs[1], inputs[2]
        if sel == LogicValue.LOW:
            return in0
        elif sel == LogicValue.HIGH:
            return in1
        return LogicValue.UNKNOWN


class DFlipFlop:
    """Edge-triggered D Flip-Flop with optional asynchronous active-high reset."""

    def __init__(
        self,
        name: str,
        clk_wire: str,
        d_wire: str,
        q_wire: str,
        q_bar_wire: Optional[str] = None,
        rst_wire: Optional[str] = None,
        delay_ns: float = 1.0
    ):
        self.name = name.strip()
        self.clk_wire = clk_wire.strip()
        self.d_wire = d_wire.strip()
        self.q_wire = q_wire.strip()
        self.q_bar_wire = q_bar_wire.strip() if q_bar_wire else None
        self.rst_wire = rst_wire.strip() if rst_wire else None
        self.delay_ns = float(delay_ns)

        # Internal state
        self.state = LogicValue.LOW
        self.last_clk = LogicValue.LOW

    def clock_transition(self, d_val: LogicValue, clk_val: LogicValue, rst_val: Optional[LogicValue]) -> Tuple[LogicValue, Optional[LogicValue]]:
        """Evaluates D-FF on clock or reset edge."""
        if rst_val == LogicValue.HIGH:
            self.state = LogicValue.LOW
        elif self.last_clk == LogicValue.LOW and clk_val == LogicValue.HIGH:  # Rising edge
            if d_val in (LogicValue.LOW, LogicValue.HIGH):
                self.state = d_val
            else:
                self.state = LogicValue.UNKNOWN
        self.last_clk = clk_val

        q_bar = LogicValue.LOW if self.state == LogicValue.HIGH else (LogicValue.HIGH if self.state == LogicValue.LOW else LogicValue.UNKNOWN)
        return self.state, q_bar


class JKFlipFlop:
    """Edge-triggered JK Flip-Flop."""

    def __init__(self, name: str, clk_wire: str, j_wire: str, k_wire: str, q_wire: str, delay_ns: float = 1.0):
        self.name = name.strip()
        self.clk_wire = clk_wire.strip()
        self.j_wire = j_wire.strip()
        self.k_wire = k_wire.strip()
        self.q_wire = q_wire.strip()
        self.delay_ns = float(delay_ns)
        self.state = LogicValue.LOW
        self.last_clk = LogicValue.LOW

    def clock_transition(self, j_val: LogicValue, k_val: LogicValue, clk_val: LogicValue) -> LogicValue:
        if self.last_clk == LogicValue.LOW and clk_val == LogicValue.HIGH:
            if j_val == LogicValue.LOW and k_val == LogicValue.LOW:
                pass  # Hold
            elif j_val == LogicValue.LOW and k_val == LogicValue.HIGH:
                self.state = LogicValue.LOW  # Reset
            elif j_val == LogicValue.HIGH and k_val == LogicValue.LOW:
                self.state = LogicValue.HIGH  # Set
            elif j_val == LogicValue.HIGH and k_val == LogicValue.HIGH:
                self.state = LogicValue.HIGH if self.state == LogicValue.LOW else LogicValue.LOW  # Toggle
        self.last_clk = clk_val
        return self.state


class ClockGenerator:
    """Produces periodic clock pulses."""

    def __init__(self, wire_name: str, period_ns: float = 10.0, duty_cycle: float = 0.5):
        self.wire_name = wire_name.strip()
        self.period_ns = float(period_ns)
        self.duty_cycle = float(duty_cycle)
        self.high_time = self.period_ns * self.duty_cycle
        self.low_time = self.period_ns - self.high_time
