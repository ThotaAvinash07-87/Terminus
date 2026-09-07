"""Logic and Bit Operation simulation blocks for Dynamic Systems."""

from __future__ import annotations
import math
import struct
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from .base import Block


class LogicalOperatorBlock(Block):
    """Perform specified logical operation on inputs: AND, OR, NAND, NOR, XOR, NOT, NXOR."""

    def __init__(self, name: str, operator: str = "AND", num_inputs: int = 2):
        self.operator = operator.upper()
        inputs_count = 1 if self.operator == "NOT" else max(2, int(num_inputs))
        super().__init__(name, num_inputs=inputs_count, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        bools = [bool(abs(float(x)) > 1e-6) for x in self.inputs]
        if self.operator == "AND":
            res = all(bools)
        elif self.operator == "OR":
            res = any(bools)
        elif self.operator == "NAND":
            res = not all(bools)
        elif self.operator == "NOR":
            res = not any(bools)
        elif self.operator == "XOR":
            # Parity XOR
            res = sum(bools) % 2 == 1
        elif self.operator == "NOT":
            res = not bools[0]
        elif self.operator == "NXOR":
            res = sum(bools) % 2 == 0
        else:
            res = all(bools)
        self.outputs[0] = 1.0 if res else 0.0
        return self.outputs


class RelationalOperatorBlock(Block):
    """Perform relational comparison: ==, ~=, !=, <, <=, >, >=."""

    def __init__(self, name: str, operator: str = "<="):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.op = operator.strip()
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u1 = float(self.inputs[0])
        u2 = float(self.inputs[1]) if len(self.inputs) > 1 else 0.0

        if self.op in ("==", "="):
            res = abs(u1 - u2) < 1e-9
        elif self.op in ("~=", "!="):
            res = abs(u1 - u2) >= 1e-9
        elif self.op == "<":
            res = u1 < u2
        elif self.op == "<=":
            res = u1 <= u2
        elif self.op == ">":
            res = u1 > u2
        elif self.op == ">=":
            res = u1 >= u2
        else:
            res = u1 <= u2

        self.outputs[0] = 1.0 if res else 0.0
        return self.outputs


class CompareToConstantBlock(Block):
    """Determine how signal compares to specified constant."""

    def __init__(self, name: str, operator: str = "==", constant: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.op = operator.strip()
        self.c = float(constant)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        if self.op in ("==", "="):
            res = abs(u - self.c) < 1e-9
        elif self.op in ("~=", "!="):
            res = abs(u - self.c) >= 1e-9
        elif self.op == "<":
            res = u < self.c
        elif self.op == "<=":
            res = u <= self.c
        elif self.op == ">":
            res = u > self.c
        elif self.op == ">=":
            res = u >= self.c
        else:
            res = u == self.c
        self.outputs[0] = 1.0 if res else 0.0
        return self.outputs


class CompareToZeroBlock(CompareToConstantBlock):
    """Determine how signal compares to zero."""

    def __init__(self, name: str, operator: str = ">="):
        super().__init__(name, operator=operator, constant=0.0)


class IntervalTestBlock(Block):
    """Determine if signal is in specified interval [lower, upper].
    Interval Closed: [a, b], Half-open [a, b) / (a, b], Open (a, b).
    """

    def __init__(
        self,
        name: str,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        interval_type: str = "Closed"  # 'Closed', 'Open', 'Half-open left', 'Half-open right'
    ):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.low = float(lower_bound)
        self.high = float(upper_bound)
        self.interval_type = interval_type.lower()
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        if "open" in self.interval_type and "half" not in self.interval_type:
            inside = (self.low < u < self.high)
        elif "left" in self.interval_type:  # (low, high]
            inside = (self.low < u <= self.high)
        elif "right" in self.interval_type:  # [low, high)
            inside = (self.low <= u < self.high)
        else:  # Closed [low, high]
            inside = (self.low <= u <= self.high)
        self.outputs[0] = 1.0 if inside else 0.0
        return self.outputs


class IntervalTestDynamicBlock(Block):
    """Determine if signal u is in dynamically specified interval [lower, upper]:
    Input 0: u (signal)
    Input 1: upper bound
    Input 2: lower bound
    """

    def __init__(self, name: str):
        super().__init__(name, num_inputs=3, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        high = float(self.inputs[1]) if len(self.inputs) > 1 else 1.0
        low = float(self.inputs[2]) if len(self.inputs) > 2 else 0.0
        if low > high:
            low, high = high, low
        inside = (low <= u <= high)
        self.outputs[0] = 1.0 if inside else 0.0
        return self.outputs


class BitClearBlock(Block):
    """Set specified bit (0-indexed) of stored integer to zero."""

    def __init__(self, name: str, bit_position: int = 0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.bit = int(bit_position)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        val = int(round(float(self.inputs[0])))
        mask = ~(1 << self.bit)
        self.outputs[0] = float(val & mask)
        return self.outputs


class BitSetBlock(Block):
    """Set specified bit (0-indexed) of stored integer to one."""

    def __init__(self, name: str, bit_position: int = 0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.bit = int(bit_position)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        val = int(round(float(self.inputs[0])))
        mask = 1 << self.bit
        self.outputs[0] = float(val | mask)
        return self.outputs


class BitwiseOperatorBlock(Block):
    """Specified bitwise operation on inputs: AND, OR, XOR, NOT, NAND, NOR, SHIFT."""

    def __init__(self, name: str, operator: str = "AND", num_inputs: int = 2):
        self.operator = operator.upper()
        inputs_count = 1 if self.operator == "NOT" else max(2, int(num_inputs))
        super().__init__(name, num_inputs=inputs_count, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        ints = [int(round(float(x))) for x in self.inputs]
        if self.operator == "AND":
            res = ints[0]
            for val in ints[1:]:
                res &= val
        elif self.operator == "OR":
            res = ints[0]
            for val in ints[1:]:
                res |= val
        elif self.operator == "XOR":
            res = ints[0]
            for val in ints[1:]:
                res ^= val
        elif self.operator == "NOT":
            res = ~ints[0]
        elif self.operator == "NAND":
            res = ints[0]
            for val in ints[1:]:
                res &= val
            res = ~res
        elif self.operator == "NOR":
            res = ints[0]
            for val in ints[1:]:
                res |= val
            res = ~res
        else:
            res = ints[0] & ints[1]
        self.outputs[0] = float(res)
        return self.outputs


class BitToIntegerConverterBlock(Block):
    """Map vector of bits (MSB first or LSB first) to corresponding integer."""

    def __init__(self, name: str, num_bits: int = 8, lsb_first: bool = False):
        super().__init__(name, num_inputs=num_bits, num_outputs=1)
        self.lsb_first = lsb_first
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        bits = [1 if abs(float(x)) > 0.5 else 0 for x in self.inputs]
        if not self.lsb_first:
            # MSB first
            val = 0
            for b in bits:
                val = (val << 1) | b
        else:
            val = 0
            for i, b in enumerate(bits):
                val |= (b << i)
        self.outputs[0] = float(val)
        return self.outputs


class IntegerToBitConverterBlock(Block):
    """Map integer to corresponding vector of bits."""

    def __init__(self, name: str, num_bits: int = 8, lsb_first: bool = False):
        self.num_bits = int(num_bits)
        super().__init__(name, num_inputs=1, num_outputs=self.num_bits)
        self.lsb_first = lsb_first
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        val = int(round(float(self.inputs[0])))
        for i in range(self.num_bits):
            if self.lsb_first:
                self.outputs[i] = float((val >> i) & 1)
            else:
                self.outputs[i] = float((val >> (self.num_bits - 1 - i)) & 1)
        return self.outputs


class ExtractBitsBlock(Block):
    """Output selection of contiguous bits [start_bit, end_bit] from integer signal."""

    def __init__(self, name: str, start_bit: int = 0, end_bit: int = 7):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.start_bit = min(start_bit, end_bit)
        self.end_bit = max(start_bit, end_bit)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        val = int(round(float(self.inputs[0])))
        num_bits = self.end_bit - self.start_bit + 1
        mask = (1 << num_bits) - 1
        extracted = (val >> self.start_bit) & mask
        self.outputs[0] = float(extracted)
        return self.outputs


class FloatExtractBitsBlock(Block):
    """Extract IEEE-754 binary representation of float32/64 to integer bits."""

    def __init__(self, name: str, is_double: bool = True):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.is_double = is_double
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        val = float(self.inputs[0])
        if self.is_double:
            packed = struct.pack('>d', val)
            int_val = struct.unpack('>Q', packed)[0]
        else:
            packed = struct.pack('>f', val)
            int_val = struct.unpack('>I', packed)[0]
        self.outputs[0] = float(int_val)
        return self.outputs


class ShiftArithmeticBlock(Block):
    """Shift bits or binary point of signal: y = u * 2^shift (or u >> shift)."""

    def __init__(self, name: str, shift_bits: int = 1, direction: str = "Left"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.shift = int(shift_bits)
        self.direction = direction.lower()
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        val = int(round(float(self.inputs[0])))
        if self.direction == "left":
            self.outputs[0] = float(val << self.shift)
        else:
            self.outputs[0] = float(val >> self.shift)
        return self.outputs


class CombinatorialLogicBlock(Block):
    """Implement truth table lookup.
    Truth table matrix of shape (2^num_inputs, num_outputs).
    """

    def __init__(self, name: str, truth_table: Sequence[Sequence[float]]):
        table = np.asarray(truth_table, dtype=float)
        num_out = table.shape[1] if table.ndim > 1 else 1
        num_in = int(math.log2(table.shape[0]))
        super().__init__(name, num_inputs=num_in, num_outputs=num_out)
        self.table = table
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        # Form integer row index from boolean inputs
        row = 0
        for x in self.inputs:
            b = 1 if abs(float(x)) > 0.5 else 0
            row = (row << 1) | b
        row = min(len(self.table) - 1, max(0, row))
        res = self.table[row]
        if isinstance(res, np.ndarray):
            self.outputs = [float(v) for v in res]
        else:
            self.outputs[0] = float(res)
        return self.outputs


class DetectChangeBlock(Block):
    """Detect change in signal value: y = 1 if u[k] != u[k-1], else 0."""

    def __init__(self, name: str, initial_value: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.prev_u = float(initial_value)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        changed = abs(u - self.prev_u) > 1e-9
        self.prev_u = u
        self.outputs[0] = 1.0 if changed else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.prev_u = 0.0


class DetectIncreaseBlock(Block):
    """Detect increase in signal value: y = 1 if u[k] > u[k-1], else 0."""

    def __init__(self, name: str, initial_value: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.prev_u = float(initial_value)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        inc = (u > self.prev_u + 1e-9)
        self.prev_u = u
        self.outputs[0] = 1.0 if inc else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.prev_u = 0.0


class DetectDecreaseBlock(Block):
    """Detect decrease in signal value: y = 1 if u[k] < u[k-1], else 0."""

    def __init__(self, name: str, initial_value: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.prev_u = float(initial_value)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        dec = (u < self.prev_u - 1e-9)
        self.prev_u = u
        self.outputs[0] = 1.0 if dec else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.prev_u = 0.0


class DetectRisePositiveBlock(Block):
    """Detect rising edge when signal increases to strictly positive value (u > 0 and prev_u <= 0)."""

    def __init__(self, name: str, initial_value: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.prev_u = float(initial_value)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        rise = (u > 0.0 and self.prev_u <= 0.0)
        self.prev_u = u
        self.outputs[0] = 1.0 if rise else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.prev_u = 0.0


class DetectRiseNonnegativeBlock(Block):
    """Detect rising edge when signal increases to nonnegative value (u >= 0 and prev_u < 0)."""

    def __init__(self, name: str, initial_value: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.prev_u = float(initial_value)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        rise = (u >= 0.0 and self.prev_u < 0.0)
        self.prev_u = u
        self.outputs[0] = 1.0 if rise else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.prev_u = 0.0


class DetectFallNegativeBlock(Block):
    """Detect falling edge when signal decreases to strictly negative value (u < 0 and prev_u >= 0)."""

    def __init__(self, name: str, initial_value: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.prev_u = float(initial_value)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        fall = (u < 0.0 and self.prev_u >= 0.0)
        self.prev_u = u
        self.outputs[0] = 1.0 if fall else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.prev_u = 0.0


class DetectFallNonpositiveBlock(Block):
    """Detect falling edge when signal decreases to nonpositive value (u <= 0 and prev_u > 0)."""

    def __init__(self, name: str, initial_value: float = 0.0):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.prev_u = float(initial_value)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        fall = (u <= 0.0 and self.prev_u > 0.0)
        self.prev_u = u
        self.outputs[0] = 1.0 if fall else 0.0
        return self.outputs

    def reset_states(self) -> None:
        self.prev_u = 0.0
