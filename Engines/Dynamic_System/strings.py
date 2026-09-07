"""String Processing, Formatting, and ASCII conversion blocks for Dynamic Systems."""

from __future__ import annotations
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from .base import Block


class StringConstantBlock(Block):
    """Output specified string constant."""

    def __init__(self, name: str, string_value: str = "Hello"):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.val = str(string_value)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[str]:
        self.outputs[0] = self.val
        return self.outputs


class StringCompareBlock(Block):
    """Compare two input strings: outputs 1 if equal, 0 otherwise."""

    def __init__(self, name: str, case_sensitive: bool = True):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.case_sensitive = case_sensitive
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        s1 = str(self.inputs[0])
        s2 = str(self.inputs[1]) if len(self.inputs) > 1 else ""
        if not self.case_sensitive:
            s1, s2 = s1.lower(), s2.lower()
        self.outputs[0] = 1.0 if s1 == s2 else 0.0
        return self.outputs


class StringConcatenateBlock(Block):
    """Concatenate input strings to form one output string."""

    def __init__(self, name: str, num_inputs: int = 2, separator: str = ""):
        super().__init__(name, num_inputs=num_inputs, num_outputs=1)
        self.sep = separator
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[str]:
        strs = [str(x) for x in self.inputs]
        self.outputs[0] = self.sep.join(strs)
        return self.outputs


class StringContainsBlock(Block):
    """Determine if string contains pattern."""

    def __init__(self, name: str, pattern: str = ""):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.pattern = pattern
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        s = str(self.inputs[0])
        self.outputs[0] = 1.0 if self.pattern in s else 0.0
        return self.outputs


class StringCountBlock(Block):
    """Count occurrences of pattern in string."""

    def __init__(self, name: str, pattern: str = "a"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.pattern = pattern
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        s = str(self.inputs[0])
        self.outputs[0] = float(s.count(self.pattern))
        return self.outputs


class StringFindBlock(Block):
    """Return 1-based index (or 0 if not found) of first occurrence of pattern string."""

    def __init__(self, name: str, pattern: str = "a"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.pattern = pattern
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        s = str(self.inputs[0])
        idx = s.find(self.pattern)
        self.outputs[0] = float(idx + 1 if idx != -1 else 0)
        return self.outputs


class StringLengthBlock(Block):
    """Output number of characters in input string."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        s = str(self.inputs[0])
        self.outputs[0] = float(len(s))
        return self.outputs


class SubstringBlock(Block):
    """Extract substring from input string: start_idx to end_idx (1-based index)."""

    def __init__(self, name: str, start_index: int = 1, end_index: int = 5):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.start_idx = max(1, int(start_index))
        self.end_idx = max(1, int(end_index))
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[str]:
        s = str(self.inputs[0])
        self.outputs[0] = s[self.start_idx - 1: self.end_idx]
        return self.outputs


class ToStringBlock(Block):
    """Convert input numeric signal to string signal."""

    def __init__(self, name: str, format_spec: str = "{:.4f}"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.fmt = format_spec
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[str]:
        val = self.inputs[0]
        try:
            self.outputs[0] = self.fmt.format(val)
        except Exception:
            self.outputs[0] = str(val)
        return self.outputs


class StringToDoubleBlock(Block):
    """Convert string signal to double float signal."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        s = str(self.inputs[0]).strip()
        try:
            self.outputs[0] = float(s)
        except ValueError:
            self.outputs[0] = 0.0
        return self.outputs


StringToSingleBlock = StringToDoubleBlock


class StringToEnumBlock(Block):
    """Map input string signal to enumerated integer index."""

    def __init__(self, name: str, enum_members: Sequence[str]):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.members = {name.lower(): idx for idx, name in enumerate(enum_members)}
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        s = str(self.inputs[0]).strip().lower()
        self.outputs[0] = float(self.members.get(s, -1))
        return self.outputs


class StringToASCIIBlock(Block):
    """Convert string signal to uint8 ASCII code vector."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        s = str(self.inputs[0])
        self.outputs[0] = [ord(c) for c in s]
        return self.outputs


class ASCIItoStringBlock(Block):
    """Convert ASCII integer code vector to string."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[str]:
        codes = self.inputs[0]
        if isinstance(codes, (list, tuple)):
            self.outputs[0] = "".join(chr(int(c)) for c in codes)
        else:
            self.outputs[0] = chr(int(codes))
        return self.outputs


class ComposeStringBlock(Block):
    """Compose output string based on format template and inputs."""

    def __init__(self, name: str, format_string: str = "Value: {}", num_inputs: int = 1):
        super().__init__(name, num_inputs=num_inputs, num_outputs=1)
        self.fmt = format_string
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[str]:
        try:
            self.outputs[0] = self.fmt.format(*self.inputs)
        except Exception:
            self.outputs[0] = self.fmt
        return self.outputs


class ScanStringBlock(Block):
    """Scan input string using regex pattern and extract matched numerical values."""

    def __init__(self, name: str, regex_pattern: str = r"[-+]?\d*\.?\d+", num_outputs: int = 1):
        super().__init__(name, num_inputs=1, num_outputs=num_outputs)
        self.pattern = re.compile(regex_pattern)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        s = str(self.inputs[0])
        matches = self.pattern.findall(s)
        for i in range(self.num_outputs):
            self.outputs[i] = float(matches[i]) if i < len(matches) else 0.0
        return self.outputs
