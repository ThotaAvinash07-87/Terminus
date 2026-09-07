"""Model-Wide Documentation, Metadata, and Linearization Utilities for Dynamic Systems."""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from .base import Block


class DocBlock(Block):
    """Documentation Block: stores markdown / text notes and model documentation."""

    def __init__(self, name: str, text: str = "System Documentation"):
        super().__init__(name, num_inputs=0, num_outputs=0)
        self.doc_text = text


class ModelInfoBlock(Block):
    """Display model metadata, author, date, and version information."""

    def __init__(self, name: str, model_name: str = "TerminusModel", version: str = "1.0", author: str = "Terminus User"):
        super().__init__(name, num_inputs=0, num_outputs=0)
        self.model_name = model_name
        self.version = version
        self.author = author


class BlockSupportTableBlock(Block):
    """View data type and code-generation support table for blocks."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=0, num_outputs=0)


class TimedBasedLinearizationBlock(Block):
    """Linearize model around operating point at specified snapshot times."""

    def __init__(self, name: str, snapshot_times: Sequence[float] = (1.0, 5.0)):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.times = list(snapshot_times)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = self.inputs[0]
        return self.outputs


class TriggerBasedLinearizationBlock(Block):
    """Linearize model when triggered."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = self.inputs[0]
        return self.outputs
