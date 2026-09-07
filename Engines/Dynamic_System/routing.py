"""Signal Routing, Buses, Multiplexers, Switches, and Data Store blocks for Dynamic Systems."""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from .base import Block


class SwitchBlock(Block):
    """3-Input Switch: outputs input 1 if input 2 (control) >= threshold, else outputs input 3."""

    def __init__(self, name: str, threshold: float = 0.0, criteria: str = ">="):
        super().__init__(name, num_inputs=3, num_outputs=1)
        self.threshold = float(threshold)
        self.criteria = criteria
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        in1, ctrl, in3 = self.inputs[0], float(self.inputs[1]) if len(self.inputs) > 1 else 0.0, self.inputs[2] if len(self.inputs) > 2 else 0.0
        if self.criteria == ">=":
            cond = ctrl >= self.threshold
        elif self.criteria == ">":
            cond = ctrl > self.threshold
        elif self.criteria == "~=":
            cond = abs(ctrl - self.threshold) > 1e-6
        else:
            cond = ctrl >= self.threshold
        self.outputs[0] = in1 if cond else in3
        return self.outputs


class MultiportSwitchBlock(Block):
    """Select output signal based on integer control input index."""

    def __init__(self, name: str, num_data_inputs: int = 3, zero_indexed: bool = False):
        self.num_data = max(1, int(num_data_inputs))
        super().__init__(name, num_inputs=1 + self.num_data, num_outputs=1)
        self.zero_indexed = zero_indexed
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        ctrl = int(round(float(self.inputs[0])))
        idx = ctrl if self.zero_indexed else ctrl - 1
        idx = max(0, min(self.num_data - 1, idx))
        self.outputs[0] = self.inputs[1 + idx]
        return self.outputs


IndexVectorBlock = MultiportSwitchBlock


class ManualSwitchBlock(Block):
    """Manual switch between two inputs (toggled by user or state)."""

    def __init__(self, name: str, switch_state: int = 0):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.state = int(switch_state)  # 0 for in0, 1 for in1
        self.direct_feedthrough = True

    def toggle(self) -> None:
        self.state = 1 if self.state == 0 else 0

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = self.inputs[1 if self.state == 1 else 0]
        return self.outputs


class MergeBlock(Block):
    """Combine multiple signal paths into a single output (outputs the active/latest update)."""

    def __init__(self, name: str, num_inputs: int = 2, initial_output: float = 0.0):
        super().__init__(name, num_inputs=num_inputs, num_outputs=1)
        self.outputs[0] = float(initial_output)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        for val in self.inputs:
            if val is not None and (not isinstance(val, (int, float)) or abs(val) > 1e-12):
                self.outputs[0] = val
        return self.outputs


class MuxBlock(Block):
    """Combine input scalar/vector signals into a virtual vector."""

    def __init__(self, name: str, num_inputs: int = 2):
        super().__init__(name, num_inputs=num_inputs, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        vector = []
        for x in self.inputs:
            if isinstance(x, (list, tuple, np.ndarray)):
                vector.extend(list(x))
            else:
                vector.append(x)
        self.outputs[0] = np.array(vector, dtype=float) if all(isinstance(v, (int, float, np.number)) for v in vector) else vector
        return self.outputs


class DemuxBlock(Block):
    """Extract and split elements of vector signal across scalar output ports."""

    def __init__(self, name: str, num_outputs: int = 2):
        super().__init__(name, num_inputs=1, num_outputs=num_outputs)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        inp = self.inputs[0]
        if isinstance(inp, (list, tuple, np.ndarray)):
            for i in range(self.num_outputs):
                self.outputs[i] = inp[i] if i < len(inp) else 0.0
        else:
            self.outputs[0] = inp
            for i in range(1, self.num_outputs):
                self.outputs[i] = 0.0
        return self.outputs


class SelectorBlock(Block):
    """Select elements from vector or matrix input using specified index array."""

    def __init__(self, name: str, indices: Sequence[int]):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.indices = list(indices)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        arr = np.asarray(self.inputs[0])
        self.outputs[0] = arr[self.indices]
        return self.outputs


class BusCreatorBlock(Block):
    """Group input signals into a structured dictionary bus."""

    def __init__(self, name: str, signal_names: Sequence[str]):
        super().__init__(name, num_inputs=len(signal_names), num_outputs=1)
        self.signal_names = list(signal_names)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        bus = {self.signal_names[i]: self.inputs[i] for i in range(len(self.signal_names))}
        self.outputs[0] = bus
        return self.outputs


class BusSelectorBlock(Block):
    """Select and output specified named signals from a bus dictionary."""

    def __init__(self, name: str, output_signals: Sequence[str]):
        super().__init__(name, num_inputs=1, num_outputs=len(output_signals))
        self.output_signals = list(output_signals)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        bus = self.inputs[0]
        if isinstance(bus, dict):
            for i, sig in enumerate(self.output_signals):
                self.outputs[i] = bus.get(sig, 0.0)
        return self.outputs


class BusAssignmentBlock(Block):
    """Assign/replace new values for specified elements in bus dictionary."""

    def __init__(self, name: str, assigned_signals: Sequence[str]):
        # Input 0: Bus, Inputs 1..N: replacement signals
        super().__init__(name, num_inputs=1 + len(assigned_signals), num_outputs=1)
        self.assigned_signals = list(assigned_signals)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        bus = dict(self.inputs[0]) if isinstance(self.inputs[0], dict) else {}
        for i, sig in enumerate(self.assigned_signals):
            if i + 1 < len(self.inputs):
                bus[sig] = self.inputs[i + 1]
        self.outputs[0] = bus
        return self.outputs


# Global tag router for Goto / From blocks
_GLOBAL_GOTO_TAGS: Dict[str, Any] = {}
_GLOBAL_DATA_STORES: Dict[str, Any] = {}


class GotoBlock(Block):
    """Pass block input to corresponding From blocks using wireless tag."""

    def __init__(self, name: str, tag: str = "A", tag_visibility: str = "global"):
        super().__init__(name, num_inputs=1, num_outputs=0)
        self.tag = tag.strip().upper()
        self.visibility = tag_visibility
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        _GLOBAL_GOTO_TAGS[self.tag] = self.inputs[0]
        return []


class FromBlock(Block):
    """Accept input wirelessly from matching Goto block tag."""

    def __init__(self, name: str, tag: str = "A"):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.tag = tag.strip().upper()
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = _GLOBAL_GOTO_TAGS.get(self.tag, 0.0)
        return self.outputs


class GotoTagVisibilityBlock(Block):
    """Define scope of Goto block tag."""

    def __init__(self, name: str, tag: str = "A"):
        super().__init__(name, num_inputs=0, num_outputs=0)
        self.tag = tag


class DataStoreMemoryBlock(Block):
    """Define global/shared data store memory."""

    def __init__(self, name: str, data_store_name: str = "dstore", initial_value: Any = 0.0):
        super().__init__(name, num_inputs=0, num_outputs=0)
        self.ds_name = data_store_name.strip().upper()
        self.ic = initial_value
        _GLOBAL_DATA_STORES[self.ds_name] = self.ic

    def reset_states(self) -> None:
        _GLOBAL_DATA_STORES[self.ds_name] = self.ic


class DataStoreReadBlock(Block):
    """Read value from data store."""

    def __init__(self, name: str, data_store_name: str = "dstore"):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.ds_name = data_store_name.strip().upper()
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = _GLOBAL_DATA_STORES.get(self.ds_name, 0.0)
        return self.outputs


class DataStoreWriteBlock(Block):
    """Write value to data store."""

    def __init__(self, name: str, data_store_name: str = "dstore"):
        super().__init__(name, num_inputs=1, num_outputs=0)
        self.ds_name = data_store_name.strip().upper()
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        _GLOBAL_DATA_STORES[self.ds_name] = self.inputs[0]
        return []


class StateReaderBlock(Block):
    """Read an external block state."""

    def __init__(self, name: str, target_block_name: str = ""):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.target_name = target_block_name.strip().upper()
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        return self.outputs


class StateWriterBlock(Block):
    """Write to an external block state."""

    def __init__(self, name: str, target_block_name: str = ""):
        super().__init__(name, num_inputs=1, num_outputs=0)
        self.target_name = target_block_name.strip().upper()
        self.direct_feedthrough = True


class ParameterWriterBlock(Block):
    """Write to a block parameter."""

    def __init__(self, name: str, target_block_name: str = "", parameter_name: str = ""):
        super().__init__(name, num_inputs=1, num_outputs=0)
        self.target_name = target_block_name.strip().upper()
        self.param_name = parameter_name


class EnvironmentControllerBlock(Block):
    """Create branches that apply to simulation vs code generation."""

    def __init__(self, name: str, is_simulation: bool = True):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.is_sim = is_simulation
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = self.inputs[0] if self.is_sim else self.inputs[1]
        return self.outputs


class VariantSourceBlock(Block):
    """Route among multiple inputs using active variant choice index."""

    def __init__(self, name: str, num_choices: int = 2, active_choice: int = 0):
        super().__init__(name, num_inputs=num_choices, num_outputs=1)
        self.active_choice = int(active_choice)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        idx = max(0, min(len(self.inputs) - 1, self.active_choice))
        self.outputs[0] = self.inputs[idx]
        return self.outputs


class VariantSinkBlock(Block):
    """Route input to one of multiple outputs using active choice."""

    def __init__(self, name: str, num_choices: int = 2, active_choice: int = 0):
        super().__init__(name, num_inputs=1, num_outputs=num_choices)
        self.active_choice = int(active_choice)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        for i in range(self.num_outputs):
            self.outputs[i] = self.inputs[0] if i == self.active_choice else 0.0
        return self.outputs


ManualVariantSourceBlock = VariantSourceBlock
ManualVariantSinkBlock = VariantSinkBlock
VariantStartBlock = VariantSourceBlock
VariantEndBlock = VariantSinkBlock
