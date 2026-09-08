"""Block diagram connection manager, DAG topological scheduler, and algebraic loop resolver."""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from .base import Block
from .blocks import ScopeSinkBlock, DisplaySinkBlock
from .catalog import DynamicBlockCatalog, parse_parameter_value


class SystemDiagram:
    """Manages dynamic system blocks, connection routing, model settings, and file serialization."""

    def __init__(self, name: str = "Untitled_Model"):
        self.name = name
        self.description = "TerminusECE Dynamic System Model"
        self.solver = "rk4"         # rk4, euler, heun, ode45, ode23
        self.dt = 0.001             # Default integration step size (s)
        self.t_start = 0.0          # Simulation start time (s)
        self.t_stop = 10.0          # Simulation stop time (s)
        self.tolerance = 1e-6       # Relative tolerance for adaptive solvers

        self.blocks: Dict[str, Block] = {}
        # List of connections: ((src_block_name, src_port_idx), (dst_block_name, dst_port_idx))
        self.connections: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []
        # Stored block metadata (type, parameters) for full serialization
        self.block_metadata: Dict[str, Dict[str, Any]] = {}

    def clear(self) -> None:
        self.blocks.clear()
        self.connections.clear()
        self.block_metadata.clear()

    def add_block(self, block: Block, block_type: str = "", **params) -> None:
        name = block.name.upper()
        self.blocks[name] = block
        if not block_type:
            block_type = block.__class__.__name__
        extracted_params = dict(getattr(block, "parameters", {}))
        if params:
            extracted_params.update(params)
        else:
            for attr in ("gain", "kp", "ki", "kd", "n_filter", "num", "den", "A", "B", "C", "D",
                         "lower_limit", "upper_limit", "rising_slew_rate", "falling_slew_rate",
                         "start_zone", "end_zone", "deadband_width", "switch_on_point", "switch_off_point",
                         "output_on", "output_off", "f_coulomb", "b_viscous", "f_static",
                         "step_time", "amplitude", "slope", "start_time", "freq", "phase", "bias",
                         "period", "duty_cycle", "noise_power", "value", "delay_time", "sample_time",
                         "signs", "operations", "function", "threshold", "tag", "operator"):
                if hasattr(block, attr):
                    val = getattr(block, attr)
                    if not callable(val):
                        extracted_params[attr] = val
        block.parameters = extracted_params
        self.block_metadata[name] = {
            "type": block_type,
            "params": extracted_params
        }

    def remove_block(self, name: str) -> bool:
        uname = name.upper().strip()
        if uname in self.blocks:
            del self.blocks[uname]
            if uname in self.block_metadata:
                del self.block_metadata[uname]
            # Remove all connections involving this block
            self.connections = [
                (src, dst) for src, dst in self.connections
                if src[0] != uname and dst[0] != uname
            ]
            return True
        return False

    def get_block(self, name: str) -> Optional[Block]:
        return self.blocks.get(name.upper().strip())

    def connect(self, src_endpoint: str, dst_endpoint: str) -> None:
        """Connects source output port to destination input port.
        
        Example: connect("Step1.0", "Sum1.0") or connect("Plant.out", "Scope1.in")
        """
        def parse_port(ep: str, is_src: bool) -> Tuple[str, int]:
            parts = ep.strip().split(".")
            bname = parts[0].upper()
            port = 0
            if len(parts) > 1:
                p_str = parts[1].lower()
                if p_str.isdigit():
                    port = int(p_str)
                elif p_str in ("out", "y", "output"):
                    port = 0
                elif p_str in ("in", "u", "input"):
                    port = 0
                elif p_str.startswith("in") and p_str[2:].isdigit():
                    port = int(p_str[2:])
                elif p_str.startswith("out") and p_str[3:].isdigit():
                    port = int(p_str[3:])
            return bname, port

        src_b, src_p = parse_port(src_endpoint, is_src=True)
        dst_b, dst_p = parse_port(dst_endpoint, is_src=False)

        if src_b not in self.blocks:
            raise KeyError(f"Source block '{src_b}' does not exist.")
        if dst_b not in self.blocks:
            raise KeyError(f"Destination block '{dst_b}' does not exist.")

        conn = ((src_b, src_p), (dst_b, dst_p))
        if conn not in self.connections:
            self.connections.append(conn)

    def disconnect(self, src_endpoint: str, dst_endpoint: str) -> bool:
        """Removes a connection between two endpoints."""
        def parse_port(ep: str, is_src: bool) -> Tuple[str, int]:
            parts = ep.strip().split(".")
            bname = parts[0].upper()
            port = 0
            if len(parts) > 1:
                p_str = parts[1].lower()
                if p_str.isdigit():
                    port = int(p_str)
            return bname, port

        src_b, src_p = parse_port(src_endpoint, is_src=True)
        dst_b, dst_p = parse_port(dst_endpoint, is_src=False)

        target = ((src_b, src_p), (dst_b, dst_p))
        if target in self.connections:
            self.connections.remove(target)
            return True
        return False

    def tune_parameter(self, block_name: str, param_name: str, value: Any) -> Any:
        """Tuning parameter on an existing block dynamically."""
        block = self.get_block(block_name)
        if not block:
            raise KeyError(f"Block '{block_name}' not found.")

        # Find descriptor from catalog to parse parameter value with correct type
        bmeta = self.block_metadata.get(block_name.upper(), {})
        btype = bmeta.get("type", block.__class__.__name__)
        desc = DynamicBlockCatalog.get_descriptor(btype)

        parsed_val = value
        if desc and param_name in desc.parameters:
            target_type = desc.parameters[param_name].param_type
            parsed_val = parse_parameter_value(value, target_type)
        elif isinstance(value, str):
            parsed_val = parse_parameter_value(value, float)

        # Set on block attribute
        if hasattr(block, param_name):
            setattr(block, param_name, parsed_val)
        block.parameters[param_name] = parsed_val
        if block_name.upper() in self.block_metadata:
            self.block_metadata[block_name.upper()]["params"][param_name] = parsed_val

        return parsed_val

    def to_dict(self) -> Dict[str, Any]:
        """Serializes diagram to dictionary representation."""
        blocks_data = {}
        for bname, block in self.blocks.items():
            meta = self.block_metadata.get(bname, {})
            btype = meta.get("type", block.__class__.__name__)
            params = {}
            for k, v in getattr(block, "parameters", {}).items():
                if hasattr(block, k):
                    live_v = getattr(block, k)
                    if not callable(live_v):
                        v = live_v
                if isinstance(v, np.ndarray):
                    params[k] = v.tolist()
                else:
                    params[k] = v
            blocks_data[bname] = {
                "type": btype,
                "parameters": params,
            }

        connections_data = [
            {"src": f"{src[0]}.{src[1]}", "dst": f"{dst[0]}.{dst[1]}"}
            for src, dst in self.connections
        ]

        return {
            "model_name": self.name,
            "description": self.description,
            "settings": {
                "solver": self.solver,
                "dt": self.dt,
                "t_start": self.t_start,
                "t_stop": self.t_stop,
                "tolerance": self.tolerance,
            },
            "blocks": blocks_data,
            "connections": connections_data,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Loads diagram structure from dictionary."""
        self.clear()
        self.name = data.get("model_name", "Model")
        self.description = data.get("description", "")
        settings = data.get("settings", {})
        self.solver = settings.get("solver", "rk4")
        self.dt = float(settings.get("dt", 0.001))
        self.t_start = float(settings.get("t_start", 0.0))
        self.t_stop = float(settings.get("t_stop", 10.0))
        self.tolerance = float(settings.get("tolerance", 1e-6))

        # Recreate blocks
        blocks_data = data.get("blocks", {})
        for bname, binfo in blocks_data.items():
            btype = binfo.get("type", "")
            params = binfo.get("parameters", {})
            block = DynamicBlockCatalog.instantiate(btype, bname, **params)
            self.add_block(block, block_type=btype, **params)

        # Recreate connections
        conns_data = data.get("connections", [])
        for c in conns_data:
            self.connect(c["src"], c["dst"])

    def save_to_file(self, file_path: str) -> None:
        """Saves model to a JSON / TMDL file."""
        data = self.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, file_path: str) -> None:
        """Loads model from a JSON / TMDL file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file '{file_path}' not found.")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.from_dict(data)


class Scheduler:
    """Computes valid execution order for block evaluations in dynamic systems."""

    def __init__(self, diagram: SystemDiagram):
        self.diagram = diagram
        self.execution_order: List[Block] = []
        self.stateful_blocks: List[Block] = []
        self.discrete_blocks: List[Block] = []

    def build_schedule(self) -> List[Block]:
        """Performs topological sort on direct-feedthrough dependencies."""
        blocks = list(self.diagram.blocks.values())
        self.stateful_blocks = [b for b in blocks if getattr(b, "num_states", 0) > 0 and getattr(b, "sample_time", 0.0) == 0.0]
        self.discrete_blocks = [b for b in blocks if getattr(b, "sample_time", 0.0) > 0.0]

        # In-degree of direct feedthrough dependencies
        adj: Dict[str, Set[str]] = {b.name.upper(): set() for b in blocks}
        in_degree: Dict[str, int] = {b.name.upper(): 0 for b in blocks}

        for (src_name, src_port), (dst_name, dst_port) in self.diagram.connections:
            if dst_name in self.diagram.blocks and src_name in self.diagram.blocks:
                dst_block = self.diagram.blocks[dst_name]
                if getattr(dst_block, "direct_feedthrough", True):
                    # dst depends on src
                    if dst_name not in adj[src_name]:
                        adj[src_name].add(dst_name)
                        in_degree[dst_name] += 1

        # Kahn's algorithm for topological sorting
        queue = [bname for bname, deg in in_degree.items() if deg == 0]
        order_names: List[str] = []

        while queue:
            curr = queue.pop(0)
            order_names.append(curr)

            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order_names) < len(blocks):
            # Algebraic loop detected: append remaining blocks
            for bname in self.diagram.blocks:
                if bname not in order_names:
                    order_names.append(bname)

        self.execution_order = [self.diagram.blocks[name] for name in order_names if name in self.diagram.blocks]
        return self.execution_order

    def propagate_signals(self, t: float) -> None:
        """Evaluates blocks in scheduled order and propagates output values through connections."""
        for block in self.execution_order:
            block.compute_output(t)
            # Propagate outputs to connected inputs
            for (src_name, src_port), (dst_name, dst_port) in self.diagram.connections:
                if src_name == block.name.upper():
                    dst_block = self.diagram.blocks.get(dst_name)
                    if dst_block:
                        if src_port < len(block.outputs) and dst_port < len(dst_block.inputs):
                            dst_block.inputs[dst_port] = block.outputs[src_port]

        # Record scopes and displays
        for block in self.execution_order:
            if isinstance(block, ScopeSinkBlock):
                block.record(t)

    def pack_states(self) -> np.ndarray:
        """Packs all continuous block state vectors into a single flat array."""
        state_list = [b.states for b in self.stateful_blocks]
        if not state_list:
            return np.zeros(0, dtype=float)
        return np.concatenate(state_list)

    def unpack_states(self, state_vec: np.ndarray) -> None:
        """Unpacks a flat state vector back into the respective block states."""
        idx = 0
        for b in self.stateful_blocks:
            n = b.num_states
            b.states = np.array(state_vec[idx : idx + n], dtype=float)
            idx += n

    def compute_all_derivatives(self, t: float) -> np.ndarray:
        """Computes concatenated time derivatives for the continuous states at time t."""
        self.propagate_signals(t)
        deriv_list = [b.compute_derivatives(t) for b in self.stateful_blocks]
        if not deriv_list:
            return np.zeros(0, dtype=float)
        return np.concatenate(deriv_list)

    def update_discrete_states(self, t: float, dt: float) -> None:
        """Fires discrete update events for discrete multi-rate blocks."""
        for b in self.discrete_blocks:
            ts = getattr(b, "sample_time", 0.0)
            if ts > 0.0:
                # Check if t is approximately a multiple of sample_time
                hit = (abs(round(t / ts) * ts - t) < (dt * 0.51))
                if hit:
                    b.discrete_update(t)
