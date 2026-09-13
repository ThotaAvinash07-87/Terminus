"""Parser for structural HDL descriptions and boolean expression truth tables."""

from __future__ import annotations
import itertools
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from CORE.common_math import parse_eng_unit
from .gates import (
    LogicValue,
    LogicGate,
    AndGate,
    OrGate,
    NotGate,
    NandGate,
    NorGate,
    XorGate,
    XnorGate,
    BufferGate,
    Mux2to1,
    DFlipFlop,
    JKFlipFlop,
    ClockGenerator,
)


class LogicCircuit:
    """Manages digital gates, flip-flops, wire interconnections, and live Verilog HDL editor buffer."""

    def __init__(self, name: str = "XilinxTop"):
        self.name = name
        self.wires: Set[str] = set()
        self.gates: Dict[str, LogicGate] = {}
        self.flip_flops: Dict[str, Union[DFlipFlop, JKFlipFlop]] = {}
        self.clocks: Dict[str, ClockGenerator] = {}
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self._source_code: str = (
            f"// ==========================================\n"
            f"// Xilinx Vivado HDL Module: {name}\n"
            f"// ==========================================\n\n"
            f"wire a, b, clk, q, out;\n"
            f"clock CLK period=10ns\n"
            f"gate G1 AND a b -> out\n"
            f"dff D1 clk=clk d=out q=q\n"
        )
        self.active_filename = f"{name}.v"
        self._file_mtime: float = 0.0

    @property
    def source_code(self) -> str:
        self.sync_from_disk()
        return self._source_code

    @source_code.setter
    def source_code(self, val: str):
        self._source_code = val
        self._recompile_hdl()

    def _recompile_hdl(self):
        """Parses the current source code into circuit gates and wires."""
        self.wires.clear()
        self.gates.clear()
        self.flip_flops.clear()
        self.clocks.clear()
        self.inputs.clear()
        self.outputs.clear()
        for line in self._source_code.splitlines():
            HDLParser.parse_line(self, line)

    def sync_from_disk(self) -> bool:
        """Checks for external editor disk modifications."""
        from CORE.storage_manager import StorageManager
        sm = StorageManager.get_instance()
        proj_dir = sm.get_mode_dir("DIGITAL") / self.name
        fpath = proj_dir / self.active_filename
        if fpath.exists():
            try:
                mtime = fpath.stat().st_mtime
                if mtime > self._file_mtime:
                    with open(fpath, "r", encoding="utf-8") as f:
                        self._source_code = f.read()
                    self._file_mtime = mtime
                    self._recompile_hdl()
                    return True
            except Exception:
                pass
        return False

    def save_to_disk(self) -> Path:
        from CORE.storage_manager import StorageManager
        sm = StorageManager.get_instance()
        proj_dir = sm.get_mode_dir("DIGITAL") / self.name
        proj_dir.mkdir(parents=True, exist_ok=True)
        fpath = proj_dir / self.active_filename
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(self._source_code)
        self._file_mtime = fpath.stat().st_mtime
        return fpath

    def edit_line(self, line_num: int, new_text: str) -> bool:
        lines = self._source_code.splitlines()
        if 1 <= line_num <= len(lines):
            lines[line_num - 1] = new_text
            self.source_code = "\n".join(lines) + "\n"
            self.save_to_disk()
            return True
        elif line_num == len(lines) + 1:
            lines.append(new_text)
            self.source_code = "\n".join(lines) + "\n"
            self.save_to_disk()
            return True
        return False

    def insert_line(self, line_num: int, new_text: str) -> bool:
        lines = self._source_code.splitlines()
        idx = max(0, min(len(lines), line_num - 1))
        lines.insert(idx, new_text)
        self.source_code = "\n".join(lines) + "\n"
        self.save_to_disk()
        return True

    def delete_line(self, line_num: int) -> bool:
        lines = self._source_code.splitlines()
        if 1 <= line_num <= len(lines):
            lines.pop(line_num - 1)
            self.source_code = "\n".join(lines) + "\n"
            self.save_to_disk()
            return True
        return False

    def append_line(self, line_text: str) -> None:
        self.source_code = (self._source_code + "\n" if self._source_code else "") + line_text + "\n"
        self.save_to_disk()

    def clear_code(self) -> None:
        self.source_code = ""
        self.save_to_disk()

    def view_code(self, cursor_line: Optional[int] = None, cursor_col: Optional[int] = None) -> str:
        self.sync_from_disk()
        if not self._source_code.strip():
            return f"// Module: {self.name} ({self.active_filename})\n// (Empty HDL script. Type 'line add <code...>' or 'ide')"
        lines = self._source_code.splitlines()
        formatted = [f"// Module: {self.name} ({self.active_filename})"]
        for i, l in enumerate(lines, 1):
            cursor_mark = "▶" if cursor_line == i else " "
            formatted.append(f"{cursor_mark}{i:02d} │ {l}")
        return "\n".join(formatted)

    def launch_external_editor(self, preferred_editor: Optional[str] = None) -> Tuple[bool, str]:
        fpath = self.save_to_disk()
        from CORE.editor_detector import HostEditorManager
        return HostEditorManager.launch(fpath, preferred_editor)

    def clear(self) -> None:
        self.wires.clear()
        self.gates.clear()
        self.flip_flops.clear()
        self.clocks.clear()
        self.inputs.clear()
        self.outputs.clear()



class HDLParser:
    """Parses structural logic netlist strings into a LogicCircuit."""

    GATE_FACTORIES = {
        "AND": AndGate,
        "OR": OrGate,
        "NOT": NotGate,
        "NAND": NandGate,
        "NOR": NorGate,
        "XOR": XorGate,
        "XNOR": XnorGate,
        "BUF": BufferGate,
        "BUFFER": BufferGate,
    }

    @classmethod
    def parse_line(cls, circuit: LogicCircuit, line: str) -> None:
        """Parses a single HDL line and modifies the logic circuit in-place."""
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            return

        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("wire", "wires"):
            wire_names = [w.strip(", ") for w in parts[1:] if w.strip(", ")]
            for w in wire_names:
                circuit.wires.add(w)

        elif cmd in ("input", "inputs"):
            inp_names = [w.strip(", ") for w in parts[1:] if w.strip(", ")]
            for w in inp_names:
                circuit.wires.add(w)
                if w not in circuit.inputs:
                    circuit.inputs.append(w)

        elif cmd in ("output", "outputs"):
            out_names = [w.strip(", ") for w in parts[1:] if w.strip(", ")]
            for w in out_names:
                circuit.wires.add(w)
                if w not in circuit.outputs:
                    circuit.outputs.append(w)

        elif cmd == "gate":
            # gate G1 AND in1 in2 -> out [delay=1ns]
            if "->" not in parts:
                return
            arrow_idx = parts.index("->")
            gname = parts[1].upper()
            gtype = parts[2].upper()
            in_wires = parts[3:arrow_idx]
            out_wire = parts[arrow_idx + 1]

            delay = 1.0
            if len(parts) > arrow_idx + 2 and "=" in parts[arrow_idx + 2]:
                delay = parse_eng_unit(parts[arrow_idx + 2].split("=")[1])

            for w in in_wires + [out_wire]:
                circuit.wires.add(w)

            factory = cls.GATE_FACTORIES.get(gtype)
            if factory:
                circuit.gates[gname] = factory(gname, in_wires, out_wire, delay_ns=delay)

        elif cmd == "dff":
            # dff D1 clk=clk d=data q=q_out [rst=rst]
            gname = parts[1].upper()
            kwargs = {}
            for p in parts[2:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    kwargs[k.lower()] = v
            clk_w = kwargs.get("clk", "clk")
            d_w = kwargs.get("d", "d")
            q_w = kwargs.get("q", "q")
            rst_w = kwargs.get("rst")
            delay = parse_eng_unit(kwargs.get("delay", "1ns"))

            circuit.wires.add(clk_w)
            circuit.wires.add(d_w)
            circuit.wires.add(q_w)
            if rst_w:
                circuit.wires.add(rst_w)

            circuit.flip_flops[gname] = DFlipFlop(gname, clk_w, d_w, q_w, rst_wire=rst_w, delay_ns=delay)

        elif cmd == "clock":
            # clock CLK period=10ns [duty=0.5]
            cname = parts[1]
            kwargs = {}
            for p in parts[2:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    kwargs[k.lower()] = v
            period = parse_eng_unit(kwargs.get("period", "10ns"))
            duty = float(kwargs.get("duty", "0.5"))
            circuit.wires.add(cname)
            circuit.clocks[cname] = ClockGenerator(cname, period_ns=period * 1e9 if period < 1e-3 else period, duty_cycle=duty)

    @classmethod
    def parse(cls, hdl_text: str) -> LogicCircuit:
        circuit = LogicCircuit()
        for line in hdl_text.strip().splitlines():
            cls.parse_line(circuit, line)
        return circuit

    @classmethod
    def generate_truth_table(cls, expr: str, var_names: Optional[List[str]] = None) -> str:
        """Generates a formatted ASCII truth table for a boolean logic expression."""
        if var_names is None:
            # Extract single-letter or word identifiers
            tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expr)
            # Remove python keywords / operators
            reserved = {"and", "or", "not", "xor", "xnor", "nand", "nor", "True", "False"}
            var_names = sorted(list(set(t for t in tokens if t.lower() not in reserved)))

        # Pythonize expression
        py_expr = expr
        py_expr = re.sub(r'~|!', ' not ', py_expr)
        py_expr = re.sub(r'&|\*', ' and ', py_expr)
        py_expr = re.sub(r'\||\+', ' or ', py_expr)
        py_expr = re.sub(r'\^', ' != ', py_expr)

        header = " | ".join(var_names) + f" | Result ({expr})"
        sep = "-" * len(header)
        rows = [header, sep]

        for combo in itertools.product([0, 1], repeat=len(var_names)):
            env = dict(zip(var_names, combo))
            try:
                res = int(bool(eval(py_expr, {"__builtins__": {}}, env)))
            except Exception:
                res = "E"
            row_vals = " | ".join(f"{v:^{len(k)}}" for k, v in env.items())
            rows.append(f"{row_vals} | {res:^{len(expr) + 8}}")

        return "\n".join(rows)
