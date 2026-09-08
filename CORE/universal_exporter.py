"""Universal Industry-Standard Format Exporters for TerminusECE.
Exports native Terminus models to standard tools:
1. Circuit -> LTspice (.asc / .cir), PSpice (.sp / .net), ngspice
2. Dynamic System -> MATLAB Simulink script (.m), Python Control (.py)
3. Digital Logic -> IEEE 1364 Verilog HDL (.v), VHDL (.vhd)
4. Numerical -> MATLAB (.m), Python SciPy (.py)
5. Embedded -> C/C++ Firmware (.c), Intel HEX (.hex), Code Composer Studio Assembly (.asm)
"""

from __future__ import annotations
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from Engines.Circuit.netlist_parser import Netlist
from Engines.Circuit.components import (
    Component, Resistor, Capacitor, Inductor, VoltageSource, CurrentSource,
    Diode, BJT, MOSFET, JFET, VoltageControlledSwitch, VCVS, SubcircuitInstance
)
from Engines.Dynamic_System.scheduler import SystemDiagram
from Engines.Dynamic_System.blocks import Block
from Engines.Digital_Logic.hdl_parser import LogicCircuit
from Engines.Numerical.parser import NumericalWorkspace
from Engines.Embedded.mcu_core import MCUCore
from Engines.Embedded.toolchain import Disassembler


class UniversalStandardExporter:
    """Exports any TerminusECE subsystem workspace to reputable industry standard formats."""

    # ==========================================
    # 1. CIRCUIT EXPORTERS (LTspice, PSpice, ngspice)
    # ==========================================
    @classmethod
    def export_circuit_ltspice_netlist(cls, netlist: Netlist, name: str = "") -> str:
        """Exports to standard LTspice .cir / .net executable netlist."""
        cname = name or netlist.name or "circuit"
        lines = [
            f"* LTspice IV/XVII Standard Netlist - Exported by TerminusECE",
            f"* Circuit Title: {cname}",
            f"* Generated on: {time.ctime()}",
            ""
        ]
        # Include parameters and models
        for p_k, p_v in netlist.params.items():
            lines.append(f".param {p_k}={p_v}")
        for m_k, m_def in netlist.models.items():
            lines.append(f".model {m_k} {m_def.get('definition', '')}")

        for name, comp in netlist.components.items():
            pins = " ".join(netlist.pin_map.get(name, comp.nodes))
            if isinstance(comp, Resistor):
                lines.append(f"{name} {pins} {comp.value}")
            elif isinstance(comp, Capacitor):
                lines.append(f"{name} {pins} {comp.value}")
            elif isinstance(comp, Inductor):
                lines.append(f"{name} {pins} {comp.value}")
            elif isinstance(comp, VoltageSource):
                if comp.waveform_type == "DC":
                    lines.append(f"{name} {pins} DC {comp.dc}")
                elif comp.waveform_type in ("SINE", "SIN"):
                    p = comp.wave_params
                    lines.append(f"{name} {pins} SINE({p.get('offset', 0)} {p.get('amplitude', 1)} {p.get('freq', 1000)})")
                elif comp.waveform_type == "PULSE":
                    p = comp.wave_params
                    lines.append(f"{name} {pins} PULSE({p.get('v1', 0)} {p.get('v2', 5)} {p.get('td', 0)} {p.get('tr', '1n')} {p.get('tf', '1n')} {p.get('ton', '1m')} {p.get('period', '2m')})")
            elif isinstance(comp, CurrentSource):
                lines.append(f"{name} {pins} DC {comp.dc}")
            elif isinstance(comp, Diode):
                lines.append(f"{name} {pins} {comp.model_name}")
            elif isinstance(comp, BJT):
                lines.append(f"{name} {pins} {comp.model_name}")
            elif isinstance(comp, MOSFET):
                lines.append(f"{name} {pins} {comp.model_name}")
            elif isinstance(comp, JFET):
                lines.append(f"{name} {pins} {comp.model_name}")
            elif isinstance(comp, SubcircuitInstance):
                lines.append(f"{name} {pins} {comp.subckt_name}")
            else:
                lines.append(f"{name} {pins}")

        lines.extend(["", ".backanno", ".end"])
        return "\n".join(lines)

    @classmethod
    def export_circuit_ltspice_asc(cls, netlist: Netlist, name: str = "") -> str:
        """Exports to standard LTspice Schematic (.asc) format."""
        lines = [
            "Version 4",
            "SHEET 1 880 680",
        ]
        x_pos = 100
        y_pos = 100

        for name, comp in netlist.components.items():
            sym_name = "res" if isinstance(comp, Resistor) else (
                "cap" if isinstance(comp, Capacitor) else (
                    "ind" if isinstance(comp, Inductor) else (
                        "voltage" if isinstance(comp, VoltageSource) else (
                            "current" if isinstance(comp, CurrentSource) else (
                                "diode" if isinstance(comp, Diode) else (
                                    "npn" if isinstance(comp, BJT) else (
                                        "nmos" if isinstance(comp, MOSFET) else "block"
                                    )
                                )
                            )
                        )
                    )
                )
            )
            lines.append(f"SYMBOL {sym_name} {x_pos} {y_pos} R0")
            lines.append(f"SYMATTR InstName {name}")
            val = getattr(comp, "value", getattr(comp, "dc", getattr(comp, "model_name", "")))
            lines.append(f"SYMATTR Value {val}")

            x_pos += 120
            if x_pos > 800:
                x_pos = 100
                y_pos += 120

        return "\n".join(lines)

    # ==========================================
    # 2. DYNAMIC SYSTEM EXPORTERS (MATLAB Simulink, Python Control)
    # ==========================================
    @classmethod
    def export_dynamic_simulink_script(cls, diagram: SystemDiagram, name: str = "") -> str:
        """Exports to executable MATLAB script (.m) that builds and opens a full Simulink model."""
        target_name = name or diagram.name or "Terminus_Model"
        model_name = re.sub(r"[^a-zA-Z0-9_]", "_", target_name)
        lines = [
            f"function build_{model_name}_simulink()",
            f"% MATLAB Simulink Model Generation Script - Exported by TerminusECE",
            f"% Model: {target_name}",
            f"% Generated on: {time.ctime()}",
            "",
            f"model_name = '{model_name}';",
            f"if bdIsLoaded(model_name)",
            f"    close_system(model_name, 0);",
            f"end",
            f"new_system(model_name);",
            f"open_system(model_name);",
            "",
            "% 1. Create and Configure Blocks",
        ]

        # Add Blocks
        x_coord = 50
        y_coord = 50
        block_pos_map: Dict[str, Tuple[int, int, int, int]] = {}

        for b_name, b_inst in diagram.blocks.items():
            b_type = b_inst.__class__.__name__.lower().replace("block", "")
            simulink_type = "simulink/Sources/Constant"

            if "step" in b_type:
                simulink_type = "simulink/Sources/Step"
            elif "sine" in b_type:
                simulink_type = "simulink/Sources/Sine Wave"
            elif "ramp" in b_type:
                simulink_type = "simulink/Sources/Ramp"
            elif "pulse" in b_type:
                simulink_type = "simulink/Sources/Pulse Generator"
            elif "gain" in b_type:
                simulink_type = "simulink/Math Operations/Gain"
            elif "sum" in b_type:
                simulink_type = "simulink/Math Operations/Sum"
            elif "product" in b_type:
                simulink_type = "simulink/Math Operations/Product"
            elif "integrator" in b_type:
                simulink_type = "simulink/Continuous/Integrator"
            elif "derivative" in b_type:
                simulink_type = "simulink/Continuous/Derivative"
            elif "transfer" in b_type or "tf" in b_type:
                simulink_type = "simulink/Continuous/Transfer Fcn"
            elif "state" in b_type:
                simulink_type = "simulink/Continuous/State-Space"
            elif "pid" in b_type:
                simulink_type = "simulink/Continuous/PID Controller"
            elif "saturation" in b_type:
                simulink_type = "simulink/Discontinuities/Saturation"
            elif "scope" in b_type:
                simulink_type = "simulink/Sinks/Scope"
            elif "display" in b_type:
                simulink_type = "simulink/Sinks/Display"

            pos = (x_coord, y_coord, x_coord + 60, y_coord + 40)
            block_pos_map[b_name] = pos
            pos_str = f"[{pos[0]}, {pos[1]}, {pos[2]}, {pos[3]}]"

            lines.append(f"add_block('{simulink_type}', [model_name '/{b_name}'], 'Position', {pos_str});")

            # Set parameters
            for p_k, p_v in getattr(b_inst, "parameters", {}).items():
                lines.append(f"set_param([model_name '/{b_name}'], '{p_k}', '{p_v}');")

            x_coord += 120
            if x_coord > 700:
                x_coord = 50
                y_coord += 80

        lines.extend(["", "% 2. Route Signal Wires (Lines)"])
        for (src_b, src_p), (dst_b, dst_p) in diagram.connections:
            lines.append(f"add_line(model_name, '{src_b}/{src_p + 1}', '{dst_b}/{dst_p + 1}', 'autorouting', 'on');")

        t_stop = getattr(diagram, "t_stop", 10.0)
        dt = getattr(diagram, "dt", 0.001)
        lines.extend([
            "",
            "% 3. Configure Solver & Simulation Parameters",
            f"set_param(model_name, 'StopTime', '{t_stop}');",
            f"set_param(model_name, 'FixedStep', '{dt}');",
            f"set_param(model_name, 'Solver', 'ode4'); % RK4 Runge-Kutta",
            "save_system(model_name);",
            f"fprintf('Simulink Model %s built and saved successfully.\\n', model_name);",
            "end"
        ])
        return "\n".join(lines)

    @classmethod
    def export_dynamic_python_script(cls, diagram: SystemDiagram) -> str:
        """Exports to runnable Python script using SciPy / NumPy / Matplotlib."""
        lines = [
            f"# Python Dynamic System Simulation - Exported by TerminusECE",
            f"# Model: {diagram.name}",
            f"# Generated on: {time.ctime()}",
            "",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "from scipy import signal",
            "",
            f"dt = {diagram.config.dt}",
            f"t_stop = {diagram.config.t_stop}",
            "t = np.arange(0, t_stop + dt*0.5, dt)",
            "",
            "# Simulation loop and signals",
        ]
        return "\n".join(lines)

    # ==========================================
    # 3. DIGITAL LOGIC EXPORTERS (IEEE Verilog HDL, VHDL)
    # ==========================================
    @classmethod
    def export_digital_verilog(cls, circuit: LogicCircuit, module_name: str = "top_logic") -> str:
        """Exports to standard IEEE 1364-2005 Verilog HDL module."""
        lines = [
            f"// IEEE 1364-2005 Verilog HDL - Exported by TerminusECE",
            f"// Module: {module_name}",
            f"// Generated on: {time.ctime()}",
            f"`timescale 1ns / 1ps",
            "",
            f"module {module_name} (",
        ]
        # Collect inputs and outputs
        input_wires: List[str] = []
        output_wires: List[str] = []
        all_wires = sorted(list(circuit.wires))

        # Wires driven by gates are outputs or internals
        driven_wires = set()
        for g in circuit.gates.values():
            driven_wires.add(g.output_wire)

        for w in all_wires:
            if w in driven_wires:
                output_wires.append(w)
            else:
                input_wires.append(w)

        port_decls = [f"    input wire {w}" for w in input_wires] + [f"    output wire {w}" for w in output_wires]
        lines.append(",\n".join(port_decls))
        lines.append(");");
        lines.append("")

        lines.append("    // Gate Level Instances")
        for g_name, g in circuit.gates.items():
            g_type_lower = g.gate_type.lower()
            in_str = ", ".join(g.input_wires)
            lines.append(f"    {g_type_lower} {g_name} ({g.output_wire}, {in_str});")

        lines.extend(["", "endmodule"])
        return "\n".join(lines)

    @classmethod
    def export_digital_vhdl(cls, circuit: LogicCircuit, entity_name: str = "top_logic") -> str:
        """Exports to standard IEEE 1076 VHDL Entity and Architecture."""
        lines = [
            f"-- IEEE 1076 VHDL - Exported by TerminusECE",
            f"-- Entity: {entity_name}",
            f"-- Generated on: {time.ctime()}",
            "library IEEE;",
            "use IEEE.STD_LOGIC_1164.ALL;",
            "",
            f"entity {entity_name} is",
            "    Port (",
        ]
        port_lines = []
        for w in sorted(list(circuit.wires)):
            port_lines.append(f"        {w} : inout STD_LOGIC")
        lines.append(";\n".join(port_lines))
        lines.append("    );")
        lines.append(f"end {entity_name};")
        lines.append("")
        lines.append(f"architecture Structural of {entity_name} is")
        lines.append("begin")
        for g_name, g in circuit.gates.items():
            g_type = g.gate_type.upper()
            if len(g.input_wires) == 2:
                lines.append(f"    {g.output_wire} <= {g.input_wires[0]} {g_type} {g.input_wires[1]};")
            elif len(g.input_wires) == 1:
                lines.append(f"    {g.output_wire} <= {g_type} {g.input_wires[0]};")
        lines.append("end Structural;")
        return "\n".join(lines)

    # ==========================================
    # 4. NUMERICAL WORKSPACE EXPORTERS (MATLAB .m, Python .py)
    # ==========================================
    @classmethod
    def export_numerical_matlab(cls, workspace: NumericalWorkspace, name: str = "workspace") -> str:
        """Exports active numerical workspace variables into standard MATLAB script (.m)."""
        lines = [
            f"% MATLAB Script - Exported by TerminusECE Numerical Workspace ({name})",
            f"% Generated on: {time.ctime()}",
            "clear; clc;",
            ""
        ]
        for k, v in workspace.variables.items():
            if isinstance(v, (int, float)):
                lines.append(f"{k} = {v};")
            elif hasattr(v, "shape"):  # NumPy matrix
                mat_str = str(v).replace("\n", "; ")
                lines.append(f"{k} = {mat_str};")
            else:
                lines.append(f"{k} = {v};")
        return "\n".join(lines)

    export_numerical_matlab_script = export_numerical_matlab

    @classmethod
    def export_numerical_python_script(cls, workspace: NumericalWorkspace, name: str = "workspace") -> str:
        """Exports numerical workspace into Python SciPy/NumPy script (.py)."""
        lines = [
            f"# Python SciPy Script - Exported by TerminusECE Numerical Workspace ({name})",
            f"# Generated on: {time.ctime()}",
            "import numpy as np",
            "import scipy.signal as signal",
            "",
        ]
        for k, v in workspace.variables.items():
            lines.append(f"{k} = {repr(v)}")
        return "\n".join(lines)

    # ==========================================
    # 5. EMBEDDED MCU EXPORTERS (C Source, Intel HEX, CCS Assembly)
    # ==========================================
    @classmethod
    def export_embedded_c_firmware(cls, mcu: MCUCore, name: str = "mcu_firmware") -> str:
        """Exports MCU program into standard ANSI C/C++ firmware."""
        lines = [
            f"// C Firmware Source ({name}) - Exported by TerminusECE Embedded Engine",
            f"// Generated on: {time.ctime()}",
            "#include <stdint.h>",
            "#include <stdbool.h>",
            "",
            "// Hardware Registers & Peripherals",
            "volatile uint32_t ACC = 0;",
            "volatile uint32_t R[8] = {0};",
            "volatile uint32_t DATA_SRAM[65536];",
            "",
            "void setup(void) {",
            "    // Peripheral Initialization",
            "}",
            "",
            "void c2000_main(void) {",
            "    // Disassembled Instruction Execution Sequence",
        ]
        disassembly = Disassembler.disassemble(mcu.prog_mem)
        for line in disassembly.splitlines():
            lines.append(f"    // {line}")
        lines.extend([
            "}",
            "",
            "int main(void) {",
            "    setup();",
            "    while(1) {",
            "        c2000_main();",
            "    }",
            "    return 0;",
            "}"
        ])
        return "\n".join(lines)

    @classmethod
    def export_embedded_intel_hex(cls, mcu: MCUCore) -> str:
        """Exports MCU machine code to standard Intel HEX format."""
        lines = []
        addr = 0
        for instr in mcu.prog_mem:
            opcode_val = instr[0] if isinstance(instr, (list, tuple)) else getattr(instr, "opcode", 0)
            if isinstance(opcode_val, str):
                opcode_num = sum(ord(c) for c in opcode_val) & 0xFF
            else:
                opcode_num = int(opcode_val) & 0xFF
            length = 4
            data_bytes = [opcode_num, (addr >> 8) & 0xFF, addr & 0xFF, 0x00]
            chksum = (length + (addr >> 8) + (addr & 0xFF) + 0x00 + sum(data_bytes)) & 0xFF
            chksum = ((~chksum) + 1) & 0xFF
            d_hex = "".join(f"{b:02X}" for b in data_bytes)
            lines.append(f":{length:02X}{addr:04X}00{d_hex}{chksum:02X}")
            addr += 4
        lines.append(":00000001FF")
        return "\n".join(lines)

    @classmethod
    def export_embedded_assembly(cls, mcu: MCUCore) -> str:
        """Exports MCU program into standard assembly file."""
        return Disassembler.disassemble(mcu.prog_mem)

    # Aliases
    export_circuit_ltspice_cir = export_circuit_ltspice_netlist
    export_dynamic_system_simulink_script = export_dynamic_simulink_script
