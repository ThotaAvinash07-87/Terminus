"""Main Textual application and unified command router for TerminusECE."""

from __future__ import annotations
import os
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Input, RichLog, Static

from CORE.common_math import parse_eng_unit, format_eng_unit, Waveform, SignalMetrics, split_smart_statements
from CORE.ascii_canvas import AsciiCanvas, AsciiPlotter, AsciiBodePlotter, SchematicVisualizer
from CORE.ipc_router import IPCRouter, IPCClient, TerminalSessionInfo, ipc_router_instance, ipc_client_instance
from CORE.storage_manager import StorageManager

from Engines.Circuit.components import (
    Component, Resistor, Capacitor, Inductor, CoupledInductors,
    VoltageSource, CurrentSource, Diode, BJT, MOSFET, JFET,
    VoltageControlledSwitch, CurrentControlledSwitch, VCVS, VCCS, CCVS, CCCS,
    BehavioralSource, OpAmpModel, SubcircuitDefinition, SubcircuitInstance
)
from Engines.Circuit.netlist_parser import Netlist, CircuitParser
from Engines.Circuit.mna_solver import MNASolver, SimulationResult
from Engines.Circuit.catalog import circuit_catalog, CircuitComponentCatalog, CircuitComponentSpec
from Engines.Circuit.diagnostics import CircuitDiagnosticChecker, CircuitDiagnosticLevel, CircuitDiagnosticReport

from Engines.Numerical.parser import NumericalWorkspace, NumericalASTParser
from Engines.Numerical.transforms import TransferFunction, DiscreteTransferFunction

from Engines.Dynamic_System.blocks import (
    Block, IntegratorBlock, DerivativeBlock, TransferFunctionBlock, StateSpaceBlock,
    TransportDelayBlock, SaturationBlock, RateLimiterBlock, DeadZoneBlock, BacklashBlock,
    RelayBlock, CoulombViscousFrictionBlock, QuantizerBlock, GainBlock, SumBlock,
    ProductBlock, MathFunctionBlock, LookupTable1DBlock, SwitchBlock,
    ZeroOrderHoldBlock, UnitDelayBlock, PIDBlock, ConstantBlock,
    StepSourceBlock, RampSourceBlock, SineSourceBlock, PulseGeneratorBlock,
    BandLimitedWhiteNoiseBlock, ScopeSinkBlock, DisplaySinkBlock
)
from Engines.Dynamic_System.scheduler import SystemDiagram
from Engines.Dynamic_System.ode_solver import DynamicSystemSimulator
from Engines.Dynamic_System.catalog import DynamicBlockCatalog, parse_parameter_value
from Engines.Dynamic_System.diagnostics import ModelDiagnosticChecker, DiagnosticReport, DiagnosticSeverity

from Engines.Digital_Logic.gates import LogicValue
from Engines.Digital_Logic.hdl_parser import LogicCircuit, HDLParser
from Engines.Digital_Logic.event_sim import EventSimulator, DigitalWaveformTracer

from Engines.Embedded.mcu_core import MCUCore
from Engines.Embedded.toolchain import Assembler, Disassembler


def split_smart_args(text: str) -> List[str]:
    """Splits command string by whitespace while preserving bracketed [1, 2, 3] and quoted groups."""
    tokens = []
    current = []
    in_bracket = 0
    in_quote = False
    quote_char = ''
    
    for ch in text:
        if ch in ('"', "'"):
            if not in_quote:
                in_quote = True
                quote_char = ch
            elif quote_char == ch:
                in_quote = False
            current.append(ch)
        elif ch == '[':
            in_bracket += 1
            current.append(ch)
        elif ch == ']':
            if in_bracket > 0:
                in_bracket -= 1
            current.append(ch)
        elif ch.isspace() and in_bracket == 0 and not in_quote:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        tokens.append("".join(current))
    return tokens


class TerminusEngineBridge:
    """Decoupled computational backend bridge that holds all engine states and executes commands."""

    def __init__(self):
        # Current active mode: 'CIRCUIT', 'NUMERICAL', 'DYNAMIC', 'DIGITAL', 'EMBEDDED', 'UNIFIED'
        self.mode = "CIRCUIT"

        # Engines
        self.circuit_netlist = Netlist()
        self.circuit_solver = MNASolver(self.circuit_netlist)
        self.last_circuit_sim: Optional[SimulationResult] = None

        self.numerical_workspace = NumericalWorkspace()
        self.numerical_parser = NumericalASTParser(self.numerical_workspace)

        self.dynamic_diagram = SystemDiagram()
        self.dynamic_simulator = DynamicSystemSimulator(self.dynamic_diagram)
        self.last_dynamic_sim: Optional[Dict[str, Waveform]] = None
        self.last_diagnostic_report: Optional[DiagnosticReport] = None

        self.logic_circuit = LogicCircuit()
        self.last_logic_traces: Optional[Dict[str, List[Tuple[float, LogicValue]]]] = None

        self.mcu = MCUCore()

        # Structured Project Storage (User's Documents/Terminus Files)
        self.storage = StorageManager.get_instance()

        # IPC
        self.ipc_client = IPCClient()
        self.ipc_router: Optional[IPCRouter] = None

        # Global Signal / Variable bus for piping
        self.global_store: Dict[str, Any] = {}

    def switch_mode(self, mode_str: str) -> str:
        m = mode_str.lower().strip()
        if m in ("circuit", "ltspice", "spice"):
            self.mode = "CIRCUIT"
        elif m in ("numerical", "matlab", "matrix", "math"):
            self.mode = "NUMERICAL"
        elif m in ("dynamic", "simulink", "control", "systems"):
            self.mode = "DYNAMIC"
        elif m in ("digital", "xilinx", "logic", "verilog", "hdl"):
            self.mode = "DIGITAL"
        elif m in ("embedded", "mcu", "c2000", "dsp"):
            self.mode = "EMBEDDED"
        elif m in ("unified", "workbench", "all"):
            self.mode = "UNIFIED"
        else:
            raise ValueError(f"Unknown mode '{mode_str}'. Valid modes: circuit, numerical, dynamic, digital, embedded, unified")
        return self.mode

    def execute_command(self, raw_command: str) -> str:
        """Executes a text command line and returns formatted output string."""
        line = raw_command.strip()
        if not line or line.startswith("#"):
            return ""

        # Semicolon-separated batch commands
        sub_cmds = split_smart_statements(line, ";")
        if len(sub_cmds) > 1:
            outputs = []
            for sc in sub_cmds:
                out = self.execute_command(sc)
                if out:
                    outputs.append(out)
            return "\n".join(outputs)

        # Pipe commands
        if "|" in line and not line.lower().startswith("connect") and not line.lower().startswith("truth"):
            pipe_parts = [p.strip() for p in line.split("|")]
            last_out = ""
            for p in pipe_parts:
                last_out = self.execute_command(p)
            return last_out

        tokens = line.split()
        first = tokens[0].lower()

        # Global commands
        if first == "help":
            return self._cmd_help()

        if first == "mode":
            if len(tokens) < 2:
                return f"Current mode: {self.mode}. Options: circuit, numerical, dynamic, digital, embedded, unified"
            new_mode = self.switch_mode(tokens[1])
            return f"Context switched to: [bold magenta]{new_mode}[/bold magenta]"

        if first == "file":
            return self._handle_file_command(line, split_smart_args(line))

        if first == "export":
            return self._cmd_export(tokens[1:])

        if first == "session":
            return self._handle_session_command(line, tokens)

        if first == "ipc":
            return self._cmd_ipc(tokens[1:])

        # Route to mode-specific handler
        if self.mode == "CIRCUIT":
            return self._handle_circuit(line, tokens)
        elif self.mode == "NUMERICAL":
            return self._handle_numerical(line, tokens)
        elif self.mode == "DYNAMIC":
            return self._handle_dynamic(line, tokens)
        elif self.mode == "DIGITAL":
            return self._handle_digital(line, tokens)
        elif self.mode == "EMBEDDED":
            return self._handle_embedded(line, tokens)
        else:
            return self._handle_unified(line, tokens)

    def _cmd_help(self) -> str:
        help_texts = [
            f"[bold cyan]=== TerminusECE Commands ({self.mode} Mode) ===[/bold cyan]",
            "Global File & Workspace Commands (Auto-saved to Documents/Terminus Files/):",
            "  file new [name]              - Create fresh model/workspace",
            "  file save [filename]         - Save current model/workspace into mode storage",
            "  file saveas <filename>       - Save current model/workspace under a new name",
            "  file open <filename>         - Open/load file from mode storage",
            "  file list / file dir         - List all saved files in current mode's storage folder",
            "  file rename <old> <new>      - Rename a file in mode storage",
            "  file delete <filename>       - Delete a file from mode storage",
            "  file close                   - Close and reset current model/workspace",
            "  file path / file storage     - Show Documents/Terminus Files folder paths",
            "  mode <subsystem>             - Switch mode (circuit, numerical, dynamic, digital, embedded)",
            "  session list / ls            - Discover and list active terminal sessions on LAN (up to 500+)",
            "  session server start [port]  - Start multi-terminal async IPC network server",
            "  session connect <ip:port>    - Connect this session to another terminal / computer",
            "  session exec <id> <cmd>      - Remotely execute command on terminal Session #<id>",
            "  session link <sig> <id>.<sig>- Bridge and live-stream signals across terminal sessions",
            "  session broadcast <msg>      - Broadcast message/variable across all terminals",
            "  export [name] [--format=csv] - Export last simulation results to Exports/ folder",
        ]
        if self.mode == "CIRCUIT":
            help_texts.extend([
                "\nCircuit Commands (LTspice-grade Workflow):",
                "  library [category]           - Browse LTspice component library (Passives, Diodes, Transistors, ICs...)",
                "  library search <query>       - Search models by name/tag (e.g. library search 1N4148, library search MOSFET)",
                "  inspect <comp_or_model>      - View detailed specs, SPICE syntax, pinouts (e.g. inspect IRF540N, inspect R1)",
                "  check / diagnose             - Run Design Rule Check (DRC) for floating nodes, shorted sources, missing ground",
                "  add <Name> <Val/Model>       - Add component (e.g. add V1 10V ac=1, add M1 IRF540N, add D1 1N4148)",
                "  connect <p1> | <p2> | <net>  - Connect pins/nets (e.g. connect V1.p | R1.a, connect R1.b | C1.a | out)",
                "  tune / set <Name>.<param> <v>- Tune component parameter (e.g. set R1.value 2.2k)",
                "  probe <net>                  - Probe waveform metrics, peak-to-peak, RMS and ASCII plot",
                "  run .op                      - Run DC Operating Point with Newton-Raphson & Gmin stepping",
                "  run .dc V1 0 10 0.1          - Run DC Parameter Sweep & transfer curves",
                "  run .ac dec 10 1Hz 100kHz    - Run AC Frequency Sweep & ASCII Bode Plot",
                "  run .tran 1u 10m             - Run Transient Simulation with nonlinear companion models",
                "  run .four 1kHz V(out) 10m    - Run Fourier Analysis and calculate Total Harmonic Distortion (THD %)",
                "  export spice [name]          - Export netlist to standard SPICE .cir file",
                "  list / clear                 - Show or reset netlist",
            ])
        elif self.mode == "NUMERICAL":
            help_texts.extend([
                "\nNumerical Commands (MATLAB-like):",
                "  A = [1 2; 3 4]               - Matrix definition",
                "  inv(A), det(A), eig(A)       - Linear algebra",
                "  H = tf([1], [1, 2, 1])       - Continuous Transfer Function H(s)",
                "  bode(H)                      - ASCII Bode plot",
                "  step(H)                      - Step response",
                "  whos / clear                 - Variable introspection",
            ])
        elif self.mode == "DYNAMIC":
            help_texts.extend([
                "\nDynamic Systems & Simulink Workflow Commands:",
                "  library [category]           - Browse 17 classified block libraries (Continuous, Discrete, Math, Sources...)",
                "  library search <query>       - Search blocks by keyword or alias",
                "  library info <type>          - View detailed block port and parameter specifications",
                "  add <type> <name> [params]   - Add block (e.g. add step Step1, add tf Plant num=[1] den=[1,2,1], add pid PID1 kp=2)",
                "  remove <name>                - Delete block and attached wires",
                "  connect <B1.out> <B2.in>     - Route signal wire between blocks (e.g. connect Step1.0 Plant.0)",
                "  disconnect <B1.out> <B2.in>  - Remove wire connection",
                "  inspect <name>               - View block properties, tuned parameters, ports, and state values",
                "  tune <name> <param>=<val>    - Tune block parameter (e.g. tune PID1 kp=3.5 ki=1.2, set Sat1.upper_limit=10)",
                "  check / diagnose             - Run Model Advisor diagnostics (checks dangling ports, algebra loops, rates)",
                "  model solver <rk4|euler|ode45> - Set ODE integration solver algorithm",
                "  model dt <step_size>         - Set simulation integration step size",
                "  model stop <t_stop>          - Set simulation stop time",
                "  sim [t_stop] [dt] [solver]   - Run high-accuracy ODE simulation with pre-flight check",
                "  step [dt]                    - Single-step simulation execution",
                "  diagram / show               - Render ASCII visual block diagram",
                "  scope <name>                 - View ASCII plot of specific scope output",
                "  clear                        - Clear diagram",
            ])
        elif self.mode == "DIGITAL":
            help_texts.extend([
                "\nDigital Logic Commands (Xilinx-like):",
                "  wire a, b, c, clk, q         - Declare wires",
                "  gate G1 AND a b -> out       - Add logic gate",
                "  dff D1 clk=clk d=a q=q       - Add D-Flip-Flop",
                "  clock CLK period=10ns        - Add clock source",
                "  sim <max_time_ns>            - Run discrete event simulation & timing diagram",
                "  truth <boolean_expression>   - Generate truth table",
            ])
        elif self.mode == "EMBEDDED":
            help_texts.extend([
                "\nEmbedded Commands (C2000 MCU):",
                "  load <asm_code_or_file>      - Assemble and load program",
                "  step                         - Single-step instruction cycle",
                "  run [max_cycles]             - Run execution",
                "  dump                         - Dump registers, flags, and memory",
                "  pwm <period> <duty>          - Configure ePWM peripheral",
            ])
        return "\n".join(help_texts)

    def _handle_file_command(self, line: str, tokens: List[str]) -> str:
        if len(tokens) < 2:
            return self._cmd_file_list()
        sub = tokens[1].lower()

        if sub == "new":
            name = tokens[2] if len(tokens) > 2 else "Untitled"
            if self.mode == "DYNAMIC":
                self.dynamic_diagram.clear()
                self.dynamic_diagram.name = name
                self.last_dynamic_sim = None
                self.last_diagnostic_report = None
            elif self.mode == "CIRCUIT":
                self.circuit_netlist.clear()
                self.last_circuit_sim = None
            elif self.mode == "NUMERICAL":
                self.numerical_workspace.variables.clear()
            elif self.mode == "DIGITAL":
                self.logic_circuit = LogicCircuit()
                self.last_logic_traces = None
            elif self.mode == "EMBEDDED":
                self.mcu.reset()
            return f"[green]Created new {self.mode} model/workspace:[/green] [bold]{name}[/bold]"

        elif sub == "save":
            name = tokens[2] if len(tokens) > 2 else ""
            return self._save_current_workspace(name)

        elif sub in ("saveas", "save_as"):
            if len(tokens) < 3:
                raise ValueError("Usage: file saveas <new_filename>")
            name = tokens[2]
            return self._save_current_workspace(name, is_save_as=True)

        elif sub in ("open", "load"):
            if len(tokens) < 3:
                raise ValueError("Usage: file open <filename>")
            name = tokens[2]
            return self._load_into_workspace(name)

        elif sub in ("list", "ls", "dir"):
            return self._cmd_file_list()

        elif sub in ("rename", "mv"):
            if len(tokens) < 4:
                raise ValueError("Usage: file rename <old_name> <new_name>")
            old_n = tokens[2]
            new_n = tokens[3]
            new_path = self.storage.rename_file(self.mode, old_n, new_n)
            return f"[green]Renamed file to:[/green] [bold]{new_path.name}[/bold] in {new_path.parent}"

        elif sub in ("delete", "remove", "rm"):
            if len(tokens) < 3:
                raise ValueError("Usage: file delete <filename>")
            fname = tokens[2]
            ok = self.storage.delete_file(self.mode, fname)
            if ok:
                return f"[green]Deleted file:[/green] [bold]{fname}[/bold] from {self.storage.get_mode_dir(self.mode)}"
            raise FileNotFoundError(f"File '{fname}' not found in {self.storage.get_mode_dir(self.mode)}")

        elif sub == "close":
            return self._handle_file_command("file new Untitled", ["file", "new", "Untitled"])

        elif sub in ("path", "paths", "where", "storage"):
            return self._cmd_file_storage_summary()

        elif sub == "info":
            if self.mode == "DYNAMIC":
                return self._cmd_dynamic_model_info()
            return self._cmd_file_list()

        else:
            raise ValueError(f"Unknown file command 'file {sub}'. Options: new, save, saveas, open, list, rename, delete, close, path")

    def _save_current_workspace(self, name: str = "", is_save_as: bool = False) -> str:
        if self.mode == "DYNAMIC":
            if not name:
                name = self.dynamic_diagram.name or "Model"
            if is_save_as:
                self.dynamic_diagram.name = Path(name).stem
            data = self.dynamic_diagram.to_dict()
            saved_path = self.storage.save_file("DYNAMIC", name, data)
            return f"[green]Saved dynamic model to:[/green] [bold]{saved_path}[/bold] ({len(self.dynamic_diagram.blocks)} blocks, {len(self.dynamic_diagram.connections)} wires)"

        elif self.mode == "CIRCUIT":
            name = name or self.circuit_netlist.name or "circuit"
            spice_content = self.circuit_netlist.export_spice()
            saved_path = self.storage.save_file("CIRCUIT", name, spice_content)
            return f"[green]Saved SPICE circuit netlist to:[/green] [bold]{saved_path}[/bold] ({len(self.circuit_netlist.components)} components)"

        elif self.mode == "NUMERICAL":
            name = name or "workspace"
            lines = [f"# Terminus Numerical Workspace - {time.ctime()}"]
            for k, v in self.numerical_workspace.variables.items():
                lines.append(f"{k} = {v}")
            saved_path = self.storage.save_file("NUMERICAL", name, "\n".join(lines))
            return f"[green]Saved numerical workspace to:[/green] [bold]{saved_path}[/bold] ({len(self.numerical_workspace.variables)} variables)"

        elif self.mode == "DIGITAL":
            name = name or "logic_circuit"
            lines = [f"// Terminus Digital Logic - {time.ctime()}"]
            for g in self.logic_circuit.gates.values():
                lines.append(str(g))
            saved_path = self.storage.save_file("DIGITAL", name, "\n".join(lines))
            return f"[green]Saved digital circuit to:[/green] [bold]{saved_path}[/bold]"

        elif self.mode == "EMBEDDED":
            name = name or "mcu_program"
            content = Disassembler.disassemble(self.mcu.prog_mem)
            saved_path = self.storage.save_file("EMBEDDED", name, content)
            return f"[green]Saved embedded program to:[/green] [bold]{saved_path}[/bold]"

        return "Workspace saved."

    def _load_into_workspace(self, name: str) -> str:
        if self.mode == "DYNAMIC":
            path, data = self.storage.load_file("DYNAMIC", name)
            if isinstance(data, dict):
                self.dynamic_diagram.from_dict(data)
            else:
                self.dynamic_diagram.load_from_file(str(path))
            return f"[green]Loaded dynamic model:[/green] [bold]{self.dynamic_diagram.name}[/bold] from {path} ({len(self.dynamic_diagram.blocks)} blocks, {len(self.dynamic_diagram.connections)} wires)"

        elif self.mode == "CIRCUIT":
            path, data = self.storage.load_file("CIRCUIT", name)
            parsed_netlist, directives = CircuitParser.parse_spice_netlist(str(data))
            parsed_netlist.name = Path(path).stem
            self.circuit_netlist = parsed_netlist
            self.circuit_solver = MNASolver(self.circuit_netlist)
            return f"[green]Loaded circuit netlist from:[/green] [bold]{path}[/bold] ({len(self.circuit_netlist.components)} components, {len(directives)} directives)"

        elif self.mode == "NUMERICAL":
            path, data = self.storage.load_file("NUMERICAL", name)
            for l in str(data).splitlines():
                if l.strip() and not l.startswith("#"):
                    self.numerical_parser.execute(l.strip())
            return f"[green]Loaded numerical workspace from:[/green] [bold]{path}[/bold]"

        elif self.mode == "DIGITAL":
            path, data = self.storage.load_file("DIGITAL", name)
            return f"[green]Loaded digital logic from:[/green] [bold]{path}[/bold]"

        elif self.mode == "EMBEDDED":
            path, data = self.storage.load_file("EMBEDDED", name)
            count = self.mcu.load_program(str(data))
            return f"[green]Loaded embedded program from:[/green] [bold]{path}[/bold] ({count} words)"

        return "File loaded."

    def _cmd_file_list(self) -> str:
        files = self.storage.list_files(self.mode)
        mode_dir = self.storage.get_mode_dir(self.mode)
        lines = [
            f"[bold cyan]Files in {self.mode} Storage ({mode_dir}):[/bold cyan]"
        ]
        if not files:
            lines.append("  [dim](No files saved yet. Use 'file save <name>' to save current workspace)[/dim]")
        else:
            for f in files:
                fname = f.get("name", "file")
                sz_str = f.get("size_formatted", "")
                mtime = f.get("modified", "")
                lines.append(f"  * [bold white]{fname:30s}[/bold white]  [dim]{sz_str:>10s}  |  {mtime}[/dim]")
        return "\n".join(lines)

    def _cmd_file_storage_summary(self) -> str:
        paths = self.storage.get_all_paths()
        lines = [
            "[bold cyan]=== Terminus Central Storage System ===[/bold cyan]",
            f"Root Base Directory : [bold green]{paths['ROOT']}[/bold green]",
            f"Circuit Storage     : {paths['CIRCUIT']}",
            f"Numerical Storage   : {paths['NUMERICAL']}",
            f"Dynamic Storage     : {paths['DYNAMIC']}",
            f"Digital Storage     : {paths['DIGITAL']}",
            f"Embedded Storage    : {paths['EMBEDDED']}",
            f"Exports Storage     : {paths['EXPORTS']}",
        ]
        return "\n".join(lines)

    def _cmd_export(self, args: List[str]) -> str:
        out_name = "export_simulation"
        fmt = "csv"
        for a in args:
            if a.startswith("--format="):
                fmt = a.split("=")[1].lower()
            elif not a.startswith("-"):
                out_name = a

        content = ""
        if self.mode == "DYNAMIC" and self.last_dynamic_sim:
            lines = [f"# TerminusECE Export - Dynamic System Simulation ({self.dynamic_diagram.name})"]
            for name, wf in self.last_dynamic_sim.items():
                lines.append(f"\n--- Scope: {name} ---")
                lines.append(wf.to_csv())
            content = "\n".join(lines)
        elif self.last_circuit_sim:
            lines = [f"# TerminusECE Export - {self.last_circuit_sim.sim_type}"]
            for name, wf in self.last_circuit_sim.waveforms.items():
                lines.append(f"\n--- {name} ---")
                lines.append(wf.to_csv())
            content = "\n".join(lines)
        else:
            return "No simulation data available to export."

        saved_path = self.storage.save_file("EXPORTS", f"{out_name}.{fmt}", content)
        return f"[green]Exported simulation data to:[/green] [bold]{saved_path}[/bold]"

    def _cmd_ipc(self, args: List[str]) -> str:
        return self._handle_session_command("session " + " ".join(args), ["session"] + args)

    # --- High-Capacity Multi-Terminal Session Network Handlers ---
    def _handle_session_command(self, line: str, tokens: List[str]) -> str:
        if len(tokens) < 2:
            return (
                "[bold cyan]=== Multi-Terminal Session Networking (500+ Terminals) ===[/bold cyan]\n"
                "  session list / ls            - Discover and list all active terminals on LAN/network\n"
                "  session info                 - View current session number, assigned IP:port, and linked signals\n"
                "  session server start [port]  - Start high-capacity async IPC router server (defaults to 8765)\n"
                "  session server stop          - Stop local IPC router server\n"
                "  session connect <ip:port>    - Connect this terminal session to a network router\n"
                "  session exec <id> <cmd>      - Remotely execute command on terminal Session #<id>\n"
                "  session link <sig> <id>.<sig>- Bridge and live-stream signal from Session #<id>\n"
                "  session push <sig> <val>     - Push signal value across network to all linked sessions\n"
                "  session broadcast <msg>      - Broadcast message/variable across all terminals\n"
                "  session limits / network     - Display architectural network capacity & scalability details\n"
            )

        sub = tokens[1].lower()

        if sub in ("list", "ls"):
            sessions = self.ipc_client.list_sessions_sync()
            if not sessions:
                if not self.ipc_client.is_connected:
                    return (
                        "[yellow]This terminal is not connected to an IPC Router.[/yellow]\n"
                        "To host a multi-terminal network on this computer: [bold]session server start[/bold]\n"
                        "To connect to an existing router: [bold]session connect 127.0.0.1:8765[/bold]"
                    )
                return "[yellow]No other active sessions detected on network router.[/yellow]"

            lines = [
                f"[bold cyan]=== Active Networked Terminal Sessions ({len(sessions)} Active / Max 500+) ===[/bold cyan]",
                f"{'Session #':<11}{'Host / IP':<22}{'Mode':<14}{'Active File':<22}{'Uptime':<10}"
            ]
            lines.append("-" * 79)
            for s in sessions:
                sid = s.get("session_id", "?")
                is_self = " (Current)" if sid == self.ipc_client.session_id else ""
                host_ip = f"{s.get('hostname', 'host')}:{s.get('port', 0)}"
                mode = s.get("mode", "Circuit")
                active = s.get("active_file", "untitled")
                uptime = f"{time.time() - s.get('connected_at', time.time()):.0f}s"
                lines.append(f"#{sid:<10}{host_ip:<22}{mode:<14}{active:<22}{uptime:<10}{is_self}")
            return "\n".join(lines)

        elif sub == "info":
            if not self.ipc_client.is_connected:
                return "[yellow]Not currently connected to any IPC Network Router. (Run 'session server start' or 'session connect <ip:port>')[/yellow]"
            lines = [
                "[bold cyan]=== Current Terminal Session Network Profile ===[/bold cyan]",
                f"Assigned Session Number: [bold green]Session #{self.ipc_client.session_id}[/bold green]",
                f"Client UUID            : {self.ipc_client.client_uuid}",
                f"Connected Router       : {self.ipc_client.host}:{self.ipc_client.port}",
                f"Current Operating Mode : {self.mode}",
                f"Linked Input Signals   : {len(self.ipc_client.linked_signals)} active",
            ]
            for k, v in self.ipc_client.linked_signals.items():
                lines.append(f"  * [bold]{k}[/bold] = {v}")
            return "\n".join(lines)

        elif sub == "server":
            if len(tokens) >= 3 and tokens[2].lower() == "start":
                port = int(tokens[3]) if len(tokens) > 3 else 8765
                ipc_router_instance.port = port
                ipc_router_instance.start()
                # Auto-connect local client to router
                self.ipc_client = ipc_client_instance
                self.ipc_client.set_command_executor(self.execute_command)
                self.ipc_client.connect(mode=self.mode, active_file=self.circuit_netlist.name)
                return f"[green]High-Capacity IPC Router Server started on 0.0.0.0:{port}[/green] (Ready for up to 500+ connections). Current terminal registered as [bold]Session #{self.ipc_client.session_id}[/bold]."
            elif len(tokens) >= 3 and tokens[2].lower() == "stop":
                ipc_router_instance.stop()
                return "[green]IPC Router Server stopped.[/green]"
            return "Usage: session server start [port] | session server stop"

        elif sub == "connect":
            if len(tokens) < 3:
                raise ValueError("Usage: session connect <ip:port> [requested_session_id]")
            endpoint = tokens[2]
            req_id = int(tokens[3]) if len(tokens) > 3 else None
            host = endpoint.split(":")[0] if ":" in endpoint else endpoint
            port = int(endpoint.split(":")[1]) if ":" in endpoint else 8765

            self.ipc_client = IPCClient(host=host, port=port)
            self.ipc_client.set_command_executor(self.execute_command)
            ok = self.ipc_client.connect(mode=self.mode, active_file=self.circuit_netlist.name, requested_session_id=req_id)
            if ok:
                return f"[green]Successfully connected to {host}:{port}[/green] as [bold]Session #{self.ipc_client.session_id}[/bold] (UUID: {self.ipc_client.client_uuid[:8]})."
            raise ConnectionError(f"Could not connect to IPC Router at {host}:{port}. Is the router server running?")

        elif sub == "exec":
            if len(tokens) < 4:
                raise ValueError("Usage: session exec <target_session_id> <command...>")
            target_sid = int(tokens[2])
            rem_cmd = " ".join(tokens[3:])
            res = self.ipc_client.remote_exec_sync(target_sid, rem_cmd)
            if res.get("status") == "dispatched":
                return f"[green]Dispatched command to Session #{target_sid}:[/green] '{rem_cmd}'"
            return f"[red]Remote execution response:[/red] {res}"

        elif sub == "link":
            if len(tokens) < 4:
                raise ValueError("Usage: session link <local_signal> <target_session_id>.<target_signal>")
            local_sig = tokens[2]
            target_spec = tokens[3]
            if "." not in target_spec:
                raise ValueError("Target must be in format <session_id>.<signal_name>, e.g. 'session link V(out) 2.Vin'")
            tgt_sid_str, tgt_sig = target_spec.split(".", 1)
            tgt_sid = int(tgt_sid_str)
            res = self.ipc_client.link_signal_sync(local_sig, tgt_sid, tgt_sig)
            return f"[green]Signal Link Established:[/green] Current Session #{self.ipc_client.session_id}:{local_sig} -> Session #{tgt_sid}:{tgt_sig}"

        elif sub == "push":
            if len(tokens) < 4:
                raise ValueError("Usage: session push <signal_name> <value>")
            sig_name = tokens[2]
            val = parse_eng_unit(tokens[3])
            self.ipc_client.push_signal_sync(sig_name, val)
            return f"[green]Pushed signal:[/green] {sig_name} = {val}"

        elif sub == "broadcast":
            if len(tokens) < 3:
                raise ValueError("Usage: session broadcast <message>")
            bmsg = " ".join(tokens[2:])
            self.ipc_client.broadcast_sync(bmsg)
            return f"[green]Broadcast sent to all terminals:[/green] '{bmsg}'"

        elif sub in ("limits", "network", "scaling"):
            return (
                "[bold cyan]=== Network Capacity & 500+ Terminal Session Architecture ===[/bold cyan]\n"
                "1. [bold white]Socket Descriptor Capacity[/bold white]:\n"
                "   Uses Python asyncio Proactor (IOCP on Windows / Epoll on Linux) supporting 10,000+ non-blocking sockets.\n"
                "   No select() 64/1024 FD limit bottlenecks.\n"
                "2. [bold white]Low-Latency Transmission[/bold white]:\n"
                "   TCP_NODELAY enabled to eliminate 40ms Nagle buffering delay for real-time signal streaming.\n"
                "3. [bold white]Scalability Bottlenecks & Mitigations[/bold white]:\n"
                "   * OS Max Open Files: Configurable via ulimit -n / registry max user sockets.\n"
                "   * Serialization Throughput: Compact UTF-8 JSON Lines streaming (<150 bytes per signal packet).\n"
                "   * Multi-Computer LAN Discovery: Subnet broadcast TCP router binding on 0.0.0.0 allows any laptop on the Wi-Fi/Ethernet to join by IP:Port.\n"
                "   * Collision-Proof Session Numbers: Atomic thread-safe session allocator ensures unique Session IDs #1..#500+ across distinct devices.\n"
            )

        raise ValueError(f"Unknown session command 'session {sub}'. Run 'session' for help.")

    # --- Circuit Handlers ---
    def _handle_circuit(self, line: str, tokens: List[str]) -> str:
        first = tokens[0].lower()

        # 1. Circuit Library Browser
        if first == "library":
            return self._cmd_circuit_library(tokens[1:])

        # 2. Design Rule Checking & Diagnostics
        if first in ("check", "diagnose", "drc"):
            return self._cmd_circuit_diagnostics()

        # 3. Component & Model Inspector
        if first in ("inspect", "details", "info"):
            if len(tokens) < 2:
                raise ValueError("Usage: inspect <component_name_or_model> (e.g. inspect R1, inspect 1N4148, inspect IRF540N)")
            return self._cmd_circuit_inspect(tokens[1])

        # 4. Probe / Measure
        if first in ("probe", "measure", "scope"):
            if len(tokens) < 2:
                raise ValueError("Usage: probe <node_or_trace> (e.g. probe out, probe V(out))")
            return self._cmd_circuit_probe(tokens[1])

        # 5. Export SPICE
        if first == "export" and len(tokens) >= 2 and tokens[1].lower() == "spice":
            name = tokens[2] if len(tokens) > 2 else self.circuit_netlist.name
            spice_code = self.circuit_netlist.export_spice()
            path = self.storage.save_file("CIRCUIT", f"{name}.cir", spice_code)
            return f"[green]Exported SPICE netlist to:[/green] [bold]{path}[/bold]"

        # Standard Netlist operations
        if first in ("add", "connect", "remove", "delete", "set", "list", "show", "clear"):
            res = CircuitParser.parse_command(self.circuit_netlist, line)
            self.circuit_solver = MNASolver(self.circuit_netlist)
            t = res.get("type")
            if t == "add":
                return f"[green]Added component:[/green] {res.get('name')} -> {res.get('component')}"
            elif t == "connect":
                return f"[green]Connected:[/green] {' | '.join(res.get('endpoints', []))}"
            elif t == "list":
                comps = res.get("components", [])
                if not comps:
                    return "Netlist is empty."
                return f"[bold cyan]Active Netlist: {self.circuit_netlist.name} ({len(comps)} components)[/bold cyan]\n" + "\n".join(f"  {c}" for c in comps)
            elif t == "clear":
                return "Circuit netlist cleared."
            elif t == "set":
                return f"Updated {res.get('target')} = {res.get('value')}"
            return str(res)

        elif first == "run":
            sim_spec = line[4:].strip()
            return self._run_circuit_simulation(sim_spec)

        if line.startswith("."):
            return self._run_circuit_simulation(line)

        raise ValueError(f"Unknown circuit command '{line}'. Type 'help' for options.")

    def _cmd_circuit_library(self, args: List[str]) -> str:
        """Browse classified LTspice component library."""
        if not args:
            cats = circuit_catalog.get_all_categories()
            lines = [
                "[bold cyan]=== LTspice Classified Component Library ===[/bold cyan]",
                "Available Categories:"
            ]
            for c in cats:
                comps = circuit_catalog.list_components_by_category(c)
                lines.append(f"  * [bold yellow]{c:<22}[/bold yellow] ({len(comps)} components) - e.g. {', '.join(spec.name for spec in comps[:3])}")
            lines.append("\nCommands:")
            lines.append("  library <category>           - List all models in a category")
            lines.append("  library search <query>       - Search by keyword or model name")
            lines.append("  inspect <model_or_comp>      - Inspect pins, SPICE syntax, and parameters")
            return "\n".join(lines)

        sub = args[0].lower()
        if sub in ("search", "find"):
            if len(args) < 2:
                raise ValueError("Usage: library search <query>")
            q = " ".join(args[1:])
            results = circuit_catalog.search_components(q)
            if not results:
                return f"[yellow]No components found matching '{q}'.[/yellow]"
            lines = [f"[bold cyan]Search Results for '{q}' ({len(results)} found):[/bold cyan]"]
            for s in results:
                lines.append(f"  * [bold green]{s.name:<16}[/bold green] [[dim]{s.category}[/dim]] - {s.description}")
            return "\n".join(lines)

        category = args[0]
        comps = circuit_catalog.list_components_by_category(category)
        if not comps:
            # Try searching as model name
            spec = circuit_catalog.get_spec(category)
            if spec:
                return self._cmd_circuit_inspect(spec.name)
            return f"[yellow]Category '{category}' not found. Run 'library' to see all categories.[/yellow]"

        lines = [f"[bold cyan]=== Library: {category} ({len(comps)} items) ===[/bold cyan]"]
        for s in comps:
            pins_str = ", ".join(s.pins)
            lines.append(f"  * [bold green]{s.name:<15}[/bold green] Pins: [{pins_str}]")
            lines.append(f"    [dim]{s.description}[/dim]")
            if s.example_usage:
                lines.append(f"    [cyan]Example:[/cyan] {s.example_usage}")
        return "\n".join(lines)

    def _cmd_circuit_diagnostics(self) -> str:
        """Runs Design Rule Checking (DRC) on the active netlist."""
        checker = CircuitDiagnosticChecker(self.circuit_netlist)
        report = checker.diagnose()

        lines = [
            f"[bold cyan]=== Circuit Design Rule & Topology Check ({self.circuit_netlist.name}) ===[/bold cyan]",
            f"Components: {report.total_components}  |  Nodes: {report.total_nodes}  |  Status: {'[bold green]PASSED[/bold green]' if report.is_valid else '[bold red]FAILED[/bold red]'}"
        ]

        if not report.issues:
            lines.append("\n[green]All Design Rule Checks passed! No floating nodes or singular matrix hazards found.[/green]")
        else:
            lines.append("\nDiagnostic Issues Detected:")
            for issue in report.issues:
                color = "red" if issue.level == CircuitDiagnosticLevel.ERROR else ("yellow" if issue.level == CircuitDiagnosticLevel.WARNING else "blue")
                lines.append(f"  [{color}][{issue.level.value}][/{color}] [bold]{issue.category}:[/bold] {issue.message}")
                if issue.suggested_fix:
                    lines.append(f"    [dim]Fix:[/dim] {issue.suggested_fix}")

        return "\n".join(lines)

    def _cmd_circuit_inspect(self, target_name: str) -> str:
        """Inspects an instance or catalog component."""
        uname = target_name.upper()
        # Check active netlist first
        if uname in self.circuit_netlist.components:
            comp = self.circuit_netlist.components[uname]
            pins = self.circuit_netlist.pin_map.get(uname, comp.nodes)
            lines = [
                f"[bold cyan]=== Circuit Component Instance: {uname} ===[/bold cyan]",
                f"Type       : {comp.__class__.__name__}",
                f"Connected  : {pins}",
            ]
            for attr in ("value", "dc", "ac_mag", "ac_phase", "waveform_type", "model_name", "expression", "gain", "beta_f", "vto"):
                if hasattr(comp, attr):
                    lines.append(f"{attr:<11}: {getattr(comp, attr)}")
            return "\n".join(lines)

        # Check catalog spec
        spec = circuit_catalog.get_spec(uname)
        if spec:
            lines = [
                f"[bold cyan]=== LTspice Component Spec: {spec.name} ===[/bold cyan]",
                f"Category   : {spec.category}",
                f"Description: {spec.description}",
                f"SPICE Type : {spec.spice_model_type}",
                f"Pins       : {', '.join(spec.pins)}",
                f"Defaults   : {spec.default_params}",
            ]
            if spec.model_statement:
                lines.append(f"SPICE Model: {spec.model_statement}")
            if spec.example_usage:
                lines.append(f"Example    : {spec.example_usage}")
            return "\n".join(lines)

        raise KeyError(f"Neither component instance nor library model '{target_name}' was found.")

    def _cmd_circuit_probe(self, trace_name: str) -> str:
        """Probes a specific node voltage or branch current from the last simulation."""
        if not self.last_circuit_sim:
            raise ValueError("No simulation results available. Run a simulation first (e.g. 'run .tran 1u 10m').")

        norm = trace_name if trace_name.startswith("V(") or trace_name.startswith("I(") else f"V({trace_name})"
        wf = self.last_circuit_sim.get_waveform(norm)
        if not wf:
            # Check operating point dictionary
            if norm in self.last_circuit_sim.op_results:
                return f"[bold cyan]Probe {norm}:[/bold cyan] [bold green]{self.last_circuit_sim.op_results[norm]:.6g} V[/bold green]"
            avail = list(self.last_circuit_sim.waveforms.keys()) + list(self.last_circuit_sim.op_results.keys())
            raise KeyError(f"Trace '{norm}' not found. Available traces: {', '.join(avail)}")

        plot_str = AsciiPlotter.plot(wf.x, wf.y, title=f"Probe: {norm}", x_label=self.last_circuit_sim.x_label, y_label=wf.y_unit)
        stats = (
            f"[bold cyan]Trace Metrics for {norm}:[/bold cyan]\n"
            f"  Max: {wf.max:.4g} {wf.y_unit}  |  Min: {wf.min:.4g} {wf.y_unit}  |  Peak-to-Peak: {wf.peak_to_peak:.4g} {wf.y_unit}  |  RMS: {wf.rms:.4g} {wf.y_unit}"
        )
        return f"{stats}\n\n{plot_str}"

    def _run_circuit_simulation(self, spec: str) -> str:
        parts = spec.split()
        if not parts:
            raise ValueError("No simulation specified. Example: 'run .ac dec 10 1Hz 100kHz'")

        self.circuit_solver = MNASolver(self.circuit_netlist)

        topo_diagram = SchematicVisualizer.render_circuit_topology(
            self.circuit_netlist.components,
            self.circuit_netlist.pin_map
        )

        sim_cmd = parts[0].lower()
        if sim_cmd == ".op":
            res = self.circuit_solver.solve_op()
            self.last_circuit_sim = res
            return f"{topo_diagram}\n\n{res.summary()}"

        elif sim_cmd == ".dc":
            if len(parts) < 5:
                raise ValueError("Usage: run .dc <Source> <Start> <Stop> <Step>")
            src = parts[1]
            start = parse_eng_unit(parts[2])
            stop = parse_eng_unit(parts[3])
            step = parse_eng_unit(parts[4])
            res = self.circuit_solver.solve_dc_sweep(src, start, stop, step)
            self.last_circuit_sim = res
            out_traces = [wf for k, wf in res.waveforms.items() if not k.startswith("I(")]
            plot_str = ""
            if out_traces:
                plot_str = AsciiPlotter.plot(out_traces[0].x, out_traces[0].y, title=f"DC Sweep: {out_traces[0].name}", x_label=f"{src} (V)")
            return f"{topo_diagram}\n\n{res.summary()}\n\n{plot_str}"

        elif sim_cmd == ".ac":
            sweep_type = "dec"
            pts = 10
            f_start = 1.0
            f_stop = 100e3

            if len(parts) >= 5:
                sweep_type = parts[1]
                pts = int(parts[2])
                f_start = parse_eng_unit(parts[3])
                f_stop = parse_eng_unit(parts[4])

            res = self.circuit_solver.solve_ac(sweep_type, pts, f_start, f_stop)
            self.last_circuit_sim = res

            out_wf = None
            for name, wf in res.waveforms.items():
                if name != "V(0)":
                    out_wf = wf

            bode_str = ""
            if out_wf:
                bode_str = AsciiBodePlotter.plot_bode(
                    out_wf.x,
                    out_wf.magnitude_db,
                    out_wf.phase_deg,
                    title=f"Bode Plot: {out_wf.name}"
                )

            return f"{topo_diagram}\n\n{res.summary()}\n\n{bode_str}"

        elif sim_cmd == ".tran":
            if len(parts) < 3:
                raise ValueError("Usage: run .tran <t_step> <t_stop> [t_start]")
            dt = parse_eng_unit(parts[1])
            t_stop = parse_eng_unit(parts[2])
            t_start = parse_eng_unit(parts[3]) if len(parts) > 3 else 0.0

            res = self.circuit_solver.solve_tran(dt, t_stop, t_start)
            self.last_circuit_sim = res

            out_wf = None
            for name, wf in res.waveforms.items():
                if name != "V(0)":
                    out_wf = wf

            plot_str = ""
            if out_wf:
                plot_str = AsciiPlotter.plot(out_wf.x, out_wf.y, title=f"Transient Response: {out_wf.name}", x_label="Time (s)", y_label="Voltage (V)")

            return f"{topo_diagram}\n\n{res.summary()}\n\n{plot_str}"

        elif sim_cmd == ".four":
            if len(parts) < 4:
                raise ValueError("Usage: run .four <fundamental_freq> <trace_name> <t_stop>")
            freq_val = parse_eng_unit(parts[1])
            tname = parts[2]
            tstop_val = parse_eng_unit(parts[3])

            res = self.circuit_solver.solve_four(freq_val, tname, tstop_val)
            self.last_circuit_sim = res
            return f"{topo_diagram}\n\n{res.summary()}"

        raise ValueError(f"Unknown simulation command '{sim_cmd}'. Supported: .op, .dc, .ac, .tran, .four")

    # --- Numerical Handlers ---
    def _handle_numerical(self, line: str, tokens: List[str]) -> str:
        if line.startswith("bode(") and line.endswith(")"):
            var_name = line[5:-1].strip()
            obj = self.numerical_workspace.variables.get(var_name)
            if isinstance(obj, TransferFunction):
                return obj.render_bode_ascii()
            raise ValueError(f"Variable '{var_name}' is not a TransferFunction.")

        if line.startswith("step(") and line.endswith(")"):
            var_name = line[5:-1].strip()
            obj = self.numerical_workspace.variables.get(var_name)
            if isinstance(obj, TransferFunction):
                wf = obj.step_response()
                return AsciiPlotter.plot(wf.x, wf.y, title=f"Step Response: {var_name}", x_label="Time (s)", y_label="Amplitude")
            raise ValueError(f"Variable '{var_name}' is not a TransferFunction.")

        val = self.numerical_parser.execute(line)
        if val is None:
            return ""
        if isinstance(val, (np.ndarray, list)):
            arr = np.asarray(val)
            if arr.ndim <= 2:
                return str(arr)
        return str(val)

    # --- Dynamic System Handlers ---
    def _handle_dynamic(self, line: str, tokens: List[str]) -> str:
        smart_tokens = split_smart_args(line)
        if smart_tokens:
            tokens = smart_tokens
        first = tokens[0].lower()

        # 1. Clear / Reset
        if first == "clear":
            self.dynamic_diagram.clear()
            self.last_dynamic_sim = None
            self.last_diagnostic_report = None
            return "[green]Dynamic system diagram cleared.[/green]"

        # 2. File Workflow (handled by global _handle_file_command)
        if first == "file":
            return self._handle_file_command(line, tokens)

        # 3. Model Configuration: model solver, model dt, model stop, model tol, model info
        if first == "model":
            if len(tokens) < 2:
                return self._cmd_dynamic_model_info()
            setting = tokens[1].lower()
            if setting in ("info", "status"):
                return self._cmd_dynamic_model_info()
            if len(tokens) >= 3:
                val = tokens[2]
                if setting in ("solver", "algorithm"):
                    self.dynamic_diagram.solver = val.lower()
                    return f"[green]Model solver set to:[/green] [bold]{self.dynamic_diagram.solver.upper()}[/bold]"
                elif setting in ("dt", "step", "step_size"):
                    self.dynamic_diagram.dt = parse_eng_unit(val)
                    return f"[green]Model integration step size dt set to:[/green] [bold]{self.dynamic_diagram.dt} s[/bold]"
                elif setting in ("stop", "t_stop", "stop_time"):
                    self.dynamic_diagram.t_stop = parse_eng_unit(val)
                    return f"[green]Model simulation stop time set to:[/green] [bold]{self.dynamic_diagram.t_stop} s[/bold]"
                elif setting in ("start", "t_start", "start_time"):
                    self.dynamic_diagram.t_start = parse_eng_unit(val)
                    return f"[green]Model simulation start time set to:[/green] [bold]{self.dynamic_diagram.t_start} s[/bold]"
                elif setting in ("tol", "tolerance"):
                    self.dynamic_diagram.tolerance = float(val)
                    return f"[green]Model adaptive tolerance set to:[/green] [bold]{self.dynamic_diagram.tolerance}[/bold]"
            # Check key=val format e.g. model solver=rk4 dt=0.001
            for kv in tokens[1:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    k = k.lower().strip()
                    if k == "solver":
                        self.dynamic_diagram.solver = v.lower().strip()
                    elif k in ("dt", "step"):
                        self.dynamic_diagram.dt = parse_eng_unit(v)
                    elif k in ("stop", "t_stop"):
                        self.dynamic_diagram.t_stop = parse_eng_unit(v)
                    elif k in ("start", "t_start"):
                        self.dynamic_diagram.t_start = parse_eng_unit(v)
                    elif k in ("tol", "tolerance"):
                        self.dynamic_diagram.tolerance = float(v)
            return self._cmd_dynamic_model_info()

        # 4. Library Browser: library, library <category>, library search <query>, library info <type>
        if first in ("library", "blocks", "catalog"):
            return self._cmd_dynamic_library(tokens[1:])

        # 5. Add Block: add <type> <name> [params...]
        if first == "add":
            return self._cmd_dynamic_add_block(line, tokens)

        # 6. Remove Block: remove <name> / delete <name>
        if first in ("remove", "delete"):
            if len(tokens) < 2:
                raise ValueError("Usage: remove <block_name>")
            bname = tokens[1].upper()
            ok = self.dynamic_diagram.remove_block(bname)
            if ok:
                return f"[green]Removed block:[/green] {bname}"
            raise KeyError(f"Block '{bname}' does not exist.")

        # 7. Connect: connect <src.port> <dst.port>
        if first == "connect":
            if len(tokens) < 3:
                raise ValueError("Usage: connect <BlockA.port> <BlockB.port>")
            self.dynamic_diagram.connect(tokens[1], tokens[2])
            return f"[green]Connected signals:[/green] {tokens[1]} -> {tokens[2]}"

        # 8. Disconnect: disconnect <src.port> <dst.port>
        if first == "disconnect":
            if len(tokens) < 3:
                raise ValueError("Usage: disconnect <BlockA.port> <BlockB.port>")
            ok = self.dynamic_diagram.disconnect(tokens[1], tokens[2])
            if ok:
                return f"[green]Disconnected wire:[/green] {tokens[1]} -x- {tokens[2]}"
            raise ValueError(f"No active wire found connecting {tokens[1]} to {tokens[2]}")

        # 9. Inspect / Properties: inspect <name>, params <name>
        if first in ("inspect", "params", "props", "properties"):
            if len(tokens) < 2:
                raise ValueError("Usage: inspect <block_name>")
            return self._cmd_dynamic_inspect(tokens[1])

        # 10. Parameter Tuning: tune <name> <param>=<val>, set <name>.<param>=<val>
        if first in ("tune", "set"):
            return self._cmd_dynamic_tune(line, tokens)

        # 11. Model Diagnostics & Verification: check, diagnose, validate
        if first in ("check", "diagnose", "validate", "advisor"):
            report = ModelDiagnosticChecker.run_diagnostics(self.dynamic_diagram)
            self.last_diagnostic_report = report
            return report.summary()

        # 12. Diagram Visualization: diagram, show, list
        if first in ("diagram", "show"):
            return SchematicVisualizer.render_dynamic_block_diagram(
                self.dynamic_diagram.blocks,
                self.dynamic_diagram.connections
            )

        if first == "list":
            return self._cmd_dynamic_list_blocks()

        # 13. Scope & Display Viewers: scope <name>, display <name>
        if first == "scope":
            sname = tokens[1].upper() if len(tokens) > 1 else ""
            if not self.last_dynamic_sim:
                raise ValueError("No simulation has been run yet. Use 'sim' first.")
            if sname and sname in self.last_dynamic_sim:
                wf = self.last_dynamic_sim[sname]
                return AsciiPlotter.plot(wf.x, wf.y, title=f"Scope: {sname}", x_label="Time (s)")
            # Show all scopes
            plots = [AsciiPlotter.plot(wf.x, wf.y, title=f"Scope: {k}", x_label="Time (s)") for k, wf in self.last_dynamic_sim.items()]
            return "\n\n".join(plots)

        if first == "display":
            if len(tokens) < 2:
                raise ValueError("Usage: display <block_name>")
            b = self.dynamic_diagram.get_block(tokens[1])
            if not b:
                raise KeyError(f"Block '{tokens[1]}' not found.")
            val = b.inputs[0] if b.inputs else getattr(b, "outputs", [0.0])[0]
            return f"Display [{b.name}]: [bold yellow]{val}[/bold yellow]"

        # 14. Step Execution: step [dt]
        if first == "step":
            dt = parse_eng_unit(tokens[1]) if len(tokens) > 1 else self.dynamic_diagram.dt
            readings = self.dynamic_simulator.step(dt=dt, solver=self.dynamic_diagram.solver)
            read_str = ", ".join(f"{k}={v:.4g}" for k, v in readings.items()) if readings else "OK"
            return f"Sim Step advanced to [bold cyan]t={self.dynamic_simulator.current_time:.4f}s[/bold cyan] ({read_str})"

        # 15. Reset Simulation: reset
        if first == "reset":
            self.dynamic_simulator.reset(self.dynamic_diagram.t_start)
            return f"[green]Simulation reset to t={self.dynamic_diagram.t_start}s.[/green]"

        # 16. Run / Simulate: sim [stop_time] [dt] [solver]
        if first in ("sim", "simulate", "run"):
            t_stop = parse_eng_unit(tokens[1]) if len(tokens) > 1 else self.dynamic_diagram.t_stop
            dt = parse_eng_unit(tokens[2]) if len(tokens) > 2 else self.dynamic_diagram.dt
            solver = tokens[3].lower() if len(tokens) > 3 else self.dynamic_diagram.solver

            # Pre-flight diagnostic verification
            report = ModelDiagnosticChecker.run_diagnostics(self.dynamic_diagram)
            self.last_diagnostic_report = report
            if not report.is_valid:
                return f"[bold red]Simulation Aborted: Model Advisor found {report.num_errors} critical error(s):[/bold red]\n\n{report.summary()}\n\n[dim]Fix issues or connect missing ports to proceed.[/dim]"

            res = self.dynamic_simulator.simulate(t_stop=t_stop, dt=dt, solver=solver)
            self.last_dynamic_sim = res

            block_diagram = SchematicVisualizer.render_dynamic_block_diagram(
                self.dynamic_diagram.blocks,
                self.dynamic_diagram.connections
            )

            plots = []
            for sname, wf in res.items():
                p = AsciiPlotter.plot(wf.x, wf.y, title=f"Scope: {sname}", x_label="Time (s)")
                plots.append(p)

            plots_str = "\n\n".join(plots) if plots else "[italic dim]No Scope sinks recorded in diagram. Add a Scope block to visualize waveforms.[/italic dim]"
            warn_header = f"[bold yellow]Note:[/bold yellow] Model simulated with {report.num_warnings} non-fatal warning(s).\n\n" if report.num_warnings > 0 else ""
            return f"{block_diagram}\n\n{warn_header}[bold green]Simulation Complete:[/bold green] {t_stop}s integrated with solver [bold cyan]{solver.upper()}[/bold cyan] (dt={dt}s).\n\n{plots_str}"

        raise ValueError(f"Unknown dynamic system command '{line}'. Type 'help' for options.")

    def _cmd_dynamic_model_info(self) -> str:
        d = self.dynamic_diagram
        cont_st = sum(b.num_states for b in d.blocks.values() if getattr(b, "sample_time", 0.0) == 0.0)
        disc_st = sum(b.num_states for b in d.blocks.values() if getattr(b, "sample_time", 0.0) > 0.0)
        lines = [
            f"[bold cyan]=== Dynamic System Model Settings: {d.name} ===[/bold cyan]",
            f"  Description      : {d.description}",
            f"  Solver Algorithm : [bold green]{d.solver.upper()}[/bold green] (Options: rk4, euler, heun, ode45, ode23)",
            f"  Step Size (dt)   : {d.dt} s",
            f"  Time Span        : {d.t_start} s to {d.t_stop} s",
            f"  Tolerance        : {d.tolerance}",
            f"  Total Blocks     : {len(d.blocks)}",
            f"  Total Wires      : {len(d.connections)}",
            f"  States           : {cont_st} Continuous, {disc_st} Discrete",
        ]
        return "\n".join(lines)

    def _cmd_dynamic_library(self, args: List[str]) -> str:
        if not args:
            # List all categories
            cats = DynamicBlockCatalog.list_categories()
            lines = ["[bold cyan]=== Simulink Classified Block Libraries ===[/bold cyan]"]
            lines.append("Type 'library <category_name>' to view blocks in a category, or 'library search <query>' to search.\n")
            for i, c in enumerate(cats, 1):
                b_count = len(DynamicBlockCatalog.get_blocks_by_category(c))
                lines.append(f"  {i:2d}. [bold magenta]{c}[/bold magenta] ({b_count} blocks)")
            return "\n".join(lines)

        sub = args[0].lower()
        if sub == "search":
            if len(args) < 2:
                raise ValueError("Usage: library search <keyword>")
            query = " ".join(args[1:])
            matches = DynamicBlockCatalog.search_blocks(query)
            if not matches:
                return f"No blocks matching query '{query}'."
            lines = [f"[bold cyan]=== Block Search Results for '{query}' ({len(matches)} matches) ===[/bold cyan]"]
            for m in matches:
                lines.append(f"  • [bold green]{m.type_name}[/bold green] ({m.display_name}) - [magenta]{m.category}[/magenta]: {m.description}")
            return "\n".join(lines)

        if sub in ("info", "spec"):
            if len(args) < 2:
                raise ValueError("Usage: library info <block_type>")
            btype = args[1]
            desc = DynamicBlockCatalog.get_descriptor(btype)
            if not desc:
                raise KeyError(f"Block type '{btype}' not found.")
            lines = [
                f"[bold cyan]=== Block Specification: {desc.display_name} ({desc.type_name}) ===[/bold cyan]",
                f"  Category    : [magenta]{desc.category}[/magenta]",
                f"  Class       : {desc.block_class.__name__}",
                f"  Description : {desc.description}",
                f"  Feedthrough : {desc.direct_feedthrough}",
                f"  Continuous  : {desc.has_continuous_states} | Discrete: {desc.has_discrete_states}",
                "\n  [bold]Input Ports:[/bold]",
            ]
            for p in desc.input_ports:
                lines.append(f"    - Port {p.port_index} ('{p.name}'): {p.data_type} ({p.description or 'signal input'})")
            if not desc.input_ports:
                lines.append("    (None - Signal Source)")

            lines.append("\n  [bold]Output Ports:[/bold]")
            for p in desc.output_ports:
                lines.append(f"    - Port {p.port_index} ('{p.name}'): {p.data_type} ({p.description or 'signal output'})")
            if not desc.output_ports:
                lines.append("    (None - Signal Sink)")

            lines.append("\n  [bold]Tunable Parameters:[/bold]")
            for pname, pdesc in desc.parameters.items():
                unit_str = f" [{pdesc.unit}]" if pdesc.unit else ""
                lines.append(f"    - [bold]{pname}[/bold]{unit_str} ({pdesc.param_type.__name__}): default={pdesc.default_value} - {pdesc.description}")
            if not desc.parameters:
                lines.append("    (No configurable parameters)")

            return "\n".join(lines)

        # List category
        cat_query = " ".join(args)
        blocks = DynamicBlockCatalog.get_blocks_by_category(cat_query)
        if not blocks:
            # Try search
            blocks = DynamicBlockCatalog.search_blocks(cat_query)
            if not blocks:
                return f"Category or block '{cat_query}' not found. Type 'library' to see all categories."

        lines = [f"[bold cyan]=== {cat_query.title()} Blocks ({len(blocks)} blocks) ===[/bold cyan]"]
        for b in blocks:
            in_c = len(b.input_ports)
            out_c = len(b.output_ports)
            lines.append(f"  • [bold green]{b.type_name:<20}[/bold green] (In:{in_c} Out:{out_c}) - {b.description}")
        return "\n".join(lines)

    def _cmd_dynamic_add_block(self, line: str, tokens: List[str]) -> str:
        if len(tokens) < 3:
            raise ValueError("Usage: add <type> <name> [param1=val1 param2=val2 ...]")
        btype = tokens[1].lower()
        bname = tokens[2].upper()

        params: Dict[str, Any] = {}

        # Check if remaining tokens are key=val format or positional
        remaining = tokens[3:]
        is_key_val = any("=" in t for t in remaining)

        if is_key_val:
            for t in remaining:
                if "=" in t:
                    k, v = t.split("=", 1)
                    params[k.strip()] = v.strip()
        else:
            # Parse transfer function brackets [num] [den] or legacy positional arguments
            brackets = re.findall(r'\[([^\]]+)\]', line)
            if btype in ("tf", "transfer_fcn", "transferfunction") and len(brackets) >= 2:
                params["num"] = [float(x) for x in re.split(r'[\s,]+', brackets[0].strip()) if x]
                params["den"] = [float(x) for x in re.split(r'[\s,]+', brackets[1].strip()) if x]
            elif btype in ("integrator", "int", "1/s"):
                if len(remaining) > 0: params["initial_condition"] = remaining[0]
                if len(remaining) > 1 and remaining[1] != "none": params["lower_limit"] = remaining[1]
                if len(remaining) > 2 and remaining[2] != "none": params["upper_limit"] = remaining[2]
            elif btype in ("derivative", "deriv", "s"):
                if len(remaining) > 0: params["tau"] = remaining[0]
            elif btype == "gain":
                if len(remaining) > 0: params["gain"] = remaining[0]
            elif btype == "sum":
                if len(remaining) > 0: params["signs"] = remaining[0]
            elif btype in ("product", "prod", "mult", "div"):
                if len(remaining) > 0: params["operations"] = remaining[0]
            elif btype in ("mathfunc", "math", "func"):
                if len(remaining) > 0: params["function"] = remaining[0]
            elif btype in ("saturation", "sat", "clamp"):
                if len(remaining) > 0: params["lower_limit"] = remaining[0]
                if len(remaining) > 1: params["upper_limit"] = remaining[1]
            elif btype in ("ratelimiter", "ratelimit", "slew"):
                if len(remaining) > 0: params["rising_slew_rate"] = remaining[0]
                if len(remaining) > 1: params["falling_slew_rate"] = remaining[1]
            elif btype in ("deadzone", "deadband"):
                if len(remaining) > 0: params["start_zone"] = remaining[0]
                if len(remaining) > 1: params["end_zone"] = remaining[1]
            elif btype in ("backlash", "hysteresis"):
                if len(remaining) > 0: params["deadband_width"] = remaining[0]
            elif btype in ("relay", "schmitt", "bangbang"):
                if len(remaining) > 0: params["switch_on_point"] = remaining[0]
                if len(remaining) > 1: params["switch_off_point"] = remaining[1]
                if len(remaining) > 2: params["output_on"] = remaining[2]
                if len(remaining) > 3: params["output_off"] = remaining[3]
            elif btype in ("friction", "fric"):
                if len(remaining) > 0: params["f_coulomb"] = remaining[0]
                if len(remaining) > 1: params["b_viscous"] = remaining[1]
                if len(remaining) > 2: params["f_static"] = remaining[2]
            elif btype in ("quantizer", "quant"):
                if len(remaining) > 0: params["quantization_interval"] = remaining[0]
            elif btype in ("delay", "transport_delay", "timedelay"):
                if len(remaining) > 0: params["delay_time"] = remaining[0]
            elif btype in ("zoh", "zero_order_hold"):
                if len(remaining) > 0: params["sample_time"] = remaining[0]
            elif btype in ("unitdelay", "unit_delay", "z^-1"):
                if len(remaining) > 0: params["sample_time"] = remaining[0]
            elif btype in ("switch", "mux2to1"):
                if len(remaining) > 0: params["threshold"] = remaining[0]
            elif btype in ("pid", "pid_controller"):
                if len(remaining) > 0: params["kp"] = remaining[0]
                if len(remaining) > 1: params["ki"] = remaining[1]
                if len(remaining) > 2: params["kd"] = remaining[2]
                if len(remaining) > 3: params["n_filter"] = remaining[3]
                if len(remaining) > 4 and remaining[4] != "none": params["lower_limit"] = remaining[4]
                if len(remaining) > 5 and remaining[5] != "none": params["upper_limit"] = remaining[5]
            elif btype in ("const", "constant"):
                if len(remaining) > 0: params["value"] = remaining[0]
            elif btype in ("step", "step_source"):
                if len(remaining) > 0: params["step_time"] = remaining[0]
                if len(remaining) > 1: params["amplitude"] = remaining[1]
            elif btype in ("ramp", "ramp_source", "slope"):
                if len(remaining) > 0: params["slope"] = remaining[0]
                if len(remaining) > 1: params["start_time"] = remaining[1]
            elif btype in ("sine", "sine_wave", "sin"):
                if len(remaining) > 0: params["freq"] = remaining[0]
                if len(remaining) > 1: params["amplitude"] = remaining[1]
            elif btype in ("pulse", "pulse_generator", "square"):
                if len(remaining) > 0: params["period"] = remaining[0]
                if len(remaining) > 1: params["duty_cycle"] = remaining[1]
            elif btype in ("noise", "white_noise", "whitenoise"):
                if len(remaining) > 0: params["noise_power"] = remaining[0]
                if len(remaining) > 1: params["sample_time"] = remaining[1]

        # Instantiate via Catalog with parameter type validation
        block = DynamicBlockCatalog.instantiate(btype, bname, **params)
        self.dynamic_diagram.add_block(block, block_type=btype, **params)

        desc = DynamicBlockCatalog.get_descriptor(btype)
        disp_name = desc.display_name if desc else btype
        return f"[green]Added Dynamic Block:[/green] [bold]{bname}[/bold] ({disp_name}) with parameters: {getattr(block, 'parameters', {})}"

    def _cmd_dynamic_inspect(self, bname: str) -> str:
        block = self.dynamic_diagram.get_block(bname)
        if not block:
            raise KeyError(f"Block '{bname}' does not exist.")

        bmeta = self.dynamic_diagram.block_metadata.get(bname.upper(), {})
        btype = bmeta.get("type", block.__class__.__name__)
        desc = DynamicBlockCatalog.get_descriptor(btype)
        disp_name = desc.display_name if desc else btype
        cat_name = desc.category if desc else "Dynamic"

        lines = [
            f"[bold cyan]=== Block Properties: {block.name.upper()} ({disp_name}) ===[/bold cyan]",
            f"  Category    : [magenta]{cat_name}[/magenta]",
            f"  Class Type  : {block.__class__.__name__}",
            f"  Feedthrough : {getattr(block, 'direct_feedthrough', True)}",
            f"  Sample Time : {getattr(block, 'sample_time', 0.0)} s (0 = Continuous)",
            f"  States      : {block.num_states} state(s) -> {block.states.tolist()}",
            "\n  [bold]Tuned Parameters:[/bold]",
        ]

        params = getattr(block, "parameters", {})
        if params:
            for k, v in params.items():
                p_desc = desc.parameters.get(k) if desc else None
                u_str = f" [{p_desc.unit}]" if (p_desc and p_desc.unit) else ""
                lines.append(f"    • [bold]{k}[/bold]{u_str} = [yellow]{v}[/yellow]")
        else:
            lines.append("    (None)")

        lines.append("\n  [bold]Port Status & Signal Wires:[/bold]")
        # Incoming
        incoming = [(src[0], src[1], dst[1]) for src, dst in self.dynamic_diagram.connections if dst[0] == block.name.upper()]
        inc_map = {dst_p: (src_b, src_p) for src_b, src_p, dst_p in incoming}
        for i in range(block.num_inputs):
            if i in inc_map:
                src_b, src_p = inc_map[i]
                lines.append(f"    - Inport {i}  : [green]Connected from {src_b}.{src_p}[/green] (val={block.inputs[i] if i < len(block.inputs) else 0.0})")
            else:
                lines.append(f"    - Inport {i}  : [bold red][UNCONNECTED][/bold red] (val=0.0)")

        # Outgoing
        for i in range(block.num_outputs):
            targets = [f"{dst[0]}.{dst[1]}" for src, dst in self.dynamic_diagram.connections if src[0] == block.name.upper() and src[1] == i]
            tar_str = ", ".join(targets) if targets else "[dim]Unconnected[/dim]"
            lines.append(f"    - Outport {i} : {tar_str} (val={block.outputs[i] if i < len(block.outputs) else 0.0})")

        return "\n".join(lines)

    def _cmd_dynamic_tune(self, line: str, tokens: List[str]) -> str:
        # tune <name> <param>=<val> ... OR set <name>.<param>=<val>
        if tokens[0].lower() == "set" and len(tokens) >= 2 and "." in tokens[1]:
            # set Block.param=val or set Block.param val
            ep = tokens[1]
            if "=" in ep:
                full_k, val = ep.split("=", 1)
                bname, param = full_k.split(".", 1)
            elif len(tokens) >= 3:
                bname, param = ep.split(".", 1)
                val = tokens[2]
            else:
                raise ValueError("Usage: set <BlockName.param>=<value>")
            new_val = self.dynamic_diagram.tune_parameter(bname, param, val)
            return f"[green]Tuned parameter:[/green] {bname.upper()}.{param} = [bold yellow]{new_val}[/bold yellow]"

        if len(tokens) < 3:
            raise ValueError("Usage: tune <block_name> <param1=val1 param2=val2 ...>")
        bname = tokens[1].upper()
        results = []
        for t in tokens[2:]:
            if "=" in t:
                pname, pval = t.split("=", 1)
                new_v = self.dynamic_diagram.tune_parameter(bname, pname, pval)
                results.append(f"{pname}={new_v}")
        if not results:
            raise ValueError("No parameters specified. Example: tune PID1 kp=3.0 ki=1.5")
        return f"[green]Tuned {bname}:[/green] " + ", ".join(results)

    def _cmd_dynamic_list_blocks(self) -> str:
        if not self.dynamic_diagram.blocks:
            return "Dynamic diagram is empty. Use 'add <type> <name>' to create blocks."
        lines = [f"[bold cyan]=== Model Blocks ({len(self.dynamic_diagram.blocks)} blocks) ===[/bold cyan]"]
        for bname, b in self.dynamic_diagram.blocks.items():
            meta = self.dynamic_diagram.block_metadata.get(bname, {})
            btype = meta.get("type", b.__class__.__name__)
            lines.append(f"  • [bold green]{bname:<16}[/bold green] ({btype}) - In:{b.num_inputs} Out:{b.num_outputs} States:{b.num_states}")

        lines.append(f"\n[bold cyan]=== Signal Connections ({len(self.dynamic_diagram.connections)} wires) ===[/bold cyan]")
        for src, dst in self.dynamic_diagram.connections:
            lines.append(f"  • {src[0]}.{src[1]} ---> {dst[0]}.{dst[1]}")
        return "\n".join(lines)

    # --- Digital Logic Handlers ---
    def _handle_digital(self, line: str, tokens: List[str]) -> str:
        first = tokens[0].lower()

        if first == "truth" or first == "truthtable":
            expr = line[len(first):].strip()
            return HDLParser.generate_truth_table(expr)

        if first in ("wire", "gate", "dff", "clock", "input", "output"):
            circuit = HDLParser.parse(line)
            self.logic_circuit.wires.update(circuit.wires)
            self.logic_circuit.gates.update(circuit.gates)
            self.logic_circuit.flip_flops.update(circuit.flip_flops)
            self.logic_circuit.clocks.update(circuit.clocks)
            return f"[green]Updated Digital Logic Circuit:[/green] {len(self.logic_circuit.gates)} gate(s), {len(self.logic_circuit.flip_flops)} FF(s)"

        if first in ("sim", "simulate", "run"):
            max_t = parse_eng_unit(tokens[1]) if len(tokens) > 1 else 100.0
            if max_t < 1e-3:
                max_t = max_t * 1e9
            sim = EventSimulator(self.logic_circuit)
            traces = sim.run(max_time_ns=max_t)
            self.last_logic_traces = traces
            return DigitalWaveformTracer.render_timing_diagram(traces, max_time_ns=max_t)

        raise ValueError(f"Unknown digital logic command '{line}'")

    # --- Embedded Handlers ---
    def _handle_embedded(self, line: str, tokens: List[str]) -> str:
        first = tokens[0].lower()

        if first == "load":
            asm_code = line[4:].strip()
            if os.path.exists(asm_code):
                with open(asm_code, "r", encoding="utf-8") as f:
                    asm_code = f.read()
            count = self.mcu.load_program(asm_code)
            return f"[green]Loaded MCU program:[/green] {count} instruction(s) assembled.\n\n" + Disassembler.disassemble(self.mcu.prog_mem, pc_highlight=0)

        if first == "step":
            ok = self.mcu.step()
            pc = self.mcu.regs.PC
            dis = Disassembler.disassemble(self.mcu.prog_mem, pc_highlight=pc)
            status = self.mcu.dump_state()
            return f"{status}\n\n{dis}"

        if first in ("run", "exec"):
            max_cyc = int(tokens[1]) if len(tokens) > 1 else 10000
            cycles = self.mcu.run(max_cycles=max_cyc)
            return f"Execution stopped after {cycles} cycles.\n\n" + self.mcu.dump_state()

        if first == "dump":
            return self.mcu.dump_state()

        if first == "reset":
            self.mcu.reset()
            return "MCU Reset complete."

        if first == "pwm":
            if len(tokens) < 3:
                raise ValueError("Usage: pwm <period_counts> <duty_counts>")
            prd = int(tokens[1])
            duty = int(tokens[2])
            self.mcu.peripherals.epwm.write(0x00, prd)
            self.mcu.peripherals.epwm.write(0x01, duty)
            wf = self.mcu.peripherals.epwm.generate_waveform(cycles=2000)
            plot_str = AsciiPlotter.plot(wf.x, wf.y, title=f"ePWM Output (Duty: {self.mcu.peripherals.epwm.duty_cycle*100:.1f}%)", x_label="Time (s)")
            return f"Configured ePWM1A: Period={prd}, Duty={duty}\n\n{plot_str}"

        raise ValueError(f"Unknown embedded command '{line}'")

    # --- Unified Handlers ---
    def _handle_unified(self, line: str, tokens: List[str]) -> str:
        first = tokens[0].lower()
        if first in ("add", "connect", "run", ".ac", ".tran", ".op", ".dc"):
            return self._handle_circuit(line, tokens)
        return self._handle_numerical(line, tokens)


class TerminusApp(App):
    """The unified Textual TUI for TerminusECE."""

    CSS = """
    #command_input {
        dock: bottom;
        margin: 0 1;
        border: heavy cyan;
    }
    #console {
        height: 100%;
        margin: 0 1;
        border: solid green;
    }
    #simulink_workspace {
        height: 100%;
    }
    .sidebar {
        width: 34;
        height: 100%;
    }
    #simulink_center {
        width: 1fr;
        height: 100%;
    }
    """

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("c", "switch_circuit", "Circuit Mode"),
        ("n", "switch_numerical", "Numerical Mode"),
        ("s", "switch_dynamic", "Simulink/Dynamic Mode"),
        ("l", "switch_digital", "Digital Logic"),
        ("e", "switch_embedded", "Embedded MCU"),
        ("f1", "show_help", "Help & Guide"),
        ("f2", "focus_library", "Library Browser"),
        ("f5", "run_sim", "Run Simulation"),
        ("f6", "step_sim", "Step Simulation"),
        ("f7", "diagnose_model", "Model Advisor Check"),
        ("q", "quit_app", "Quit Terminus"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bridge = TerminusEngineBridge()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="console", highlight=True, markup=True)
        yield Input(placeholder="Terminus [CIRCUIT] > Enter command (e.g. 'help', 'mode dynamic', 'add V1 10V')...", id="command_input")
        yield Footer()

    def on_ready(self) -> None:
        log = self.query_one(RichLog)
        log.write("[bold green]=== TerminusECE Unified EDA Workbench Initialized ===[/bold green]")
        log.write("Lightning-fast command-driven workspace for Electrical & Computer Engineering.")
        log.write("Current mode: [bold magenta]CIRCUIT (LTspice)[/bold magenta]. Type [bold cyan]'help'[/bold cyan] for commands.")
        log.write("Simulink Workflow: Type [bold cyan]'mode dynamic'[/bold cyan] or press [bold yellow]'s'[/bold yellow] to enter Dynamic Systems.")
        self.query_one(Input).focus()

    def action_quit_app(self) -> None:
        self.exit()

    def action_show_help(self) -> None:
        log = self.query_one(RichLog)
        log.write(self.bridge.execute_command("help"))

    def action_run_sim(self) -> None:
        log = self.query_one(RichLog)
        if self.bridge.mode == "DYNAMIC":
            log.write(f"[bold cyan]> sim[/bold cyan]")
            log.write(self.bridge.execute_command("sim"))
        elif self.bridge.mode == "CIRCUIT":
            log.write(f"[bold cyan]> run .tran 1u 10m[/bold cyan]")
            log.write(self.bridge.execute_command("run .tran 1u 10m"))
        elif self.bridge.mode == "DIGITAL":
            log.write(f"[bold cyan]> sim 100ns[/bold cyan]")
            log.write(self.bridge.execute_command("sim 100ns"))

    def action_step_sim(self) -> None:
        log = self.query_one(RichLog)
        if self.bridge.mode == "DYNAMIC":
            log.write(f"[bold cyan]> step[/bold cyan]")
            log.write(self.bridge.execute_command("step"))
        elif self.bridge.mode == "EMBEDDED":
            log.write(f"[bold cyan]> step[/bold cyan]")
            log.write(self.bridge.execute_command("step"))

    def action_diagnose_model(self) -> None:
        log = self.query_one(RichLog)
        if self.bridge.mode == "DYNAMIC":
            log.write(f"[bold cyan]> check[/bold cyan]")
            log.write(self.bridge.execute_command("check"))

    def action_focus_library(self) -> None:
        log = self.query_one(RichLog)
        if self.bridge.mode == "DYNAMIC":
            log.write(self.bridge.execute_command("library"))

    def action_switch_circuit(self) -> None:
        self._set_mode("CIRCUIT")

    def action_switch_numerical(self) -> None:
        self._set_mode("NUMERICAL")

    def action_switch_dynamic(self) -> None:
        self._set_mode("DYNAMIC")

    def action_switch_digital(self) -> None:
        self._set_mode("DIGITAL")

    def action_switch_embedded(self) -> None:
        self._set_mode("EMBEDDED")

    def _set_mode(self, mode: str) -> None:
        new_mode = self.bridge.switch_mode(mode)
        inp = self.query_one(Input)
        inp.placeholder = f"Terminus [{new_mode}] > Enter command..."
        log = self.query_one(RichLog)
        log.write(f"[bold magenta]Switched context to {new_mode}[/bold magenta]")
        if new_mode == "DYNAMIC":
            log.write("Simulink Dynamic Systems Active: Type [bold cyan]'library'[/bold cyan] to browse blocks, [bold cyan]'file new <name>'[/bold cyan], [bold cyan]'check'[/bold cyan] for diagnostics.")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        log = self.query_one(RichLog)
        cmd = event.value.strip()
        if not cmd:
            return

        log.write(f"[bold cyan]> {cmd}[/bold cyan]")
        try:
            out = self.bridge.execute_command(cmd)
            if out:
                log.write(out)
        except Exception as err:
            log.write(f"[bold red]Error:[/bold red] {err}")

        # Update input placeholder if mode changed
        self.query_one(Input).placeholder = f"Terminus [{self.bridge.mode}] > Enter command..."
        event.input.value = ""
