"""Main Textual application and unified command router for TerminusECE."""

from __future__ import annotations
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Input, RichLog, Static

from CORE.common_math import parse_eng_unit, format_eng_unit, Waveform, SignalMetrics, split_smart_statements
from CORE.ascii_canvas import AsciiCanvas, AsciiPlotter, AsciiBodePlotter, SchematicVisualizer
from CORE.ipc_router import (
    IPCRouter, IPCClient, TerminalSessionInfo, CrossModeSignalValidator,
    CrossModeValidationResult, ipc_router_instance, ipc_client_instance
)
from CORE.figure_renderer import TerminusFigure, TerminalImagePreviewer, TechnicalReportMetrics, active_figure
from CORE.storage_manager import StorageManager
from CORE.lib_importer import CustomModelImporter, MCUSpecification
from CORE.universal_exporter import UniversalStandardExporter

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
from Engines.Numerical.dsp_engine import DSPSignalEngine
from Engines.Numerical.matlab_manager import MatlabProjectManager, MatlabProjectFile

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

from Engines.Embedded import (
    BoardSpecification, BoardCatalog, board_catalog,
    USBPortManager, USBDeviceInfo,
    ProjectSketchManager, ProjectFile,
    HardwareFirmwareFlasher, HardwareFlashResult,
    RealSerialMonitor, RealAsciiSerialPlotter,
    VirtualDevice, LEDDevice, RGBLEDDevice, ButtonDevice,
    PotentiometerDevice, DHTSensorDevice, UltrasonicSensorDevice,
    ServoDevice, LCD16x2Device, OLEDSSD1306Device, DeviceManager,
    ArduinoPinState, ArduinoSerialStream, ArduinoRuntimeEngine,
    CompilationResult, SerialPortInfo, scan_com_ports,
    HardwareSerialBridge, AsciiSerialPlotter, FirmwareFlasher
)
from Engines.Embedded.mcu_core import MCUCore
from Engines.Embedded.toolchain import Assembler, Disassembler

from Engines.KiCad import (
    FootprintCatalog, FootprintSpec, PadSpec,
    KiCadSchematic, SchematicComponent, SchematicPin,
    KiCadPCB, PlacedComponent, CopperTrack, PCBVia, PCBPadLocation,
    GerberExporter, KiCadCalculator, KiCadProject
)



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
        self.matlab_proj = MatlabProjectManager("SignalWorkspace")
        self.figure = active_figure

        self.dynamic_diagram = SystemDiagram()
        self.dynamic_simulator = DynamicSystemSimulator(self.dynamic_diagram)
        self.last_dynamic_sim: Optional[Dict[str, Waveform]] = None
        self.last_diagnostic_report: Optional[DiagnosticReport] = None

        self.logic_circuit = LogicCircuit()
        self.last_logic_traces: Optional[Dict[str, List[Tuple[float, LogicValue]]]] = None

        self.mcu = MCUCore()
        self.arduino = ArduinoRuntimeEngine("UNO")
        self.target_board = board_catalog.get("UNO")
        self.sketch_proj = ProjectSketchManager("MyProject")
        self.active_com_port = USBPortManager.auto_detect_primary_port() or "COM3"
        self.real_flasher = HardwareFirmwareFlasher(self.target_board, self.active_com_port)
        self.real_serial = RealSerialMonitor(self.active_com_port, 115200)
        self.ascii_plotter = self.real_serial.plotter
        self.active_serial_bridge: Optional[HardwareSerialBridge] = None

        # Structured Project Storage (User's Documents/Terminus Files)
        self.storage = StorageManager.get_instance()

        # KiCad Schematics & PCB Design Engine
        self.kicad_proj = KiCadProject("MyPCBProject", storage=self.storage)

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
        elif m in ("embedded", "mcu", "c2000", "dsp", "arduino", "arduino_ide"):
            self.mode = "EMBEDDED"
        elif m in ("kicad", "pcb", "board", "eda", "schematic"):
            self.mode = "KICAD"
        elif m in ("unified", "workbench", "all"):
            self.mode = "UNIFIED"
        else:
            raise ValueError(f"Unknown mode '{mode_str}'. Valid modes: circuit, numerical, dynamic, digital, embedded, kicad, unified")
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

        if first in ("import", "include"):
            return self._handle_import_command(line, tokens)

        if first == "export":
            return self._cmd_export(tokens[1:])

        if first in ("session", "room", "team"):
            if first in ("room", "team"):
                return self._handle_session_command(f"session room {' '.join(tokens[1:])}", ["session", "room"] + tokens[1:])
            return self._handle_session_command(line, tokens)

        if first == "ipc":
            return self._cmd_ipc(tokens[1:])

        # Universal High-Resolution PNG Figure Generator & In-Terminal Truecolor Previewer
        if first in ("png", "figure", "preview", "image", "render"):
            return self._handle_figure_image_command(line, tokens)

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
        elif self.mode == "KICAD":
            return self._handle_kicad(line, tokens)
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
            "  png [filename.png]           - Universal: Export active plot/scope/circuit to high-res PNG & Truecolor terminal preview",
            "  preview <filename.png>       - Universal: Render 24-bit Truecolor ANSI graphic preview in terminal",
            "  figure clear / reset         - Universal: Clear active figure canvas and subplots",
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
                "  png [circuit_plot.png]       - Export circuit simulation to high-res PNG + Truecolor terminal preview",
                "  export spice [name]          - Export netlist to standard SPICE .cir file",
                "  list / clear                 - Show or reset netlist",
            ])
        elif self.mode == "NUMERICAL":
            help_texts.extend([
                "\nMATLAB & Signal Processing (DSP) Workbench Commands:",
                "  project <name>               - Create/switch MATLAB project directory in Documents/Terminus Files/Numerical/",
                "  code / script                - View active MATLAB script with line numbers",
                "  edit / edit open             - Launch external editor (Notepad/VS Code) on project .m script",
                "  edit set <code_block>        - Update full MATLAB script code buffer",
                "  run / run <file.m>           - Execute MATLAB script and evaluate variables & plots",
                "  t = 0:0.001:1; x = sin(2*pi*50*t) - Vectorized range, array, and expression generation",
                "  plot(t, x) / stem(n, x)      - Plot continuous waveform or discrete pulses (ASCII + Figure canvas)",
                "  subplot(r, c, i)             - Set multi-graph grid layout (e.g. subplot(2, 1, 1))",
                "  bode(H) / step(H)            - Frequency response Bode diagram & Step response",
                "  butter(...) / firwin(...)    - Digital filter design (Butterworth, Chebyshev, FIR Windowing)",
                "  psd(x, Fs) / specgram(x, Fs) - Power Spectral Density & Time-Frequency Spectrogram",
                "  png [fig_name.png]           - Export high-res 300 DPI graphic with technical metric banner & Truecolor preview",
                "  whos / clear                 - Inspect workspace variables or clear memory",
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
            port_tag = f"[bold green]{self.active_com_port}[/bold green]" if self.active_com_port else "[dim]None[/dim]"
            help_texts.extend([
                f"\nEmbedded & Arduino Real-Hardware IDE Commands (Target: {self.target_board.name} | Port: {port_tag}):",
                "  ports / com                  - Scan connected physical USB microcontroller boards & COM ports",
                "  port <COMx>                  - Select target USB COM port (e.g. port COM3, port COM4)",
                "  port test / reset            - Test port communication or send DTR/RTS hardware reset pulse",
                "  board [list]                 - Browse 30+ supported boards (Arduino, ESP32, STM32, Pico, TI, RISC-V)",
                "  board <board_id>             - Select active hardware board (e.g. board UNO, board ESP32, board STM32_BLUEPILL)",
                "  board info                   - Inspect hardware specs, FQBN, memory sizes, and upload protocols",
                "  code / sketch                - View full project code in editor with line numbers",
                "  code new <ProjectName>       - Create a new project folder (Documents/Terminus Files/Embedded/<ProjectName>/)",
                "  edit / edit open             - Open sketch in external text editor (Notepad, VS Code) or terminal",
                "  edit set <text> / paste      - Replace active sketch buffer with new code text",
                "  edit append <text>           - Append code block to current sketch buffer",
                "  sketch example <name>        - Load hardware templates (blink, sensor_telemetry, analog_read, servo_sweep)",
                "  compile / verify             - Compile sketch via host toolchains, output binary & Flash/RAM usage",
                "  upload / flash               - Upload/dump compiled binary over USB cable to physical board",
                "  dump [output.bin]            - Read & dump raw binary firmware from physical chip over USB",
                "  serial / monitor             - Open live USB Serial Monitor to read real physical sensor stream",
                "  serial plot / plotter        - Display live real-time ASCII telemetry waveform graph from physical sensors",
                "  serial send <text>           - Send commands/strings over USB cable to running physical board",
                "  serial baud <rate>           - Change baud rate (9600, 115200, 230400, etc.)",
                "  file save [name]             - Save project to Documents/Terminus Files/Embedded/<ProjectName>/",
                "  file open <name>             - Open project from storage",
            ])
        elif self.mode == "KICAD":
            help_texts.extend([
                f"\nKiCad EDA Schematic & PCB Layout Commands (Project: {self.kicad_proj.name}):",
                "  library [category]           - Browse KiCad symbol & footprint catalog (Passives, Diodes, Transistors, ICs...)",
                "  library search <query>       - Search symbols and footprints by keyword or model name",
                "  inspect <Ref|Footprint>      - Inspect component pins, footprint pads, and physical dimensions",
                "  add <Ref> <Value> [Footprint]- Add component (e.g. add R1 10k 0805, add C1 100nF 0603, add U1 NE555 SOIC-8)",
                "  connect <p1> | <p2> | <Net>  - Connect pins/nets (e.g. connect V1.1 | R1.1 | VCC, connect R1.2 | C1.1 | Net_Out)",
                "  net <NetName> <p1> <p2> ...  - Create named net interconnect",
                "  remove <Ref>                 - Remove component from schematic and PCB",
                "  schematic / show             - Render paper-style ASCII circuit schematic with short symbol blocks",
                "  place <Ref> <x_mm> <y_mm>    - Place component on PCB board canvas",
                "  route <NetName> / autoroute  - Route copper tracks between component pads on PCB",
                "  pcb / board / layout         - Render 2D terminal PCB layout canvas with traces, vias, and silkscreen",
                "  erc                          - Run Electrical Rules Check (checks floating pins, missing ground, power rails)",
                "  drc                          - Run Design Rules Check (checks clearance, min track width, via sizes)",
                "  probe <Ref>                  - Probe and calculate node voltage, branch current, and power dissipation",
                "  calc track <I_amp> [dT] [oz] - Calculate IPC-2152/2221 PCB track width, resistance, and voltage drop",
                "  calc via <drill_mm> [pad_mm] - Calculate via DC resistance, inductance, capacitance, and ampacity",
                "  calc microstrip <W> <H> [Er] - Calculate RF microstrip transmission line impedance Z0 and delay",
                "  calc 555 <R1> <R2> <C>       - Calculate astable 555 timer frequency, period, and duty cycle",
                "  calc opamp <inv|noninv> <R1> <Rf> - Calculate op-amp closed loop gain (V/V and dB)",
                "  calc divider <Vin> <R1> <R2> - Calculate voltage divider output voltage, ratio, and dissipation",
                "  calc filter <rc|rl|lc> <R> <C> - Calculate filter cutoff frequency (-3dB fc) and bandwidth",
                "  calc reactance <freq> <C/L>  - Calculate capacitive (Xc) or inductive (Xl) reactance",
                "  calc power <V> <I|R>         - Calculate Ohm's law and power dissipation",
                "  gerber / gerbers             - Export complete industry-standard RS-274X Gerber suite & drill files",
                "  bom                          - Export Bill of Materials CSV (.tbom)",
                "  file save [name]             - Save project files (.tkcad, .tsch, .tpcb, .tbom, gerbers) in project folder",
                "  file open <name>             - Open project from storage",
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
            elif self.mode == "KICAD":
                self.kicad_proj = KiCadProject(name=name, storage=self.storage)
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
            proj_name = name or self.matlab_proj.project_name or "SignalWorkspace"
            ok, msg = self.matlab_proj.save_project(proj_name)
            return f"[green]{msg}[/green]"

        elif self.mode == "DIGITAL":
            name = name or "logic_circuit"
            lines = [f"// Terminus Digital Logic - {time.ctime()}"]
            for g in self.logic_circuit.gates.values():
                lines.append(str(g))
            saved_path = self.storage.save_file("DIGITAL", name, "\n".join(lines))
            return f"[green]Saved digital circuit to:[/green] [bold]{saved_path}[/bold]"

        elif self.mode == "EMBEDDED":
            proj_name = name or self.sketch_proj.project_name or "MyProject"
            ok, msg = self.sketch_proj.save_project(proj_name)
            return f"[green]{msg}[/green]"

        elif self.mode == "KICAD":
            proj_name = name or self.kicad_proj.name or "MyPCBProject"
            ok, msg = self.kicad_proj.save_project(proj_name)
            return f"[green]{msg}[/green]"

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
            ok, msg = self.matlab_proj.load_project(name)
            if ok:
                self.numerical_workspace.clear()
                self.numerical_parser.execute_script(self.matlab_proj.source_code)
                return f"[green]{msg}[/green]"
            raise FileNotFoundError(f"MATLAB Project '{name}' not found in {self.storage.get_mode_dir('NUMERICAL')}")

        elif self.mode == "DIGITAL":
            path, data = self.storage.load_file("DIGITAL", name)
            return f"[green]Loaded digital logic from:[/green] [bold]{path}[/bold]"

        elif self.mode == "EMBEDDED":
            ok, msg = self.sketch_proj.load_project(name)
            if ok:
                self.arduino.load_sketch(self.sketch_proj.source_code)
                return f"[green]{msg}[/green]"
            raise FileNotFoundError(f"Project '{name}' not found in {self.storage.get_mode_dir('EMBEDDED')}")

        elif self.mode == "KICAD":
            ok, msg = self.kicad_proj.load_project(name)
            if ok:
                return f"[green]{msg}[/green]"
            raise FileNotFoundError(msg)

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

    def _handle_import_command(self, line: str, tokens: List[str]) -> str:
        """Imports custom SPICE .lib/.mod models or microcontroller .lib/.txt specifications."""
        from pathlib import Path
        if len(tokens) < 2:
            return (
                "[bold cyan]=== Custom Component & MCU Model Importer ===[/bold cyan]\n"
                "  import <filepath>            - Import SPICE .lib/.mod file or MCU .lib/.txt specification\n"
                "  import spice <filepath>      - Import custom SPICE models/subcircuits into component catalog\n"
                "  import mcu <filepath>        - Import microcontroller characteristics into MCU engine\n"
            )

        file_path_arg = tokens[-1].strip('"\'')
        if tokens[1].lower() in ("spice", "lib", "mcu") and len(tokens) >= 3:
            file_path_arg = tokens[2].strip('"\'')

        target_path = Path(file_path_arg)
        if not target_path.exists():
            mode_p = self.storage.get_mode_dir(self.mode) / file_path_arg
            if mode_p.exists():
                target_path = mode_p
            else:
                root_p = self.storage.base_dir / file_path_arg
                if root_p.exists():
                    target_path = root_p
                else:
                    raise FileNotFoundError(f"Import file '{file_path_arg}' not found on disk.")

        models_count, subckts_count, mcu_spec = CustomModelImporter.import_file(str(target_path), circuit_catalog)
        lines = [
            f"[bold green]Successfully imported model library:[/bold green] [bold]{target_path.name}[/bold]",
            f"  * Source Path: {target_path}",
        ]
        if models_count > 0:
            lines.append(f"  * SPICE Models Registered: [bold cyan]{models_count}[/bold cyan]")
        if subckts_count > 0:
            lines.append(f"  * SPICE Subcircuits Registered: [bold cyan]{subckts_count}[/bold cyan]")
        if mcu_spec:
            lines.append(f"  * MCU Hardware Specification: [bold yellow]{mcu_spec.name}[/bold yellow] ({mcu_spec.architecture})")
            lines.append(f"    - Clock: {mcu_spec.clock_freq_mhz} MHz | Flash: {mcu_spec.flash_kb} KB | SRAM: {mcu_spec.sram_kb} KB")
            lines.append(f"    - ADC: {mcu_spec.adc_channels} Channels ({mcu_spec.adc_resolution_bits}-bit, {mcu_spec.adc_input_range[0]}V..{mcu_spec.adc_input_range[1]}V)")
            lines.append(f"    - Peripherals: {mcu_spec.epwm_channels} ePWMs, {mcu_spec.gpio_count} GPIOs")
            self.mcu.apply_specification(mcu_spec)
            lines.append("  * [bold green]Applied hardware characteristics to MCU core.[/bold green]")

        return "\n".join(lines)

    def _cmd_export(self, args: List[str]) -> str:
        """Universal industry-standard exporter for all Terminus modes."""
        from pathlib import Path
        if not args:
            return (
                "[bold cyan]=== Industry Standard Universal Exporter ===[/bold cyan]\n"
                "  export ltspice [name]        - Export Circuit to LTspice .cir / .asc standard schematic\n"
                "  export simulink [name]       - Export Dynamic System to MATLAB Simulink .m script\n"
                "  export matlab [name]         - Export Numerical workspace / TF to MATLAB .m script\n"
                "  export verilog [name]        - Export Digital Logic to IEEE 1364 Verilog HDL (.v)\n"
                "  export vhdl [name]           - Export Digital Logic to IEEE 1076 VHDL (.vhd)\n"
                "  export c [name]              - Export Embedded program to ANSI C/C++ firmware (.c)\n"
                "  export hex [name]            - Export Embedded program to Intel HEX format (.hex)\n"
                "  export asm [name]            - Export Embedded program to Assembly (.asm)\n"
                "  export python [name]         - Export Numerical workspace to Python SciPy script (.py)\n"
                "  export csv [name]            - Export last simulation traces to CSV\n"
            )

        target_fmt = args[0].lower()
        out_name = args[1] if len(args) > 1 else ""

        # 1. Circuit Exporters
        if target_fmt in ("ltspice", "spice", "cir", "net"):
            name = out_name or self.circuit_netlist.name or "circuit_export"
            content = UniversalStandardExporter.export_circuit_ltspice_cir(self.circuit_netlist, name)
            path = self.storage.save_file("EXPORTS", f"{name}.cir", content)
            return (
                f"[bold green]Exported Industry-Standard SPICE/LTspice Netlist:[/bold green] [bold]{path}[/bold]\n"
                f"  * Compatibility: LTspice XVII/24, OrCAD PSpice, ngspice, Cadence Virtuoso\n"
                f"  * Components: {len(self.circuit_netlist.components)} elements"
            )

        elif target_fmt in ("asc", "schematic"):
            name = out_name or self.circuit_netlist.name or "schematic_export"
            content = UniversalStandardExporter.export_circuit_ltspice_asc(self.circuit_netlist, name)
            path = self.storage.save_file("EXPORTS", f"{name}.asc", content)
            return (
                f"[bold green]Exported LTspice Schematic File:[/bold green] [bold]{path}[/bold]\n"
                f"  * Compatibility: Native LTspice Schematic Editor (.asc)\n"
                f"  * Symbols Placed: {len(self.circuit_netlist.components)}"
            )

        # 2. Dynamic Systems Exporters
        elif target_fmt in ("simulink", "slx", "mdl"):
            name = out_name or self.dynamic_diagram.name or "simulink_model"
            content = UniversalStandardExporter.export_dynamic_system_simulink_script(self.dynamic_diagram, name)
            path = self.storage.save_file("EXPORTS", f"build_{name}_simulink.m", content)
            return (
                f"[bold green]Exported MATLAB Simulink Model Generator Script:[/bold green] [bold]{path}[/bold]\n"
                f"  * Compatibility: MATLAB R2018b - R2024b+ (Simulink add_block / add_line API)\n"
                f"  * Blocks: {len(self.dynamic_diagram.blocks)} | Signal Wires: {len(self.dynamic_diagram.connections)}\n"
                f"  * Run in MATLAB: >> {Path(path).stem}"
            )

        # 3. Numerical Exporters
        elif target_fmt in ("matlab", "m"):
            name = out_name or "numerical_workspace"
            content = UniversalStandardExporter.export_numerical_matlab_script(self.numerical_workspace, name)
            path = self.storage.save_file("EXPORTS", f"{name}.m", content)
            return (
                f"[bold green]Exported MATLAB Script:[/bold green] [bold]{path}[/bold]\n"
                f"  * Compatibility: Standard MATLAB / GNU Octave\n"
                f"  * Variables: {len(self.numerical_workspace.variables)}"
            )

        elif target_fmt in ("python", "py", "scipy"):
            name = out_name or "numerical_scipy"
            content = UniversalStandardExporter.export_numerical_python_script(self.numerical_workspace, name)
            path = self.storage.save_file("EXPORTS", f"{name}.py", content)
            return f"[bold green]Exported Python SciPy Script:[/bold green] [bold]{path}[/bold]"

        # 4. Digital Logic Exporters
        elif target_fmt in ("verilog", "v", "hdl"):
            name = out_name or "digital_module"
            content = UniversalStandardExporter.export_digital_verilog(self.logic_circuit, name)
            path = self.storage.save_file("EXPORTS", f"{name}.v", content)
            return (
                f"[bold green]Exported IEEE 1364-2005 Verilog HDL Module:[/bold green] [bold]{path}[/bold]\n"
                f"  * Compatibility: AMD/Xilinx Vivado, Intel Quartus Prime, ModelSim, Synopsys Design Compiler\n"
                f"  * Logic Gates: {len(self.logic_circuit.gates)}"
            )

        elif target_fmt in ("vhdl", "vhd"):
            name = out_name or "digital_module"
            content = UniversalStandardExporter.export_digital_vhdl(self.logic_circuit, name)
            path = self.storage.save_file("EXPORTS", f"{name}.vhd", content)
            return (
                f"[bold green]Exported IEEE 1076 VHDL Entity/Architecture:[/bold green] [bold]{path}[/bold]\n"
                f"  * Compatibility: Vivado, Quartus, GHDL, Synopsys\n"
                f"  * Logic Elements: {len(self.logic_circuit.gates)}"
            )

        # 5. Embedded Firmware Exporters
        elif target_fmt in ("c", "cpp", "firmware"):
            name = out_name or "mcu_firmware"
            content = UniversalStandardExporter.export_embedded_c_firmware(self.mcu, name)
            path = self.storage.save_file("EXPORTS", f"{name}.c", content)
            return (
                f"[bold green]Exported ANSI C/C++ Embedded Firmware:[/bold green] [bold]{path}[/bold]\n"
                f"  * Compatibility: GCC / Clang / TI Code Composer Studio / Keil MDK / IAR Embedded Workbench\n"
                f"  * Program Words: {len(self.mcu.prog_mem)}"
            )

        elif target_fmt in ("hex", "ihex", "intel_hex"):
            name = out_name or "firmware_image"
            content = UniversalStandardExporter.export_embedded_intel_hex(self.mcu)
            path = self.storage.save_file("EXPORTS", f"{name}.hex", content)
            return (
                f"[bold green]Exported Intel HEX Flash Memory Image:[/bold green] [bold]{path}[/bold]\n"
                f"  * Compatibility: J-Link, ST-Link, OpenOCD, Flash Programmers\n"
                f"  * Address Space: 0x0000 - 0x{len(self.mcu.prog_mem):04X}"
            )

        elif target_fmt in ("asm", "assembly", "s"):
            name = out_name or "program"
            content = UniversalStandardExporter.export_embedded_assembly(self.mcu)
            path = self.storage.save_file("EXPORTS", f"{name}.asm", content)
            return f"[bold green]Exported Assembly Source:[/bold green] [bold]{path}[/bold]"

        # 6. Fallback: CSV Simulation export
        name = target_fmt if target_fmt != "csv" else (out_name or "export_simulation")
        content = ""
        if self.mode == "DYNAMIC" and self.last_dynamic_sim:
            lines = [f"# TerminusECE Export - Dynamic System Simulation ({self.dynamic_diagram.name})"]
            for sname, wf in self.last_dynamic_sim.items():
                lines.append(f"\n--- Scope: {sname} ---")
                lines.append(wf.to_csv())
            content = "\n".join(lines)
        elif self.last_circuit_sim:
            lines = [f"# TerminusECE Export - {self.last_circuit_sim.sim_type}"]
            for sname, wf in self.last_circuit_sim.waveforms.items():
                lines.append(f"\n--- {sname} ---")
                lines.append(wf.to_csv())
            content = "\n".join(lines)
        else:
            return "No simulation data available to export to CSV."

        saved_path = self.storage.save_file("EXPORTS", f"{name}.csv", content)
        return f"[green]Exported simulation data to:[/green] [bold]{saved_path}[/bold]"

    def _cmd_ipc(self, args: List[str]) -> str:
        return self._handle_session_command("session " + " ".join(args), ["session"] + args)

    # --- High-Capacity Multi-Terminal Session Network Handlers ---
    def _handle_session_command(self, line: str, tokens: List[str]) -> str:
        if len(tokens) < 2:
            return (
                "[bold cyan]=== Multi-Terminal Session Networking (500+ Terminals) ===[/bold cyan]\n"
                "  session room <code_or_name>  - Join / set Team Room code across all devices\n"
                "  session list / ls            - Discover and list active terminals on LAN/room\n"
                "  session info                 - View current session number, room code, and linked signals\n"
                "  session server start [port]  - Start high-capacity async IPC router server (defaults to 8765)\n"
                "  session server stop          - Stop local IPC router server\n"
                "  session connect <ip:port>    - Connect this terminal session to a network router\n"
                "  session exec <id> <cmd>      - Remotely execute command on terminal Session #<id>\n"
                "  session link <sig> <id>.<sig>- Bridge and live-stream signal with cross-mode validation\n"
                "  session push <sig> <val>     - Push signal value across network to all linked sessions\n"
                "  session broadcast <msg>      - Broadcast message/variable across all terminals\n"
                "  session limits / network     - Display architectural network capacity & scalability details\n"
            )

        sub = tokens[1].lower()

        if sub in ("room", "team", "join"):
            if len(tokens) < 3:
                curr_room = getattr(self.ipc_client, "room_code", "DEFAULT")
                return f"Current Team Room: [bold cyan]{curr_room}[/bold cyan]. Use 'session room <CODE>' to join/create a team room."
            room_code = tokens[2].upper().strip()
            res = self.ipc_client.join_room_sync(room_code)
            members = res.get("members", [self.ipc_client.session_id])
            return (
                f"[bold green]Joined Team Room:[/bold green] [bold cyan]{room_code}[/bold cyan]\n"
                f"Current Session #{self.ipc_client.session_id} is now linked in room '{room_code}'.\n"
                f"Active Members in Room: {members}"
            )

        elif sub in ("list", "ls"):
            room_filter = getattr(self.ipc_client, "room_code", "DEFAULT")
            sessions = self.ipc_client.list_sessions_sync(room=room_filter)
            if not sessions and room_filter != "DEFAULT":
                # Fallback to all sessions if room is empty or unrouted
                sessions = self.ipc_client.list_sessions_sync()

            if not sessions:
                if not self.ipc_client.is_connected:
                    return (
                        "[yellow]This terminal is not connected to an IPC Router.[/yellow]\n"
                        "To host a multi-terminal network on this computer: [bold]session server start[/bold]\n"
                        "To connect to an existing router: [bold]session connect 127.0.0.1:8765[/bold]"
                    )
                return f"[yellow]No other active sessions detected on network router (Room: {room_filter}).[/yellow]"

            lines = [
                f"[bold cyan]=== Active Networked Terminal Sessions (Room: {room_filter} | {len(sessions)} Active / Max 500+) ===[/bold cyan]",
                f"{'Session #':<11}{'Host / IP':<22}{'Mode':<14}{'Active File':<20}{'Room':<10}{'Uptime':<8}"
            ]
            lines.append("-" * 85)
            for s in sessions:
                sid = s.get("session_id", "?")
                is_self = " (Current)" if sid == self.ipc_client.session_id else ""
                host_ip = f"{s.get('hostname', 'host')}:{s.get('port', 0)}"
                mode = s.get("mode", "Circuit")
                active = s.get("active_file", "untitled")
                s_room = s.get("room_code", "DEFAULT")
                uptime = f"{time.time() - s.get('connected_at', time.time()):.0f}s"
                lines.append(f"#{sid:<10}{host_ip:<22}{mode:<14}{active:<20}{s_room:<10}{uptime:<8}{is_self}")
            return "\n".join(lines)

        elif sub == "info":
            if not self.ipc_client.is_connected:
                return "[yellow]Not currently connected to any IPC Network Router. (Run 'session server start' or 'session connect <ip:port>')[/yellow]"
            lines = [
                "[bold cyan]=== Current Terminal Session Network Profile ===[/bold cyan]",
                f"Assigned Session Number: [bold green]Session #{self.ipc_client.session_id}[/bold green]",
                f"Team Room Code         : [bold cyan]{getattr(self.ipc_client, 'room_code', 'DEFAULT')}[/bold cyan]",
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
                raise ValueError("Usage: session connect <ip:port> [requested_session_id] [room_code]")
            endpoint = tokens[2]
            req_id = int(tokens[3]) if len(tokens) > 3 and tokens[3].isdigit() else None
            room_c = tokens[4].upper() if len(tokens) > 4 else (tokens[3].upper() if len(tokens) > 3 and not tokens[3].isdigit() else "DEFAULT")
            host = endpoint.split(":")[0] if ":" in endpoint else endpoint
            port = int(endpoint.split(":")[1]) if ":" in endpoint else 8765

            self.ipc_client = IPCClient(host=host, port=port)
            self.ipc_client.set_command_executor(self.execute_command)
            ok = self.ipc_client.connect(mode=self.mode, active_file=self.circuit_netlist.name, requested_session_id=req_id, room_code=room_c)
            if ok:
                return f"[green]Successfully connected to {host}:{port}[/green] as [bold]Session #{self.ipc_client.session_id}[/bold] in Team Room '[bold cyan]{room_c}[/bold cyan]' (UUID: {self.ipc_client.client_uuid[:8]})."
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
            
            val_info = res.get("validation", {})
            v_status = val_info.get("status", "COMPATIBLE")
            v_msg = val_info.get("message", "Semantic connection verified.")
            v_adapter = val_info.get("adapter", "Direct 1:1 Signal Wire")
            v_warnings = val_info.get("warnings", [])

            color = "green" if v_status == "COMPATIBLE" else "yellow" if v_status == "VALID_WITH_ADAPTER" else "red"
            lines = [
                f"[bold green]Signal Link Established:[/bold green] Session #{self.ipc_client.session_id}:{local_sig} -> Session #{tgt_sid}:{tgt_sig}",
                f"  * Validation Status : [bold {color}]{v_status}[/bold {color}]",
                f"  * Semantic Handshake: {v_msg}",
                f"  * Signal Adapter    : {v_adapter}",
            ]
            for w in v_warnings:
                lines.append(f"  * [bold yellow]Limitation Warning[/bold yellow]: {w}")
            return "\n".join(lines)

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

        # 5. Paper-Style Schematic Visualization
        if first in ("schematic", "paper", "topology"):
            return SchematicVisualizer.render_circuit_topology(self.circuit_netlist.components, self.circuit_netlist.pin_map)

        # 6. Engineering Calculator in Circuit mode
        if first in ("calc", "calculate"):
            return self._cmd_kicad_calc(tokens[1:])

        # 7. Export SPICE
        if first == "export" and len(tokens) >= 2 and tokens[1].lower() == "spice":
            name = tokens[2] if len(tokens) > 2 else self.circuit_netlist.name
            spice_code = self.circuit_netlist.export_spice()
            path = self.storage.save_file("CIRCUIT", f"{name}.cir", spice_code)
            return f"[green]Exported SPICE netlist to:[/green] [bold]{path}[/bold]"

        # Standard Netlist operations
        if first in ("add", "connect", "remove", "delete", "set", "list", "show", "clear"):
            if first in ("show", "list") and len(tokens) == 1:
                return SchematicVisualizer.render_circuit_topology(self.circuit_netlist.components, self.circuit_netlist.pin_map)
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

    # --- Universal High-Resolution PNG Figure & In-Terminal Truecolor Preview Handlers ---
    def _handle_figure_image_command(self, line: str, tokens: List[str]) -> str:
        first = tokens[0].lower()

        # 1. Preview existing PNG file
        if first == "preview" or (len(tokens) > 1 and tokens[1].lower() in ("preview", "view", "show") and any(t.lower().endswith(".png") for t in tokens)):
            target = tokens[1] if len(tokens) > 1 and tokens[1].lower() != "preview" else (tokens[2] if len(tokens) > 2 else "")
            if not target:
                # Find most recent PNG in mode storage
                mode_dir = self.storage.get_mode_dir(self.mode)
                pngs = sorted(mode_dir.glob("**/*.png"), key=os.path.getmtime, reverse=True)
                if not pngs:
                    return f"[yellow]No PNG files found in {mode_dir}. Use 'png <filename>' to export a figure.[/yellow]"
                target = str(pngs[0])
            else:
                p = Path(target)
                if not p.exists():
                    p_mode = self.storage.get_mode_dir(self.mode) / target
                    if p_mode.exists():
                        target = str(p_mode)
                    else:
                        p_proj = self.storage.get_mode_dir(self.mode) / self.matlab_proj.project_name / target
                        if p_proj.exists():
                            target = str(p_proj)
            return TerminalImagePreviewer.preview_image(target)

        # 2. Figure Management: figure clear / figure reset / subplot
        if first == "figure":
            sub = tokens[1].lower() if len(tokens) > 1 else "show"
            if sub in ("clear", "reset", "new"):
                self.figure.clear()
                return "[green]Figure canvas and subplots cleared.[/green]"
            if sub == "subplot" and len(tokens) >= 5:
                r, c, idx = int(tokens[2]), int(tokens[3]), int(tokens[4])
                self.figure.set_subplot_grid(r, c, idx)
                return f"[green]Active subplot set to:[/green] ({r}x{c}, index {idx})"

        # 3. Export PNG: png [filename.png] / figure save [filename.png]
        out_name = tokens[1] if (len(tokens) > 1 and tokens[1].lower() not in ("save", "export", "render")) else (tokens[2] if len(tokens) > 2 else "")
        if not out_name:
            if self.mode == "NUMERICAL":
                out_name = f"{self.matlab_proj.project_name}_figure.png"
            elif self.mode == "CIRCUIT":
                out_name = f"{self.circuit_netlist.name}_circuit.png"
            elif self.mode == "DYNAMIC":
                out_name = f"{self.dynamic_diagram.name}_scopes.png"
            elif self.mode == "EMBEDDED":
                out_name = f"{self.sketch_proj.project_name}_telemetry.png"
            else:
                out_name = f"{self.mode.lower()}_figure.png"

        if not out_name.lower().endswith(".png"):
            out_name += ".png"

        # Determine target folder: project directory or mode directory
        if self.mode == "NUMERICAL":
            dest_path = self.matlab_proj.get_project_dir() / out_name
        elif self.mode == "EMBEDDED":
            dest_path = self.sketch_proj.get_project_dir() / out_name
        else:
            dest_path = self.storage.get_mode_dir(self.mode) / out_name

        # Ensure figure has trace data from current mode
        has_traces = any(sp.traces for sp in self.figure.subplots.values())
        if not has_traces:
            self._auto_populate_figure_from_active_mode()

        ok, msg = self.figure.export_to_png(dest_path, dpi=200)
        if not ok:
            return f"[bold red]Figure Export Failed:[/bold red] {msg}"

        # Generate Truecolor In-Terminal ANSI preview
        preview_str = TerminalImagePreviewer.preview_image(dest_path)
        return (
            f"[bold green]=== High-Resolution PNG Graphic Generated ===[/bold green]\n"
            f"  Subsystem Mode : [bold cyan]{self.mode}[/bold cyan]\n"
            f"  Resolution     : 2400 x 1200 px (300 DPI)\n"
            f"  Saved Image    : [bold white]{dest_path}[/bold white]\n\n"
            f"{preview_str}\n"
            f"[dim]Saved to project directory. Ready for reports, papers, and presentations.[/dim]"
        )

    def _auto_populate_figure_from_active_mode(self):
        """Populates self.figure with current mode simulation results if figure is empty."""
        self.figure.clear()
        if self.mode == "CIRCUIT" and self.last_circuit_sim and self.last_circuit_sim.waveforms:
            self.figure.title = f"LTspice Circuit Analysis: {self.circuit_netlist.name}"
            for name, wf in self.last_circuit_sim.waveforms.items():
                if name != "V(0)":
                    self.figure.add_trace(wf.x, wf.y, label=name)
                    self.figure.metrics = TechnicalReportMetrics.compute_from_waveform(wf.x, wf.y, mode="CIRCUIT", project=self.circuit_netlist.name)
            self.figure.set_labels(xlabel=self.last_circuit_sim.x_label, ylabel="Voltage (V)")

        elif self.mode == "DYNAMIC" and self.last_dynamic_sim:
            self.figure.title = f"Simulink Dynamic Model: {self.dynamic_diagram.name}"
            for name, wf in self.last_dynamic_sim.items():
                self.figure.add_trace(wf.x, wf.y, label=name)
                if self.figure.metrics is None:
                    self.figure.metrics = TechnicalReportMetrics.compute_from_waveform(wf.x, wf.y, mode="DYNAMIC", project=self.dynamic_diagram.name)
            self.figure.set_labels(xlabel="Time (s)", ylabel="Scope Signal Amplitude")

        elif self.mode == "DIGITAL" and self.last_logic_traces:
            self.figure.title = f"Digital Logic Timing Diagram: {self.logic_circuit.name}"
            for name, trace in self.last_logic_traces.items():
                if trace:
                    x_vals = [t for t, _ in trace]
                    y_vals = [1.0 if val == LogicValue.ONE else 0.0 for _, val in trace]
                    self.figure.add_trace(x_vals, y_vals, label=name, style="step")
            self.figure.set_labels(xlabel="Time (ns)", ylabel="Logic Level (0/1)")

        elif self.mode == "EMBEDDED":
            self.figure.title = f"Embedded USB Serial Telemetry: {self.sketch_proj.project_name}"
            if self.ascii_plotter.channels:
                for ch_name, vals in self.ascii_plotter.channels.items():
                    y_arr = np.array(vals)
                    x_arr = np.arange(len(y_arr))
                    self.figure.add_trace(x_arr, y_arr, label=ch_name)
                    if self.figure.metrics is None:
                        self.figure.metrics = TechnicalReportMetrics.compute_from_waveform(x_arr, y_arr, mode="EMBEDDED", project=self.sketch_proj.project_name)
            self.figure.set_labels(xlabel="Sample Index", ylabel="Sensor Value")

        elif self.mode == "NUMERICAL":
            self.figure.title = f"MATLAB Numerical Analysis: {self.matlab_proj.project_name}"
            # Look for 1D arrays in variables
            for k, v in self.numerical_workspace.variables.items():
                if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) > 1:
                    x_arr = np.arange(len(v))
                    self.figure.add_trace(x_arr, v, label=k)
                    self.figure.metrics = TechnicalReportMetrics.compute_from_waveform(x_arr, v, mode="NUMERICAL", project=self.matlab_proj.project_name)
                    break
            self.figure.set_labels(xlabel="Index / Time", ylabel="Amplitude")

    # --- MATLAB & Signal Processing Handlers ---
    def _handle_numerical(self, line: str, tokens: List[str]) -> str:
        first = tokens[0].lower()

        # 1. Project & Script Management (MATLAB .m workflow)
        if first in ("project", "script", "code", "edit"):
            if first in ("code", "script") and (len(tokens) == 1 or tokens[1].lower() in ("show", "view", "cat")):
                return self.matlab_proj.view_code()

            sub = tokens[1].lower() if len(tokens) > 1 else ("open" if first == "edit" else "show")

            if first == "project" and len(tokens) > 1 and sub not in ("new", "create", "open", "save", "list"):
                pname = tokens[1]
                self.matlab_proj.new_project(pname)
                return f"[green]Created MATLAB project '{pname}'[/green] in {self.matlab_proj.get_project_dir()}"

            if sub in ("new", "create"):
                pname = tokens[2] if len(tokens) > 2 else "SignalWorkspace"
                self.matlab_proj.new_project(pname)
                return f"[green]Created new MATLAB project:[/green] [bold]{pname}[/bold] in {self.matlab_proj.get_project_dir()}"

            if sub in ("open", "external", "launch") or (first == "edit" and len(tokens) == 1):
                ok, msg = self.matlab_proj.launch_external_editor()
                return f"[{'green' if ok else 'yellow'}]{msg}[/{'green' if ok else 'yellow'}]"

            if sub in ("set", "paste", "replace"):
                new_code = line[line.lower().find(sub) + len(sub):].strip()
                self.matlab_proj.set_content(new_code)
                return f"[green]Updated MATLAB script buffer[/green] ({len(self.matlab_proj.source_code.splitlines())} lines)."

            if sub == "append":
                append_txt = line[line.lower().find("append") + 6:].strip()
                self.matlab_proj.append_content(append_txt)
                return f"[green]Appended line to MATLAB script[/green] (Total: {len(self.matlab_proj.source_code.splitlines())} lines)."

            if sub == "clear":
                self.matlab_proj.set_content("")
                return "MATLAB script buffer cleared."

            # Direct append if 'code <text>'
            direct_text = line[len(tokens[0]):].strip()
            self.matlab_proj.append_content(direct_text)
            return f"[green]Appended line to MATLAB script[/green] Total lines: {len(self.matlab_proj.source_code.splitlines())}"

        # 2. Run Script (.m Execution)
        if first in ("run", "exec"):
            target_code = self.matlab_proj.source_code
            if len(tokens) > 1 and tokens[1].endswith(".m"):
                f_path = self.matlab_proj.get_project_dir() / tokens[1]
                if f_path.exists():
                    with open(f_path, "r", encoding="utf-8") as f:
                        target_code = f.read()

            logs = self.numerical_parser.execute_script(target_code)
            out_lines = [f"[bold green]=== Executed MATLAB Script: {self.matlab_proj.project_name} ({len(logs)} statements) ===[/bold green]"]
            for idx, stmt, res in logs:
                if str(res).startswith("Error"):
                    out_lines.append(f"  [bold red]L{idx:>3} | {stmt}[/bold red] -> {res}")
                else:
                    res_str = f" = {res}" if res is not None and not isinstance(res, str) else (f": {res}" if res is not None else "")
                    out_lines.append(f"  [dim]L{idx:>3} |[/dim] [cyan]{stmt}[/cyan]{res_str}")

            # If figure was populated during execution, notify
            if any(sp.traces for sp in self.figure.subplots.values()):
                out_lines.append("\n[bold cyan]Figure generated.[/bold cyan] Type [bold green]'png'[/bold green] to render high-res image & in-terminal Truecolor preview.")
            return "\n".join(out_lines)

        # 3. Subplot Layout: subplot(r, c, i) or subplot r c i
        if first == "subplot" or line.startswith("subplot(") or line.startswith("subplot "):
            m = re.search(r'subplot\s*\(?\s*(\d+)[\s,]+(\d+)[\s,]+(\d+)\s*\)?', line)
            if m:
                r, c, idx = int(m.group(1)), int(m.group(2)), int(m.group(3))
                self.figure.set_subplot_grid(r, c, idx)
                return f"[green]Set active subplot:[/green] Grid {r}x{c} -> Index {idx}"

        # 4. Interactive Plotting: plot(...) and stem(...)
        if first in ("plot", "stem") or line.startswith("plot(") or line.startswith("stem("):
            is_stem = first == "stem" or line.startswith("stem(")
            call_content = line[line.find("(") + 1 : line.rfind(")")].strip() if "(" in line else " ".join(tokens[1:])
            args = [a.strip() for a in re.split(r'[\s,]+', call_content) if a.strip()]

            ctx = self.numerical_workspace.get_eval_context()
            ctx["np"] = np
            eval_builtins = {
                "float": float, "int": int, "len": len, "range": range, "abs": abs,
                "max": max, "min": min, "sum": sum, "complex": complex, "bool": bool, "str": str
            }

            x_arr = None
            y_arr = None
            lbl = "Signal"

            if len(args) == 1:
                val = eval(args[0], {"__builtins__": eval_builtins}, ctx)
                y_arr = np.asarray(val, dtype=float)
                x_arr = np.arange(len(y_arr))
                lbl = args[0]
            elif len(args) >= 2:
                v1 = eval(args[0], {"__builtins__": eval_builtins}, ctx)
                v2 = eval(args[1], {"__builtins__": eval_builtins}, ctx)
                x_arr = np.asarray(v1, dtype=float)
                y_arr = np.asarray(v2, dtype=float)
                lbl = args[1]
                if len(args) >= 3:
                    lbl = args[2].strip("'\"")

            if x_arr is not None and y_arr is not None:
                # Add to high-resolution figure
                self.figure.add_trace(x_arr, y_arr, label=lbl, style="stem" if is_stem else "-")
                self.figure.metrics = TechnicalReportMetrics.compute_from_waveform(x_arr, y_arr, mode="MATLAB", project=self.matlab_proj.project_name)
                
                # Render ASCII preview in terminal
                plot_str = AsciiPlotter.plot(x_arr, y_arr, title=f"MATLAB Plot: {lbl}", x_label="Index / Time (s)", y_label="Amplitude")
                m = self.figure.metrics
                stats_str = f"  [dim]Vpk: {m.v_peak:.3f}V | Vrms: {m.v_rms:.3f}V | Vpp: {m.v_pp:.3f}V"
                if m.dominant_freq: stats_str += f" | Peak Freq: {m.dominant_freq:.1f}Hz"
                if m.thd_pct: stats_str += f" | THD: {m.thd_pct:.2f}%"
                stats_str += "[/dim]\n  [bold cyan]> Type 'png' to convert to high-res graphic & in-terminal Truecolor preview.[/bold cyan]"
                return f"{plot_str}\n{stats_str}"

        # 5. Bode & Frequency Response: bode(H) or bode(num, den)
        if line.startswith("bode(") and line.endswith(")"):
            inner = line[5:-1].strip()
            args = [a.strip() for a in inner.split(",") if a.strip()]
            ctx = self.numerical_workspace.get_eval_context()
            if len(args) == 1:
                obj = ctx.get(args[0])
                if isinstance(obj, TransferFunction):
                    # Bode response computation
                    w_arr, mag_db, phase_deg = DSPSignalEngine.bode_response(obj.num, obj.den)
                    self.figure.clear()
                    self.figure.set_subplot_grid(2, 1, 1)
                    self.figure.add_trace(w_arr, mag_db, label="Magnitude", color="#00e5ff")
                    self.figure.set_labels(title=f"Bode Diagram - Magnitude ({args[0]})", xlabel="Frequency (Hz)", ylabel="Magnitude (dB)")
                    self.figure.subplots[0].xscale = "log"

                    self.figure.set_subplot_grid(2, 1, 2)
                    self.figure.add_trace(w_arr, phase_deg, label="Phase", color="#ffab00")
                    self.figure.set_labels(title="Bode Diagram - Phase", xlabel="Frequency (Hz)", ylabel="Phase (deg)")
                    self.figure.subplots[1].xscale = "log"

                    ascii_bode = obj.render_bode_ascii()
                    return f"{ascii_bode}\n\n[dim]Type 'png' to export high-res Bode PNG & in-terminal Truecolor preview.[/dim]"
                raise ValueError(f"Variable '{args[0]}' is not a TransferFunction. Create with H = tf([1], [1, 2, 1]).")

        # 6. Step & Impulse Response
        if line.startswith("step(") and line.endswith(")"):
            var_name = line[5:-1].strip()
            obj = self.numerical_workspace.variables.get(var_name)
            if isinstance(obj, TransferFunction):
                wf = obj.step_response()
                self.figure.add_trace(wf.x, wf.y, label=f"Step Response: {var_name}")
                self.figure.metrics = TechnicalReportMetrics.compute_from_waveform(wf.x, wf.y, mode="MATLAB", project=self.matlab_proj.project_name)
                plot_str = AsciiPlotter.plot(wf.x, wf.y, title=f"Step Response: {var_name}", x_label="Time (s)", y_label="Amplitude")
                return f"{plot_str}\n\n[dim]Type 'png' to export high-res graphic & in-terminal Truecolor preview.[/dim]"
            raise ValueError(f"Variable '{var_name}' is not a TransferFunction.")

        # 7. Standard MATLAB Line Execution
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

    # --- Embedded Handlers (Real Hardware Arduino & Microcontroller IDE) ---
    def _handle_embedded(self, line: str, tokens: List[str]) -> str:
        first = tokens[0].lower()

        # 1. Physical USB COM Ports & Device Discovery
        if first in ("ports", "com", "port"):
            if first in ("ports", "com") or (first == "port" and (len(tokens) == 1 or tokens[1].lower() in ("list", "scan", "ls"))):
                devices = USBPortManager.scan_ports()
                lines = [f"[bold cyan]=== Physical USB COM & Serial Ports ({len(devices)} detected) ===[/bold cyan]"]
                if not devices:
                    lines.append("  [dim]No USB serial ports detected. Connect your microcontroller via USB cable.[/dim]")
                else:
                    for d in devices:
                        is_active = (d.port.upper() == self.active_com_port.upper())
                        active_badge = "[bold green][ACTIVE][/bold green]" if is_active else "        "
                        chip_info = f" -> {d.detected_board_hint}" if d.detected_board_hint else ""
                        lines.append(f"  {active_badge} [bold]{d.port:<8}[/bold] : {d.description}{chip_info}")
                lines.append(f"\nActive Target Port: [bold green]{self.active_com_port}[/bold green]")
                lines.append("[dim]To change port: type 'port <COMx>' (e.g. port COM3, port COM4)[/dim]")
                return "\n".join(lines)

            sub = tokens[1].lower()
            if sub in ("select", "set") or (tokens[1].upper().startswith("COM") or tokens[1].startswith("/dev/")):
                p_name = tokens[2] if sub in ("select", "set") and len(tokens) > 2 else tokens[1]
                self.active_com_port = p_name.upper()
                self.real_flasher.port = self.active_com_port
                self.real_serial.port = self.active_com_port
                return f"[green]Target USB port set to:[/green] [bold]{self.active_com_port}[/bold]"

            if sub in ("test", "check"):
                p_target = tokens[2] if len(tokens) > 2 else self.active_com_port
                ok, msg = USBPortManager.test_connection(p_target, self.real_serial.baudrate)
                color = "green" if ok else "red"
                return f"[{color}]{msg}[/{color}]"

            if sub in ("reset", "dtr", "rts"):
                p_target = tokens[2] if len(tokens) > 2 else self.active_com_port
                ok, msg = USBPortManager.hardware_reset(p_target)
                color = "green" if ok else "red"
                return f"[{color}]{msg}[/{color}]"

        # 2. Target Board Selection & Catalog
        if first in ("board", "boards", "target"):
            if len(tokens) == 1 or tokens[1].lower() in ("list", "ls"):
                cat = tokens[2].upper() if len(tokens) > 2 else ""
                table_str = board_catalog.generate_summary_table(cat)
                curr = f"Active Target: [bold green]{self.target_board.name} ({self.target_board.mcu})[/bold green] [{self.target_board.id}]"
                return f"{curr}\n\n{table_str}\n\n[dim]To select a board, type 'board <id>' (e.g. board ESP32, board STM32_BLUEPILL, board RPI_PICO)[/dim]"

            if tokens[1].lower() in ("info", "inspect", "show"):
                target_id = tokens[2].upper() if len(tokens) > 2 else self.target_board.id
                spec = board_catalog.get(target_id)
                if not spec:
                    raise ValueError(f"Unknown board ID '{target_id}'. Use 'board list' to see all.")
                lines = [
                    f"[bold cyan]=== Microcontroller Specification: {spec.name} ===[/bold cyan]",
                    f"  Board ID     : {spec.id} (Family: {spec.family})",
                    f"  MCU / Core   : {spec.mcu} ({spec.architecture}) @ {spec.clock_mhz} MHz ({spec.voltage_v}V logic)",
                    f"  Flash Memory : {spec.flash_kb} KB ({spec.flash_kb*1024:,} bytes) [Base: {spec.flash_base_addr}]",
                    f"  SRAM Memory  : {spec.ram_kb} KB ({spec.ram_kb*1024:,} bytes) [Base: {spec.ram_base_addr}]",
                    f"  EEPROM Memory: {spec.eeprom_bytes} bytes",
                    f"  I/O Pins     : {spec.gpio_count} Digital GPIOs, {spec.adc_channels} ADC channels ({spec.adc_resolution_bits}-bit), {spec.pwm_channels} PWM pins",
                    f"  Protocols    : UART={spec.has_uart}, I2C={spec.has_i2c}, SPI={spec.has_spi}, CAN={spec.has_can}, WiFi={spec.has_wifi}, BLE={spec.has_ble}",
                    f"  Toolchain    : {spec.upload_tool} ({spec.upload_protocol} @ {spec.default_baud} baud)",
                ]
                if spec.fqbn:
                    lines.append(f"  Arduino FQBN : {spec.fqbn}")
                return "\n".join(lines)

            # Set board
            new_id = tokens[1].upper()
            spec = board_catalog.get(new_id)
            if not spec:
                matches = board_catalog.search(tokens[1])
                if matches:
                    spec = matches[0]
                else:
                    raise ValueError(f"Board '{tokens[1]}' not found in catalog. Type 'board list' for all supported boards.")
            self.target_board = spec
            self.real_flasher.board = spec
            self.arduino.set_board(spec.id)
            return f"[green]Target board switched to:[/green] [bold]{spec.name}[/bold] ({spec.mcu}, {spec.clock_mhz}MHz, {spec.flash_kb}KB Flash, {spec.ram_kb}KB RAM)"

        # 3. Sketch Code Editor & Project Management
        if first in ("code", "sketch", "project", "edit", "template"):
            if first == "template":
                ex_name = tokens[1].lower() if len(tokens) > 1 else "blink"
                return self._load_template(ex_name)

            if first in ("code", "sketch") and (len(tokens) == 1 or tokens[1].lower() in ("show", "view", "cat")):
                return self.sketch_proj.view_code()

            sub = tokens[1].lower() if len(tokens) > 1 else ("open" if first == "edit" else "show")

            if first == "project" and len(tokens) > 1 and sub not in ("new", "create", "open", "save", "list", "files"):
                # Shorthand: 'project MyProj' -> create project
                pname = tokens[1]
                self.sketch_proj.new_project(pname)
                self.arduino.load_sketch(self.sketch_proj.source_code)
                return f"[green]Created project '{pname}'[/green] in {self.sketch_proj.get_project_dir()}"

            if sub in ("new", "create"):
                pname = tokens[2] if len(tokens) > 2 else "NewProject"
                self.sketch_proj.new_project(pname)
                self.arduino.load_sketch(self.sketch_proj.source_code)
                return f"[green]Created new project:[/green] [bold]{pname}[/bold] in {self.sketch_proj.get_project_dir()}"

            if sub in ("open", "external", "launch") or first == "edit":
                # Launch external editor if no extra args
                if len(tokens) == 1 or (len(tokens) == 2 and sub in ("open", "external")):
                    ok, msg = self.sketch_proj.launch_external_editor()
                    return f"[{'green' if ok else 'yellow'}]{msg}[/{'green' if ok else 'yellow'}]"

                if sub in ("set", "paste", "replace"):
                    new_code = line[line.lower().find(sub) + len(sub):].strip()
                    self.sketch_proj.set_content(new_code)
                    self.arduino.load_sketch(self.sketch_proj.source_code)
                    return f"[green]Updated sketch buffer[/green] ({len(self.sketch_proj.source_code.splitlines())} lines)."

                if sub == "append":
                    append_txt = line[line.lower().find("append") + 6:].strip()
                    self.sketch_proj.append_content(append_txt)
                    self.arduino.load_sketch(self.sketch_proj.source_code)
                    return f"[green]Appended line to sketch buffer[/green] (Total: {len(self.sketch_proj.source_code.splitlines())} lines)."

            if sub == "clear":
                self.sketch_proj.set_content("")
                self.arduino.source_code = ""
                return "Sketch buffer cleared."

            if sub == "example":
                ex_name = tokens[2].lower() if len(tokens) > 2 else "blink"
                return self._load_template(ex_name)

            # Direct append
            direct_text = line[len(tokens[0]):].strip()
            self.sketch_proj.append_content(direct_text)
            self.arduino.load_sketch(self.sketch_proj.source_code)
            return f"[green]Appended line to sketch.[/green] Total lines: {len(self.sketch_proj.source_code.splitlines())}"

        # 4. Compilation & Verification
        if first in ("compile", "verify", "build"):
            # Save project first to ensure build directory is in sync
            self.sketch_proj.save_project()
            main_path = self.sketch_proj.get_main_filepath()

            ok, report, bin_path = self.real_flasher.compile_project(main_path)
            if not ok:
                return f"[bold red]Compilation Failed:[/bold red]\n{report}"

            lines = [
                report,
                f"\n[bold green]Ready for hardware upload.[/bold green] Type 'upload' or 'flash' to deploy to {self.active_com_port}."
            ]
            return "\n".join(lines)

        # 5. Real Hardware Upload / Flashing
        if first in ("upload", "flash", "deploy", "dump", "read_flash"):
            if first in ("dump", "read_flash"):
                out_name = tokens[1] if len(tokens) > 1 else f"{self.sketch_proj.project_name}_dump.bin"
                out_path = self.sketch_proj.get_project_dir() / out_name
                ok, msg = self.real_flasher.dump_flash_from_hardware(out_path, self.active_com_port)
                color = "green" if ok else "red"
                return f"[{color}]{msg}[/{color}]"

            # Auto-compile before upload
            self.sketch_proj.save_project()
            main_path = self.sketch_proj.get_main_filepath()
            ok_comp, report, bin_path = self.real_flasher.compile_project(main_path)
            if not ok_comp:
                return f"[bold red]Upload aborted: Compilation failed.[/bold red]\n{report}"

            # Target port
            target_port = tokens[1].upper() if len(tokens) > 1 and tokens[1].upper().startswith("COM") else self.active_com_port

            # Close serial monitor if open to prevent port locking
            if self.real_serial.connected and self.real_serial.port == target_port:
                self.real_serial.disconnect()

            res = self.real_flasher.upload_to_hardware(bin_path, target_port)
            if res.success:
                # Auto-reconnect serial monitor after upload
                self.real_serial.connect(target_port)
                return (
                    f"[bold green]=== Upload Succeeded ===[/bold green]\n"
                    f"  Target Board : {self.target_board.name} ({self.target_board.mcu})\n"
                    f"  Port         : {res.port}\n"
                    f"  Toolchain    : {res.toolchain_used}\n"
                    f"  Payload Size : {res.bytes_written:,} bytes\n"
                    f"  Duration     : {res.duration_s:.2f} s\n\n"
                    f"[bold cyan]Serial Monitor reconnected to {target_port}. Type 'serial monitor' or 'serial plot' to view telemetry.[/bold cyan]"
                )
            else:
                return res.message

        # 6. Physical USB Serial Monitor & Waveform Plotter
        if first in ("serial", "monitor", "plot", "plotter", "send"):
            if first == "send":
                msg = line[len(tokens[0]):].strip()
                ok, out_msg = self.real_serial.write_data(msg)
                color = "green" if ok else "red"
                return f"[{color}]{out_msg}[/{color}]"

            if first in ("plot", "plotter") or (len(tokens) > 1 and tokens[1].lower() in ("plot", "plotter", "chart")):
                if len(tokens) > 1 and tokens[1].lower() == "status":
                    return f"Live ASCII Telemetry Plotter: active, channels={len(self.real_serial.plotter.channels)}"
                return self.real_serial.plotter.render()

            if len(tokens) == 1 or tokens[1].lower() in ("monitor", "show", "read", "view"):
                if not self.real_serial.connected:
                    self.real_serial.connect()
                return self.real_serial.get_monitor_view()

            sub = tokens[1].lower()
            if sub == "status":
                conn_str = f"Connected to {self.real_serial.port} @ {self.real_serial.baudrate} baud" if self.real_serial.connected else "Disconnected"
                return f"Physical USB Serial Monitor: {conn_str}, Buffer count: {len(self.real_serial.rx_buffer)}"

            if sub == "clear":
                self.real_serial.clear_buffer()
                return "Serial Monitor and Plotter buffers cleared."

            if sub in ("send", "write", "tx"):
                msg = " ".join(tokens[2:])
                ok, out_msg = self.real_serial.write_data(msg)
                color = "green" if ok else "red"
                return f"[{color}]{out_msg}[/{color}]"

            if sub in ("baud", "speed", "rate"):
                if len(tokens) < 3:
                    return f"Current Serial Baud Rate: {self.real_serial.baudrate} baud."
                new_baud = int(tokens[2])
                self.real_serial.baudrate = new_baud
                self.real_serial.connect(baudrate=new_baud)
                return f"[green]Serial baud rate switched to:[/green] [bold]{new_baud} baud[/bold]."

            if sub in ("connect", "open"):
                p = tokens[2] if len(tokens) > 2 else self.active_com_port
                ok, msg = self.real_serial.connect(port=p)
                color = "green" if ok else "red"
                return f"[{color}]{msg}[/{color}]"

            if sub in ("disconnect", "close"):
                self.real_serial.disconnect()
                return f"[yellow]Serial Monitor disconnected from {self.active_com_port}.[/yellow]"

        # 7. Hybrid / Simulation Fallbacks
        if first in ("run", "sim", "step", "pins"):
            if first in ("run", "sim"):
                cycles = int(tokens[1]) if len(tokens) > 1 else 10
                self.arduino.load_sketch(self.sketch_proj.source_code)
                ok, logs = self.arduino.run_continuous(cycles=cycles)
                if logs:
                    self.ascii_plotter.feed_lines(logs)
                return f"[bold green]Executed {cycles} simulated cycle(s) on {self.target_board.name}.[/bold green]\n" + "\n".join(f"  > {l}" for l in logs[-15:])
            if first == "step":
                self.arduino.load_sketch(self.sketch_proj.source_code)
                delta = self.arduino.step_loop()
                return f"[bold green]Stepped 1 iteration.[/bold green] Serial: " + " | ".join(delta)
            if first == "pins":
                return self.arduino.get_pinout_ascii()

        # 8. Assembly-Level C2000 Legacy Fallback
        if first == "load":
            asm_code = line[4:].strip()
            if os.path.exists(asm_code):
                with open(asm_code, "r", encoding="utf-8") as f:
                    asm_code = f.read()
            count = self.mcu.load_program(asm_code)
            return f"[green]Loaded MCU program:[/green] {count} instruction(s) assembled.\n\n" + Disassembler.disassemble(self.mcu.prog_mem, pc_highlight=0)

        if first == "dump":
            return self.mcu.dump_state()

        if first == "reset":
            self.arduino.reset()
            self.mcu.reset()
            return "Embedded MCU and IDE Workspace Reset complete."

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

        raise ValueError(f"Unknown embedded command '{line}'. Type 'help' for all commands.")

    def _load_template(self, ex_name: str) -> str:
        examples = {
            "blink": (
                "// ==========================================\n"
                "// Built-in Hardware Template: LED Blink\n"
                "// Connect LED anode to Pin 13 / onboard LED\n"
                "// ==========================================\n"
                "const int ledPin = 13;\n\n"
                "void setup() {\n"
                "  pinMode(ledPin, OUTPUT);\n"
                "  Serial.begin(115200);\n"
                "  Serial.println(\"=== Hardware LED Blink Active ===\");\n"
                "}\n\n"
                "void loop() {\n"
                "  digitalWrite(ledPin, HIGH);\n"
                "  Serial.println(\"LED: HIGH\");\n"
                "  delay(500);\n"
                "  digitalWrite(ledPin, LOW);\n"
                "  Serial.println(\"LED: LOW\");\n"
                "  delay(500);\n"
                "}\n"
            ),
            "sensor_telemetry": (
                "// ==========================================\n"
                "// Hardware Template: Multi-Channel Sensor Telemetry\n"
                "// Connect Analog Sensors to A0, A1, A2\n"
                "// ==========================================\n"
                "void setup() {\n"
                "  Serial.begin(115200);\n"
                "  pinMode(A0, INPUT);\n"
                "  pinMode(A1, INPUT);\n"
                "}\n\n"
                "void loop() {\n"
                "  float v_a0 = analogRead(A0) * (5.0 / 1023.0);\n"
                "  float v_a1 = analogRead(A1) * (5.0 / 1023.0);\n"
                "  // Output format for Serial Plotter\n"
                "  Serial.print(v_a0);\n"
                "  Serial.print(\", \");\n"
                "  Serial.println(v_a1);\n"
                "  delay(100);\n"
                "}\n"
            ),
            "servo_sweep": (
                "// ==========================================\n"
                "// Hardware Template: Servo Motor Sweep\n"
                "// Connect Servo PWM signal to Pin 9\n"
                "// ==========================================\n"
                "#include <Servo.h>\n"
                "Servo myServo;\n\n"
                "void setup() {\n"
                "  myServo.attach(9);\n"
                "  Serial.begin(115200);\n"
                "  Serial.println(\"Servo Sweep Initialized on Pin 9\");\n"
                "}\n\n"
                "void loop() {\n"
                "  for (int pos = 0; pos <= 180; pos += 45) {\n"
                "    myServo.write(pos);\n"
                "    Serial.print(\"Servo Angle: \");\n"
                "    Serial.println(pos);\n"
                "    delay(200);\n"
                "  }\n"
                "}\n"
            ),
            "serial_echo": (
                "// ==========================================\n"
                "// Hardware Template: Bidirectional Serial Echo\n"
                "// Interact via 'serial send <command>'\n"
                "// ==========================================\n"
                "void setup() {\n"
                "  Serial.begin(115200);\n"
                "  Serial.println(\"Ready. Send characters to receive echo.\");\n"
                "}\n\n"
                "void loop() {\n"
                "  if (Serial.available() > 0) {\n"
                "    String incoming = Serial.readStringUntil('\\n');\n"
                "    Serial.print(\"Echo Received: \");\n"
                "    Serial.println(incoming);\n"
                "  }\n"
                "}\n"
            ),
        }
        if ex_name not in examples:
            valid_ex = ", ".join(examples.keys())
            raise ValueError(f"Unknown template '{ex_name}'. Available: {valid_ex}")

        self.sketch_proj.set_content(examples[ex_name])
        self.arduino.load_sketch(self.sketch_proj.source_code)
        return f"[green]Loaded hardware template '{ex_name}' into project.[/green]\nType 'compile' then 'upload' to dump onto real board."

    # --- KiCad EDA & PCB Design Handlers ---
    def _handle_kicad(self, line: str, tokens: List[str]) -> str:
        first = tokens[0].lower()

        # 1. KiCad Library & Footprint Browser
        if first == "library":
            return self._cmd_kicad_library(tokens[1:])

        # 2. Inspect component, footprint, or model
        if first in ("inspect", "details", "info"):
            if len(tokens) < 2:
                raise ValueError("Usage: inspect <component_ref_or_footprint> (e.g. inspect R1, inspect 0805, inspect SOIC-8)")
            return self._cmd_kicad_inspect(tokens[1])

        # 3. Add Component (e.g. add R1 10k 0805, add U1 NE555 SOIC-8)
        if first == "add":
            if len(tokens) < 3:
                raise ValueError("Usage: add <Ref> <Value> [Footprint] (e.g. add R1 10k 0805, add C1 100nF 0603, add U1 NE555 SOIC-8)")
            ref = tokens[1].upper()
            val = tokens[2]
            fp = tokens[3] if len(tokens) > 3 else None
            comp = self.kicad_proj.schematic.add_component(ref, val, footprint=fp)
            # Sync to PCB
            self.kicad_proj.pcb.add_component(ref, comp.footprint_name, value=val, comp_type=comp.comp_type)
            return f"[green]Added KiCad Component:[/green] [bold]{ref}[/bold] ({comp.value}) [dim]Footprint: {comp.footprint_name} | Pins: {len(comp.pins)}[/dim]"

        # 4. Connect Pins / Nets (e.g. connect V1.1 | R1.1 | VCC)
        if first in ("connect", "wire"):
            args = line[len(tokens[0]):].strip()
            endpoints = [p.strip() for p in args.split("|") if p.strip()]
            if not endpoints:
                endpoints = tokens[1:]
            if len(endpoints) < 2:
                raise ValueError("Usage: connect <pin1> | <pin2> | [NetName] (e.g. connect V1.1 | R1.1 | VCC)")
            net_name = None
            if len(endpoints) >= 3 and not ("." in endpoints[-1]):
                net_name = endpoints.pop()
            for i in range(len(endpoints) - 1):
                self.kicad_proj.schematic.connect(endpoints[i], endpoints[i+1], net_name=net_name)
            # Sync nets to PCB
            for net_name, pins in self.kicad_proj.schematic.nets.items():
                for ref, pnum in pins:
                    self.kicad_proj.pcb.set_pad_net(ref, pnum, net_name)
            return f"[green]Connected:[/green] {' ─── '.join(endpoints)} [dim](Net: {net_name or 'Auto'})[/dim]"

        # 5. Named Net interconnect (e.g. net VCC V1.1 R1.1 U1.8)
        if first == "net":
            if len(tokens) < 3:
                raise ValueError("Usage: net <NetName> <pin1> <pin2> ... (e.g. net VCC V1.1 R1.1)")
            net_name = tokens[1]
            pins = tokens[2:]
            for i in range(len(pins) - 1):
                self.kicad_proj.schematic.connect(pins[i], pins[i+1], net_name=net_name)
            for ref, pnum in self.kicad_proj.schematic.nets.get(net_name, set()):
                self.kicad_proj.pcb.set_pad_net(ref, pnum, net_name)
            return f"[green]Created Net [{net_name}]:[/green] {', '.join(pins)}"

        # 6. Remove Component
        if first in ("remove", "delete", "rm"):
            if len(tokens) < 2:
                raise ValueError("Usage: remove <Ref>")
            ref = tokens[1].upper()
            if ref in self.kicad_proj.schematic.components:
                del self.kicad_proj.schematic.components[ref]
            if ref in self.kicad_proj.pcb.placed_components:
                del self.kicad_proj.pcb.placed_components[ref]
            return f"[green]Removed component:[/green] [bold]{ref}[/bold]"

        # 7. Paper-Style Schematic View
        if first in ("schematic", "sch", "show", "circuit"):
            return self.kicad_proj.schematic.render_schematic_ascii()

        # 8. PCB Placement (e.g. place R1 10 20 90)
        if first == "place":
            if len(tokens) < 4:
                raise ValueError("Usage: place <Ref> <x_mm> <y_mm> [rotation_deg] (e.g. place R1 10 15 0)")
            ref = tokens[1].upper()
            x = float(tokens[2])
            y = float(tokens[3])
            rot = float(tokens[4]) if len(tokens) > 4 else 0.0
            comp = self.kicad_proj.schematic.components.get(ref)
            fp_name = comp.footprint_name if comp else "0805"
            self.kicad_proj.pcb.place_component(ref, x, y, rotation_deg=rot, footprint=fp_name)
            return f"[green]Placed on PCB:[/green] [bold]{ref}[/bold] at ({x:.1f}, {y:.1f}) mm, rot={rot:.0f}°"

        # 9. PCB Routing (e.g. route VCC, autoroute)
        if first in ("route", "autoroute"):
            if first == "autoroute" or (len(tokens) > 1 and tokens[1].lower() == "all"):
                ok, cnt = self.kicad_proj.pcb.autoroute_all_nets()
                return f"[green]Auto-routed {cnt} copper tracks across all nets.[/green]"
            elif len(tokens) >= 2:
                net_name = tokens[1]
                ok, msg = self.kicad_proj.pcb.route_net(net_name)
                return f"[green]{msg}[/green]"
            else:
                ok, cnt = self.kicad_proj.pcb.autoroute_all_nets()
                return f"[green]Auto-routed {cnt} copper tracks.[/green]"

        # 10. PCB 2D Board Canvas View
        if first in ("pcb", "board", "layout"):
            return self.kicad_proj.pcb.render_board_ascii()

        # 11. Electrical Rules Check (ERC)
        if first in ("erc", "rules", "check_erc"):
            return self._cmd_kicad_erc()

        # 12. Design Rules Check (DRC)
        if first in ("drc", "check_drc", "check"):
            return self._cmd_kicad_drc()

        # 13. Probe Component / Value calculation
        if first in ("probe", "calc_comp", "measure"):
            if len(tokens) < 2:
                raise ValueError("Usage: probe <Ref> (e.g. probe R1, probe C1, probe U1)")
            return self._cmd_kicad_probe(tokens[1])

        # 14. Engineering Calculators
        if first in ("calc", "calculate"):
            return self._cmd_kicad_calc(tokens[1:])

        # 15. Export Gerbers
        if first in ("gerber", "gerbers"):
            res = self.kicad_proj.export_gerbers()
            lines = [f"[bold green]Generated Complete RS-274X Gerber & Drill Suite ({len(res)} files):[/bold green]"]
            for layer_name, p in res.items():
                lines.append(f"  * [cyan]{layer_name:<16}[/cyan] -> {p.name}")
            return "\n".join(lines)

        # 16. Bill of Materials (BOM)
        if first == "bom":
            pdir = self.kicad_proj.get_project_dir()
            tbom_path = pdir / f"{self.kicad_proj.name}.tbom"
            self.kicad_proj.save_project()
            with open(tbom_path, "r", encoding="utf-8") as f:
                content = f.read()
            return f"[bold cyan]=== Bill of Materials ({self.kicad_proj.name}) ===[/bold cyan]\n{content}"

        # 17. Clear / Reset
        if first == "clear":
            self.kicad_proj.schematic.clear()
            self.kicad_proj.pcb.clear()
            return "[yellow]Cleared active KiCad schematic and PCB layout.[/yellow]"

        raise ValueError(f"Unknown KiCad command '{line}'. Type 'help' for available commands.")

    def _cmd_kicad_library(self, args: List[str]) -> str:
        """Browse KiCad standard symbol & footprint libraries."""
        FootprintCatalog.initialize()
        fps = FootprintCatalog.list_footprints()
        if not args:
            categories: Dict[str, List[FootprintSpec]] = {}
            for fp in fps:
                categories.setdefault(fp.category, []).append(fp)
            lines = [
                "[bold cyan]=== KiCad EDA Classified Component & Footprint Library ===[/bold cyan]",
                "Standard Categories & Footprints:"
            ]
            for cat, specs in categories.items():
                examples = ", ".join(s.name for s in specs[:4])
                lines.append(f"  * [bold yellow]{cat:<18}[/bold yellow] ({len(specs)} footprints) - e.g. {examples}")
            lines.append("\nCommands:")
            lines.append("  library <category>           - List all footprints in a category (Passive, Semiconductor, IC, Connector...)")
            lines.append("  library search <query>       - Search footprints by name/dimensions")
            lines.append("  inspect <FootprintName>      - View physical pad dimensions, pitch, and drill specs")
            return "\n".join(lines)

        sub = args[0].lower()
        if sub in ("search", "find"):
            if len(args) < 2:
                raise ValueError("Usage: library search <query>")
            q = args[1].lower()
            matches = [fp for fp in fps if q in fp.name.lower() or q in fp.category.lower() or q in fp.description.lower()]
            if not matches:
                return f"[yellow]No footprints found matching '{q}'.[/yellow]"
            lines = [f"[bold cyan]Search Results for '{q}' ({len(matches)} found):[/bold cyan]"]
            for m in matches:
                tht_smd = "SMD" if m.is_smd else "THT"
                lines.append(f"  * [bold green]{m.name:<25}[/bold green] [[dim]{m.category} | {tht_smd} | {m.pad_count} pads[/dim]] - {m.description}")
            return "\n".join(lines)

        category = args[0]
        matches = [fp for fp in fps if fp.category.lower() == category.lower()]
        if not matches:
            # Try as footprint name
            fp_spec = FootprintCatalog.get_footprint(category)
            if fp_spec:
                return self._cmd_kicad_inspect(fp_spec.name)
            return f"[yellow]Category '{category}' not found. Run 'library' to list all categories.[/yellow]"

        lines = [f"[bold cyan]=== KiCad Library: {category} ({len(matches)} items) ===[/bold cyan]"]
        for m in matches:
            tht_smd = "SMD" if m.is_smd else "THT"
            lines.append(f"  * [bold green]{m.name:<25}[/bold green] ({tht_smd}, {m.width_mm:.1f}x{m.height_mm:.1f}mm, {m.pad_count} pads) - [dim]{m.description}[/dim]")
        return "\n".join(lines)

    def _cmd_kicad_inspect(self, target: str) -> str:
        """Inspects component instance or footprint specification."""
        utarget = target.strip().upper()
        # Check active schematic components
        if utarget in self.kicad_proj.schematic.components:
            c = self.kicad_proj.schematic.components[utarget]
            fp_spec = FootprintCatalog.get_footprint(c.footprint_name)
            lines = [
                f"[bold cyan]=== Schematic Component: {c.ref} ===[/bold cyan]",
                f"Type       : {c.comp_type}",
                f"Value      : {c.value}",
                f"Footprint  : {c.footprint_name}",
                f"Pins ({len(c.pins)}):"
            ]
            for pnum, pin in c.pins.items():
                net_str = f"[bold green]{pin.net}[/bold green]" if pin.net else "[dim]Floating[/dim]"
                lines.append(f"  * Pin {pnum} ({pin.name}): Net -> {net_str}")
            return "\n".join(lines)

        # Check footprint catalog
        fp_spec = FootprintCatalog.get_footprint(target)
        if fp_spec:
            lines = [
                f"[bold cyan]=== KiCad Footprint Spec: {fp_spec.name} ===[/bold cyan]",
                f"Category   : {fp_spec.category}",
                f"Dimensions : {fp_spec.width_mm:.2f} x {fp_spec.height_mm:.2f} mm",
                f"Mounting   : {'Surface Mount (SMD)' if fp_spec.is_smd else 'Through-Hole (THT)'}",
                f"Pad Count  : {fp_spec.pad_count}",
                f"Description: {fp_spec.description}",
                "Pads Layout:"
            ]
            for pad in fp_spec.pads:
                drill_str = f" | Drill: {pad.drill_mm:.2f}mm" if pad.is_tht else ""
                lines.append(f"  * Pad {pad.number} ({pad.name}): offset=({pad.x_offset_mm:+.2f}, {pad.y_offset_mm:+.2f})mm, size={pad.width_mm:.2f}x{pad.height_mm:.2f}mm [{pad.shape}{drill_str}]")
            return "\n".join(lines)

        raise KeyError(f"Neither component '{target}' nor footprint '{target}' was found.")

    def _cmd_kicad_probe(self, ref: str) -> str:
        """Probes electrical parameters and calculations for a component."""
        data = self.kicad_proj.schematic.probe_component(ref)
        lines = [
            f"[bold cyan]=== Electrical Calculation & Probe: {data.get('ref')} ===[/bold cyan]",
            f"Component Type : {data.get('type')}",
            f"Value          : {data.get('value')}",
        ]
        if "voltage_drop_v" in data:
            lines.append(f"Voltage Drop   : [bold green]{data['voltage_drop_v']:.3f} V[/bold green]")
        if "current_ma" in data:
            lines.append(f"Branch Current : [bold green]{data['current_ma']:.3f} mA[/bold green] ({data['current_a']:.6g} A)")
        if "power_mw" in data:
            lines.append(f"Power Dissipated: [bold yellow]{data['power_mw']:.3f} mW[/bold yellow]")
        if "reactance_1khz_ohms" in data:
            lines.append(f"Reactance @1kHz: [bold cyan]{data['reactance_1khz_ohms']:.2f} Ω[/bold cyan]")
        if "pins" in data:
            lines.append("Pin Connections:")
            for p, net in data["pins"].items():
                net_str = net if net else "Floating"
                lines.append(f"  * Pin {p} -> Net [{net_str}]")
        return "\n".join(lines)

    def _cmd_kicad_erc(self) -> str:
        """Runs Electrical Rules Check (ERC) on KiCad schematic."""
        issues = self.kicad_proj.schematic.run_erc()
        lines = [
            f"[bold cyan]=== KiCad Electrical Rules Check (ERC): {self.kicad_proj.name} ===[/bold cyan]",
            f"Components: {len(self.kicad_proj.schematic.components)} | Nets: {len(self.kicad_proj.schematic.nets)}"
        ]
        if not issues:
            lines.append("[bold green]PASSED: No electrical rule errors or floating pins detected.[/bold green]")
        else:
            for issue in issues:
                lvl = issue.get("level", "WARNING")
                color = "red" if lvl == "ERROR" else "yellow"
                lines.append(f"  [{color}][{lvl}][/{color}] {issue.get('message')}")
        return "\n".join(lines)

    def _cmd_kicad_drc(self) -> str:
        """Runs Design Rules Check (DRC) on KiCad PCB layout."""
        report = self.kicad_proj.pcb.run_drc()
        lines = [
            f"[bold cyan]=== KiCad PCB Design Rules Check (DRC): {self.kicad_proj.name} ===[/bold cyan]",
            f"Board Size: {self.kicad_proj.pcb.width_mm}x{self.kicad_proj.pcb.height_mm}mm | Tracks: {len(self.kicad_proj.pcb.tracks)} | Vias: {len(self.kicad_proj.pcb.vias)} | Status: {'[bold green]PASSED[/bold green]' if report.get('is_valid') else '[bold red]VIOLATIONS[/bold red]'}"
        ]
        issues = report.get("issues", [])
        if not issues:
            lines.append("[bold green]PASSED: All clearances, track widths, and annular rings meet manufacturing constraints.[/bold green]")
        else:
            for issue in issues:
                lines.append(f"  [red][VIOLATION][/red] {issue}")
        return "\n".join(lines)

    def _cmd_kicad_calc(self, args: List[str]) -> str:
        """Built-in engineering calculators."""
        if not args:
            return (
                "[bold cyan]=== Terminus Engineering Calculators ===[/bold cyan]\n"
                "  calc track <current_A> [temp_rise_C] [copper_oz] [len_mm] - IPC-2152/2221 PCB track width & resistance\n"
                "  calc via <drill_mm> [pad_mm] [pcb_thick_mm]               - Via DC resistance, inductance, capacitance\n"
                "  calc microstrip <width_mm> <height_mm> [Er]               - RF Microstrip impedance Z0 & propagation delay\n"
                "  calc 555 <R1> <R2> <C>                                    - Astable 555 timer frequency, duty cycle, period\n"
                "  calc opamp <inv|noninv> <R1> <Rf>                         - Op-Amp closed loop gain (V/V & dB)\n"
                "  calc divider <Vin> <R1> <R2>                              - Voltage divider output voltage & power\n"
                "  calc regulator <Vout> [Vref] [R1]                         - LM317 / linear regulator resistor divider\n"
                "  calc filter <rc|rl|lc> <R> <C/L>                          - Filter cutoff frequency (-3dB fc) & bandwidth\n"
                "  calc reactance <freq_Hz> <C_or_L>                         - Capacitive (Xc) or Inductive (Xl) reactance\n"
                "  calc power <Voltage> <Current|Resistance>                 - Ohm's Law and Power Dissipation\n"
            )

        tool = args[0].lower()
        if tool in ("track", "trace", "width"):
            if len(args) < 2:
                raise ValueError("Usage: calc track <current_A> [temp_rise_C] [copper_oz] [length_mm]")
            i = parse_eng_unit(args[1])
            dt = parse_eng_unit(args[2]) if len(args) > 2 else 10.0
            oz = parse_eng_unit(args[3]) if len(args) > 3 else 1.0
            l = parse_eng_unit(args[4]) if len(args) > 4 else 50.0
            res = KiCadCalculator.calculate_track_width(i, temp_rise_deg_c=dt, copper_thickness_oz=oz, track_length_mm=l)
            return (
                f"[bold cyan]=== IPC-2152 / IPC-2221 PCB Track Width Calculator ===[/bold cyan]\n"
                f"  Design Parameters: Current={res['current_amp']}A, ΔT={res['temp_rise_c']}°C, Copper={res['copper_oz']}oz, Length={res['length_mm']}mm\n"
                f"  * [bold green]External Track Width[/bold green]: {res['external_width_mm']:.3f} mm ({res['external_width_mil']:.1f} mil)\n"
                f"  * [bold green]Internal Track Width[/bold green]: {res['internal_width_mm']:.3f} mm ({res['internal_width_mil']:.1f} mil)\n"
                f"  * Track DC Resistance   : {res['resistance_ohm']:.4f} Ω\n"
                f"  * Voltage Drop          : {res['voltage_drop_v']:.4f} V\n"
                f"  * Power Loss            : {res['power_loss_w']:.4f} W"
            )

        elif tool == "via":
            drill = parse_eng_unit(args[1]) if len(args) > 1 else 0.3
            pad = parse_eng_unit(args[2]) if len(args) > 2 else 0.6
            thick = parse_eng_unit(args[3]) if len(args) > 3 else 1.6
            res = KiCadCalculator.calculate_via_parasitics(drill_dia_mm=drill, pad_dia_mm=pad, pcb_thickness_mm=thick)
            return (
                f"[bold cyan]=== PCB Via Parasitics & Ampacity Calculator ===[/bold cyan]\n"
                f"  Via Geometry: Drill={res['drill_mm']:.2f}mm, Pad={res['pad_mm']:.2f}mm, PCB Thickness={res['pcb_thickness_mm']:.2f}mm\n"
                f"  * [bold green]DC Resistance[/bold green] : {res['resistance_mohm']:.2f} mΩ\n"
                f"  * [bold green]Inductance[/bold green]    : {res['inductance_nh']:.2f} nH\n"
                f"  * [bold green]Capacitance[/bold green]   : {res['capacitance_pf']:.2f} pF\n"
                f"  * [bold green]Max Current[/bold green]   : {res['max_current_amp']:.2f} A (@10°C rise)"
            )

        elif tool in ("microstrip", "rf", "transmission"):
            if len(args) < 3:
                raise ValueError("Usage: calc microstrip <width_mm> <height_mm> [Er]")
            w = parse_eng_unit(args[1])
            h = parse_eng_unit(args[2])
            er = parse_eng_unit(args[3]) if len(args) > 3 else 4.5
            res = KiCadCalculator.calculate_microstrip(w, h, substrate_er=er)
            return (
                f"[bold cyan]=== RF Microstrip Transmission Line Calculator ===[/bold cyan]\n"
                f"  Substrate: Width={res['width_mm']:.2f}mm, Dielectric Height={res['height_mm']:.2f}mm, Relative Permittivity Er={res['substrate_er']:.1f}\n"
                f"  * [bold green]Characteristic Impedance (Z0)[/bold green]: {res['z0_ohms']:.2f} Ω\n"
                f"  * [bold green]Effective Permittivity (Eff Er)[/bold green]: {res['effective_er']:.3f}\n"
                f"  * [bold green]Propagation Delay[/bold green]            : {res['delay_ps_per_mm']:.2f} ps/mm"
            )

        elif tool in ("555", "timer"):
            if len(args) < 4:
                raise ValueError("Usage: calc 555 <R1_ohms> <R2_ohms> <C_farads> (e.g. calc 555 10k 47k 100n)")
            r1 = parse_eng_unit(args[1])
            r2 = parse_eng_unit(args[2])
            c = parse_eng_unit(args[3])
            res = KiCadCalculator.calculate_555_timer(r1, r2, c)
            return (
                f"[bold cyan]=== Astable 555 Timer Calculator ===[/bold cyan]\n"
                f"  Components: R1={format_eng_unit(r1)}Ω, R2={format_eng_unit(r2)}Ω, C={format_eng_unit(c)}F\n"
                f"  * [bold green]Oscillation Frequency[/bold green]: {format_eng_unit(res['freq_hz'])}Hz\n"
                f"  * [bold green]Period[/bold green]               : {format_eng_unit(res['period_s'])}s\n"
                f"  * [bold green]Duty Cycle[/bold green]           : {res['duty_cycle_pct']:.2f} %\n"
                f"  * High Time (T_high) : {format_eng_unit(res['t_high_s'])}s\n"
                f"  * Low Time (T_low)   : {format_eng_unit(res['t_low_s'])}s"
            )

        elif tool in ("opamp", "op-amp", "gain"):
            if len(args) < 4:
                raise ValueError("Usage: calc opamp <inv|noninv> <R1_ohms> <Rf_ohms> (e.g. calc opamp noninv 10k 100k)")
            op_type = args[1]
            r1 = parse_eng_unit(args[2])
            rf = parse_eng_unit(args[3])
            res = KiCadCalculator.calculate_opamp_gain(op_type, r1, rf)
            return (
                f"[bold cyan]=== Op-Amp Closed Loop Gain Calculator ===[/bold cyan]\n"
                f"  Configuration: {res['circuit_type']} | R1={format_eng_unit(r1)}Ω, Rf={format_eng_unit(rf)}Ω\n"
                f"  * [bold green]Voltage Gain (Av)[/bold green]: {res['gain_v_per_v']:.3f} V/V\n"
                f"  * [bold green]Gain in dB[/bold green]       : {res['gain_db']:.2f} dB"
            )

        elif tool in ("divider", "vdiv"):
            if len(args) < 4:
                raise ValueError("Usage: calc divider <Vin_V> <R1_ohms> <R2_ohms> (e.g. calc divider 12V 10k 2.2k)")
            vin = parse_eng_unit(args[1])
            r1 = parse_eng_unit(args[2])
            r2 = parse_eng_unit(args[3])
            res = KiCadCalculator.calculate_voltage_divider(vin, r1, r2)
            return (
                f"[bold cyan]=== Voltage Divider Calculator ===[/bold cyan]\n"
                f"  Input: Vin={res['vin_v']}V | R1={format_eng_unit(r1)}Ω, R2={format_eng_unit(r2)}Ω\n"
                f"  * [bold green]Output Voltage (Vout)[/bold green]: {res['vout_v']:.4f} V\n"
                f"  * [bold green]Divider Ratio[/bold green]         : {res['ratio']:.4f} ({res['ratio']*100:.1f}%)\n"
                f"  * Bleeder Current     : {res['current_ma']:.3f} mA\n"
                f"  * Total Power Loss    : {res['total_power_mw']:.2f} mW (R1: {res['p_r1_mw']:.2f}mW, R2: {res['p_r2_mw']:.2f}mW)"
            )

        elif tool in ("regulator", "lm317"):
            if len(args) < 2:
                raise ValueError("Usage: calc regulator <Vout_V> [Vref_V] [R1_ohms]")
            vout = parse_eng_unit(args[1])
            vref = parse_eng_unit(args[2]) if len(args) > 2 else 1.25
            r1 = parse_eng_unit(args[3]) if len(args) > 3 else 240.0
            res = KiCadCalculator.calculate_regulator_divider(vout, vref=vref, r1_ohms=r1)
            return (
                f"[bold cyan]=== Linear Regulator Resistor Calculator (LM317 / AMS1117) ===[/bold cyan]\n"
                f"  Target Vout={res['target_vout']}V, Vref={res['vref_v']}V, R1={format_eng_unit(r1)}Ω\n"
                f"  * [bold green]Required Resistor R2[/bold green]: {format_eng_unit(res['r2_calculated_ohms'])}Ω"
            )

        elif tool in ("filter", "rc", "rl", "lc"):
            ftype = args[1] if tool == "filter" and len(args) > 1 else tool
            if ftype == "filter":
                ftype = "rc"
            r = parse_eng_unit(args[2]) if (tool == "filter" and len(args) > 2) else (parse_eng_unit(args[1]) if len(args) > 1 else 1000.0)
            c_or_l = parse_eng_unit(args[3]) if (tool == "filter" and len(args) > 3) else (parse_eng_unit(args[2]) if len(args) > 2 else 100e-9)
            if "rc" in ftype:
                res = KiCadCalculator.calculate_filter("rc", r_ohms=r, c_farads=c_or_l)
            elif "rl" in ftype:
                res = KiCadCalculator.calculate_filter("rl", r_ohms=r, l_henries=c_or_l)
            else:
                res = KiCadCalculator.calculate_filter("lc", l_henries=r, c_farads=c_or_l)
            lines = [f"[bold cyan]=== {res.get('filter_type', 'Filter')} Calculator ===[/bold cyan]"]
            if "cutoff_freq_hz" in res:
                lines.append(f"  * [bold green]Cutoff Frequency (-3dB fc)[/bold green]: {format_eng_unit(res['cutoff_freq_hz'])}Hz")
                lines.append(f"  * Time Constant (τ)            : {format_eng_unit(res['time_constant_s'])}s")
            elif "resonant_freq_hz" in res:
                lines.append(f"  * [bold green]Resonant Frequency (f0)[/bold green]    : {format_eng_unit(res['resonant_freq_hz'])}Hz")
                lines.append(f"  * Characteristic Impedance (Z0): {format_eng_unit(res['characteristic_z0_ohms'])}Ω")
            return "\n".join(lines)

        elif tool in ("reactance", "xc", "xl"):
            if len(args) < 3:
                raise ValueError("Usage: calc reactance <freq_Hz> <C_or_L> (e.g. calc reactance 1kHz 100nF, calc reactance 100kHz 10uH)")
            f = parse_eng_unit(args[1])
            val_str = args[2].lower()
            val = parse_eng_unit(val_str)
            is_cap = "f" in val_str or "c" in val_str or val < 1e-4
            if is_cap:
                res = KiCadCalculator.calculate_reactance(f, c_farads=val)
                return f"[bold cyan]Capacitive Reactance (Xc):[/bold cyan] [bold green]{format_eng_unit(res['xc_ohms'])}Ω[/bold green] @ {format_eng_unit(f)}Hz (C={format_eng_unit(val)}F)"
            else:
                res = KiCadCalculator.calculate_reactance(f, l_henries=val)
                return f"[bold cyan]Inductive Reactance (Xl):[/bold cyan] [bold green]{format_eng_unit(res['xl_ohms'])}Ω[/bold green] @ {format_eng_unit(f)}Hz (L={format_eng_unit(val)}H)"

        elif tool in ("power", "ohm"):
            if len(args) < 3:
                raise ValueError("Usage: calc power <Voltage_V> <Current_A_or_Resistance_Ohm>")
            v = parse_eng_unit(args[1])
            val_str = args[2].lower()
            val = parse_eng_unit(val_str)
            is_curr = "a" in val_str or "ma" in val_str or "ua" in val_str
            if is_curr:
                res = KiCadCalculator.calculate_power(v, current_a=val)
                return f"[bold cyan]Power Dissipation:[/bold cyan] [bold yellow]{format_eng_unit(res['power_w'])}W[/bold yellow] | Current={format_eng_unit(res['current_a'])}A, Resistance={format_eng_unit(res['resistance_ohm'])}Ω"
            else:
                res = KiCadCalculator.calculate_power(v, resistance_ohm=val)
                return f"[bold cyan]Power Dissipation:[/bold cyan] [bold yellow]{format_eng_unit(res['power_w'])}W[/bold yellow] | Current={format_eng_unit(res['current_a'])}A, Resistance={format_eng_unit(res['resistance_ohm'])}Ω"

        raise ValueError(f"Unknown calculator tool '{tool}'. Run 'calc' for options.")

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
