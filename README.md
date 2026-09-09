# Terminus

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Status: Production Ready](https://img.shields.io/badge/Status-Active%20Workspace-blue.svg)]()
[![Tests: 140+ Passing](https://img.shields.io/badge/Tests-142%20Passed-success.svg)]()

> **The Command-Driven, Terminal-Based Unified Engineering Workspace for Electrical and Computer Engineering.**

---

##  Overview

The traditional ECE software ecosystem is severely fragmented by bloated, heavy GUI applications—juggling **KiCad** for PCB design, **LTspice** for analog circuits, **Simulink** for dynamic systems, **MATLAB** for signal processing, **Xilinx Vivado** for digital logic, and **Arduino / Code Composer Studio** for embedded hardware.

**Terminus** unifies all six foundational engineering disciplines into a single, lightning-fast, keyboard-driven terminal environment. Operating on a strict philosophy: **Engineering is mathematics, and mathematics is best expressed with precision, clarity, and speed.**

```
       Terminus Unified Engineering Architecture
┌────────────────────────────────────────────────────────┐
│  KiCad EDA  │  LTspice  │  Simulink  │  MATLAB  │  Xilinx │
│  PCB Design │  Circuits │  Dynamics  │  DSP/AST │  Logic  │
└────────────────────────────────────────────────────────┘
│            Real-Hardware Embedded IDE (30+ MCUs)       │
├────────────────────────────────────────────────────────┤
│  500+ Multi-Terminal LAN IPC Network & Signal Bus      │
├────────────────────────────────────────────────────────┤
│  Paper-Style ASCII Schematics & 300 DPI PNG Previews   │
└────────────────────────────────────────────────────────┘
```

---

##  Key Capabilities & Subsystems

### 1.  KiCad EDA Schematic & PCB Layout Mode (`mode kicad`)
* **Single-Folder Project Management**: All design files are organized together in `Documents/Terminus Files/KiCad/<ProjectName>/`:
  * `<Project>.tkcad` — Terminus EDA Project Manifest & DRC constraints
  * `<Project>.tsch` — Schematic sheets, components, and pin interconnects
  * `<Project>.tpcb` — 2-layer PCB layout, tracks, vias, and pad coordinates
  * `<Project>.tbom` — Bill of Materials CSV
  * `<Project>-gerbers/` — Complete suite of RS-274X Gerber layers (`F.Cu`, `B.Cu`, `F.SilkS`, `B.SilkS`, `F.Mask`, `B.Mask`, `Edge.Cuts`) and Excellon drill files
* **Interactive CLI Workflow**: Add components, connect nets, place footprints on board canvas, and autoroute copper traces with text commands.
* **Paper-Style Standard Circuit Schematics**: Clean ASCII/Unicode circuit art following industry conventions (Sources & Inputs on the left, Grounds `⏚` at the bottom, Power rails `⏉` on top, Outputs on the right) with short observable symbols (`[ R1 10k ]`, `[ C1 100nF ]`, `[ U1 NE555 ]`).
* **Electrical Rules Check (ERC)**: Automatically detects floating pins, unconnected inputs, missing ground references, and conflicting rails.
* **Design Rules Check (DRC)**: Validates board edge clearances, minimum track widths, annular rings, and courtyard overlaps.
* **500+ Component & Footprint Library**: Full catalog of SMD passives (0201 to 2512), THT resistors/capacitors, SOD/SMA/SMB diodes, SOT-23/SOT-223/TO-92/TO-220 transistors, SOIC/DIP/QFP/QFN ICs, pin headers, screw terminals, and USB-C.

### 2.  Analog Devices LTspice Circuit Engine (`mode circuit`)
* **Modified Nodal Analysis (MNA)**: High-accuracy SPICE simulation engine with companion models for resistors, capacitors, inductors, diodes, BJTs, MOSFETs, and independent/dependent sources.
* **Full Simulation Directives**:
  * `.op` — DC Operating Point with Newton-Raphson iteration & Gmin stepping
  * `.dc <Src> <Start> <Stop> <Step>` — DC Parameter Sweep & transfer curves
  * `.ac <dec|oct|lin> <Points> <Fstart> <Fstop>` — AC Frequency Response & ASCII Bode diagrams
  * `.tran <Tstep> <Tstop>` — Transient time-domain simulation with non-linear companion integration
  * `.four <Freq> <Trace> <Tstop>` — Fourier Harmonic Analysis & Total Harmonic Distortion (THD %)
* **Design Rule & Topology Diagnostics**: Instant detection of floating nodes, shorted sources, singular matrix hazards, and missing ground references.
* **Interactive Probing**: Measure node voltages, branch currents, peak-to-peak, and RMS values with live ASCII waveform rendering.

### 3.  MathWorks Simulink Dynamic Systems Engine (`mode dynamic`)
* **17 Classified Block Libraries (100+ Blocks)**: Continuous, Discrete, Math, Discontinuities, Logic & Bit, Lookup Tables, Matrix Ops, Routing, Sources, Sinks, Signal Attributes, Dashboard, Verification, and Subsystems.
* **Multi-Algorithm Numerical ODE Solvers**: Runge-Kutta 4th Order (`rk4`), Forward Euler (`euler`), Heun 2nd Order (`heun`), and Adaptive Dormand-Prince (`ode45`).
* **Model Advisor Diagnostics**: Automatic pre-flight verification identifying direct-feedthrough algebraic loops, unconnected input ports, parameter bounds violations, and multi-rate transition issues.
* **Real-Time Parameter Tuning**: Tune gains, time constants, and limits on live running models without restarting simulation.

### 4.  MathWorks MATLAB & DSP Workbench (`mode numerical`)
* **Vectorized AST Mathematical Evaluator**: Evaluate matrix algebra, trigonometric expressions, and NumPy-backed linear algebra directly from the command bar.
* **DSP Signal Processing Suite**: Design Butterworth, Chebyshev, and FIR windowed digital filters; compute Power Spectral Density (PSD), Time-Frequency Spectrograms, and Fast Fourier Transforms (FFT).
* **Control Systems & Transforms**: Continuous and discrete transfer functions $H(s)$ and $H(z)$, Laplace transforms, Z-plane analysis, and step response.
* **Universal 300 DPI PNG Figure Exporter**: Render publication-quality technical plots with metric banners and view 24-bit Truecolor ANSI graphic previews directly in your terminal.

### 5.  AMD Xilinx Digital Logic & HDL Engine (`mode digital`)
* **Combinational & Sequential Simulation**: Logic gates (AND, OR, NOT, XOR, NAND, NOR, XNOR) and Flip-Flops (D-FF, JK-FF, T-FF).
* **Discrete-Event Simulation**: Cycle-accurate clock generation, signal propagation delays, and multi-channel ASCII timing diagrams.
* **Truth Table Generator**: Automatically evaluate boolean expressions and generate complete truth tables.

### 6.  Real-Hardware Embedded IDE & Microcontroller Workbench (`mode embedded`)
* **Auto USB COM Port Detection**: Scans physical USB buses for connected hardware signatures (FTDI, CH340, CP2102, Atmel, ST-Link, ESP-JTAG).
* **30+ Supported Board Profiles**: Arduino Uno/Nano/Mega, ESP32, ESP8266, STM32 BluePill/Nucleo, Raspberry Pi Pico, TI C2000, and RISC-V.
* **Toolchain Compiler & Firmware Uploader**: Seamless compilation via local toolchains (`avr-gcc`, `arduino-cli`) with Flash/SRAM usage analysis and direct hardware uploading over USB.
* **Live Bidirectional USB Serial Monitor**: Multi-threaded serial terminal streaming live sensor data, sending commands, and rendering real-time ASCII telemetry graphs.
* **Virtual Breadboard**: Simulate virtual LEDs, RGB modules, push buttons, potentiometers, DHT22 sensors, ultrasonic HC-SR04, servo motors, LCD 16x2, and SSD1306 OLED displays.

---

##  Built-in Engineering Calculators

Access engineering calculation tools in both **KiCad** and **LTspice / Circuit** modes via the `calc` command:

| Calculator | Command Syntax | Description |
| :--- | :--- | :--- |
| **PCB Track Width** | `calc track <I_amp> [dT] [oz] [len]` | IPC-2152 / IPC-2221 track width, DC resistance, voltage drop & power loss |
| **Via Parasitics** | `calc via <drill_mm> [pad_mm] [thick_mm]` | Via DC resistance, parasitic inductance ($L$), capacitance ($C$) & ampacity |
| **RF Microstrip** | `calc microstrip <W_mm> <H_mm> [Er]` | Characteristic impedance ($Z_0$) and propagation delay for transmission lines |
| **555 Timer** | `calc 555 <R1> <R2> <C>` | Astable 555 timer oscillation frequency, period, and duty cycle percentage |
| **Op-Amp Gain** | `calc opamp <inv\|noninv> <R1> <Rf>` | Closed-loop inverting or non-inverting voltage gain ($V/V$ and $\text{dB}$) |
| **Voltage Divider** | `calc divider <Vin> <R1> <R2>` | Output voltage, division ratio, bleeder current, and resistor dissipation |
| **Voltage Regulator**| `calc regulator <Vout> [Vref] [R1]` | LM317 / AMS1117 adjustable regulator feedback resistor $R_2$ calculation |
| **Filter Cutoff** | `calc filter <rc\|rl\|lc> <R> <C/L>` | $-3\text{dB}$ cutoff frequency ($f_c$), time constant ($\tau$), or resonant frequency |
| **Reactance** | `calc reactance <freq> <C_or_L>` | Capacitive reactance ($X_C = \frac{1}{2\pi f C}$) or Inductive reactance ($X_L = 2\pi f L$) |
| **Power & Ohm** | `calc power <Voltage> <Current\|Resistance>` | Ohm's Law and power dissipation calculations |

---

##  Centralized Storage & Custom Terminus File Extensions

Terminus maintains organized, mode-specific project folders inside `Documents/Terminus Files/`:

```
Documents/Terminus Files/
├── KiCad/          (*.tkcad, *.tsch, *.tpcb, *.tbom, *-gerbers/)
├── LTspice/        (*.tckt)
├── Simulink/       (*.tmdl)
├── MATLAB/         (*.tmat)
├── Xilinx/         (*.tdig)
├── Arduino_IDE/    (*.tembed)
└── Exports/        (*.csv, *.cir, *.asc, *.v, *.vhd, *.c, *.hex, *.m)
```

---

##  High-Capacity Multi-Terminal Session Networking

Terminus supports high-capacity multi-terminal networking connecting **500+ concurrent terminal sessions** across local and LAN devices:

* **Async Proactor Socket Engine**: Uses non-blocking asynchronous I/O supporting 10,000+ socket descriptors with zero select limits.
* **Real-Time Signal Streaming (`session link`)**: Live-stream simulated waveforms or physical sensor telemetry from one terminal directly into another terminal's math workspace.
* **Remote Command Execution (`session exec`)**: Remotely dispatch and execute commands on other terminal instances.
* **LAN Subnet Discovery**: Bind to `0.0.0.0` to collaborate seamlessly across multiple computers on the same network.

---

##  Command Cheat Sheet

### Switching Modes
```sh
mode kicad        # Switch to KiCad EDA Schematic & PCB Layout mode
mode circuit      # Switch to Analog Devices LTspice Circuit Simulation mode
mode dynamic      # Switch to MathWorks Simulink Dynamic Systems mode
mode numerical    # Switch to MathWorks MATLAB & DSP Workbench mode
mode digital      # Switch to AMD Xilinx Digital Logic & HDL mode
mode embedded     # Switch to Embedded Hardware IDE & Serial Monitor mode
```

### KiCad EDA Workflow
```sh
# Add components & connect pins
add R1 10k 0805
add C1 100nF 0603
add V1 5V PinHeader_1x2_P2.54mm
connect V1.1 | R1.1 | VCC
connect R1.2 | C1.1 | Net_Out
connect C1.2 | V1.2 | GND

# Verify & visualize
erc               # Run Electrical Rules Check
schematic         # Display paper-style standard ASCII schematic
probe R1          # Probe voltage, current, and power dissipation

# PCB layout & manufacturing
place R1 10 10 0
place C1 20 10 90
autoroute         # Auto-route copper tracks on PCB
pcb               # View 2D terminal PCB layout canvas
drc               # Run Design Rules Check
gerber            # Export complete RS-274X Gerber suite & drill files
bom               # Export Bill of Materials (.tbom)
file save MyPowerCircuit
```

### LTspice Circuit Workflow
```sh
add V1 10V ac=1
add R1 1k
add C1 1u
connect V1.p | R1.a
connect R1.b | C1.a | out
connect C1.b | V1.n | 0

check             # Run DRC topology check
schematic         # View paper-style circuit diagram
run .ac dec 10 1Hz 100kHz   # Run AC sweep & ASCII Bode plot
run .tran 1u 10m            # Run transient simulation
probe out         # Probe waveform metrics & ASCII plot
png circuit_bode.png        # Export high-res PNG & Truecolor preview
```

### Simulink Dynamic Systems Workflow
```sh
library Continuous
add step Step1
add tf Plant num=[1] den=[1, 2, 1]
add scope Scope1
connect Step1.0 Plant.0
connect Plant.0 Scope1.0

check             # Run Model Advisor diagnostics
sim 10 0.01 rk4   # Run ODE simulation
scope Scope1      # Display scope waveform plot
export simulink   # Export to MATLAB Simulink generator script
```

---

## 🛠️ Installation & Quick Start

### Requirements
* Python 3.9 or higher
* Recommended terminal: Windows Terminal, macOS Terminal, or Linux Bash with UTF-8 support

### Setup
```sh
# Clone repository
git clone https://github.com/ThotaAvinash07-87/TerminusECE.git
cd TerminusECE

# Install dependencies
pip install rich textual numpy scipy matplotlib pyserial

# Run interactive CLI shell
python Terminus.py

# Launch directly into KiCad mode
python Terminus.py --mode kicad

# Launch full-screen TUI
python Terminus.py --tui
```

### Running Automated Test Suite
```sh
python -m unittest discover -s tests
```

---

##  License

Terminus is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
