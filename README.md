# Terminus v1.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Status: Production Ready](https://img.shields.io/badge/Status-v1.0%20Release-blue.svg)]()
[![Tests: 148 Passing](https://img.shields.io/badge/Tests-148%20Passed-success.svg)]()

> **The Unified, Keyboard-Driven Terminal Workspace for Electrical and Computer Engineering.**

---

## 🌟 Overview

The traditional ECE software ecosystem is severely fragmented by bloated, heavy GUI applications—juggling **KiCad** for PCB design, **LTspice** for analog circuits, **Simulink** for dynamic systems, **MATLAB** for signal processing, **Xilinx Vivado** for digital logic, and **Arduino IDE** for embedded hardware.

**Terminus v1.0** unifies all six foundational engineering disciplines into a single, lightning-fast, keyboard-driven terminal environment. Operating on a strict philosophy: **Engineering is mathematics, and mathematics is best expressed with precision, clarity, and speed.**

```
       Terminus v1.0 Unified Engineering Architecture
┌────────────────────────────────────────────────────────┐
│  LTspice   │  KiCad EDA │  Simulink  │  MATLAB  │  Xilinx │
│  Circuits  │  PCB & Sch │  Dynamics  │  DSP/AST │  Logic  │
└────────────────────────────────────────────────────────┘
│           Arduino IDE Real-Hardware MCU Studio          │
├────────────────────────────────────────────────────────┤
│  60% Left Interactive Canvas / 40% Right Tech Dashboard │
├────────────────────────────────────────────────────────┤
│  Standard VS Code Height Integrated Command History    │
├────────────────────────────────────────────────────────┤
│  500+ Multi-Terminal LAN IPC Network & Signal Bus      │
├────────────────────────────────────────────────────────┤
│  High-Precision Vector PNGs & Technical Metrics Cards  │
└────────────────────────────────────────────────────────┘
```

---

## 🖥️ 60/40 Split Interactive Workspace

Every mode renders an authentic, framed engineering workspace:

```
┌─────────────────────────── [ LTspice / KiCad / Simulink Workspace ] ──────────────────────────┐
│ ◄────────────── 60% Left Interactive Canvas ──────────────►│◄────── 40% Right Dashboard ─────►│
│                                                            │ [LIVE GRAPH / TECHNICAL DATA]    │
│   +12V ──────► [ R R1:10k ] (12.0V, 1.2mA) ──► Node [out]  │ 12.0 ┌───────────·-·-            │
│                     │                                      │  8.0 │          /       \         │
│               [ C C1:100n ] (0.0V)                         │  0.0 └─────────/─────────\───     │
│                     │                                      │                                  │
│                    GND                                     │ [METRICS: Vpk=12V, Fs=10kHz]     │
│                                                            ├──────────────────────────────────┤
│   [Interactive: Click/Arrow keys to select & move blocks]  │ [COMMAND HISTORY & LOGS]         │
│   [Selected: R1 (10k, Pos: (12, 4)) -> 'move R1 15 4']     │ > add R1 10k 0805 [OK]           │
│   [Pins: 1 -> VCC, 2 -> out | Probed: 1.2mA, 14.4mW]       │ > connect R1.2 | C1.1 [OK]       │
├────────────────────────────────────────────────────────────┴──────────────────────────────────┤
│ [ VS Code Style Bottom Command History & Output Terminal ]                                    │
│ Terminus [KICAD] #1 > _ (Commands: add, connect, move, place, route, run, calc, png, etc.)    │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Layout Mapping by Subsystem:
* **LTspice (`mode circuit`)**: 60% Left is interactive movable schematic canvas (`[ R R1:10k ]`, `[ C C1:100n ]`, `[ U U1:LM358 ]`); 40% Right is live scope waveforms, Bode plots, DC operating point bar charts, and probed node data ($V, I, P$).
* **KiCad (`mode kicad`)**: 60% Left is 2D PCB board layout canvas with orthogonal copper tracks, vias, and silkscreen; 40% Right is real-time DRC violation list, layer stack status (`F.Cu`/`B.Cu`), and BOM table.
* **Simulink (`mode dynamic`)**: 60% Left is 2D dynamic block diagram canvas; 40% Right is multi-channel scope traces, model state vectors, and stability status.
* **MATLAB (`mode numerical`)**: 60% Left is live script code buffer with line numbers; 40% Right is continuous ASCII Braille plots, FFT spectrum, Welch PSD, and variable workspace table (`whos`).
* **Xilinx (`mode digital`)**: 60% Left is Verilog/HDL netlist buffer; 40% Right is multi-trace timing logic waveforms (0/1/Z), truth tables, and FPGA resource utilization.
* **Arduino IDE (`mode embedded`)**: 60% Left is embedded C/C++ sketch editor; 40% Right is real-time USB serial monitor stream, live sensor telemetry plotter, and flash memory metrics.
* **Bottom Panel**: Standard VS Code terminal height displaying scrolling, color-highlighted command history and error feeds.

---

## 📸 High-Precision Vector PNG & Technical Card Export Pipeline

Terminus provides publication-grade 300 DPI vector graphic exports directly saved into each mode's project folder (`Documents/Terminus Files/<Mode>/<Project>/`):

| Command | Output Description |
| :--- | :--- |
| `png graph [name.png]` | **Pure High-Precision Vector Graph**: High-res vector curves without banners or technical cards. |
| `png card [name.png]` | **Technical Report Metrics Card**: Dark-themed publication card detailing $V_{pk}, V_{rms}, V_{pp}, f_0, \text{THD} \%, \text{SNR dB}$. |
| `png pcb [name.png]` | **KiCad PCB Layout Artwork**: 2-layer board layout graphic with authentic solder mask green, `F.Cu` red, and `B.Cu` blue tracks. |
| `png [name.png]` | **Combined Engineering Figure**: Complete multi-subplot figure with top banner and technical metrics. |
| `preview <name.png>` | **In-Terminal Truecolor Preview**: 24-bit ANSI unicode half-block preview rendered directly in the terminal console. |

---

## 🧮 Built-in Engineering Calculators

Run calculations anytime in **KiCad** and **LTspice / Circuit** modes via the `calc` command:

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
| **Reactance** | `calc reactance <freq> <C_or_L>` | Capacitive reactance ($X_C$) or Inductive reactance ($X_L$) |
| **Power & Ohm** | `calc power <Voltage> <Current\|Resistance>` | Ohm's Law and power dissipation calculations |

---

## 📂 Centralized Industry-Standard Storage Structure

All projects and exported artifacts are structured cleanly in `Documents/Terminus Files/`:

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

## ⚡ Quick Start

```powershell
# Start Terminus Interactive REPL Shell
python Terminus.py

# Switch between modes
mode circuit       # Analog Devices LTspice
mode kicad         # KiCad EDA Schematic & PCB
mode dynamic       # MathWorks Simulink
mode numerical     # MathWorks MATLAB & DSP
mode digital       # AMD Xilinx Vivado HDL
mode embedded      # Arduino IDE & Real Hardware
```

---

## 🧪 Automated Test Suite

Terminus includes 148 automated unit and integration tests across all computational engines and UI renderers:

```powershell
python -m unittest discover -s tests
# 148 tests passed (100% OK)
```

---

## 📄 License

Terminus is open-source software licensed under the **MIT License**.
