"""Custom Component Model and Microcontroller (.lib, .mod, .txt) Importer.
Parses user-defined device characteristics, SPICE macromodels, and MCU hardware specifications.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from Engines.Circuit.catalog import circuit_catalog, CircuitComponentSpec
from Engines.Circuit.components import (
    Component, Resistor, Capacitor, Inductor, VoltageSource, CurrentSource,
    Diode, BJT, MOSFET, JFET, VoltageControlledSwitch, VCVS, SubcircuitDefinition
)
from Engines.Circuit.netlist_parser import CircuitParser
from CORE.common_math import parse_eng_unit


@dataclass
class MCUSpecification:
    """Hardware characteristic specification parsed from .lib / .txt file."""
    device_name: str = "GENERIC_MCU"
    architecture: str = "RISC_DSP_32"
    clock_freq_hz: float = 60e6
    supply_voltage: float = 3.3
    flash_bytes: int = 65536
    sram_bytes: int = 65536
    adc_resolution_bits: int = 12
    adc_input_range: Tuple[float, float] = (0.0, 3.3)
    adc_channels: int = 16
    epwm_channels: int = 8
    gpio_count: int = 32
    timers: int = 4
    interrupt_sources: List[str] = field(default_factory=list)
    raw_properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.device_name

    @name.setter
    def name(self, val: str) -> None:
        self.device_name = val

    @property
    def clock_freq_mhz(self) -> float:
        return self.clock_freq_hz / 1e6

    @clock_freq_mhz.setter
    def clock_freq_mhz(self, val: float) -> None:
        self.clock_freq_hz = float(val) * 1e6

    @property
    def flash_kb(self) -> int:
        return int(self.flash_bytes / 1024)

    @flash_kb.setter
    def flash_kb(self, val: int) -> None:
        self.flash_bytes = int(val) * 1024

    @property
    def sram_kb(self) -> int:
        return int(self.sram_bytes / 1024)

    @sram_kb.setter
    def sram_kb(self, val: int) -> None:
        self.sram_bytes = int(val) * 1024

    def summary(self) -> str:
        lines = [
            f"[bold cyan]=== Microcontroller Specification: {self.device_name} ===[/bold cyan]",
            f"Architecture      : {self.architecture}",
            f"Clock Frequency   : {self.clock_freq_hz / 1e6:.1f} MHz",
            f"Supply Voltage    : {self.supply_voltage:.2f} V",
            f"Flash Memory      : {self.flash_bytes / 1024:.0f} KB",
            f"SRAM Memory       : {self.sram_bytes / 1024:.0f} KB",
            f"ADC Module        : {self.adc_channels} Channels @ {self.adc_resolution_bits}-bit (Range: {self.adc_input_range[0]}V - {self.adc_input_range[1]}V)",
            f"ePWM Channels     : {self.epwm_channels} High-Resolution PWM outputs",
            f"GPIO Pins         : {self.gpio_count} Digital I/O lines",
            f"Hardware Timers   : {self.timers}",
        ]
        return "\n".join(lines)


class CustomModelImporter:
    """Parses and registers SPICE `.lib`, `.mod`, and MCU hardware `.txt` files."""

    @classmethod
    def import_spice_library(cls, file_path_or_text: str, target_catalog: Optional[CircuitComponentCatalog] = None) -> Tuple[int, int, Dict[str, Any]]:
        """Parses SPICE .lib / .mod / .txt file with .model and .subckt definitions into CircuitComponentCatalog."""
        cat = target_catalog or circuit_catalog
        content = ""
        src_name = "custom_lib"
        if os.path.exists(file_path_or_text):
            src_name = Path(file_path_or_text).name
            with open(file_path_or_text, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        else:
            content = file_path_or_text

        imported_models: List[str] = []
        imported_subckts: List[str] = []

        current_subckt: Optional[SubcircuitDefinition] = None
        lines = content.splitlines()

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("*"):
                continue

            # Strip comments
            if ";" in line:
                line = line.split(";", 1)[0].strip()

            tokens = line.split()
            if not tokens:
                continue
            first_tok = tokens[0].upper()

            # 1. Subcircuit start: .SUBCKT name pin1 pin2 ...
            if first_tok == ".SUBCKT":
                if len(tokens) >= 3:
                    s_name = tokens[1].upper()
                    s_pins = tokens[2:]
                    current_subckt = SubcircuitDefinition(s_name, s_pins)
                continue

            # 2. Subcircuit end: .ENDS
            elif first_tok == ".ENDS":
                if current_subckt:
                    spec = CircuitComponentSpec(
                        name=current_subckt.name,
                        category="Custom / Subcircuits",
                        symbol_prefix="X",
                        description=f"Imported Subcircuit model from {src_name}",
                        pins=current_subckt.pins,
                        default_params={},
                        spice_model_type="SUBCKT",
                        subcircuit_template=current_subckt,
                        example_usage=f"add X1 {' '.join(current_subckt.pins)} {current_subckt.name}",
                        tags=["custom", "subcircuit", "imported", current_subckt.name.lower()]
                    )
                    cat._register(spec)
                    imported_subckts.append(current_subckt.name)
                    current_subckt = None
                continue

            # 3. Model statement: .MODEL name type(param=val ...)
            elif first_tok == ".MODEL":
                if len(tokens) >= 3:
                    m_name = tokens[1].upper()
                    m_def = " ".join(tokens[2:])
                    m_type = tokens[2].upper().split("(")[0]
                    cat_name = "Custom / Diodes" if m_type in ("D", "DIODE") else (
                        "Custom / Transistors" if m_type in ("NPN", "PNP", "NMOS", "PMOS", "NJF", "PJF") else "Custom / Models"
                    )
                    spec = CircuitComponentSpec(
                        name=m_name,
                        category=cat_name,
                        symbol_prefix=m_type[0],
                        description=f"Imported SPICE model ({m_type}) from {src_name}",
                        pins=["anode", "cathode"] if m_type == "D" else ["pin1", "pin2", "pin3"],
                        default_params={},
                        spice_model_type=m_type,
                        model_statement=line,
                        example_usage=f"add {m_type[0]}1 node1 node2 {m_name}",
                        tags=["custom", "model", "imported", m_name.lower()]
                    )
                    cat._register(spec)
                    imported_models.append(m_name)
                continue

            # If inside subcircuit, add child component
            if current_subckt:
                cname = tokens[0].upper()
                prefix = cname[0]
                if prefix == "R" and len(tokens) >= 4:
                    current_subckt.add_component(Resistor(cname, tokens[1], tokens[2], tokens[3]))
                elif prefix == "C" and len(tokens) >= 4:
                    current_subckt.add_component(Capacitor(cname, tokens[1], tokens[2], tokens[3]))
                elif prefix == "L" and len(tokens) >= 4:
                    current_subckt.add_component(Inductor(cname, tokens[1], tokens[2], tokens[3]))
                elif prefix == "D" and len(tokens) >= 3:
                    current_subckt.add_component(Diode(cname, tokens[1], tokens[2]))
                elif prefix == "Q" and len(tokens) >= 4:
                    current_subckt.add_component(BJT(cname, tokens[1], tokens[2], tokens[3]))
                elif prefix == "M" and len(tokens) >= 5:
                    current_subckt.add_component(MOSFET(cname, tokens[1], tokens[2], tokens[3], tokens[4]))
                elif prefix == "E" and len(tokens) >= 6:
                    gain = parse_eng_unit(tokens[5])
                    current_subckt.add_component(VCVS(cname, tokens[1], tokens[2], tokens[3], tokens[4], gain=gain))

        info = {
            "source": src_name,
            "models_count": len(imported_models),
            "subckts_count": len(imported_subckts),
            "models": imported_models,
            "subcircuits": imported_subckts
        }
        return len(imported_models), len(imported_subckts), info

    @classmethod
    def import_mcu_specification(cls, file_path_or_text: str) -> MCUSpecification:
        """Parses custom microcontroller characteristics from .lib / .txt definition file."""
        content = ""
        if os.path.exists(file_path_or_text):
            with open(file_path_or_text, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        else:
            content = file_path_or_text

        spec = MCUSpecification()
        lines = content.splitlines()

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("*") or line.startswith("//"):
                continue

            if "=" in line:
                k, v = [p.strip() for p in line.split("=", 1)]
                ku = k.upper()
                spec.raw_properties[ku] = v

                if "DEVICE" in ku or "NAME" in ku or "MCU" in ku:
                    spec.device_name = v
                elif "ARCH" in ku:
                    spec.architecture = v
                elif "CLOCK" in ku or "FREQ" in ku:
                    val = parse_eng_unit(v)
                    if "MHZ" in ku and not any(ch in v.upper() for ch in ("K", "M", "G")):
                        val *= 1e6
                    spec.clock_freq_hz = val
                elif "VOLTAGE" in ku or "VDD" in ku or "VCC" in ku:
                    spec.supply_voltage = parse_eng_unit(v)
                elif "FLASH" in ku:
                    val = parse_eng_unit(v)
                    if "KB" in ku and not any(ch in v.upper() for ch in ("K", "M", "G")):
                        val *= 1024
                    spec.flash_bytes = int(val)
                elif "SRAM" in ku or "RAM" in ku:
                    val = parse_eng_unit(v)
                    if "KB" in ku and not any(ch in v.upper() for ch in ("K", "M", "G")):
                        val *= 1024
                    spec.sram_bytes = int(val)
                elif "ADC_RES" in ku or "BITS" in ku:
                    spec.adc_resolution_bits = int(v)
                elif "ADC_CHAN" in ku or "ADC_MODULE" in ku:
                    spec.adc_channels = int(v)
                elif "ADC_RANGE" in ku:
                    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", v)
                    if len(nums) >= 2:
                        spec.adc_input_range = (float(nums[0]), float(nums[1]))
                elif "EPWM" in ku or "PWM" in ku:
                    spec.epwm_channels = int(v)
                elif "GPIO" in ku:
                    spec.gpio_count = int(v)
                elif "TIMER" in ku:
                    spec.timers = int(v)
                elif "INTERRUPT" in ku:
                    spec.interrupt_sources.extend([i.strip() for i in v.split(",")])

        return spec

    @classmethod
    def import_file(cls, file_path_or_text: str, target_catalog: Optional[CircuitComponentCatalog] = None) -> Tuple[int, int, Optional[MCUSpecification]]:
        """Unified importer that auto-detects SPICE models and MCU characteristics in a file."""
        content = ""
        if os.path.exists(file_path_or_text):
            with open(file_path_or_text, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        else:
            content = file_path_or_text

        m_count = 0
        s_count = 0
        mcu_spec: Optional[MCUSpecification] = None

        cu = content.upper()
        if ".MODEL" in cu or ".SUBCKT" in cu:
            m_count, s_count, _ = cls.import_spice_library(content, target_catalog)

        if "DEVICE" in cu or "CLOCK_FREQ" in cu or "ADC_RESOLUTION" in cu or "FLASH_KB" in cu:
            mcu_spec = cls.import_mcu_specification(content)

        return m_count, s_count, mcu_spec

    # Aliases
    import_spice_lib = import_spice_library
    import_mcu_spec = import_mcu_specification
