"""Classified Component Library and Model Catalog for Circuit Engine.
Inspired by LTspice standard library, ltwiki.org components, and standard SPICE semiconductor models.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from .components import (
    Component,
    Resistor,
    Capacitor,
    Inductor,
    CoupledInductors,
    VoltageSource,
    CurrentSource,
    Diode,
    BJT,
    MOSFET,
    JFET,
    VoltageControlledSwitch,
    CurrentControlledSwitch,
    VCVS,
    VCCS,
    CCVS,
    CCCS,
    BehavioralSource,
    OpAmpModel,
    SubcircuitDefinition,
    SubcircuitInstance,
)


@dataclass
class CircuitComponentSpec:
    """Metadata specification for a catalog circuit component / model."""
    name: str
    category: str
    symbol_prefix: str
    description: str
    pins: List[str]
    default_params: Dict[str, Any]
    spice_model_type: str
    model_statement: Optional[str] = None
    subcircuit_template: Optional[SubcircuitDefinition] = None
    example_usage: str = ""
    tags: List[str] = field(default_factory=list)


class CircuitComponentCatalog:
    """Comprehensive Classified LTspice Component & Subcircuit Library Browser."""

    def __init__(self):
        self._specs: Dict[str, CircuitComponentSpec] = {}
        self._subcircuits: Dict[str, SubcircuitDefinition] = {}
        self._init_catalog()

    def _register(self, spec: CircuitComponentSpec) -> None:
        self._specs[spec.name.upper()] = spec
        if spec.subcircuit_template:
            self._subcircuits[spec.subcircuit_template.name.upper()] = spec.subcircuit_template

    def _init_catalog(self) -> None:
        # ==========================================
        # 1. PASSIVES & LINEAR ELEMENTS
        # ==========================================
        self._register(CircuitComponentSpec(
            name="RESISTOR",
            category="Passives",
            symbol_prefix="R",
            description="Linear Resistor (standard, precision, or power)",
            pins=["p (+)", "n (-)"],
            default_params={"resistance": "1k", "tc1": 0.0, "tc2": 0.0},
            spice_model_type="R",
            example_usage="add R1 in out 10k",
            tags=["resistor", "passive", "impedance", "divider"]
        ))

        self._register(CircuitComponentSpec(
            name="POTENTIOMETER",
            category="Passives",
            symbol_prefix="POT",
            description="3-Terminal Adjustable Potentiometer / Trimmer",
            pins=["term1", "wiper", "term2"],
            default_params={"total_r": "10k", "ratio": 0.5},
            spice_model_type="SUBCKT",
            example_usage="add POT1 vcc out gnd 10k ratio=0.7",
            tags=["potentiometer", "variable", "trimmer", "passive"]
        ))

        self._register(CircuitComponentSpec(
            name="CAPACITOR",
            category="Passives",
            symbol_prefix="C",
            description="Linear Capacitor (Ceramic, Film, Electrolytic)",
            pins=["p (+)", "n (-)"],
            default_params={"capacitance": "100n", "ic": 0.0, "rser": 0.0},
            spice_model_type="C",
            example_usage="add C1 out 0 10u ic=0",
            tags=["capacitor", "passive", "filtering", "bypass"]
        ))

        self._register(CircuitComponentSpec(
            name="INDUCTOR",
            category="Passives",
            symbol_prefix="L",
            description="Linear Inductor with series winding resistance",
            pins=["p (+)", "n (-)"],
            default_params={"inductance": "10m", "ic": 0.0, "rser": 0.01},
            spice_model_type="L",
            example_usage="add L1 in out 100u",
            tags=["inductor", "choke", "magnetic", "filter"]
        ))

        self._register(CircuitComponentSpec(
            name="TRANSFORMER",
            category="Passives",
            symbol_prefix="K",
            description="Coupled Inductor Transformer (Mutual Inductance Coupling)",
            pins=["pri_p", "pri_n", "sec_p", "sec_n"],
            default_params={"L1": "10m", "L2": "10m", "k": 0.999},
            spice_model_type="K",
            example_usage="add K1 L1 L2 0.99",
            tags=["transformer", "coupled", "mutual", "isolation"]
        ))

        # ==========================================
        # 2. DIODES & RECTIFIERS
        # ==========================================
        self._register(CircuitComponentSpec(
            name="1N4148",
            category="Diodes",
            symbol_prefix="D",
            description="High-speed small-signal silicon switching diode (100V, 200mA, 4ns)",
            pins=["anode", "cathode"],
            default_params={"is_sat": 2.52e-9, "n_ideal": 1.752, "rs": 0.568, "cjo": 4e-12, "bv": 100.0},
            spice_model_type="D",
            model_statement=".model 1N4148 D(Is=2.52n N=1.752 Rs=0.568 Cjo=4p Bv=100 Ibv=100u)",
            example_usage="add D1 in out 1N4148",
            tags=["diode", "switching", "fast", "rectifier", "1n4148"]
        ))

        self._register(CircuitComponentSpec(
            name="1N4007",
            category="Diodes",
            symbol_prefix="D",
            description="General purpose power rectifier diode (1000V, 1A)",
            pins=["anode", "cathode"],
            default_params={"is_sat": 7.02e-9, "n_ideal": 1.8, "rs": 0.034, "cjo": 15e-12, "bv": 1000.0},
            spice_model_type="D",
            model_statement=".model 1N4007 D(Is=7.02n N=1.8 Rs=0.034 Cjo=15p Bv=1000)",
            example_usage="add D1 ac_in dc_out 1N4007",
            tags=["diode", "power", "rectifier", "mains", "1n4007"]
        ))

        self._register(CircuitComponentSpec(
            name="1N5819",
            category="Diodes",
            symbol_prefix="D",
            description="Schottky barrier rectifier diode (40V, 1A, Low Forward Drop)",
            pins=["anode", "cathode"],
            default_params={"is_sat": 3.16e-7, "n_ideal": 1.05, "rs": 0.05, "cjo": 110e-12, "bv": 40.0},
            spice_model_type="D",
            model_statement=".model 1N5819 D(Is=316n N=1.05 Rs=0.05 Cjo=110p Bv=40)",
            example_usage="add D1 in out 1N5819",
            tags=["schottky", "low-drop", "fast", "diode", "1n5819"]
        ))

        self._register(CircuitComponentSpec(
            name="BZX84C5V1",
            category="Diodes",
            symbol_prefix="D",
            description="5.1V Zener voltage regulator diode (350mW)",
            pins=["anode", "cathode"],
            default_params={"is_sat": 1e-14, "n_ideal": 1.0, "rs": 5.0, "bv": 5.1, "ibv": 5e-3},
            spice_model_type="D",
            model_statement=".model BZX84C5V1 D(Is=10f N=1.0 Rs=5.0 Bv=5.1 Ibv=5m)",
            example_usage="add DZ1 0 v_reg BZX84C5V1",
            tags=["zener", "voltage-regulator", "clamp", "5.1v"]
        ))

        self._register(CircuitComponentSpec(
            name="LED_RED",
            category="Diodes",
            symbol_prefix="D",
            description="Standard Red Light Emitting Diode (Vf ~ 1.8V, 20mA)",
            pins=["anode", "cathode"],
            default_params={"is_sat": 1e-19, "n_ideal": 2.0, "rs": 2.0, "bv": 5.0},
            spice_model_type="D",
            model_statement=".model LED_RED D(Is=1e-19 N=2.0 Rs=2.0 Bv=5)",
            example_usage="add DLED out 0 LED_RED",
            tags=["led", "optical", "indicator", "red"]
        ))

        # ==========================================
        # 3. BIPOLAR JUNCTION TRANSISTORS (BJT)
        # ==========================================
        self._register(CircuitComponentSpec(
            name="2N2222",
            category="Transistors",
            symbol_prefix="Q",
            description="NPN General Purpose switching & amplifier transistor (40V, 800mA, hFE~100-300)",
            pins=["collector", "base", "emitter"],
            default_params={"bjt_type": "NPN", "beta_f": 200.0, "is_sat": 1e-14, "vaf": 100.0, "vbe_on": 0.7},
            spice_model_type="Q",
            model_statement=".model 2N2222 NPN(Is=14.34f Xti=3 Eg=1.11 Vaf=74.03 Bf=255.9)",
            example_usage="add Q1 c b e 2N2222",
            tags=["npn", "bjt", "transistor", "amplifier", "2n2222"]
        ))

        self._register(CircuitComponentSpec(
            name="2N3904",
            category="Transistors",
            symbol_prefix="Q",
            description="NPN Small Signal Audio & Switching Transistor (40V, 200mA)",
            pins=["collector", "base", "emitter"],
            default_params={"bjt_type": "NPN", "beta_f": 300.0, "is_sat": 1.4e-14, "vaf": 74.0, "vbe_on": 0.68},
            spice_model_type="Q",
            model_statement=".model 2N3904 NPN(Is=1e-14 Bf=300 Vaf=100)",
            example_usage="add Q1 c b e 2N3904",
            tags=["npn", "bjt", "small-signal", "audio", "2n3904"]
        ))

        self._register(CircuitComponentSpec(
            name="2N3906",
            category="Transistors",
            symbol_prefix="Q",
            description="PNP Small Signal Audio & Switching Transistor (-40V, -200mA)",
            pins=["collector", "base", "emitter"],
            default_params={"bjt_type": "PNP", "beta_f": 200.0, "is_sat": 1.4e-14, "vaf": 50.0, "vbe_on": 0.68},
            spice_model_type="Q",
            model_statement=".model 2N3906 PNP(Is=1.4e-14 Bf=200 Vaf=50)",
            example_usage="add Q1 c b e 2N3906",
            tags=["pnp", "bjt", "transistor", "complementary", "2n3906"]
        ))

        self._register(CircuitComponentSpec(
            name="BC547",
            category="Transistors",
            symbol_prefix="Q",
            description="NPN High-Gain Low-Noise European Audio Transistor (45V, 100mA, hFE~400)",
            pins=["collector", "base", "emitter"],
            default_params={"bjt_type": "NPN", "beta_f": 400.0, "is_sat": 1.8e-14, "vaf": 80.0, "vbe_on": 0.66},
            spice_model_type="Q",
            model_statement=".model BC547 NPN(Is=1.8e-14 Bf=400 Vaf=80)",
            example_usage="add Q1 c b e BC547",
            tags=["npn", "bc547", "low-noise", "audio"]
        ))

        # ==========================================
        # 4. FIELD EFFECT TRANSISTORS (MOSFET / JFET)
        # ==========================================
        self._register(CircuitComponentSpec(
            name="IRF540N",
            category="Transistors",
            symbol_prefix="M",
            description="N-Channel Power MOSFET (100V, 33A, Rds(on)=44mOhm, HEXFET)",
            pins=["drain", "gate", "source", "body"],
            default_params={"mos_type": "NMOS", "vto": 3.5, "kp": 0.05, "w": 1e-3, "l": 1e-6, "rd": 0.044, "rs": 0.01},
            spice_model_type="M",
            model_statement=".model IRF540N NMOS(Vto=3.5 Kp=0.05 Rd=0.044 Rs=0.01 Lambda=0.01)",
            example_usage="add M1 d g s s IRF540N",
            tags=["nmos", "power-mosfet", "switch", "motor-drive", "irf540n"]
        ))

        self._register(CircuitComponentSpec(
            name="2N7000",
            category="Transistors",
            symbol_prefix="M",
            description="N-Channel Small Signal MOSFET (60V, 200mA, Logic-Level Gate)",
            pins=["drain", "gate", "source", "body"],
            default_params={"mos_type": "NMOS", "vto": 2.1, "kp": 0.012, "w": 1e-4, "l": 1e-6, "rd": 1.2, "rs": 0.5},
            spice_model_type="M",
            model_statement=".model 2N7000 NMOS(Vto=2.1 Kp=0.012 Rd=1.2 Rs=0.5)",
            example_usage="add M1 d g s s 2N7000",
            tags=["nmos", "logic-level", "2n7000", "switching"]
        ))

        self._register(CircuitComponentSpec(
            name="IRF9540",
            category="Transistors",
            symbol_prefix="M",
            description="P-Channel Power MOSFET (-100V, -19A, Rds(on)=117mOhm)",
            pins=["drain", "gate", "source", "body"],
            default_params={"mos_type": "PMOS", "vto": -3.5, "kp": 0.04, "w": 1e-3, "l": 1e-6, "rd": 0.117, "rs": 0.01},
            spice_model_type="M",
            model_statement=".model IRF9540 PMOS(Vto=-3.5 Kp=0.04 Rd=0.117 Rs=0.01)",
            example_usage="add M1 d g s s IRF9540",
            tags=["pmos", "power-mosfet", "high-side", "irf9540"]
        ))

        self._register(CircuitComponentSpec(
            name="2N5457",
            category="Transistors",
            symbol_prefix="J",
            description="N-Channel JFET General Purpose Audio & Pre-amp (25V, Idss=1-5mA)",
            pins=["drain", "gate", "source"],
            default_params={"jfet_type": "NJFET", "vto": -2.0, "beta": 1.2e-3, "lambda_param": 0.01},
            spice_model_type="J",
            model_statement=".model 2N5457 NJF(Vto=-2.0 Beta=1.2m Lambda=0.01)",
            example_usage="add J1 d g s 2N5457",
            tags=["jfet", "njfet", "preamp", "high-impedance", "2n5457"]
        ))

        # ==========================================
        # 5. OP-AMPS & ANALOG INTEGRATED CIRCUITS
        # ==========================================
        self._register(CircuitComponentSpec(
            name="LM741",
            category="Integrated Circuits",
            symbol_prefix="U",
            description="Classic General-Purpose Operational Amplifier (1MHz GBW, Avol=200k, Slew=0.5V/us)",
            pins=["out", "in- (inv)", "in+ (noninv)", "V+", "V-"],
            default_params={"avol": 200000.0, "gbw": 1e6, "rin": 2e6, "rout": 75.0, "slew_rate": 0.5e6},
            spice_model_type="SUBCKT",
            example_usage="add U1 out in_minus in_plus vcc vee LM741",
            tags=["opamp", "analog", "amplifier", "lm741", "ic"]
        ))

        self._register(CircuitComponentSpec(
            name="TL072",
            category="Integrated Circuits",
            symbol_prefix="U",
            description="Dual Low-Noise JFET-Input Audio Op-Amp (3MHz GBW, Slew=13V/us, High Zin)",
            pins=["out", "in- (inv)", "in+ (noninv)", "V+", "V-"],
            default_params={"avol": 200000.0, "gbw": 3e6, "rin": 1e12, "rout": 50.0, "slew_rate": 13e6},
            spice_model_type="SUBCKT",
            example_usage="add U1 out in_minus in_plus vcc vee TL072",
            tags=["opamp", "jfet-input", "low-noise", "audio", "tl072"]
        ))

        self._register(CircuitComponentSpec(
            name="LM358",
            category="Integrated Circuits",
            symbol_prefix="U",
            description="Dual Single-Supply Operational Amplifier (Ground-Sensing Input, 1MHz)",
            pins=["out", "in- (inv)", "in+ (noninv)", "V+", "V-"],
            default_params={"avol": 100000.0, "gbw": 1e6, "rin": 1e6, "rout": 100.0, "slew_rate": 0.6e6},
            spice_model_type="SUBCKT",
            example_usage="add U1 out in_minus in_plus vcc 0 LM358",
            tags=["opamp", "single-supply", "lm358", "sensor"]
        ))

        self._register(CircuitComponentSpec(
            name="LM311",
            category="Integrated Circuits",
            symbol_prefix="U",
            description="High-Speed Voltage Comparator with Open-Collector Output (200ns)",
            pins=["out", "in- (inv)", "in+ (noninv)", "V+", "V-"],
            default_params={"avol": 200000.0, "rin": 1e7, "rout": 10.0},
            spice_model_type="SUBCKT",
            example_usage="add U1 out in_minus in_plus vcc 0 LM311",
            tags=["comparator", "open-collector", "detector", "lm311"]
        ))

        self._register(CircuitComponentSpec(
            name="NE555",
            category="Integrated Circuits",
            symbol_prefix="U",
            description="Precision Timer IC (Astable / Monostable Oscillator)",
            pins=["GND (1)", "TRIG (2)", "OUT (3)", "RESET (4)", "CTRL (5)", "THRES (6)", "DISCH (7)", "VCC (8)"],
            default_params={"vcc": 5.0},
            spice_model_type="SUBCKT",
            example_usage="add U1 0 trig out vcc ctrl thres disch vcc NE555",
            tags=["timer", "555", "oscillator", "pwm", "pulse"]
        ))

        self._register(CircuitComponentSpec(
            name="LM7805",
            category="Integrated Circuits",
            symbol_prefix="U",
            description="Fixed +5V Linear Voltage Regulator (1.5A max output, Thermal Protection)",
            pins=["in", "gnd", "out"],
            default_params={"v_out": 5.0, "drop_out": 2.0, "i_max": 1.5},
            spice_model_type="SUBCKT",
            example_usage="add U1 raw_in 0 v5_out LM7805",
            tags=["voltage-regulator", "linear", "5v", "lm7805", "power"]
        ))

        # ==========================================
        # 6. SOURCES & SIGNAL GENERATORS
        # ==========================================
        self._register(CircuitComponentSpec(
            name="VDC",
            category="Sources",
            symbol_prefix="V",
            description="DC Independent Voltage Source (Power Rail or Bias)",
            pins=["p (+)", "n (-)"],
            default_params={"dc": "5V"},
            spice_model_type="V",
            example_usage="add V1 vcc 0 5V",
            tags=["voltage", "dc", "power", "source", "battery"]
        ))

        self._register(CircuitComponentSpec(
            name="VSINE",
            category="Sources",
            symbol_prefix="V",
            description="Sinusoidal AC Signal Generator with Offset, Amplitude, Freq, Phase",
            pins=["p (+)", "n (-)"],
            default_params={"offset": 0.0, "amplitude": 1.0, "freq": 1000.0, "delay": 0.0, "phase": 0.0},
            spice_model_type="V",
            example_usage="add V1 in 0 SINE(0 1 1k)",
            tags=["ac", "sine", "oscillator", "generator", "audio"]
        ))

        self._register(CircuitComponentSpec(
            name="VPULSE",
            category="Sources",
            symbol_prefix="V",
            description="Pulse Waveform Generator (Square, Triangle, Ramp, Clock)",
            pins=["p (+)", "n (-)"],
            default_params={"v1": 0.0, "v2": 5.0, "td": 0.0, "tr": "1n", "tf": "1n", "ton": "0.5m", "period": "1m"},
            spice_model_type="V",
            example_usage="add V1 clk 0 PULSE(0 5 0 1n 1n 0.5m 1m)",
            tags=["pulse", "square-wave", "clock", "pwm", "generator"]
        ))

        self._register(CircuitComponentSpec(
            name="VPWL",
            category="Sources",
            symbol_prefix="V",
            description="Piecewise Linear Arbitrary Waveform Voltage Source",
            pins=["p (+)", "n (-)"],
            default_params={"points": [(0.0, 0.0), (1e-3, 5.0), (2e-3, 2.5), (3e-3, 0.0)]},
            spice_model_type="V",
            example_usage="add V1 in 0 PWL(0 0 1m 5 2m 2.5 3m 0)",
            tags=["pwl", "arbitrary", "waveform", "table"]
        ))

        self._register(CircuitComponentSpec(
            name="BEHAVIORAL_SOURCE",
            category="Sources",
            symbol_prefix="B",
            description="Nonlinear Arbitrary Behavioral Source (V or I = mathematical expression of nets)",
            pins=["p (+)", "n (-)"],
            default_params={"source_type": "V", "expression": "V(in1)*V(in2)"},
            spice_model_type="B",
            example_usage="add B1 out 0 V=V(in)*2 + sin(2*pi*1000*time)",
            tags=["behavioral", "b-source", "math", "multiplier", "nonlinear"]
        ))

        # ==========================================
        # 7. SWITCHES & CONTROLLED ELEMENTS
        # ==========================================
        self._register(CircuitComponentSpec(
            name="SW_VOLTAGE",
            category="Switches",
            symbol_prefix="S",
            description="Voltage-Controlled Switch (smooth switching Ron/Roff based on control voltage)",
            pins=["sw+ (1)", "sw- (2)", "ctrl+ (3)", "ctrl- (4)"],
            default_params={"ron": 1.0, "roff": 1e6, "von": 1.0, "voff": 0.0},
            spice_model_type="S",
            example_usage="add S1 in out ctrl 0 SW_MODEL",
            tags=["switch", "relay", "controlled", "analog-switch"]
        ))

        # ==========================================
        # 8. PRE-BUILT CIRCUITS & SUBCIRCUITS
        # ==========================================
        self._build_standard_subcircuits()

    def _build_standard_subcircuits(self) -> None:
        """Registers composite ready-to-simulate subcircuits and modular building blocks."""
        # 1. Full-Wave Bridge Rectifier (4 Diodes)
        bridge = SubcircuitDefinition("BRIDGE_RECTIFIER", ["AC1", "AC2", "DC_POS", "DC_NEG"])
        bridge.add_component(Diode("D1", "AC1", "DC_POS", "1N4007"))
        bridge.add_component(Diode("D2", "DC_NEG", "AC1", "1N4007"))
        bridge.add_component(Diode("D3", "AC2", "DC_POS", "1N4007"))
        bridge.add_component(Diode("D4", "DC_NEG", "AC2", "1N4007"))
        self._subcircuits["BRIDGE_RECTIFIER"] = bridge
        self._register(CircuitComponentSpec(
            name="BRIDGE_RECTIFIER",
            category="Subcircuits",
            symbol_prefix="X",
            description="Full-Wave 4-Diode Bridge Rectifier Module (AC to DC)",
            pins=["AC1", "AC2", "DC+", "DC-"],
            default_params={},
            spice_model_type="SUBCKT",
            subcircuit_template=bridge,
            example_usage="add X1 ac1 ac2 dc_pos 0 BRIDGE_RECTIFIER",
            tags=["bridge", "rectifier", "power-supply", "ac-dc"]
        ))

        # 2. Sallen-Key 2nd-Order Low-Pass Filter
        sk_lpf = SubcircuitDefinition("SALLEN_KEY_LPF", ["IN", "OUT", "VCC", "VEE", "GND"])
        sk_lpf.add_component(Resistor("R1", "IN", "N1", 10000.0))
        sk_lpf.add_component(Resistor("R2", "N1", "N2", 10000.0))
        sk_lpf.add_component(Capacitor("C1", "N1", "OUT", 10e-9))
        sk_lpf.add_component(Capacitor("C2", "N2", "GND", 10e-9))
        sk_lpf.add_component(VCVS("E_OPAMP", "OUT", "GND", "N2", "OUT", 100000.0))
        self._subcircuits["SALLEN_KEY_LPF"] = sk_lpf
        self._register(CircuitComponentSpec(
            name="SALLEN_KEY_LPF",
            category="Subcircuits",
            symbol_prefix="X",
            description="2nd-Order Sallen-Key Active Low-Pass Filter (Fc ~ 1.59 kHz, Butterworth)",
            pins=["IN", "OUT", "VCC", "VEE", "GND"],
            default_params={"r1": "10k", "r2": "10k", "c1": "10n", "c2": "10n"},
            spice_model_type="SUBCKT",
            subcircuit_template=sk_lpf,
            example_usage="add X1 in out vcc vee 0 SALLEN_KEY_LPF",
            tags=["filter", "active-filter", "sallen-key", "lowpass", "audio"]
        ))

        # 3. Non-Inverting Op-Amp Gain Stage
        opamp_amp = SubcircuitDefinition("OPAMP_NONINV_AMP", ["IN", "OUT", "VCC", "VEE", "GND"])
        opamp_amp.add_component(Resistor("RIN", "N_INV", "GND", 1000.0))
        opamp_amp.add_component(Resistor("RFB", "OUT", "N_INV", 9000.0))  # Gain = 1 + 9k/1k = 10 (20dB)
        opamp_amp.add_component(VCVS("E1", "OUT", "GND", "IN", "N_INV", 100000.0))
        self._subcircuits["OPAMP_NONINV_AMP"] = opamp_amp
        self._register(CircuitComponentSpec(
            name="OPAMP_NONINV_AMP",
            category="Subcircuits",
            symbol_prefix="X",
            description="Non-Inverting Op-Amp Amplifier (Gain = 1 + Rf/Rin = 10x / +20dB)",
            pins=["IN", "OUT", "VCC", "VEE", "GND"],
            default_params={"gain": 10.0, "rin": "1k", "rfb": "9k"},
            spice_model_type="SUBCKT",
            subcircuit_template=opamp_amp,
            example_usage="add X1 audio_in audio_out vcc vee 0 OPAMP_NONINV_AMP",
            tags=["amplifier", "opamp", "gain", "noninverting", "20db"]
        ))

    def get_all_categories(self) -> List[str]:
        """Returns list of all component categories."""
        cats = set(s.category for s in self._specs.values())
        return sorted(list(cats))

    def list_components_by_category(self, category: Optional[str] = None) -> List[CircuitComponentSpec]:
        """Lists components filtered by category or all."""
        if not category or category.lower() in ("all", "*"):
            return list(self._specs.values())
        cat_lower = category.lower()
        return [s for s in self._specs.values() if s.category.lower() == cat_lower]

    def search_components(self, query: str) -> List[CircuitComponentSpec]:
        """Performs search across name, description, category, and tags."""
        q = query.lower().strip()
        results: List[CircuitComponentSpec] = []
        for spec in self._specs.values():
            if (
                q in spec.name.lower()
                or q in spec.description.lower()
                or q in spec.category.lower()
                or any(q in t for t in spec.tags)
            ):
                results.append(spec)
        return results

    def get_spec(self, name: str) -> Optional[CircuitComponentSpec]:
        """Retrieves specification for component or model name."""
        return self._specs.get(name.upper())

    def get_subcircuit(self, name: str) -> Optional[SubcircuitDefinition]:
        """Retrieves subcircuit definition template."""
        return self._subcircuits.get(name.upper())


# Global singleton catalog instance
circuit_catalog = CircuitComponentCatalog()
