"""KiCad Component & Footprint Library for TerminusECE.

Provides standard SMD and THT footprints, pad geometries, physical dimensions,
pin pitch, and pinout definitions based on KiCad standard libraries.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PadSpec:
    """Pad geometry and location relative to component centroid in mm."""
    number: str
    name: str
    x_offset_mm: float
    y_offset_mm: float
    width_mm: float
    height_mm: float
    shape: str = "RECT"  # RECT, ROUND, OVAL, CIRCLE
    is_tht: bool = False
    drill_mm: float = 0.0


@dataclass
class FootprintSpec:
    """Footprint physical and geometric definition."""
    name: str
    category: str  # Passive, Semiconductor, IC, Connector, Module
    width_mm: float
    height_mm: float
    pad_count: int
    pads: List[PadSpec] = field(default_factory=list)
    description: str = ""
    is_smd: bool = True
    courtyard_clearance_mm: float = 0.25


class FootprintCatalog:
    """Standard KiCad footprint registry."""

    _catalog: Dict[str, FootprintSpec] = {}

    @classmethod
    def initialize(cls):
        if cls._catalog:
            return

        # 1. Standard SMD Passives (0402, 0603, 0805, 1206, 1210, 2010, 2512)
        cls._register_2pin_smd("0402", 1.0, 0.5, pad_w=0.6, pad_h=0.5, pad_spacing=0.9, cat="Passive", desc="SMD 0402 (1005 Metric)")
        cls._register_2pin_smd("0603", 1.6, 0.8, pad_w=0.8, pad_h=0.8, pad_spacing=1.6, cat="Passive", desc="SMD 0603 (1608 Metric)")
        cls._register_2pin_smd("0805", 2.0, 1.25, pad_w=1.0, pad_h=1.3, pad_spacing=2.0, cat="Passive", desc="SMD 0805 (2012 Metric)")
        cls._register_2pin_smd("1206", 3.2, 1.6, pad_w=1.1, pad_h=1.6, pad_spacing=3.2, cat="Passive", desc="SMD 1206 (3216 Metric)")
        cls._register_2pin_smd("1210", 3.2, 2.5, pad_w=1.1, pad_h=2.5, pad_spacing=3.2, cat="Passive", desc="SMD 1210 (3225 Metric)")
        cls._register_2pin_smd("2512", 6.4, 3.2, pad_w=1.3, pad_h=3.2, pad_spacing=6.4, cat="Passive", desc="SMD 2512 (6432 Metric)")

        # 2. Standard THT Passives
        cls._register_tht_resistor("Axial_D2.5mm_P7.62mm", w=7.62, h=2.5, pitch=7.62, drill=0.8, pad_dia=1.6, desc="THT Axial Resistor 1/4W 7.62mm pitch")
        cls._register_tht_resistor("Axial_D3.5mm_P10.16mm", w=10.16, h=3.5, pitch=10.16, drill=0.9, pad_dia=1.8, desc="THT Axial Resistor 1/2W 10.16mm pitch")
        cls._register_tht_cap("Radial_D6.3mm_P2.5mm", dia=6.3, pitch=2.5, drill=0.8, pad_dia=1.6, desc="THT Electrolytic Cap D6.3mm Pitch 2.5mm")
        cls._register_tht_cap("Radial_D8.0mm_P3.5mm", dia=8.0, pitch=3.5, drill=0.8, pad_dia=1.6, desc="THT Electrolytic Cap D8.0mm Pitch 3.5mm")

        # 3. Diodes & LEDs
        cls._register_2pin_smd("SOD-123", 2.7, 1.6, pad_w=0.9, pad_h=1.2, pad_spacing=3.3, cat="Diode", desc="SMD Diode SOD-123")
        cls._register_2pin_smd("SMA", 4.3, 2.6, pad_w=1.5, pad_h=1.7, pad_spacing=4.6, cat="Diode", desc="SMD Diode DO-214AC (SMA)")
        cls._register_2pin_smd("SMB", 4.6, 3.6, pad_w=2.1, pad_h=2.2, pad_spacing=4.8, cat="Diode", desc="SMD Diode DO-214AA (SMB)")
        cls._register_2pin_smd("LED_0805", 2.0, 1.25, pad_w=1.0, pad_h=1.3, pad_spacing=2.0, cat="LED", desc="SMD LED 0805")
        cls._register_tht_resistor("LED_D3.0mm_P2.54mm", w=3.0, h=3.0, pitch=2.54, drill=0.8, pad_dia=1.6, desc="THT LED 3mm Pitch 2.54mm")
        cls._register_tht_resistor("LED_D5.0mm_P2.54mm", w=5.0, h=5.0, pitch=2.54, drill=0.8, pad_dia=1.6, desc="THT LED 5mm Pitch 2.54mm")

        # 4. Transistors & Small Regulators (SOT-23, SOT-223, TO-92, TO-220)
        cls._register_sot23()
        cls._register_sot223()
        cls._register_to92()
        cls._register_to220()

        # 5. ICs (SOIC, DIP, QFP, QFN)
        cls._register_soic(8, "SOIC-8", 4.9, 3.9, pitch=1.27)
        cls._register_soic(14, "SOIC-14", 8.65, 3.9, pitch=1.27)
        cls._register_soic(16, "SOIC-16", 9.9, 3.9, pitch=1.27)

        cls._register_dip(8, "DIP-8", 9.6, 7.62, pitch=2.54, row_spacing=7.62)
        cls._register_dip(14, "DIP-14", 19.2, 7.62, pitch=2.54, row_spacing=7.62)
        cls._register_dip(16, "DIP-16", 20.0, 7.62, pitch=2.54, row_spacing=7.62)
        cls._register_dip(28, "DIP-28", 35.6, 15.24, pitch=2.54, row_spacing=15.24)

        cls._register_qfp(32, "QFP-32", 7.0, 7.0, pitch=0.8)
        cls._register_qfn(32, "QFN-32", 5.0, 5.0, pitch=0.5)

        # 6. Connectors & Headers
        cls._register_pin_header(2, 1, "PinHeader_1x2_P2.54mm", pitch=2.54)
        cls._register_pin_header(4, 1, "PinHeader_1x4_P2.54mm", pitch=2.54)
        cls._register_pin_header(6, 1, "PinHeader_1x6_P2.54mm", pitch=2.54)
        cls._register_pin_header(8, 2, "PinHeader_2x4_P2.54mm", pitch=2.54)
        cls._register_pin_header(10, 2, "PinHeader_2x5_P2.54mm", pitch=2.54)
        cls._register_screw_terminal(2, "ScrewTerminal_1x2_P5.08mm", pitch=5.08)
        cls._register_usb_c()

    @classmethod
    def get_footprint(cls, name: str) -> Optional[FootprintSpec]:
        cls.initialize()
        clean = name.strip()
        # Direct match
        if clean in cls._catalog:
            return cls._catalog[clean]
        # Case-insensitive match
        for k, v in cls._catalog.items():
            if k.lower() == clean.lower():
                return v
        # Keyword heuristic lookup
        for k, v in cls._catalog.items():
            if clean.lower() in k.lower():
                return v
        # Default fallback
        return cls._catalog.get("0805")

    @classmethod
    def list_footprints(cls) -> List[FootprintSpec]:
        cls.initialize()
        return list(cls._catalog.values())

    @classmethod
    def _register_2pin_smd(cls, name: str, w: float, h: float, pad_w: float, pad_h: float, pad_spacing: float, cat: str, desc: str):
        half_sp = pad_spacing / 2.0
        pads = [
            PadSpec(number="1", name="1", x_offset_mm=-half_sp, y_offset_mm=0.0, width_mm=pad_w, height_mm=pad_h, shape="RECT"),
            PadSpec(number="2", name="2", x_offset_mm=half_sp, y_offset_mm=0.0, width_mm=pad_w, height_mm=pad_h, shape="RECT"),
        ]
        cls._catalog[name] = FootprintSpec(name=name, category=cat, width_mm=w, height_mm=h, pad_count=2, pads=pads, description=desc, is_smd=True)

    @classmethod
    def _register_tht_resistor(cls, name: str, w: float, h: float, pitch: float, drill: float, pad_dia: float, desc: str):
        half_p = pitch / 2.0
        pads = [
            PadSpec(number="1", name="1", x_offset_mm=-half_p, y_offset_mm=0.0, width_mm=pad_dia, height_mm=pad_dia, shape="CIRCLE", is_tht=True, drill_mm=drill),
            PadSpec(number="2", name="2", x_offset_mm=half_p, y_offset_mm=0.0, width_mm=pad_dia, height_mm=pad_dia, shape="CIRCLE", is_tht=True, drill_mm=drill),
        ]
        cls._catalog[name] = FootprintSpec(name=name, category="Passive", width_mm=w, height_mm=h, pad_count=2, pads=pads, description=desc, is_smd=False)

    @classmethod
    def _register_tht_cap(cls, name: str, dia: float, pitch: float, drill: float, pad_dia: float, desc: str):
        half_p = pitch / 2.0
        pads = [
            PadSpec(number="1", name="+", x_offset_mm=-half_p, y_offset_mm=0.0, width_mm=pad_dia, height_mm=pad_dia, shape="RECT", is_tht=True, drill_mm=drill),
            PadSpec(number="2", name="-", x_offset_mm=half_p, y_offset_mm=0.0, width_mm=pad_dia, height_mm=pad_dia, shape="CIRCLE", is_tht=True, drill_mm=drill),
        ]
        cls._catalog[name] = FootprintSpec(name=name, category="Passive", width_mm=dia, height_mm=dia, pad_count=2, pads=pads, description=desc, is_smd=False)

    @classmethod
    def _register_sot23(cls):
        pads = [
            PadSpec(number="1", name="B", x_offset_mm=-0.95, y_offset_mm=-1.0, width_mm=0.6, height_mm=0.8, shape="RECT"),
            PadSpec(number="2", name="E", x_offset_mm=0.95, y_offset_mm=-1.0, width_mm=0.6, height_mm=0.8, shape="RECT"),
            PadSpec(number="3", name="C", x_offset_mm=0.0, y_offset_mm=1.0, width_mm=0.6, height_mm=0.8, shape="RECT"),
        ]
        cls._catalog["SOT-23"] = FootprintSpec(name="SOT-23", category="Semiconductor", width_mm=2.9, height_mm=1.3, pad_count=3, pads=pads, description="SMD Transistor SOT-23 (TO-236AB)", is_smd=True)

    @classmethod
    def _register_sot223(cls):
        pads = [
            PadSpec(number="1", name="1", x_offset_mm=-2.3, y_offset_mm=-3.1, width_mm=0.95, height_mm=1.5, shape="RECT"),
            PadSpec(number="2", name="2", x_offset_mm=0.0, y_offset_mm=-3.1, width_mm=0.95, height_mm=1.5, shape="RECT"),
            PadSpec(number="3", name="3", x_offset_mm=2.3, y_offset_mm=-3.1, width_mm=0.95, height_mm=1.5, shape="RECT"),
            PadSpec(number="4", name="TAB", x_offset_mm=0.0, y_offset_mm=3.1, width_mm=3.25, height_mm=1.5, shape="RECT"),
        ]
        cls._catalog["SOT-223"] = FootprintSpec(name="SOT-223", category="Semiconductor", width_mm=6.5, height_mm=3.5, pad_count=4, pads=pads, description="SMD Regulator/Transistor SOT-223", is_smd=True)

    @classmethod
    def _register_to92(cls):
        pads = [
            PadSpec(number="1", name="1", x_offset_mm=-1.27, y_offset_mm=0.0, width_mm=1.4, height_mm=1.4, shape="CIRCLE", is_tht=True, drill_mm=0.8),
            PadSpec(number="2", name="2", x_offset_mm=0.0, y_offset_mm=1.27, width_mm=1.4, height_mm=1.4, shape="CIRCLE", is_tht=True, drill_mm=0.8),
            PadSpec(number="3", name="3", x_offset_mm=1.27, y_offset_mm=0.0, width_mm=1.4, height_mm=1.4, shape="CIRCLE", is_tht=True, drill_mm=0.8),
        ]
        cls._catalog["TO-92"] = FootprintSpec(name="TO-92", category="Semiconductor", width_mm=4.8, height_mm=3.8, pad_count=3, pads=pads, description="THT Transistor TO-92 Inline/Staggered", is_smd=False)

    @classmethod
    def _register_to220(cls):
        pads = [
            PadSpec(number="1", name="IN", x_offset_mm=-2.54, y_offset_mm=0.0, width_mm=1.8, height_mm=1.8, shape="RECT", is_tht=True, drill_mm=1.0),
            PadSpec(number="2", name="GND", x_offset_mm=0.0, y_offset_mm=0.0, width_mm=1.8, height_mm=1.8, shape="CIRCLE", is_tht=True, drill_mm=1.0),
            PadSpec(number="3", name="OUT", x_offset_mm=2.54, y_offset_mm=0.0, width_mm=1.8, height_mm=1.8, shape="CIRCLE", is_tht=True, drill_mm=1.0),
        ]
        cls._catalog["TO-220"] = FootprintSpec(name="TO-220", category="Semiconductor", width_mm=10.0, height_mm=4.5, pad_count=3, pads=pads, description="THT Power Package TO-220-3", is_smd=False)

    @classmethod
    def _register_soic(cls, count: int, name: str, w: float, h: float, pitch: float):
        pads = []
        n_side = count // 2
        y_span = (n_side - 1) * pitch
        y_start = -y_span / 2.0
        x_off = 2.7  # pad center to center / 2

        # Left side pins (1 .. n_side)
        for i in range(n_side):
            pnum = str(i + 1)
            pads.append(PadSpec(number=pnum, name=pnum, x_offset_mm=-x_off, y_offset_mm=y_start + i * pitch, width_mm=1.5, height_mm=0.6, shape="RECT"))

        # Right side pins (count .. n_side+1)
        for i in range(n_side):
            pnum = str(count - i)
            pads.append(PadSpec(number=pnum, name=pnum, x_offset_mm=x_off, y_offset_mm=y_start + i * pitch, width_mm=1.5, height_mm=0.6, shape="RECT"))

        cls._catalog[name] = FootprintSpec(name=name, category="IC", width_mm=w, height_mm=h, pad_count=count, pads=pads, description=f"SOIC-{count} Package", is_smd=True)

    @classmethod
    def _register_dip(cls, count: int, name: str, w: float, h: float, pitch: float, row_spacing: float):
        pads = []
        n_side = count // 2
        y_span = (n_side - 1) * pitch
        y_start = -y_span / 2.0
        x_off = row_spacing / 2.0

        for i in range(n_side):
            pnum = str(i + 1)
            shape = "RECT" if i == 0 else "CIRCLE"
            pads.append(PadSpec(number=pnum, name=pnum, x_offset_mm=-x_off, y_offset_mm=y_start + i * pitch, width_mm=1.6, height_mm=1.6, shape=shape, is_tht=True, drill_mm=0.8))

        for i in range(n_side):
            pnum = str(count - i)
            pads.append(PadSpec(number=pnum, name=pnum, x_offset_mm=x_off, y_offset_mm=y_start + i * pitch, width_mm=1.6, height_mm=1.6, shape="CIRCLE", is_tht=True, drill_mm=0.8))

        cls._catalog[name] = FootprintSpec(name=name, category="IC", width_mm=w, height_mm=h, pad_count=count, pads=pads, description=f"DIP-{count} Package (Through-Hole)", is_smd=False)

    @classmethod
    def _register_qfp(cls, count: int, name: str, w: float, h: float, pitch: float):
        pads = []
        side_pins = count // 4
        span = (side_pins - 1) * pitch
        start = -span / 2.0
        r_off = (w / 2.0) + 0.5

        # Side 1 (Bottom): 1 .. side_pins
        for i in range(side_pins):
            pads.append(PadSpec(number=str(i + 1), name=str(i + 1), x_offset_mm=start + i * pitch, y_offset_mm=-r_off, width_mm=0.4, height_mm=1.2, shape="RECT"))
        # Side 2 (Right): side_pins+1 .. 2*side_pins
        for i in range(side_pins):
            p = side_pins + 1 + i
            pads.append(PadSpec(number=str(p), name=str(p), x_offset_mm=r_off, y_offset_mm=start + i * pitch, width_mm=1.2, height_mm=0.4, shape="RECT"))
        # Side 3 (Top): 2*side_pins+1 .. 3*side_pins
        for i in range(side_pins):
            p = 2 * side_pins + 1 + i
            pads.append(PadSpec(number=str(p), name=str(p), x_offset_mm=-start - i * pitch, y_offset_mm=r_off, width_mm=0.4, height_mm=1.2, shape="RECT"))
        # Side 4 (Left): 3*side_pins+1 .. count
        for i in range(side_pins):
            p = 3 * side_pins + 1 + i
            pads.append(PadSpec(number=str(p), name=str(p), x_offset_mm=-r_off, y_offset_mm=-start - i * pitch, width_mm=1.2, height_mm=0.4, shape="RECT"))

        cls._catalog[name] = FootprintSpec(name=name, category="IC", width_mm=w, height_mm=h, pad_count=count, pads=pads, description=f"QFP-{count} Surface Mount Package", is_smd=True)

    @classmethod
    def _register_qfn(cls, count: int, name: str, w: float, h: float, pitch: float):
        pads = []
        side_pins = count // 4
        span = (side_pins - 1) * pitch
        start = -span / 2.0
        r_off = (w / 2.0) - 0.2

        for i in range(side_pins):
            pads.append(PadSpec(number=str(i + 1), name=str(i + 1), x_offset_mm=start + i * pitch, y_offset_mm=-r_off, width_mm=0.25, height_mm=0.6, shape="RECT"))
        for i in range(side_pins):
            p = side_pins + 1 + i
            pads.append(PadSpec(number=str(p), name=str(p), x_offset_mm=r_off, y_offset_mm=start + i * pitch, width_mm=0.6, height_mm=0.25, shape="RECT"))
        for i in range(side_pins):
            p = 2 * side_pins + 1 + i
            pads.append(PadSpec(number=str(p), name=str(p), x_offset_mm=-start - i * pitch, y_offset_mm=r_off, width_mm=0.25, height_mm=0.6, shape="RECT"))
        for i in range(side_pins):
            p = 3 * side_pins + 1 + i
            pads.append(PadSpec(number=str(p), name=str(p), x_offset_mm=-r_off, y_offset_mm=-start - i * pitch, width_mm=0.6, height_mm=0.25, shape="RECT"))

        # Thermal pad (EP)
        pads.append(PadSpec(number="EP", name="EP", x_offset_mm=0.0, y_offset_mm=0.0, width_mm=w*0.6, height_mm=h*0.6, shape="RECT"))

        cls._catalog[name] = FootprintSpec(name=name, category="IC", width_mm=w, height_mm=h, pad_count=count + 1, pads=pads, description=f"QFN-{count} Surface Mount Package with Exposed Pad", is_smd=True)

    @classmethod
    def _register_pin_header(cls, total_pins: int, rows: int, name: str, pitch: float = 2.54):
        cols = total_pins // rows
        pads = []
        x_start = -((cols - 1) * pitch) / 2.0
        y_start = -((rows - 1) * pitch) / 2.0

        pnum = 1
        for r in range(rows):
            for c in range(cols):
                shape = "RECT" if pnum == 1 else "CIRCLE"
                pads.append(PadSpec(
                    number=str(pnum), name=str(pnum),
                    x_offset_mm=x_start + c * pitch, y_offset_mm=y_start + r * pitch,
                    width_mm=1.6, height_mm=1.6, shape=shape, is_tht=True, drill_mm=1.0
                ))
                pnum += 1

        cls._catalog[name] = FootprintSpec(
            name=name, category="Connector",
            width_mm=cols * pitch, height_mm=rows * pitch,
            pad_count=total_pins, pads=pads,
            description=f"Pin Header {rows}x{cols} Pitch {pitch}mm", is_smd=False
        )

    @classmethod
    def _register_screw_terminal(cls, pins: int, name: str, pitch: float = 5.08):
        pads = []
        x_start = -((pins - 1) * pitch) / 2.0
        for i in range(pins):
            pads.append(PadSpec(
                number=str(i + 1), name=str(i + 1),
                x_offset_mm=x_start + i * pitch, y_offset_mm=0.0,
                width_mm=2.5, height_mm=2.5, shape="CIRCLE", is_tht=True, drill_mm=1.4
            ))
        cls._catalog[name] = FootprintSpec(name=name, category="Connector", width_mm=pins * pitch, height_mm=8.0, pad_count=pins, pads=pads, description=f"Screw Terminal Block {pins}-pin Pitch {pitch}mm", is_smd=False)

    @classmethod
    def _register_usb_c(cls):
        pads = [
            PadSpec(number="A1", name="GND", x_offset_mm=-3.2, y_offset_mm=0.0, width_mm=0.6, height_mm=1.2, shape="RECT"),
            PadSpec(number="A4", name="VBUS", x_offset_mm=-2.4, y_offset_mm=0.0, width_mm=0.6, height_mm=1.2, shape="RECT"),
            PadSpec(number="A5", name="CC1", x_offset_mm=-1.6, y_offset_mm=0.0, width_mm=0.3, height_mm=1.2, shape="RECT"),
            PadSpec(number="A6", name="DP1", x_offset_mm=-0.8, y_offset_mm=0.0, width_mm=0.3, height_mm=1.2, shape="RECT"),
            PadSpec(number="A7", name="DN1", x_offset_mm=0.0, y_offset_mm=0.0, width_mm=0.3, height_mm=1.2, shape="RECT"),
            PadSpec(number="A8", name="SBU1", x_offset_mm=0.8, y_offset_mm=0.0, width_mm=0.3, height_mm=1.2, shape="RECT"),
            PadSpec(number="A9", name="VBUS", x_offset_mm=1.6, y_offset_mm=0.0, width_mm=0.6, height_mm=1.2, shape="RECT"),
            PadSpec(number="A12", name="GND", x_offset_mm=2.4, y_offset_mm=0.0, width_mm=0.6, height_mm=1.2, shape="RECT"),
            PadSpec(number="SH1", name="SHIELD", x_offset_mm=-4.3, y_offset_mm=-2.0, width_mm=1.2, height_mm=1.8, shape="RECT", is_tht=True, drill_mm=0.8),
            PadSpec(number="SH2", name="SHIELD", x_offset_mm=4.3, y_offset_mm=-2.0, width_mm=1.2, height_mm=1.8, shape="RECT", is_tht=True, drill_mm=0.8),
        ]
        cls._catalog["USB_C_Receptacle"] = FootprintSpec(name="USB_C_Receptacle", category="Connector", width_mm=8.94, height_mm=7.35, pad_count=len(pads), pads=pads, description="USB Type-C 16-pin / 24-pin Receptacle", is_smd=True)
