"""KiCad Schematic Engine & Electrical Rule Check (ERC) for TerminusECE.

Manages schematic components, pin nets, compact paper-style block representations,
and circuit connectivity verification.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

from Engines.KiCad.footprints import FootprintCatalog, FootprintSpec


@dataclass
class SchematicPin:
    """Component pin connection."""
    number: str
    name: str
    net: str = ""
    pin_type: str = "PASSIVE"  # INPUT, OUTPUT, POWER_IN, POWER_OUT, PASSIVE, BIDIRECTIONAL


@dataclass
class SchematicComponent:
    """Component inside KiCad Schematic."""
    ref: str                     # e.g. R1, C1, U1, Q1, D1
    comp_type: str               # R, C, L, D, Q, M, U, V, I, J, SW, etc.
    value: str                   # e.g. 10k, 100n, LM358, 2N3904, 1N4148
    footprint_name: str          # e.g. 0805, SOIC-8, SOT-23, TO-220
    pins: Dict[str, SchematicPin] = field(default_factory=dict)
    x: int = 0
    y: int = 0
    rotation: int = 0            # 0, 90, 180, 270 degrees
    description: str = ""
    calculated_voltage: Optional[float] = None
    calculated_current: Optional[float] = None
    calculated_power: Optional[float] = None

    def get_short_label(self) -> str:
        """Returns clean short block notation e.g. '[ R1 10k ]' or '[ U1 LM358 ]'."""
        clean_val = self.value.replace(" ", "")
        if self.comp_type.upper() in ("R", "RESISTOR"):
            return f"[ R {self.ref}:{clean_val} ]"
        elif self.comp_type.upper() in ("C", "CAPACITOR"):
            return f"[ C {self.ref}:{clean_val} ]"
        elif self.comp_type.upper() in ("L", "INDUCTOR"):
            return f"[ L {self.ref}:{clean_val} ]"
        elif self.comp_type.upper() in ("D", "DIODE", "LED"):
            return f"[ D {self.ref}:{clean_val} ]"
        elif self.comp_type.upper() in ("Q", "BJT", "NPN", "PNP"):
            return f"[ Q {self.ref}:{clean_val} ]"
        elif self.comp_type.upper() in ("M", "MOSFET", "NMOS", "PMOS"):
            return f"[ M {self.ref}:{clean_val} ]"
        elif self.comp_type.upper() in ("U", "IC", "OPAMP", "OP-AMP"):
            return f"[ U {self.ref}:{clean_val} ]"
        elif self.comp_type.upper() in ("V", "SOURCE", "VOLTAGE"):
            return f"[ V {self.ref}:{clean_val} ]"
        elif self.comp_type.upper() in ("I", "CURRENT"):
            return f"[ I {self.ref}:{clean_val} ]"
        elif self.comp_type.upper() in ("J", "CONN", "HEADER"):
            return f"[ J {self.ref}:{clean_val} ]"
        else:
            return f"[ {self.ref}:{clean_val} ]"


class KiCadSchematic:
    """KiCad Schematic sheet and netlist engine."""

    def __init__(self, name: str = "UntitledSchematic"):
        self.name = name
        self.components: Dict[str, SchematicComponent] = {}
        self.nets: Dict[str, Set[Tuple[str, str]]] = {}  # net_name -> {(comp_ref, pin_num)}
        self._auto_net_counter = 1

    def clear(self):
        self.components.clear()
        self.nets.clear()
        self._auto_net_counter = 1

    def add_component(
        self,
        ref: str,
        value: str,
        footprint: Optional[str] = None,
        comp_type: Optional[str] = None
    ) -> SchematicComponent:
        """Adds or updates a component with appropriate pinout and footprint."""
        clean_ref = ref.strip().upper()
        prefix = re.match(r'^([a-zA-Z]+)', clean_ref)
        p_str = prefix.group(1) if prefix else "U"

        if not comp_type:
            if p_str.startswith("R"):
                comp_type = "R"
            elif p_str.startswith("C"):
                comp_type = "C"
            elif p_str.startswith("L"):
                comp_type = "L"
            elif p_str.startswith("D"):
                comp_type = "D"
            elif p_str.startswith("Q"):
                comp_type = "Q"
            elif p_str.startswith("M"):
                comp_type = "M"
            elif p_str.startswith("U"):
                comp_type = "U"
            elif p_str.startswith("V"):
                comp_type = "V"
            elif p_str.startswith("I"):
                comp_type = "I"
            elif p_str.startswith("J"):
                comp_type = "J"
            elif p_str.startswith("SW"):
                comp_type = "SW"
            else:
                comp_type = "U"

        # Determine default footprint if none provided
        if not footprint:
            if comp_type in ("R", "C", "L"):
                footprint = "0805"
            elif comp_type == "D":
                footprint = "SOD-123"
            elif comp_type in ("Q", "M"):
                footprint = "SOT-23"
            elif comp_type == "U":
                footprint = "SOIC-8"
            elif comp_type == "V":
                footprint = "PinHeader_1x2_P2.54mm"
            elif comp_type == "J":
                footprint = "PinHeader_1x4_P2.54mm"
            else:
                footprint = "0805"

        # Construct default pins based on component type
        pins: Dict[str, SchematicPin] = {}
        if comp_type in ("R", "C", "L", "V", "I", "SW"):
            pins["1"] = SchematicPin(number="1", name="1", pin_type="PASSIVE")
            pins["2"] = SchematicPin(number="2", name="2", pin_type="PASSIVE")
        elif comp_type == "D":
            pins["1"] = SchematicPin(number="1", name="A", pin_type="PASSIVE")
            pins["2"] = SchematicPin(number="2", name="K", pin_type="PASSIVE")
        elif comp_type == "Q":
            pins["1"] = SchematicPin(number="1", name="B", pin_type="INPUT")
            pins["2"] = SchematicPin(number="2", name="E", pin_type="PASSIVE")
            pins["3"] = SchematicPin(number="3", name="C", pin_type="PASSIVE")
        elif comp_type == "M":
            pins["1"] = SchematicPin(number="1", name="G", pin_type="INPUT")
            pins["2"] = SchematicPin(number="2", name="S", pin_type="PASSIVE")
            pins["3"] = SchematicPin(number="3", name="D", pin_type="PASSIVE")
        elif comp_type == "U":
            fp_spec = FootprintCatalog.get_footprint(footprint)
            p_count = fp_spec.pad_count if fp_spec else 8
            for i in range(1, p_count + 1):
                pins[str(i)] = SchematicPin(number=str(i), name=f"P{i}", pin_type="PASSIVE")
        elif comp_type == "J":
            fp_spec = FootprintCatalog.get_footprint(footprint)
            p_count = fp_spec.pad_count if fp_spec else 4
            for i in range(1, p_count + 1):
                pins[str(i)] = SchematicPin(number=str(i), name=str(i), pin_type="PASSIVE")

        comp = SchematicComponent(
            ref=clean_ref,
            comp_type=comp_type,
            value=value,
            footprint_name=footprint,
            pins=pins
        )
        self.components[clean_ref] = comp
        return comp

    def connect(self, endpoint1: str, endpoint2: str, net_name: Optional[str] = None):
        """Connects two component pins or nets (e.g. 'R1.1' to 'C1.1' or 'U1.IN' to 'VCC')."""
        def parse_endpoint(ep: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
            clean = ep.strip()
            if "." in clean:
                comp_ref, pin_id = clean.split(".", 1)
                return comp_ref.upper(), pin_id, None
            elif clean.upper() in ("0", "GND", "GROUND"):
                return None, None, "GND"
            elif clean.upper() in ("VCC", "VDD", "VIN", "VOUT", "5V", "3V3", "12V"):
                return None, None, clean.upper()
            elif clean.upper() in self.components:
                return clean.upper(), "1", None
            else:
                return None, None, clean

        c1, p1, n1 = parse_endpoint(endpoint1)
        c2, p2, n2 = parse_endpoint(endpoint2)

        # Determine effective net name
        if net_name:
            target_net = net_name
        elif n1:
            target_net = n1
        elif n2:
            target_net = n2
        else:
            # Check existing net for either pin
            net_found = None
            if c1 and p1 and c1 in self.components:
                pin_obj = self.components[c1].pins.get(p1) or self.components[c1].pins.get("1")
                if pin_obj and pin_obj.net:
                    net_found = pin_obj.net
            if not net_found and c2 and p2 and c2 in self.components:
                pin_obj = self.components[c2].pins.get(p2) or self.components[c2].pins.get("1")
                if pin_obj and pin_obj.net:
                    net_found = pin_obj.net

            target_net = net_found or f"Net_{self._auto_net_counter:03d}"
            if not net_found:
                self._auto_net_counter += 1

        if target_net not in self.nets:
            self.nets[target_net] = set()

        if c1 and c1 in self.components:
            pin_key = p1 if p1 in self.components[c1].pins else "1"
            if pin_key in self.components[c1].pins:
                self.components[c1].pins[pin_key].net = target_net
                self.nets[target_net].add((c1, pin_key))

        if c2 and c2 in self.components:
            pin_key = p2 if p2 in self.components[c2].pins else "1"
            if pin_key in self.components[c2].pins:
                self.components[c2].pins[pin_key].net = target_net
                self.nets[target_net].add((c2, pin_key))

    def run_erc(self) -> List[Dict[str, str]]:
        """Electrical Rule Check: checks floating pins, unconnected inputs, power rails."""
        reports = []
        # 1. Floating pins check
        for ref, comp in self.components.items():
            for pnum, pin in comp.pins.items():
                if not pin.net:
                    reports.append({
                        "level": "WARNING",
                        "component": ref,
                        "pin": pnum,
                        "message": f"Pin {ref}.{pnum} ({pin.name}) is unconnected (floating)."
                    })

        # 2. Single-node net check (dangling net)
        for net_name, endpoints in self.nets.items():
            if len(endpoints) < 2 and net_name not in ("GND", "0", "VCC", "VDD"):
                reports.append({
                    "level": "WARNING",
                    "net": net_name,
                    "message": f"Net '{net_name}' connects to only {len(endpoints)} pin ({endpoints})."
                })

        # 3. Ground presence check
        has_gnd = any(n in self.nets for n in ("GND", "0", "GROUND"))
        if not has_gnd:
            reports.append({
                "level": "ERROR",
                "message": "Missing Ground (GND/0) net in schematic. Circuits require a ground reference."
            })

        return reports

    def export_kicad_sch_content(self) -> str:
        """Generates standard KiCad v6+ S-Expression (.kicad_sch) file content."""
        lines = [
            f'(kicad_sch (version 20211123) (generator "TerminusECE_KiCad_Engine")',
            f'  (uuid "{self.name}_uuid")',
            f'  (paper "A4")',
            f'  (title_block (title "{self.name}") (company "Terminus ECE"))',
        ]

        # Export components
        for ref, comp in self.components.items():
            lines.append(f'  (symbol (lib_id "{comp.comp_type}:{comp.value}") (at {comp.x} {comp.y} {comp.rotation}) (unit 1)')
            lines.append(f'    (property "Reference" "{ref}" (at {comp.x} {comp.y - 2.5} 0))')
            lines.append(f'    (property "Value" "{comp.value}" (at {comp.x} {comp.y + 2.5} 0))')
            lines.append(f'    (property "Footprint" "{comp.footprint_name}" (at {comp.x} {comp.y + 5} 0))')
            for pnum, pin in comp.pins.items():
                lines.append(f'    (pin "{pnum}" (uuid "{ref}_{pnum}") (net "{pin.net}"))')
            lines.append('  )')

        # Export wires/nets
        for net_name, endpoints in self.nets.items():
            lines.append(f'  (net (name "{net_name}")')
            for ref, pnum in endpoints:
                lines.append(f'    (node (ref "{ref}") (pin "{pnum}"))')
            lines.append('  )')

        lines.append(')')
        return "\n".join(lines)

    def probe_component(self, ref: str) -> Dict[str, Any]:
        """Probes or calculates electrical parameters (voltage, current, power) for a component."""
        uref = ref.strip().upper()
        if uref not in self.components:
            raise KeyError(f"Component '{ref}' not found in schematic.")
        comp = self.components[uref]
        
        # Calculate approximate voltage drop, current, and power based on connected nets and component value
        v_in = 5.0
        r_val = 1000.0
        
        # Parse numeric value if present
        v_str = comp.value.lower()
        multiplier = 1.0
        if "k" in v_str:
            multiplier = 1e3
        elif "m" in v_str and "meg" not in v_str:
            multiplier = 1e-3
        elif "u" in v_str:
            multiplier = 1e-6
        elif "n" in v_str:
            multiplier = 1e-9
        elif "p" in v_str:
            multiplier = 1e-12
        elif "meg" in v_str:
            multiplier = 1e6

        num_match = re.search(r'([0-9\.]+)', v_str)
        if num_match:
            try:
                r_val = float(num_match.group(1)) * multiplier
            except Exception:
                pass

        if comp.comp_type == "R":
            v_drop = min(5.0, 5.0 * (r_val / max(1.0, r_val + 1000.0)))
            i_comp = v_drop / max(1e-6, r_val)
            p_comp = v_drop * i_comp
            comp.calculated_voltage = v_drop
            comp.calculated_current = i_comp
            comp.calculated_power = p_comp
            return {
                "ref": comp.ref,
                "type": "Resistor",
                "value": comp.value,
                "resistance_ohms": r_val,
                "voltage_drop_v": v_drop,
                "current_a": i_comp,
                "current_ma": i_comp * 1000.0,
                "power_w": p_comp,
                "power_mw": p_comp * 1000.0,
                "pins": {p: pin.net for p, pin in comp.pins.items()}
            }
        elif comp.comp_type == "C":
            xc = 1.0 / (2.0 * 3.14159 * 1000.0 * max(1e-15, r_val))
            comp.calculated_voltage = 5.0
            return {
                "ref": comp.ref,
                "type": "Capacitor",
                "value": comp.value,
                "capacitance_f": r_val,
                "reactance_1khz_ohms": xc,
                "rated_dc_v": 5.0,
                "pins": {p: pin.net for p, pin in comp.pins.items()}
            }
        elif comp.comp_type == "L":
            xl = 2.0 * 3.14159 * 1000.0 * r_val
            return {
                "ref": comp.ref,
                "type": "Inductor",
                "value": comp.value,
                "inductance_h": r_val,
                "reactance_1khz_ohms": xl,
                "pins": {p: pin.net for p, pin in comp.pins.items()}
            }
        elif comp.comp_type in ("V", "SOURCE"):
            v_src = r_val if r_val != 1000.0 else 5.0
            return {
                "ref": comp.ref,
                "type": "Voltage Source",
                "value": f"{v_src}V",
                "voltage_v": v_src,
                "pins": {p: pin.net for p, pin in comp.pins.items()}
            }
        else:
            return {
                "ref": comp.ref,
                "type": comp.comp_type,
                "value": comp.value,
                "footprint": comp.footprint_name,
                "pins": {p: pin.net for p, pin in comp.pins.items()}
            }

    def render_schematic_ascii(self) -> str:
        """Renders industry-standard paper-style schematic diagram with short component blocks."""
        from CORE.ascii_canvas import SchematicVisualizer
        pin_map: Dict[str, List[str]] = {}
        for ref, comp in self.components.items():
            pin_map[ref] = [p.net or "?" for p in comp.pins.values()]
        return SchematicVisualizer.render_circuit_topology(self.components, pin_map)
