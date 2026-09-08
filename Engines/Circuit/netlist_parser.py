"""Netlist manager and parser for interactive CLI circuit commands and SPICE format.
Supports hierarchical subcircuits (.SUBCKT), models (.MODEL), parameters (.PARAM),
behavioral sources (B), semiconductor models (Diode, BJT, MOSFET, JFET), and controlled sources.
"""

from __future__ import annotations
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
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
from .catalog import circuit_catalog
from CORE.common_math import parse_eng_unit


class Netlist:
    """Manages the graph of circuit components, subcircuits, and node connections."""

    def __init__(self, name: str = "Circuit"):
        self.name = name
        self.components: Dict[str, Component] = {}
        # Map: component_name -> list of pin net names
        self.pin_map: Dict[str, List[str]] = {}
        # Map alias nodes (e.g., 'gnd', 'GND', '0' -> '0')
        self.node_aliases: Dict[str, str] = {"gnd": "0", "GND": "0", "0": "0"}
        # Subcircuit definitions local to netlist
        self.subcircuits: Dict[str, SubcircuitDefinition] = {}
        # Models registered (.model)
        self.models: Dict[str, Dict[str, Any]] = {}
        # Parameters (.param)
        self.params: Dict[str, float] = {}

    def clear(self) -> None:
        self.components.clear()
        self.pin_map.clear()
        self.node_aliases = {"gnd": "0", "GND": "0", "0": "0"}
        self.subcircuits.clear()
        self.models.clear()
        self.params.clear()

    def normalize_node(self, node: str) -> str:
        n = str(node).strip()
        return self.node_aliases.get(n, n)

    def add_component(self, comp: Component) -> None:
        uname = comp.name.upper()
        self.components[uname] = comp
        self.pin_map[uname] = [self.normalize_node(n) for n in comp.nodes]

    def remove_component(self, name: str) -> bool:
        uname = name.upper()
        if uname in self.components:
            del self.components[uname]
            if uname in self.pin_map:
                del self.pin_map[uname]
            return True
        return False

    def get_all_nodes(self) -> List[str]:
        """Returns all unique node names with '0' (ground) at index 0 if present."""
        nodes: Set[str] = set()
        for cname, pins in self.pin_map.items():
            for p in pins:
                norm = self.normalize_node(p)
                if norm != "?":
                    nodes.add(norm)
        
        node_list = sorted(list(nodes))
        if "0" in node_list:
            node_list.remove("0")
            node_list.insert(0, "0")
        return node_list

    def resolve_pin_index(self, comp_name: str, pin_spec: str) -> int:
        """Resolves pin name/index to pin position."""
        comp = self.components.get(comp_name.upper())
        if not comp:
            raise KeyError(f"Component '{comp_name}' does not exist in netlist.")

        spec = pin_spec.lower().strip()

        if isinstance(comp, (Resistor, Capacitor, Inductor, VoltageSource, CurrentSource, BehavioralSource)):
            if spec in ("p", "a", "plus", "+", "1", "anode", "in"):
                return 0
            if spec in ("n", "b", "minus", "-", "2", "cathode", "out"):
                return 1
        elif isinstance(comp, Diode):
            if spec in ("a", "p", "anode", "plus", "1"):
                return 0
            if spec in ("k", "c", "n", "cathode", "minus", "2"):
                return 1
        elif isinstance(comp, BJT):
            if spec in ("c", "collector", "1"):
                return 0
            if spec in ("b", "base", "2"):
                return 1
            if spec in ("e", "emitter", "3"):
                return 2
        elif isinstance(comp, (MOSFET, JFET)):
            if spec in ("d", "drain", "1"):
                return 0
            if spec in ("g", "gate", "2"):
                return 1
            if spec in ("s", "source", "3"):
                return 2
            if spec in ("b", "body", "bulk", "4"):
                return 3
        elif isinstance(comp, (VCVS, VCCS)):
            if spec in ("out_p", "out+", "1"):
                return 0
            if spec in ("out_n", "out-", "2"):
                return 1
            if spec in ("in_p", "in+", "3"):
                return 2
            if spec in ("in_n", "in-", "4"):
                return 3
        elif isinstance(comp, VoltageControlledSwitch):
            if spec in ("sw_p", "sw+", "1"):
                return 0
            if spec in ("sw_n", "sw-", "2"):
                return 1
            if spec in ("ctrl_p", "ctrl+", "3"):
                return 2
            if spec in ("ctrl_n", "ctrl-", "4"):
                return 3
        elif isinstance(comp, OpAmpModel):
            if spec in ("out", "1"):
                return 0
            if spec in ("in-", "inv", "2"):
                return 1
            if spec in ("in+", "noninv", "3"):
                return 2
            if spec in ("v+", "vcc", "4"):
                return 3
            if spec in ("v-", "vee", "5"):
                return 4

        # Numerical index fallback (1-based)
        if spec.isdigit():
            idx = int(spec) - 1
            if 0 <= idx < len(comp.nodes):
                return idx

        raise ValueError(f"Unknown pin specification '{pin_spec}' for component '{comp_name}'.")

    def connect_pins(self, targets: List[str]) -> None:
        """Connects multiple pins and/or explicit net names together to form a common node."""
        if len(targets) < 2:
            raise ValueError("Connect command requires at least 2 endpoints/nets.")

        chosen_net: Optional[str] = None
        pins_to_wire: List[Tuple[str, int]] = []

        for item in targets:
            item = item.strip()
            if not item:
                continue
            if "." in item:
                comp_name, pin_name = item.split(".", 1)
                comp_name_u = comp_name.upper()
                if comp_name_u not in self.components:
                    raise KeyError(f"Component '{comp_name}' not found.")
                pin_idx = self.resolve_pin_index(comp_name_u, pin_name)
                pins_to_wire.append((comp_name_u, pin_idx))
            else:
                norm = self.normalize_node(item)
                if norm == "0":
                    chosen_net = "0"
                elif chosen_net is None:
                    chosen_net = norm

        if chosen_net is None:
            for cname, pidx in pins_to_wire:
                curr_node = self.pin_map[cname][pidx]
                if curr_node != "?" and curr_node != "0":
                    chosen_net = curr_node
                    break
            if chosen_net is None:
                first_c, first_p = pins_to_wire[0]
                chosen_net = f"net_{first_c.lower()}_{first_p}"

        for cname, pidx in pins_to_wire:
            self.pin_map[cname][pidx] = chosen_net
            self.components[cname].nodes[pidx] = chosen_net

    def flatten_subcircuits(self) -> Netlist:
        """Flattens any SubcircuitInstance (X...) into base components for MNA solving."""
        flat_netlist = Netlist(name=self.name)
        flat_netlist.node_aliases = dict(self.node_aliases)
        flat_netlist.params = dict(self.params)
        flat_netlist.models = dict(self.models)

        for cname, comp in self.components.items():
            if isinstance(comp, SubcircuitInstance):
                # Lookup subcircuit definition from netlist or catalog
                sub_def = self.subcircuits.get(comp.subckt_name) or circuit_catalog.get_subcircuit(comp.subckt_name)
                if not sub_def:
                    raise KeyError(f"Subcircuit definition '{comp.subckt_name}' not found for instance '{comp.name}'.")
                expanded = sub_def.instantiate(comp.name, comp.nodes, comp.instance_params)
                for exp_comp in expanded:
                    flat_netlist.add_component(exp_comp)
            else:
                flat_netlist.add_component(comp)

        return flat_netlist

    def export_spice(self) -> str:
        """Exports netlist to standard SPICE (.cir) netlist format."""
        lines = [f"* SPICE Netlist generated by Terminus: {self.name}"]
        for p_name, p_val in self.params.items():
            lines.append(f".param {p_name}={p_val}")

        for m_name, m_spec in self.models.items():
            lines.append(f".model {m_name} {m_spec}")

        for cname, comp in self.components.items():
            pins_str = " ".join(self.pin_map.get(cname, comp.nodes))
            if isinstance(comp, Resistor):
                lines.append(f"{cname} {pins_str} {comp.value}")
            elif isinstance(comp, Capacitor):
                lines.append(f"{cname} {pins_str} {comp.value}")
            elif isinstance(comp, Inductor):
                lines.append(f"{cname} {pins_str} {comp.value}")
            elif isinstance(comp, VoltageSource):
                if comp.waveform_type == "DC":
                    lines.append(f"{cname} {pins_str} DC {comp.dc}")
                elif comp.waveform_type in ("SINE", "SIN"):
                    off = comp.wave_params.get("offset", 0.0)
                    amp = comp.wave_params.get("amplitude", 1.0)
                    freq = comp.wave_params.get("freq", 1000.0)
                    lines.append(f"{cname} {pins_str} SINE({off} {amp} {freq})")
                elif comp.waveform_type == "PULSE":
                    v1 = comp.wave_params.get("v1", 0.0)
                    v2 = comp.wave_params.get("v2", 5.0)
                    lines.append(f"{cname} {pins_str} PULSE({v1} {v2})")
            elif isinstance(comp, CurrentSource):
                lines.append(f"{cname} {pins_str} DC {comp.dc}")
            elif isinstance(comp, Diode):
                lines.append(f"{cname} {pins_str} {comp.model_name}")
            elif isinstance(comp, BJT):
                lines.append(f"{cname} {pins_str} {comp.model_name}")
            elif isinstance(comp, MOSFET):
                lines.append(f"{cname} {pins_str} {comp.model_name}")
            elif isinstance(comp, JFET):
                lines.append(f"{cname} {pins_str} {comp.model_name}")
            elif isinstance(comp, BehavioralSource):
                lines.append(f"{cname} {pins_str} {comp.source_type}={comp.expression}")
            elif isinstance(comp, SubcircuitInstance):
                lines.append(f"{cname} {pins_str} {comp.subckt_name}")
            else:
                lines.append(f"{cname} {pins_str}")

        lines.append(".end")
        return "\n".join(lines)


class CircuitParser:
    """Parses interactive commands and SPICE text files into Netlist structures and simulation specs."""

    @classmethod
    def parse_command(cls, netlist: Netlist, cmd_line: str) -> Dict[str, Union[str, Dict, bool, Any]]:
        """Parses a single interactive circuit command."""
        line = cmd_line.strip()
        if not line or line.startswith("#") or line.startswith("*"):
            return {"type": "comment", "raw": line}

        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "clear":
            netlist.clear()
            return {"type": "clear", "status": "ok", "message": "Circuit cleared."}

        elif cmd in ("list", "show"):
            summary = []
            for name, comp in netlist.components.items():
                pins = netlist.pin_map.get(name, comp.nodes)
                val = getattr(comp, "value", getattr(comp, "dc", getattr(comp, "model_name", getattr(comp, "expression", ""))))
                summary.append(f"{name} [{', '.join(pins)}] = {val}")
            return {"type": "list", "status": "ok", "components": summary}

        elif cmd == "add":
            if len(parts) < 3:
                raise ValueError("Usage: add <Name> [nodes...] <Value/Model> [param=val ...]")
            name = parts[1].upper()
            prefix = name[0]

            # Extract kwargs (param=val)
            kwargs = {}
            pos_args = []
            for p in parts[2:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    kwargs[k.lower()] = v
                else:
                    pos_args.append(p)

            # Check if positional arguments include explicit nodes
            # 2-terminal: R, C, L, V, I, D, B (if 3 positional args: node_p, node_n, value)
            if prefix in ("R", "C", "L", "V", "I", "D", "B") and len(pos_args) == 3:
                node_p, node_n, val_or_type = pos_args[0], pos_args[1], pos_args[2]
                comp = cls._create_component_from_spec(name, val_or_type, kwargs)
                comp.nodes = [node_p, node_n]
            # 3-terminal: Q, J (if 4 positional args: c, b, e, model)
            elif prefix in ("Q", "J") and len(pos_args) == 4:
                n1, n2, n3, val_or_type = pos_args[0], pos_args[1], pos_args[2], pos_args[3]
                comp = cls._create_component_from_spec(name, val_or_type, kwargs)
                comp.nodes = [n1, n2, n3]
            # 4-terminal: M, S, E, G (if 5 positional args: d, g, s, b, model)
            elif prefix in ("M", "S", "E", "G") and len(pos_args) == 5:
                n1, n2, n3, n4, val_or_type = pos_args[0], pos_args[1], pos_args[2], pos_args[3], pos_args[4]
                comp = cls._create_component_from_spec(name, val_or_type, kwargs)
                comp.nodes = [n1, n2, n3, n4]
            elif pos_args:
                val_or_type = pos_args[-1] if len(pos_args) == 1 else pos_args[-1]
                comp = cls._create_component_from_spec(name, val_or_type, kwargs)
                if len(pos_args) > 1:
                    comp.nodes = pos_args[:-1]
            else:
                val_or_type = "1k"
                comp = cls._create_component_from_spec(name, val_or_type, kwargs)

            netlist.add_component(comp)
            return {"type": "add", "status": "ok", "name": comp.name, "component": repr(comp)}

        elif cmd == "connect":
            raw_conns = " ".join(parts[1:])
            endpoints = [e.strip() for e in raw_conns.split("|") if e.strip()]
            if len(endpoints) < 2:
                endpoints = parts[1:]
            netlist.connect_pins(endpoints)
            return {"type": "connect", "status": "ok", "endpoints": endpoints}

        elif cmd in ("remove", "delete", "del"):
            if len(parts) < 2:
                raise ValueError("Usage: remove <Name>")
            name = parts[1]
            ok = netlist.remove_component(name)
            return {"type": "remove", "status": "ok" if ok else "not_found", "name": name}

        elif cmd == "set":
            if len(parts) < 3:
                raise ValueError("Usage: set <Name>.<param> <Value>")
            target = parts[1]
            val = parts[2]
            if "." in target:
                comp_name, param = target.split(".", 1)
                comp = netlist.components.get(comp_name.upper())
                if not comp:
                    raise KeyError(f"Component '{comp_name}' not found.")
                num_val = parse_eng_unit(val)
                setattr(comp, param.lower(), num_val)
                return {"type": "set", "status": "ok", "target": target, "value": num_val}
            else:
                raise ValueError("Set target must be in format <Component>.<param>")

        elif cmd == "run":
            sim_spec_str = " ".join(parts[1:])
            return {"type": "run", "status": "ok", "sim_spec": sim_spec_str}

        raise ValueError(f"Unknown circuit command: '{cmd}'")

    @classmethod
    def _create_component_from_spec(
        cls,
        name: str,
        val_or_type: str,
        kwargs: Dict[str, str]
    ) -> Component:
        prefix = name[0].upper()

        if prefix == "R":
            return Resistor(name, "?", "?", val_or_type)
        elif prefix == "C":
            ic = float(kwargs.get("ic", 0.0))
            return Capacitor(name, "?", "?", val_or_type, ic=ic)
        elif prefix == "L":
            ic = float(kwargs.get("ic", 0.0))
            return Inductor(name, "?", "?", val_or_type, ic=ic)
        elif prefix == "V":
            ac_mag = float(parse_eng_unit(kwargs.get("ac", "0.0")))
            ac_ph = float(kwargs.get("phase", "0.0"))
            wtype = kwargs.get("type", "DC").upper()
            wave_params = {}
            if "freq" in kwargs:
                wave_params["freq"] = parse_eng_unit(kwargs["freq"])
            if "amplitude" in kwargs or "amp" in kwargs:
                wave_params["amplitude"] = parse_eng_unit(kwargs.get("amplitude", kwargs.get("amp", "1.0")))
            if "offset" in kwargs:
                wave_params["offset"] = parse_eng_unit(kwargs["offset"])
            return VoltageSource(
                name, "?", "?",
                dc=val_or_type,
                ac_mag=ac_mag,
                ac_phase=ac_ph,
                waveform_type=wtype,
                wave_params=wave_params
            )
        elif prefix == "I":
            ac_mag = float(parse_eng_unit(kwargs.get("ac", "0.0")))
            ac_ph = float(kwargs.get("phase", "0.0"))
            return CurrentSource(name, "?", "?", dc=val_or_type, ac_mag=ac_mag, ac_phase=ac_ph)
        elif prefix == "D":
            return Diode(name, "?", "?", model_name=val_or_type)
        elif prefix == "Q":
            beta = float(kwargs.get("beta", "100.0"))
            bjt_type = kwargs.get("type", "NPN").upper()
            return BJT(name, "?", "?", "?", model_name=val_or_type, bjt_type=bjt_type, beta_f=beta)
        elif prefix == "M":
            mos_type = kwargs.get("type", "NMOS").upper()
            return MOSFET(name, "?", "?", "?", "?", model_name=val_or_type, mos_type=mos_type)
        elif prefix == "J":
            jfet_type = kwargs.get("type", "NJFET").upper()
            return JFET(name, "?", "?", "?", model_name=val_or_type, jfet_type=jfet_type)
        elif prefix == "S":
            return VoltageControlledSwitch(name, "?", "?", "?", "?", model_name=val_or_type)
        elif prefix == "B":
            stype = "V" if val_or_type.upper().startswith("V=") else ("I" if val_or_type.upper().startswith("I=") else "V")
            expr = val_or_type.split("=", 1)[1] if "=" in val_or_type else val_or_type
            return BehavioralSource(name, "?", "?", source_type=stype, expression=expr)
        elif prefix in ("X", "U"):
            return SubcircuitInstance(name, val_or_type, ["?", "?", "?"])
        elif prefix == "E":
            gain = parse_eng_unit(val_or_type)
            return VCVS(name, "?", "?", "?", "?", gain=gain)

        return Resistor(name, "?", "?", val_or_type)

    @classmethod
    def parse_spice_netlist(cls, spice_text: str) -> Tuple[Netlist, List[str]]:
        """Parses a full standard SPICE netlist string including models, subcircuits, and directives."""
        netlist = Netlist()
        sim_commands: List[str] = []
        current_subckt: Optional[SubcircuitDefinition] = None

        lines = spice_text.strip().splitlines()
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("*"):
                continue

            # Strip inline comments with semicolon or $
            if ";" in line:
                line = line.split(";", 1)[0].strip()
            if not line:
                continue

            # Directive lines (.SUBCKT, .ENDS, .MODEL, .PARAM, .TRAN, .AC, .OP, .DC, etc.)
            if line.startswith("."):
                directive_parts = line.split()
                dir_name = directive_parts[0].upper()

                if dir_name == ".SUBCKT":
                    if len(directive_parts) >= 3:
                        s_name = directive_parts[1].upper()
                        s_pins = directive_parts[2:]
                        current_subckt = SubcircuitDefinition(s_name, s_pins)
                    continue

                elif dir_name == ".ENDS":
                    if current_subckt:
                        netlist.subcircuits[current_subckt.name] = current_subckt
                        current_subckt = None
                    continue

                elif dir_name == ".MODEL":
                    if len(directive_parts) >= 3:
                        m_name = directive_parts[1].upper()
                        m_def = " ".join(directive_parts[2:])
                        netlist.models[m_name] = {"definition": m_def}
                    continue

                elif dir_name == ".PARAM":
                    for param_pair in directive_parts[1:]:
                        if "=" in param_pair:
                            p_k, p_v = param_pair.split("=", 1)
                            try:
                                netlist.params[p_k.upper()] = parse_eng_unit(p_v)
                            except Exception:
                                pass
                    continue

                elif dir_name in (".TRAN", ".AC", ".OP", ".DC", ".NOISE", ".TF", ".FOUR", ".STEP"):
                    sim_commands.append(line)
                    continue

                elif dir_name == ".END":
                    break
                else:
                    sim_commands.append(line)
                    continue

            tokens = line.split()
            if len(tokens) < 3:
                continue

            cname = tokens[0].upper()
            prefix = cname[0]

            # 1. Resistor: R1 n1 n2 1k
            if prefix == "R":
                comp = Resistor(cname, tokens[1], tokens[2], tokens[3])
            # 2. Capacitor: C1 n1 n2 1u [IC=0]
            elif prefix == "C":
                ic = 0.0
                for tok in tokens[4:]:
                    if tok.upper().startswith("IC="):
                        ic = parse_eng_unit(tok.split("=")[1])
                comp = Capacitor(cname, tokens[1], tokens[2], tokens[3], ic=ic)
            # 3. Inductor: L1 n1 n2 1m [IC=0]
            elif prefix == "L":
                ic = 0.0
                for tok in tokens[4:]:
                    if tok.upper().startswith("IC="):
                        ic = parse_eng_unit(tok.split("=")[1])
                comp = Inductor(cname, tokens[1], tokens[2], tokens[3], ic=ic)
            # 4. Mutual Inductance: K1 L1 L2 0.99
            elif prefix == "K":
                comp = CoupledInductors(cname, tokens[1], tokens[2], float(tokens[3]))
            # 5. Voltage Source: V1 n1 n2 [DC] 10 [AC 1 0] [SIN(...)]
            elif prefix == "V":
                dc_val = 0.0
                ac_mag = 0.0
                ac_ph = 0.0
                wtype = "DC"
                wparams = {}
                idx = 3
                while idx < len(tokens):
                    tok = tokens[idx].upper()
                    if tok == "DC" and idx + 1 < len(tokens):
                        dc_val = parse_eng_unit(tokens[idx + 1])
                        idx += 2
                    elif tok == "AC" and idx + 1 < len(tokens):
                        ac_mag = parse_eng_unit(tokens[idx + 1])
                        idx += 2
                        if idx < len(tokens) and not tokens[idx].startswith("."):
                            try:
                                ac_ph = float(tokens[idx])
                                idx += 1
                            except ValueError:
                                pass
                    elif tok.startswith("SIN"):
                        wtype = "SINE"
                        idx += 1
                    elif tok.startswith("PULSE"):
                        wtype = "PULSE"
                        idx += 1
                    else:
                        try:
                            dc_val = parse_eng_unit(tok)
                        except ValueError:
                            pass
                        idx += 1
                comp = VoltageSource(cname, tokens[1], tokens[2], dc=dc_val, ac_mag=ac_mag, ac_phase=ac_ph, waveform_type=wtype, wave_params=wparams)
            # 6. Current Source: I1 n1 n2 DC 1m
            elif prefix == "I":
                comp = CurrentSource(cname, tokens[1], tokens[2], dc=tokens[3] if len(tokens) > 3 else "0")
            # 7. Diode: D1 anode cathode [model]
            elif prefix == "D":
                mname = tokens[3] if len(tokens) > 3 else "1N4148"
                comp = Diode(cname, tokens[1], tokens[2], model_name=mname)
            # 8. BJT: Q1 c b e [model]
            elif prefix == "Q":
                mname = tokens[4] if len(tokens) > 4 else "2N2222"
                comp = BJT(cname, tokens[1], tokens[2], tokens[3], model_name=mname)
            # 9. MOSFET: M1 d g s b [model]
            elif prefix == "M":
                if len(tokens) >= 6:
                    comp = MOSFET(cname, tokens[1], tokens[2], tokens[3], tokens[4], model_name=tokens[5])
                else:
                    comp = MOSFET(cname, tokens[1], tokens[2], tokens[3], tokens[3], model_name=tokens[4] if len(tokens) > 4 else "IRF540N")
            # 10. JFET: J1 d g s [model]
            elif prefix == "J":
                mname = tokens[4] if len(tokens) > 4 else "2N5457"
                comp = JFET(cname, tokens[1], tokens[2], tokens[3], model_name=mname)
            # 11. Switch: S1 sw+ sw- ctrl+ ctrl- [model]
            elif prefix == "S":
                mname = tokens[5] if len(tokens) > 5 else "SW"
                comp = VoltageControlledSwitch(cname, tokens[1], tokens[2], tokens[3], tokens[4], model_name=mname)
            # 12. Behavioral Source: B1 out 0 V=V(in)*2
            elif prefix == "B":
                full_expr = " ".join(tokens[3:])
                stype = "V"
                if full_expr.upper().startswith("V="):
                    stype = "V"
                    full_expr = full_expr[2:]
                elif full_expr.upper().startswith("I="):
                    stype = "I"
                    full_expr = full_expr[2:]
                comp = BehavioralSource(cname, tokens[1], tokens[2], source_type=stype, expression=full_expr)
            # 13. Subcircuit Instance: X1 pin1 pin2 ... subckt_name
            elif prefix in ("X", "U"):
                sub_name = tokens[-1].upper()
                inst_pins = tokens[1:-1]
                comp = SubcircuitInstance(cname, sub_name, inst_pins)
            # 14. VCVS: E1 out+ out- in+ in- gain
            elif prefix == "E":
                gain = parse_eng_unit(tokens[5]) if len(tokens) > 5 else 1.0
                comp = VCVS(cname, tokens[1], tokens[2], tokens[3], tokens[4], gain=gain)
            # 15. VCCS: G1 out+ out- in+ in- gm
            elif prefix == "G":
                gm = parse_eng_unit(tokens[5]) if len(tokens) > 5 else 1.0
                comp = VCCS(cname, tokens[1], tokens[2], tokens[3], tokens[4], transconductance=gm)
            else:
                # Default fallback
                comp = Resistor(cname, tokens[1], tokens[2], tokens[3] if len(tokens) > 3 else "1k")

            if current_subckt:
                current_subckt.add_component(comp)
            else:
                netlist.add_component(comp)

        return netlist, sim_commands
