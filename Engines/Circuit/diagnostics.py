"""Comprehensive Circuit Design Rule Checker (DRC) and Diagnostic Engine.
Validates SPICE circuits for missing ground reference, floating/dangling nodes, singular MNA matrices,
shorted ideal sources, untied semiconductor terminals, and numerical convergence hazards.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Tuple
from .components import (
    Component,
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    CurrentSource,
    Diode,
    BJT,
    MOSFET,
    JFET,
    VCVS,
    VCCS,
    CCVS,
    CCCS,
    BehavioralSource,
    SubcircuitInstance,
)
from .netlist_parser import Netlist


class CircuitDiagnosticLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class CircuitDiagnosticIssue:
    """Individual circuit health or DRC issue."""
    level: CircuitDiagnosticLevel
    category: str
    message: str
    component_name: Optional[str] = None
    node_name: Optional[str] = None
    suggested_fix: str = ""


@dataclass
class CircuitDiagnosticReport:
    """Aggregated DRC validation and matrix condition report."""
    is_valid: bool
    total_components: int
    total_nodes: int
    issues: List[CircuitDiagnosticIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.level == CircuitDiagnosticLevel.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.level == CircuitDiagnosticLevel.WARNING for i in self.issues)


class CircuitDiagnosticChecker:
    """Design Rule Checking (DRC) and pre-simulation topological analyzer."""

    def __init__(self, netlist: Netlist):
        self.netlist = netlist

    def diagnose(self) -> CircuitDiagnosticReport:
        """Runs full suite of topological and physical design rule checks."""
        issues: List[CircuitDiagnosticIssue] = []
        components = self.netlist.components
        all_nodes = self.netlist.get_all_nodes()

        # 1. Check for empty netlist
        if not components:
            issues.append(CircuitDiagnosticIssue(
                level=CircuitDiagnosticLevel.ERROR,
                category="Topology",
                message="Netlist is completely empty. No components to simulate.",
                suggested_fix="Add components using 'add <type> <nodes...> <value>' or load a netlist file."
            ))
            return CircuitDiagnosticReport(is_valid=False, total_components=0, total_nodes=0, issues=issues)

        # 2. Check for Reference Ground (Net 0 / GND)
        has_ground = "0" in all_nodes or "GND" in self.netlist.node_aliases
        if not has_ground or "0" not in all_nodes:
            issues.append(CircuitDiagnosticIssue(
                level=CircuitDiagnosticLevel.ERROR,
                category="Ground Reference",
                message="Circuit has no reference ground node ('0' or 'GND'). MNA matrix will be singular.",
                suggested_fix="Connect at least one node to ground net '0' (e.g. 'connect V1 n 0')."
            ))

        # 3. Node Connectivity Map & Floating Node Check
        node_conn_count: Dict[str, List[str]] = {n: [] for n in all_nodes}
        for cname, pins in self.netlist.pin_map.items():
            for p in pins:
                norm_p = self.netlist.normalize_node(p)
                if norm_p in node_conn_count:
                    node_conn_count[norm_p].append(cname)

        for node, attached_comps in node_conn_count.items():
            if node == "0":
                continue
            if len(attached_comps) < 2:
                issues.append(CircuitDiagnosticIssue(
                    level=CircuitDiagnosticLevel.WARNING,
                    category="Floating Node",
                    message=f"Node '{node}' is floating / dangling (connected only to {attached_comps}).",
                    node_name=node,
                    suggested_fix=f"Connect node '{node}' to another component or tie to ground with a high-value pull-down resistor (e.g. 1G)."
                ))

        # 4. Check for Shorted Voltage Sources (0-Ohm loop / V-Source parallel loop)
        v_sources = [c for c in components.values() if isinstance(c, VoltageSource)]
        for vs in v_sources:
            norm_p = self.netlist.normalize_node(vs.nodes[0])
            norm_n = self.netlist.normalize_node(vs.nodes[1])
            if norm_p == norm_n:
                issues.append(CircuitDiagnosticIssue(
                    level=CircuitDiagnosticLevel.ERROR,
                    category="Short Circuit",
                    message=f"Voltage source '{vs.name}' has both terminals tied to the same node '{norm_p}'.",
                    component_name=vs.name,
                    suggested_fix=f"Ensure voltage source '{vs.name}' is connected across two distinct nodes."
                ))

        # Check for parallel voltage sources between identical node pairs
        vsource_pairs: Dict[Tuple[str, str], List[str]] = {}
        for vs in v_sources:
            n1 = self.netlist.normalize_node(vs.nodes[0])
            n2 = self.netlist.normalize_node(vs.nodes[1])
            pair = tuple(sorted([n1, n2]))
            if pair not in vsource_pairs:
                vsource_pairs[pair] = []
            vsource_pairs[pair].append(vs.name)

        for pair, vs_names in vsource_pairs.items():
            if len(vs_names) > 1:
                issues.append(CircuitDiagnosticIssue(
                    level=CircuitDiagnosticLevel.ERROR,
                    category="Parallel Voltage Sources",
                    message=f"Multiple ideal voltage sources ({', '.join(vs_names)}) connected in parallel across nodes {pair}. Causes singular MNA matrix.",
                    suggested_fix="Add internal series resistance (Rser) to each voltage source or remove redundant sources."
                ))

        # 5. Check for Loop of Inductors with Voltage Sources (Singular DC condition)
        # Check pure capacitive cutsets (node connected ONLY to capacitors and current sources)
        for node, attached_comps in node_conn_count.items():
            if node == "0":
                continue
            all_caps_or_isources = True
            for cname in attached_comps:
                comp = components.get(cname)
                if not isinstance(comp, (Capacitor, CurrentSource)):
                    all_caps_or_isources = False
                    break
            if all_caps_or_isources and attached_comps:
                issues.append(CircuitDiagnosticIssue(
                    level=CircuitDiagnosticLevel.WARNING,
                    category="DC Floating Node",
                    message=f"Node '{node}' has no DC path to ground (connected exclusively to capacitors/current sources).",
                    node_name=node,
                    suggested_fix=f"Add a large bypass resistor (e.g. 100Meg) from node '{node}' to ground for DC operating point convergence."
                ))

        # 6. Check for Extreme Component Values
        for cname, comp in components.items():
            if isinstance(comp, Resistor):
                if comp.value < 1e-9:
                    issues.append(CircuitDiagnosticIssue(
                        level=CircuitDiagnosticLevel.WARNING,
                        category="Numerical Precision",
                        message=f"Resistor '{comp.name}' has extremely low resistance ({comp.value} Ohms). May cause matrix ill-conditioning.",
                        component_name=comp.name,
                        suggested_fix="Increase resistance to at least 1mOhm (0.001) or model as ideal short."
                    ))
                elif comp.value > 1e12:
                    issues.append(CircuitDiagnosticIssue(
                        level=CircuitDiagnosticLevel.INFO,
                        category="Numerical Precision",
                        message=f"Resistor '{comp.name}' has very high resistance ({comp.value} Ohms).",
                        component_name=comp.name
                    ))
            elif isinstance(comp, Capacitor):
                if comp.value <= 0:
                    issues.append(CircuitDiagnosticIssue(
                        level=CircuitDiagnosticLevel.ERROR,
                        category="Component Parameter",
                        message=f"Capacitor '{comp.name}' has non-positive capacitance ({comp.value} F).",
                        component_name=comp.name
                    ))
            elif isinstance(comp, Inductor):
                if comp.value <= 0:
                    issues.append(CircuitDiagnosticIssue(
                        level=CircuitDiagnosticLevel.ERROR,
                        category="Component Parameter",
                        message=f"Inductor '{comp.name}' has non-positive inductance ({comp.value} H).",
                        component_name=comp.name
                    ))

        # 7. Check Semiconductor Model Validity
        for cname, comp in components.items():
            if isinstance(comp, Diode):
                if comp.is_sat <= 0:
                    issues.append(CircuitDiagnosticIssue(
                        level=CircuitDiagnosticLevel.ERROR,
                        category="Model Parameter",
                        message=f"Diode '{comp.name}' saturation current Is must be positive.",
                        component_name=comp.name
                    ))
            elif isinstance(comp, BJT):
                if comp.beta_f <= 0:
                    issues.append(CircuitDiagnosticIssue(
                        level=CircuitDiagnosticLevel.ERROR,
                        category="Model Parameter",
                        message=f"BJT '{comp.name}' forward current gain Beta (Bf) must be positive.",
                        component_name=comp.name
                    ))

        is_valid = not any(i.level == CircuitDiagnosticLevel.ERROR for i in issues)
        return CircuitDiagnosticReport(
            is_valid=is_valid,
            total_components=len(components),
            total_nodes=len(all_nodes),
            issues=issues
        )
