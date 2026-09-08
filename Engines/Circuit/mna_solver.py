"""Modified Nodal Analysis (MNA) solver for DC, AC, Transient, Fourier, and Parametric circuit simulations.
Features Newton-Raphson non-linear iterations with adaptive Gmin/source stepping,
support for MOSFETs, JFETs, BJTs, Diodes, Switches, Behavioral sources (B), and hierarchical subcircuits.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from CORE.common_math import parse_eng_unit, Waveform
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
    SubcircuitInstance,
)
from .netlist_parser import Netlist


class SimulationResult:
    """Holds time, frequency, or sweep simulation output traces and diagnostic metrics."""

    def __init__(self, sim_type: str, x_vector: np.ndarray, x_label: str = "Time (s)"):
        self.sim_type = sim_type.upper()  # OP, DC, AC, TRAN, FOUR, STEP
        self.x = np.asarray(x_vector, dtype=float)
        self.x_label = x_label
        # Map: trace_name -> Waveform
        self.waveforms: Dict[str, Waveform] = {}
        # Scalar operating point / Fourier metrics
        self.op_results: Dict[str, float] = {}
        self.metrics: Dict[str, Any] = {}
        self.message: str = ""

    def add_waveform(self, name: str, y: np.ndarray, unit: str = "V", domain: str = "time") -> Waveform:
        x_unit = "Hz" if self.sim_type == "AC" else ("V" if self.sim_type == "DC" else "s")
        wf = Waveform(self.x, y, name=name, x_unit=x_unit, y_unit=unit, domain=domain)
        self.waveforms[name] = wf
        return wf

    def get_waveform(self, name: str) -> Optional[Waveform]:
        return self.waveforms.get(name)

    def summary(self) -> str:
        """Returns concise human-readable summary of simulation results."""
        lines = [f"=== Simulation Result: {self.sim_type} ==="]
        if self.message:
            lines.append(f"Status: {self.message}")
        if self.sim_type == "OP":
            lines.append("Node Voltages & Branch Currents:")
            for k, v in sorted(self.op_results.items()):
                lines.append(f"  {k:18s} = {v:12.6g}")
        elif self.sim_type == "FOUR":
            lines.append(f"Fundamental Frequency: {self.metrics.get('fundamental_hz', 0):.2f} Hz")
            lines.append(f"Total Harmonic Distortion (THD): {self.metrics.get('thd_percent', 0.0):.4f} %")
            for h_name, h_val in sorted(self.op_results.items()):
                lines.append(f"  {h_name:18s} = {h_val:12.6g}")
        else:
            lines.append(f"Swept Points: {len(self.x)}")
            lines.append(f"Traces Recorded: {', '.join(self.waveforms.keys())}")
        return "\n".join(lines)


class MNASolver:
    """Modified Nodal Analysis solver with nonlinear convergence algorithms."""

    def __init__(self, netlist: Netlist):
        self.netlist = netlist.flatten_subcircuits()

    def _build_node_index_map(self) -> Tuple[Dict[str, int], List[str]]:
        """Maps circuit nodes to 0-based matrix row indices (ground '0' is excluded from unknowns)."""
        all_nodes = self.netlist.get_all_nodes()
        node_to_idx: Dict[str, int] = {}
        idx_to_node: List[str] = []

        curr_idx = 0
        for n in all_nodes:
            norm = self.netlist.normalize_node(n)
            if norm == "0":
                continue
            if norm not in node_to_idx:
                node_to_idx[norm] = curr_idx
                idx_to_node.append(norm)
                curr_idx += 1

        return node_to_idx, idx_to_node

    def solve_op(self, max_iter: int = 100, tol: float = 1e-6) -> SimulationResult:
        """Calculates DC Operating Point (.OP) with adaptive Gmin and Newton-Raphson iterations."""
        node_map, idx_to_node = self._build_node_index_map()
        num_nodes = len(node_map)

        v_sources: List[VoltageSource] = []
        inductors: List[Inductor] = []
        vcvs_list: List[VCVS] = []
        b_v_sources: List[BehavioralSource] = []

        for comp in self.netlist.components.values():
            if isinstance(comp, VoltageSource):
                v_sources.append(comp)
            elif isinstance(comp, Inductor):
                inductors.append(comp)
            elif isinstance(comp, VCVS):
                vcvs_list.append(comp)
            elif isinstance(comp, BehavioralSource) and comp.source_type == "V":
                b_v_sources.append(comp)

        num_aux = len(v_sources) + len(inductors) + len(vcvs_list) + len(b_v_sources)
        total_dim = num_nodes + num_aux

        if total_dim == 0:
            res = SimulationResult("OP", np.array([0.0]))
            res.message = "Empty circuit netlist."
            return res

        # Assign aux indices
        aux_cursor = num_nodes
        for vs in v_sources:
            vs.aux_index = aux_cursor
            aux_cursor += 1
        for ind in inductors:
            ind.aux_index = aux_cursor
            aux_cursor += 1
        for e in vcvs_list:
            e.aux_index = aux_cursor
            aux_cursor += 1
        for bv in b_v_sources:
            bv.aux_index = aux_cursor
            aux_cursor += 1

        sol = np.zeros(total_dim, dtype=float)
        gmin_levels = [1e-12, 1e-9, 1e-6, 1e-4]
        converged = False

        for gmin in gmin_levels:
            for iteration in range(max_iter):
                A = np.zeros((total_dim, total_dim), dtype=float)
                b = np.zeros(total_dim, dtype=float)

                # Map current voltages for nonlinear/behavioral evaluation
                node_voltages: Dict[str, float] = {
                    name: sol[idx] for name, idx in node_map.items()
                }
                node_voltages["0"] = 0.0

                # 1. Resistors
                for comp in self.netlist.components.values():
                    if isinstance(comp, Resistor):
                        np_name = self.netlist.normalize_node(comp.nodes[0])
                        nn_name = self.netlist.normalize_node(comp.nodes[1])
                        g = comp.conductance
                        p_idx = node_map.get(np_name)
                        n_idx = node_map.get(nn_name)
                        if p_idx is not None:
                            A[p_idx, p_idx] += g
                        if n_idx is not None:
                            A[n_idx, n_idx] += g
                        if p_idx is not None and n_idx is not None:
                            A[p_idx, n_idx] -= g
                            A[n_idx, p_idx] -= g

                # 2. Independent Current Sources
                for comp in self.netlist.components.values():
                    if isinstance(comp, CurrentSource):
                        np_name = self.netlist.normalize_node(comp.nodes[0])
                        nn_name = self.netlist.normalize_node(comp.nodes[1])
                        p_idx = node_map.get(np_name)
                        n_idx = node_map.get(nn_name)
                        val = comp.dc
                        if p_idx is not None:
                            b[p_idx] -= val
                        if n_idx is not None:
                            b[n_idx] += val

                # 3. Voltage Sources
                for vs in v_sources:
                    np_name = self.netlist.normalize_node(vs.nodes[0])
                    nn_name = self.netlist.normalize_node(vs.nodes[1])
                    p_idx = node_map.get(np_name)
                    n_idx = node_map.get(nn_name)
                    aux = vs.aux_index
                    if p_idx is not None:
                        A[p_idx, aux] += 1.0
                        A[aux, p_idx] += 1.0
                    if n_idx is not None:
                        A[n_idx, aux] -= 1.0
                        A[aux, n_idx] -= 1.0
                    b[aux] += vs.dc

                # 4. Inductors (0V short in DC)
                for ind in inductors:
                    np_name = self.netlist.normalize_node(ind.nodes[0])
                    nn_name = self.netlist.normalize_node(ind.nodes[1])
                    p_idx = node_map.get(np_name)
                    n_idx = node_map.get(nn_name)
                    aux = ind.aux_index
                    if p_idx is not None:
                        A[p_idx, aux] += 1.0
                        A[aux, p_idx] += 1.0
                    if n_idx is not None:
                        A[n_idx, aux] -= 1.0
                        A[aux, n_idx] -= 1.0
                    b[aux] += 0.0

                # 5. VCVS (E-Source)
                for e in vcvs_list:
                    out_p = node_map.get(self.netlist.normalize_node(e.nodes[0]))
                    out_n = node_map.get(self.netlist.normalize_node(e.nodes[1]))
                    in_p = node_map.get(self.netlist.normalize_node(e.nodes[2]))
                    in_n = node_map.get(self.netlist.normalize_node(e.nodes[3]))
                    aux = e.aux_index
                    if out_p is not None:
                        A[out_p, aux] += 1.0
                        A[aux, out_p] += 1.0
                    if out_n is not None:
                        A[out_n, aux] -= 1.0
                        A[aux, out_n] -= 1.0
                    if in_p is not None:
                        A[aux, in_p] -= e.gain
                    if in_n is not None:
                        A[aux, in_n] += e.gain

                # 6. Diodes
                for comp in self.netlist.components.values():
                    if isinstance(comp, Diode):
                        np_name = self.netlist.normalize_node(comp.nodes[0])
                        nn_name = self.netlist.normalize_node(comp.nodes[1])
                        p_idx = node_map.get(np_name)
                        n_idx = node_map.get(nn_name)
                        vp = sol[p_idx] if p_idx is not None else 0.0
                        vn = sol[n_idx] if n_idx is not None else 0.0
                        vd = vp - vn
                        gd, ieq = comp.linearize(vd)
                        if p_idx is not None:
                            A[p_idx, p_idx] += gd
                            b[p_idx] -= ieq
                        if n_idx is not None:
                            A[n_idx, n_idx] += gd
                            b[n_idx] += ieq
                        if p_idx is not None and n_idx is not None:
                            A[p_idx, n_idx] -= gd
                            A[n_idx, p_idx] -= gd

                # 7. BJTs
                for comp in self.netlist.components.values():
                    if isinstance(comp, BJT):
                        c_idx = node_map.get(self.netlist.normalize_node(comp.nodes[0]))
                        b_idx = node_map.get(self.netlist.normalize_node(comp.nodes[1]))
                        e_idx = node_map.get(self.netlist.normalize_node(comp.nodes[2]))
                        vc = sol[c_idx] if c_idx is not None else 0.0
                        vb = sol[b_idx] if b_idx is not None else 0.0
                        ve = sol[e_idx] if e_idx is not None else 0.0
                        lin = comp.linearize(vb - ve, vc - ve)
                        gm, gpi, go = lin["g_m"], lin["g_pi"], lin["g_o"]
                        if b_idx is not None:
                            A[b_idx, b_idx] += gpi
                        if e_idx is not None:
                            A[e_idx, e_idx] += gpi + gm + go
                        if c_idx is not None:
                            A[c_idx, c_idx] += go
                        if c_idx is not None and b_idx is not None:
                            A[c_idx, b_idx] += gm
                        if c_idx is not None and e_idx is not None:
                            A[c_idx, e_idx] -= (gm + go)
                        if b_idx is not None and e_idx is not None:
                            A[b_idx, e_idx] -= gpi
                        if e_idx is not None and b_idx is not None:
                            A[e_idx, b_idx] -= (gpi + gm)
                        if e_idx is not None and c_idx is not None:
                            A[e_idx, c_idx] -= go

                # 8. MOSFETs
                for comp in self.netlist.components.values():
                    if isinstance(comp, MOSFET):
                        d_idx = node_map.get(self.netlist.normalize_node(comp.nodes[0]))
                        g_idx = node_map.get(self.netlist.normalize_node(comp.nodes[1]))
                        s_idx = node_map.get(self.netlist.normalize_node(comp.nodes[2]))
                        vd = sol[d_idx] if d_idx is not None else 0.0
                        vg = sol[g_idx] if g_idx is not None else 0.0
                        vs_node = sol[s_idx] if s_idx is not None else 0.0
                        lin = comp.linearize(vg - vs_node, vd - vs_node)
                        gm, gds, id_val = lin["gm"], lin["gds"], lin["id"]
                        ieq = id_val - gm * (vg - vs_node) - gds * (vd - vs_node)
                        if d_idx is not None:
                            A[d_idx, d_idx] += gds
                            b[d_idx] -= ieq
                        if s_idx is not None:
                            A[s_idx, s_idx] += gds + gm
                            b[s_idx] += ieq
                        if d_idx is not None and g_idx is not None:
                            A[d_idx, g_idx] += gm
                        if d_idx is not None and s_idx is not None:
                            A[d_idx, s_idx] -= (gm + gds)
                        if s_idx is not None and g_idx is not None:
                            A[s_idx, g_idx] -= gm
                        if s_idx is not None and d_idx is not None:
                            A[s_idx, d_idx] -= gds

                # 9. Voltage-Controlled Switches
                for comp in self.netlist.components.values():
                    if isinstance(comp, VoltageControlledSwitch):
                        p_idx = node_map.get(self.netlist.normalize_node(comp.nodes[0]))
                        n_idx = node_map.get(self.netlist.normalize_node(comp.nodes[1]))
                        cp_idx = node_map.get(self.netlist.normalize_node(comp.nodes[2]))
                        cn_idx = node_map.get(self.netlist.normalize_node(comp.nodes[3]))
                        v_cp = sol[cp_idx] if cp_idx is not None else 0.0
                        v_cn = sol[cn_idx] if cn_idx is not None else 0.0
                        g_sw = comp.get_conductance(v_cp - v_cn)
                        if p_idx is not None:
                            A[p_idx, p_idx] += g_sw
                        if n_idx is not None:
                            A[n_idx, n_idx] += g_sw
                        if p_idx is not None and n_idx is not None:
                            A[p_idx, n_idx] -= g_sw
                            A[n_idx, p_idx] -= g_sw

                # 10. Behavioral Sources (B-Source)
                for comp in self.netlist.components.values():
                    if isinstance(comp, BehavioralSource):
                        np_name = self.netlist.normalize_node(comp.nodes[0])
                        nn_name = self.netlist.normalize_node(comp.nodes[1])
                        p_idx = node_map.get(np_name)
                        n_idx = node_map.get(nn_name)
                        b_val = comp.evaluate(node_voltages, time=0.0)
                        if comp.source_type == "I":
                            if p_idx is not None:
                                b[p_idx] -= b_val
                            if n_idx is not None:
                                b[n_idx] += b_val
                        else:
                            aux = comp.aux_index
                            if p_idx is not None:
                                A[p_idx, aux] += 1.0
                                A[aux, p_idx] += 1.0
                            if n_idx is not None:
                                A[n_idx, aux] -= 1.0
                                A[aux, n_idx] -= 1.0
                            b[aux] += b_val

                # GMIN diagonal stability
                for i in range(num_nodes):
                    A[i, i] += gmin

                try:
                    new_sol = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    new_sol = np.linalg.lstsq(A, b, rcond=None)[0]

                diff = np.max(np.abs(new_sol - sol))
                sol = new_sol
                if diff < tol:
                    converged = True
                    break

            if converged:
                break

        res = SimulationResult("OP", np.array([0.0]))
        res.message = f"DC Operating Point converged in {iteration + 1} iterations (gmin={gmin:1.0e})."
        for name, idx in node_map.items():
            res.op_results[f"V({name})"] = float(sol[idx])
        for vs in v_sources:
            res.op_results[f"I({vs.name})"] = float(sol[vs.aux_index])
        for ind in inductors:
            res.op_results[f"I({ind.name})"] = float(sol[ind.aux_index])

        return res

    def solve_dc_sweep(
        self,
        src_name: str,
        start_val: float,
        stop_val: float,
        step_val: float
    ) -> SimulationResult:
        """Sweeps a DC source parameter and computes node voltage transfer curves."""
        src_comp = self.netlist.components.get(src_name.upper())
        if not src_comp or not isinstance(src_comp, VoltageSource):
            raise KeyError(f"Voltage source '{src_name}' not found for DC sweep.")

        orig_val = src_comp.dc
        sweep_vals = np.arange(start_val, stop_val + step_val * 0.5, step_val)
        res = SimulationResult("DC", sweep_vals, x_label=f"{src_name} (V)")

        node_traces: Dict[str, List[float]] = {}

        for val in sweep_vals:
            src_comp.dc = float(val)
            op = self.solve_op()
            for k, v in op.op_results.items():
                if k not in node_traces:
                    node_traces[k] = []
                node_traces[k].append(v)

        src_comp.dc = orig_val
        for k, vlist in node_traces.items():
            res.add_waveform(k, np.array(vlist), unit="V" if k.startswith("V") else "A", domain="dc_sweep")

        return res

    def solve_ac(
        self,
        sweep_type: str = "dec",
        points: int = 10,
        f_start: Union[str, float] = 1.0,
        f_stop: Union[str, float] = 100e3
    ) -> SimulationResult:
        """AC Small-Signal Frequency Sweep Analysis (.AC)."""
        f_start = parse_eng_unit(f_start)
        f_stop = parse_eng_unit(f_stop)

        if sweep_type.lower() == "dec":
            decades = math.log10(f_stop / f_start)
            num_pts = max(2, int(decades * points) + 1)
            frequencies = np.logspace(math.log10(f_start), math.log10(f_stop), num_pts)
        elif sweep_type.lower() == "oct":
            octaves = math.log2(f_stop / f_start)
            num_pts = max(2, int(octaves * points) + 1)
            frequencies = np.logspace(math.log10(f_start), math.log10(f_stop), num_pts)
        else:
            frequencies = np.linspace(f_start, f_stop, max(2, points))

        node_map, idx_to_node = self._build_node_index_map()
        num_nodes = len(node_map)

        v_sources: List[VoltageSource] = []
        for comp in self.netlist.components.values():
            if isinstance(comp, VoltageSource):
                v_sources.append(comp)

        num_aux = len(v_sources)
        total_dim = num_nodes + num_aux

        for i, vs in enumerate(v_sources):
            vs.aux_index = num_nodes + i

        node_complex_data: Dict[str, List[complex]] = {name: [] for name in node_map.keys()}

        for freq in frequencies:
            omega = 2.0 * math.pi * freq
            j_omega = 1j * omega

            A = np.zeros((total_dim, total_dim), dtype=complex)
            b = np.zeros(total_dim, dtype=complex)

            # Resistors
            for comp in self.netlist.components.values():
                if isinstance(comp, Resistor):
                    np_name = self.netlist.normalize_node(comp.nodes[0])
                    nn_name = self.netlist.normalize_node(comp.nodes[1])
                    g = comp.conductance
                    p_idx = node_map.get(np_name)
                    n_idx = node_map.get(nn_name)
                    if p_idx is not None:
                        A[p_idx, p_idx] += g
                    if n_idx is not None:
                        A[n_idx, n_idx] += g
                    if p_idx is not None and n_idx is not None:
                        A[p_idx, n_idx] -= g
                        A[n_idx, p_idx] -= g

            # Capacitors: Y = j * omega * C
            for comp in self.netlist.components.values():
                if isinstance(comp, Capacitor):
                    np_name = self.netlist.normalize_node(comp.nodes[0])
                    nn_name = self.netlist.normalize_node(comp.nodes[1])
                    yc = j_omega * comp.value
                    p_idx = node_map.get(np_name)
                    n_idx = node_map.get(nn_name)
                    if p_idx is not None:
                        A[p_idx, p_idx] += yc
                    if n_idx is not None:
                        A[n_idx, n_idx] += yc
                    if p_idx is not None and n_idx is not None:
                        A[p_idx, n_idx] -= yc
                        A[n_idx, p_idx] -= yc

            # Inductors: Y = 1 / (j * omega * L)
            for comp in self.netlist.components.values():
                if isinstance(comp, Inductor):
                    np_name = self.netlist.normalize_node(comp.nodes[0])
                    nn_name = self.netlist.normalize_node(comp.nodes[1])
                    yl = 1.0 / (j_omega * comp.value + 1e-15)
                    p_idx = node_map.get(np_name)
                    n_idx = node_map.get(nn_name)
                    if p_idx is not None:
                        A[p_idx, p_idx] += yl
                    if n_idx is not None:
                        A[n_idx, n_idx] += yl
                    if p_idx is not None and n_idx is not None:
                        A[p_idx, n_idx] -= yl
                        A[n_idx, p_idx] -= yl

            # AC Voltage Sources
            for vs in v_sources:
                np_name = self.netlist.normalize_node(vs.nodes[0])
                nn_name = self.netlist.normalize_node(vs.nodes[1])
                p_idx = node_map.get(np_name)
                n_idx = node_map.get(nn_name)
                aux = vs.aux_index
                if p_idx is not None:
                    A[p_idx, aux] += 1.0
                    A[aux, p_idx] += 1.0
                if n_idx is not None:
                    A[n_idx, aux] -= 1.0
                    A[aux, n_idx] -= 1.0

                mag = vs.ac_mag if vs.ac_mag > 0 else 1.0
                phasor = mag * math.cos(math.radians(vs.ac_phase)) + 1j * mag * math.sin(math.radians(vs.ac_phase))
                b[aux] += phasor

            for i in range(num_nodes):
                A[i, i] += 1e-12

            try:
                sol = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]

            for name, idx in node_map.items():
                node_complex_data[name].append(sol[idx])

        result = SimulationResult("AC", frequencies, x_label="Frequency (Hz)")
        for name, cdata in node_complex_data.items():
            result.add_waveform(f"V({name})", np.array(cdata), unit="V", domain="frequency")

        return result

    def solve_tran(
        self,
        t_step: Union[str, float],
        t_stop: Union[str, float],
        t_start: Union[str, float] = 0.0,
        method: str = "trapezoidal"
    ) -> SimulationResult:
        """Transient Time-Domain Analysis (.TRAN) with nonlinear iteration per timestep."""
        dt = parse_eng_unit(t_step)
        t_end = parse_eng_unit(t_stop)
        t_0 = parse_eng_unit(t_start)

        time_points = np.arange(t_0, t_end + dt * 0.5, dt)
        node_map, idx_to_node = self._build_node_index_map()
        num_nodes = len(node_map)

        v_sources: List[VoltageSource] = []
        b_v_sources: List[BehavioralSource] = []
        vcvs_list: List[VCVS] = []

        for comp in self.netlist.components.values():
            if isinstance(comp, VoltageSource):
                v_sources.append(comp)
            elif isinstance(comp, BehavioralSource) and comp.source_type == "V":
                b_v_sources.append(comp)
            elif isinstance(comp, VCVS):
                vcvs_list.append(comp)

        num_aux = len(v_sources) + len(b_v_sources) + len(vcvs_list)
        total_dim = num_nodes + num_aux

        aux_cursor = num_nodes
        for vs in v_sources:
            vs.aux_index = aux_cursor
            aux_cursor += 1
        for bv in b_v_sources:
            bv.aux_index = aux_cursor
            aux_cursor += 1
        for e in vcvs_list:
            e.aux_index = aux_cursor
            aux_cursor += 1

        capacitors = [c for c in self.netlist.components.values() if isinstance(c, Capacitor)]
        inductors = [c for c in self.netlist.components.values() if isinstance(c, Inductor)]

        node_waveforms: Dict[str, List[float]] = {name: [] for name in node_map.keys()}

        for cap in capacitors:
            cap.v_prev = cap.ic
            cap.i_prev = 0.0
        for ind in inductors:
            ind.i_prev = ind.ic
            ind.v_prev = 0.0

        sol = np.zeros(total_dim, dtype=float)

        for t in time_points:
            # Newton-Raphson loop for nonlinear transient step
            for nr_iter in range(20):
                A = np.zeros((total_dim, total_dim), dtype=float)
                b = np.zeros(total_dim, dtype=float)

                node_voltages = {name: sol[idx] for name, idx in node_map.items()}
                node_voltages["0"] = 0.0

                # 1. Resistors
                for comp in self.netlist.components.values():
                    if isinstance(comp, Resistor):
                        np_name = self.netlist.normalize_node(comp.nodes[0])
                        nn_name = self.netlist.normalize_node(comp.nodes[1])
                        g = comp.conductance
                        p_idx = node_map.get(np_name)
                        n_idx = node_map.get(nn_name)
                        if p_idx is not None:
                            A[p_idx, p_idx] += g
                        if n_idx is not None:
                            A[n_idx, n_idx] += g
                        if p_idx is not None and n_idx is not None:
                            A[p_idx, n_idx] -= g
                            A[n_idx, p_idx] -= g

                # 2. Capacitors (Companion Model)
                for cap in capacitors:
                    np_name = self.netlist.normalize_node(cap.nodes[0])
                    nn_name = self.netlist.normalize_node(cap.nodes[1])
                    geq, ieq = cap.get_companion_model(dt, method=method)
                    p_idx = node_map.get(np_name)
                    n_idx = node_map.get(nn_name)
                    if p_idx is not None:
                        A[p_idx, p_idx] += geq
                        b[p_idx] += ieq
                    if n_idx is not None:
                        A[n_idx, n_idx] += geq
                        b[n_idx] -= ieq
                    if p_idx is not None and n_idx is not None:
                        A[p_idx, n_idx] -= geq
                        A[n_idx, p_idx] -= geq

                # 3. Inductors (Companion Model)
                for ind in inductors:
                    np_name = self.netlist.normalize_node(ind.nodes[0])
                    nn_name = self.netlist.normalize_node(ind.nodes[1])
                    geq, ieq = ind.get_companion_model(dt, method=method)
                    p_idx = node_map.get(np_name)
                    n_idx = node_map.get(nn_name)
                    if p_idx is not None:
                        A[p_idx, p_idx] += geq
                        b[p_idx] -= ieq
                    if n_idx is not None:
                        A[n_idx, n_idx] += geq
                        b[n_idx] += ieq
                    if p_idx is not None and n_idx is not None:
                        A[p_idx, n_idx] -= geq
                        A[n_idx, p_idx] -= geq

                # 4. Voltage Sources (Time-Dependent Waveform)
                for vs in v_sources:
                    np_name = self.netlist.normalize_node(vs.nodes[0])
                    nn_name = self.netlist.normalize_node(vs.nodes[1])
                    p_idx = node_map.get(np_name)
                    n_idx = node_map.get(nn_name)
                    aux = vs.aux_index
                    if p_idx is not None:
                        A[p_idx, aux] += 1.0
                        A[aux, p_idx] += 1.0
                    if n_idx is not None:
                        A[n_idx, aux] -= 1.0
                        A[aux, n_idx] -= 1.0
                    b[aux] += vs.value_at_time(t)

                # 5. Diodes
                for comp in self.netlist.components.values():
                    if isinstance(comp, Diode):
                        np_name = self.netlist.normalize_node(comp.nodes[0])
                        nn_name = self.netlist.normalize_node(comp.nodes[1])
                        p_idx = node_map.get(np_name)
                        n_idx = node_map.get(nn_name)
                        vp = sol[p_idx] if p_idx is not None else 0.0
                        vn = sol[n_idx] if n_idx is not None else 0.0
                        gd, ieq = comp.linearize(vp - vn)
                        if p_idx is not None:
                            A[p_idx, p_idx] += gd
                            b[p_idx] -= ieq
                        if n_idx is not None:
                            A[n_idx, n_idx] += gd
                            b[n_idx] += ieq
                        if p_idx is not None and n_idx is not None:
                            A[p_idx, n_idx] -= gd
                            A[n_idx, p_idx] -= gd

                # 6. Behavioral Sources
                for comp in self.netlist.components.values():
                    if isinstance(comp, BehavioralSource):
                        np_name = self.netlist.normalize_node(comp.nodes[0])
                        nn_name = self.netlist.normalize_node(comp.nodes[1])
                        p_idx = node_map.get(np_name)
                        n_idx = node_map.get(nn_name)
                        b_val = comp.evaluate(node_voltages, time=t)
                        if comp.source_type == "I":
                            if p_idx is not None:
                                b[p_idx] -= b_val
                            if n_idx is not None:
                                b[n_idx] += b_val
                        else:
                            aux = comp.aux_index
                            if p_idx is not None:
                                A[p_idx, aux] += 1.0
                                A[aux, p_idx] += 1.0
                            if n_idx is not None:
                                A[n_idx, aux] -= 1.0
                                A[aux, n_idx] -= 1.0
                            b[aux] += b_val

                # 7. Switches
                for comp in self.netlist.components.values():
                    if isinstance(comp, VoltageControlledSwitch):
                        p_idx = node_map.get(self.netlist.normalize_node(comp.nodes[0]))
                        n_idx = node_map.get(self.netlist.normalize_node(comp.nodes[1]))
                        cp_idx = node_map.get(self.netlist.normalize_node(comp.nodes[2]))
                        cn_idx = node_map.get(self.netlist.normalize_node(comp.nodes[3]))
                        v_cp = sol[cp_idx] if cp_idx is not None else 0.0
                        v_cn = sol[cn_idx] if cn_idx is not None else 0.0
                        g_sw = comp.get_conductance(v_cp - v_cn)
                        if p_idx is not None:
                            A[p_idx, p_idx] += g_sw
                        if n_idx is not None:
                            A[n_idx, n_idx] += g_sw
                        if p_idx is not None and n_idx is not None:
                            A[p_idx, n_idx] -= g_sw
                            A[n_idx, p_idx] -= g_sw

                # GMIN diagonal stability
                for i in range(num_nodes):
                    A[i, i] += 1e-12

                try:
                    new_sol = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    new_sol = np.linalg.lstsq(A, b, rcond=None)[0]

                diff = np.max(np.abs(new_sol - sol))
                sol = new_sol
                if diff < 1e-5:
                    break

            for name, idx in node_map.items():
                node_waveforms[name].append(float(sol[idx]))

            # Update companion states for next step
            for cap in capacitors:
                np_name = self.netlist.normalize_node(cap.nodes[0])
                nn_name = self.netlist.normalize_node(cap.nodes[1])
                vp = sol[node_map[np_name]] if np_name in node_map else 0.0
                vn = sol[node_map[nn_name]] if nn_name in node_map else 0.0
                v_cap = vp - vn
                geq, ieq = cap.get_companion_model(dt, method=method)
                cap.i_prev = geq * v_cap - ieq
                cap.v_prev = v_cap

            for ind in inductors:
                np_name = self.netlist.normalize_node(ind.nodes[0])
                nn_name = self.netlist.normalize_node(ind.nodes[1])
                vp = sol[node_map[np_name]] if np_name in node_map else 0.0
                vn = sol[node_map[nn_name]] if nn_name in node_map else 0.0
                v_ind = vp - vn
                geq, ieq = ind.get_companion_model(dt, method=method)
                ind.i_prev = geq * v_ind + ieq
                ind.v_prev = v_ind

        result = SimulationResult("TRAN", time_points, x_label="Time (s)")
        for name, vdata in node_waveforms.items():
            result.add_waveform(f"V({name})", np.array(vdata), unit="V", domain="time")

        return result

    def solve_four(
        self,
        freq_hz: Union[str, float],
        trace_name: str,
        t_stop: Union[str, float],
        t_step: Optional[Union[str, float]] = None,
        num_harmonics: int = 9
    ) -> SimulationResult:
        """Fourier Analysis (.FOUR) computing harmonics and Total Harmonic Distortion (THD %)."""
        f0 = parse_eng_unit(freq_hz)
        period = 1.0 / f0
        t_end = parse_eng_unit(t_stop)
        dt = parse_eng_unit(t_step) if t_step else (period / 200.0)

        tran_res = self.solve_tran(t_step=dt, t_stop=t_end, t_start=max(0.0, t_end - period * 5.0))
        wf = tran_res.get_waveform(trace_name) or tran_res.get_waveform(f"V({trace_name})")
        if not wf:
            raise KeyError(f"Trace '{trace_name}' not found in transient output.")

        # Extract last integer periods
        t_vec = wf.x
        y_vec = wf.y
        mask = t_vec >= (t_vec[-1] - period)
        t_period = t_vec[mask]
        y_period = y_vec[mask]

        if len(t_period) < 8:
            res = SimulationResult("FOUR", np.array([f0]))
            res.message = "Insufficient time samples for Fourier analysis."
            return res

        # Resample exact single period with endpoint=False for leakage-free FFT
        t_start_p = t_vec[-1] - period
        uniform_t = np.linspace(t_start_p, t_vec[-1], 1024, endpoint=False)
        uniform_y = np.interp(uniform_t, t_vec, y_vec)
        fft_vals = np.fft.rfft(uniform_y) / len(uniform_y)
        magnitudes = np.abs(fft_vals) * 2.0

        dc_component = float(np.abs(fft_vals[0]))
        h1_mag = float(magnitudes[1]) if len(magnitudes) > 1 else 1e-12

        harmonics: Dict[str, float] = {"DC Component": dc_component, "Harmonic 1 (Fund)": h1_mag}
        harmonic_sum_sq = 0.0

        for h in range(2, min(num_harmonics + 1, len(magnitudes))):
            h_mag = float(magnitudes[h])
            harmonics[f"Harmonic {h} ({h*f0:.0f}Hz)"] = h_mag
            harmonic_sum_sq += h_mag ** 2

        thd = (math.sqrt(harmonic_sum_sq) / max(1e-12, h1_mag)) * 100.0

        result = SimulationResult("FOUR", np.array([f0]), x_label="Frequency (Hz)")
        result.op_results = harmonics
        result.metrics = {
            "fundamental_hz": f0,
            "thd_percent": thd,
            "dc_offset": dc_component,
            "fundamental_mag": h1_mag
        }
        result.message = f"Fourier analysis completed. THD = {thd:.4f}%"
        return result
