"""Circuit component definitions for MNA (Modified Nodal Analysis) solver.
Includes Passives, Semiconductor Devices (Diode, BJT, MOSFET, JFET), Controlled/Behavioral Sources,
Switches, Subcircuits, and Op-Amp macromodels compliant with SPICE/LTspice standards.
"""

from __future__ import annotations
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from CORE.common_math import parse_eng_unit


class Component:
    """Base class for all circuit components."""
    
    def __init__(self, name: str, nodes: List[str]):
        self.name = name.strip()
        self.nodes = [n.strip() for n in nodes]
        self.params: Dict[str, Any] = {}

    def get_node(self, terminal_index: int) -> str:
        if 0 <= terminal_index < len(self.nodes):
            return self.nodes[terminal_index]
        return "0"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name} nodes={self.nodes}>"


class Resistor(Component):
    """Linear Resistor (R)."""
    
    def __init__(self, name: str, node_p: str, node_n: str, resistance: Union[str, float], tc1: float = 0.0, tc2: float = 0.0):
        super().__init__(name, [node_p, node_n])
        self.value = parse_eng_unit(resistance)
        self.tc1 = tc1
        self.tc2 = tc2
        if self.value <= 0:
            raise ValueError(f"Resistance for {name} must be positive, got {self.value}")

    @property
    def conductance(self) -> float:
        return 1.0 / self.value

    def get_resistance_at_temp(self, temp_c: float, tnom: float = 27.0) -> float:
        dt = temp_c - tnom
        return self.value * (1.0 + self.tc1 * dt + self.tc2 * (dt ** 2))


class Capacitor(Component):
    """Linear Capacitor (C) with companion model integration states."""
    
    def __init__(
        self,
        name: str,
        node_p: str,
        node_n: str,
        capacitance: Union[str, float],
        ic: float = 0.0,
        rser: float = 0.0,
        lser: float = 0.0
    ):
        super().__init__(name, [node_p, node_n])
        self.value = parse_eng_unit(capacitance)
        self.ic = ic  # Initial condition voltage
        self.rser = rser  # Equivalent series resistance
        self.lser = lser  # Equivalent series inductance
        self.v_prev = ic
        self.i_prev = 0.0

    def get_companion_model(self, dt: float, method: str = "trapezoidal") -> Tuple[float, float]:
        """Calculates equivalent conductance G_eq and current source I_eq for time-stepping."""
        if method == "backward_euler":
            geq = self.value / dt
            ieq = geq * self.v_prev
        else:  # trapezoidal
            geq = 2.0 * self.value / dt
            ieq = geq * self.v_prev + self.i_prev
        return geq, ieq


class Inductor(Component):
    """Linear Inductor (L) with companion model integration states."""
    
    def __init__(
        self,
        name: str,
        node_p: str,
        node_n: str,
        inductance: Union[str, float],
        ic: float = 0.0,
        rser: float = 0.0,
        rpar: float = 0.0
    ):
        super().__init__(name, [node_p, node_n])
        self.value = parse_eng_unit(inductance)
        self.ic = ic  # Initial current
        self.rser = rser  # Series winding resistance
        self.rpar = rpar  # Parallel resistance
        self.v_prev = 0.0
        self.i_prev = ic
        self.aux_index: int = -1  # Assigned by MNA solver

    def get_companion_model(self, dt: float, method: str = "trapezoidal") -> Tuple[float, float]:
        """Calculates equivalent conductance G_eq and current source I_eq."""
        if method == "backward_euler":
            geq = dt / self.value
            ieq = self.i_prev
        else:  # trapezoidal
            geq = dt / (2.0 * self.value)
            ieq = self.i_prev + geq * self.v_prev
        return geq, ieq


class CoupledInductors(Component):
    """Mutual Inductance / Transformer Coupling (K)."""

    def __init__(self, name: str, inductor1: str, inductor2: str, coupling_k: float = 1.0):
        super().__init__(name, [])
        self.inductor1 = inductor1.strip().upper()
        self.inductor2 = inductor2.strip().upper()
        self.k = float(coupling_k)
        if not (-1.0 <= self.k <= 1.0):
            raise ValueError(f"Coupling coefficient k must be between -1.0 and 1.0, got {self.k}")


class VoltageSource(Component):
    """Independent Voltage Source (DC, AC, Sine, Pulse, PWL, SFFM, EXP)."""
    
    def __init__(
        self,
        name: str,
        node_p: str,
        node_n: str,
        dc: Union[str, float] = 0.0,
        ac_mag: float = 0.0,
        ac_phase: float = 0.0,
        waveform_type: str = "DC",
        wave_params: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, [node_p, node_n])
        self.dc = parse_eng_unit(dc)
        self.ac_mag = parse_eng_unit(ac_mag) if ac_mag else 0.0
        self.ac_phase = float(ac_phase)
        self.waveform_type = waveform_type.upper()
        self.wave_params = wave_params or {}
        self.aux_index: int = -1  # Row/col in MNA matrix

    def value_at_time(self, t: float) -> float:
        """Evaluates time-dependent voltage."""
        if self.waveform_type == "DC":
            return self.dc
        
        elif self.waveform_type in ("SINE", "SIN"):
            offset = self.wave_params.get("offset", self.dc)
            amplitude = self.wave_params.get("amplitude", 1.0)
            freq = self.wave_params.get("freq", 1000.0)
            delay = self.wave_params.get("delay", 0.0)
            damping = self.wave_params.get("damping", 0.0)
            phase = self.wave_params.get("phase", 0.0)
            if t < delay:
                return offset
            dt = t - delay
            damp_factor = math.exp(-damping * dt) if damping > 0 else 1.0
            return offset + amplitude * damp_factor * math.sin(2.0 * math.pi * freq * dt + math.radians(phase))
        
        elif self.waveform_type == "PULSE":
            v1 = self.wave_params.get("v1", 0.0)
            v2 = self.wave_params.get("v2", self.dc if self.dc != 0 else 5.0)
            td = self.wave_params.get("td", 0.0)
            tr = max(1e-15, self.wave_params.get("tr", 1e-9))
            tf = max(1e-15, self.wave_params.get("tf", 1e-9))
            ton = self.wave_params.get("ton", 1e-3)
            period = self.wave_params.get("period", 2e-3)

            if t < td:
                return v1
            t_rel = (t - td) % period
            if t_rel < tr:
                return v1 + (v2 - v1) * (t_rel / tr)
            elif t_rel < tr + ton:
                return v2
            elif t_rel < tr + ton + tf:
                return v2 - (v2 - v1) * ((t_rel - (tr + ton)) / tf)
            else:
                return v1

        elif self.waveform_type == "PWL":
            # Piecewise Linear: points = [(t0, v0), (t1, v1), ...]
            points = self.wave_params.get("points", [(0.0, self.dc)])
            if not points:
                return self.dc
            if t <= points[0][0]:
                return points[0][1]
            if t >= points[-1][0]:
                return points[-1][1]
            for i in range(len(points) - 1):
                t0, v0 = points[i]
                t1, v1 = points[i+1]
                if t0 <= t <= t1:
                    frac = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
                    return v0 + (v1 - v0) * frac
            return points[-1][1]

        elif self.waveform_type == "EXP":
            # Exponential: v1, v2, td1, tau1, td2, tau2
            v1 = self.wave_params.get("v1", 0.0)
            v2 = self.wave_params.get("v2", 1.0)
            td1 = self.wave_params.get("td1", 0.0)
            tau1 = max(1e-15, self.wave_params.get("tau1", 1e-3))
            td2 = self.wave_params.get("td2", 2e-3)
            tau2 = max(1e-15, self.wave_params.get("tau2", 1e-3))
            if t < td1:
                return v1
            elif t < td2:
                return v1 + (v2 - v1) * (1.0 - math.exp(-(t - td1) / tau1))
            else:
                v_at_td2 = v1 + (v2 - v1) * (1.0 - math.exp(-(td2 - td1) / tau1))
                return v1 + (v_at_td2 - v1) * math.exp(-(t - td2) / tau2)

        elif self.waveform_type == "SFFM":
            # Single-Frequency FM: vo, va, fc, mdi, fs
            vo = self.wave_params.get("vo", 0.0)
            va = self.wave_params.get("va", 1.0)
            fc = self.wave_params.get("fc", 1000.0)
            mdi = self.wave_params.get("mdi", 1.0)
            fs = self.wave_params.get("fs", 100.0)
            return vo + va * math.sin(2.0 * math.pi * fc * t + mdi * math.sin(2.0 * math.pi * fs * t))

        return self.dc


class CurrentSource(Component):
    """Independent Current Source (enters node_p, leaves node_n)."""
    
    def __init__(
        self,
        name: str,
        node_p: str,
        node_n: str,
        dc: Union[str, float] = 0.0,
        ac_mag: float = 0.0,
        ac_phase: float = 0.0,
        waveform_type: str = "DC",
        wave_params: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, [node_p, node_n])
        self.dc = parse_eng_unit(dc)
        self.ac_mag = parse_eng_unit(ac_mag) if ac_mag else 0.0
        self.ac_phase = float(ac_phase)
        self.waveform_type = waveform_type.upper()
        self.wave_params = wave_params or {}

    def value_at_time(self, t: float) -> float:
        if self.waveform_type == "DC":
            return self.dc
        elif self.waveform_type in ("SINE", "SIN"):
            offset = self.wave_params.get("offset", self.dc)
            amplitude = self.wave_params.get("amplitude", 1.0)
            freq = self.wave_params.get("freq", 1000.0)
            phase = self.wave_params.get("phase", 0.0)
            return offset + amplitude * math.sin(2.0 * math.pi * freq * t + math.radians(phase))
        elif self.waveform_type == "PULSE":
            i1 = self.wave_params.get("i1", 0.0)
            i2 = self.wave_params.get("i2", self.dc if self.dc != 0 else 1.0)
            td = self.wave_params.get("td", 0.0)
            tr = max(1e-15, self.wave_params.get("tr", 1e-9))
            tf = max(1e-15, self.wave_params.get("tf", 1e-9))
            ton = self.wave_params.get("ton", 1e-3)
            period = self.wave_params.get("period", 2e-3)
            if t < td:
                return i1
            t_rel = (t - td) % period
            if t_rel < tr:
                return i1 + (i2 - i1) * (t_rel / tr)
            elif t_rel < tr + ton:
                return i2
            elif t_rel < tr + ton + tf:
                return i2 - (i2 - i1) * ((t_rel - (tr + ton)) / tf)
            else:
                return i1
        return self.dc


class Diode(Component):
    """Semiconductor Diode using Shockley model with series resistance and breakdown (Zener)."""
    
    def __init__(
        self,
        name: str,
        node_anode: str,
        node_cathode: str,
        model_name: str = "D1N4148",
        is_sat: float = 1e-14,
        n_ideal: float = 1.0,
        vt: float = 0.02585,  # ~26mV at 300K
        rs: float = 0.0,
        cjo: float = 0.0,
        bv: float = 100.0,   # Breakdown voltage
        ibv: float = 1e-3    # Breakdown current
    ):
        super().__init__(name, [node_anode, node_cathode])
        self.model_name = model_name
        self.is_sat = is_sat
        self.n_ideal = n_ideal
        self.vt = vt
        self.rs = rs
        self.cjo = cjo
        self.bv = bv
        self.ibv = ibv
        self.vd_prev = 0.6  # Initial guess

    def linearize(self, vd: float) -> Tuple[float, float]:
        """Calculates linearized companion conductance gd and current id_eq for Newton-Raphson iteration.
        Includes forward conduction and reverse breakdown (Zener).
        """
        vd_limited = max(-self.bv * 1.2, min(vd, 1.2))  # Voltage limiting for stability
        nvt = self.n_ideal * self.vt

        if vd_limited >= -5.0 * nvt:
            # Forward and normal reverse region
            exp_term = math.exp(min(40.0, vd_limited / nvt))
            id_val = self.is_sat * (exp_term - 1.0)
            gd = max(1e-12, (self.is_sat / nvt) * exp_term)
        else:
            # Deep reverse / breakdown
            if self.bv > 0 and vd_limited <= -self.bv:
                # Breakdown region
                v_zener = -vd_limited - self.bv
                exp_z = math.exp(min(40.0, v_zener / (nvt * 2.0)))
                id_val = -self.ibv * exp_z
                gd = max(1e-12, (self.ibv / (nvt * 2.0)) * exp_z)
            else:
                id_val = -self.is_sat
                gd = 1e-12

        ieq = id_val - gd * vd_limited
        return gd, ieq


class BJT(Component):
    """Bipolar Junction Transistor (NPN / PNP standard Ebers-Moll / Gummel-Poon model)."""
    
    def __init__(
        self,
        name: str,
        node_c: str,
        node_b: str,
        node_e: str,
        model_name: str = "2N2222",
        bjt_type: str = "NPN",
        beta_f: float = 100.0,
        beta_r: float = 1.0,
        is_sat: float = 1e-14,
        vbe_on: float = 0.7,
        vaf: float = 100.0,  # Early voltage
        vt: float = 0.02585
    ):
        super().__init__(name, [node_c, node_b, node_e])
        self.model_name = model_name
        self.bjt_type = bjt_type.upper()
        self.beta_f = beta_f
        self.beta_r = beta_r
        self.is_sat = is_sat
        self.vbe_on = vbe_on
        self.vaf = vaf
        self.vt = vt

    def linearize(self, v_be: float, v_ce: float) -> Dict[str, float]:
        """Calculates linearized conductance stamps (g_pi, g_m, g_o, i_eq) for BJT."""
        sign = 1.0 if self.bjt_type == "NPN" else -1.0
        vbe = sign * v_be
        vce = sign * v_ce
        vbe_lim = max(-5.0, min(vbe, 0.9))

        exp_be = math.exp(min(40.0, vbe_lim / self.vt))
        ib = (self.is_sat / self.beta_f) * (exp_be - 1.0)
        ic = self.is_sat * (exp_be - 1.0) * (1.0 + max(0.0, vce) / self.vaf)

        g_pi = max(1e-12, (self.is_sat / (self.beta_f * self.vt)) * exp_be)
        g_m = max(1e-12, (self.is_sat / self.vt) * exp_be * (1.0 + max(0.0, vce) / self.vaf))
        g_o = max(1e-12, ic / self.vaf if self.vaf > 0 else 1e-6)

        return {
            "g_pi": g_pi,
            "g_m": g_m,
            "g_o": g_o,
            "ib": ib,
            "ic": ic,
            "ie": ib + ic
        }


class MOSFET(Component):
    """MOSFET Transistor (NMOS / PMOS Shichman-Hodges Level 1-3 Model)."""

    def __init__(
        self,
        name: str,
        node_d: str,
        node_g: str,
        node_s: str,
        node_b: Optional[str] = None,
        model_name: str = "IRF540N",
        mos_type: str = "NMOS",
        vto: float = 2.0,
        kp: float = 0.02,
        w: float = 1e-4,
        l: float = 1e-6,
        lambda_param: float = 0.01,
        rd: float = 0.01,
        rs: float = 0.01
    ):
        body_node = node_b if node_b else node_s
        super().__init__(name, [node_d, node_g, node_s, body_node])
        self.model_name = model_name
        self.mos_type = mos_type.upper()
        self.vto = vto
        self.kp = kp
        self.w = w
        self.l = l
        self.beta = kp * (w / l)
        self.lambda_param = lambda_param
        self.rd = rd
        self.rs = rs

    def linearize(self, v_gs: float, v_ds: float) -> Dict[str, float]:
        """Linearized transconductance gm, output conductance gds, and drain current Id."""
        sign = 1.0 if self.mos_type == "NMOS" else -1.0
        vgs = sign * v_gs
        vds = max(0.0, sign * v_ds)
        v_ov = vgs - self.vto

        if v_ov <= 0:
            # Cutoff region
            id_val = 1e-12
            gm = 1e-12
            gds = 1e-12
        elif vds < v_ov:
            # Triode / Linear / Ohmic region
            id_val = self.beta * (v_ov * vds - 0.5 * vds ** 2) * (1.0 + self.lambda_param * vds)
            gm = max(1e-12, self.beta * vds * (1.0 + self.lambda_param * vds))
            gds = max(1e-12, self.beta * (v_ov - vds) * (1.0 + self.lambda_param * vds) + self.lambda_param * id_val)
        else:
            # Saturation / Active region
            id_val = 0.5 * self.beta * (v_ov ** 2) * (1.0 + self.lambda_param * vds)
            gm = max(1e-12, self.beta * v_ov * (1.0 + self.lambda_param * vds))
            gds = max(1e-12, 0.5 * self.beta * (v_ov ** 2) * self.lambda_param)

        return {
            "id": id_val,
            "gm": gm,
            "gds": gds
        }


class JFET(Component):
    """Junction Field Effect Transistor (NJFET / PJFET)."""

    def __init__(
        self,
        name: str,
        node_d: str,
        node_g: str,
        node_s: str,
        model_name: str = "2N5457",
        jfet_type: str = "NJFET",
        vto: float = -2.0,
        beta: float = 1e-3,
        lambda_param: float = 0.01
    ):
        super().__init__(name, [node_d, node_g, node_s])
        self.model_name = model_name
        self.jfet_type = jfet_type.upper()
        self.vto = vto
        self.beta = beta
        self.lambda_param = lambda_param

    def linearize(self, v_gs: float, v_ds: float) -> Dict[str, float]:
        sign = 1.0 if self.jfet_type == "NJFET" else -1.0
        vgs = sign * v_gs
        vds = max(0.0, sign * v_ds)
        v_gst = vgs - self.vto

        if v_gst <= 0:
            return {"id": 1e-12, "gm": 1e-12, "gds": 1e-12}
        elif vds < v_gst:
            id_val = self.beta * (2.0 * v_gst * vds - vds ** 2) * (1.0 + self.lambda_param * vds)
            gm = self.beta * 2.0 * vds * (1.0 + self.lambda_param * vds)
            gds = self.beta * 2.0 * (v_gst - vds) + self.lambda_param * id_val
        else:
            id_val = self.beta * (v_gst ** 2) * (1.0 + self.lambda_param * vds)
            gm = self.beta * 2.0 * v_gst * (1.0 + self.lambda_param * vds)
            gds = self.beta * (v_gst ** 2) * self.lambda_param

        return {"id": id_val, "gm": max(1e-12, gm), "gds": max(1e-12, gds)}


class VoltageControlledSwitch(Component):
    """Voltage-Controlled Switch (SW / S)."""

    def __init__(
        self,
        name: str,
        node_sw_p: str,
        node_sw_n: str,
        node_ctrl_p: str,
        node_ctrl_n: str,
        model_name: str = "SW_DEFAULT",
        ron: float = 1.0,
        roff: float = 1e6,
        von: float = 1.0,
        voff: float = 0.0
    ):
        super().__init__(name, [node_sw_p, node_sw_n, node_ctrl_p, node_ctrl_n])
        self.model_name = model_name
        self.ron = max(1e-6, ron)
        self.roff = max(1.0, roff)
        self.von = von
        self.voff = voff

    def get_conductance(self, v_ctrl: float) -> float:
        """Returns dynamic conductance with smooth transition between Ron and Roff."""
        v_mid = 0.5 * (self.von + self.voff)
        v_span = max(1e-3, abs(self.von - self.voff))
        # Logistic / tanh smooth switching curve
        k = 6.0 / v_span
        s = 1.0 / (1.0 + math.exp(-k * (v_ctrl - v_mid))) if self.von > self.voff else 1.0 / (1.0 + math.exp(k * (v_ctrl - v_mid)))
        r_eff = self.roff * (1.0 - s) + self.ron * s
        return 1.0 / max(1e-6, r_eff)


class CurrentControlledSwitch(Component):
    """Current-Controlled Switch (CSW / W)."""

    def __init__(
        self,
        name: str,
        node_sw_p: str,
        node_sw_n: str,
        vsource_ctrl: str,
        model_name: str = "CSW_DEFAULT",
        ron: float = 1.0,
        roff: float = 1e6,
        ion: float = 1e-3,
        ioff: float = 0.0
    ):
        super().__init__(name, [node_sw_p, node_sw_n])
        self.vsource_ctrl = vsource_ctrl.strip().upper()
        self.model_name = model_name
        self.ron = max(1e-6, ron)
        self.roff = max(1.0, roff)
        self.ion = ion
        self.ioff = ioff

    def get_conductance(self, i_ctrl: float) -> float:
        i_mid = 0.5 * (self.ion + self.ioff)
        i_span = max(1e-6, abs(self.ion - self.ioff))
        k = 6.0 / i_span
        s = 1.0 / (1.0 + math.exp(-k * (i_ctrl - i_mid)))
        r_eff = self.roff * (1.0 - s) + self.ron * s
        return 1.0 / max(1e-6, r_eff)


class VCVS(Component):
    """Voltage-Controlled Voltage Source (E-source)."""
    
    def __init__(
        self,
        name: str,
        node_out_p: str,
        node_out_n: str,
        node_in_p: str,
        node_in_n: str,
        gain: Union[str, float] = 1.0
    ):
        super().__init__(name, [node_out_p, node_out_n, node_in_p, node_in_n])
        self.gain = parse_eng_unit(gain)
        self.aux_index: int = -1


class VCCS(Component):
    """Voltage-Controlled Current Source (G-source / Transconductance)."""

    def __init__(
        self,
        name: str,
        node_out_p: str,
        node_out_n: str,
        node_in_p: str,
        node_in_n: str,
        transconductance: Union[str, float] = 1.0
    ):
        super().__init__(name, [node_out_p, node_out_n, node_in_p, node_in_n])
        self.gm = parse_eng_unit(transconductance)


class CCVS(Component):
    """Current-Controlled Voltage Source (H-source / Transresistance)."""

    def __init__(
        self,
        name: str,
        node_out_p: str,
        node_out_n: str,
        vsource_ctrl: str,
        transresistance: Union[str, float] = 1.0
    ):
        super().__init__(name, [node_out_p, node_out_n])
        self.vsource_ctrl = vsource_ctrl.strip().upper()
        self.rm = parse_eng_unit(transresistance)
        self.aux_index: int = -1


class CCCS(Component):
    """Current-Controlled Current Source (F-source / Current Gain)."""

    def __init__(
        self,
        name: str,
        node_out_p: str,
        node_out_n: str,
        vsource_ctrl: str,
        gain: Union[str, float] = 1.0
    ):
        super().__init__(name, [node_out_p, node_out_n])
        self.vsource_ctrl = vsource_ctrl.strip().upper()
        self.gain = parse_eng_unit(gain)


class BehavioralSource(Component):
    """Arbitrary Behavioral Source (B-source) evaluating nonlinear math expressions:
    e.g., V = V(in1)*V(in2) + 2.5, or I = sin(2*pi*1000*time) * exp(-V(ctrl)).
    """

    def __init__(
        self,
        name: str,
        node_p: str,
        node_n: str,
        source_type: str = "V",  # 'V' or 'I'
        expression: str = "0.0"
    ):
        super().__init__(name, [node_p, node_n])
        self.source_type = source_type.upper()
        self.expression = expression.strip()
        self.aux_index: int = -1 if self.source_type == "V" else -2

    def evaluate(self, node_voltages: Dict[str, float], time: float = 0.0) -> float:
        """Safely evaluates expression given current node voltages and time."""
        safe_env = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "exp": math.exp,
            "log": math.log,
            "ln": math.log,
            "log10": math.log10,
            "sqrt": math.sqrt,
            "abs": abs,
            "min": min,
            "max": max,
            "pi": math.pi,
            "e": math.e,
            "time": time,
            "t": time,
            "limit": lambda val, low, high: max(low, min(val, high)),
            "stp": lambda x: 1.0 if x >= 0 else 0.0,
            "sgn": lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0),
        }

        # Voltage function helper V(...)
        def v_func(node1: str, node2: Optional[str] = None) -> float:
            v1 = node_voltages.get(str(node1).strip(), 0.0) if str(node1).strip() != "0" else 0.0
            if node2 is None:
                return v1
            v2 = node_voltages.get(str(node2).strip(), 0.0) if str(node2).strip() != "0" else 0.0
            return v1 - v2

        safe_env["V"] = v_func
        safe_env["v"] = v_func

        # Replace standard SPICE token syntax if present, e.g. V(n1)
        expr_py = self.expression
        # Transform ^ to **
        expr_py = expr_py.replace("^", "**")

        try:
            val = eval(expr_py, {"__builtins__": {}}, safe_env)
            return float(val)
        except Exception:
            return 0.0


class OpAmpModel(Component):
    """Macromodel for Operational Amplifiers (LM741, TL072, Ideal OpAmp)."""

    def __init__(
        self,
        name: str,
        node_out: str,
        node_in_inv: str,
        node_in_noninv: str,
        node_vpos: Optional[str] = None,
        node_vneg: Optional[str] = None,
        model_name: str = "LM741",
        avol: float = 200000.0,
        gbw: float = 1e6,
        rin: float = 2e6,
        rout: float = 75.0,
        slew_rate: float = 0.5e6  # V/s
    ):
        vpos = node_vpos if node_vpos else "VCC"
        vneg = node_vneg if node_vneg else "VEE"
        super().__init__(name, [node_out, node_in_inv, node_in_noninv, vpos, vneg])
        self.model_name = model_name
        self.avol = avol
        self.gbw = gbw
        self.rin = rin
        self.rout = rout
        self.slew_rate = slew_rate
        self.aux_index: int = -1


class SubcircuitDefinition:
    """Definition template for a SPICE .SUBCKT block."""

    def __init__(self, name: str, pins: List[str], params: Optional[Dict[str, Any]] = None):
        self.name = name.strip().upper()
        self.pins = [p.strip() for p in pins]
        self.params = params or {}
        self.internal_components: List[Component] = []
        self.models: Dict[str, Dict[str, Any]] = {}

    def add_component(self, comp: Component) -> None:
        self.internal_components.append(comp)

    def instantiate(self, instance_name: str, connected_nodes: List[str], inst_params: Optional[Dict[str, Any]] = None) -> List[Component]:
        """Flattens and expands subcircuit instance with prefixed internal node names."""
        if len(connected_nodes) != len(self.pins):
            raise ValueError(
                f"Subcircuit '{self.name}' expects {len(self.pins)} pins ({self.pins}), but got {len(connected_nodes)} ({connected_nodes})"
            )
        
        node_map = dict(zip(self.pins, connected_nodes))
        # Ground node '0' is always global
        node_map["0"] = "0"
        node_map["GND"] = "0"
        node_map["gnd"] = "0"

        expanded_components: List[Component] = []
        for comp in self.internal_components:
            # Create mapped node list
            mapped_nodes = [node_map.get(n, f"{instance_name}_{n}") for n in comp.nodes]
            # Clone component with unique name
            c_type = type(comp)
            c_name = f"{instance_name}_{comp.name}"
            
            if c_type == Resistor:
                expanded_components.append(Resistor(c_name, mapped_nodes[0], mapped_nodes[1], getattr(comp, "value", 1000.0)))
            elif c_type == Capacitor:
                expanded_components.append(Capacitor(c_name, mapped_nodes[0], mapped_nodes[1], getattr(comp, "value", 1e-6)))
            elif c_type == Inductor:
                expanded_components.append(Inductor(c_name, mapped_nodes[0], mapped_nodes[1], getattr(comp, "value", 1e-3)))
            elif c_type == Diode:
                expanded_components.append(Diode(c_name, mapped_nodes[0], mapped_nodes[1], getattr(comp, "model_name", "D1N4148")))
            elif c_type == BJT:
                expanded_components.append(BJT(c_name, mapped_nodes[0], mapped_nodes[1], mapped_nodes[2], getattr(comp, "model_name", "2N2222"), getattr(comp, "bjt_type", "NPN")))
            elif c_type == MOSFET:
                b_node = mapped_nodes[3] if len(mapped_nodes) > 3 else mapped_nodes[2]
                expanded_components.append(MOSFET(c_name, mapped_nodes[0], mapped_nodes[1], mapped_nodes[2], b_node, getattr(comp, "model_name", "IRF540N"), getattr(comp, "mos_type", "NMOS")))
            elif c_type == VCVS:
                expanded_components.append(VCVS(c_name, mapped_nodes[0], mapped_nodes[1], mapped_nodes[2], mapped_nodes[3], getattr(comp, "gain", 1.0)))
            else:
                # General fallback
                clone = type(comp)(c_name, *mapped_nodes)
                expanded_components.append(clone)

        return expanded_components


class SubcircuitInstance(Component):
    """Subcircuit Instance in SPICE netlist (X-prefix)."""

    def __init__(self, name: str, subckt_name: str, connected_nodes: List[str], params: Optional[Dict[str, Any]] = None):
        super().__init__(name, connected_nodes)
        self.subckt_name = subckt_name.strip().upper()
        self.instance_params = params or {}
