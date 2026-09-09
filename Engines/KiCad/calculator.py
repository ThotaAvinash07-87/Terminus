"""KiCad Built-in Engineering Calculators for TerminusECE.

Implements standard PCB and circuit engineering calculations:
1. IPC-2152 / IPC-2221 PCB Track Width vs Current Capacity
2. PCB Via Resistance, Capacitance, Inductance, & Ampacity
3. RF Microstrip Transmission Line Impedance (Z0)
4. Astable 555 Timer Frequency & Duty Cycle
5. Op-Amp Closed-Loop Gain & Feedback Network
6. Voltage Regulator (LM317) Resistor Divider
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple


class KiCadCalculator:
    """Built-in KiCad Engineering Calculators."""

    @staticmethod
    def calculate_track_width(
        current_amp: float,
        temp_rise_deg_c: float = 10.0,
        copper_thickness_oz: float = 1.0,
        track_length_mm: float = 50.0
    ) -> Dict[str, Any]:
        """Calculates required PCB track width using IPC-2221 formulas.

        Formula: Area [mils^2] = (I / (k * (dT^0.44)))^(1/0.725)
        k = 0.048 for external layers, 0.024 for internal layers.
        """
        i = max(0.001, current_amp)
        dt = max(1.0, temp_rise_deg_c)
        thickness_mil = copper_thickness_oz * 1.378  # 1 oz = 1.378 mil = 35 um

        # External layer
        k_ext = 0.048
        area_ext_mil2 = (i / (k_ext * (dt ** 0.44))) ** (1.0 / 0.725)
        width_ext_mil = area_ext_mil2 / thickness_mil
        width_ext_mm = width_ext_mil * 0.0254

        # Internal layer
        k_int = 0.024
        area_int_mil2 = (i / (k_int * (dt ** 0.44))) ** (1.0 / 0.725)
        width_int_mil = area_int_mil2 / thickness_mil
        width_int_mm = width_int_mil * 0.0254

        # Resistance & Voltage drop for external track
        # Copper resistivity rho = 1.724e-8 ohm-meter
        area_ext_m2 = (width_ext_mm * 1e-3) * (thickness_mil * 0.0254 * 1e-3)
        res_ext_ohm = (1.724e-8 * (track_length_mm * 1e-3)) / max(1e-12, area_ext_m2)
        v_drop = i * res_ext_ohm
        p_loss = (i ** 2) * res_ext_ohm

        return {
            "current_amp": i,
            "temp_rise_c": dt,
            "copper_oz": copper_thickness_oz,
            "length_mm": track_length_mm,
            "external_width_mm": width_ext_mm,
            "external_width_mil": width_ext_mil,
            "internal_width_mm": width_int_mm,
            "internal_width_mil": width_int_mil,
            "resistance_ohm": res_ext_ohm,
            "voltage_drop_v": v_drop,
            "power_loss_w": p_loss,
        }

    @staticmethod
    def calculate_via_parasitics(
        drill_dia_mm: float = 0.3,
        pad_dia_mm: float = 0.6,
        pcb_thickness_mm: float = 1.6,
        copper_plating_thickness_um: float = 25.0,
        er_substrate: float = 4.5
    ) -> Dict[str, Any]:
        """Calculates via electrical parameters: DC resistance, inductance, capacitance, and ampacity."""
        drill_m = drill_dia_mm * 1e-3
        h_m = pcb_thickness_mm * 1e-3
        t_m = copper_plating_thickness_um * 1e-6

        # Cross-sectional area of hollow copper barrel = pi * d_mean * t
        d_mean = drill_m - t_m
        area_m2 = math.pi * d_mean * t_m

        # Via Resistance (approx 1.724e-8 ohm-m)
        r_dc = (1.724e-8 * h_m) / max(1e-12, area_m2)

        # Via Inductance (Johnson & Graham formula): L = 5.08 * h * [ln(4*h/d) + 1] nH (with h, d in inches)
        h_in = pcb_thickness_mm / 25.4
        d_in = drill_dia_mm / 25.4
        l_nh = 5.08 * h_in * (math.log(4.0 * h_in / max(1e-6, d_in)) + 1.0)

        # Via Capacitance: C = (1.41 * Er * T * D1) / (D2 - D1) pF (approx)
        d1_in = pad_dia_mm / 25.4
        d2_in = (pad_dia_mm + 0.5) / 25.4
        c_pf = (1.41 * er_substrate * h_in * d1_in) / max(1e-6, d2_in - d1_in)

        # Safe Current carrying capacity (approx 1.5A for 0.3mm drill @ 10C rise)
        i_safe = 1.75 * math.sqrt(drill_dia_mm / 0.3)

        return {
            "drill_mm": drill_dia_mm,
            "pad_mm": pad_dia_mm,
            "pcb_thickness_mm": pcb_thickness_mm,
            "resistance_mohm": r_dc * 1000.0,
            "inductance_nh": max(0.01, l_nh),
            "capacitance_pf": max(0.01, c_pf),
            "max_current_amp": i_safe,
        }

    @staticmethod
    def calculate_microstrip(
        width_mm: float,
        height_mm: float,
        substrate_er: float = 4.5,
        copper_thickness_um: float = 35.0
    ) -> Dict[str, Any]:
        """Calculates Characteristic Impedance (Z0) of microstrip transmission line using Wheeler / IPC formulas."""
        w = max(0.01, width_mm)
        h = max(0.01, height_mm)
        er = max(1.0, substrate_er)

        ratio = w / h
        if ratio < 1.0:
            eff_er = (er + 1.0)/2.0 + ((er - 1.0)/2.0) * ((1.0 / math.sqrt(1.0 + 12.0/ratio)) + 0.04 * (1.0 - ratio)**2)
            z0 = (60.0 / math.sqrt(eff_er)) * math.log(8.0/ratio + 0.25*ratio)
        else:
            eff_er = (er + 1.0)/2.0 + ((er - 1.0)/2.0) * (1.0 / math.sqrt(1.0 + 12.0/ratio))
            z0 = (120.0 * math.pi) / (math.sqrt(eff_er) * (ratio + 1.393 + 0.667 * math.log(ratio + 1.444)))

        # Propagation delay (ps/inch) = 84.72 * sqrt(eff_er)
        delay_ps_mm = (math.sqrt(eff_er) / 3e8) * 1e9  # ps per mm

        return {
            "width_mm": w,
            "height_mm": h,
            "substrate_er": er,
            "effective_er": eff_er,
            "z0_ohms": z0,
            "delay_ps_per_mm": delay_ps_mm,
        }

    @staticmethod
    def calculate_555_timer(r1_ohms: float, r2_ohms: float, c_farads: float) -> Dict[str, Any]:
        """Calculates astable 555 timer output frequency, duty cycle, and timings."""
        r1 = max(1.0, r1_ohms)
        r2 = max(1.0, r2_ohms)
        c = max(1e-15, c_farads)

        t_high = 0.693 * (r1 + r2) * c
        t_low = 0.693 * r2 * c
        period = t_high + t_low
        freq = 1.0 / max(1e-15, period)
        duty_cycle = (t_high / max(1e-15, period)) * 100.0

        return {
            "r1_ohms": r1,
            "r2_ohms": r2,
            "c_farads": c,
            "freq_hz": freq,
            "period_s": period,
            "t_high_s": t_high,
            "t_low_s": t_low,
            "duty_cycle_pct": duty_cycle,
        }

    @staticmethod
    def calculate_opamp_gain(op_type: str, r1_ohms: float, rf_ohms: float) -> Dict[str, Any]:
        """Calculates inverting or non-inverting op-amp closed loop gain."""
        r1 = max(1.0, r1_ohms)
        rf = max(0.0, rf_ohms)
        is_inv = "inv" in op_type.lower() and "non" not in op_type.lower()

        if is_inv:
            gain_v_v = -(rf / r1)
        else:
            gain_v_v = 1.0 + (rf / r1)

        gain_db = 20.0 * math.log10(max(1e-9, abs(gain_v_v)))
        return {
            "circuit_type": "Inverting" if is_inv else "Non-Inverting",
            "r1_ohms": r1,
            "rf_ohms": rf,
            "gain_v_per_v": gain_v_v,
            "gain_db": gain_db,
        }

    @staticmethod
    def calculate_regulator_divider(target_vout: float, vref: float = 1.25, r1_ohms: float = 240.0) -> Dict[str, Any]:
        """Calculates LM317 / linear regulator R2 for desired Vout: Vout = Vref * (1 + R2/R1)."""
        vout = max(vref, target_vout)
        r1 = max(1.0, r1_ohms)
        # R2 = R1 * (Vout / Vref - 1)
        r2 = r1 * ((vout / vref) - 1.0)
        return {
            "target_vout": vout,
            "vref_v": vref,
            "r1_ohms": r1,
            "r2_calculated_ohms": max(0.0, r2),
        }

    @staticmethod
    def calculate_voltage_divider(vin_v: float, r1_ohms: float, r2_ohms: float) -> Dict[str, Any]:
        """Calculates voltage divider output voltage, divider ratio, and power dissipation."""
        r1 = max(1e-6, r1_ohms)
        r2 = max(1e-6, r2_ohms)
        r_total = r1 + r2
        i_bleed = vin_v / r_total
        vout = vin_v * (r2 / r_total)
        p_total = (vin_v ** 2) / r_total
        p_r1 = (i_bleed ** 2) * r1
        p_r2 = (i_bleed ** 2) * r2
        return {
            "vin_v": vin_v,
            "r1_ohms": r1,
            "r2_ohms": r2,
            "vout_v": vout,
            "ratio": r2 / r_total,
            "current_ma": i_bleed * 1000.0,
            "total_power_mw": p_total * 1000.0,
            "p_r1_mw": p_r1 * 1000.0,
            "p_r2_mw": p_r2 * 1000.0,
        }

    @staticmethod
    def calculate_filter(filter_type: str, r_ohms: float = 1000.0, c_farads: float = 100e-9, l_henries: float = 10e-3) -> Dict[str, Any]:
        """Calculates cutoff frequency fc (-3dB point) and bandwidth for RC, RL, and LC filters."""
        ft = filter_type.lower()
        if "rc" in ft:
            r = max(1e-6, r_ohms)
            c = max(1e-18, c_farads)
            fc = 1.0 / (2.0 * math.pi * r * c)
            tau = r * c
            return {
                "filter_type": "RC Filter",
                "r_ohms": r,
                "c_farads": c,
                "cutoff_freq_hz": fc,
                "time_constant_s": tau,
            }
        elif "rl" in ft:
            r = max(1e-6, r_ohms)
            l = max(1e-12, l_henries)
            fc = r / (2.0 * math.pi * l)
            tau = l / r
            return {
                "filter_type": "RL Filter",
                "r_ohms": r,
                "l_henries": l,
                "cutoff_freq_hz": fc,
                "time_constant_s": tau,
            }
        elif "lc" in ft or "rlc" in ft:
            l = max(1e-12, l_henries)
            c = max(1e-18, c_farads)
            fc = 1.0 / (2.0 * math.pi * math.sqrt(l * c))
            z0 = math.sqrt(l / c)
            return {
                "filter_type": "LC Resonance Filter",
                "l_henries": l,
                "c_farads": c,
                "resonant_freq_hz": fc,
                "characteristic_z0_ohms": z0,
            }
        else:
            return {"error": f"Unknown filter type '{filter_type}'. Options: rc, rl, lc"}

    @staticmethod
    def calculate_reactance(freq_hz: float, c_farads: Optional[float] = None, l_henries: Optional[float] = None) -> Dict[str, Any]:
        """Calculates capacitive reactance Xc = 1/(2*pi*f*C) and inductive reactance Xl = 2*pi*f*L."""
        f = max(1e-6, freq_hz)
        res: Dict[str, Any] = {"freq_hz": f}
        if c_farads is not None:
            c = max(1e-18, c_farads)
            xc = 1.0 / (2.0 * math.pi * f * c)
            res["c_farads"] = c
            res["xc_ohms"] = xc
        if l_henries is not None:
            l = max(1e-18, l_henries)
            xl = 2.0 * math.pi * f * l
            res["l_henries"] = l
            res["xl_ohms"] = xl
        return res

    @staticmethod
    def calculate_power(voltage_v: float, current_a: Optional[float] = None, resistance_ohm: Optional[float] = None) -> Dict[str, Any]:
        """Calculates Ohm's Law and Power Dissipation."""
        v = voltage_v
        if current_a is not None:
            i = current_a
            p = v * i
            r = v / max(1e-12, i)
            return {"voltage_v": v, "current_a": i, "power_w": p, "resistance_ohm": r}
        elif resistance_ohm is not None:
            r = max(1e-12, resistance_ohm)
            i = v / r
            p = (v ** 2) / r
            return {"voltage_v": v, "current_a": i, "power_w": p, "resistance_ohm": r}
        return {"voltage_v": v}
