"""Unit and integration test suite for LTspice circuit library, DRC diagnostics, MNA solver,
and High-Capacity Multi-Terminal Session Networking (500+ sessions).
"""

import math
import os
import sys
import time
import unittest
import numpy as np

from Engines.Circuit.components import (
    Component, Resistor, Capacitor, Inductor, CoupledInductors,
    VoltageSource, CurrentSource, Diode, BJT, MOSFET, JFET,
    VoltageControlledSwitch, BehavioralSource, OpAmpModel, SubcircuitDefinition, SubcircuitInstance
)
from Engines.Circuit.catalog import circuit_catalog, CircuitComponentCatalog
from Engines.Circuit.diagnostics import CircuitDiagnosticChecker, CircuitDiagnosticLevel
from Engines.Circuit.netlist_parser import Netlist, CircuitParser
from Engines.Circuit.mna_solver import MNASolver, SimulationResult
from CORE.ipc_router import IPCRouter, IPCClient
from UI.app import TerminusEngineBridge


class TestCircuitCatalogAndDiagnostics(unittest.TestCase):
    """Tests for the classified LTspice component library and DRC diagnostic checker."""

    def test_catalog_categories_and_search(self):
        cats = circuit_catalog.get_all_categories()
        self.assertIn("Passives", cats)
        self.assertIn("Diodes", cats)
        self.assertIn("Transistors", cats)
        self.assertIn("Integrated Circuits", cats)
        self.assertIn("Sources", cats)
        self.assertIn("Subcircuits", cats)

        # Test category list
        diodes = circuit_catalog.list_components_by_category("Diodes")
        self.assertGreater(len(diodes), 0)
        diode_names = [d.name for d in diodes]
        self.assertIn("1N4148", diode_names)
        self.assertIn("1N4007", diode_names)

        # Test search
        mosfets = circuit_catalog.search_components("MOSFET")
        self.assertGreater(len(mosfets), 0)
        self.assertTrue(any("IRF540N" in m.name for m in mosfets))

        # Test inspect spec
        spec_741 = circuit_catalog.get_spec("LM741")
        self.assertIsNotNone(spec_741)
        self.assertEqual(spec_741.category, "Integrated Circuits")
        self.assertIn("out", spec_741.pins)

    def test_drc_missing_ground(self):
        nl = Netlist("NoGroundCircuit")
        nl.add_component(Resistor("R1", "n1", "n2", 1000.0))
        nl.add_component(VoltageSource("V1", "n1", "n2", dc=5.0))

        checker = CircuitDiagnosticChecker(nl)
        report = checker.diagnose()
        self.assertFalse(report.is_valid)
        self.assertTrue(report.has_errors)
        self.assertTrue(any(i.category == "Ground Reference" for i in report.issues))

    def test_drc_floating_node(self):
        nl = Netlist("FloatingCircuit")
        nl.add_component(VoltageSource("V1", "1", "0", dc=10.0))
        nl.add_component(Resistor("R1", "1", "0", 1000.0))
        nl.add_component(Resistor("R2", "1", "dangling_node", 1000.0))

        checker = CircuitDiagnosticChecker(nl)
        report = checker.diagnose()
        self.assertTrue(report.has_warnings)
        self.assertTrue(any(i.category == "Floating Node" for i in report.issues))

    def test_drc_shorted_voltage_source(self):
        nl = Netlist("ShortedSourceCircuit")
        nl.add_component(VoltageSource("V1", "0", "0", dc=5.0))

        checker = CircuitDiagnosticChecker(nl)
        report = checker.diagnose()
        self.assertFalse(report.is_valid)
        self.assertTrue(any(i.category == "Short Circuit" for i in report.issues))

    def test_drc_healthy_circuit(self):
        nl = Netlist("HealthyCircuit")
        nl.add_component(VoltageSource("V1", "in", "0", dc=10.0))
        nl.add_component(Resistor("R1", "in", "out", 1000.0))
        nl.add_component(Resistor("R2", "out", "0", 1000.0))

        checker = CircuitDiagnosticChecker(nl)
        report = checker.diagnose()
        self.assertTrue(report.is_valid)
        self.assertFalse(report.has_errors)


class TestMNASolverAndAnalyses(unittest.TestCase):
    """Tests for MNA simulation engine (.OP, .DC, .AC, .TRAN, .FOUR)."""

    def test_dc_op_voltage_divider(self):
        nl = Netlist("Divider")
        nl.add_component(VoltageSource("V1", "in", "0", dc=12.0))
        nl.add_component(Resistor("R1", "in", "mid", 3000.0))
        nl.add_component(Resistor("R2", "mid", "0", 1000.0))

        solver = MNASolver(nl)
        res = solver.solve_op()
        self.assertAlmostEqual(res.op_results["V(in)"], 12.0, places=4)
        self.assertAlmostEqual(res.op_results["V(mid)"], 3.0, places=4)
        self.assertAlmostEqual(res.op_results["I(V1)"], -0.003, places=4)

    def test_dc_sweep(self):
        nl = Netlist("SweepTest")
        nl.add_component(VoltageSource("V1", "in", "0", dc=0.0))
        nl.add_component(Resistor("R1", "in", "out", 1000.0))
        nl.add_component(Resistor("R2", "out", "0", 1000.0))

        solver = MNASolver(nl)
        res = solver.solve_dc_sweep("V1", 0.0, 10.0, 1.0)
        self.assertEqual(len(res.x), 11)
        wf_out = res.get_waveform("V(out)")
        self.assertIsNotNone(wf_out)
        self.assertAlmostEqual(wf_out.y[0], 0.0, places=4)
        self.assertAlmostEqual(wf_out.y[-1], 5.0, places=4)

    def test_ac_rc_filter(self):
        nl = Netlist("RC_Filter")
        nl.add_component(VoltageSource("V1", "in", "0", dc=0.0, ac_mag=1.0))
        nl.add_component(Resistor("R1", "in", "out", 1000.0))
        nl.add_component(Capacitor("C1", "out", "0", 1e-6))  # Fc ~ 159.15 Hz

        solver = MNASolver(nl)
        res = solver.solve_ac("dec", 10, 1.0, 100e3)
        wf_out = res.get_waveform("V(out)")
        self.assertIsNotNone(wf_out)
        # At 1 Hz (DC-like), magnitude should be ~ 0 dB (gain ~ 1.0)
        self.assertAlmostEqual(wf_out.magnitude_db[0], 0.0, delta=0.5)
        # At high freq (100kHz), magnitude should be heavily attenuated (< -40 dB)
        self.assertLess(wf_out.magnitude_db[-1], -40.0)

    def test_transient_rc_step(self):
        nl = Netlist("RC_Step")
        # Pulse voltage source: 0V to 5V step
        nl.add_component(VoltageSource("V1", "in", "0", waveform_type="PULSE", wave_params={"v1": 0.0, "v2": 5.0, "tr": 1e-6, "ton": 10e-3, "period": 20e-3}))
        nl.add_component(Resistor("R1", "in", "out", 1000.0))
        nl.add_component(Capacitor("C1", "out", "0", 1e-6))  # tau = 1ms

        solver = MNASolver(nl)
        res = solver.solve_tran(t_step="50u", t_stop="5m")
        wf_out = res.get_waveform("V(out)")
        self.assertIsNotNone(wf_out)
        # Check asymptotic charging towards 5V
        self.assertAlmostEqual(wf_out.y[0], 0.0, delta=0.1)
        self.assertGreater(wf_out.y[-1], 4.8)

    def test_fourier_analysis_thd(self):
        nl = Netlist("SineTest")
        # Clean 1kHz sine wave 1V amplitude
        nl.add_component(VoltageSource("V1", "in", "0", waveform_type="SINE", wave_params={"offset": 0.0, "amplitude": 2.0, "freq": 1000.0}))
        nl.add_component(Resistor("R1", "in", "out", 1000.0))
        nl.add_component(Resistor("R2", "out", "0", 1000.0))

        solver = MNASolver(nl)
        res = solver.solve_four(freq_hz=1000.0, trace_name="V(out)", t_stop="10m")
        self.assertEqual(res.sim_type, "FOUR")
        self.assertIn("thd_percent", res.metrics)
        # Linear voltage divider creates no harmonic distortion (THD < 0.5%)
        self.assertLess(res.metrics["thd_percent"], 0.5)

    def test_subcircuit_instantiation(self):
        # Create subcircuit LPF
        sub = SubcircuitDefinition("TEST_LPF", ["IN_PIN", "OUT_PIN", "GND_PIN"])
        sub.add_component(Resistor("R1", "IN_PIN", "OUT_PIN", 2000.0))
        sub.add_component(Capacitor("C1", "OUT_PIN", "GND_PIN", 1e-6))

        top_nl = Netlist("TopNetlist")
        top_nl.subcircuits["TEST_LPF"] = sub
        top_nl.add_component(VoltageSource("V1", "src", "0", dc=10.0))
        top_nl.add_component(SubcircuitInstance("X1", "TEST_LPF", ["src", "load", "0"]))

        flat = top_nl.flatten_subcircuits()
        self.assertIn("X1_R1", flat.components)
        self.assertIn("X1_C1", flat.components)

        solver = MNASolver(flat)
        op = solver.solve_op()
        self.assertAlmostEqual(op.op_results["V(load)"], 10.0, places=4)


class TestMultiTerminalSessionNetworking(unittest.TestCase):
    """Tests for scalable multi-terminal networking, session discovery, and cross-terminal linking."""

    @classmethod
    def setUpClass(cls):
        cls.test_port = 8799
        cls.router = IPCRouter(host="127.0.0.1", port=cls.test_port)
        cls.router.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.router.stop()

    def test_session_allocation_and_listing(self):
        client1 = IPCClient(host="127.0.0.1", port=self.test_port)
        client2 = IPCClient(host="127.0.0.1", port=self.test_port)

        ok1 = client1.connect(mode="Circuit", active_file="ckt1.cir")
        ok2 = client2.connect(mode="Dynamic", active_file="model1.tmdl")

        self.assertTrue(ok1)
        self.assertTrue(ok2)
        self.assertGreater(client1.session_id, 0)
        self.assertGreater(client2.session_id, 0)
        self.assertNotEqual(client1.session_id, client2.session_id)

        # List active sessions
        sessions = client1.list_sessions_sync()
        self.assertGreaterEqual(len(sessions), 2)
        sids = [s["session_id"] for s in sessions]
        self.assertIn(client1.session_id, sids)
        self.assertIn(client2.session_id, sids)

    def test_remote_command_execution(self):
        # Setup target client with command executor
        client_target = IPCClient(host="127.0.0.1", port=self.test_port)
        bridge = TerminusEngineBridge()
        client_target.set_command_executor(bridge.execute_command)
        client_target.connect(mode="Circuit", active_file="remote_test.cir")

        client_sender = IPCClient(host="127.0.0.1", port=self.test_port)
        client_sender.connect(mode="Dynamic", active_file="sender.tmdl")

        # Sender dispatches command to target session
        res = client_sender.remote_exec_sync(client_target.session_id, "add R1 in out 5k")
        self.assertEqual(res.get("status"), "dispatched")
        time.sleep(0.2)

        # Verify component was added in target bridge
        self.assertIn("R1", bridge.circuit_netlist.components)

    def test_cross_terminal_signal_linking(self):
        client_rx = IPCClient(host="127.0.0.1", port=self.test_port)
        client_rx.connect(mode="Dynamic", active_file="rx_model.tmdl")

        client_tx = IPCClient(host="127.0.0.1", port=self.test_port)
        client_tx.connect(mode="Circuit", active_file="tx_ckt.cir")

        # Link signal: Tx.V_out -> Rx.motor_voltage
        link_res = client_tx.link_signal_sync("V_out", client_rx.session_id, "motor_voltage")
        self.assertIn("linked", link_res)

        # Push signal from Tx
        client_tx.push_signal_sync("V_out", 12.5)
        time.sleep(0.2)

        # Verify Rx received bridged value
        self.assertEqual(client_rx.linked_signals.get("motor_voltage"), 12.5)

    def test_high_concurrency_connections(self):
        """Spawns 40 concurrent async terminal connections to test non-blocking socket handling."""
        clients = []
        for i in range(40):
            c = IPCClient(host="127.0.0.1", port=self.test_port)
            ok = c.connect(mode="Circuit", active_file=f"test_{i}.cir")
            self.assertTrue(ok)
            clients.append(c)

        # Verify all sessions are registered on router
        sessions = clients[0].list_sessions_sync()
        self.assertGreaterEqual(len(sessions), 40)


class TestUnifiedCLIBridgeIntegration(unittest.TestCase):
    """Tests interactive CLI commands through TerminusEngineBridge."""

    def test_circuit_library_cli_commands(self):
        bridge = TerminusEngineBridge()
        bridge.switch_mode("circuit")

        # Browse library
        out_lib = bridge.execute_command("library")
        self.assertIn("LTspice Classified Component Library", out_lib)

        # Search library
        out_search = bridge.execute_command("library search 1N4148")
        self.assertIn("1N4148", out_search)

        # Inspect component spec
        out_inspect = bridge.execute_command("inspect 1N4148")
        self.assertIn("1N4148", out_inspect)

    def test_circuit_drc_cli_command(self):
        bridge = TerminusEngineBridge()
        bridge.switch_mode("circuit")
        bridge.execute_command("add V1 10V")
        bridge.execute_command("add R1 1k")
        bridge.execute_command("connect V1.p | R1.a")
        bridge.execute_command("connect V1.n | 0")

        out_drc = bridge.execute_command("check")
        self.assertIn("Circuit Design Rule", out_drc)

    def test_session_cli_commands(self):
        bridge = TerminusEngineBridge()
        out_session = bridge.execute_command("session")
        self.assertIn("Multi-Terminal Session Networking", out_session)

        out_limits = bridge.execute_command("session limits")
        self.assertIn("Socket Descriptor Capacity", out_limits)


if __name__ == "__main__":
    unittest.main()
