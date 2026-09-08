"""Comprehensive unit and integration tests for the Simulink-like frontend workflow,
classified block library browser, parameter tuner, deep model diagnostics, and multi-solver ODEs.
"""

import unittest
import os
import tempfile
import numpy as np

from UI.app import TerminusEngineBridge
from Engines.Dynamic_System.catalog import DynamicBlockCatalog, parse_parameter_value
from Engines.Dynamic_System.diagnostics import ModelDiagnosticChecker, DiagnosticSeverity
from Engines.Dynamic_System.scheduler import SystemDiagram
from Engines.Dynamic_System.ode_solver import DynamicSystemSimulator
from Engines.Dynamic_System.blocks import (
    IntegratorBlock, GainBlock, SumBlock, SaturationBlock,
    StepSourceBlock, ScopeSinkBlock, TransferFunctionBlock, PIDBlock, UnitDelayBlock
)


class TestSimulinkFrontendWorkflow(unittest.TestCase):
    """Test suite for the Simulink frontend workflow and diagnostics."""

    def setUp(self):
        self.bridge = TerminusEngineBridge()
        self.bridge.switch_mode("dynamic")

    def test_catalog_categories_and_search(self):
        """Test library catalog indexing and search."""
        cats = DynamicBlockCatalog.list_categories()
        self.assertGreaterEqual(len(cats), 15)
        self.assertIn("Continuous", cats)
        self.assertIn("Discrete", cats)
        self.assertIn("Math Operations", cats)
        self.assertIn("Discontinuities", cats)
        self.assertIn("Sources", cats)
        self.assertIn("Sinks", cats)

        # Get blocks by category
        cont_blocks = DynamicBlockCatalog.get_blocks_by_category("Continuous")
        self.assertTrue(any(b.type_name == "integrator" for b in cont_blocks))
        self.assertTrue(any(b.type_name == "pid" for b in cont_blocks))

        # Search
        results = DynamicBlockCatalog.search_blocks("saturation")
        self.assertTrue(any(r.type_name == "saturation" for r in results))

        results_alias = DynamicBlockCatalog.search_blocks("1/s")
        self.assertTrue(any(r.type_name == "integrator" for r in results_alias))

    def test_catalog_block_instantiation_and_parameter_parsing(self):
        """Test instantiating blocks with type-checked parameters."""
        # Integrator
        int_block = DynamicBlockCatalog.instantiate(
            "integrator", "Int1",
            initial_condition="5.0",
            lower_limit="-10.0",
            upper_limit="10.0"
        )
        self.assertIsInstance(int_block, IntegratorBlock)
        self.assertEqual(int_block.initial_condition, 5.0)
        self.assertEqual(int_block.lower_limit, -10.0)
        self.assertEqual(int_block.upper_limit, 10.0)

        # PID
        pid_block = DynamicBlockCatalog.instantiate(
            "pid", "PID1",
            kp="2.5",
            ki="1.2",
            kd="0.05",
            n_filter="50"
        )
        self.assertIsInstance(pid_block, PIDBlock)
        self.assertEqual(pid_block.kp, 2.5)
        self.assertEqual(pid_block.ki, 1.2)
        self.assertEqual(pid_block.kd, 0.05)
        self.assertEqual(pid_block.n_filter, 50.0)

        # Transfer Function
        tf_block = DynamicBlockCatalog.instantiate(
            "tf", "Plant",
            num="[10.0]",
            den="[1.0, 3.0, 2.0]"
        )
        self.assertIsInstance(tf_block, TransferFunctionBlock)
        self.assertEqual(tf_block.num, [10.0])
        self.assertEqual(tf_block.den, [1.0, 3.0, 2.0])

    def test_diagnostics_unconnected_ports(self):
        """Test diagnostic detection of unconnected input ports."""
        d = SystemDiagram("TestUnconnected")
        step = StepSourceBlock("Step1")
        gain = GainBlock("G1", gain=2.0)
        scope = ScopeSinkBlock("Scope1")

        d.add_block(step)
        d.add_block(gain)
        d.add_block(scope)

        # Connect step to scope, leaving gain input unconnected
        d.connect("Step1.0", "Scope1.0")

        report = ModelDiagnosticChecker.run_diagnostics(d)
        self.assertFalse(report.is_valid)
        self.assertGreaterEqual(report.num_errors, 1)

        # Check that G1 is flagged
        unconnected_issues = [i for i in report.issues if i.category == "Unconnected Port" and i.block_name == "G1"]
        self.assertEqual(len(unconnected_issues), 1)
        self.assertEqual(unconnected_issues[0].port_index, 0)

    def test_diagnostics_algebraic_loop(self):
        """Test diagnostic detection of direct-feedthrough algebraic loops."""
        d = SystemDiagram("TestAlgLoop")
        g1 = GainBlock("G1", gain=2.0)
        g2 = GainBlock("G2", gain=0.5)

        d.add_block(g1)
        d.add_block(g2)

        # Cyclic feedthrough loop: G1 -> G2 -> G1
        d.connect("G1.0", "G2.0")
        d.connect("G2.0", "G1.0")

        report = ModelDiagnosticChecker.run_diagnostics(d)
        self.assertTrue(report.has_algebraic_loops)
        loop_issues = [i for i in report.issues if i.category == "Algebraic Loop"]
        self.assertGreaterEqual(len(loop_issues), 1)

    def test_diagnostics_parameter_bounds(self):
        """Test diagnostic checking for illegal parameter ranges."""
        d = SystemDiagram("TestParams")
        sat = SaturationBlock("Sat1", lower_limit=10.0, upper_limit=5.0)  # Invalid: lower > upper
        d.add_block(sat)

        report = ModelDiagnosticChecker.run_diagnostics(d)
        param_issues = [i for i in report.issues if i.category == "Parameter Bounds" and i.block_name == "SAT1"]
        self.assertEqual(len(param_issues), 1)
        self.assertEqual(param_issues[0].severity, DiagnosticSeverity.ERROR)

    def test_parameter_tuning_dynamically(self):
        """Test dynamic parameter tuning on diagrams."""
        d = SystemDiagram("TuningTest")
        pid = PIDBlock("PID1", kp=1.0, ki=0.0, kd=0.0)
        d.add_block(pid, block_type="pid")

        d.tune_parameter("PID1", "kp", "3.5")
        d.tune_parameter("PID1", "ki", 1.8)

        self.assertEqual(pid.kp, 3.5)
        self.assertEqual(pid.ki, 1.8)
        self.assertEqual(pid.parameters["kp"], 3.5)
        self.assertEqual(pid.parameters["ki"], 1.8)

    def test_model_serialization_save_load(self):
        """Test saving and opening model files (.tmdl / .json)."""
        d = SystemDiagram("FeedbackLoop")
        d.solver = "euler"
        d.dt = 0.005
        d.t_stop = 8.0

        step = StepSourceBlock("Step1", step_time=1.0, amplitude=5.0)
        gain = GainBlock("Gain1", gain=2.5)
        scope = ScopeSinkBlock("Scope1")

        d.add_block(step, block_type="step", step_time=1.0, amplitude=5.0)
        d.add_block(gain, block_type="gain", gain=2.5)
        d.add_block(scope, block_type="scope")

        d.connect("Step1.0", "Gain1.0")
        d.connect("Gain1.0", "Scope1.0")

        with tempfile.NamedTemporaryFile(suffix=".tmdl", delete=False) as tf:
            temp_path = tf.name

        try:
            d.save_to_file(temp_path)

            d2 = SystemDiagram("LoadedModel")
            d2.load_from_file(temp_path)

            self.assertEqual(d2.name, "FeedbackLoop")
            self.assertEqual(d2.solver, "euler")
            self.assertEqual(d2.dt, 0.005)
            self.assertEqual(d2.t_stop, 8.0)
            self.assertEqual(len(d2.blocks), 3)
            self.assertEqual(len(d2.connections), 2)
            self.assertIn("GAIN1", d2.blocks)
            self.assertEqual(d2.blocks["GAIN1"].gain, 2.5)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_multi_solver_ode_simulation(self):
        """Test simulation with RK4, Euler, Heun, and adaptive ODE45."""
        d = SystemDiagram("PlantSim")
        step = StepSourceBlock("Step1", step_time=0.0, amplitude=1.0)
        plant = TransferFunctionBlock("Plant", num=[1.0], den=[1.0, 1.0])  # First order: 1/(s+1)
        scope = ScopeSinkBlock("Scope1")

        d.add_block(step)
        d.add_block(plant)
        d.add_block(scope)

        d.connect("Step1.0", "Plant.0")
        d.connect("Plant.0", "Scope1.0")

        sim = DynamicSystemSimulator(d)

        # 1. RK4
        res_rk4 = sim.simulate(t_stop=5.0, dt=0.01, solver="rk4")
        self.assertIn("Scope1", res_rk4)
        final_rk4 = res_rk4["Scope1"].y[-1]
        self.assertAlmostEqual(final_rk4, 1.0 - np.exp(-5.0), delta=0.01)

        # 2. Euler
        res_euler = sim.simulate(t_stop=5.0, dt=0.01, solver="euler")
        self.assertIn("Scope1", res_euler)
        final_euler = res_euler["Scope1"].y[-1]
        self.assertAlmostEqual(final_euler, 1.0 - np.exp(-5.0), delta=0.05)

        # 3. Heun
        res_heun = sim.simulate(t_stop=5.0, dt=0.01, solver="heun")
        self.assertIn("Scope1", res_heun)
        final_heun = res_heun["Scope1"].y[-1]
        self.assertAlmostEqual(final_heun, 1.0 - np.exp(-5.0), delta=0.02)

        # 4. Adaptive ODE45
        res_ode45 = sim.simulate(t_stop=5.0, dt=0.01, solver="ode45")
        self.assertIn("Scope1", res_ode45)
        final_ode45 = res_ode45["Scope1"].y[-1]
        self.assertAlmostEqual(final_ode45, 1.0 - np.exp(-5.0), delta=0.02)

    def test_full_command_workflow_in_bridge(self):
        """Test full user workflow from command lines in TerminusEngineBridge."""
        # Create model
        out = self.bridge.execute_command("file new Cruise_Controller")
        self.assertIn("Created new", out)
        self.assertIn("Cruise_Controller", out)

        # Browse library
        out_lib = self.bridge.execute_command("library Continuous")
        self.assertIn("Continuous Blocks", out_lib)

        # Add blocks
        self.bridge.execute_command("add sources/step TargetSpeed step_time=1.0 amplitude=60.0")
        self.bridge.execute_command("add math/sum Error signs=+-")
        self.bridge.execute_command("add continuous/pid Controller kp=2.0 ki=0.5 kd=0.1")
        self.bridge.execute_command("add discontinuities/saturation Actuator lower_limit=0.0 upper_limit=100.0")
        self.bridge.execute_command("add continuous/tf Vehicle num=[1] den=[10, 1]")
        self.bridge.execute_command("add sinks/scope SpeedScope")

        # Connect wire network
        self.bridge.execute_command("connect TargetSpeed.0 Error.0")
        self.bridge.execute_command("connect Error.0 Controller.0")
        self.bridge.execute_command("connect Controller.0 Actuator.0")
        self.bridge.execute_command("connect Actuator.0 Vehicle.0")
        self.bridge.execute_command("connect Vehicle.0 SpeedScope.0")
        self.bridge.execute_command("connect Vehicle.0 Error.1")

        # Inspect block
        out_insp = self.bridge.execute_command("inspect Controller")
        self.assertIn("Block Properties: CONTROLLER", out_insp)
        self.assertIn("kp", out_insp)

        # Tune parameter
        out_tune = self.bridge.execute_command("tune Controller kp=3.0 ki=0.8")
        self.assertIn("Tuned CONTROLLER", out_tune)

        # Model Advisor Pre-Flight Diagnostics
        out_check = self.bridge.execute_command("check")
        self.assertIn("PASSED", out_check)
        self.assertIn("0 Error(s)", out_check)

        # Simulate
        out_sim = self.bridge.execute_command("sim 15 0.01 rk4")
        self.assertIn("Simulation Complete", out_sim)
        self.assertIn("SPEEDSCOPE", out_sim)


if __name__ == "__main__":
    unittest.main()
