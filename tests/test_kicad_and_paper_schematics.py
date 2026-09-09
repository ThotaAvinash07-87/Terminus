"""Unit tests for KiCad EDA Mode, Paper-Style Schematic Visualizer, and Engineering Calculators."""

import unittest
from pathlib import Path

from UI.app import TerminusEngineBridge
from Engines.KiCad import KiCadProject, KiCadSchematic, KiCadPCB, KiCadCalculator, FootprintCatalog
from CORE.ascii_canvas import SchematicVisualizer
from CORE.storage_manager import StorageManager


class TestKiCadAndPaperSchematics(unittest.TestCase):
    """Tests KiCad mode workflows, paper-style ASCII schematics, calculations, and file persistence."""

    def setUp(self):
        self.bridge = TerminusEngineBridge()
        self.bridge.switch_mode("KICAD")

    def test_kicad_command_workflow(self):
        # 1. Add components
        res1 = self.bridge.execute_command("add R1 10k 0805")
        self.assertIn("Added KiCad Component", res1)
        self.assertIn("R1", res1)
        res2 = self.bridge.execute_command("add C1 100nF 0603")
        self.assertIn("Added KiCad Component", res2)
        self.assertIn("C1", res2)
        res3 = self.bridge.execute_command("add V1 5V PinHeader_1x2_P2.54mm")
        self.assertIn("Added KiCad Component", res3)
        self.assertIn("V1", res3)

        # 2. Connect components
        res_c1 = self.bridge.execute_command("connect V1.1 | R1.1 | VCC")
        self.assertIn("Connected:", res_c1)
        res_c2 = self.bridge.execute_command("connect R1.2 | C1.1 | Net_Out")
        self.assertIn("Connected:", res_c2)
        res_c3 = self.bridge.execute_command("connect C1.2 | V1.2 | GND")
        self.assertIn("Connected:", res_c3)

        # 3. Schematic paper-style visualization
        sch_art = self.bridge.execute_command("schematic")
        self.assertIn("CIRCUIT SCHEMATIC TOPOLOGY BLOCK", sch_art)
        self.assertIn("R1:10k", sch_art)
        self.assertIn("C1:100nF", sch_art)
        self.assertIn("GND", sch_art)

        # 4. Electrical Rules Check (ERC)
        erc_res = self.bridge.execute_command("erc")
        self.assertIn("Electrical Rules Check", erc_res)

        # 5. Place and route on PCB
        res_p1 = self.bridge.execute_command("place R1 10 10 0")
        self.assertIn("Placed on PCB", res_p1)
        self.assertIn("R1", res_p1)
        res_p2 = self.bridge.execute_command("place C1 20 10 90")
        self.assertIn("Placed on PCB", res_p2)
        self.assertIn("C1", res_p2)
        res_route = self.bridge.execute_command("autoroute")
        self.assertIn("Auto-routed", res_route)

        # 6. PCB 2D Board view
        pcb_art = self.bridge.execute_command("pcb")
        self.assertIn("PCB:", pcb_art)

        # 7. Design Rules Check (DRC)
        drc_res = self.bridge.execute_command("drc")
        self.assertIn("Design Rules Check", drc_res)

        # 8. Probe component calculation
        probe_r1 = self.bridge.execute_command("probe R1")
        self.assertIn("Voltage Drop", probe_r1)
        self.assertIn("Branch Current", probe_r1)
        self.assertIn("Power Dissipated", probe_r1)

        # 9. Gerbers & BOM
        gerber_res = self.bridge.execute_command("gerber")
        self.assertIn("Generated Complete RS-274X Gerber", gerber_res)
        bom_res = self.bridge.execute_command("bom")
        self.assertIn("Reference,Value,Footprint", bom_res)

        # 10. Save and Load project (.tkcad, .tsch, .tpcb)
        save_res = self.bridge.execute_command("file save TestProjectEDA")
        self.assertIn("Saved Terminus KiCad EDA project", save_res)
        load_res = self.bridge.execute_command("file open TestProjectEDA")
        self.assertIn("Loaded Terminus KiCad EDA project", load_res)

    def test_engineering_calculators(self):
        # 1. Track width
        res_track = self.bridge.execute_command("calc track 2.5 10 1 50")
        self.assertIn("External Track Width", res_track)
        self.assertIn("Internal Track Width", res_track)

        # 2. Via parasitics
        res_via = self.bridge.execute_command("calc via 0.3 0.6 1.6")
        self.assertIn("DC Resistance", res_via)
        self.assertIn("Inductance", res_via)

        # 3. Microstrip
        res_rf = self.bridge.execute_command("calc microstrip 1.5 1.6 4.5")
        self.assertIn("Characteristic Impedance (Z0)", res_rf)

        # 4. 555 timer
        res_555 = self.bridge.execute_command("calc 555 10k 47k 100n")
        self.assertIn("Oscillation Frequency", res_555)
        self.assertIn("Duty Cycle", res_555)

        # 5. Op-Amp gain
        res_op = self.bridge.execute_command("calc opamp noninv 10k 100k")
        self.assertIn("Voltage Gain (Av)", res_op)

        # 6. Voltage divider
        res_div = self.bridge.execute_command("calc divider 12V 10k 2.2k")
        self.assertIn("Output Voltage (Vout)", res_div)

        # 7. Regulator
        res_reg = self.bridge.execute_command("calc regulator 5V 1.25 240")
        self.assertIn("Required Resistor R2", res_reg)

        # 8. Filter
        res_filt = self.bridge.execute_command("calc filter rc 1k 100n")
        self.assertIn("Cutoff Frequency", res_filt)

        # 9. Reactance
        res_x = self.bridge.execute_command("calc reactance 1kHz 100nF")
        self.assertIn("Capacitive Reactance", res_x)

        # 10. Power
        res_pow = self.bridge.execute_command("calc power 5V 10mA")
        self.assertIn("Power Dissipation", res_pow)

    def test_circuit_mode_schematic_and_calc(self):
        self.bridge.switch_mode("CIRCUIT")
        self.bridge.execute_command("add V1 10V ac=1")
        self.bridge.execute_command("add R1 1k")
        self.bridge.execute_command("add C1 1u")
        self.bridge.execute_command("connect V1.p | R1.a")
        self.bridge.execute_command("connect R1.b | C1.a | out")
        self.bridge.execute_command("connect C1.b | V1.n | 0")

        sch_art = self.bridge.execute_command("schematic")
        self.assertIn("CIRCUIT SCHEMATIC TOPOLOGY BLOCK", sch_art)
        self.assertIn("R1:1.000k", sch_art)
        self.assertIn("C1:1.000u", sch_art)

        # Check calc command works in Circuit mode too
        res_calc = self.bridge.execute_command("calc filter rc 1k 1u")
        self.assertIn("Cutoff Frequency", res_calc)

        # Check pcb command works in Circuit / LTspice mode too
        pcb_art = self.bridge.execute_command("pcb")
        self.assertIn("PCB:", pcb_art)
        self.assertIn("R1", pcb_art)
        self.assertIn("C1", pcb_art)
        self.assertIn("V1", pcb_art)


if __name__ == "__main__":
    unittest.main()
