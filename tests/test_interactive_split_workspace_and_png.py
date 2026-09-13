"""Unit tests for the 60/40 Split Interactive Workspace, Compact Electronic Symbols,
Multi-Mode Canvases, High-Precision Vector PNG Exports, and Technical Metrics Cards.
"""

import unittest
from pathlib import Path
import os
import shutil

from UI.app import TerminusEngineBridge
from UI.split_workspace import SplitWorkspaceRenderer
from CORE.interactive_canvas import InteractiveSchematicCanvas
from CORE.ascii_canvas import AsciiCanvas, SchematicVisualizer
from CORE.storage_manager import StorageManager
from CORE.figure_renderer import TerminusFigure, TechnicalReportMetrics


class TestInteractiveSplitWorkspaceAndPNG(unittest.TestCase):
    """Tests 60/40 split screen rendering, interactive block movement, and PNG generators."""

    def setUp(self):
        self.bridge = TerminusEngineBridge()

    def test_compact_symbol_formatting(self):
        """Validates standard short electrical notations (R, L, C, U:OP-AMP, etc.)."""
        # Test compact block styling
        r_blk = AsciiCanvas.format_component_short_block("R1", "10k")
        self.assertIn("R", r_blk)
        self.assertIn("R1:10k", r_blk)

        c_blk = AsciiCanvas.format_component_short_block("C1", "100n")
        self.assertIn("C", c_blk)
        self.assertIn("C1:100n", c_blk)

        op_blk = AsciiCanvas.format_component_short_block("U1", "LM358", "OP-AMP")
        self.assertIn("OP-AMP", op_blk)
        self.assertIn("U1", op_blk)

    def test_interactive_canvas_movement_and_routing(self):
        """Validates 2D movable blocks and orthogonal Manhattan routing."""
        canvas = InteractiveSchematicCanvas(width_chars=60, height_chars=16)
        b1 = canvas.add_or_update_block("R1", "[ R R1:10k ]", x=5, y=3)
        b2 = canvas.add_or_update_block("C1", "[ C C1:100n ]", x=25, y=3)
        canvas.add_wire("R1", "2", "C1", "1", net_name="out")

        # Verify initial rendering
        art = canvas.render()
        self.assertIn("R1", art)
        self.assertIn("C1", art)

        # Move component
        ok = canvas.move_block("C1", 30, 6)
        self.assertTrue(ok)
        self.assertEqual(b2.x, 30)
        self.assertEqual(b2.y, 6)

        # Select component
        sel = canvas.select_block("R1")
        self.assertIsNotNone(sel)
        self.assertTrue(sel.selected)

    def test_split_workspace_rendering_across_all_modes(self):
        """Validates 60% Left / 40% Right + VS Code bottom history terminal layout in all 6 modes."""
        modes = ["CIRCUIT", "NUMERICAL", "DYNAMIC", "DIGITAL", "EMBEDDED", "KICAD"]
        for m in modes:
            self.bridge.switch_mode(m)
            ws_view = self.bridge.render_split_workspace(f"Testing Mode {m}")
            # Verify outer framing
            self.assertIn("VS Code Terminal History & Output Log", ws_view)
            self.assertIn("Testing Mode", ws_view)

    def test_matlab_numerical_code_buffer_and_split_screen(self):
        """Validates MATLAB mode code writing on left 60% and figure/data on right 40%."""
        self.bridge.switch_mode("NUMERICAL")
        self.bridge.execute_command("project DSPWorkspace")
        self.bridge.execute_command("t = 0:0.001:0.1")
        self.bridge.execute_command("sig = sin(2*pi*50*t)")

        ws_view = self.bridge.render_split_workspace()
        self.assertIn("MATLAB", ws_view)
        self.assertIn("Workspace Variables", ws_view)

    def test_xilinx_digital_logic_code_and_timing_split_screen(self):
        """Validates Xilinx mode HDL editor on left 60% and timing diagrams on right 40%."""
        self.bridge.switch_mode("DIGITAL")
        self.bridge.execute_command("wire a, b, out")
        self.bridge.execute_command("gate G1 AND a b -> out")

        ws_view = self.bridge.render_split_workspace()
        self.assertIn("Xilinx", ws_view)
        self.assertIn("FPGA Resource Utilization", ws_view)

    def test_png_high_precision_exports(self):
        """Validates png graph, png card, and png pcb separate exports."""
        # 1. Circuit mode simulation & exports
        self.bridge.switch_mode("CIRCUIT")
        self.bridge.execute_command("add V1 10V ac=1")
        self.bridge.execute_command("add R1 1k")
        self.bridge.execute_command("add C1 100n")
        self.bridge.execute_command("connect V1.p | R1.a")
        self.bridge.execute_command("connect R1.b | C1.a | out")
        self.bridge.execute_command("connect C1.b | V1.n | 0")
        self.bridge.execute_command("run .tran 1u 1m")

        # Export pure graph
        res_graph = self.bridge.execute_command("png graph test_circuit_graph.png")
        self.assertIn("High-Precision Vector Graph PNG", res_graph)
        self.assertIn("test_circuit_graph.png", res_graph)

        # Export technical metrics card
        res_card = self.bridge.execute_command("png card test_circuit_card.png")
        self.assertIn("Technical Report Metrics Card PNG", res_card)
        self.assertIn("test_circuit_card.png", res_card)

        # 2. KiCad PCB artwork PNG export
        self.bridge.switch_mode("KICAD")
        self.bridge.execute_command("add R1 10k 0805")
        self.bridge.execute_command("add C1 100n 0603")
        self.bridge.execute_command("place R1 10 10 0")
        self.bridge.execute_command("place C1 20 10 90")

        res_pcb = self.bridge.execute_command("png pcb test_kicad_pcb.png")
        self.assertIn("KiCad PCB Layout Artwork PNG", res_pcb)
        self.assertIn("test_kicad_pcb.png", res_pcb)


if __name__ == "__main__":
    unittest.main()
